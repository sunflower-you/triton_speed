# 02. `01-vector-add.py` 调用栈

本文只跟一个例子：

```bash
cd /home/moffett/workspace/triton_speed
MPLBACKEND=Agg TRITON_INTERPRET=0 python python/tutorials/01-vector-add.py
```

你已经跑通过，关键输出是：

```text
The maximum difference between torch and triton is 0.0
```

## 第一层：用户代码

文件：

```text
python/tutorials/01-vector-add.py
```

核心 kernel：

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

小白理解：

- `pid`：当前 GPU 小程序块的编号。
- `BLOCK_SIZE=1024`：每个小程序块处理 1024 个元素。
- `offsets`：当前程序块负责的元素下标。
- `mask`：防止越界。
- `tl.load`：从 GPU 内存读。
- `tl.store`：写回 GPU 内存。

调用位置：

```python
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
```

这里最重要：

```text
add_kernel[grid]
```

不是普通 Python list 下标，而是触发 Triton 的 kernel launch 语法。

## 插一层：`grid = lambda meta: ...` 是怎么算的

教程里 launch 前有这一句：

```python
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
```

它的作用是：根据输入元素个数和 `BLOCK_SIZE`，计算要启动多少个 Triton program。

`triton.cdiv(a, b)` 是向上取整除法：

```python
triton.cdiv(n_elements, BLOCK_SIZE)
```

等价于：

```python
(n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
```

举例：

```python
n_elements = 10000
BLOCK_SIZE = 1024
```

那么：

```python
triton.cdiv(10000, 1024) = 10
```

因为 9 个 program 只能处理：

```text
9 * 1024 = 9216
```

还剩一些元素，所以需要第 10 个 program。第 10 个 program 里多出来的 offset 会被 kernel 里的 mask 挡住：

```python
mask = offsets < n_elements
```

所以最终 grid 是：

```python
(10,)
```

这个 tuple 表示一维 launch grid，有 10 个 Triton program。

### `meta` 从哪里来

用户代码里：

```python
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
```

其中：

```python
BLOCK_SIZE=1024
```

是传给 JIT kernel 的 keyword 参数。

进入 `JITFunction.run()` 后，源码里会先执行：

```python
bound_args, specialization, options = binder(*args, **kwargs)
```

这里的 `bound_args` 大概长这样：

```python
{
    "x_ptr": x,
    "y_ptr": y,
    "output_ptr": output,
    "n_elements": n_elements,
    "BLOCK_SIZE": 1024,
}
```

然后 `JITFunction.run()` 里有这段：

```python
if callable(grid):
    grid = grid(bound_args)
```

所以教程里 lambda 的参数名虽然叫 `meta`：

```python
lambda meta: ...
```

实际传进去的是 `bound_args`。也就是说：

```python
meta['BLOCK_SIZE']
```

就是：

```python
bound_args['BLOCK_SIZE']
```

也就是 launch 时传入的：

```python
BLOCK_SIZE=1024
```

这条小调用链是：

```text
BLOCK_SIZE=1024
  -> 进入 add_kernel[grid](...) 的 kwargs
  -> binder(*args, **kwargs)
  -> bound_args["BLOCK_SIZE"] = 1024
  -> grid(bound_args)
  -> lambda meta: meta["BLOCK_SIZE"]
  -> 算出 launch grid
```

## 第二层：`@triton.jit` 把函数包起来

文件：

```text
python/triton/runtime/jit.py
```

入口：

```python
def jit(...):
    ...
    return JITFunction(...)
```

你写：

```python
@triton.jit
def add_kernel(...):
    ...
```

大概等价于：

```python
add_kernel = triton.jit(add_kernel)
```

所以 `add_kernel` 不再是普通 Python 函数，而是 `JITFunction` 对象。

## 第三层：`kernel[grid]` 生成可调用对象

相关代码：

```text
python/triton/runtime/jit.py
```

关键方法：

```python
class KernelInterface:
    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
```

小白理解：

```python
add_kernel[grid]
```

会返回一个“带着 grid 信息的函数”。

后面再调用：

```python
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
```

实际进入：

```python
JITFunction.run(...)
```

## 第四层：`JITFunction.run` 决定编译还是用缓存

文件：

```text
python/triton/runtime/jit.py
```

关键方法：

```python
def run(self, *args, grid, warmup, **kwargs):
    device = driver.active.get_current_device()
    stream = driver.active.get_current_stream(device)
    kernel_cache, ..., backend, binder = self.device_caches[device]
    bound_args, specialization, options = binder(*args, **kwargs)
    key = compute_cache_key(...)
    kernel = kernel_cache.get(key, None)
    if kernel is None:
        kernel = self._do_compile(...)
    ...
    kernel.run(...)
```

小白理解：

这一步做 4 件事：

1. 找当前 GPU。
2. 找当前 CUDA stream。
3. 看这个 kernel 以前有没有编译过。
4. 没编译过就编译，编译过就直接 launch。

为什么要缓存？

因为编译很贵。相同参数类型、相同 `BLOCK_SIZE`、相同 GPU 架构时，可以复用 CUBIN。

## 第五层：`_do_compile` 调用编译器

文件：

```text
python/triton/runtime/jit.py
```

关键代码：

```python
src = self.ASTSource(self, signature, constexprs, attrs)
kernel = self.compile(src, target=target, options=options.__dict__)
```

这里的 `ASTSource` 表示：

```text
我要从 Python 函数 AST 开始编译
```

`compile` 来自：

```text
python/triton/compiler/compiler.py
```

## 第六层：`compiler.compile` 走完整编译 pipeline

文件：

```text
python/triton/compiler/compiler.py
```

关键流程：

```python
backend.add_stages(stages, options, src.language)
module = src.make_ir(...)
for ext, compile_ir in stages:
    module = compile_ir(module, metadata)
return CompiledKernel(...)
```

小白理解：

`compiler.compile` 是总调度员。它不一定自己干活，而是让 backend 加入不同编译阶段。

在 CUDA backend 下，阶段大概是：

```text
ttir
  -> ttgir
  -> llir
  -> ptx
  -> cubin
```

这些名字可以这样理解：

- `ttir`：Triton 的基础中间表示。
- `ttgir`：带 GPU layout/thread 信息的 Triton IR。
- `llir`：LLVM IR。
- `ptx`：NVIDIA GPU 汇编文本。
- `cubin`：真正能被 CUDA driver 加载执行的二进制。

## 第七层：CUDA backend 定义编译阶段

文件：

```text
third_party/nvidia/backend/compiler.py
```

关键函数：

```python
def make_ttir(...)
def make_ttgir(...)
def make_llir(...)
def make_ptx(...)
def make_cubin(...)
```

小白理解：

CUDA backend 负责回答：

```text
我要怎么把 Triton kernel 变成 NVIDIA GPU 能执行的东西？
```

里面会调用很多 MLIR pass，例如：

```python
passes.ttir.add_convert_to_ttgpuir(...)
passes.ttgpuir.add_coalesce(...)
nvidia.passes.ttgpuir.add_to_llvmir(...)
passes.convert.add_nvvm_to_llvm(...)
```

这些 pass 的意思是：

```text
一步一步改写 IR，让它越来越接近 GPU 硬件能执行的形式。
```

## 第八层：生成 CUBIN 后，加载到 GPU

文件：

```text
python/triton/compiler/compiler.py
third_party/nvidia/backend/driver.py
```

编译完成返回：

```python
CompiledKernel(...)
```

它里面有：

```python
self.kernel = self.asm["cubin"]
```

当真正要运行时：

```python
driver.active.utils.load_binary(...)
```

加载 CUBIN，拿到 CUDA function handle。

## 第九层：真正 launch

文件：

```text
third_party/nvidia/backend/driver.py
```

关键类：

```python
class CudaLauncher:
    def __call__(...):
        self.launch(...)
```

`self.launch` 最终来自 C 扩展：

```text
third_party/nvidia/backend/driver.c
```

最终做的事情就是 CUDA driver API 的 kernel launch。

## 一张完整调用栈图

```text
python/tutorials/01-vector-add.py
  add_kernel[grid](...)

python/triton/runtime/jit.py
  jit()
  JITFunction.__init__()
  KernelInterface.__getitem__()
  JITFunction.run()
  JITFunction._do_compile()

python/triton/compiler/compiler.py
  ASTSource(...)
  compile(...)
  src.make_ir(...)
  ast_to_ttir(...)
  backend.add_stages(...)
  CompiledKernel(...)

python/triton/compiler/code_generator.py
  ast_to_ttir()
  CodeGenerator.visit_FunctionDef()
  CodeGenerator.visit_Assign()
  CodeGenerator.visit_Call()

python/triton/language/core.py
  tl.program_id()
  tl.arange()
  tl.load()
  tl.store()

third_party/nvidia/backend/compiler.py
  CUDABackend.make_ttir()
  CUDABackend.make_ttgir()
  CUDABackend.make_llir()
  CUDABackend.make_ptx()
  CUDABackend.make_cubin()

third_party/nvidia/backend/driver.py
  CudaDriver.get_current_target()
  CudaLauncher.__call__()

third_party/nvidia/backend/driver.c
  CUDA driver launch

GPU
  execute kernel
```

