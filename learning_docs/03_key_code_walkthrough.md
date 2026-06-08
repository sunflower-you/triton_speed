# 03. 关键代码逐段讲解

本文挑最重要的代码讲，不试图把整个 Triton 每一行都讲完。

## 1. `@triton.jit`：普通函数变成 JITFunction

文件：

```text
python/triton/runtime/jit.py
```

核心：

```python
def jit(fn=None, ...):
    def decorator(fn):
        if knobs.runtime.interpret:
            return InterpretedFunction(...)
        else:
            return JITFunction(fn, ...)
```

小白理解：

`@triton.jit` 是一个装饰器。它把你写的 Python 函数包装成 `JITFunction`。

普通 Python 函数：

```text
调用后马上执行 Python 代码
```

Triton JIT 函数：

```text
第一次调用时先编译，再在 GPU 上执行
```

## 2. `JITFunction.run`：运行入口

文件：

```text
python/triton/runtime/jit.py
```

关键逻辑：

```python
kernel = kernel_cache.get(key, None)
if kernel is None:
    kernel = self._do_compile(...)
...
kernel.run(...)
```

小白理解：

Triton 运行 kernel 前会问：

```text
这个 kernel 当前参数组合以前编译过吗？
```

如果编译过，直接拿缓存执行。

如果没编译过，进入 `_do_compile`。

## 3. `ASTSource`：告诉编译器“源码是 Python AST”

文件：

```text
python/triton/compiler/compiler.py
```

核心：

```python
class ASTSource:
    def make_ir(...):
        from .code_generator import ast_to_ttir
        return ast_to_ttir(...)
```

小白理解：

Triton 编译器不直接执行你的 Python 函数，而是读它的 AST。

AST 可以理解成：

```text
Python 代码的树形结构
```

例如：

```python
output = x + y
```

AST 会表达成：

```text
赋值语句
  左边：output
  右边：加法
    左操作数：x
    右操作数：y
```

`ast_to_ttir` 会把这棵树转成 Triton IR。

## 4. `CodeGenerator`：把 Python 语法翻译成 Triton IR

文件：

```text
python/triton/compiler/code_generator.py
```

重要方法：

```python
visit_FunctionDef
visit_Assign
visit_Call
```

它继承自：

```python
ast.NodeVisitor
```

小白理解：

Python AST 是一棵树，`CodeGenerator` 就像一个“遍历树的人”。

它看到不同节点，会执行不同函数：

```text
看到函数定义 -> visit_FunctionDef
看到赋值语句 -> visit_Assign
看到函数调用 -> visit_Call
```

例如教程里：

```python
pid = tl.program_id(axis=0)
```

大概会走：

```text
visit_Assign
  -> visit_Call
    -> tl.program_id
      -> semantic.program_id
        -> 创建 IR 操作
```

## 5. `tl.program_id`

文件：

```text
python/triton/language/core.py
```

核心：

```python
@builtin
def program_id(axis, _semantic=None):
    axis = _unwrap_if_constexpr(axis)
    return _semantic.program_id(axis)
```

小白理解：

`tl.program_id(0)` 不是普通 Python 算法，它会生成一个 IR 节点，表示：

```text
运行时读取当前 program 在 axis=0 上的编号
```

如果 grid 是：

```python
grid = (97,)
```

那就会启动 97 个 program，每个 program 的 `pid` 分别是 0 到 96。

## 6. `tl.arange`

文件：

```text
python/triton/language/core.py
```

核心：

```python
@builtin
def arange(start, end, _semantic=None):
    return _semantic.arange(start, end)
```

小白理解：

`tl.arange(0, BLOCK_SIZE)` 不是 CPU 上的 Python list。

它表示 GPU program 内部的一组向量下标：

```text
[0, 1, 2, ..., BLOCK_SIZE - 1]
```

在 vector-add 里：

```python
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

如果：

```text
pid = 2
BLOCK_SIZE = 1024
```

那么：

```text
block_start = 2048
offsets = [2048, 2049, ..., 3071]
```

## 7. `tl.load` 和 `tl.store`

文件：

```text
python/triton/language/core.py
```

核心：

```python
def load(pointer, mask=None, other=None, ...):
    return _semantic.load(...)

def store(pointer, value, mask=None, ...):
    return _semantic.store(...)
```

小白理解：

`tl.load(x_ptr + offsets, mask=mask)` 的意思是：

```text
从 x_ptr 指向的 GPU 内存里，读取 offsets 对应的那些元素。
mask 为 False 的地方不要读。
```

`tl.store(output_ptr + offsets, output, mask=mask)` 的意思是：

```text
把 output 写到 output_ptr + offsets 这些地址。
mask 为 False 的地方不要写。
```

为什么需要 mask？

因为数据长度不一定刚好是 `BLOCK_SIZE` 的整数倍。

假设 `n_elements=98432`，`BLOCK_SIZE=1024`。

最后一个 program 可能只有一部分下标有效，超过范围的下标必须屏蔽。

## 8. CUDA backend 的编译 pipeline

文件：

```text
third_party/nvidia/backend/compiler.py
```

核心阶段：

```python
make_ttir
make_ttgir
make_llir
make_ptx
make_cubin
```

小白理解：

这像翻译一本书：

```text
Python 写法
  -> Triton 自己的语言
  -> 带 GPU 线程和内存布局的信息
  -> LLVM 语言
  -> NVIDIA PTX 汇编
  -> GPU 可以加载的二进制
```

### `make_ttir`

做通用优化。

比如：

- inline
- canonicalize
- common subexpression elimination
- loop unroll

### `make_ttgir`

加入 GPU 相关信息。

比如：

- layout
- thread locality
- coalescing
- pipeline
- matmul acceleration

### `make_llir`

把 TritonGPU IR 降到 LLVM IR。

这一步开始接近 LLVM/MLIR 世界。

### `make_ptx`

调用 LLVM NVPTX backend，生成 PTX 文本。

PTX 类似 NVIDIA GPU 的汇编语言。

### `make_cubin`

调用 `ptxas`：

```text
PTX -> CUBIN
```

CUBIN 是 CUDA driver 可以加载的二进制。

## 9. `CompiledKernel`

文件：

```text
python/triton/compiler/compiler.py
```

核心：

```python
class CompiledKernel:
    self.asm = ...
    self.kernel = self.asm[binary_ext]
```

小白理解：

编译完以后，Triton 得到一个 `CompiledKernel` 对象。

它里面保存：

- metadata
- 不同阶段 IR
- 最终 cubin
- runtime launch 需要的信息

真正运行前，它会调用：

```python
driver.active.utils.load_binary(...)
```

把 cubin 加载进 CUDA driver。

## 10. `CudaLauncher`

文件：

```text
third_party/nvidia/backend/driver.py
```

核心：

```python
class CudaLauncher:
    def __call__(...):
        self.launch(...)
```

小白理解：

`CudaLauncher` 负责把 Python 里的参数整理成 CUDA driver 能理解的形状。

比如：

- tensor 指针变成 `CUdeviceptr`
- constexpr 不作为 kernel 参数传入
- tuple 参数展开
- 分配 scratch memory
- 传 gridX/gridY/gridZ
- 传 CUDA stream

最后调用 C 扩展里的 `launch`。

## 11. 最终 C 层

文件：

```text
third_party/nvidia/backend/driver.c
```

这是 Python 和 CUDA Driver API 的桥。

小白可以暂时这样理解：

```text
Python 不能直接高效调用 CUDA driver 的所有细节，
所以 Triton 写了一层 C 扩展。
```

