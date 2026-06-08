# 04. 调试入口和术语表

## 你现在环境的实际情况

当前环境里：

```text
Python: /opt/conda/envs/triton_speed/bin/python
Triton 源码: /home/moffett/workspace/triton_speed
LLVM 源码: /home/moffett/workspace/llvm-project_gpu_triton
LLVM Debug build: /home/moffett/workspace/llvm-project_gpu_triton/build_87717bf9f81f7b29466c5d9a30a3453bdfc93941_debug
PyTorch 源码: /home/moffett/workspace/pytorch
```

注意：

```text
Triton 和 LLVM 已经是源码 Debug 构建。
PyTorch 源码已同步，但当前可运行 torch 来自 nightly wheel。
```

原因见：

```text
readme_my.md
```

里面记录了 PyTorch source build 在 CUDA Toolkit 12.1 下遇到的问题。

## 推荐调试命令

### 跑 vector add

```bash
cd /home/moffett/workspace/triton_speed
conda run -n triton_speed env MPLBACKEND=Agg TRITON_INTERPRET=0 \
  python python/tutorials/01-vector-add.py
```

### 打开 IR dump

可以让 Triton 把中间 IR dump 出来：

```bash
cd /home/moffett/workspace/triton_speed
conda run -n triton_speed env \
  TRITON_INTERPRET=0 \
  TRITON_KERNEL_DUMP=1 \
  MPLBACKEND=Agg \
  python python/tutorials/01-vector-add.py
```

常见可观察内容：

- `.ttir`
- `.ttgir`
- `.llir`
- `.ptx`
- `.cubin`

如果你想学编译器，最重要的是看这些文件如何一步步变化。

## 建议断点

### Python 层

1. `python/triton/runtime/jit.py`

推荐断点：

```text
jit
JITFunction.__init__
JITFunction.run
JITFunction._do_compile
```

你会看到：

```text
@triton.jit 怎么包装函数
kernel[grid](...) 怎么进 run
什么时候查缓存
什么时候触发 compile
```

2. `python/triton/compiler/compiler.py`

推荐断点：

```text
ASTSource.make_ir
compile
CompiledKernel.__init__
CompiledKernel._init_handles
```

你会看到：

```text
编译 pipeline 怎么跑
生成的 metadata 是什么
cubin 什么时候加载
```

3. `python/triton/compiler/code_generator.py`

推荐断点：

```text
ast_to_ttir
CodeGenerator.visit_FunctionDef
CodeGenerator.visit_Assign
CodeGenerator.visit_Call
```

你会看到：

```text
Python AST 怎么被遍历
tl.program_id / tl.load / tl.store 怎么变成 IR
```

4. `third_party/nvidia/backend/compiler.py`

推荐断点：

```text
CUDABackend.make_ttir
CUDABackend.make_ttgir
CUDABackend.make_llir
CUDABackend.make_ptx
CUDABackend.make_cubin
```

你会看到：

```text
每个编译阶段输入输出是什么
ptxas 是怎么被调用的
```

5. `third_party/nvidia/backend/driver.py`

推荐断点：

```text
CudaDriver.get_current_target
CudaLauncher.__init__
CudaLauncher.__call__
```

你会看到：

```text
GPU 架构怎么识别
参数怎么打包
kernel 怎么 launch
```

### C++/MLIR 层

如果用 gdb/lldb，重点看：

```text
lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp
lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp
lib/Conversion/TritonGPUToLLVM/SPMDOpToLLVM.cpp
lib/Dialect/Triton/IR/Ops.cpp
lib/Dialect/TritonGPU/IR/Ops.cpp
```

这部分更难。建议先看 Python 调用栈，再看 IR dump，最后再进 C++ pass。

## 小白术语表

### Kernel

GPU 上执行的函数。

普通 Python 函数在 CPU 上执行，Triton kernel 在 GPU 上执行。

### JIT

Just-In-Time compilation，即“运行时才编译”。

Triton 不会在你定义函数时马上编译，而是在第一次调用 kernel 时，根据参数类型和 GPU 架构编译。

### Grid

启动多少个 program。

例子：

```python
grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
```

如果 `n_elements=98432`，`BLOCK_SIZE=1024`，grid 大约是 97。

意思是启动 97 个 Triton program。

### Program

Triton 的一个执行单位。

可以粗略理解成：

```text
一个 program 负责处理一小块数据
```

在 vector-add 里，一个 program 处理 1024 个元素。

### `tl.program_id`

当前 program 的编号。

### `tl.arange`

生成当前 program 内部的一段向量下标。

### `tl.constexpr`

编译期常量。

例如：

```python
BLOCK_SIZE: tl.constexpr
```

这表示 `BLOCK_SIZE` 在编译时就确定，不是普通运行时参数。

为什么重要？

因为编译器可以根据 `BLOCK_SIZE=1024` 做优化。

### TTIR

Triton IR。

刚从 Python AST 生成出来的 Triton 中间表示。

### TTGIR

Triton GPU IR。

比 TTIR 更接近 GPU，带有 layout、warp、thread、memory 等信息。

### LLVM IR

LLVM 的中间表示。

Triton 最终借助 LLVM 的 NVPTX backend 生成 NVIDIA PTX。

### PTX

NVIDIA GPU 的汇编文本。

### CUBIN

NVIDIA GPU 可加载执行的二进制。

### `ptxas`

NVIDIA 工具，把 PTX 编译成 CUBIN。

### Cache

Triton 会缓存编译结果。

同一个 kernel、同样参数类型、同样 constexpr、同样 GPU 架构，下一次可以直接复用。

## 小白学习路线

第一阶段，只看用户层：

```text
python/tutorials/01-vector-add.py
python/tutorials/03-matrix-multiplication.py
```

第二阶段，看 Python runtime：

```text
python/triton/runtime/jit.py
python/triton/compiler/compiler.py
```

第三阶段，看语言 API：

```text
python/triton/language/core.py
python/triton/language/semantic.py
```

第四阶段，看 CUDA backend：

```text
third_party/nvidia/backend/compiler.py
third_party/nvidia/backend/driver.py
```

第五阶段，看 C++/MLIR：

```text
lib/Dialect
lib/Conversion
lib/Analysis
```

不要一开始就从 C++ pass 看起，会很痛苦。先从一个能跑的 tutorial 出发，沿调用栈往下挖。

