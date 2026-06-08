# 01. 项目地图

## 先用一句话看全局

Triton 让你这样写 GPU 程序：

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    ...

add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
```

表面上是 Python，实际流程是：

```text
Python kernel 函数
  -> Triton JIT 包装
  -> Python AST
  -> Triton IR / TritonGPU IR
  -> LLVM IR
  -> PTX
  -> CUBIN
  -> CUDA Driver launch
  -> GPU 执行
```

## 目录分别干什么

### `python/triton`

这是 Python 用户最常接触的层。

重要文件：

- `python/triton/runtime/jit.py`
  - 处理 `@triton.jit`
  - 处理 `kernel[grid](...)`
  - 管理编译缓存
  - 调用编译器
  - 调用 GPU launcher

- `python/triton/compiler/compiler.py`
  - 编译总控
  - 把一个 kernel 依次编译成不同阶段的 IR
  - 管理 cache、dump、metadata

- `python/triton/compiler/code_generator.py`
  - 把 Python AST 转成 Triton IR
  - 你写的 `pid = tl.program_id(0)`、`x = tl.load(...)` 都会在这里被访问和转换

- `python/triton/language/core.py`
  - `tl.program_id`
  - `tl.arange`
  - `tl.load`
  - `tl.store`
  - `tl.constexpr`
  - Triton 语言里用户可调用的基础 API

- `python/triton/language/semantic.py`
  - 真正把 `tl.load`、`tl.store` 等语义变成 IR builder 操作

### `third_party/nvidia`

NVIDIA CUDA backend。

重要文件：

- `third_party/nvidia/backend/compiler.py`
  - CUDA 后端编译 pipeline
  - TTIR -> TTGIR -> LLVM IR -> PTX -> CUBIN
  - 调用 `ptxas`

- `third_party/nvidia/backend/driver.py`
  - CUDA runtime/driver 接口
  - 加载 CUBIN
  - 构造 launcher
  - 最终调用 CUDA driver launch kernel

- `third_party/nvidia/backend/driver.c`
  - C 层 CUDA driver API 封装
  - Python launcher 最终会调用这里编译出来的扩展模块

### `third_party/amd`

AMD backend，结构和 NVIDIA backend 类似，但目标是 AMD GPU。

如果你当前主要看 CUDA，可以先跳过这里。

### `lib`

这是 C++/MLIR 编译器核心。

重要目录：

- `lib/Dialect`
  - Triton 自己定义的 MLIR Dialect
  - 可以理解成 Triton 编译器的“中间语言语法”

- `lib/Conversion`
  - IR 降级逻辑
  - 比如 TritonGPU IR 怎么变成 LLVM IR

- `lib/Analysis`
  - 编译器分析
  - 比如内存、轴信息、别名、buffer region

- `lib/Tools`
  - layout、swizzle 等辅助工具

### `include`

C++ 头文件，给 `lib` 里面的 C++ 代码用。

### `test` / `unittest`

编译器测试。很多 `.mlir` 文件用于测试某个 pass 是否把 IR 改成预期样子。

### `python/tutorials`

用户教程。最适合小白从这里开始。

当前最关键文件：

- `python/tutorials/01-vector-add.py`

它是最小闭环：

```text
写 kernel -> 编译 -> 跑 CUDA -> 和 torch 比结果 -> benchmark
```

### `cmake`

构建配置。

重要文件：

- `cmake/llvm-info.json`
  - 指定 Triton 当前需要的 LLVM commit
  - 这就是为什么安装时要先看 `llvm_hash`

### `setup.py`

源码安装入口。

执行：

```bash
python -m pip install -e . --no-build-isolation -v
```

会触发：

```text
setup.py
  -> CMake
  -> 编译 C++/MLIR 扩展
  -> 生成 python/triton/_C/libtriton.so
```

`libtriton.so` 是 Python 和 C++ 编译器核心之间的桥。

## 你现在最应该先看哪些文件

按顺序：

1. `python/tutorials/01-vector-add.py`
2. `python/triton/runtime/jit.py`
3. `python/triton/compiler/compiler.py`
4. `python/triton/compiler/code_generator.py`
5. `python/triton/language/core.py`
6. `third_party/nvidia/backend/compiler.py`
7. `third_party/nvidia/backend/driver.py`
8. `lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp`
9. `lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp`

