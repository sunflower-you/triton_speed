# 源码同步、Full Debug 构建与 CUDA 验证

## Summary

- 先同步最新 `triton-lang/triton.git` 到 `triton_speed/develop`。
- 同步后读取最新 `cmake/llvm-info.json` 的 `llvm_hash`，再把 `/home/moffett/workspace/llvm-project_gpu_triton` checkout 到对应 LLVM commit 并 Full Debug 编译。
- 使用 `/home/moffett/workspace/pytorch` 的最新 PyTorch `main` 尝试源码 Debug 安装；本机 CUDA Toolkit 12.1 与当前 PyTorch main 的 CUDA C++20 编译路径不兼容，实际改用 CUDA nightly wheel 作为运行依赖。
- Triton 和 LLVM 使用源码路径和 debug 符号，保证可用 gdb/lldb 进入 Triton/LLVM 源码。PyTorch 源码已同步到最新 main，可用于源码阅读；当前环境中的可运行 PyTorch 来自 nightly wheel。
- 最后跑一个 LLVM NVPTX case 和 Triton CUDA backend 的 `python/tutorials/01-vector-add.py`。

## Key Changes

- README 先写完整中文步骤，覆盖：
  - Triton upstream 同步
  - 从 `llvm-info.json` 读取 `llvm_hash`
  - LLVM Full Debug build
  - PyTorch 最新源码 Full Debug 安装
  - Triton Full Debug editable 安装
  - CUDA backend 验证命令

## Implementation Steps

1. 准备当前环境：

   ```bash
   conda env remove -n triton_speed -y
   conda create -n triton_speed python=3.13 cmake ninja git -y
   conda activate triton_speed
   python -m pip install --upgrade pip setuptools wheel
   ```

   实际环境：

   ```bash
   /opt/conda/envs/triton_speed/bin/python --version
   # Python 3.13.13
   ```

2. 同步 Triton 最新上游：

   ```bash
   cd /home/moffett/workspace/triton_speed
   git remote add upstream https://github.com/triton-lang/triton.git || true
   git fetch upstream
   git fetch origin
   git checkout develop
   git merge upstream/main
   git push origin develop
   ```

   实际结果：

   ```bash
   git rev-parse HEAD
   # 334f00f5a455e6b3f256aeca43bde28b05f48931
   ```

   merge conflict 只出现在 `README.md` 头部。处理方式：保留 fork 的标题/logo，再接上 upstream 最新 README 内容，然后提交并 push `origin/develop`。

3. 读取最新 LLVM hash：

   ```bash
   cd /home/moffett/workspace/triton_speed
   LLVM_HASH=$(python -c "import json; print(json.load(open('cmake/llvm-info.json'))['llvm_hash'])")
   echo "$LLVM_HASH"
   ```

   实际 hash：

   ```bash
   LLVM_HASH=87717bf9f81f7b29466c5d9a30a3453bdfc93941
   ```

4. Full Debug 编译 LLVM：

   ```bash
   cd /home/moffett/workspace/llvm-project_gpu_triton
   git fetch origin
   git checkout "$LLVM_HASH"

   LLVM_BUILD_DIR=/home/moffett/workspace/llvm-project_gpu_triton/build_${LLVM_HASH}_debug

   cmake -G Ninja \
     -S /home/moffett/workspace/llvm-project_gpu_triton/llvm \
     -B "$LLVM_BUILD_DIR" \
     -DCMAKE_BUILD_TYPE=Debug \
     -DLLVM_CCACHE_BUILD=OFF \
     -DLLVM_ENABLE_ASSERTIONS=ON \
     -DCMAKE_C_COMPILER=clang \
     -DCMAKE_CXX_COMPILER=clang++ \
     -DLLVM_ENABLE_LLD=ON \
     -DLLVM_OPTIMIZED_TABLEGEN=ON \
     -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
     -DLLVM_ENABLE_ZSTD=OFF \
     -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld;clang" \
     -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
     -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
     -DCMAKE_INSTALL_PREFIX=/home/moffett/workspace/llvm-project_gpu_triton/install_${LLVM_HASH}_debug

   ninja -C "$LLVM_BUILD_DIR" -j64
   ```

   实际结果：LLVM Debug build 完成，产物目录：

   ```bash
   /home/moffett/workspace/llvm-project_gpu_triton/build_87717bf9f81f7b29466c5d9a30a3453bdfc93941_debug
   ```

5. 同步并源码安装最新 PyTorch：

   ```bash
   cd /home/moffett/workspace/pytorch
   git fetch origin
   git checkout -B main origin/main
   git submodule sync
   git submodule update --init --recursive

   python -m pip install -r requirements.txt
   DEBUG=1 USE_CUDA=1 BUILD_TEST=0 MAX_JOBS=32 python setup.py develop
   ```

   实际同步结果：

   ```bash
   git rev-parse HEAD
   # 8f0273ecb721fc56d80dd38c418c6f9f17e382bd
   ```

   实际源码构建尝试命令：

   ```bash
   conda run -n triton_speed python -m pip install -r requirements.txt

   conda run -n triton_speed env \
     DEBUG=1 USE_CUDA=1 BUILD_TEST=0 MAX_JOBS=64 USE_DISTRIBUTED=0 \
     CC=/usr/bin/gcc CXX=/usr/bin/g++ \
     python setup.py develop
   ```

   遇到的问题和处理：

   - 第一次失败：旧 `build/CMakeCache.txt` 记录了不存在的 `/opt/conda/envs/triton_speed/bin/cc` 和 `/opt/conda/envs/triton_speed/bin/c++`。
   - 处理：删除 PyTorch `build` 目录，显式指定 `CC=/usr/bin/gcc CXX=/usr/bin/g++`。

   ```bash
   rm -rf build
   ```

   - 第二次失败：内置 `flash_attention` CUDA 文件编译失败，错误来自 `third_party/cutlass` 的 `half`/`bfloat16` 转换。
   - 处理：关闭 PyTorch 内置 flash attention 和 mem efficient attention 后继续尝试。

   ```bash
   conda run -n triton_speed env \
     DEBUG=1 USE_CUDA=1 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 \
     BUILD_TEST=0 MAX_JOBS=64 USE_DISTRIBUTED=0 \
     CC=/usr/bin/gcc CXX=/usr/bin/g++ \
     python setup.py develop
   ```

   - 第三次失败：`ActivationPreluKernel.cu` 在 CUDA 12.1 + C++20 device code 下报 `std::apply` 不能从 device code 调用。
   - 尝试改用 GCC 13：

   ```bash
   rm -rf build
   conda run -n triton_speed env \
     DEBUG=1 USE_CUDA=1 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 \
     BUILD_TEST=0 MAX_JOBS=64 USE_DISTRIBUTED=0 \
     CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDAHOSTCXX=/usr/bin/g++-13 \
     python setup.py develop
   ```

   - GCC 13 被 CUDA 12.1 拦截：`gcc versions later than 12 are not supported`。
   - 再加 `CMAKE_CUDA_FLAGS=-allow-unsupported-compiler` 后，CUDA 12.1 前端仍无法解析 libstdc++ 13 头文件。

   结论：本机只有 CUDA Toolkit 12.1，而当前 PyTorch main 已要求 CUDA >= 12.1 且使用 C++20；在该机器工具链组合下，PyTorch CUDA 源码 Debug build 没有完成。为了继续验证 Triton CUDA backend，安装 CUDA nightly wheel 作为运行依赖：

   ```bash
   conda run -n triton_speed python -m pip install --pre torch \
     --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

   实际安装结果：

   ```bash
   torch==2.12.0.dev20260408+cu128
   torch.version.cuda == 12.8
   ```

6. Full Debug 源码安装 Triton：

   ```bash
   cd /home/moffett/workspace/triton_speed

   python -m pip install -r python/requirements.txt

   LLVM_BUILD_DIR=/home/moffett/workspace/llvm-project_gpu_triton/build_${LLVM_HASH}_debug

   DEBUG=1 \
   LLVM_SYSPATH=$LLVM_BUILD_DIR \
   MAX_JOBS=64 \
   python -m pip install -e . --no-build-isolation -v
   ```

   实际结果：

   ```bash
   triton==3.7.0+git334f00f5
   triton.__file__ == /home/moffett/workspace/triton_speed/python/triton/__init__.py
   ```

   注意：PyTorch nightly wheel 会依赖它自带的 `triton==3.7.0+git282c8251`。源码安装 Triton 后会覆盖 wheel 版 Triton，pip 会提示依赖版本不完全匹配，这是预期结果；目标就是验证当前 `triton_speed` 源码。

7. Tutorial 依赖：

   `python/tutorials/01-vector-add.py` 的 benchmark 绘图阶段需要 `matplotlib`，实际也安装了 `pandas`：

   ```bash
   conda run -n triton_speed python -m pip install matplotlib pandas
   ```

## Test Plan

- LLVM NVPTX case：

  ```bash
  "$LLVM_BUILD_DIR/bin/llvm-lit" -sv \
    /home/moffett/workspace/llvm-project_gpu_triton/llvm/test/CodeGen/NVPTX/simple-call.ll
  ```

- 确认 PyTorch/Triton 路径：

  ```bash
  python -c "import torch, triton, sys; print(sys.executable); print(torch.__file__); print(torch.version.cuda); print(triton.__file__); print(triton.__version__)"
  ```

  实际输出要点：

  ```text
  python /opt/conda/envs/triton_speed/bin/python
  torch 2.12.0.dev20260408+cu128 /opt/conda/envs/triton_speed/lib/python3.13/site-packages/torch/__init__.py 12.8
  triton 3.7.0 /home/moffett/workspace/triton_speed/python/triton/__init__.py
  ```

- 确认 CUDA backend：

  ```bash
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
  ```

  实际输出：

  ```text
  True
  NVIDIA GeForce RTX 3090
  ```

- 跑指定 Triton case：

  ```bash
  cd /home/moffett/workspace/triton_speed
  MPLBACKEND=Agg TRITON_INTERPRET=0 python python/tutorials/01-vector-add.py
  ```

  实际结果：

  ```text
  The maximum difference between torch and triton is 0.0
  vector-add-performance:
             size  Triton (GB/s)  Torch (GB/s)
  0        4096.0      12.000000     12.000000
  ...
  15  134217728.0     849.737435    849.737435
  ```

## Assumptions

- `triton_speed` Conda 环境允许重建，并使用 Python 3.13。
- PyTorch 使用 `/home/moffett/workspace/pytorch` 的最新 `origin/main`。
- Full Debug 是明确要求；LLVM/Triton 构建会明显更慢、产物更大。
- 当前机器驱动支持 CUDA 12.9，但本地 CUDA Toolkit 只有 12.1。PyTorch 最新 main 的源码 CUDA Debug build 需要更新的 Toolkit/兼容 host compiler 才能完成。
- 如果 `torch.cuda.is_available()` 为 false，则问题归类为 CUDA/driver/GPU 环境问题，而不是 Triton 源码安装成功与否。
