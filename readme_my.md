# 源码同步、Full Debug 构建与 CUDA 验证

## Summary

- 先同步最新 `triton-lang/triton.git` 到 `triton_speed/develop`。
- 同步后读取最新 `cmake/llvm-info.json` 的 `llvm_hash`，再把 `/home/moffett/workspace/llvm-project_gpu_triton` checkout 到对应 LLVM commit 并 Full Debug 编译。
- 使用 `/home/moffett/workspace/pytorch` 的最新 PyTorch `main`，源码安装到当前 Conda 环境。
- Triton、PyTorch、LLVM 都用源码路径和 debug 符号，保证可用 gdb/lldb 进入源码。
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

3. 读取最新 LLVM hash：

   ```bash
   cd /home/moffett/workspace/triton_speed
   LLVM_HASH=$(python -c "import json; print(json.load(open('cmake/llvm-info.json'))['llvm_hash'])")
   echo "$LLVM_HASH"
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
     -DLLVM_ENABLE_ASSERTIONS=ON \
     -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
     -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"

   ninja -C "$LLVM_BUILD_DIR"
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

6. Full Debug 源码安装 Triton：

   ```bash
   cd /home/moffett/workspace/triton_speed

   python -m pip install -r python/requirements.txt
   python -m pip install -r python/test-requirements.txt

   LLVM_BUILD_DIR=/home/moffett/workspace/llvm-project_gpu_triton/build_${LLVM_HASH}_debug

   DEBUG=1 \
   TRITON_BUILD_WITH_CLANG_LLD=1 \
   TRITON_BUILD_WITH_CCACHE=0 \
   LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
   LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
   LLVM_SYSPATH=$LLVM_BUILD_DIR \
   python -m pip install -e . --no-build-isolation -v
   ```

## Test Plan

- LLVM NVPTX case：

  ```bash
  "$LLVM_BUILD_DIR/bin/llvm-lit" -sv \
    /home/moffett/workspace/llvm-project_gpu_triton/llvm/test/CodeGen/NVPTX/simple-call.ll
  ```

- 确认 PyTorch/Triton 都来自源码：

  ```bash
  python -c "import torch, triton, sys; print(sys.executable); print(torch.__file__); print(torch.version.cuda); print(triton.__file__); print(triton.__version__)"
  ```

- 确认 CUDA backend：

  ```bash
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
  ```

- 跑指定 Triton case：

  ```bash
  cd /home/moffett/workspace/triton_speed
  TRITON_INTERPRET=0 python python/tutorials/01-vector-add.py
  ```

## Assumptions

- `triton_speed` Conda 环境允许重建，并使用 Python 3.13。
- PyTorch 使用 `/home/moffett/workspace/pytorch` 的最新 `origin/main`。
- Full Debug 是明确要求；构建会明显更慢、产物更大。
- 如果 `torch.cuda.is_available()` 为 false，则问题归类为 CUDA/driver/GPU 环境问题，而不是 Triton 源码安装成功与否。
