# 05. `triton.runtime.driver.active.get_active_torch_device()` 调用栈

本文只看教程里常见的这一行：

```python
DEVICE = triton.runtime.driver.active.get_active_torch_device()
```

它的作用是：找到当前 Triton 选中的 GPU backend，然后返回一个 PyTorch 能用的 device，例如：

```python
torch.device("cuda:0")
```

## 先看完整路径

第一次执行时，调用链大概是：

```text
triton.runtime.driver
  -> DriverConfig.active
    -> DriverConfig.default
      -> _create_driver()
        -> 找到所有已注册 backend
        -> 调用每个 backend.driver.is_active()
        -> 选择唯一 active 的 driver class
        -> 实例化 CudaDriver() 或 HIPDriver()
    -> 缓存到 self._active
  -> active_driver.get_active_torch_device()
    -> torch.device("cuda", active_driver.get_current_device())
```

如果之前已经访问过 `triton.runtime.driver.active`，driver 已经缓存在 `self._active` 里，后续就会短很多：

```text
DriverConfig.active
  -> return self._active
  -> get_active_torch_device()
```

## 第一层：`triton.runtime.driver` 是什么

文件：

```text
python/triton/runtime/__init__.py
```

关键代码：

```python
from .driver import driver
```

所以用户写的：

```python
triton.runtime.driver
```

不是 `python/triton/runtime/driver.py` 这个模块，而是从里面 import 出来的 `driver` 对象。

这个对象定义在：

```text
python/triton/runtime/driver.py
```

关键代码：

```python
driver = DriverConfig()
```

小白理解：

- `DriverConfig` 是一个“当前 driver 管理器”。
- 它负责懒加载默认 driver。
- 它也允许以后通过 `set_active()` 手动切换 active driver。

## 第二层：访问 `.active`

文件：

```text
python/triton/runtime/driver.py
```

关键代码：

```python
@property
def active(self) -> DriverBase:
    if self._active is None:
        self._active = self.default
    return self._active
```

这里 `.active` 是一个 Python property，不是普通字段。

第一次访问时：

```text
self._active is None
```

所以会继续访问：

```python
self.default
```

## 第三层：访问 `.default`

同一个文件：

```text
python/triton/runtime/driver.py
```

关键代码：

```python
@property
def default(self) -> DriverBase:
    if self._default is None:
        self._default = _create_driver()
    return self._default
```

第一次访问时：

```text
self._default is None
```

所以会调用：

```python
_create_driver()
```

这一步才是真正选择 CUDA / HIP backend 的地方。

## 第四层：`_create_driver()` 怎么选 backend

文件：

```text
python/triton/runtime/driver.py
```

关键逻辑：

```python
selected = os.environ.get("TRITON_DEFAULT_BACKEND", None)
```

如果用户设置了环境变量：

```bash
TRITON_DEFAULT_BACKEND=nvidia
```

或者：

```bash
TRITON_DEFAULT_BACKEND=amd
```

Triton 会尝试直接使用指定 backend。

如果没有设置，就扫描所有 backend：

```python
active_drivers = [x.driver for x in backends.values() if x.driver.is_active()]
```

然后要求 active driver 数量必须刚好是 1：

```python
if len(active_drivers) != 1:
    raise RuntimeError(...)
```

小白理解：

- 没有 GPU：可能是 `0 active drivers`。
- CUDA 和 HIP 同时被判断为 active：可能是 `2 active drivers`。
- 正常情况下应该刚好选出一个。

## 第五层：`backends` 从哪里来

文件：

```text
python/triton/backends/__init__.py
```

关键代码：

```python
backends: dict[str, Backend] = _discover_backends()
```

默认会通过 Python entry points 找到 Triton backend。

每个 backend 包含两个东西：

```python
@dataclass(frozen=True)
class Backend:
    compiler: Type[BaseBackend]
    driver: Type[DriverBase]
```

也就是说，一个 backend 至少有：

- `compiler`：负责把 Triton IR 编译到目标 GPU。
- `driver`：负责设备查询、加载二进制、launch kernel。

对于当前代码树，常见 backend 是：

```text
third_party/nvidia/backend
third_party/amd/backend
```

## 第六层：NVIDIA 怎么判断 active

文件：

```text
third_party/nvidia/backend/driver.py
```

NVIDIA driver class 是：

```python
class CudaDriver(GPUDriver):
    ...
```

它的 active 判断：

```python
@staticmethod
def is_active():
    return _cuda_driver_is_active()
```

`_cuda_driver_is_active()` 大概做这些事：

```text
加载 libcuda.so.1
  -> cuInit(0)
  -> cuDeviceGetCount(&count)
  -> count > 0
```

所以 NVIDIA backend active 的含义是：

```text
系统能加载 CUDA driver，并且 CUDA driver 能看到至少一张 GPU。
```

## 第七层：AMD/HIP 怎么判断 active

文件：

```text
third_party/amd/backend/driver.py
```

AMD driver class 是：

```python
class HIPDriver(GPUDriver):
    ...
```

它的 active 判断：

```python
@staticmethod
def is_active():
    try:
        import torch
        return torch.cuda.is_available() and (torch.version.hip is not None)
    except ImportError:
        return False
```

注意：PyTorch 里 HIP 设备接口也挂在 `torch.cuda` 下面。

## 第八层：实例化具体 driver

如果选中 NVIDIA，`_create_driver()` 最后会做：

```python
return active_drivers[0]()
```

也就是实例化：

```python
CudaDriver()
```

文件：

```text
third_party/nvidia/backend/driver.py
```

关键代码：

```python
def __init__(self):
    self.utils = CudaUtils()
    self.launcher_cls = CudaLauncher
    if sys.modules.get("torch") is not None:
        super().__init__()
    else:
        self.get_device_capability = self._get_device_capability
        self.get_current_stream = self._get_current_stream
        self.get_current_device = self._get_current_device
        self.set_current_device = self._set_current_device
```

这里有一个重要分支：

- 如果 `torch` 已经 import 过，就走 `GPUDriver.__init__()`，绑定 PyTorch 的当前设备和 stream 接口。
- 如果没有 import torch，就走 Triton 自己封装的 CUDA driver API。

`GPUDriver.__init__()` 在：

```text
python/triton/backends/driver.py
```

关键代码：

```python
self.get_current_device = torch.cuda.current_device
self.set_current_device = torch.cuda.set_device
```

## 第九层：真正返回 `DEVICE`

NVIDIA 实现：

```text
third_party/nvidia/backend/driver.py
```

关键代码：

```python
def get_active_torch_device(self):
    import torch
    return torch.device("cuda", self.get_current_device())
```

AMD/HIP 实现：

```text
third_party/amd/backend/driver.py
```

关键代码：

```python
def get_active_torch_device(self):
    import torch
    # when using hip devices, the device string in pytorch is "cuda"
    return torch.device("cuda", self.get_current_device())
```

所以无论 NVIDIA 还是 HIP，返回给 PyTorch 的 device 字符串通常都是：

```text
cuda:<current_device>
```

例如当前 PyTorch device 是 0：

```python
torch.device("cuda", 0)
```

等价于：

```python
torch.device("cuda:0")
```

## 一句话总结

这行代码不是直接问 PyTorch “当前设备是什么”，而是先通过 Triton 的 backend 系统确定当前 active driver，再让这个 driver 用自己的设备接口返回 PyTorch device。

最核心的路径是：

```text
DriverConfig.active
  -> _create_driver()
  -> CudaDriver/HIPDriver.is_active()
  -> CudaDriver/HIPDriver()
  -> get_active_torch_device()
  -> torch.device("cuda", current_device)
```

