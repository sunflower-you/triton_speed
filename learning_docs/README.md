# Triton 小白学习文档

这个目录是用“understand-anything 式”的方式整理的：不先陷进每个细节，而是先回答 4 个问题。

1. 这个项目是干什么的？
2. 代码大概分成哪些层？
3. 一个最小例子 `01-vector-add.py` 从 Python 调用到 GPU 执行，中间发生了什么？
4. 如果我要调试，应该在哪些文件下断点？

建议阅读顺序：

1. [01_project_map.md](01_project_map.md)：先建立项目地图。
2. [02_vector_add_call_stack.md](02_vector_add_call_stack.md)：沿 `python/tutorials/01-vector-add.py` 看调用栈。
3. [03_key_code_walkthrough.md](03_key_code_walkthrough.md)：解释关键代码在做什么。
4. [04_debug_and_terms.md](04_debug_and_terms.md)：常见术语、调试入口、你现在环境里的实际情况。
5. [05_driver_active_call_stack.md](05_driver_active_call_stack.md)：解释 `triton.runtime.driver.active.get_active_torch_device()` 怎么选 backend 和返回 device。

一句话理解 Triton：

> Triton 是一个“用 Python 写 GPU kernel，然后自动编译成 CUDA/AMD GPU 可执行二进制”的编译器项目。

你在教程里写的不是普通 Python 函数，而是一段会被 Triton 编译器拿去分析、降级、优化、生成 PTX/CUBIN，并最终丢给 GPU 执行的 kernel。
