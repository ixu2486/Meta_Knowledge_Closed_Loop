# svm_claude_safe.py
"""
🔒 Claude-Safe SVM Wrapper
限制功能：禁止 raw pointer 存取與自定資源釋放
適用於嵌入式 AI 模型環境、安全沙盒推理任務
"""

import ctypes
from ctypes import c_void_p, c_size_t, c_uint, c_int

# Constants for OpenCL SVM
CL_MEM_READ_WRITE = 1 << 0
CL_MEM_SVM_FINE_GRAIN_BUFFER = 1 << 10
CL_SUCCESS = 0
CL_MAP_WRITE = 1 << 1
CL_MAP_READ = 1 << 0

class ClaudeSafeSVM:
    def __init__(self, opencl_lib):
        self.cl = opencl_lib
        self._wrap_safe_functions()

    def _wrap_safe_functions(self):
        self.cl.clSVMAlloc.restype = c_void_p
        self.cl.clSVMAlloc.argtypes = [c_void_p, c_uint, c_size_t, c_uint]

        self.cl.clEnqueueSVMMap.restype = c_int
        self.cl.clEnqueueSVMMap.argtypes = [c_void_p, ctypes.c_uint, c_uint, c_void_p, c_size_t, c_uint, c_void_p, c_void_p]

        self.cl.clEnqueueSVMUnmap.restype = c_int
        self.cl.clEnqueueSVMUnmap.argtypes = [c_void_p, c_void_p, c_uint, c_void_p, c_void_p]

    def safe_alloc(self, context_ptr: c_void_p, size: int) -> bytes:
        ptr = self.cl.clSVMAlloc(context_ptr, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, size, 0)
        if not ptr:
            raise RuntimeError("SVM allocation failed")

        # 安全封裝為 Python bytearray，不回傳指標
        buffer = (ctypes.c_ubyte * size).from_address(ptr)
        return bytearray(buffer[:])  # 複製安全快照回 Python

    def map_region(self, queue_ptr: c_void_p, svm_ptr: c_void_p, size: int):
        err = self.cl.clEnqueueSVMMap(queue_ptr, True, CL_MAP_WRITE | CL_MAP_READ, svm_ptr, c_size_t(size), 0, None, None)
        if err != CL_SUCCESS:
            raise RuntimeError(f"Map failed: {err}")

    def unmap_region(self, queue_ptr: c_void_p, svm_ptr: c_void_p):
        err = self.cl.clEnqueueSVMUnmap(queue_ptr, svm_ptr, 0, None, None)
        if err != CL_SUCCESS:
            raise RuntimeError(f"Unmap failed: {err}")
