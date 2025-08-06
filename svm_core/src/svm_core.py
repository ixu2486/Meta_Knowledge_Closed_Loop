# svm_core.py
"""
🔒 RetryIX SVM Core Module (OpenCL 2.0 Compatible)
提供 OpenCL SVM 記憶體分配與 map/unmap 操作的高層安全封裝。
✅ 適合語義 AI 使用，可開源提供非商業研究用途。
"""

import ctypes
from ctypes import c_void_p, c_size_t, c_uint, c_int
import numpy as np

# OpenCL 常數定義
CL_MEM_READ_WRITE = 1 << 0
CL_MEM_SVM_FINE_GRAIN_BUFFER = 1 << 10
CL_SUCCESS = 0
CL_MAP_WRITE = 1 << 1
CL_MAP_READ = 1 << 0

class RetryIXSVM:
    def __init__(self, lib_path: str):
        self.cl = ctypes.CDLL(lib_path)
        self._load_functions()

    def _load_functions(self):
        self.cl.clSVMAlloc.restype = c_void_p
        self.cl.clSVMAlloc.argtypes = [c_void_p, c_uint, c_size_t, c_uint]

        self.cl.clSVMFree.restype = None
        self.cl.clSVMFree.argtypes = [c_void_p, c_void_p]

    def alloc(self, context_ptr: c_void_p, size: int, flags: int = CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER) -> c_void_p:
        ptr = self.cl.clSVMAlloc(context_ptr, flags, size, 0)
        if not ptr:
            raise RuntimeError("SVM allocation failed")
        return ptr

    def free(self, context_ptr: c_void_p, svm_ptr: c_void_p):
        self.cl.clSVMFree(context_ptr, svm_ptr)

    def map(self, queue_ptr: c_void_p, svm_ptr: c_void_p, size: int, flags: int = CL_MAP_WRITE | CL_MAP_READ):
        err = self.cl.clEnqueueSVMMap(queue_ptr, True, flags, svm_ptr, c_size_t(size), 0, None, None)
        if err != CL_SUCCESS:
            raise RuntimeError(f"SVM map failed: error code {err}")

    def unmap(self, queue_ptr: c_void_p, svm_ptr: c_void_p):
        err = self.cl.clEnqueueSVMUnmap(queue_ptr, svm_ptr, 0, None, None)
        if err != CL_SUCCESS:
            raise RuntimeError(f"SVM unmap failed: error code {err}")
