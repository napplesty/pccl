import os
import ctypes
import torch
from pccl.envs import USE_CUDA, USE_HIP, UCX_HOME, CUDA_HOME, CUDA_VERSION

if USE_CUDA:
    cuda_lib_path = f'{CUDA_HOME}-{CUDA_VERSION}/targets/x86_64-linux/lib/stubs/libcuda.so'
    if os.path.exists(cuda_lib_path):
        ctypes.CDLL(cuda_lib_path, mode=ctypes.RTLD_GLOBAL)

from _pccl import *

__all__ = []
