import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

USE_IBVERBS = True

CURRENT_DIR = Path(__file__).parent.parent.absolute().as_posix()
USE_CUDA = True #torch.cuda.is_available() and torch.version.cuda is not None
USE_HIP  = torch.cuda.is_available() and torch.version.hip is not None
CUDA_VERSION = None
HIP_VERSION = None
if USE_CUDA:
    CUDA_VERSION = torch.version.cuda
if USE_HIP:
    HIP_VERSION = torch.version.hip

FLAGS = ['-O2', '-std=c++20']

LIBRARIES = []

LIBRARY_DIRS = []

INCLUDE_DIRS = [
    f"{CURRENT_DIR}/include",
]

CSRCS = [
    "csrc/config.cc",
    "csrc/python.cc",
]

if USE_CUDA:
    LIBRARY_DIRS.append(f'{CUDA_HOME}/lib64')
    LIBRARY_DIRS.append(f'{CUDA_HOME}/targets/x86_64-linux/lib/stubs/')
    INCLUDE_DIRS.append(f'{CUDA_HOME}/include')
elif USE_HIP:
    LIBRARY_DIRS.append(f'{ROCM_HOME}/lib')
    INCLUDE_DIRS.append(f'{ROCM_HOME}/include')

if USE_IBVERBS:
    FLAGS.append("-DUSE_IBVERBS")

