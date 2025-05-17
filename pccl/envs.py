import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

UCX_HOME = "/opt/hpcx/ucx/"
USE_UCX = os.path.exists(UCX_HOME)

USE_IBVERBS = True

CURRENT_DIR = Path(__file__).parent.parent.absolute().as_posix()
USE_CUDA = torch.cuda.is_available() and torch.version.cuda is not None
USE_HIP  = torch.cuda.is_available() and torch.version.hip is not None
CUDA_VERSION = None
HIP_VERSION = None
if USE_CUDA:
    CUDA_VERSION = torch.version.cuda
if USE_HIP:
    HIP_VERSION = torch.version.hip

FLAGS = ['-O2']

LIBRARIES = []

LIBRARY_DIRS = []

INCLUDE_DIRS = [
    f"{CURRENT_DIR}/include",
    f"{CURRENT_DIR}/thirdparty/json/include",
    f"{CURRENT_DIR}/thirdparty/cpp-httplib",
]

CSRCS = [
    "src/python.cc",
    "src/device.cc",
    "src/config.cc",
    "src/runtime.cc",
    "src/utils.cc",
    "src/component/endpoint.cc",
    "src/component/operator.cc",
    "src/component/profile.cc",
    "src/component/proxy.cc",
    "src/plugin/ib.cc",
    "src/plugin/sock.cc",
]

CUSRCS = [
    "src/cuda/reduce_kernel.cu",
    "src/cuda/kernel.cu",
    "src/cuda/packet.cu",
    "src/cuda/connection_context.cu",
    "src/cuda/connection.cu",
    "src/cuda/registered_memory.cu",
]

if USE_CUDA:
    assert CUDA_VERSION >= "12.4", "CUDA 12.4 or higher is required"
    CUDA_ARCH = "8.9"
    os.environ["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH
    FLAGS.append('-DUSE_CUDA')

    LIBRARY_DIRS.append(f'{CUDA_HOME}/lib64')
    LIBRARY_DIRS.append(f'{CUDA_HOME}/targets/x86_64-linux/lib/stubs/')
    INCLUDE_DIRS.append(f'{CUDA_HOME}/include')
elif USE_HIP:
    HIP_ARCH = "gfx1100"
    os.environ["TORCH_HIP_ARCH_LIST"] = HIP_ARCH
    FLAGS.append('-DUSE_HIP')

    LIBRARY_DIRS.append(f'{ROCM_HOME}/lib')
    INCLUDE_DIRS.append(f'{ROCM_HOME}/include')
else:
    assert False, "No GPU detected, does not support now"

if USE_UCX:
    LIBRARIES.append('ucp')
    LIBRARY_DIRS.append(f'{UCX_HOME}/lib')
    INCLUDE_DIRS.append(f'{UCX_HOME}/include')

if USE_IBVERBS:
    FLAGS.append("-DUSE_IBVERBS")
