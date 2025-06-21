import sys
import importlib.util
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

ROOT_DIR = Path(__file__).parent

envs = load_module_from_path('envs', ROOT_DIR / 'pccl' / 'envs.py')

setup(
    name='pccl',
    version='1.0.0',
    packages=find_packages(include=['pccl']),
    include_package_data=True,
    ext_modules=[
        CppExtension(
            name='_pccl',
            sources=envs.CSRCS,
            extra_compile_args=envs.FLAGS,
            include_dirs=envs.INCLUDE_DIRS,
            library_dirs=envs.LIBRARY_DIRS,
            libraries=envs.LIBRARIES,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
