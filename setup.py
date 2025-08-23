import os
import setuptools
import shutil
import subprocess
import glob
from pathlib import Path
from setuptools import find_packages
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import CppExtension, CUDA_HOME

current_dir = os.path.dirname(os.path.realpath(__file__))

def get_all_files(directory):
    all_files = glob.glob(os.path.join(directory, "**", "*"), recursive=True)
    files = [f[len(current_dir)+1:] for f in all_files if os.path.isfile(f)]
    return files

sources = get_all_files(os.path.join(current_dir, 'csrc'))
build_include_dirs = [
    f'{CUDA_HOME}/include',
    'include',
    'thirdparty/cutlass/include',
    'thirdparty/composable_kernel/include',
    'thirdparty/json/include',
]

build_libraries = ['cuda', 'cudart', 'nvrtc']
build_library_dirs = [
    f'{CUDA_HOME}/lib64',
    f'{CUDA_HOME}/lib64/stubs'
]

cxx_flags = ['-std=c++20',
             '-O3',
             '-fPIC',
             '-fvisibility=hidden']

data_include_dirs = [
    'include/plugins/acu',
    'include/plugins/aroc',
    'thirdparty/cutlass/include/cute',
    'thirdparty/cutlass/include/cutlass',
    'thirdparty/composable_kernel/include/ck',
    'thirdparty/composable_kernel/include/ck_tile',
]

class CustomBuildPy(build_py):
    def run(self):
        self.prepare_includes()
        build_py.run(self)

    def prepare_includes(self):
        build_include_dir = os.path.join(self.build_lib, 'pccl/include')
        os.makedirs(build_include_dir, exist_ok=True)

        for d in data_include_dirs:
            dirname = d.split('/')[-1]
            src_dir = os.path.join(current_dir, d)
            dst_dir = os.path.join(build_include_dir, dirname)

            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)

            shutil.copytree(src_dir, dst_dir)

print(sources)

if __name__ == '__main__':
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except:
        revision = ''

    setuptools.setup(
        name='pccl',
        version='0.1.0' + revision,
        packages=find_packages('.'),
        package_data={
            'pccl': [
                'include/pccl/**/*',
                'include/cute/**/*',
                'include/cutlass/**/*',
                'include/ck/**/*',
                'include/ck_tile/**/*',
            ]
        },
        ext_modules=[
            CppExtension(name='pccl._C',
                        sources=sources,
                        include_dirs=build_include_dirs,
                        libraries=build_libraries,
                        library_dirs=build_library_dirs,
                        extra_compile_args=cxx_flags)
        ],
        zip_safe=False,
        cmdclass={
            'build_py': CustomBuildPy,
        },
    )

