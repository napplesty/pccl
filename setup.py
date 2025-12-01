import os
import setuptools
import shutil
import subprocess
import glob
import subprocess
from setuptools import find_packages
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME

current_dir = os.path.dirname(os.path.realpath(__file__))

def build_hwloc(build_libs, make_jobs=None, verbose=False):
    build_libs = os.path.abspath(os.path.expanduser(build_libs))
    if not os.path.isdir(build_libs):
        os.makedirs(build_libs)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    hwloc_path = os.path.join(current_dir, 'thirdparty', 'hwloc')
    if not os.path.isdir(hwloc_path):
        raise FileNotFoundError(hwloc_path)

    make_jobs = make_jobs or (os.cpu_count() or 1)

    def _run(cmd, cwd=None):
        cwd = cwd or hwloc_path
        if verbose:
            print(' '.join(cmd))
        subprocess.run(cmd, cwd=cwd, capture_output=not verbose, text=True, check=False)

    _run(['./autogen.sh'])
    _run(['./configure'])
    _run(['make', '-j{}'.format(make_jobs)])

    src_so = os.path.join(hwloc_path, 'hwloc', '.libs', 'libhwloc.so.0.0.0')
    if not os.path.isfile(src_so):
        raise FileNotFoundError(src_so)

    dst_so = os.path.join(build_libs, 'libhwloc.so')
    shutil.copy2(src_so, dst_so)

def get_all_files(directory):
    all_files = glob.glob(os.path.join(directory, "**", "*"), recursive=True)
    files = [f[len(current_dir)+1:] for f in all_files if os.path.isfile(f)]
    return files

sources = get_all_files(os.path.join(current_dir, 'csrc'))

build_include_dirs = [
    f'{CUDA_HOME}/include',
    f'{current_dir}/include',
    f'{current_dir}/thirdparty/cutlass/include',
    f'{current_dir}/thirdparty/composable_kernel/include',
    f'{current_dir}/thirdparty/json/include',
    f'{current_dir}/thirdparty/hwloc/include',
    f'{current_dir}/thirdparty/spdlog/include',
    f'{current_dir}/thirdparty/asio/asio/include',
]

build_libraries = ['cuda', 'cudart', 'nvrtc']
build_library_dirs = [
    f'{CUDA_HOME}/lib64',
    f'{CUDA_HOME}/lib64/stubs',
    f'{CUDA_HOME}/targets/x86_64-linux/lib/stubs/',
]

cxx_flags = ['-std=c++20',
             '-fPIC',
             '-fvisibility=hidden',
             '-DASIO_STANDALONE',
             '-DASIO_HEADER_ONLY',
            ]

cuda_flags = ['-std=c++20',
              '-O3',
             ]

debug = True
if debug:
    cxx_flags.extend(['-DPCCL_DEBUG'])
    cuda_flags.extend(['-DPCCL_DEBUG'])

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
                'lib/*'
            ]
        },
        ext_modules=[
            CppExtension(name='pccl.engine_c',
                         sources=sources,
                         include_dirs=build_include_dirs,
                         libraries=build_libraries,
                         library_dirs=build_library_dirs,
                         extra_compile_args={'cxx': cxx_flags, 
                                             'cuda': cuda_flags},
            )
        ],
        zip_safe=False,
        cmdclass={
            'build_ext': BuildExtension,
            'build_py': build_py,
        },
    )
