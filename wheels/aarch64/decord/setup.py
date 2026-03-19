import setuptools

from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel
import platform
import os

def get_platform_tag():
    machine = platform.machine()
    system = platform.system().lower()
    
    if system == 'linux':
        if machine in ['aarch64', 'arm64']:
            return 'linux_aarch64'
        elif machine in ['x86_64', 'amd64']:
            return 'linux_x86_64'
    elif system == 'darwin':  # macOS
        if machine in ['aarch64', 'arm64']:
            return 'macosx_11_0_arm64'
        else:
            return 'macosx_10_9_x86_64'
    elif system == 'windows':
        return 'win_amd64'
    
    return 'any'

class PlatformAwareBdistWheel(bdist_wheel):
    def get_tag(self):
        python_tag = 'py3'
        abi_tag = 'none'
        platform_tag = get_platform_tag()
        return python_tag, abi_tag, platform_tag

class BinaryDistribution(setuptools.Distribution):
    def has_ext_modules(self):
        return True

setup(
    name="decord",
    version="0.6.0",
    packages=find_packages(),
    package_data={"decord": ["lib/libdecord.so", "_ffi/*", "_ffi/_cython/*", "_ffi/_ctypes/*"]},
    include_package_data=True,
    install_requires=["numpy"],
    python_requires=">=3.7",
    distclass=BinaryDistribution,
    cmdclass={
        'bdist_wheel': PlatformAwareBdistWheel,
    },
)
