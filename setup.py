from setuptools import setup, find_packages

# NOTE: The original CUDA extension (ext_modules) is commented out because
# kat-rational==0.4 never completed the build setup. The Triton-based
# implementation in kat_rational/rational_triton.py is used instead —
# it requires no compilation and ships with standard PyTorch.

setup(
    name='kat_rational',
    version='0.4',
    author='adamdad',
    author_email='yxy_adadm@qq.com',
    description='Group-wise rational activation for KAT, with Triton kernels',
    packages=['kat_rational', 'rational_kat_cu'],
    # triton is not a standalone pip package on Linux — it ships bundled with PyTorch
    install_requires=[],
)