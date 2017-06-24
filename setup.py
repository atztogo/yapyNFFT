import os
import sys
from setuptools import setup, Extension

include_dirs = []
include_dirs += get_numpy_include_dirs()

extra_compile_args = []
extra_link_args = ['-lnfft3', '-lfftw3', '-lm']
define_macros = []

extension = Extension('yapyNFFT._yapyNFFT',
                      include_dirs=include_dirs,
                      sources=['c/_yapyNFFT.c'],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args,
                      define_macros=define_macros)

setup(name='yapyNFFT',
      version=version,
      setup_requires=['numpy', 'setuptools'],
      description='This is the yapyNFFT module.',
      author='Atsushi Togo',
      author_email='atz.togo@gmail.com',
      packages=['yapyNFFT'],
      install_requires=['numpy'],
      provides=['yapyNFFT'],
      platforms=['all'],
      ext_modules=[extension])

