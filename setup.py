import os
import sys
import numpy
from setuptools import setup, Extension

include_dirs = [numpy.get_include(), '/home/togo/code/nfft/include']

extra_compile_args = []
#extra_link_args = ['-L/home/togo/code/nfft/lib', '-lnfft3', '-lfftw3', '-lm']
extra_link_args = ['-L/home/togo/code/nfft/lib', '-lnfft3_threads', '-lfftw3', '-lm']
define_macros = []

extension = Extension('yapyNFFT._yapyNFFT',
                      include_dirs=include_dirs,
                      sources=['c/_yapyNFFT.c'],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args,
                      define_macros=define_macros)

setup(name='yapyNFFT',
      version='0.1',
      setup_requires=['numpy', 'setuptools'],
      description='This is the yapyNFFT module.',
      author='Atsushi Togo',
      author_email='atz.togo@gmail.com',
      packages=['yapyNFFT'],
      install_requires=['numpy'],
      provides=['yapyNFFT'],
      platforms=['all'],
      ext_modules=[extension])

