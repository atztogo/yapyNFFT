#!/usr/bin/env python

"""This is an example of yapyNFFT.

yapyNFFT is a simple python wrapper of NFFT (http://www.nfft.org).

I think the license has to be GPL to distribute because NFFT employs
GPL. But if it is not the case, I want to change the license to BSD-3.

In this example, NFFT and FFTW are simply compared.
Note that pyFFTW is necessary to run this example.
https://github.com/pyFFTW/pyFFTW
This may be installed via conda-forge by

% conda install -c conda-forge pyfftw

with this, multithreaded FFTW3 is also installed.

Atsushi Togo

"""

import numpy as np
import pyfftw
import yapyNFFT

def show(array):
    for i, v in enumerate(array):
        for j, x in enumerate(v):
            print("%3d %3d  %13.5e  %13.5e" % (i, j, x.real, x.imag))
        print("")
    print("")

def generate_data_2d(dims, decimals=None):
    """2D data of exp(2pi i * r), where r = (x, y)^T"""

    Nx, Ny = dims
    x = np.arange(Nx, dtype='double') / float(Nx)
    y = np.arange(Ny, dtype='double') / float(Ny)
    exp_x = np.exp(2j * np.pi * x)
    exp_y = np.exp(2j * np.pi * y)
    data = np.empty((Nx, Ny), dtype='complex128', order='C')
    data[:] = np.outer(exp_x, exp_y)
    if decimals is not None:
        data = data.round(decimals=decimals)
    return data

def generate_nodes_2d(dims):
    Nx, Ny = dims
    ndim = 2
    x = np.arange(Nx, dtype='double') / float(Nx)
    x = np.where(x < 0.5, x, x - 1) # [0, 1) -> [0, 0.5) + [-0.5, 0)
    y = np.arange(Ny, dtype='double') / float(Ny)
    y = np.where(y < 0.5, y, y - 1) # [0, 1) -> [0, 0.5) + [-0.5, 0)
    nodes = np.zeros([Nx * Ny, ndim], dtype='double')
    for i, n in enumerate(nodes):
        i_x = i // Ny
        i_y = i % Ny
        n[0] = x[i_x]
        n[1] = y[i_y]
    return nodes

def init(dims, nnode):
    yapyNFFT.init(dims, nnode)

def run_nfft(dims, data, nodes):
    init(dims, nodes.shape[0])
    yapyNFFT.set(nodes, data)
    yapyNFFT.precompute_one_psi()
    print("# nfft")
    yapyNFFT.trafo()
    f = yapyNFFT.get_f().reshape(dims)
    show(f)
    yapyNFFT.finalize()

def run_fftw(dims, data):
    print("# fftw")
    a = pyfftw.empty_aligned(dims, dtype='complex128')
    a[:] = data
    b = pyfftw.interfaces.numpy_fft.fftn(a)
    show(b)

def run_numpy_fft(data):
    print("# numpy.fft")
    show(np.fft.fftn(data))

def main():
    dims = [20, 30]
    data = generate_data_2d(dims)
    nodes = generate_nodes_2d(dims)
    run_nfft(dims, data, nodes)
    run_fftw(dims, data)
    run_numpy_fft(data)

if __name__ == '__main__':
    main()
