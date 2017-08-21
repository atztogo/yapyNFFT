#!/usr/bin/env python

"""This is an example of yapyNFFT.

yapyNFFT is a simple python wrapper of NFFT (http://www.nfft.org).

I think the license has to be GPL to distribute because NFFT employs
GPL. But if it is not the case, I want to change the license to BSD-3.

nfft_plan is stored in yapyNFFT modlue as a static variable. So
currently only one nfft_plan can be stored for each import of
yapyNFFT. Finalization (finalize) is necessary to release allocated
memory.

Atsushi Togo

"""

import numpy as np
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
    x = np.arange(Nx, dtype='double') / float(Nx) - 0.5
    y = np.arange(Ny, dtype='double') / float(Ny) - 0.5
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
    x = np.arange(Nx, dtype='double') / float(Nx) - 0.5
    y = np.arange(Ny, dtype='double') / float(Ny) - 0.5
    nodes = np.zeros([Nx * Ny, ndim], dtype='double')
    for i, n in enumerate(nodes):
        i_x = i // Ny
        i_y = i % Ny
        n[0] = x[i_x]
        n[1] = y[i_y]
    return nodes

def init_guru(dims, nnode, sigma, cutoff=6):
    """

    Args:
        dims: Dimensions of data on uniform grids for f_hat
        nnode: Number of sampling points for x
        sigma: Over sampling factor. According to the comment written in nfft3.h
               the default value is 2 <= sigma < 4, where over sampling
               dimensions for FFTW will be a power of 2.
        cutoff: According to the comment in nfft3.h, the default value is:
                Kaiser Bessel 6
                Sinc power    9
                B spline     11
                Gaussian     12
    """

    odims = [int(N * sigma) for N in dims]
    print(("Over sampling: [ " + "%d " * len(dims) + "] -> [" +
           " %d" * len(dims) + " ]") % (tuple(dims) + tuple(odims)))
    yapyNFFT.init_guru(dims, odims, nnode, cutoff=cutoff)

def init(dims, nnode):
    yapyNFFT.init(dims, nnode)

def main():
    dims = [20, 30]
    ndim = len(dims)

    # Data on a uniform grid points.
    data = generate_data_2d(dims, decimals=9)

    # Points on which data are transformed. These can be non-uniform.
    nodes = generate_nodes_2d(dims)
    nnode = nodes.shape[0]

    # nfft_init is done with simple version or guru version.
    # Simple version: probably default values of sigma and cutoff are used.
    init(dims, nnode)
    # Guru version:
    # init_guru(dims, nnode, sigma=2.5, cutoff=6)

    # Store nodes and data in nfft_plan.x and nfft_plan.f_hat, respectively.
    yapyNFFT.set(nodes, data)

    # Precompute
    yapyNFFT.precompute_one_psi()

    #
    # Forward trnsformation from f_hat to f
    #
    # Transformed data are stored in nfft_plan.f.
    # ndft (nfft_trafo_direct) and nfft (nfft_trafo) are expected to show
    # enough similar results if sigma and cutoff are properly chosen.
    # Transformed result 
    print("# ndft")
    yapyNFFT.trafo_direct()
    f = yapyNFFT.get_f().reshape(dims)
    show(f)

    print("# nfft")
    yapyNFFT.trafo()
    f = yapyNFFT.get_f().reshape(dims)
    show(f)

    #
    # Backward transformation from f to f_hat
    #
    # Transformed data are stored in nfft_plan.f_hat.
    # adjoint ndft (nfft_adjoint_direct) and adjoint nfft (nfft_adjoint)
    # are expected to show enough similar results if sigma and cutoff are
    # properly chosen.
    print("# adjoint ndft")
    yapyNFFT.adjoint_direct()
    f_hat = yapyNFFT.get_f_hat()
    show(f_hat)

    print("# adjoint nfft")
    yapyNFFT.adjoint()
    f_hat = yapyNFFT.get_f_hat()
    show(f_hat)

    # Deallocate nfft_plan.
    yapyNFFT.finalize()



if __name__ == '__main__':
    main()
