#!/usr/bin/env python

import numpy as np
import yapyNFFT

ndim = 2
Nx = 20
Ny = 30
x = np.arange(Nx) / float(Nx) - 0.5
y = np.arange(Ny) / float(Ny) - 0.5

exp_x = np.exp(2j * np.pi * x)
exp_y = np.exp(2j * np.pi * y)
exp_2d = np.outer(exp_x, exp_y)
dtype = 'f%d' % (exp_2d.itemsize / 2)
exp_2d_double = exp_2d.ravel().view(dtype=dtype)

nodes = np.zeros([Nx * Ny, ndim], dtype='double')

dx = 1.0 / Nx
dy = 1.0 / Ny
count = 0
x = -0.5
for i in range(Nx):
    y = -0.5
    for j in range(Ny):
        nodes[count, 0] = x
        nodes[count, 1] = y
        count += 1
        y += dy
    x += dx

yapyNFFT.init_guru(ndim, [Nx, Ny], [Nx, Ny], cutoff=3)
yapyNFFT.set(nodes, exp_2d_double)
yapyNFFT.precompute_one_psi()
print("# ndft")
yapyNFFT.trafo_direct()
f = yapyNFFT.get([Nx, Ny])
print(f)
print("# nfft")
yapyNFFT.trafo()
f = yapyNFFT.get([Nx, Ny])
print(f)
yapyNFFT.finalize()


