import numpy as np
import yapyNFFT._yapyNFFT

def init(dims_N, nnode):
    _yapyNFFT.nfft_init(np.array(dims_N, dtype='intc'),
                        int(nnode))

def init_guru(dims_N, dims_n, nnode, cutoff=6):
    _yapyNFFT.nfft_init_guru(np.array(dims_N, dtype='intc'),
                             np.array(dims_n, dtype='intc'),
                             int(nnode),
                             int(cutoff))

def finalize():
    id = 0
    _yapyNFFT.nfft_finalize(id)

def precompute_one_psi():
    _yapyNFFT.nfft_precompute_one_psi()

def trafo():
    _yapyNFFT.nfft_trafo()

def trafo_direct():
    _yapyNFFT.nfft_trafo_direct()

def adjoint():
    _yapyNFFT.nfft_adjoint()

def adjoint_direct():
    _yapyNFFT.nfft_adjoint_direct()

def set(x, f_hat, x_order='C', shift_grid=False):
    """Set data

    Args: (n is number of dimensions)
        x: numpy (nnode, n)-array, dtype='double', order='C'
        f_hat: numpy (N0, N1, N2, ..., Nn) where Nn are even numbers.
            dtype='complex128' or 'float64', order='C'
        x_order: Order in each row of x, e.g. for 3D, xyz ('C') or zyx ('F') in
            each row.
        shift_grid (bool): f_hat data are shifted in memory +Nn/2 along all
            dimensions.

    """

    if (np.array(f_hat.shape) % 2 != 0).any():
        print("All dimentions have to be even numbers.")
        raise ValueError

    itemsize = np.dtype('double').itemsize
    if f_hat.dtype.name[0] == 'c' and f_hat.itemsize == itemsize * 2:
        dtype = 'f%d' % (f_hat.itemsize // 2)
        f_hat_double = f_hat.view(dtype=dtype)
        _yapyNFFT.nfft_set(x, f_hat_double, 'c', x_order, shift_grid * 1)
    elif f_hat.dtype.name[0] == 'f' and f_hat.itemsize == itemsize:
        _yapyNFFT.nfft_set(x, f_hat, 'f', x_order, shift_grid * 1)
    else:
        print("Second argument of yapyNFFT.set has to have dtype='complex%d'"
              % (itemsize * 16))
        print("or 'double' (equivalently 'float%d')." % (itemsize * 8))
        raise TypeError

def get_f():
    M = _yapyNFFT.nfft_get_M()
    f = np.zeros((M, 2), dtype='double', order='C')
    _yapyNFFT.nfft_get_f(f)
    dtype = 'c%d' % (f.itemsize * 2)
    return f.view(dtype=dtype).reshape((M,))

def get_f_hat():
    N = _yapyNFFT.nfft_get_N()
    f_hat = np.zeros(tuple(N) + (2,), dtype='double', order='C')
    _yapyNFFT.nfft_get_f_hat(f_hat)
    dtype = 'c%d' % (f_hat.itemsize * 2)
    return f_hat.view(dtype=dtype).reshape((N,))

def get_N():
    return _yapyNFFT.nfft_get_N()

def get_M():
    return _yapyNFFT.nfft_get_M()
