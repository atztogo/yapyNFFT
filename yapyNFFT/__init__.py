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

def set(x, f_hat):
    """Set data

    Args: (n is number of dimensions)
        x: numpy (nnode, n)-array, dtype='double', order='C'
        f_hat: numpy (N0, N1, N2, ..., Nn) 
            dtype='complex128' or 'float64', order='C'

    """

    itemsize = np.dtype('double').itemsize
    if f_hat.dtype.name[0] == 'c' and f_hat.itemsize == itemsize * 2:
        dtype = 'f%d' % (f_hat.itemsize // 2)
        f_hat_double = f_hat.view(dtype=dtype)
        _yapyNFFT.nfft_set(x, f_hat_double, 'c')
    elif f_hat.dtype.name[0] == 'f' and f_hat.itemsize == itemsize:
        _yapyNFFT.nfft_set(x, f_hat, 'f')
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
    return f.view(dtype=dtype)

def get_f_hat():
    N = _yapyNFFT.nfft_get_N()
    f_hat = np.zeros(tuple(N) + (2,), dtype='double', order='C')
    _yapyNFFT.nfft_get_f_hat(f_hat)
    dtype = 'c%d' % (f_hat.itemsize * 2)
    return f_hat.view(dtype=dtype)

def get_N():
    return _yapyNFFT.nfft_get_N()

def get_M():
    return _yapyNFFT.nfft_get_M()
