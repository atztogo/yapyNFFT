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

    Args:
        x: numpy 1d array, dtype='double', order='C'
        f_hat: numpy 1d array, dtype='double', order='C'

    """

    _yapyNFFT.nfft_set(x, f_hat)

def get_f(dims_x):
    f = np.zeros(tuple(dims_x) + (2,), dtype='double', order='C')
    _yapyNFFT.nfft_get_f(f)
    dtype = 'c%d' % (f.itemsize * 2)
    return f.view(dtype=dtype)

def get_f_hat(dims_f_hat):
    f_hat = np.zeros(tuple(dims_f_hat) + (2,), dtype='double', order='C')
    _yapyNFFT.nfft_get_f_hat(f_hat)
    dtype = 'c%d' % (f_hat.itemsize * 2)
    return f_hat.view(dtype=dtype)
