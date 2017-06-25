import numpy as np
import yapyNFFT._yapyNFFT

def init_guru(ndim, dims_x, dims_f_hat, cutoff=6):
    _yapyNFFT.nfft_init_guru(ndim,
                             np.array(dims_x, dtype='intc'),
                             np.array(dims_f_hat, dtype='intc'),
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

def set(x, f_hat):
    """Set data

    Args:
        x: numpy 1d array, dtype='double', order='C'
        f_hat: numpy 1d array, dtype='double', order='C'

    """

    _yapyNFFT.nfft_set(x, f_hat)

def get(dims_x):
    f = np.zeros(tuple(dims_x) + (2,), dtype='double', order='C')
    _yapyNFFT.nfft_get(f)
    dtype = 'c%d' % (f.itemsize * 2)
    return f.view(dtype=dtype)
