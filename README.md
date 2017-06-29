# yapyNFFT

yapyNFFT is a simple python wrapper of NFFT (http://www.nfft.org).

I think the license has to be GPL to distribute because NFFT employs
GPL. But if it is not the case, I want to change the license to BSD-3.

nfft_plan is stored in yapyNFFT modlue as a static variable. So
currently only one nfft_plan can be stored for each import of
yapyNFFT. Finalization (finalize) is necessary to release allocated
memory.

## How to install

First NFFT library has to be prepared. Because NFFT relies on FFTW, FFTW3 has to be also properly installed. Then modify `setup.py` so that it can find NFFT and FFTW3 library, e.g.,
```
extra_link_args = ['-L/home/togo/code/nfft/lib', '-lnfft3', '-lfftw3', '-lm']
```
if `/home/togo/code/nfft/lib` is the directory where NFFT library is installed and FFTW3 is installed the location where automatically found (in the system library directory or in `LD_LIBRARY_PATH`). After setting up this, run `setup.py`, e.g.,
```
% python setup.py install --user
```

## Example

Examples are found in the `example` directory.
