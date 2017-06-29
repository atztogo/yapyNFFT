# yapyNFFT

yapyNFFT is a simple python wrapper of NFFT (http://www.nfft.org).

I think the license has to be GPL to distribute because NFFT employs
GPL. But if it is not the case, I want to change the license to BSD-3.

nfft_plan is stored in yapyNFFT modlue as a static variable. So
currently only one nfft_plan can be stored for each import of
yapyNFFT. Finalization (finalize) is necessary to release allocated
memory.
