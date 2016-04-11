#ifndef CUDA_MODULE_H
#define CUDA_MODULE_H
#include <complex>

//Pointer to samples, number of fft windows, averageing, pointer to output processed and averaged samples
//This function will malloc the necessary amount of memory for the output array
//float* dothething(std::complex<short>*, const int, const int);
void dothething(std::complex<short>*, float*, const int, const int);
float* dothething_overlap(std::complex<short>*, const int, float*, const int, const int);

#endif