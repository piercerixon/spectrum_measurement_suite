#ifndef FFT_MODULE_H
#define FFT_MODULE_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <complex>


typedef short2 cuComplexShort;


static __global__ void avg_out(float* out, cuComplex* d_fft, const int num_wins, const int averaging, const float offset, const int resolution);

static __global__ void avg_out_filter(float* out, cuComplex* d_fft, const int num_wins, const int averaging, const float offset, const int resolution);

static __global__ void cufft_prep(cuComplex* d_fft, cuComplexShort* d_s, float* d_w, const int num_wins, const int resolution);

void cuda_memcheck();

const int CU_THD = 1024;

#ifndef PI
#define PI
static const float pi = float(std::acos(-1.0));
#endif


#endif