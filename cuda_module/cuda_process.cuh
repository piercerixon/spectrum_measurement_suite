#ifndef CUDA_PROCESS_H
#define CUDA_PROCESS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <complex>


typedef short2 cuComplexShort;
//void dothething(std::complex<short>*, const int);
static __global__ void avg_out(float*, cuComplex*, const int, const int, const float);
static __global__ void cufft_prep(cuComplex*, cuComplexShort*, float*, const int);
static __global__ void filter_test(float*, const int);

static __global__ void avg_out_overlap(float*, cuComplex*, const int, const int, const float, int);
static __global__ void cufft_prep_overlap(cuComplex*, cuComplexShort*, float*, const int, int);

static __device__ __host__ cuComplex ComplexRMul(cuComplex, float);

//const int NUM_SAMPS = 131072;
//const int NUM_SAMPS = 16384;
//const int NUM_SAMPS = 2048;
const int NUM_SAMPS = 131072;

const int CU_THD = 1024;

#ifndef PI
#define PI
	static const float pi = float(std::acos(-1.0));
#endif


#endif