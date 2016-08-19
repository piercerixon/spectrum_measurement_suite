#include "cuda_process.cuh"
#include "cuda_module.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_functions.h"
#include "helper_cuda.h"

#include <stdio.h>
#include <iostream>
#include <cufft.h>
#include <complex>

//int main(int argc, char **argv){}

//The cufft must be invoked by the host, not as part of a kernel. 
//num_wins is the total number of windows, without averaging. Therefore total windows abould be number of output frames * averaging to ensure no wasted samples.
//float* dothething(std::complex<short>* h_samp_arry, const int averaging, const int num_wins) {
void dothething(std::complex<short>* h_samp_arry, float* h_out, const int averaging, const int num_wins) {
	/*
	if (num_wins == 0) {
		std::cout << "AMG NO WINS!\n";
		num_wins = sizeof(h_samp_arry) / (2 * NUM_SAMPS);
		std::cout << "Number of windows: " << num_wins << std::endl;
	} */
	//const int num_wins = 1;
	//cuComplex* samp[NUM_SAMPS];
	//std::complex<short>* d_samp;

	cudaError_t cudaStatus;

	// for outputting of averaged and processed samples /
	
	//float* h_out = (float*)malloc(sizeof(float) * NUM_SAMPS * num_wins/averaging);
	//h_out = (float*)calloc(NUM_SAMPS * num_wins / averaging, sizeof(float));
	//if (h_out == NULL) {
	//	fprintf(stderr, "h_out Malloc failed!");
	//	goto Error;
	//}
	
	float* d_out;
	
	cuComplexShort* h_samp_ptr = (cuComplexShort*)&h_samp_arry[0];

	//std::cout << h_samp_arry[0].real() << "," << h_samp_arry[0].imag() << " cuCmplx" << h_samp_ptr[0].x << "," << h_samp_ptr[0].y << std::endl;
	
	float h_win[NUM_SAMPS];
	float* d_win;

	cuComplexShort* d_samp;
	cuComplex* d_fftbuff;

	float win_power = 0;
	int rx_gain = 30;

	//Create coefficient array and x axis index for plotting
	for (int i = 0; i < NUM_SAMPS; i++) {
		h_win[i] = 0.35875 - 0.48829*cos(2 * pi*i / (NUM_SAMPS - 1)) + 0.14128*cos(4 * pi*i / (NUM_SAMPS - 1)) - 0.01168*cos(6 * pi*i / (NUM_SAMPS - 1)); //blackmann harris window		
		win_power += (h_win[i] * h_win[i]); //this computes the total window power and normalises it to account for DC gain due to the window.
	}
	win_power /= NUM_SAMPS; //normalise the total window power across each sample.

	const float offset = -10 - rx_gain + 10 * std::log10(win_power); //-10 is the MAX power detected by the ADC and take into account the gain of the frontend.

	//printf("GPU Offset: %f", offset);

	//allocate the memory for the GPU
	cudaStatus = cudaMalloc((float**)&d_out, sizeof(float)*NUM_SAMPS * num_wins / averaging);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaMemset(d_out, 0, sizeof(float)*NUM_SAMPS * num_wins / averaging); //initialise to zero

	cudaStatus = cudaMalloc((cuComplexShort**)&d_samp, sizeof(cuComplexShort)*NUM_SAMPS*num_wins);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((float**)&d_win, sizeof(float)*NUM_SAMPS);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((cuComplex**)&d_fftbuff, sizeof(cuComplex)*NUM_SAMPS*num_wins);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	//Transfer data to GPU
	cudaStatus = cudaMemcpy(d_win, h_win, sizeof(float)*NUM_SAMPS, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to Device failed!");
		goto Error;
	}
	
	cudaStatus = cudaMemcpy(d_samp, h_samp_ptr, sizeof(cuComplexShort)*NUM_SAMPS*num_wins, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to Device failed!");
		goto Error;
	}
	
	//Create cufft plan, turns out cufft handles its own memory transfers, so we must use callbacks in order to avoid numerous reads and writes in the device
	//Will however use multiple kernels initially, then see what the performance improvement is with callbacks at a later stage. n.n
	cufftHandle plan;
	cufftPlan1d(&plan, NUM_SAMPS, CUFFT_C2C, num_wins);

	//printf("h_samp[%d]=%f,%f ", 0, s_ptr[0].x, s_ptr[0].y);
	//printf("d_samp[%d]=%f,%f\n", 0,d_samp[0].x,d_samp[0].y);
	// Kernel calls lah <<blocks,threads>>
	
	cufft_prep <<< (NUM_SAMPS*num_wins) / CU_THD, CU_THD >>>(d_fftbuff, d_samp, d_win, num_wins); //This will create (WIN_SAMPS*num_wins)/CU_THD blocks, with 1024 threads per block
	
	//inplace fft
	if (cufftExecC2C(plan, d_fftbuff, d_fftbuff, CUFFT_FORWARD)){
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		goto Error;
	}
	
	//Do something with the fft'd samples, like average them, then output them to the host, where the host can perform detection.
	avg_out <<< NUM_SAMPS / CU_THD, CU_THD >> >(d_out, d_fftbuff, num_wins, averaging, offset);
	
	filter_test <<< NUM_SAMPS / CU_THD, CU_THD >> >(d_out, num_wins/averaging);

	cudaStatus = cudaMemcpy(h_out, d_out, sizeof(float)*NUM_SAMPS * num_wins/averaging, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to Host failed!");
		goto Error;
	}
	
	/*
	std::cout << "GPU: ";
	
	for (int i = 0; i < NUM_SAMPS; i++) {
		std::cout << h_win[i] << ",";
	}
	
	std::cout << "Please note these are not flipped around samples/2 correctly" << std::endl;
	*/

Error:

	cufftDestroy(plan);
	checkCudaErrors(cudaFree(d_out));
	checkCudaErrors(cudaFree(d_samp));
	checkCudaErrors(cudaFree(d_win));
	checkCudaErrors(cudaFree(d_fftbuff));

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}

	//return h_out;
}

//Kernel Call
//https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-use-cufft-callbacks-custom-data-processing/ for inspiration
static __global__ void cufft_prep(cuComplex* d_fft, cuComplexShort* d_s, float* d_w, const int num_wins) {

	int idx = threadIdx.x;
	
	//blockDim = number of threads in a block
	//This will take an array of complex shorts (14b samples) an array of cuComplex and a window array, will convert the com_short to cuComplex (com_float), correctly scale the samples and apply the appropriate window prepping it for fft
	for (int i = blockIdx.x * blockDim.x + idx; i < NUM_SAMPS*num_wins; i += blockDim.x * gridDim.x){
		d_fft[i].x = (d_s[i].x*1.0f / 32767.0f) * d_w[i%NUM_SAMPS];
		d_fft[i].y = (d_s[i].y*1.0f / 32767.0f) * d_w[i%NUM_SAMPS];
	}
	//if(idx == 0) printf("d_s[%d]: %f,%f fftbuff %f,%f\n", idx, d_s[idx].x, d_s[idx].y, d_s[idx].x, d_s[idx].x);
}

static __global__ void filter_test(float*out, const int num_wins){

	int idx = threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	float* out_ptr = &out[0];
	const int fiveby_filter_level = 13; //normally 16 for 5x5, 13 for aggressive.
	const int filter_level = 5; // 3x3 kernel 

	bool FIVEBY = true; //for use later

	//increment loop by 1, and decrease total run by 1 to accomodate for edges of the kernel
	if (!FIVEBY){
		for (int i = (blockIdx.x * blockDim.x + idx) + stride; i < NUM_SAMPS*(num_wins - 1); i += stride){

			if (out_ptr[i] == 0 && (blockIdx.x + idx != 0 || blockIdx.x + idx != NUM_SAMPS - 1)){
				if ((out_ptr[i - 1 - stride] + out_ptr[i - stride] + out_ptr[i + 1 - stride] +
					out_ptr[i - 1] + out_ptr[i + 1] +
					out_ptr[i - 1 + stride] + out_ptr[i + stride] + out_ptr[i + 1 + stride]) > filter_level){

					out_ptr[i] = 1;
				}
			}
		}
	}
	if (FIVEBY){
		//special case code for handling the beginning and end of the image, note that edges are ignored as they are significantly less impactful on window generation

		for (int i = blockIdx.x * blockDim.x + idx, j = 0; i < stride*(num_wins); i += stride, j++){

			if (out_ptr[i] == 0 && (blockIdx.x * blockDim.x + idx != 0 && blockIdx.x * blockDim.x + idx != NUM_SAMPS - 1 &&
				blockIdx.x * blockDim.x + idx != 1 && blockIdx.x * blockDim.x + idx != NUM_SAMPS - 2)){

				if (j == 0){
					if (( //unrolled here for efficiencies
						out_ptr[i - 2] + out_ptr[i - 1] + out_ptr[i + 1] + out_ptr[i + 2] +
						out_ptr[i - 2 + stride] + out_ptr[i - 1 + stride] + out_ptr[i + stride] + out_ptr[i + 1 + stride] + out_ptr[i + 2 + stride] +
						out_ptr[i - 2 + 2 * stride] + out_ptr[i - 1 + 2 * stride] + out_ptr[i + 2 * stride] + out_ptr[i + 1 + 2 * stride] + out_ptr[i + 2 + 2 * stride]) > fiveby_filter_level - 6)
					{
						out_ptr[i] = 1;
					}
				}

				else if (j == 1){
					if (( //unrolled here for efficiencies
						out_ptr[i - 2 - stride] + out_ptr[i - 1 - stride] + out_ptr[i - stride] + out_ptr[i + 1 - stride] + out_ptr[i + 2 - stride] +
						out_ptr[i - 2] + out_ptr[i - 1] + out_ptr[i + 1] + out_ptr[i + 2] +
						out_ptr[i - 2 + stride] + out_ptr[i - 1 + stride] + out_ptr[i + stride] + out_ptr[i + 1 + stride] + out_ptr[i + 2 + stride] +
						out_ptr[i - 2 + 2 * stride] + out_ptr[i - 1 + 2 * stride] + out_ptr[i + 2 * stride] + out_ptr[i + 1 + 2 * stride] + out_ptr[i + 2 + 2 * stride]) > fiveby_filter_level - 3)
					{
						out_ptr[i] = 1;
					}
				}

				else if (j >= 2 && j < num_wins - 2){
					if (( //unrolled here for efficiencies
						out_ptr[i - 2 - 2 * stride] + out_ptr[i - 1 - 2 * stride] + out_ptr[i - 2 * stride] + out_ptr[i + 1 - 2 * stride] + out_ptr[i + 2 - 2 * stride] +
						out_ptr[i - 2 - stride] + out_ptr[i - 1 - stride] + out_ptr[i - stride] + out_ptr[i + 1 - stride] + out_ptr[i + 2 - stride] +
						out_ptr[i - 2] + out_ptr[i - 1] + out_ptr[i + 1] + out_ptr[i + 2] +
						out_ptr[i - 2 + stride] + out_ptr[i - 1 + stride] + out_ptr[i + stride] + out_ptr[i + 1 + stride] + out_ptr[i + 2 + stride] +
						out_ptr[i - 2 + 2 * stride] + out_ptr[i - 1 + 2 * stride] + out_ptr[i + 2 * stride] + out_ptr[i + 1 + 2 * stride] + out_ptr[i + 2 + 2 * stride]) > fiveby_filter_level)
					{
						out_ptr[i] = 1;
					}
				}

				else if (j == num_wins - 2){
					if (( //unrolled here for efficiencies
						out_ptr[i - 2 - 2 * stride] + out_ptr[i - 1 - 2 * stride] + out_ptr[i - 2 * stride] + out_ptr[i + 1 - 2 * stride] + out_ptr[i + 2 - 2 * stride] +
						out_ptr[i - 2 - stride] + out_ptr[i - 1 - stride] + out_ptr[i - stride] + out_ptr[i + 1 - stride] + out_ptr[i + 2 - stride] +
						out_ptr[i - 2] + out_ptr[i - 1] + out_ptr[i + 1] + out_ptr[i + 2] +
						out_ptr[i - 2 + stride] + out_ptr[i - 1 + stride] + out_ptr[i + stride] + out_ptr[i + 1 + stride] + out_ptr[i + 2 + stride]) > fiveby_filter_level - 3)
					{
						out_ptr[i] = 1;
					}
				}

				else if (j == num_wins - 1){
					if (( //unrolled here for efficiencies
						out_ptr[i - 2 - 2 * stride] + out_ptr[i - 1 - 2 * stride] + out_ptr[i - 2 * stride] + out_ptr[i + 1 - 2 * stride] + out_ptr[i + 2 - 2 * stride] +
						out_ptr[i - 2 - stride] + out_ptr[i - 1 - stride] + out_ptr[i - stride] + out_ptr[i + 1 - stride] + out_ptr[i + 2 - stride] +
						out_ptr[i - 2] + out_ptr[i - 1] + out_ptr[i + 1] + out_ptr[i + 2]) > fiveby_filter_level - 6)
					{
						out_ptr[i] = 1;
					}
				}
			}
		}
	}
}

static __global__ void avg_out(float* out, cuComplex* d_fft, const int num_wins, const int averaging, const float offset) {

	int idx = threadIdx.x;
	float* out_ptr = &out[0];
	cuComplex* d_fft_ptr = &d_fft[0];
	const float threshold = -116;

	bool THRESHOLD = true;

	for (int j = 0; j < num_wins / averaging; j++){

		for (int i = blockIdx.x * blockDim.x + idx; i < NUM_SAMPS*averaging; i += blockDim.x * gridDim.x){

			out_ptr[((NUM_SAMPS / 2) + i) % NUM_SAMPS] += ( //This is done to correctly flip the fft result
				10 * log10(abs(d_fft_ptr[i].x * d_fft_ptr[i].x + d_fft_ptr[i].y * d_fft_ptr[i].y) / NUM_SAMPS) //DFT bin magnitude
				);
		}

		//		__syncthreads();

		if (THRESHOLD){
			out_ptr[(NUM_SAMPS / 2 + blockIdx.x * blockDim.x + idx) % NUM_SAMPS] = ((out_ptr[(NUM_SAMPS / 2 + blockIdx.x * blockDim.x + idx) % NUM_SAMPS] / averaging + offset) <= threshold) ? 1 : 0;
		}
		else {
			out_ptr[(NUM_SAMPS / 2 + blockIdx.x * blockDim.x + idx) % NUM_SAMPS] = (out_ptr[(NUM_SAMPS / 2 + blockIdx.x * blockDim.x + idx) % NUM_SAMPS] / averaging + offset);
		}

		out_ptr += NUM_SAMPS; //increment out_ptr by one frame of averages
		d_fft_ptr += NUM_SAMPS*averaging; //increment d_fft_ptr by number of frames averaged
	}
}

/* BACKUP 
static __global__ void avg_out(float* out, cuComplex* d_fft, const int num_wins, const int averaging) {

	int idx = threadIdx.x;
	float* out_ptr = &out[0];
	cuComplex* d_fft_ptr = &d_fft[0];

	for (int j = 0; j < num_wins / averaging; j++){

		for (int i = blockIdx.x * blockDim.x + idx; i < NUM_SAMPS*averaging; i += blockDim.x * gridDim.x){

			out_ptr[i%NUM_SAMPS] += (
				10 * log10(abs(d_fft_ptr[i].x * d_fft_ptr[i].x + d_fft_ptr[i].y * d_fft_ptr[i].y) / NUM_SAMPS) //DFT bin magnitude
				);
		}
		out_ptr += NUM_SAMPS; //increment out_ptr by one frame of averages
		d_fft_ptr += NUM_SAMPS*averaging; //increment d_fft_ptr by number of frames averaged
	}
}*/

// Complex multiplication with a real
static __device__ __host__ inline cuComplex ComplexRMul(cuComplex cmplx, float real)
{
	cuComplex c;
	c.x = cmplx.x * real;
	c.y = cmplx.y * real;
	return c;
}
