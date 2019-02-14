#include "fft_module.cuh"
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
//num_wins = number of requested frames + averaging -1, this is required to ensure the correct averaging parameters. 
//a requested frame is an FFT of size = resolution
//by the same logic, h_samp_array must be = num_wins * resolution. h_out will be = (num_wins - (averaging-1)) * resolution
void perform_fft(std::complex<short>* h_samp_arry, float* h_out, const int resolution, const int averaging, const int num_wins) {
	/*
	if (num_wins == 0) {
		std::cout << "AMG NO WINS!\n";
		num_wins = sizeof(h_samp_arry) / (2 * resolution);
		std::cout << "Number of windows: " << num_wins << std::endl;
	} */
	//const int num_wins = 1;
	//cuComplex* samp[resolution];
	//std::complex<short>* d_samp;

	cudaError_t cudaStatus;
	cufftResult fftStatus;

	//Create cufft plan, turns out cufft handles its own memory transfers, so we must use callbacks in order to avoid numerous reads and writes in the device
	//Will however use multiple kernels initially, then see what the performance improvement is with callbacks at a later stage. n.n
	cufftHandle plan;
	fftStatus = cufftPlan1d(&plan, resolution, CUFFT_C2C, (num_wins + averaging - 1)); //is deprecated
	//int n[1] = { resolution };
	//fftStatus = cufftPlanMany(&plan, 1, n, 
	//	NULL, 1, resolution, 
	///	NULL, 1, resolution, 
	// CUFFT_C2C, (num_wins + averaging - 1));
	if (fftStatus != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error creating plan: %d\n", fftStatus);
		goto Error;
	}

	// for outputting of averaged and processed samples /
	
	float* d_out;
	
	//cast stl complex to cuda complex
	cuComplexShort* h_samp_ptr = (cuComplexShort*)&h_samp_arry[0];

	//std::cout << h_samp_arry[0].real() << "," << h_samp_arry[0].imag() << " cuCmplx" << h_samp_ptr[0].x << "," << h_samp_ptr[0].y << std::endl;
	
	float* h_coef;
	h_coef = (float*)malloc(sizeof(float)*resolution);
	float* d_coef;

	cuComplexShort* d_samp;
	cuComplex* d_fftbuff;

	float win_power = 0;
	int rx_gain = 30;

	//Create coefficient array and x axis index for plotting
	for (int i = 0; i < resolution; i++) {
		h_coef[i] = 0.35875 - 0.48829*cos(2 * pi*i / (resolution - 1)) + 0.14128*cos(4 * pi*i / (resolution - 1)) - 0.01168*cos(6 * pi*i / (resolution - 1)); //blackmann harris window		
		win_power += (h_coef[i] * h_coef[i]); //this computes the total window power and normalises it to account for DC gain due to the window.
	}
	win_power /= resolution; //normalise the total window power across each sample.

	const float offset = 10 - rx_gain + 10 * std::log10(win_power); //10 is the MAX power detected by the ADC and take into account the gain of the frontend.

	//printf("GPU Offset: %f", offset);

	cuda_memcheck();



	//allocate the memory for the GPU
	cudaStatus = cudaMalloc((cuComplexShort**)&d_samp, sizeof(cuComplexShort)* resolution*(num_wins + averaging - 1));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "d_samp cudaMalloc failed! %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cuda_memcheck();
	
	cudaStatus = cudaMalloc((float**)&d_coef, sizeof(float)*resolution);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "d_coef cudaMalloc failed! %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
	cuda_memcheck();

	cudaStatus = cudaMalloc((cuComplex**)&d_fftbuff, sizeof(cuComplex)*resolution*(num_wins + averaging - 1));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "d_fftbuff cudaMalloc failed! %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
	cuda_memcheck();

	//Transfer data to GPU
	cudaStatus = cudaMemcpy(d_coef, h_coef, sizeof(float)*resolution, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to Device failed! %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
	cudaStatus = cudaMemcpy(d_samp, h_samp_ptr, sizeof(cuComplexShort)*resolution*(num_wins + averaging - 1), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to Device failed! %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}


	cufft_prep <<< (resolution*num_wins) / CU_THD, CU_THD >> > (d_fftbuff, d_samp, d_coef, (num_wins + averaging - 1), resolution); //This will create (WIN_SAMPS*num_wins)/CU_THD blocks, with 1024 threads per block
	
	checkCudaErrors(cudaFree(d_samp));
	checkCudaErrors(cudaFree(d_coef));

	//inplace fft
	fftStatus = cufftExecC2C(plan, d_fftbuff, d_fftbuff, CUFFT_FORWARD);
	if (fftStatus != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed %d\n", fftStatus);
		goto Error;
	}

	cudaStatus = cudaMalloc((float**)&d_out, sizeof(float)*resolution * num_wins);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaMemset(d_out, 0, sizeof(float)*resolution * num_wins); //initialise to zero
	
	//Do something with the fft'd samples, like average them, then output them to the host, where the host can perform detection.
	avg_out <<< resolution / CU_THD, CU_THD >>> (d_out, d_fftbuff, num_wins, averaging, offset, resolution);
	filter <<< resolution / CU_THD, CU_THD >>> (d_out, num_wins, resolution); //As this uses the correct moving average, num_wins does not have to be divided out

	cudaStatus = cudaMemcpy(h_out, d_out, sizeof(float)*resolution * num_wins, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to Host failed! %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:

	cufftDestroy(plan);
	checkCudaErrors(cudaFree(d_out));
	//checkCudaErrors(cudaFree(d_samp));
	//checkCudaErrors(cudaFree(d_coef));
	checkCudaErrors(cudaFree(d_fftbuff));

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed! %s", cudaGetErrorString(cudaStatus));
	}

	//return h_out;
}

//Kernel Call
//https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-use-cufft-callbacks-custom-data-processing/ for inspiration
static __global__ void cufft_prep(cuComplex* d_fft, cuComplexShort* d_s, float* d_w, const int num_wins, const int resolution) {

	int idx = threadIdx.x;
	
	//blockDim = number of threads in a block
	//This will take an array of complex shorts (14b samples) an array of cuComplex and a window array, will convert the com_short to cuComplex (com_float), correctly scale the samples and apply the appropriate window prepping it for fft
	for (int i = blockIdx.x * blockDim.x + idx; i < resolution*num_wins; i += blockDim.x * gridDim.x){
		d_fft[i].x = (d_s[i].x*1.0f / 32767.0f) * d_w[i%resolution];
		d_fft[i].y = (d_s[i].y*1.0f / 32767.0f) * d_w[i%resolution];
	}
	//if(idx == 0) printf("d_s[%d]: %f,%f fftbuff %f,%f\n", idx, d_s[idx].x, d_s[idx].y, d_s[idx].x, d_s[idx].x);
}

static __global__ void filter(float*out, const int num_wins, const int resolution){

	int idx = threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	float* out_ptr = &out[0];
	const int fiveby_filter_level = 13; //normally 16 for 5x5, 13 for aggressive.
	const int filter_level = 5; // 3x3 kernel 

	bool FIVEBY = true; //for use later

	//increment loop by 1, and decrease total run by 1 to accomodate for edges of the kernel
	if (!FIVEBY){
		for (int i = (blockIdx.x * blockDim.x + idx) + stride; i < resolution*(num_wins - 1); i += stride){

			if (out_ptr[i] == 0 && (blockIdx.x + idx != 0 || blockIdx.x + idx != resolution - 1)){
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

			if (out_ptr[i] == 0 && (blockIdx.x * blockDim.x + idx != 0 && blockIdx.x * blockDim.x + idx != resolution - 1 &&
				blockIdx.x * blockDim.x + idx != 1 && blockIdx.x * blockDim.x + idx != resolution - 2)){

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
					if (( //unrolled here for efficiencies, this isnt called ... bug
						out_ptr[i - 2 - 2 * stride] + out_ptr[i - 1 - 2 * stride] + out_ptr[i - 2 * stride] + out_ptr[i + 1 - 2 * stride] + out_ptr[i + 2 - 2 * stride] +
						out_ptr[i - 2 - stride] + out_ptr[i - 1 - stride] + out_ptr[i - stride] + out_ptr[i + 1 - stride] + out_ptr[i + 2 - stride] +
						out_ptr[i - 2] + out_ptr[i - 1] + out_ptr[i + 1] + out_ptr[i + 2] +
						out_ptr[i - 2 + stride] + out_ptr[i - 1 + stride] + out_ptr[i + stride] + out_ptr[i + 1 + stride] + out_ptr[i + 2 + stride]) > fiveby_filter_level - 3)
					{
						out_ptr[i] = 1;
					}
				}

				else if (j == num_wins - 1){
					if (( //unrolled here for efficiencies, neither is this ... BUG!!!!!
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

static __global__ void avg_out(float* out, cuComplex* d_fft, const int num_wins, const int averaging, const float offset, const int resolution) {
	
	//Need to modify for appropriate averaging output
	int idx = threadIdx.x;
	float* out_ptr = &out[0];
	cuComplex* d_fft_ptr = &d_fft[0];
	const float threshold = -96;

	bool THRESHOLD = true;

	for (int j = 0; j < num_wins; j++){ //what about the final set of frames? They should be retained and re-computed to maintain accurate averaging ...

		for (int i = blockIdx.x * blockDim.x + idx; i < resolution*averaging; i += blockDim.x * gridDim.x){
			//Moving average of each output bin according to the 'averaging' value - typically set to 10
			out_ptr[((resolution / 2) + i) % resolution] += (
				10 * log10(abs(d_fft_ptr[i].x * d_fft_ptr[i].x + d_fft_ptr[i].y * d_fft_ptr[i].y) / resolution) //DFT bin magnitude
				);
		}

//		__syncthreads();

		if (THRESHOLD){
			out_ptr[(resolution / 2 + blockIdx.x * blockDim.x + idx) % resolution] = ((out_ptr[(resolution / 2 + blockIdx.x * blockDim.x + idx) % resolution] / averaging + offset) <= threshold) ? 1 : 0;
		}
		else {
			out_ptr[(resolution / 2 + blockIdx.x * blockDim.x + idx) % resolution] = (out_ptr[(resolution / 2 + blockIdx.x * blockDim.x + idx) % resolution] / averaging + offset);
		}
//		if (out_ptr[blockIdx.x * blockDim.x + idx] <= threshold) out_ptr[blockIdx.x * blockDim.x + idx] = 1;
//		elseP out_ptr[blockIdx.x * blockDim.x + idx] = 0;

		out_ptr += resolution; //increment out_ptr by one frame of averages
		d_fft_ptr += resolution; //increment d_fft_ptr by one frame to maintain rolling average
	}
}

/* DEPRECATED
static __global__ void avg_out_filter(float* out, cuComplex* d_fft, const int num_wins, const int averaging, const float offset, const int resolution) {

	//Need to modify for appropriate averaging output
	//Remember whitespace is a 1
	int idx = threadIdx.x;
	float* out_ptr = &out[0];
	cuComplex* d_fft_ptr = &d_fft[0];
	const float threshold = -96;
	const int filter_level = 13; //normally 16 for fiveby, 13 for aggressive

	bool THRESHOLD = true;
	bool FILTER = true;
	bool FIVEBY = false;

	for (int j = 0; j < num_wins; j++){

		for (int i = blockIdx.x * blockDim.x + idx; i < resolution*averaging; i += blockDim.x * gridDim.x){

			out_ptr[((resolution / 2) + i) % resolution] += (
				10 * log10(abs(d_fft_ptr[i].x * d_fft_ptr[i].x + d_fft_ptr[i].y * d_fft_ptr[i].y) / resolution) //DFT bin magnitude
				);
		}

		//		__syncthreads();

		if (THRESHOLD){
			out_ptr[(resolution / 2 + blockIdx.x * blockDim.x + idx) % resolution] = ((out_ptr[(resolution / 2 + blockIdx.x * blockDim.x + idx) % resolution] / averaging + offset) <= threshold) ? 1 : 0;
		}
		else {
			out_ptr[(resolution / 2 + blockIdx.x * blockDim.x + idx) % resolution] = (out_ptr[(resolution / 2 + blockIdx.x * blockDim.x + idx) % resolution] / averaging + offset);
		}
		//		if (out_ptr[blockIdx.x * blockDim.x + idx] <= threshold) out_ptr[blockIdx.x * blockDim.x + idx] = 1;
		//		elseP out_ptr[blockIdx.x * blockDim.x + idx] = 0;

		out_ptr += resolution; //increment out_ptr by one frame of averages
		d_fft_ptr += resolution; //increment d_fft_ptr by number of frames averaged
	}

	//Now perform filtering, only if thresholding is performed

	if (THRESHOLD && FILTER && !FIVEBY){
		//Zero out pointer
		out_ptr = &out[0 + resolution]; //we dont want to filter the first row - at this stage anyway
		int absthreadidx = blockIdx.x * blockDim.x + threadIdx.x; //I wanted to be more explicit before, shortcutting here

		//j starts at 1 and ends at num_wins-1 to give the sufficient spacing for the 3x3 kernel
		for (int j = 1; j < num_wins-1; j++){

			if (j == 0) { //first row
			}
			else if (j == num_wins - 1) { //last row
			
			}

			if (absthreadidx == 0) { //left edge
			}
			else if (absthreadidx == resolution - 1) { //right edge
			}

			else { //everything else
				//If the centre of a kernel = 1, take a 3 by 3 kernel, and sum the edge cells, if greater than 7, can assume this is noise
				if (out_ptr[absthreadidx] == 0)
				{
					//Currently set to detect a lone cell. Can increase this for more agressive filtering. Though the kernel size may have to increase also
					if ((out_ptr[absthreadidx - resolution - 1] + out_ptr[absthreadidx - resolution] + out_ptr[absthreadidx - resolution + 1] +
						out_ptr[absthreadidx - 1] + out_ptr[absthreadidx + 1] +
						out_ptr[absthreadidx + resolution - 1] + out_ptr[absthreadidx + resolution] + out_ptr[absthreadidx + resolution + 1]) > 6)
					{
						out_ptr[absthreadidx] = 1;
					}
				}
			}

			out_ptr += resolution; //next row of output array (as the 2d output is really just a very long 1d array)
		}
	}

	else if (THRESHOLD && FILTER && FIVEBY){
		//Zero out pointer
		out_ptr = &out[0 + 2 * resolution]; //we dont want to filter the first 2 rows - at this stage anyway
		int absthreadidx = blockIdx.x * blockDim.x + threadIdx.x; //I wanted to be more explicit before, shortcutting here

		//j starts at 1 and ends at num_wins-1 to give the sufficient spacing for the 3x3 kernel
		for (int j = 2; j < num_wins - 2; j++){

			if (j == 0 || j == 1) { //first row
			}
			else if (j == num_wins - 2 || j == num_wins - 1) { //last row

			}

			if (absthreadidx == 0 || absthreadidx == 1) { //left edge
			}
			else if (absthreadidx == resolution - 1 || absthreadidx == resolution - 2) { //right edge
			}

			else { //everything else
				//If the centre of a kernel = 1, take a 3 by 3 kernel, and sum the edge cells, if greater than filter_level, can assume this is noise
				if (out_ptr[absthreadidx] == 0)
				{
					//Currently set to detect a lone cell. Can increase this for more agressive filtering. Though the kernel size may have to increase also
					if ((
						out_ptr[absthreadidx - 2 * resolution - 2] + out_ptr[absthreadidx - 2 * resolution - 1] + out_ptr[absthreadidx - 2 * resolution] + out_ptr[absthreadidx - 2 * resolution + 1] + out_ptr[absthreadidx - 2 * resolution + 2] +
						out_ptr[absthreadidx - resolution - 2] + out_ptr[absthreadidx - resolution - 1] + out_ptr[absthreadidx - resolution] + out_ptr[absthreadidx - resolution + 1] + out_ptr[absthreadidx - resolution + 2] +
						out_ptr[absthreadidx - 2] + out_ptr[absthreadidx - 1] + out_ptr[absthreadidx + 1] + out_ptr[absthreadidx + 2] +
						out_ptr[absthreadidx + resolution - 2] + out_ptr[absthreadidx + resolution - 1] + out_ptr[absthreadidx + resolution] + out_ptr[absthreadidx + resolution + 1] + out_ptr[absthreadidx + resolution + 2] +
						out_ptr[absthreadidx + 2 * resolution - 2] + out_ptr[absthreadidx + 2 * resolution - 1] + out_ptr[absthreadidx + 2 * resolution] + out_ptr[absthreadidx + 2 * resolution + 1] + out_ptr[absthreadidx + 2 * resolution + 2]
						) > filter_level)
					{
						out_ptr[absthreadidx] = 1;
					}
				}
			}

			out_ptr += resolution; //next row of output array (as the 2d output is really just a very long 1d array)
		}
	}

}
/*

/* BACKUP LOL
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

void cuda_memcheck() {
	size_t free_byte;

	size_t total_byte;

	cudaError_t cudaStatus;

	cudaStatus = cudaMemGetInfo(&free_byte, &total_byte);

	size_t used_byte = total_byte - free_byte;

	if (cudaStatus != cudaSuccess){

		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cudaStatus));

		exit(1);

	}
	else printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

		used_byte / 1024.0 / 1024.0, free_byte / 1024.0 / 1024.0, total_byte / 1024.0 / 1024.0);
}