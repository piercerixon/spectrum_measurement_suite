#ifndef PROCESS_H
#define PROCESS_H

#include "window.h"
#include <complex>
#include <vector>


/**************** Prototypes *****************/
//void init();
//void plot_samples(mglFLTK*, mglData*);
//void fftf_proc(std::complex<float>*,int,mglData*);
//void fftf_proc_s (std::complex<short>*, int, mglData*, double);
//void update(mglFLTK*, mglData*);
//void update_once(mglFLTK*, mglData*);

//Choose where the sample files exist
void select_file(char*);

//void detect(unsigned char* ws_frame, const int num_wins, int* ws_array, int *overlap, window window_vec)
void detect(float*, const int, int*, int*, std::vector<window>*, int*);
void detect_once(unsigned char*, const int, int*, int*, std::vector<window>*, int);
void detect_dev(unsigned char*, const int, int*, int*, std::vector<window>*, int);

void detect_ts(float*, const int, int*, std::vector<window>*, int*);
void detect_ts_once(unsigned char*, const int, int*, std::vector<window>*, int);

void detect_ts_rec(float*, double*, const int, int*, std::vector<window>*, int*);
void detect_ts_rec_once(unsigned char*, double*, const int, int*, std::vector<window>*, int);

void bar_plot(std::vector<window>*);
void read_samples(char*);
void read_samples_plot(char*);

void test();
template<class T> 
void print_vector(std::vector<T> &); //for convenience
int minimum(std::vector<int>&,int);


/**************** Measurement Globals/typedef ********************/

typedef unsigned int u_int; 

#ifndef PI
#define PI
static const float pi = float(std::acos(-1.0));
#endif

//const int WIN_SAMPS = 65536; //temporarly set here to be a definite number
const int WIN_SAMPS = 131072; //temporarly set here to be a definite number (OMFG THE DAMN THING RAN OUT OF STACK MEMORY LOL)
//const int WIN_SAMPS = 262144;
//const int WIN_SAMPS = 128;

//Program conditions for different modes
static bool DISPLAY = false;
static bool FFT = true;

//Misc options
static bool TIMING = false;
const int NUM_THREADS = 1;
const bool VERBOSE = false;
const bool LOG = true;

#endif