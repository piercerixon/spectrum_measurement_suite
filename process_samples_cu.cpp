/****** Process_Samples.cpp ******/

//	2015/16, Pierce Rixon, pierce.rixon@gmail.com
//	This program is intended to be used in conjunction with the files generated from Record_RX_Samples || rx_samples_to_file v10+. 

/*********************************/

//#include <uhd/types/tune_request.hpp>
//#include <uhd/utils/thread_priority.hpp>
//#include <uhd/utils/safe_main.hpp>
//#include <uhd/usrp/multi_usrp.hpp>
//#include <uhd/exception.hpp>
//#include <uhd/convert.hpp>
//#include <convert/convert_common.hpp>
#include "process.h"
#include "cuda_process.h"

#include <Windows.h>
#include <sys/stat.h>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <iostream>
#include <fstream>
#include <csignal>
#include <complex>
#include <cmath>
#include <fftw3/fftw3.h>
#include <mgl2/mgl.h>
#include <mgl2/fltk.h>

//Globals for processing
float centre_freq, samp_rate, bandwidth, rx_gain; //Read in from config file
float threshold = -113;

int frame_count = 0;
std::string device, filename;
long long ws_count = 0;

bool endfile = false;
//THIS IS THE WORST ITS ONLY FOR TESTS FORGIVE PLS
bool COMPARE = false;
const int GA_ROWS = 25;
unsigned char GA[GA_ROWS][WIN_SAMPS];

//Container for Plot
mglFLTK *globalr = NULL;
const int num_disp_pts = 2048;

//Global for logfile
std::ofstream logfile;

//mutexes for updating display
boost::mutex display_mtx, stream_mtx, fft_mtx;
boost::condition_variable_any disp_cond;//, fft_cond, stream_cond;

//mutexes and cond vars for FFT threads and receiver
boost::mutex fft_mutex[NUM_THREADS], recv_mutex, super_mutex;
boost::condition_variable_any fft_cond, recv_cond, super_cond;

//constructor definition
window::window(int t, int bw, int f, int ws, int n) {
	timescale = t;
	bandwidth = bw;
	frequency = f;
	whitespace = ws; //could be long long, will need to test experimentally, u_int should be enough
	frame_no = n;
}

int main(int argc, char*argv[]) {

	char select_file[MAX_PATH] = "";

	OPENFILENAMEA openfile = {0};  //http://www.cplusplus.com/forum/windows/82432/ Thanks Disch!

	openfile.lStructSize = sizeof(openfile);
	openfile.lpstrFilter = "data file\0*.dat\0config file\0*.cfg\0";
	openfile.nFilterIndex = 1; //because all indexes start at 1 :D
	openfile.Flags = OFN_FILEMUSTEXIST;  //only allow the user to open files that actually exist

	// the most important bits:
	openfile.lpstrFile = select_file;
	openfile.nMaxFile = MAX_PATH;  // size of our 'buffer' buffer
	
	std::cout << "\nSelect samples" << std::endl;
	//Call the open file dialog
	if( !GetOpenFileNameA( &openfile ) )
	{
	  //'Cancel'
		std::cout << "No file selected!" << std::endl;
	}
	else
	{
	  //'OK'
		std::cout << "Loading file: " << select_file << std::endl;
	}

	if(COMPARE) {
	for (int i = 0; i < GA_ROWS; i++) {
		for (int j = 0; j < WIN_SAMPS; j++) {
			GA[i][j] = 0;
		}
	}
	std::ifstream win_dmp ("Window_dump_t128k.csv",std::ifstream::in); //perform comparison of freshly computed windows, versus a window preservation file. 
	std::string win;
	std::istringstream dmp;

	int ts = 0;
	int freq = 0;
	int bw = 0;
	int frame = 0;

	if(win_dmp.is_open()){
		std::getline(win_dmp,win);
		while(std::getline(win_dmp,win,',')){ //timescale
			dmp.clear();
			dmp.str(win);
			dmp >> ts;

			std::getline(win_dmp,win,','); // frequency
			dmp.clear();
			dmp.str(win);
			dmp >> freq;

			std::getline(win_dmp,win,','); //bandwidth
			dmp.clear();
			dmp.str(win);
			dmp >> bw;

			std::getline(win_dmp,win,','); //discard whitespace
			std::getline(win_dmp,win); // frame_no
			dmp.clear();
			dmp.str(win);
			dmp >> frame;

			for(int k = (frame - ts); k < frame; k++){
				for(int q = freq; q < (freq + bw); q++) {
					GA[k][q] = 1;
				}
			}
		}
	}
	std::cout << "TEST: Test array populated\n";
	//system("Pause");
	}
	
	//test();
	//return 0;

	//Logfile
	if(LOG) logfile.open("logfile.log",std::ofstream::out);

	//Setup Processing
	//Configuration
	std::string cfgfile = "sample_config.cfg"; //FIXME!!!!!!!!!!!!!!!!!!!!!!!!!! 23/2/16 - srs
	std::ifstream config (cfgfile, std::ifstream::in);
	if(config.fail()) {
		std::cout << "\nCan you actually pick a sample set with a config file bro?\n\nThis program will now terminate" << std::endl; 
		return 0;
	}

	//Read in configuration File and set appropriate program parameters
	// use argv[1] to grab the config filename specified by the user - or use a file picker?
	std::string line;
	std::istringstream sin;
	int i = 0;

	if(config.is_open()){
		while (std::getline(config,line)) {
			sin.clear();
			sin.str(line);
			if(i==0) sin >> filename; //Currently stops at a space - its an istringstream problem 
			else if(i==1) sin >> device;
			else if(i==2) sin >> centre_freq;
			else if(i==3) sin >> bandwidth;
			else if(i==4) sin >> samp_rate;
			else if(i==5) sin >> rx_gain;
			else if(i > 5) {
				std::cout << "Too many arguments in configuration ... terminating" << std::endl;
				return 0;
			}
			std::cout << line << std::endl;
			i++;
		}
		if(LOG) logfile << "Configuration successful\n";
	}	

	std::cout << "Now testing CUDA compatibility" << std::endl;
	cufft_proc <<<1, 1024 >>>();
	
	//Samples Setup 
	std::ifstream in_samples;
	//std::string filename = "samples_0";
	//std::string sampfile = filename + ".dat"; //This should be taken from the cfg file
	std::string sampfile = select_file;
	in_samples.open(sampfile, std::ifstream::binary);
	in_samples.seekg(0, std::ifstream::beg);

	struct _stat64 stat_buff;
	_stat64(sampfile.c_str(), &stat_buff);
	double file_size = stat_buff.st_size;

	std::cout << "Filesize: " << file_size/std::pow(2.0,20) << " MB" << std::endl;

	if(LOG) logfile << "Samples found\n";

	std::complex<short>* re;//Buffer for read in samples
	re = (std::complex<short>*) malloc(file_size);

	if(in_samples.read((char*)re,file_size)) {
		std::cout << "File Loaded" << std::endl;
	} else {
		std::cout << "File Failed to Load" << std::endl;
		system("Pause");
	}
	
	//DISPLAY setup
	mglData *mglY = new mglData(num_disp_pts); //mglData object for plotting 	

	if(DISPLAY)	{
		boost::thread spawnWindow(init);	//Thread runs 5ever
		boost::this_thread::sleep(boost::posix_time::milliseconds(500));
		boost::thread display_thread(boost::bind(update,boost::ref(globalr),boost::ref(mglY)));
	}

	//This is where the magic happens
	fftf_proc_s(re,0,mglY,file_size);
		
	std::cout << "We done lah" << std::endl;

	std::cout << "SuperV: Cleaning up" << std::endl;
	//TODO: Cleanup
	if(in_samples.is_open()){
		in_samples.close();
	}
	if(LOG && logfile.is_open()){
		in_samples.close();
	}
	if(config.is_open()){
		in_samples.close();
	}
	free(re);
	system("pause");
}

/*
void fftf_proc (std::complex<float>* buff, int idx, mglData *mglY) {
	
	boost::unique_lock<boost::mutex> lock(fft_mutex[idx]);
	//Prepare your body for the FFT
	//CONVENTION fftwf = floats, instead of doubles used before. AS we only have 14 bits of precision it doesnt make sense to use double point precision
	// but im using doubles anyway :D
	

	fftwf_complex *fft_buff; 
	fftwf_plan plan;
	float win_power = 0;
	float *w_array, *avg_buff;
	int avg_count = 0;	
	const char* filename = "fftwf_plan";

	int frame_count = 0;

	int error_counter = 0;

	const int NUM_WINS = 1;

	int ws_array[WIN_SAMPS];
	unsigned char ws_frame[WIN_SAMPS];
	int overlap[WIN_SAMPS];
	for(int i = 0; i < WIN_SAMPS; i++){
		ws_array[i] = 0;
		ws_frame[i] = 0;
		overlap[i] = 0;
	}
	//long long ws_count = 0;
	std::vector<window> window_vec;
	std::vector<window>* w_vec_ptr = &window_vec;
	int frame_number = 0;
	bool init = false;

	fft_buff = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * WIN_SAMPS); //buffer allocated for Output
	w_array = (float*) malloc(sizeof(float) * WIN_SAMPS); //allocate memory for window coefficients
	
	avg_buff = (float*) malloc(sizeof(float) * WIN_SAMPS); //buffer allocated for averaging FFT results
	
	//Cast the std::complex<double> as a fftw_complex
	if (!fftwf_import_wisdom_from_filename(filename)) {
		std::cout << "should use wisdom next time :)" << std::endl;
		plan = fftwf_plan_dft_1d(WIN_SAMPS, reinterpret_cast<fftwf_complex*>(&buff[0]), fft_buff, FFTW_FORWARD, FFTW_MEASURE);
		std::cout << "FFTW PLAN: " << fftwf_export_wisdom_to_filename(filename) << std::endl;
	}
	//plan = fftwf_plan_dft_1d(WIN_SAMPS, reinterpret_cast<fftwf_complex*>(&buff[0]), fft_buff, FFTW_FORWARD, FFTW_PATIENT);
	else plan = fftwf_plan_dft_1d(WIN_SAMPS, reinterpret_cast<fftwf_complex*>(&buff[0]), fft_buff, FFTW_FORWARD, FFTW_MEASURE);
	
	//Create coefficient array and x axis index for plotting
	for (int i = 0; i < WIN_SAMPS; i++) {
			w_array[i] = 0.35875 - 0.48829*cos(2*pi*i/(WIN_SAMPS-1)) + 0.14128*cos(4*pi*i/(WIN_SAMPS-1)) - 0.01168*cos(6*pi*i/(WIN_SAMPS-1)); //blackmann harris window		
			win_power += (w_array[i]*w_array[i]); //this computes the total window power and normalises it to account for DC gain due to the window.
	}
	win_power /= WIN_SAMPS; //normalise the total window power across each sample.

	//std::cout << -10*std::log10(win_power) << std::endl;
	//double offset = (- 10*std::log10(win_power) //DC gain of window
	//				 - (174 - 10*std::log10(bandwidth/samp_rate))); //Noise floor
	float offset = - 10 - rx_gain; // - 10*std::log10(win_power); //-10 is the MAX power detected by the ADC and take into account the gain of the frontend.

	std::cout << "FFT[" << idx << "]: Spawned" << std::endl;
	if(LOG) logfile << "FFT[" << idx << "]: Initialised and Sleeping\n";

	super_cond.notify_one();
	fft_cond.wait(lock);

	std::ofstream fftout;
	fftout.open ("fftout.csv");

	while(true) {
		if(LOG) logfile << "FFT[" << idx << "]: Awake ";
		boost::system_time now = boost::get_system_time();

		//Apply window to samples
		//if ((int)(*num_rx_samps) == WIN_SAMPS*NUM_THREADS) {
			for (int i = 0; i < WIN_SAMPS; i++) {
				buff[i] *= w_array[i];
			}

			//Spin the FFT yoooooooooo
			fftwf_execute(plan);

			//Keep a moving average of every FFT that has happened until the averaging says so :D
			//Probably need to optimise this better. Pointer arithmetic is probably the go. 
			//#pragma omp parallel for

			for (int i = 0; i < WIN_SAMPS; i++) { 
				avg_buff[i] += ( 
					10*std::log10(std::abs(fft_buff[i][0]*fft_buff[i][0] + fft_buff[i][1]*fft_buff[i][1])/WIN_SAMPS) //DFT bin magnitude
					); 
			}				
						
			//Count the number of FFT frames that have been processed thus far
			avg_count++;
			
			//Perform ws detection 

			if (avg_count%10 == 0 || endfile) {
				//frame_count++;
				
				for(int i = 0; i < WIN_SAMPS; i++) {

					if(!endfile) {
						if((avg_buff[((WIN_SAMPS/2)+i)%WIN_SAMPS]/avg_count + offset) > threshold) ws_frame[i] = 0; //whitespace does not exist
						else {
							ws_frame[i] = 1; //whitespace exists
							ws_count++;
						}
					}
					else ws_frame[i] = 0; //Set final frame to all 0's to assist with correct window detection.
				}

				if(COMPARE) {
					//std::cout << "Comparing" << std::endl;
					bool error = false;
					int temp = 0;
				
					for(int i = 0; i < WIN_SAMPS; i++) {
						if(GA[frame_number][i] != ws_frame[i]) {
							if(error) {
								std::cout << "TEST: " << frame_number << "," << i;
								temp = i;
							}
							error = false;
							error_counter++;
							if(GA[frame_number][i] == 1) fftout << 3 << ',';
							else if(GA[frame_number][i] == 0) fftout << 4 << ',';
							else fftout << 7 << ',';
						}
						else { 
							if(!error && temp > 0) std::cout << "TEST: Num Err: " << i - temp << std::endl;
							fftout << (int)ws_frame[i] << ',';
							error = true;
						}
						}
						fftout << "\n";
						//if(error) system("Pause");
						
				} else {
					for(int i = 0; i < WIN_SAMPS; i++) {
						fftout << (int)ws_frame[i] << ',';
					}
					fftout << "\n";
				}

				if(init) detect_dev(ws_frame, NUM_WINS, ws_array, overlap, w_vec_ptr, frame_number);
				else {
					for(int i = 0; i < WIN_SAMPS; i++) {
						ws_array[i] = (int)ws_frame[i]; //This is done because the first entry in ws_array has to be initialised with the first frame
					}
					init = true;
				}

				frame_number++;
			//	for (int i = 0; i < WIN_SAMPS; i++) {
			//		std::cout << (int)ws_frame[i] << ',';
			//	}

				

			//if(avg_count%10 == 0) {
			// DISPLAY CODE, got to flip the buffer around correctly !!!!

				//display the averaged samples
				if (idx == 0 && display_mtx.try_lock() && DISPLAY) { //ENTER CRITICAL REGION, will not block the FFT thread.
					if(LOG) logfile << "FFT[" << idx << "]: Displaying ";

					std::cout << "Updating frame with averages: " << avg_count << std::endl;

					int step = WIN_SAMPS/num_disp_pts;
					float val_max, val_min = 0;
					float temp = 0;
					int index = 0;

					for (int i = 0; i < num_disp_pts; i++) {
						index = i*step;
						val_max = -1000;
						val_min = 0;
						temp = -1000; //If you actually get a level lower than this, you have issues :)

						for (int j = 0; j < step; j++) {//Grab the max and min values from each step set
							temp = (avg_buff[index + j]/avg_count); 

							if(temp > val_max) {val_max = temp;} //Take the MAX value
							else if(temp < val_min) {val_min = temp;} //Else Take the MIN value
						
							//avg_buff[index + j] = 0;

						//std::cout << "point: " << ((WIN_SAMPS/2)+i)%WIN_SAMPS << " val: " << (avg_buff[i] / avg_count) << std::endl;
						}

						val_max += offset;
						mglY->a[((num_disp_pts/2)+i)%num_disp_pts] = val_max;
					}
					
					display_mtx.unlock();
					disp_cond.notify_one();
				}
				//Clear the average once it has been processed, to prepare it for the next set.
				for(int i = 0; i < WIN_SAMPS; i++) avg_buff[i] = 0;
				avg_count = 0;
			}

		//This is here to provide the final window detection and outputting of the window vector to file.
		if(endfile) {
			std::cout << "\nFFT["<< idx << "]: Error Count: " << error_counter << std::endl;

			std::ofstream window_dump;
			window_dump.open ("Window_dump.csv");
			window_dump << 	"timescale,frequency,bandwidth,whitespace,frame_no\n";
			std::vector<window>::iterator WINit; //Window iterator
			std::cout << "FFT["<< idx << "]: Outputting Whitespace Windows" << std::endl;
			for(WINit = window_vec.begin(); WINit < window_vec.end(); WINit++) {
				window_dump << WINit->timescale << "," << WINit->frequency << "," << WINit->bandwidth << "," << WINit->whitespace << "," << WINit->frame_no << "\n"; 
			}
			std::cout << "FFT["<< idx << "]: Output Complete" << std::endl;
			window_dump.flush();
			window_dump.close();
			fftout.flush();
			fftout.close();

			bar_plot(&window_vec); //If the end of the data has been reached, then display the data
			break;
		}	
		
		//Once all done, go back to sleep
		if(LOG) logfile << "FFT[" << idx << "]: Sleeping ";
		super_cond.notify_one();
		fft_cond.wait(lock);
	}

	std::cout << "FFT["<< idx << "]: Terminating" << std::endl;
	//destroy fft plan when all the fun is over
	fftwf_destroy_plan(plan);
	fftwf_free(fft_buff);
	fftwf_cleanup_threads();
	free(w_array);
}
*/
//fftf_proc_s( array of samples, index of fft thread, mgl container for display, size of the file being read)
void fftf_proc_s (std::complex<short>* re, int idx, mglData *mglY, double fsz) {
	
	//boost::unique_lock<boost::mutex> lock(fft_mutex[idx]);
	//Prepare your body for the FFT
	//CONVENTION fftwf = floats, instead of doubles used before. AS we only have 14 bits of precision it doesnt make sense to use double point precision
	// but im using doubles anyway :D

	std::complex<float>* buff;
	buff = (std::complex<float>*) malloc(sizeof(std::complex<float>) * WIN_SAMPS); //Buffer for floated samples n.n

	//std::complex<short>* s_ptr = &re[0];

	std::complex<float>* f_ptr = buff;

	fftwf_complex *fft_buff; 
	fftwf_plan plan;
	float win_power = 0;
	float *w_array, *avg_buff;
	int avg_count = 0;	
	const char* filename = "fftwf_plan";

	int frame_count = 0;
	int error_counter = 0;
	const int NUM_WINS = 1;

	int ws_array[WIN_SAMPS];
	unsigned char ws_frame[WIN_SAMPS];
	int overlap[WIN_SAMPS];
	for(int i = 0; i < WIN_SAMPS; i++){
		ws_array[i] = 0;
		ws_frame[i] = 0;
		overlap[i] = 0;
	}
	//long long ws_count = 0;
	std::vector<window> window_vec;
	std::vector<window>* w_vec_ptr = &window_vec;
	int frame_number = 0;
	bool init = false;

	fft_buff = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * WIN_SAMPS); //buffer allocated for Output
	w_array = (float*) malloc(sizeof(float) * WIN_SAMPS); //allocate memory for window coefficients
	
	avg_buff = (float*) malloc(sizeof(float) * WIN_SAMPS); //buffer allocated for averaging FFT results
	
	//Cast the std::complex<double> as a fftw_complex
	if (!fftwf_import_wisdom_from_filename(filename)) {
		std::cout << "should use wisdom next time :)" << std::endl;
		plan = fftwf_plan_dft_1d(WIN_SAMPS, reinterpret_cast<fftwf_complex*>(&buff[0]), fft_buff, FFTW_FORWARD, FFTW_MEASURE);
		std::cout << "FFTW PLAN: " << fftwf_export_wisdom_to_filename(filename) << std::endl;
	}
	//plan = fftwf_plan_dft_1d(WIN_SAMPS, reinterpret_cast<fftwf_complex*>(&buff[0]), fft_buff, FFTW_FORWARD, FFTW_PATIENT);
	else plan = fftwf_plan_dft_1d(WIN_SAMPS, reinterpret_cast<fftwf_complex*>(&buff[0]), fft_buff, FFTW_FORWARD, FFTW_MEASURE);
	
	//Create coefficient array and x axis index for plotting
	for (int i = 0; i < WIN_SAMPS; i++) {
			w_array[i] = 0.35875 - 0.48829*cos(2*pi*i/(WIN_SAMPS-1)) + 0.14128*cos(4*pi*i/(WIN_SAMPS-1)) - 0.01168*cos(6*pi*i/(WIN_SAMPS-1)); //blackmann harris window		
			win_power += (w_array[i]*w_array[i]); //this computes the total window power and normalises it to account for DC gain due to the window.
	}
	win_power /= WIN_SAMPS; //normalise the total window power across each sample.

	//std::cout << 10*std::log10(win_power) << std::endl;

	//double offset = (- 10*std::log10(win_power) //DC gain of window
	//				 - (174 - 10*std::log10(bandwidth/samp_rate))); //Noise floor
	float offset = - 10 - rx_gain + 10*std::log10(win_power); //-10 is the MAX power detected by the ADC and take into account the gain of the frontend.

	std::cout << "FFT[" << idx << "]: Spawned" << std::endl;

	std::ofstream fftout;
	fftout.open ("fftout.csv");

	long long samp = 0;
	double chars_read = 0.0;
	int reduce = 0;
	double last_count = 0;
	double rate = 0;
	double progress = 0;
	boost::posix_time::time_duration diff,read;
	boost::posix_time::ptime now1;
	double time_diff;
	//std::cout << std::fixed << std::setprecision(2);

	LARGE_INTEGER wind, ftpl, avge, perffreq;
	double wf, fa, ae, pf;
	QueryPerformanceFrequency(&perffreq);
	pf = perffreq.QuadPart;

	std::vector<float> plot(num_disp_pts, 0);

	while(true) {
		boost::system_time now = boost::get_system_time();

		//Apply window to samples
		//if ((int)(*num_rx_samps) == WIN_SAMPS*NUM_THREADS) {
			if (samp > reduce*fsz/1000) {	//Progress Bar
			
				if(reduce%10 == 0) {
					diff = boost::posix_time::microsec_clock::local_time() - now1;
					time_diff = (double)boost::posix_time::time_duration::ticks_per_second() / (double)diff.ticks();
					now1 = boost::posix_time::microsec_clock::local_time();
					rate = ((samp - last_count)/std::pow(2.0,20))*time_diff;
					last_count = samp;
				}

				reduce++;
				progress = (samp/fsz)*100;
				std::cout << progress << " % complete at " << rate << " MB/s \t\t\r";
				
			}

			QueryPerformanceCounter(&wind);
			//for (int i = 0, j = 0; i < WIN_SAMPS*2; i+=2, j++) { //WIN_SAMPS*2 BECAUSE THEY ARE COMPLEX
			for (int i = 0; i < WIN_SAMPS; i++) { 
				//Looks janky, but is actually for good reason. The modulo, is to ensure the f_ptr does not overrun buff, as buff is complex, there is 2 floats per 'float'
				//the s_ptr points to the entire dataset cached in memory, thus it can run untill the end of the file
				//f_ptr[i] = ((s_ptr[samp++])*1.0f/32767.0f) * w_array[j]; 
				//f_ptr[i+1] = ((s_ptr[samp++])*1.0f/32767.0f) * w_array[j];
				f_ptr[i].real( ((*re).real()*1.0f/32767.0f) * w_array[i] ); 
				f_ptr[i].imag( ((*re).imag()*1.0f/32767.0f) * w_array[i] );
				re++;
				samp+=4;
			}
			
			QueryPerformanceCounter(&ftpl);

			//Spin the FFT yoooooooooo
			fftwf_execute(plan);

			QueryPerformanceCounter(&avge);
			//Keep a moving average of every FFT that has happened until the averaging says so :D
			//Probably need to optimise this better. Pointer arithmetic is probably the go. 
			//#pragma omp parallel for

			for (int i = 0; i < WIN_SAMPS; i++) { 
				avg_buff[i] += ( 
					10*std::log10(std::abs(fft_buff[i][0]*fft_buff[i][0] + fft_buff[i][1]*fft_buff[i][1])/WIN_SAMPS) //DFT bin magnitude
					); 
			}				

			wf = (ftpl.QuadPart - wind.QuadPart)/pf;
			QueryPerformanceCounter(&wind);	
			fa = (avge.QuadPart - ftpl.QuadPart)/pf;
			ae = (wind.QuadPart - avge.QuadPart)/pf;
			//std::cout << "Windowing: " << wf << "s FFT: " << fa << "s Avg: " << ae << "s" << std::endl; 
			//Count the number of FFT frames that have been processed thus far
			avg_count++;
			
			//Perform ws detection 

			if (avg_count%10 == 0 || samp >= fsz) {
				//frame_count++;
				
				for(int i = 0; i < WIN_SAMPS; i++) {

					if(samp < fsz) {
						if((avg_buff[((WIN_SAMPS/2)+i)%WIN_SAMPS]/avg_count + offset) > threshold) ws_frame[i] = 0; //whitespace does not exist
						else {
							ws_frame[i] = 1; //whitespace exists
							ws_count++;
						}
					}
					else ws_frame[i] = 0; //Set final frame to all 0's to assist with correct window detection.
				}

				if(COMPARE) {
					//std::cout << "Comparing" << std::endl;
					bool error = false;
					int temp = 0;
				
					for(int i = 0; i < WIN_SAMPS; i++) {
						if(GA[frame_number][i] != ws_frame[i]) {
							if(error) {
								std::cout << "TEST: " << frame_number << "," << i;
								temp = i;
							}
							error = false;
							error_counter++;
							if(GA[frame_number][i] == 1) fftout << 3 << ',';
							else if(GA[frame_number][i] == 0) fftout << 4 << ',';
							else fftout << 7 << ',';
						}
						else { 
							if(!error && temp > 0) std::cout << "TEST: Num Err: " << i - temp << std::endl;
							fftout << (int)ws_frame[i] << ',';
							error = true;
						}
						}
						fftout << "\n";
						//if(error) system("Pause");
						
				} else {
					for(int i = 0; i < WIN_SAMPS; i++) {
						fftout << (int)ws_frame[i] << ',';
					}
					fftout << "\n";
				}

				//if(init) detect_dev(ws_frame, NUM_WINS, ws_array, overlap, w_vec_ptr, frame_number);
				if(init) detect(ws_frame, NUM_WINS, ws_array, overlap, w_vec_ptr, frame_number);
				else {
					for(int i = 0; i < WIN_SAMPS; i++) {
						ws_array[i] = (int)ws_frame[i]; //This is done because the first entry in ws_array has to be initialised with the first frame
					}
					init = true;
				}

				frame_number++;

			// DISPLAY CODE, got to flip the buffer around correctly !!!!

				//display the averaged samples
				if (idx == 0 && display_mtx.try_lock() && DISPLAY) { //ENTER CRITICAL REGION, will not block the FFT thread.
					if(LOG) logfile << "FFT[" << idx << "]: Displaying ";

					std::cout << "Updating frame with averages: " << avg_count << std::endl;

					int step = WIN_SAMPS/num_disp_pts;
					float val_max, val_min = 0;
					float temp = 0;
					int index = 0;

					for (int i = 0; i < num_disp_pts; i++) {
						index = i*step;
						val_max = -1000;
						val_min = 0;
						temp = -1000; //If you actually get a level lower than this, you have issues :)

						for (int j = 0; j < step; j++) {//Grab the max and min values from each step set
							temp = (avg_buff[index + j]/avg_count); 

							if(temp > val_max) {val_max = temp;} //Take the MAX value
							else if(temp < val_min) {val_min = temp;} //Else Take the MIN value
						
							//avg_buff[index + j] = 0;

						//std::cout << "point: " << ((WIN_SAMPS/2)+i)%WIN_SAMPS << " val: " << (avg_buff[i] / avg_count) << std::endl;
						}

						val_max += offset;
						plot[((num_disp_pts / 2) + i) % num_disp_pts] = val_max;
					}
					
					(*mglY).Set(plot);
					display_mtx.unlock();
					disp_cond.notify_one();
				}
				//Clear the average once it has been processed, to prepare it for the next set.
				for(int i = 0; i < WIN_SAMPS; i++) avg_buff[i] = 0;
				avg_count = 0;
			}

		//This is here to provide the final window detection and outputting of the window vector to file.
		if(samp >= fsz) {
			std::cout << "\nFFT["<< idx << "]: Error Count: " << error_counter << std::endl;

			std::ofstream window_dump;
			window_dump.open ("Window_dump.csv");
			window_dump << 	"timescale,frequency,bandwidth,whitespace,frame_no\n";
			std::vector<window>::iterator WINit; //Window iterator
			std::cout << "FFT["<< idx << "]: Outputting Whitespace Windows" << std::endl;
			for(WINit = window_vec.begin(); WINit < window_vec.end(); WINit++) {
				window_dump << WINit->timescale << "," << WINit->frequency << "," << WINit->bandwidth << "," << WINit->whitespace << "," << WINit->frame_no << "\n"; 
			}
			std::cout << "FFT["<< idx << "]: Output Complete" << std::endl;
			window_dump.flush();
			window_dump.close();
			fftout.flush();
			fftout.close();

			bar_plot(&window_vec); //If the end of the data has been reached, then display the data
			break;
		}	
	}

	std::cout << "FFT["<< idx << "]: Terminating" << std::endl;
	//destroy fft plan when all the fun is over
	fftwf_destroy_plan(plan);
	fftwf_free(fft_buff);
	fftwf_cleanup_threads();
	free(w_array);
}


//Thread lives fiveever
void update(mglFLTK *psr, mglData *mglArr) {

	boost::unique_lock<boost::mutex> lock(display_mtx);
	
	while(true) {

		if (lock.owns_lock()) {
			//std::cout << "Updated sir!" << std::endl;
			psr->Adjust();
			psr->ClearFrame(); //clear 
			psr->Plot(*mglArr,"q0-");
			psr->Box();
			psr->SetTicks('y',10,9);
			psr->Axis("xy");
			psr->Grid("y!","W-");
			psr->Update(); //update display
		
			frame_count++;
		}

		disp_cond.wait(lock);
	}
}

void update_once(mglFLTK *psr, mglData *mglArr) {

	mglFLTK *gr = new mglFLTK("Power Spectrum"); //Yes the window object has to be spawned and then referenced within this thread. 
	mglData x(10000), y(10000), z(10000);  gr->Fill(x,"2*rnd-1");
	gr->Fill(y,"2*rnd-1"); gr->Fill(z,"exp(-6*(v^2+w^2))",x,y);
	mglData xx=gr->Hist(x,z), yy=gr->Hist(y,z);	xx.Norm(0,1);
	yy.Norm(0,1);
	gr->MultiPlot(3,3,3,2,2,"");   gr->SetRanges(-1,1,-1,1,0,1);
	gr->Axis("xy"); gr->Dots(x,y,z,"wyrRk"); gr->Box();
	gr->MultiPlot(3,3,0,2,1,"");   gr->SetRanges(-1,1,0,1);
	gr->Box(); gr->Axis("xy"); gr->Bars(xx);
	gr->MultiPlot(3,3,5,1,2,"");   gr->SetRanges(0,1,-1,1);
	gr->Box(); gr->Axis("xy"); gr->Barh(yy);
	gr->SubPlot(3,3,2);
	//gr->Puts(mglPoint(0.5,0.5),"Hist and\nMultiPlot\nsample","a",-6);
	gr->Run();
}

//Thread spawns plot window then runs 5ever
//The calc function can be easily used to send updates to the plot window, either in its own thread or not (referencing globalr - window object).
void init() {
	mglFLTK *gr = new mglFLTK("Power Spectrum"); //Yes the window object has to be spawned and then referenced within this thread. 
	// No i dont know why
	// Just accept it.
	globalr = gr;
	gr->SetDelay(.001);
	gr->SetRange('y',-130,-10);
	gr->SetRange('x',centre_freq - samp_rate/2,centre_freq + samp_rate/2);
	gr->Run();
}

void plot_samples(mglFLTK *psr, mglData *mglArr) {
	
	//Perform the Plotting!
	psr->Adjust();
	psr->ClearFrame(); //clear 
	psr->Plot(*mglArr,"q0-");
	psr->Box();
	psr->Axis("xy");
	psr->Update(); //update display
	
	frame_count++;

	display_mtx.unlock(); //Unlocking the mtx to enable next update.
	//std::cout << "Frame Time: " << psr->GetDelay() << std::endl;
}