/****** Process_Samples.cpp ******/

//	2015/16, Pierce Rixon, pierce.rixon@gmail.com
//	This program is intended to be used in conjunction with the files generated from STF (Samples_to_file)

/*********************************/

//#include <uhd/types/tune_request.hpp>
//#include <uhd/utils/thread_priority.hpp>
//#include <uhd/utils/safe_main.hpp>
//#include <uhd/usrp/multi_usrp.hpp>
//#include <uhd/exception.hpp>
//#include <uhd/convert.hpp>
//#include <convert/convert_common.hpp>
#include "process.h"

#include <Windows.h>
#include <sys/stat.h>
#include <thread>
//#include <boost/python.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <fstream>
#include <csignal>
#include <complex>
#include <cmath>
#include <fftw3/fftw3.h>
//#include <mgl2/mgl.h>
//#include <mgl2/fltk.h>

//Also ensure the library is included along with the header
#include <cuda_module.h>


//Python things
#ifndef _DEBUG
#include "pyheader.h"
#include <numpy/arrayobject.h>

//Global Python objects for outputting plots
PyObject *pFunc, *pArg;
PyArrayObject* pArray;
PyObject *pTup = PyTuple_New(4);
#endif


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
//mglFLTK *globalr = NULL;
const int num_disp_pts = 2048;

//Global for logfile
std::ofstream logfile;

//mutexes for updating display
boost::mutex display_mtx, stream_mtx, fft_mtx;
boost::condition_variable_any disp_cond;//, fft_cond, stream_cond;

//mutexes and cond vars for FFT threads and receiver
boost::mutex fft_mutex[NUM_THREADS], recv_mutex, super_mutex;
boost::condition_variable_any fft_cond, recv_cond, super_cond;


const int MIN5 = 5722; //int(300,000/52.4288) ms (@131072 - no averaging)
//const int MIN5 = 366208; //@2048 ...

//constructor definition
window::window(int t, int bw, int f, int ws, int n) {
	timescale = t;
	bandwidth = bw;
	frequency = f;
	whitespace = ws; //could be long long, will need to test experimentally, u_int should be enough
	frame_no = n;
}

//destructor
window::~window() {};

//Returns the filepath of the beginning of the selected dataset (.dat file)
void select_file(char* file ) {

	//char select_file[MAX_PATH] = "";

	OPENFILENAMEA openfile = { 0 };  //http://www.cplusplus.com/forum/windows/82432/ Thanks Disch!

	openfile.lStructSize = sizeof(openfile);
	openfile.lpstrFilter = "data file\0*.dat\0config file\0*.cfg\0";
	openfile.nFilterIndex = 1; //because all indexes start at 1 :D
	openfile.Flags = OFN_FILEMUSTEXIST;  //only allow the user to open files that actually exist

	// the most important bits:
	openfile.lpstrFile = file;
	openfile.nMaxFile = MAX_PATH;  // size of our 'buffer' buffer

	std::cout << "\nSelect .dat sample file, ensure valid .cfg file is in the same directory" << std::endl;
	//Call the open file dialog
	if (!GetOpenFileNameA(&openfile))
	{
		//No file selected
		std::cout << "No file selected!\n\n" << "Terminating";
		exit(0);
	}
	else
	{
		//File selected
		std::cout << "Loading file: " << file << std::endl;

#ifndef _DEBUG
		std::cout << "Initialising Python Extension" << std::endl;
		init_py_plot();

#endif
//		return file;
	}
}

#ifndef _DEBUG

int python_test() { //This will be deprecated in due time
	//Test pythoney things n.n
	PyThreadState *_save;
	_save = PyEval_SaveThread();

	PyGILState_STATE gstate = PyGILState_Ensure();
	std::cout << "Python\n";

	char* def = "return_fig";
	char* pyfile = "plot_helper";
	PyObject* pFunction = init_py_function(pyfile, def);
	//PyObject* pShow = init_py_function(pyfile, (char*)"display_fig");
	PyObject* pShow = init_py_function(pyfile, (char*)"update_plt");
	PyObject* pUpdate = init_py_function(pyfile, (char*)"display_fig");

	PyObject* pFig;

	if (PyCallable_Check(pFunction)) {


		pFig = PyObject_CallObject(pFunction, NULL);

		//PyObject_CallObject(pFunction, NULL);



		//Just incase ;)
		PyErr_Print();
	}

	PyObject *pUpdateArgs = PyTuple_New(1);
	PyTuple_SET_ITEM(pUpdateArgs, 0, pFig);
	PyObject_CallObject(pUpdate, pUpdateArgs);


	if (PyCallable_Check(pShow)) {

		std::cout << "Call plot" << std::endl;
		PyObject *pShowArgs = PyTuple_New(1);
		PyTuple_SET_ITEM(pShowArgs, 0, pFig);
		//PyObject_CallObject(pShow, NULL);
		PyObject_CallObject(pShow, pShowArgs);

		//Just incase ;)
		PyErr_Print();

	}
	for (int i = 0; i < 5000; i++) std::cout << i << "\r";
	PyGILState_Release(gstate);
	PyEval_RestoreThread(_save);

	return 0;
}

int main(int argc, char*argv[]) {
	//Very exciting main

	char dataset[MAX_PATH] = "";
	
	select_file(dataset);

	//read_samples_plot(dataset); //BW = 1
	read_samples_outwins(dataset); // Greedy allocation
	return(0);

	//Program should never get past here (with GPU), preserving for just incasesies ;)
	//Probably should look into some kind of realtime plotting display for use with GPU ... one day... WE HAVE IT! Qt to the rescue do do-do do! 
}

/*
	//Logfile
	if(LOG) logfile.open("logfile.log",std::ofstream::out);

	//Setup Processing
	//Configuration
	std::string base = select_file;
	std::string path = base.substr(0, base.find_last_of("/\\") + 1);
	std::string token = base.substr(base.find_last_of("/\\") + 1);

	std::string filenum_s = token.substr(token.find_last_of("_") + 1, (token.find(".dat") - token.find_last_of("_") - 1)); //extract the number of the sample file
	int filenum_i = 0;

	struct _stat64 stat;
	std::string search;
	double stat_size = 0;

	std::cout << "Only sequential sample files will be included" << std::endl;
	for (int i = stoi(filenum_s); i < 100; i++) {
	search = base.substr(0, base.find_last_of("_") + 1) + boost::lexical_cast<std::string>(i) + ".dat";

	if (_stat64(search.c_str(), &stat) != -1) {
	std::cout << "File: " << i << " included" << std::endl;
	stat_size += stat.st_size;
	_stat64(search.c_str(), &stat);
	}
	else {
	std::cout << std::endl << i << " sample files discovered\n\n" << "Cumulative filesize: " << stat_size / (1 << 20) << " MB\n\n";
	filenum_i = i;
	break;
	}
	}
	std::cout << token << std::endl;
	std::cout << path << std::endl;
	std::cout << filenum_s << std::endl;
	std::string cfgfile = path + "sample_config.cfg";
	std::cout << cfgfile << std::endl;
	double three_gb = std::pow(2.0, 31) + std::pow(2.0, 30);

	int remain = (((int64_t)stat_size / (4 * WIN_SAMPS)) % 10);
	std::cout << "Total number of frames: " << (stat_size / (4 * WIN_SAMPS*10)) << " Frames dropped: " << remain << std::endl;
	std::cout << "Frames per 3GB: " << int( three_gb / (4 * WIN_SAMPS * 10));
	//return 0;

	std::ifstream config (cfgfile, std::ifstream::in);
	if(config.fail()) {
	std::cout << "\nPlease ensure that cfg file is in the same directory as the samples!\n\nThis program will now terminate" << std::endl;
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

	//Samples Setup
	std::ifstream in_samples;
	std::string sampfile = select_file;
	in_samples.open(sampfile, std::ifstream::binary);
	in_samples.seekg(0, std::ifstream::beg);

	struct _stat64 stat_buff;
	_stat64(sampfile.c_str(), &stat_buff);
	double file_size = stat_buff.st_size;

	std::cout << "Filesize: " << file_size/std::pow(2.0,20) << " MB" << std::endl;

	// **********HEREBEDRAGONS**********
	bool HYBRID = true;

	if (HYBRID){

	LARGE_INTEGER ftpl, wind, perffreq;
	double wf, pf;
	QueryPerformanceFrequency(&perffreq);
	pf = perffreq.QuadPart;
	QueryPerformanceCounter(&ftpl);

	std::complex<short>* re;//Buffer for read in samples
	re = (std::complex<short>*) malloc(file_size/2);

	float* processed_ptr;
	int averaging = 10;
	int num_wins = file_size / (4 * WIN_SAMPS*2); //Total number of bytes, divided by the size of a complex double (4) times the number of samples for a window (131072), times 2 (to only take half of the sample file)
	//std::cout << "NUMBER OF FRAMES: " << file_size / (4 * WIN_SAMPS * 2) << std::endl;


	//size_t read_in = WIN_SAMPS*averaging*

	//		if (num_wins % averaging != 0) std::cout << "Last " << num_wins%averaging << " windows will be dropped!" << std::endl;

	int frame_number = 0;
	bool init = true;

	int ws_array[WIN_SAMPS];
	unsigned char ws_frame[WIN_SAMPS];
	int overlap[WIN_SAMPS];

	for (int i = 0; i < WIN_SAMPS; i++){
	ws_array[i] = 0;
	ws_frame[i] = 0;
	overlap[i] = 0;
	}

	//Window vector for outputting windows
	std::vector<window> window_vec;
	std::vector<window>* w_vec_ptr = &window_vec;
	size_t bytesread = 0;

	int f = 0;

	while (true){

	in_samples.read((char*)re, file_size / 3);
	bytesread = in_samples.gcount();
	std::cout << bytesread << " Bytes read\n";

	if (in_samples.eof()) std::cout << "WE AT DA END\n";
	if (bytesread) {
	std::cout << "File part "<< ++f << " Loaded" << std::endl;
	}
	else {
	std::cout << "End of File" << std::endl;
	break;
	}

	//processed_ptr = (float*)malloc(sizeof(float) * WIN_SAMPS * num_wins / averaging);
	processed_ptr = dothething(re, averaging, processed_ptr, num_wins);
	std::cout << "Round " << f << " complete!" << std::endl;

	//Perform the detection! :D
	if (init) {
	for (int i = 0; i < WIN_SAMPS; i++) {
	ws_array[i] = processed_ptr[i]; //This is done because the first entry in ws_array has to be initialised with the first frame
	}
	init = false;
	processed_ptr += WIN_SAMPS;
	frame_number++;

	detect(processed_ptr, (num_wins / averaging) - 1, ws_array, overlap, w_vec_ptr, &frame_number);
	}
	else {
	frame_number++;
	detect(processed_ptr, (num_wins / averaging), ws_array, overlap, w_vec_ptr, &frame_number);
	}
	}

	std::cout << "Finishing up" << std::endl;
	//Close the samples off
	frame_number++;
	detect_once(ws_frame, 1, ws_array, overlap, w_vec_ptr, frame_number);

	//This is here to provide the final window detection and outputting of the window vector to file.
	std::ofstream window_dump;
	window_dump.open("Window_dump.csv");
	window_dump << "timescale,frequency,bandwidth,whitespace,frame_no\n";
	std::vector<window>::iterator WINit; //Window iterator
	std::cout << "Outputting Whitespace Windows" << std::endl;
	for (WINit = window_vec.begin(); WINit < window_vec.end(); WINit++) {
	window_dump << WINit->timescale << "," << WINit->frequency << "," << WINit->bandwidth << "," << WINit->whitespace << "," << WINit->frame_no << "\n";
	}
	std::cout << "Output Complete" << std::endl;
	window_dump.flush();
	window_dump.close();

	//bar_plot(&window_vec); //If the end of the data has been reached, then display the data
	exit(0);
	}


	// **********End dragons n.n**********

	std::complex<short>* re;//Buffer for read in samples
	re = (std::complex<short>*) malloc(file_size);

	if (in_samples.read((char*)re, file_size)) {
	std::cout << "File Loaded" << std::endl;
	}
	else {
	std::cout << "File Failed to Load" << std::endl;
	system("Pause");
	}

	//DISPLAY setup
	mglData *mglY = new mglData(num_disp_pts); //mglData object for plotting

	if (DISPLAY)	{
	boost::thread spawnWindow(init);	//Thread runs 5ever
	boost::this_thread::sleep(boost::posix_time::milliseconds(500));
	boost::thread display_thread(boost::bind(update, boost::ref(globalr), boost::ref(mglY)));
	}

	//This is where the magic happens on the CPU routine (if GPU not installed)
	fftf_proc_s(re, 0, mglY, file_size);

	std::cout << "We done lah" << std::endl;

	std::cout << "SuperV: Cleaning up" << std::endl;
	//TODO: Cleanup
	if (in_samples.is_open()){
	in_samples.close();
	}
	if (LOG && logfile.is_open()){
	in_samples.close();
	}
	if (config.is_open()){
	in_samples.close();
	}
	free(re);
	system("pause");
	}
	*/

void create_folders(){

	//Might want to change where these are created?
	char path0[] = "\.\\5min";
	CreateDirectory(path0, NULL);
	/*char path1[] = "\.\\10min";
	CreateDirectory(path1, NULL);
	char path2[] = "\.\\15min";
	CreateDirectory(path2, NULL);
	char path3[] = "\.\\30min";
	CreateDirectory(path3, NULL);
	char path4[] = "\.\\1hr";
	CreateDirectory(path4, NULL);
	char path5[] = "\.\\24hr";
	CreateDirectory(path5, NULL);*/

	char path01[] = "\.\\5min\\log";
	CreateDirectory(path01, NULL);
	char path02[] = "\.\\5min\\lin";
	CreateDirectory(path02, NULL);
}

void call_py_plot(double *inArr, int scale, int id, double total) {

	if (PyCallable_Check(pFunc)) {

		const int arry_H = 1;
		//npy_intp arry_W[arry_H] = { MIN5*10 }; //The way to read this is: array[array_height][array_width]

		npy_intp arry_W[arry_H] = { WIN_SAMPS/2 - WIN_SAMPS/10}; //The way to read this is: array[array_height][array_width]

		//THIS NEEDS TO BE MODIFIED to adapt to the longest timescale that is detected. 
		
		//std::cout << "Cast array\n";
		pArg = PyArray_SimpleNewFromData(arry_H, arry_W, NPY_DOUBLE, reinterpret_cast<void*>(inArr));
		pArray = reinterpret_cast<PyArrayObject*>(pArg);
		//std::cout << "Call Python\n";

		PyTuple_SET_ITEM(pTup, 0, pArg);
		PyTuple_SET_ITEM(pTup, 1, PyLong_FromLong(scale));
		PyTuple_SET_ITEM(pTup, 2, PyLong_FromLong(id));
		PyTuple_SET_ITEM(pTup, 3, PyLong_FromDouble(total));

		PyGILState_STATE gstate = PyGILState_Ensure();

		PyObject_CallObject(pFunc, pTup);

		//Just incase ;)
		PyErr_Print();
	}
	else {
		std::cout << "Plotting function unavailable\n\nTerminating\n";
		PyErr_Print();
		system("pause");
		exit(0);
	}
}

int init_py_plot() {

	_putenv_s("PYTHONPATH", ".");
	Py_Initialize();
	import_array();

	PyObject *pName, *pModule, *pDict;

	char* py_file = "plot_helper";

	create_folders();

	pName = PyUnicode_FromString(py_file);
	pModule = PyImport_Import(pName);
	if (pModule) {
		std::cout << "Module Loaded\n";
		pDict = PyModule_GetDict(pModule);
		pFunc = PyObject_GetAttrString(pModule, (char*)"plot_array");

	}
	else {
		PyErr_Print();
	}

	Py_DECREF(pModule);
	Py_DECREF(pName);

	//Test Routine
	/*
	double x[131072] = { 50, 50, 403, 20, 1 };
	call_py_plot(x,5,0,297);
	x[5] = 10000;
	call_py_plot(x, 5, 1, 497);

	exit(0);
	*/
	return 0; 
}

PyObject* init_py_function(char* py_file, char* fName) {

	PyObject *pName, *pModule, *pFn;

	create_folders();

	pName = PyUnicode_FromString(py_file);
	pModule = PyImport_Import(pName);
	if (pModule) {
		std::cout << "Module " << fName <<  " Loaded\n";
		pFn = PyObject_GetAttrString(pModule, fName);

	}
	else {
		PyErr_Print();
	}

	Py_DECREF(pModule);
	Py_DECREF(pName);

	return pFn;
}

#endif

void read_samples(char* sampbase){

	//Configuration
	std::string base = sampbase;
	std::string path = base.substr(0, base.find_last_of("/\\") + 1);
	std::string token = base.substr(base.find_last_of("/\\") + 1);

	std::string cfgfile = path + "sample_config.cfg";
	std::cout << cfgfile << std::endl;

	std::ifstream config(cfgfile, std::ifstream::in);
	if (config.fail()) {
		std::cout << "\nPlease ensure that cfg file is in the same directory as the samples!\n\nThis program will now terminate" << std::endl;
		exit(0);
	}

	//Read in configuration File and set appropriate program parameters.
	// IMPORTANT: configuration file must have the name "sample_config.cfg" and be in the same directory as the samples
	std::string line;
	std::istringstream sin;
	int i = 0;

	if (config.is_open()){
		while (std::getline(config, line)) {
			sin.clear();
			sin.str(line);
			if (i == 0) sin >> filename; //Currently stops at a space - its an istringstream problem 
			else if (i == 1) sin >> device;
			else if (i == 2) sin >> centre_freq;
			else if (i == 3) sin >> bandwidth;
			else if (i == 4) sin >> samp_rate;
			else if (i == 5) sin >> rx_gain;
			else if (i > 5) {
				std::cout << "Too many arguments in configuration ... terminating" << std::endl;
				exit(0);
			}
			std::cout << line << std::endl;
			i++;
		}
	}

	//Sample Files
	std::string filenum_s = token.substr(token.find_last_of("_") + 1, (token.find(".dat") - token.find_last_of("_") - 1)); //extract the number of the sample file
	int filenum_i = stoi(filenum_s);

	struct _stat64 stat;
	std::string search;
	int64_t stat_size = 0;

	std::cout << "Only sequential sample files will be included" << std::endl;
	while (true) {
		search = base.substr(0, base.find_last_of("_") + 1) + boost::lexical_cast<std::string>(filenum_i)+".dat";

		if (_stat64(search.c_str(), &stat) != -1) {
			std::cout << "File: " << filenum_i++ << " included" << std::endl;
			stat_size += stat.st_size;
			_stat64(search.c_str(), &stat);

		}
		else {
			std::cout << std::endl << filenum_i << " sample file(s) discovered\n\n" << "Cumulative filesize: " << stat_size / (1 << 20) << " MB\n\n";
			break;
		}
	}
	std::cout << token << std::endl;
	std::cout << path << std::endl;
	std::cout << filenum_s << std::endl;
	
	int64_t three_gb = std::pow(2.0, 31) + std::pow(2.0, 30);
	int64_t two_gb = std::pow(2.0, 31);
	int remain = (((int64_t)stat_size / (4 * WIN_SAMPS)) % 10);
	std::cout << "Total number of frames: " << (stat_size / (4 * WIN_SAMPS * 10)) << " Windows dropped: " << remain << std::endl;
	//std::cout << "Frames per 3GB: " << int(three_gb / (4 * WIN_SAMPS * 10));
	//return 0;

	//Timing variables
	/*
	LARGE_INTEGER ftpl, wind, perffreq;
	double wf, pf;
	QueryPerformanceFrequency(&perffreq);
	pf = perffreq.QuadPart;
	QueryPerformanceCounter(&ftpl);
	*/

	//Samples Setup 
	std::ifstream in_samples;
	//std::string sampfile = select_file;
	std::string sampfile = sampbase;
	in_samples.open(sampfile, std::ifstream::binary);
	in_samples.seekg(0, std::ifstream::beg);

	struct _stat64 stat_buff;
	_stat64(sampfile.c_str(), &stat_buff);
	double file_size = stat_buff.st_size;

	float* processed_ptr;
	int64_t averaging = 10;
	int64_t num_wins = file_size / (4 * WIN_SAMPS * 2); //Total number of bytes, divided by the size of a complex double (4) times the number of samples for a window (131072), times 2 (to only take half of the sample file)
	//std::cout << "NUMBER OF FRAMES: " << file_size / (4 * WIN_SAMPS * 2) << std::endl;

	//		if (num_wins % averaging != 0) std::cout << "Last " << num_wins%averaging << " windows will be dropped!" << std::endl;

	int frame_number = 0;
	bool init = true;

	int ws_array[WIN_SAMPS];
	unsigned char zero_frame[WIN_SAMPS];
	int overlap[WIN_SAMPS];

	for (int i = 0; i < WIN_SAMPS; i++){
		ws_array[i] = 0;
		zero_frame[i] = 0;
		overlap[i] = 0;
	}

	//Window vector for outputting windows
	std::vector<window> window_vec;
	std::vector<window>* w_vec_ptr = &window_vec;
	size_t bytesread = 1;
	size_t bytesleft = 0;
	size_t samp_overlap = 8;

	int currfile = stoi(filenum_s);
	int64_t bytes_to_read = (two_gb / (sizeof(std::complex<short>) * WIN_SAMPS * averaging)) *(WIN_SAMPS * sizeof(std::complex<short>)  * averaging);
//	int64_t bytes_to_read = ((two_gb / (4 * WIN_SAMPS * averaging * samp_overlap)) *(WIN_SAMPS * 4 * averaging));

	std::complex<short>* re;//Buffer for read in samples
	re = (std::complex<short>*) malloc(bytes_to_read);
	int currwins = 0;
	

	//std::cout << std::endl <<  std::endl << bytes_to_read << " Read this many samps please\n\n";
	//This is where all the processing happens
	while (bytesread != 0){

		in_samples.read((char*)re, bytes_to_read);
		bytesread = in_samples.gcount();
		std::cout << std::endl<< bytesread << " Bytes read. ";
		bytesleft = bytes_to_read - bytesread;

		if (bytesleft > 0 && currfile < filenum_i) {

			if (in_samples.is_open()) in_samples.close();
			sampfile = base.substr(0, base.find_last_of("_") + 1) + boost::lexical_cast<std::string>(++currfile)+".dat";
			in_samples.open(sampfile, std::ifstream::binary);
			if (in_samples.is_open()) {
				in_samples.seekg(0, std::ifstream::beg);
				in_samples.read((char*)re, bytesleft);

				std::cout << in_samples.gcount() << " additional Bytes read. " << in_samples.gcount() + bytesread << " total Bytes read.";
				bytesread += in_samples.gcount();
			}
		}

		currwins = (bytesread / (4 * WIN_SAMPS * averaging)) * averaging;
		std::cout << "Windows: " << currwins << std::endl;

		if (bytesread) {
			std::cout << "\nFile part " << currfile << " of " << filenum_i << " Loaded" << std::endl;

			processed_ptr = (float*)realloc(processed_ptr, sizeof(float) * WIN_SAMPS * currwins);
			perform_fft(re, processed_ptr, WIN_SAMPS, averaging, currwins);
			//processed_ptr = dothething(re, averaging, currwins);
			//processed_ptr = dothething_overlap(re, averaging, processed_ptr, currwins, samp_overlap);

			//Perform the detection! :D
			if (init) {
				for (int i = 0; i < WIN_SAMPS; i++) {
					ws_array[i] = processed_ptr[i]; //This is done because the first entry in ws_array has to be initialised with the first frame
				}
				init = false;
				processed_ptr += WIN_SAMPS;
				frame_number++;

				//detect(processed_ptr, (currwins / averaging) - 1, ws_array, overlap, w_vec_ptr, &frame_number);
				detect_ts(processed_ptr, currwins - 1, ws_array, w_vec_ptr, &frame_number);
				//detect(processed_ptr, ((currwins*samp_overlap) / averaging) - 1, ws_array, overlap, w_vec_ptr, &frame_number);
			}
			else {
				frame_number++;
				//detect(processed_ptr, (currwins / averaging), ws_array, overlap, w_vec_ptr, &frame_number);
				detect_ts(processed_ptr, currwins, ws_array, w_vec_ptr, &frame_number);
				//detect(processed_ptr, ((currwins*samp_overlap) / averaging), ws_array, overlap, w_vec_ptr, &frame_number);
			}
		}
	}

	std::cout << "Finishing up" << std::endl;
	//Close the samples off
	frame_number++;
	//detect_once(zero_frame, 1, ws_array, overlap, w_vec_ptr, frame_number);
	detect_ts_once(zero_frame, 1, ws_array, w_vec_ptr, frame_number);

	//This is here to provide the final window detection and outputting of the window vector to file.
	std::ofstream window_dump;
	window_dump.open("Window_dump.csv");
	window_dump << "timescale,frequency,bandwidth,whitespace,frame_no\n";
	std::vector<window>::iterator WINit; //Window iterator
	std::cout << "Outputting Whitespace Windows" << std::endl;
	for (WINit = window_vec.begin(); WINit < window_vec.end(); WINit++) {
		window_dump << WINit->timescale << "," << WINit->frequency << "," << WINit->bandwidth << "," << WINit->whitespace << "," << WINit->frame_no << "\n";
	}
	std::cout << "Output Complete" << std::endl;
	window_dump.flush();
	window_dump.close();


	//TODO: Cleanup
	if (in_samples.is_open()){
		in_samples.close();
	}
	if (config.is_open()){
		in_samples.close();
	}
	free(re);

	//bar_plot(&window_vec); //If the end of the data has been reached, then display the data
}

void read_samples_plot(char* sampbase){

	//Configuration
	std::string base = sampbase;
	std::string path = base.substr(0, base.find_last_of("/\\") + 1);
	std::string token = base.substr(base.find_last_of("/\\") + 1);

	std::string cfgfile = path + "sample_config.cfg";
	std::cout << cfgfile << std::endl;

	std::ifstream config(cfgfile, std::ifstream::in);
	if (config.fail()) {
		std::cout << "\nPlease ensure that cfg file is in the same directory as the samples!\n\nThis program will now terminate" << std::endl;
		exit(0);
	}

	//Read in configuration File and set appropriate program parameters.
	// IMPORTANT: configuration file must have the name "sample_config.cfg" and be in the same directory as the samples
	std::string line;
	std::istringstream sin;
	int i = 0;

	if (config.is_open()){
		while (std::getline(config, line)) {
			sin.clear();
			sin.str(line);
			if (i == 0) sin >> filename; //Currently stops at a space - its an istringstream problem 
			else if (i == 1) sin >> device;
			else if (i == 2) sin >> centre_freq;
			else if (i == 3) sin >> bandwidth;
			else if (i == 4) sin >> samp_rate;
			else if (i == 5) sin >> rx_gain;
			else if (i > 5) {
				std::cout << "Too many arguments in configuration ... terminating" << std::endl;
				exit(0);
			}
			std::cout << line << std::endl;
			i++;
		}
	}

	//Sample Files
	std::string filenum_s = token.substr(token.find_last_of("_") + 1, (token.find(".dat") - token.find_last_of("_") - 1)); //extract the number of the sample file
	int filenum_i = stoi(filenum_s);

	struct _stat64 stat;
	std::string search;
	int64_t stat_size = 0;

	std::cout << "Only sequential sample files will be included" << std::endl;
	while (true) {
		search = base.substr(0, base.find_last_of("_") + 1) + boost::lexical_cast<std::string>(filenum_i)+".dat";

		if (_stat64(search.c_str(), &stat) != -1) {
			std::cout << "File: " << filenum_i++ << " included" << std::endl;
			stat_size += stat.st_size;
			_stat64(search.c_str(), &stat);

		}
		else {
			std::cout << std::endl << filenum_i << " sample file(s) discovered\n\n" << "Cumulative filesize: " << stat_size / (1 << 20) << " MB\n\n";
			break;
		}
	}
	std::cout << token << std::endl;
	std::cout << path << std::endl;
	std::cout << filenum_s << std::endl;

	int64_t three_gb = std::pow(2.0, 31) + std::pow(2.0, 30);
	int64_t two_gb = std::pow(2.0, 31);

	//FIXME

	int remain = (((int64_t)stat_size / (4 * WIN_SAMPS)) % 10);
	
	std::cout << "Total number of frames: " << (stat_size / (4 * WIN_SAMPS)) << " Windows dropped: " << remain << std::endl;
	
	//std::cout << "Frames per 3GB: " << int(three_gb / (4 * WIN_SAMPS * 10));
	//return 0;

	//Timing variables
	/*
	LARGE_INTEGER ftpl, wind, perffreq;
	double wf, pf;
	QueryPerformanceFrequency(&perffreq);
	pf = perffreq.QuadPart;
	QueryPerformanceCounter(&ftpl);
	*/

	//Samples Setup 
	std::ifstream in_samples;
	//std::string sampfile = select_file;
	std::string sampfile = sampbase;
	in_samples.open(sampfile, std::ifstream::binary);
	in_samples.seekg(0, std::ifstream::beg);

	struct _stat64 stat_buff;
	_stat64(sampfile.c_str(), &stat_buff);
	double file_size = stat_buff.st_size;

	float* processed_ptr_base = NULL;
	float* processed_ptr = NULL;
	int64_t averaging = 10;
	int64_t num_wins = file_size / (4 * WIN_SAMPS * 2); //Total number of bytes, divided by the size of a complex double (4) times the number of samples for a window (131072), times 2 (to only take half of the sample file)
	//std::cout << "NUMBER OF FRAMES: " << file_size / (4 * WIN_SAMPS * 2) << std::endl;

	//		if (num_wins % averaging != 0) std::cout << "Last " << num_wins%averaging << " windows will be dropped!" << std::endl;

	int frame_number = 0;
	bool init = true;

	int ws_array[WIN_SAMPS];
	unsigned char zero_frame[WIN_SAMPS];
	int overlap[WIN_SAMPS];

	for (int i = 0; i < WIN_SAMPS; i++){
		ws_array[i] = 0;
		zero_frame[i] = 0;
		overlap[i] = 0;
	}

	//Window vector for outputting windows
	std::vector<window> window_vec;
	std::vector<window>* w_vec_ptr = &window_vec;
	size_t bytesread = 1;
	size_t bytesleft = 0;
	size_t samp_overlap = 8;

	int currfile = stoi(filenum_s);

	//FIX ME?
	//int64_t bytes_to_read = (two_gb / (sizeof(std::complex<short>) * WIN_SAMPS * averaging)) *(WIN_SAMPS * sizeof(std::complex<short>)  * averaging);
	
	int64_t bytes_to_read = 2 * std::pow(2, 30) + (averaging - 1) * 4 * WIN_SAMPS; //4GB plus 9 frames
	
	//	int64_t bytes_to_read = ((two_gb / (4 * WIN_SAMPS * averaging * samp_overlap)) *(WIN_SAMPS * 4 * averaging));

	std::complex<short>* re;//Buffer for read in samples
	re = (std::complex<short>*) malloc(bytes_to_read);
	int currwins = 0;

	double ts_array[MIN5*10] = { 0 }; //intialise array ton 0, BEWARE!!!! index 0 is actaully Timescale 1. (for the purposes of plotting n.n)
	double rec_array[(WIN_SAMPS / 2) - (WIN_SAMPS / 10)] = { 0 }; //initialise a clipped recording array at half of the total spanned spectrum (it is impossible to get more than half of the spectrum as bandwidth, due to the power on the centre frequency from the LO)
	double total_ws = 0;
	int plot_id = 0;
	const int SCALE = 5;
	int frames_remain = 0;

	double bw_ws_count = 0;

	std::ofstream window_dump;
	std::string csv_filename;
	std::vector<window>::iterator WINit; //Window iterator

	//std::cout << std::endl <<  std::endl << bytes_to_read << " Read this many samps please\n\n";
	//This is where all the processing happens
	//This will generate unique plots for every 5722 frames (MIN5) (set at OLD 10 AVERAGING ... i.e. frame TS = 52.4288ms) ... this number would be significantly higher once that is fixed
	while (bytesread != 0){

		in_samples.read((char*)re, bytes_to_read);
		bytesread = in_samples.gcount();
		std::cout << std::endl << bytesread << " Bytes read. ";
		bytesleft = bytes_to_read - bytesread;

		if (bytesleft > 0 && currfile < filenum_i) {

			if (in_samples.is_open()) in_samples.close();
			sampfile = base.substr(0, base.find_last_of("_") + 1) + boost::lexical_cast<std::string>(++currfile) + ".dat";
			in_samples.open(sampfile, std::ifstream::binary);
			if (in_samples.is_open()) {
				in_samples.seekg(0, std::ifstream::beg);
				in_samples.read((char*)re, bytesleft);

				std::cout << in_samples.gcount() << " additional Bytes read. " << in_samples.gcount() + bytesread << " total Bytes read.";
				bytesread += in_samples.gcount();

				if (bytesleft >= WIN_SAMPS*(averaging - 1)*sizeof(std::complex<short>)) in_samples.seekg(bytesleft - WIN_SAMPS*(averaging - 1)*sizeof(std::complex<short>));
				else if (bytesleft - WIN_SAMPS*(averaging - 1)*sizeof(std::complex<short>) < 0) { //edge case if average buffer spans 2 files
					in_samples.close();
					std::cout << "reopening previous file as required by average buffer" << std::endl;
					sampfile = base.substr(0, base.find_last_of("_") + 1) + boost::lexical_cast<std::string>(--currfile) + ".dat";
					in_samples.open(sampfile, std::ifstream::binary);
					in_samples.seekg(-1 + bytesleft - WIN_SAMPS*(averaging - 1)*sizeof(std::complex<short>), std::ifstream::end);
				}
			}
		}

		currwins = (bytesread / (sizeof(std::complex<short>)  * WIN_SAMPS)) - (averaging - 1);
		std::cout << "Windows: " << currwins << std::endl;

		if (bytesread) {
			std::cout << "\nFile part " << currfile << " of " << filenum_i << " Loaded" << std::endl;

			processed_ptr_base = (float*)realloc(processed_ptr_base, sizeof(float) * WIN_SAMPS * currwins);
			processed_ptr = processed_ptr_base;
			perform_fft(re, processed_ptr, WIN_SAMPS, averaging, currwins);
			//processed_ptr = dothething_overlap(re, averaging, processed_ptr, currwins, samp_overlap);

			//Perform the detection! :D
			if (init) {
				for (int i = 0; i < WIN_SAMPS; i++) {
					ws_array[i] = processed_ptr[i]; //This is done because the first entry in ws_array has to be initialised with the first frame
				}
				init = false;
				processed_ptr += WIN_SAMPS;
				frame_number++;

				//detect(processed_ptr, (currwins / averaging) - 1, ws_array, overlap, w_vec_ptr, &frame_number);
				
				detect_ts_rec(processed_ptr, ts_array, currwins - 1, ws_array, w_vec_ptr, &frame_number);
				//detect_bw_rec(processed_ptr, rec_array, currwins - 1, ws_array, w_vec_ptr, &frame_number, &bw_ws_count);
				
				//detect(processed_ptr, ((currwins*samp_overlap) / averaging) - 1, ws_array, overlap, w_vec_ptr, &frame_number);
			}
			else {
				frame_number++;
				if (frame_number + currwins >= MIN5*averaging) { //SO HACKY

					frames_remain = MIN5*averaging - frame_number;

					//last of the frames for this 5 minute segment
					detect_ts_rec(processed_ptr, ts_array, frames_remain, ws_array, w_vec_ptr, &frame_number);
					//detect_bw_rec(processed_ptr, rec_array, frames_remain, ws_array, w_vec_ptr, &frame_number, &bw_ws_count);

					//cap it off
					frame_number++;
					
					detect_ts_rec_once(zero_frame, ts_array, 1, ws_array, w_vec_ptr, frame_number);
					//detect_bw_rec_once(zero_frame, rec_array, 1, ws_array, w_vec_ptr, frame_number, &bw_ws_count);

					//increment processed_ptr appropriately
					processed_ptr += WIN_SAMPS * frames_remain;

					//need scale, ID, total and double array
					
					
					if (false) {
						for (int i = 0; i < MIN5*averaging; i++) {
							total_ws += ts_array[i];
						}
					}

					for (int i = 0; i < ((WIN_SAMPS / 2) - (WIN_SAMPS / 10)); i++){
						total_ws += (rec_array[i] * (i+1));
						rec_array[i] *= (i + 1);
					}

					//call plot
					//for (int k = 0; k < MIN5; k++) std::cout << ts_array[k] << " ,";
					//std::cout << "\nTotal: " << total_ws << std::endl;

#ifndef _DEBUG
					//call_py_plot(ts_array, SCALE, plot_id, total_ws); //It looks like ts_array is going out of scope when this is being plotted T.T

					if (false) call_py_plot(rec_array, SCALE, plot_id, total_ws); //It looks like ts_array is going out of scope when this is being plotted T.T
#endif
	
					if (true) {
						//This is here to provide the final window detection and outputting of the window vector to file.
						csv_filename = "\.\\partitioning\\window_dump_" + boost::lexical_cast<std::string>(plot_id)+"_BW=1.csv";
						window_dump.open(csv_filename);
						window_dump << "timescale,frequency,bandwidth,whitespace,frame_no\n";
						std::cout << "Outputting Whitespace Windows" << std::endl;
						for (WINit = window_vec.begin(); WINit < window_vec.end(); WINit++) {
							window_dump << WINit->timescale << "," << WINit->frequency << "," << WINit->bandwidth << "," << WINit->whitespace << "," << WINit->frame_no << "\n";
						}
						std::cout << "window dump csv " << plot_id << " saved" << std::endl;
						window_dump.flush();
						window_dump.close();
					}
					else std::cout << "Skipping window output" << std::endl;
					std::cout << "Total Whitespace: " << bw_ws_count << std::endl;
					bw_ws_count = 0;
					window_vec.clear(); //remove previous recorded set of windows

					//reset for next block
					for (int i = 0; i < WIN_SAMPS; i++){
						ws_array[i] = processed_ptr[i]; // <- frame number 0 ;)
					}

					processed_ptr += WIN_SAMPS;
					total_ws = 0;
					plot_id++;
					frame_number = 1;

					//clear previous ts_array
					for (int i = 0; i < MIN5*10; i++) {
						ts_array[i] = 0;
					}

					for (int i = 0; i < ((WIN_SAMPS / 2) - (WIN_SAMPS / 10)); i++){
						rec_array[i] = 0;
					}

					//record the remainder of the windows
					detect_ts_rec(processed_ptr, ts_array, (currwins - frames_remain) - 1, ws_array, w_vec_ptr, &frame_number); //IT IS NOT CURRWINS/AVERAGING
					//detect_bw_rec(processed_ptr, rec_array, (currwins - frames_remain) - 1, ws_array, w_vec_ptr, &frame_number, &bw_ws_count);
				}
				else {
					//detect(processed_ptr, (currwins / averaging), ws_array, overlap, w_vec_ptr, &frame_number);
					detect_ts_rec(processed_ptr, ts_array, currwins, ws_array, w_vec_ptr, &frame_number);
					//detect_bw_rec(processed_ptr, rec_array, currwins, ws_array, w_vec_ptr, &frame_number, &bw_ws_count);
					//detect(processed_ptr, ((currwins*samp_overlap) / averaging), ws_array, overlap, w_vec_ptr, &frame_number);
				}
			}
		}
	}

	std::cout << "Finishing up" << std::endl;
	//Close the samples off
	frame_number++;
	//detect_once(ws_frame, 1, ws_array, overlap, w_vec_ptr, frame_number);
	detect_ts_rec_once(zero_frame, ts_array, 1, ws_array, w_vec_ptr, frame_number);
	//detect_bw_rec_once(zero_frame, rec_array, 1, ws_array, w_vec_ptr, frame_number, &bw_ws_count);

	if (false) {
		for (int i = 0; i < MIN5; i++) {
			total_ws += ts_array[i];
		}
	}

	for (int i = 0; i < ((WIN_SAMPS / 2) - (WIN_SAMPS / 10)); i++){
		total_ws += (rec_array[i] * (i + 1));
		rec_array[i] *= (i + 1);
	}

	//call final plot
#ifndef _DEBUG
	if (false) call_py_plot(rec_array, SCALE, plot_id, total_ws);
#endif
	
	if (true) {

		//This is here to provide the final window detection and outputting of the window vector to file.
		csv_filename = "\.\\partitioning\\window_dump_" + boost::lexical_cast<std::string>(plot_id)+"_BW=1.csv";
		window_dump.open(csv_filename);
		window_dump << "timescale,frequency,bandwidth,whitespace,frame_no\n";
		std::cout << "Outputting final whitespace windows" << std::endl;
		for (WINit = window_vec.begin(); WINit < window_vec.end(); WINit++) {
			window_dump << WINit->timescale << "," << WINit->frequency << "," << WINit->bandwidth << "," << WINit->whitespace << "," << WINit->frame_no << "\n";
		}
		std::cout << "window dump csv " << plot_id << " saved" << std::endl;
		std::cout << "Output Complete" << std::endl;
		window_dump.flush();
		window_dump.close();
	}
	else std::cout << "Skipping final window output" << std::endl;
	std::cout << "Total Whitespace: " << bw_ws_count << std::endl;
	bw_ws_count = 0;
	//TODO: Cleanup
	if (in_samples.is_open()){
		in_samples.close();
	}
	if (config.is_open()){
		in_samples.close();
	}
	free(re);
	free(processed_ptr);

	//bar_plot(&window_vec); //If the end of the data has been reached, then display the data
}

void read_samples_outwins(char* sampbase){

	//Configuration
	std::string base = sampbase;
	std::string path = base.substr(0, base.find_last_of("/\\") + 1);
	std::string token = base.substr(base.find_last_of("/\\") + 1);

	std::string cfgfile = path + "sample_config.cfg";
	std::cout << cfgfile << std::endl;

	std::ifstream config(cfgfile, std::ifstream::in);
	if (config.fail()) {
		std::cout << "\nPlease ensure that cfg file is in the same directory as the samples!\n\nThis program will now terminate" << std::endl;
		exit(0);
	}

	//Read in configuration File and set appropriate program parameters.
	// IMPORTANT: configuration file must have the name "sample_config.cfg" and be in the same directory as the samples
	std::string line;
	std::istringstream sin;
	int i = 0;

	if (config.is_open()){
		while (std::getline(config, line)) {
			sin.clear();
			sin.str(line);
			if (i == 0) sin >> filename; //Currently stops at a space - its an istringstream problem 
			else if (i == 1) sin >> device;
			else if (i == 2) sin >> centre_freq;
			else if (i == 3) sin >> bandwidth;
			else if (i == 4) sin >> samp_rate;
			else if (i == 5) sin >> rx_gain;
			else if (i > 5) {
				std::cout << "Too many arguments in configuration ... terminating" << std::endl;
				exit(0);
			}
			std::cout << line << std::endl;
			i++;
		}
	}

	//Sample Files
	std::string filenum_s = token.substr(token.find_last_of("_") + 1, (token.find(".dat") - token.find_last_of("_") - 1)); //extract the number of the sample file
	int filenum_i = stoi(filenum_s);

	struct _stat64 stat;
	std::string search;
	int64_t stat_size = 0;

	std::cout << "Only sequential sample files will be included" << std::endl;
	while (true) {
		search = base.substr(0, base.find_last_of("_") + 1) + boost::lexical_cast<std::string>(filenum_i)+".dat";

		if (_stat64(search.c_str(), &stat) != -1) {
			std::cout << "File: " << filenum_i++ << " included" << std::endl;
			stat_size += stat.st_size;
			_stat64(search.c_str(), &stat);

		}
		else {
			std::cout << std::endl << filenum_i << " sample file(s) discovered\n\n" << "Cumulative filesize: " << stat_size / (1 << 20) << " MB\n\n";
			break;
		}
	}
	std::cout << token << std::endl;
	std::cout << path << std::endl;
	std::cout << filenum_s << std::endl;

	int64_t three_gb = std::pow(2.0, 31) + std::pow(2.0, 30);
	int64_t two_gb = std::pow(2.0, 31);

	//FIXME

	int remain = (((int64_t)stat_size / (4 * WIN_SAMPS)) % 10);

	std::cout << "Total number of frames: " << (stat_size / (4 * WIN_SAMPS)) << " Windows dropped: " << remain << std::endl;

	//std::cout << "Frames per 3GB: " << int(three_gb / (4 * WIN_SAMPS * 10));
	//return 0;

	//Timing variables
	/*
	LARGE_INTEGER ftpl, wind, perffreq;
	double wf, pf;
	QueryPerformanceFrequency(&perffreq);
	pf = perffreq.QuadPart;
	QueryPerformanceCounter(&ftpl);
	*/

	//Samples Setup 
	std::ifstream in_samples;
	//std::string sampfile = select_file;
	std::string sampfile = sampbase;
	in_samples.open(sampfile, std::ifstream::binary);
	in_samples.seekg(0, std::ifstream::beg);

	struct _stat64 stat_buff;
	_stat64(sampfile.c_str(), &stat_buff);
	double file_size = stat_buff.st_size;

	float* processed_ptr_base = NULL;
	float* processed_ptr = NULL;
	int64_t averaging = 10;
	int64_t num_wins = file_size / (4 * WIN_SAMPS * 2); //Total number of bytes, divided by the size of a complex double (4) times the number of samples for a window (131072), times 2 (to only take half of the sample file)
	//std::cout << "NUMBER OF FRAMES: " << file_size / (4 * WIN_SAMPS * 2) << std::endl;

	//		if (num_wins % averaging != 0) std::cout << "Last " << num_wins%averaging << " windows will be dropped!" << std::endl;

	int frame_number = 0;
	bool init = true;

	int ws_array[WIN_SAMPS][2]; //currently 2, as 0 is the actual WS frame, and 1 is the current window that is spanning that opportunity
	float zero_frame[WIN_SAMPS];
	int overlap[WIN_SAMPS];

	for (int i = 0; i < WIN_SAMPS; i++){
		ws_array[i][0] = 0;
		ws_array[i][1] = 0;
		zero_frame[i] = 0;
		overlap[i] = 0;
	}

	//Window vector for outputting windows
	std::vector<window> window_vec;
	std::vector<window>* w_vec_ptr = &window_vec;
	size_t bytesread = 1;
	size_t bytesleft = 0;
	size_t samp_overlap = 8;

	int currfile = stoi(filenum_s);

	//FIX ME?
	//int64_t bytes_to_read = (two_gb / (sizeof(std::complex<short>) * WIN_SAMPS * averaging)) *(WIN_SAMPS * sizeof(std::complex<short>)  * averaging);

	int64_t bytes_to_read = 2 * std::pow(2, 30) + (averaging - 1) * 4 * WIN_SAMPS; //4GB plus 9 frames

	//	int64_t bytes_to_read = ((two_gb / (4 * WIN_SAMPS * averaging * samp_overlap)) *(WIN_SAMPS * 4 * averaging));

	std::complex<short>* re;//Buffer for read in samples
	re = (std::complex<short>*) malloc(bytes_to_read);
	int currwins = 0;

	double ts_array[MIN5 * 10] = { 0 }; //intialise array ton 0, BEWARE!!!! index 0 is actaully Timescale 1. (for the purposes of plotting n.n)
	double rec_array[(WIN_SAMPS / 2) - (WIN_SAMPS / 10)] = { 0 }; //initialise a clipped recording array at half of the total spanned spectrum (it is impossible to get more than half of the spectrum as bandwidth, due to the power on the centre frequency from the LO)
	double total_ws = 0;
	int plot_id = 0;
	const int SCALE = 5;
	int frames_remain = 0;

	double bw_ws_count = 0;

	std::ofstream window_dump;
	std::string csv_filename;
	std::vector<window>::iterator WINit; //Window iterator

	//std::cout << std::endl <<  std::endl << bytes_to_read << " Read this many samps please\n\n";
	//This is where all the processing happens
	//This will generate unique plots for every 57220 frames (MIN5)
	while (bytesread != 0){

		in_samples.read((char*)re, bytes_to_read);
		bytesread = in_samples.gcount();
		std::cout << std::endl << bytesread << " Bytes read. ";
		bytesleft = bytes_to_read - bytesread;

		if (bytesleft > 0 && currfile < filenum_i) {

			if (in_samples.is_open()) in_samples.close();
			sampfile = base.substr(0, base.find_last_of("_") + 1) + boost::lexical_cast<std::string>(++currfile) + ".dat";
			in_samples.open(sampfile, std::ifstream::binary);
			if (in_samples.is_open()) {
				in_samples.seekg(0, std::ifstream::beg);
				in_samples.read((char*)re, bytesleft);

				std::cout << in_samples.gcount() << " additional Bytes read. " << in_samples.gcount() + bytesread << " total Bytes read.";
				bytesread += in_samples.gcount();

				//Correctly retain samples that are not completely averaged yet.
				if (bytesleft >= WIN_SAMPS*(averaging - 1)*sizeof(std::complex<short>)) in_samples.seekg(bytesleft - WIN_SAMPS*(averaging - 1)*sizeof(std::complex<short>));
				else if (bytesleft - WIN_SAMPS*(averaging - 1)*sizeof(std::complex<short>) < 0) { //edge case if average buffer spans 2 files
					in_samples.close();
					std::cout << "reopening previous file as required by average buffer" << std::endl;
					sampfile = base.substr(0, base.find_last_of("_") + 1) + boost::lexical_cast<std::string>(--currfile) + ".dat";
					in_samples.open(sampfile, std::ifstream::binary);
					in_samples.seekg(-1 + bytesleft - WIN_SAMPS*(averaging - 1)*sizeof(std::complex<short>), std::ifstream::end);
				}
			}
		}

		currwins = (bytesread / (sizeof(std::complex<short>)  * WIN_SAMPS)) - (averaging - 1);
		std::cout << "Windows: " << currwins << std::endl;

		if (bytesread) {
			std::cout << "\nFile part " << currfile << " of " << filenum_i << " Loaded" << std::endl;

			processed_ptr_base = (float*)realloc(processed_ptr_base, sizeof(float) * WIN_SAMPS * currwins);
			processed_ptr = processed_ptr_base;
			perform_fft(re, processed_ptr, WIN_SAMPS, averaging, currwins);
			//processed_ptr = dothething_overlap(re, averaging, processed_ptr, currwins, samp_overlap);

			//Perform the detection! :D
			if (init) {
				for (int i = 0; i < WIN_SAMPS; i++) {
					ws_array[i][0] = processed_ptr[i]; //This is done because the first entry in ws_array has to be initialised with the first frame
				}
				init = false;
				processed_ptr += WIN_SAMPS;
				frame_number++;

				//detect(processed_ptr, (currwins / averaging) - 1, ws_array, overlap, w_vec_ptr, &frame_number);
				//detect_ts_rec(processed_ptr, ts_array, currwins - 1, ws_array, w_vec_ptr, &frame_number);
				//detect_bw_rec(processed_ptr, rec_array, currwins - 1, ws_array, w_vec_ptr, &frame_number, &bw_ws_count);
				detect_greedy(processed_ptr, currwins - 1, ws_array, w_vec_ptr, &frame_number, &bw_ws_count);

				//detect(processed_ptr, ((currwins*samp_overlap) / averaging) - 1, ws_array, overlap, w_vec_ptr, &frame_number);
			}
			else {
				frame_number++;
				if (frame_number + currwins >= MIN5*averaging) { //SO HACKY

					frames_remain = MIN5*averaging - frame_number;

					//last of the frames for this 5 minute segment
					//detect_ts_rec(processed_ptr, ts_array, frames_remain, ws_array, w_vec_ptr, &frame_number);
					//detect_bw_rec(processed_ptr, rec_array, frames_remain, ws_array, w_vec_ptr, &frame_number, &bw_ws_count);
					detect_greedy(processed_ptr, frames_remain, ws_array, w_vec_ptr, &frame_number, &bw_ws_count);

					//cap it off
					frame_number++;

					//detect_ts_rec_once(zero_frame, ts_array, 1, ws_array, w_vec_ptr, frame_number);
					//detect_bw_rec_once(zero_frame, rec_array, 1, ws_array, w_vec_ptr, frame_number, &bw_ws_count);
					detect_greedy(zero_frame, 1, ws_array, w_vec_ptr, &frame_number, &bw_ws_count);

					//increment processed_ptr appropriately
					processed_ptr += WIN_SAMPS * frames_remain;

					//need scale, ID, total and double array


					if (false) {
						for (int i = 0; i < MIN5*averaging; i++) {
							total_ws += ts_array[i];
						}


						for (int i = 0; i < ((WIN_SAMPS / 2) - (WIN_SAMPS / 10)); i++){
							total_ws += (rec_array[i] * (i + 1));
							rec_array[i] *= (i + 1);
						}
					}
					//call plot?
					//for (int k = 0; k < MIN5; k++) std::cout << ts_array[k] << " ,";
					//std::cout << "\nTotal: " << total_ws << std::endl;
					//call_py_plot(ts_array, SCALE, plot_id, total_ws); //It looks like ts_array is going out of scope when this is being plotted T.T

					if (true) {
						//This is here to provide the final window detection and outputting of the window vector to file.
						csv_filename = "\.\\partitioning\\window_dump_" + boost::lexical_cast<std::string>(plot_id)+".csv";
						window_dump.open(csv_filename);
						window_dump << "timescale,frequency,bandwidth,whitespace,frame_no\n";
						std::cout << "Outputting Whitespace Windows" << std::endl;
						for (WINit = window_vec.begin(); WINit < window_vec.end(); WINit++) {
							window_dump << WINit->timescale << "," << WINit->frequency << "," << WINit->bandwidth << "," << WINit->whitespace << "," << WINit->frame_no << "\n";
						}
						window_dump << "\n\n\n" << "Total Whitespace: " << bw_ws_count << "\n";
						std::cout << "window dump csv " << plot_id << " saved" << std::endl;
						window_dump.flush();
						window_dump.close();
					}
					else std::cout << "Skipping window output" << std::endl;
					std::cout << "Total Whitespace: " << bw_ws_count << std::endl;
					bw_ws_count = 0;
					window_vec.clear(); //remove previous recorded set of windows

					//reset for next block
					for (int i = 0; i < WIN_SAMPS; i++){
						ws_array[i][0] = processed_ptr[i]; // <- frame number 0 ;)
					}

					processed_ptr += WIN_SAMPS;
					total_ws = 0;
					plot_id++;
					frame_number = 1;

					//clear previous ts_array
					for (int i = 0; i < MIN5 * 10; i++) {
						ts_array[i] = 0;
					}

					for (int i = 0; i < ((WIN_SAMPS / 2) - (WIN_SAMPS / 10)); i++){
						rec_array[i] = 0;
					}

					//record the remainder of the windows
					//detect_ts_rec(processed_ptr, ts_array, (currwins - frames_remain) - 1, ws_array, w_vec_ptr, &frame_number); //IT IS NOT CURRWINS/AVERAGING
					//detect_bw_rec(processed_ptr, rec_array, (currwins - frames_remain) - 1, ws_array, w_vec_ptr, &frame_number, &bw_ws_count);
					detect_greedy(processed_ptr, (currwins - frames_remain) - 1, ws_array, w_vec_ptr, &frame_number, &bw_ws_count);
				}
				else {
					//detect(processed_ptr, (currwins / averaging), ws_array, overlap, w_vec_ptr, &frame_number);
					//detect_ts_rec(processed_ptr, ts_array, currwins, ws_array, w_vec_ptr, &frame_number);
					//detect_bw_rec(processed_ptr, rec_array, currwins, ws_array, w_vec_ptr, &frame_number, &bw_ws_count);
					detect_greedy(processed_ptr, currwins, ws_array, w_vec_ptr, &frame_number, &bw_ws_count);
					//detect(processed_ptr, ((currwins*samp_overlap) / averaging), ws_array, overlap, w_vec_ptr, &frame_number);
				}
			}
		}
	}

	std::cout << "Finishing up" << std::endl;
	//Close the samples off
	frame_number++;
	//detect_once(ws_frame, 1, ws_array, overlap, w_vec_ptr, frame_number);
	//detect_ts_rec_once(zero_frame, ts_array, 1, ws_array, w_vec_ptr, frame_number);
	//detect_bw_rec_once(zero_frame, rec_array, 1, ws_array, w_vec_ptr, frame_number, &bw_ws_count);
	detect_greedy(zero_frame, 1, ws_array, w_vec_ptr, &frame_number, &bw_ws_count);

	if (false) {
		for (int i = 0; i < MIN5; i++) {
			total_ws += ts_array[i];
		}
	}

	for (int i = 0; i < ((WIN_SAMPS / 2) - (WIN_SAMPS / 10)); i++){
		total_ws += (rec_array[i] * (i + 1));
		rec_array[i] *= (i + 1);
	}

	//call final plot
//#ifndef _DEBUG
//	call_py_plot(rec_array, SCALE, plot_id, total_ws);
//#endif

	if (true) {

		//This is here to provide the final window detection and outputting of the window vector to file.
		csv_filename = "\.\\partitioning\\window_dump_" + boost::lexical_cast<std::string>(plot_id)+".csv";
		window_dump.open(csv_filename);
		window_dump << "timescale,frequency,bandwidth,whitespace,frame_no\n";
		std::cout << "Outputting final whitespace windows" << std::endl;
		for (WINit = window_vec.begin(); WINit < window_vec.end(); WINit++) {
			window_dump << WINit->timescale << "," << WINit->frequency << "," << WINit->bandwidth << "," << WINit->whitespace << "," << WINit->frame_no << "\n";
		}
		window_dump << "\n\n\n" << "Total Whitespace: " << bw_ws_count << "\n";
		std::cout << "window dump csv " << plot_id << " saved" << std::endl;
		std::cout << "Output Complete" << std::endl;
		window_dump.flush();
		window_dump.close();
	}
	else std::cout << "Skipping final window output" << std::endl;
	std::cout << "Total Whitespace: " << bw_ws_count << std::endl;
	bw_ws_count = 0;
	//TODO: Cleanup
	if (in_samples.is_open()){
		in_samples.close();
	}
	if (config.is_open()){
		in_samples.close();
	}
	free(re);
	free(processed_ptr);

	//bar_plot(&window_vec); //If the end of the data has been reached, then display the data
}
template<class T>
void print_vector(std::vector<T> &vect){

	std::vector<T>::iterator it;
	for (it = vect.begin(); it < vect.end(); it++) {
		if (it != vect.end() - 1) std::cout << *it << ",";
		else std::cout << *it << std::endl;
	}
}

//fftf_proc_s( array of samples, index of fft thread, mgl container for display, size of the file being read)
//This is only for use with a CPU only system
/*
void fftf_proc_s (std::complex<short>* re, int idx, mglData *mglY, double fsz) {
	
	//boost::unique_lock<boost::mutex> lock(fft_mutex[idx]);
	//Prepare your body for the FFT
	//CONVENTION fftwf = floats, instead of doubles used before. AS we only have 14 bits of precision it doesnt make sense to use double point precision
	// but im using doubles anyway :D
	//LARGE_INTEGER wind;
	//QueryPerformanceCounter(&wind);

	std::complex<float>* buff;
	buff = (std::complex<float>*) malloc(sizeof(std::complex<float>) * WIN_SAMPS); //Buffer for floated samples n.n

	std::complex<short>* s_ptr = &re[0];

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

	//long long ws_count = 0;
	std::vector<window> window_vec;
	std::vector<window>* w_vec_ptr = &window_vec;
	int frame_number = 0;
	bool init = false;

	fft_buff = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * WIN_SAMPS); //buffer allocated for Output
	w_array = (float*) malloc(sizeof(float) * WIN_SAMPS); //allocate memory for window coefficients
	avg_buff = (float*) malloc(sizeof(float) * WIN_SAMPS); //buffer allocated for averaging FFT results

	for (int i = 0; i < WIN_SAMPS; i++){
		ws_array[i] = 0;
		ws_frame[i] = 0;
		overlap[i] = 0;
		avg_buff[i] = 0;
	}
	
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

	LARGE_INTEGER ftpl, wind, avge, perffreq;
	double wf, fa, ae, pf;
	QueryPerformanceFrequency(&perffreq);
	pf = perffreq.QuadPart;
	int testGPU = 0;

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
				//std::cout << progress << " % complete at " << rate << " MB/s \t\t\r";
				
			}

			//QueryPerformanceCounter(&wind);
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
			
			//QueryPerformanceCounter(&ftpl);

			//Spin the FFT yoooooooooo
			fftwf_execute(plan);

			//QueryPerformanceCounter(&avge);
			//Keep a moving average of every FFT that has happened until the averaging says so :D
			//Probably need to optimise this better. Pointer arithmetic is probably the go. 
			//#pragma omp parallel for

			for (int i = 0; i < WIN_SAMPS; i++) { 
				avg_buff[i] += ( 
					10*std::log10(std::abs(fft_buff[i][0]*fft_buff[i][0] + fft_buff[i][1]*fft_buff[i][1])/WIN_SAMPS) //DFT bin magnitude
					); 
			}				

			//wf = (ftpl.QuadPart - wind.QuadPart)/pf;
			//QueryPerformanceCounter(&wind);	
			//fa = (avge.QuadPart - ftpl.QuadPart)/pf;
			//ae = (wind.QuadPart - avge.QuadPart)/pf;
			//std::cout << "Windowing: " << wf << "s FFT: " << fa << "s Avg: " << ae << "s" << std::endl; 
			//Count the number of FFT frames that have been processed thus far
			avg_count++;


			//Perform ws detection 

			/*
			if (avg_count > 0 && avg_count%10 == 0 || samp >= fsz) {
				//frame_count++;

				
				testGPU++;
				QueryPerformanceCounter(&ftpl);
				wf = (ftpl.QuadPart - wind.QuadPart) / pf;
				//std::cout << "CPU Time : " << wf << std::endl;
				//std::cout << "CPU OFFSET : " << offset << std::endl;
				
				std::cout << "CPUsam: ";
				float temp;
				for (int i = 0; i < WIN_SAMPS; i++) {
					temp = (avg_buff[((WIN_SAMPS / 2) + i) % WIN_SAMPS] / 10 + offset);// <= threshold ? 1 : 0;
					std::cout << temp << ", ";
					//std::cout << avg_buff[((WIN_SAMPS / 2) + i) % WIN_SAMPS] << ", ";
				}
				std::cout << std::endl;
				
				if (testGPU == 10) {

					QueryPerformanceCounter(&wind);
					float* processed_ptr;
					int num_wins = 100;
					int averaging = 10;
					//printf("about to do the thing!");
					//processed_ptr = (float*)malloc(sizeof(float) * WIN_SAMPS * num_wins / averaging);
					processed_ptr = dothething(s_ptr, averaging, num_wins); //GPU Time @ 10 : 0.745034 ... GPU Time @ 1 : 0.743284 ... GPU Time @ 1024 : 0.935398
					QueryPerformanceCounter(&ftpl);
					wf = (ftpl.QuadPart - wind.QuadPart) / pf;
					std::cout << std::endl;

					for (int p = 0; p < 10; p++) {

						std::cout << "GPUsam: ";
						for (int m = 0; m < WIN_SAMPS; m++) std::cout << processed_ptr[m] << ", ";
						std::cout << std::endl;
						processed_ptr += WIN_SAMPS;
					}

					std::cout << "End." << std::endl;
					exit(0);
					}
				
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

				//if(init) detect_dev(ws_frame, 1, ws_array, overlap, w_vec_ptr, frame_number);
				if(init) detect_once(ws_frame, 1, ws_array, overlap, w_vec_ptr, frame_number);
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
			*/
/*
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
*/

/*
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
*/
/*
void py_plot_legacy() {

	std::cout << "To the Python mobile!" << std::endl << std::endl;


	PyObject *pName, *pModule, *pDict;

	char* py_file = "plot_helper";

	npy_intp dims[1] = { 4 }; //this would be NUM_SAMPS
	float x[4] = { 0, 5, 0, 9 };

	pName = PyUnicode_FromString(py_file);
	pModule = PyImport_Import(pName);
	if (pModule) {
		std::cout << "Module Loaded\n";
		pDict = PyModule_GetDict(pModule);
		pFunc = PyObject_GetAttrString(pModule, (char*)"pyitup");

		if (PyCallable_Check(pFunc)) {

			std::cout << "Build array\n";
			pArg = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, reinterpret_cast<void*>(x));
			std::cout << "Reinterpret\n";
			pArray = reinterpret_cast<PyArrayObject*>(pArg);
			std::cout << "Call Python\n";
			//pArg = PyTuple_New(1);
			//PyTuple_SetItem(pArg, 0, PyUnicode_FromString((char*)"FIRE ZE CANNON"));
			//PyTuple_SetItem(pArg, 0, reinterpret_cast<PyObject*>());
			PyGILState_STATE gstate = PyGILState_Ensure();
			PyObject_CallFunctionObjArgs(pFunc, pArray, NULL);

			x[3] = 150;

			PyObject_CallFunctionObjArgs(pFunc, pArray, NULL);
		}
		else {
			PyErr_Print();
		}

		Py_DECREF(pModule);
	}
	else {
		PyErr_Print();
	}


	Py_DECREF(pName);

	system("PAUSE");
	exit(0);

}*/