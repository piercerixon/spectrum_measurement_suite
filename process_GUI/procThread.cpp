#include "gui.h"
#include "procThread.h"
#include "window.h"
#include "cuda_module.h"
#include <boost/lexical_cast.hpp>
#include <QDebug>
#include <iostream>

//Constructor
procThread::procThread(QObject* parent) : QThread(parent) {

	restart = false;
	abort = false;

	start(HighPriority); //Unsure if there is a higher priority

}

procThread::~procThread(){

	qDebug() << "I have been terminated!";
	mutex.lock();
	abort = true; // <- this is called from the parent thread, thus a mutex is necessary when destructing as the object will exist in a separate worker thread
	condition.wakeOne();
	mutex.unlock();

	wait();
}

void procThread::run(){

	setup();

	qDebug() << "I have awoken!";

	OutputDebugStringA("\nCheck Selection: ");
	OutputDebugStringA(selection);

	//Samples Setup 
	std::ifstream in_samples;
	std::string sampfile;

	mutex.lock(); //these are likely unnecessary
	std::string filename = this->selection;
	int currfile = this->filenum_base;
	int pltLength = this->plotLength;
	int t_maxFrames = this->maxFrames;
	mutex.unlock(); //end unnecessary

	//struct _stat64 stat_buff;
	//_stat64(sampfile.c_str(), &stat_buff);
	//double file_size = stat_buff.st_size;

	//For accessing and storing data returned by GPU
	float* processed_ptr_base = NULL;
	float* processed_ptr = NULL;

	int frame_number = 0;
	bool init = true;

	/*
	int ws_array[WIN_SAMPS];
	unsigned char zero_frame[WIN_SAMPS];
	int overlap[WIN_SAMPS];

	for (int i = 0; i < WIN_SAMPS; i++){
		ws_array[i] = 0;
		zero_frame[i] = 0;
		overlap[i] = 0;
	}
	*/

	//Window vector for storing and outputting windows 
	//to be used with detect algorithms (currently unimplemented in class 11/4/16)
	std::vector<window> window_vec;
	std::vector<window>* w_vec_ptr = &window_vec;

	size_t bytesread = 1;
	size_t bytesleft = 0;
	int64_t bytes_to_read = TWO_GB;

	std::complex<short>* re;//Buffer for read in samples
	//Note this will only work for GPUs with a memory of 12GB+
	re = (std::complex<short>*) malloc(TWO_GB);

	double byteDepth = 0;
	double reqFile = 0;
	int currwins = 0;
	double ts_array[MIN5] = { 0 }; //intialise array to 0, BEWARE!!!! index 0 is actaully Timescale 1. (for the purposes of plotting n.n)
	double total_ws = 0;
	int plot_id = 0;
	const int SCALE = 5;
	int frames_remain = 0;
	int64_t sampstart = 0;

	bool unInit = true;
	int t_rng_max, t_rng_min;

	const int64_t FRAME_SIZE = WIN_SAMPS*sizeof(std::complex<short>)*averaging; //size of a frame in bytes

	qDebug() << bytes_to_read / FRAME_SIZE; //409.6 T.T

	mutex.lock();
	int filenum_max = this->filenum_max;
	mutex.unlock();
	qDebug() << "Display first plot";
	//Infinite event loop //forever
	while (true) {

		mutex.lock();
		//copy across necessary variables from the object
		//THIS IS TO BE USED TO COPY STUFF FROM THE NOTIFY FUNCTION VIA THE OBJECT
		//for example, the ranges should be copied accross
		int reqFrame = this->requestedFrame;
		int frameMax = this->frame_rng_max;
		int frameMin = this->frame_rng_min;
		mutex.unlock();

		qDebug() << "reqFrame: " << reqFrame;

		//Core processing
		//calculate where in the sample set the reqFrame exists THIS CURRENTLY ASSUMES THE MAX FILE SIZE IS 4GB

		//frame number * number of samples required to build a frame / max filesize = file number
		//this will absolutely need to be modified when the averaging is fixed.
		if (reqFrame < frameMin || reqFrame > frameMax || unInit) {
			qDebug() << "req: " << reqFrame << " rng_min: " << frameMin << " rng_max: " << frameMax;
			unInit = false;

			//reqFrame - 1 to take into account the samples required prior to perform the averaging
			if (t_maxFrames < 409){
				qDebug() << "This is a pretty small file ... cant handle it ... yet, let me know if its necessary";
				this->~procThread();
			}
			else if (reqFrame <= 100) { //case where we are closer than 100 to the beginning
				reqFile = filenum_base;
				sampstart = 0;
				t_rng_min = 0;
				t_rng_max = 408;

				qDebug() << "start - min: " << t_rng_min << " max: " << t_rng_max;
			}
			else if (reqFrame + 308 > t_maxFrames) { //case where we will overflow
				reqFile = filenum_base + std::floor(((t_maxFrames - 408) * FRAME_SIZE) / FOUR_GB);
				sampstart = (((t_maxFrames - 408) * FRAME_SIZE)) % FOUR_GB;
				t_rng_min = t_maxFrames - 408;
				t_rng_max = t_maxFrames;

				qDebug() << "end - min: " << t_rng_min << " max: " << t_rng_max;
			}
			else { //every other case
				reqFile = filenum_base + std::floor(((reqFrame - 100) * FRAME_SIZE) / FOUR_GB);
				sampstart = (((reqFrame - 100) * FRAME_SIZE)) % FOUR_GB;
				t_rng_min = reqFrame - 100;
				t_rng_max = reqFrame + 308;

				qDebug() << "general - min: " << t_rng_min << " max: " << t_rng_max;
			}

			//calculate sampstart
			
			qDebug() << "reqFile: " << reqFile << " sampstart: " << sampstart;

			mutex.lock();
			this->frame_rng_min = t_rng_min;
			this->frame_rng_max = t_rng_max;
			mutex.unlock();

		//as the samples to read are only half that of the typical sample file, this does not have to be recursive.			
		//	while (bytesleft > 0 && reqFile < filenum_max){

			//check to see if the file requested is already opened, if not, continue, otherwise close it and open the one requested.
			if (sampfile != filename.substr(0, filename.find_last_of("_") + 1) + boost::lexical_cast<std::string>(reqFile)+".dat") {
				in_samples.close();
				sampfile = filename.substr(0, filename.find_last_of("_") + 1) + boost::lexical_cast<std::string>(reqFile)+".dat";
				in_samples.open(sampfile, std::ifstream::binary);
				qDebug() << "Opening new sample file: " << QString::fromStdString(sampfile);
			}
			else if (!in_samples.is_open()){
				sampfile = filename.substr(0, filename.find_last_of("_") + 1) + boost::lexical_cast<std::string>(reqFile)+".dat";
				in_samples.open(sampfile, std::ifstream::binary);
				qDebug() << "Sample file not previously open" << QString::fromStdString(sampfile);
			}

			//grab the required number of samples, if we dont have enough, open the next file to get more n.n, the previous frame range settings should protect this from accessing things that dont exist
			if (in_samples.is_open()) {
				in_samples.seekg(sampstart, std::ifstream::beg);
				in_samples.read((char*)re, bytes_to_read);
				bytesread = in_samples.gcount();
				qDebug() << "Bytes read: " << bytesread;
				bytesleft = bytes_to_read - bytesread;

				//if need to span multiple files
				if (bytesleft > 0 && reqFile < filenum_max) {
					//read remaining bytes required
					sampstart = 0;
					in_samples.close();
					sampfile = filename.substr(0, filename.find_last_of("_") + 1) + boost::lexical_cast<std::string>(++reqFile) + ".dat";
					in_samples.open(sampfile, std::ifstream::binary);
					in_samples.read((char*)re, bytesleft);
					qDebug() << "Additional bytes read: " << in_samples.gcount();
					bytesread += in_samples.gcount();
				}
			}

			currwins = (bytesread / (4 * WIN_SAMPS * averaging)) * averaging; //this is done to drop the decimal
		//	}

			processed_ptr_base = (float*)realloc(processed_ptr_base, sizeof(float) * WIN_SAMPS * currwins / averaging);

			//FFT the samples and store them in processed_ptr_base
			dothething(re, processed_ptr_base, averaging, currwins);

		} //if the frames dont have to be recalculated, this statement will not be executed.

		//emit some things after junk done
		processed_ptr = &processed_ptr_base[(reqFrame - t_rng_min)*WIN_SAMPS];
		qDebug() << "processed_ptr idx: " << (reqFrame - t_rng_min) << "* WIN_SAMPS";
		//processed_ptr += ((reqFrame - t_rng_min)*WIN_SAMPS);
		preparePlot(processed_ptr, WIN_SAMPS);

		//Need to do cheeky cuda things here, the biggest blocking call is reading the samples in

		//at the end of the junk, wait to be asked to do more junk n.n
		mutex.lock();
		if (!restart) {
			qDebug() << "yawn";
			condition.wait(&mutex); //will wait untill told to run again
		}							//i.e. another plot request
		restart = false;
		mutex.unlock();

	}

	/*

	std::cout << "Finishing up" << std::endl;

	//Close the samples off
	frame_number++;
	//detect_once(zero_frame, 1, ws_array, overlap, w_vec_ptr, frame_number);
	//detect_ts_once(zero_frame, 1, ws_array, w_vec_ptr, frame_number);

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
	*/

	//Cleanup
	if (in_samples.is_open()){
		in_samples.close();
	}
	free(re);

	qDebug() << "Processing thread reached the end D:";
}

void procThread::preparePlot(float* powerArray, int size){

	//plotLength set to 2048, so only a max of 2048 datapoints are displayed.
	//additionally, the max datapoint under each step will be the only
	int step = size / plotLength;
	if (step < 1) {
		plotLength = size;
		step = 1;
		qDebug() << "Plot length < 2048";
	}
	else qDebug() << "Step size set to: " << step; 

	double temp = 0;
	double max = -1000;
	QVector <double> plotAvg(size,0), plotMax(size,0);
	for (int i = 0; i < plotLength; i++){
		for (int j = 0; j < step; j++) {

			temp += powerArray[i*step + j];
			if (powerArray[i*step + j] > max) max = powerArray[i*step + j];
		}
		temp /= step;
		plotAvg[i] = temp;
		plotMax[i] = max;
		temp = 0;
		max = -1000;
	}

	emit plotSignal(plotAvg, plotMax);

	qDebug() << "emit plotSignal";
}
