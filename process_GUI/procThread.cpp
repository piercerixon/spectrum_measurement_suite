#include "gui.h"
#include "procThread.h"
#include "window.h"
#include <QDebug>
#include <iostream>

//Constructor
procThread::procThread(QObject* parent) : QThread(parent) {

	restart = false;
	abort = false;

	setup();

	//start(HighPriority); //Unsure if there is a higher priority

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
	qDebug() << "I have awoken!";

	OutputDebugStringA("\nCheck Selection: ");
	OutputDebugStringA(selection);

	//Send trigger to initialise GUI perhaps?

	//Samples Setup 
	std::ifstream in_samples;

	mutex.lock();
	std::string sampfile = this->selection;
	int currfile = this->filenum_base;
	int pltLength = this->plotLength;
	mutex.unlock();

	struct _stat64 stat_buff;
	_stat64(sampfile.c_str(), &stat_buff);
	double file_size = stat_buff.st_size;

	float* processed_ptr_base = NULL;
	float* processed_ptr = NULL;

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

	//Window vector for storing and outputting windows 
	//to be used with detect algorithms (currently unimplemented in class 11/4/16)
	std::vector<window> window_vec;
	std::vector<window>* w_vec_ptr = &window_vec;

	size_t bytesread = 1;
	size_t bytesleft = 0;

	std::complex<short>* re;//Buffer for read in samples
	re = (std::complex<short>*) malloc(bytes_to_read);
	int currwins = 0;
	double ts_array[MIN5] = { 0 }; //intialise array to 0, BEWARE!!!! index 0 is actaully Timescale 1. (for the purposes of plotting n.n)
	double total_ws = 0;
	int plot_id = 0;
	const int SCALE = 5;
	int frames_remain = 0;


	//Infinite event loop
	while (true) {

		mutex.lock();
		//copy across necessary variables from the object
		//THIS IS TO BE USED TO COPY STUFF FROM THE NOTIFY FUNCTION VIA THE OBJECT
		//for example, the ranges should be copied accross
		mutex.unlock();

		//Core processing
				

		in_samples.open(sampfile, std::ifstream::binary);
		in_samples.seekg(0, std::ifstream::beg);

		//emit some things after junk done

		//at the end of the junk, wait to be asked to do more junk n.n
		mutex.lock();
		if (!restart)
			condition.wait(&mutex); //will wait untill told to run again
		restart = false;
		mutex.unlock();

	}

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


	//Cleanup
	if (in_samples.is_open()){
		in_samples.close();
	}
	if (config.is_open()){
		in_samples.close();
	}
	free(re);
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

}
