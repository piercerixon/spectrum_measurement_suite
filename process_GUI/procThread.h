#ifndef PROCTHREAD_H
#define PROCTHREAD_H

#include <QtCore>
#include <fstream>
#include <complex>
#include <stdint.h>
#include <Windows.h>
#include <sys/stat.h>
//#include "process.h"

const int MIN5 = 5722;
const int WIN_SAMPS = 131072;
const double TWO_GB = std::pow(2.0, 31);

class procThread : public QThread {
	
	Q_OBJECT

public: 
	procThread(QObject *parent = 0);
	~procThread();	

	//Invoked by the parent thread to select frame to display
	void requestUpdate(int);

signals:
	void plotSignal(QVector<double>, QVector<double>);

private:
	void run();
	//Takes an array of floats and the size of the array
	void preparePlot(float*, int);
	void setup();
	void procData(float*, int);
	void selectFile(char*);

	//Operational parameters for thread
	QMutex mutex;
	QWaitCondition condition;
	bool restart;
	bool abort;

	int plotLength = 2048;
	char selection[MAX_PATH];

	//TEMPORARY
	int64_t averaging = 10;

	int64_t bytes_to_read = (TWO_GB / (sizeof(std::complex<short>) * WIN_SAMPS * averaging)) *(WIN_SAMPS * sizeof(std::complex<short>)  * averaging);

	//Configuration variables
	float centre_freq, samp_rate, bandwidth, rx_gain;
	long long maxFrames = 0; //Note, under the current implementation, this will result in dropped frames (remainder from %10).
	std::string device, filename;
	std::ifstream config; //watch this one for errors bru

	int filenum_base = -1; //setup will modify this, if -1, then no samples have been selected.
	int filenum_max = -1; //similarly to base, if -1, no max.

	int requestedFrame = -1;
	int frame_rng_min = -1;
	int frame_rng_max = -1;

};

#endif