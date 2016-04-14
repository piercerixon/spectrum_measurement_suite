#include "procThread.h"

#include <string>
#include <iostream>
#include <csignal>
#include <sys/stat.h>
#include <boost/lexical_cast.hpp>

void procThread::selectFile(char* file) {

	sleep(1);

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
		qDebug() << "No file selected! Destroying";
		this->~procThread();
	}
	else
	{
		//File selected
		std::cout << "Loading file: " << file << std::endl;
	}

}

void procThread::setup() {

	mutex.lock();
	selectFile(selection);
	
	if (selection == NULL) {
		qDebug() << "No file has been selected, this thread will now terminate";
		this->~procThread();
	}

	//Configuration
	std::string base = selection;	
	qDebug(base.c_str());
	mutex.unlock();

	std::string path = base.substr(0, base.find_last_of("/\\") + 1);
	std::string token = base.substr(base.find_last_of("/\\") + 1);
	std::string cfgfile = path + "sample_config.cfg";
	std::cout << cfgfile << std::endl;

	std::ifstream config;
	config = std::ifstream(cfgfile, std::ifstream::in);
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
		mutex.lock();
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
		mutex.unlock();
	}

	if (config.is_open()) config.close();

	//Sample Files
	std::string filenum_s = token.substr(token.find_last_of("_") + 1, (token.find(".dat") - token.find_last_of("_") - 1)); //extract the number of the sample file

	//int filenum_base = boost::lexical_cast<int>(filenum_s);
	//int filenum_max = boost::lexical_cast<int>(filenum_s);

	int t_filenum_max = stoi(filenum_s);
	
	struct _stat64 stat;
	std::string search;
	int64_t stat_size = 0;

	std::cout << "Only sequential sample files will be included" << std::endl;

	while (true) {
		search = base.substr(0, base.find_last_of("_") + 1) + boost::lexical_cast<std::string>(t_filenum_max)+".dat";

		if (_stat64(search.c_str(), &stat) != -1) {
			qDebug() << "File: " << t_filenum_max << " included";
			std::cout << "File: " << t_filenum_max++ << " included" << std::endl;
			
			stat_size += stat.st_size;
			_stat64(search.c_str(), &stat);

		}
		else {
			std::cout << std::endl << t_filenum_max << " sample file(s) discovered\n\n" << "Cumulative filesize: " << stat_size / (1 << 20) << " MB\n\n";
			qDebug() << t_filenum_max - stoi(filenum_s) << " sample file(s) discovered\n\n" << "Cumulative filesize: " << stat_size / (1 << 20) << " MB";
			break;
		}
	}

	//int64_t three_gb = std::pow(2.0, 31) + std::pow(2.0, 30);
	int remain = (((int64_t)stat_size / (4 * WIN_SAMPS)) % 10);
	
	//Copy temp variables into object variables
	mutex.lock();
	this->filenum_max = t_filenum_max;
	this->filenum_base = stoi(filenum_s);
	this->maxFrames = (stat_size / (4 * WIN_SAMPS * 10));
	mutex.unlock();

	std::cout << "Total number of frames: " << maxFrames << " Windows dropped: " << remain << std::endl;
	qDebug() << "\nFrames: " << maxFrames;
}


void procThread::requestUpdate(int frame) {

	//perform initial checks
	if (filenum_base == -1 || filenum_max == -1) { 
		qDebug() << "Illegal filenumber range, destroying";
		this->~procThread();
	}
	if (frame >= 0 && frame < maxFrames) {

		QMutexLocker locker(&mutex);
		//Copy across the parameters of the update

		this->requestedFrame = frame;

		if (!isRunning()) {
			start(HighPriority);
		}
		else {
			restart = true;
			condition.wakeOne();
		}
	}
	else qDebug() << "Illegal frame request";
}

void procThread::procData(float* out, int frame) {
	//calculate where the frame would be located within the sample set
	//return a 2GB slice of processed data to re (approx 409 frames at 131072 sample depth)

}