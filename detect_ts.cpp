// Detect TS, this is a single dimension detection algorithm, to just search through all the samples and output the timescales of windows of a single bandwidth bin

#include "process.h"
#include <complex>
#include <cmath>
#include <set>
#include <vector>
#include <iostream>

void detect_ts(float* ws_frame, const int num_wins, int* ws_array, std::vector<window>* window_vec, int* frame_num) {

	//std::cout << "Commence Whitespace Detection" << std::endl;

	//Process the ws
	float* ws_ptr = ws_frame;

	const int active_bw = WIN_SAMPS / 10;

	for (int m = 0; m < num_wins; m++) { //This will loop as many times as num_wins is specified

		if (m != 0) {
			ws_ptr += WIN_SAMPS;  //set ws_ptr to the beginning of the current whitespace frame (this is used for multiple frames in an array)
			(*frame_num)++;
		}
		
		std::cout << "Detecting " << m + 1 << " of " << num_wins << std::endl;

		for (int i = 0 + active_bw; i < WIN_SAMPS - active_bw; i++) {

			//Scans through the next frame for a 0 (no ws) that preceeded a previous 1 (ws), upon finding a zero, the associated whitespace opportunity that ended is then analysed 
			if ((int)ws_ptr[i] == 0 && ws_array[i] != 0){
				window_vec->push_back(window(ws_array[i], 1, i, ws_array[i], *frame_num));
			}
		}

		//Once Detection is performed for the "previous" frame, update with current frame.
		for (int i = 0; i < WIN_SAMPS; i++) { //If whitespace does exist at this bin, increment appropriate entry 

			if ((int)ws_ptr[i] != 0) {
				ws_array[i]++;
			}
			else ws_array[i] = 0;
		}

	}

}

void detect_ts_once(unsigned char* ws_frame, const int num_wins, int* ws_array, std::vector<window>* window_vec, int frame_num) {

	//std::cout << "Commence Whitespace Detection" << std::endl;

	//Process the ws
	unsigned char* ws_ptr = &ws_frame[0];

	const int active_bw = WIN_SAMPS / 10;

	for (int i = 0 + active_bw; i < WIN_SAMPS - active_bw; i++) {

		//Scans through the next frame for a 0 (no ws) that preceeded a previous 1 (ws), upon finding a zero, the associated whitespace opportunity that ended is then analysed 
		if ((int)ws_ptr[i] == 0 && ws_array[i] != 0){
			window_vec->push_back(window(ws_array[i], 1, i, ws_array[i], frame_num));
		}
	}

	//Once Detection is performed for the "previous" frame, update with current frame.
	for (int i = 0; i < WIN_SAMPS; i++) { //If whitespace does exist at this bin, increment appropriate entry 

		if ((int)ws_ptr[i] != 0) {
			ws_array[i]++;
		}
		else ws_array[i] = 0;
	}
}

void detect_ts_rec(float* ws_frame, double* record, const int num_wins, int* ws_array, std::vector<window>* window_vec, int* frame_num) {

	//std::cout << "Commence Whitespace Detection" << std::endl;

	//Process the ws
	float* ws_ptr = ws_frame;
	int temp = 0;
	const int active_bw = WIN_SAMPS / 10;

	for (int m = 0; m < num_wins; m++) { //This will loop as many times as num_wins is specified

		if (m != 0) {
			ws_ptr += WIN_SAMPS;  //set ws_ptr to the beginning of the current whitespace frame (this is used for multiple frames in an array)
			(*frame_num)++;
		}
		temp = (m + 1) % 100;
		if (temp == 0 || m+1 == num_wins) std::cout << "Detecting " << m + 1 << " of " << num_wins;

		for (int i = 0 + active_bw; i < WIN_SAMPS - active_bw; i++) {

			//Scans through the next frame for a 0 (no ws) that preceeded a previous 1 (ws), upon finding a zero, the associated whitespace opportunity that ended is then analysed 
			if ((int)ws_ptr[i] == 0 && ws_array[i] != 0){
				window_vec->push_back(window(ws_array[i], 1, i, ws_array[i], *frame_num));
				//TS of 1 currently set to 52.4288ms as each window is averaged accross 10 frames ... THIS WILL BE FIXED SOON. A frame is 5.24288ms: 131,072/25e6 (num samps/sample rate)
				//fixed ...
				if (ws_array[i] - 1 < 0) {
					std::cout << "Oh we gon have a problem here - less than 0 T.T " << (ws_array[i] - 1) << "\n";
					system("pause");
				}
				else if (ws_array[i] - 1 >= 5722*10) {
					std::cout << "Too big lah! - dat overphlow " << (ws_array[i] - 1) << "\n";
					system("pause");
				}
				else record[ws_array[i]-1] += ws_array[i];
			}
		}

		//Once Detection is performed for the "previous" frame, update with current frame.
		if (temp == 0 || m + 1 == num_wins) std::cout << " ..." << std::endl;
		for (int i = 0; i < WIN_SAMPS; i++) { //If whitespace does exist at this bin, increment appropriate entry 

			if ((int)ws_ptr[i] != 0) {
				ws_array[i]++;
			}
			else ws_array[i] = 0;
		}

	}

}

void detect_ts_rec_once(unsigned char* ws_frame, double* record, const int num_wins, int* ws_array, std::vector<window>* window_vec, int frame_num) {

	//std::cout << "Commence Whitespace Detection" << std::endl;

	//Process the ws
	unsigned char* ws_ptr = &ws_frame[0];

	const int active_bw = WIN_SAMPS / 10;

	for (int i = 0 + active_bw; i < WIN_SAMPS - active_bw; i++) {

		//Scans through the next frame for a 0 (no ws) that preceeded a previous 1 (ws), upon finding a zero, the associated whitespace opportunity that ended is then analysed 
		if ((int)ws_ptr[i] == 0 && ws_array[i] != 0){
			window_vec->push_back(window(ws_array[i], 1, i, ws_array[i], frame_num));
			//TS of 1 currently set to 52.4288ms as each window is averaged accross 10 frames ... THIS WILL BE FIXED SOON. A frame is 5.24288ms: 131,072/25e6 (num samps/sample rate)
			record[ws_array[i]-1] += ws_array[i];
		}
	}

	//Once Detection is performed for the "previous" frame, update with current frame.
	for (int i = 0; i < WIN_SAMPS; i++) { //If whitespace does exist at this bin, increment appropriate entry 

		if ((int)ws_ptr[i] != 0) {
			ws_array[i]++;
		}
		else ws_array[i] = 0;
	}

}