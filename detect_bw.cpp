// Detect TS, this is a single dimension detection algorithm, to just search through all the samples and output the timescales of windows of a single bandwidth bin

#include "process.h"
#include <complex>
#include <cmath>
#include <set>
#include <vector>
#include <iostream>

void detect_bw_rec(float* ws_frame, double* record, const int num_wins, int* ws_array, std::vector<window>* window_vec, int* frame_num) {

	//std::cout << "Commence Whitespace Detection" << std::endl;

	//Process the ws
	float* ws_ptr = ws_frame;
	int temp = 0;
	const int active_bw = WIN_SAMPS / 10;
	int s_idx, e_idx, bw_count;

	for (int m = 0; m < num_wins; m++) { //This will loop as many times as num_wins is specified

		if (m != 0) {
			ws_ptr += WIN_SAMPS;  //set ws_ptr to the beginning of the current whitespace frame (this is used for multiple frames in an array)
			(*frame_num)++;
		}
		temp = (m + 1) % 500;
		if (temp == 0 || m + 1 == num_wins) std::cout << "Detecting " << m + 1 << " of " << num_wins;

		for (int i = 0 + active_bw; i < WIN_SAMPS - active_bw; i++) {

			//Grab all the windows of max bandwidth, forcing TS to be 1
			if (ws_array[i] != 0){

				s_idx = i;
				e_idx = i;

				while (s_idx > 0 + active_bw && ws_array[s_idx - 1] != 0) s_idx--; //Find leftmost ws bound
				while (e_idx + 1 < WIN_SAMPS - active_bw && ws_array[e_idx + 1] != 0) e_idx++; //Find rightmost ws bound

				bw_count = e_idx - s_idx + 1;

				window_vec->push_back(window(1, bw_count, s_idx, bw_count, *frame_num));
				//fixed ...
				if (bw_count - 1 < 0) {
					std::cout << "Oh we gon have a problem here - less than 0 T.T " << (ws_array[i] - 1) << "\n";
					system("pause");
				}
				else if (bw_count - 1 >= (WIN_SAMPS / 2) - (WIN_SAMPS / 10)) {
					std::cout << "Too big lah! - dat overphlow " << (ws_array[i] - 1) << "\n";
					system("pause");
				}
				else record[bw_count-1]++;

				i = e_idx;
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

void detect_bw_rec_once(unsigned char* ws_frame, double* record, const int num_wins, int* ws_array, std::vector<window>* window_vec, int frame_num, double* count) {

	//std::cout << "Commence Whitespace Detection" << std::endl;

	//Process the ws
	unsigned char* ws_ptr = &ws_frame[0];

	const int active_bw = WIN_SAMPS / 10;
	int s_idx, e_idx, bw_count;


	for (int i = 0 + active_bw; i < WIN_SAMPS - active_bw; i++) if (ws_ptr[i] == 1) (*count)++;

	for (int i = 0 + active_bw; i < WIN_SAMPS - active_bw; i++) {


		//Grab all the windows of max bandwidth, forcing TS to be 1
		if (ws_array[i] != 0){


		s_idx = i;
		e_idx = i;

		while (s_idx > 0 + active_bw && ws_array[s_idx - 1] != 0) s_idx--; //Find leftmost ws bound
		while (e_idx + 1 < WIN_SAMPS - active_bw && ws_array[e_idx + 1] != 0) e_idx++; //Find rightmost ws bound

		bw_count = e_idx - s_idx;

		window_vec->push_back(window(1, bw_count, s_idx, bw_count, frame_num));

		record[bw_count-1]++;

		i = e_idx;
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