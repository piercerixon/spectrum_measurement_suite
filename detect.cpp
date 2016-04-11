#include "process.h"
#include <complex>
#include <cmath>
#include <set>
#include <vector>
#include <iostream>

void detect(float* ws_frame, const int num_wins, int* ws_array, int *overlap, std::vector<window>* window_vec, int* frame_num) {

	//std::cout << "Commence Whitespace Detection" << std::endl;
		
	//Process the ws
	float* ws_ptr = ws_frame;
	//int overlap[w_samps];

	std::vector<int> zero_val_idx;
	std::vector<int> zero_mask;

	std::vector<window> temp_window_vec;
	std::vector<window>::iterator win_itr;
	bool unique;
		
	std::set<int>::iterator itr;
	std::set<int> win_vals_set; //Consider order, descending? - defaults to ascending

	//std::vector<window> window_vec;

	int s_idx, e_idx;
	int bw_count;
	int contribution = 0;
	int lwb_idx, rwb_idx;
	int min;

	//const int active_bw = 0;
	const int active_bw = WIN_SAMPS/10;

	for (int m = 0; m < num_wins; m++) { //This will loop as many times as num_wins is specified

		if (m != 0) {
			ws_ptr += WIN_SAMPS;  //set ws_ptr to the beginning of the current whitespace frame (this is used for multiple frames in an array)
			(*frame_num)++;
		}
		std::cout << "Detecting " << m+1 << " of " << num_wins << std::endl;

		for (int i = 0 + active_bw; i < WIN_SAMPS - active_bw; i++) {

			//Scans through the next frame for a 0 (no ws) that preceeded a previous 1 (ws), upon finding a zero, the associated whitespace opportunity that ended is then analysed 
			if ((int)ws_ptr[i] == 0 && ws_array[i] != 0){

				s_idx = i;
				e_idx = i;
				zero_val_idx.clear();
				win_vals_set.clear();

				while (s_idx > 0 + active_bw && ws_array[s_idx - 1] != 0) s_idx--; //Find leftmost ws bound
				while (e_idx + 1 < WIN_SAMPS - active_bw && ws_array[e_idx + 1] != 0) e_idx++; //Find rightmost ws bound

				for (int j = s_idx; j <= e_idx; j++) {
					win_vals_set.insert(ws_array[j]); //Track unique reference of every possible TS under window
					if (ws_ptr[j] == 0) zero_val_idx.push_back(j); //Copy zero locations for appropriate window
				}

				//Iterate through every detected zero, for each unique TS to find all windows
				for (int j = 0; j < zero_val_idx.size(); j++){


					//Note: this make sense for an ascending ordered set
					for (itr = win_vals_set.begin(); itr != win_vals_set.end() && ws_array[zero_val_idx[j]] >= *itr; itr++) {

						lwb_idx = zero_val_idx[j]; //Left Whitespace Bound Index
						rwb_idx = zero_val_idx[j]; //Right Whitespace Bound Index

						//Find Window
						while (lwb_idx > 0 + active_bw &&  *itr <= ws_array[lwb_idx - 1]) lwb_idx--; //Find leftmost ws bound of sub-windwow	
						while (rwb_idx + 1 < WIN_SAMPS - active_bw && *itr <= ws_array[rwb_idx + 1]) rwb_idx++; //Find rightmost ws bound of sub-windwow

						//Calculate BW
						bw_count = rwb_idx - lwb_idx + 1;

						if (bw_count == 1){ //If BW is only 1, then the TS is obviously equal to whatever the corresponsing ws_array value is, and does not need to iterate through the set
							itr = win_vals_set.find(ws_array[zero_val_idx[j]]);
						}

						unique = true;
						//Check if output is unique, then save result
						for (win_itr = temp_window_vec.begin(); win_itr < temp_window_vec.end();){
							if (*itr <= win_itr->timescale && bw_count == win_itr->bandwidth && lwb_idx == win_itr->frequency) {
								unique = false;
								break;
								// if the new timescale is longer than previous windows, merge the contributions of those windows into the larger timescale window, then erase them, as they are (effectively) identical
							}
							else if (*itr > win_itr->timescale && bw_count == win_itr->bandwidth && lwb_idx == win_itr->frequency) {
								contribution += win_itr->whitespace;
								win_itr = temp_window_vec.erase(win_itr); //If a larger TS is found, remove
							}
							else win_itr++;
						}

						if (unique) {
							// Calculate Overlap
							for (int p = lwb_idx; p <= rwb_idx; p++){

								//if(ws_ptr[p] == 0) {
								min = ((ws_array[p] - overlap[p]) >= *itr) ? (*itr) : (ws_array[p] - overlap[p]);
								if (min < 0) min = 0;
								contribution += min;
								overlap[p] += min;
								//}
							}
							//std::cout << " BW: " << bw_count << " TS: " << *itr << " Con: " << contribution << std::endl << std::endl; //BW of ws is equal to e_idx-s_idx+1
							window temp = window(*itr, bw_count, lwb_idx, contribution, *frame_num); //TS,BW,F,WS,F_n
							temp_window_vec.push_back(temp);
							contribution = 0;
						}
						//Increment to the next TS level
						//if(itr != win_vals_set.end()) itr++;
					}
				}

				//Save all the unique windows created by this detection phase
				for (win_itr = temp_window_vec.begin(); win_itr < temp_window_vec.end(); win_itr++){
					window_vec->push_back(*win_itr);
				}
				//Clear temporary window vector
				temp_window_vec.clear();

				//Increment the counter by the width of the total detected window.
				i = e_idx;
			}
		}

		//Once Detection is performed for the "previous" frame, update with current frame.
		for (int i = 0; i < WIN_SAMPS; i++) { //If whitespace does exist at this bin, increment appropriate entry 

			overlap[i] *= (int)ws_ptr[i]; //Mask overlap array appropriately

			if ((int)ws_ptr[i] != 0) {
				ws_array[i]++;
			}
			else ws_array[i] *= ws_ptr[i];
		}

	}

}

void detect_once(unsigned char* ws_frame, const int num_wins, int* ws_array, int *overlap, std::vector<window>* window_vec, int frame_num) {

	//std::cout << "Commence Whitespace Detection" << std::endl;

	//Process the ws
	unsigned char* ws_ptr = &ws_frame[0];
	//int overlap[w_samps];

	std::vector<int> zero_val_idx;
	std::vector<int> zero_mask;

	std::vector<window> temp_window_vec;
	std::vector<window>::iterator win_itr;
	bool unique;

	std::set<int>::iterator itr;
	std::set<int> win_vals_set; //Consider order, descending? - defaults to ascending

	//std::vector<window> window_vec;

	int s_idx, e_idx;
	int bw_count;
	int contribution = 0;
	int lwb_idx, rwb_idx;
	int min;

	//const int active_bw = 0;
const int active_bw = WIN_SAMPS / 10;

	for (int i = 0 + active_bw; i < WIN_SAMPS - active_bw; i++) {

		//Scans through the next frame for a 0 (no ws) that preceeded a previous 1 (ws), upon finding a zero, the associated whitespace opportunity that ended is then analysed 
		if ((int)ws_ptr[i] == 0 && ws_array[i] != 0){

			s_idx = i;
			e_idx = i;
			zero_val_idx.clear();
			win_vals_set.clear();

			while (s_idx > 0 + active_bw && ws_array[s_idx - 1] != 0) s_idx--; //Find leftmost ws bound
			while (e_idx + 1 < WIN_SAMPS - active_bw && ws_array[e_idx + 1] != 0) e_idx++; //Find rightmost ws bound

			for (int j = s_idx; j <= e_idx; j++) {
				win_vals_set.insert(ws_array[j]); //Track unique reference of every possible TS under window
				if ((int)ws_ptr[j] == 0) zero_val_idx.push_back(j); //Copy zero locations for appropriate window
			}

			//Iterate through every detected zero, for each unique TS to find all windows
			for (int j = 0; j < zero_val_idx.size(); j++){


				//Note: this make sense for an ascending ordered set
				for (itr = win_vals_set.begin(); itr != win_vals_set.end() && ws_array[zero_val_idx[j]] >= *itr; itr++) {

					lwb_idx = zero_val_idx[j]; //Left Whitespace Bound Index
					rwb_idx = zero_val_idx[j]; //Right Whitespace Bound Index

					//Find Window
					while (lwb_idx > 0 + active_bw &&  *itr <= ws_array[lwb_idx - 1]) lwb_idx--; //Find leftmost ws bound of sub-windwow	
					while (rwb_idx + 1 < WIN_SAMPS - active_bw && *itr <= ws_array[rwb_idx + 1]) rwb_idx++; //Find rightmost ws bound of sub-windwow

					//Calculate BW
					bw_count = rwb_idx - lwb_idx + 1;

					if (bw_count == 1){ //If BW is only 1, then the TS is obviously equal to whatever the corresponsing ws_array value is, and does not need to iterate through the set
						itr = win_vals_set.find(ws_array[zero_val_idx[j]]);
					}

					unique = true;
					//Check if output is unique, then save result
					for (win_itr = temp_window_vec.begin(); win_itr < temp_window_vec.end();){
						if (*itr <= win_itr->timescale && bw_count == win_itr->bandwidth && lwb_idx == win_itr->frequency) {
							unique = false;
							break;
							// if the new timescale is longer than previous windows, merge the contributions of those windows into the larger timescale window, then erase them, as they are (effectively) identical
						}
						else if (*itr > win_itr->timescale && bw_count == win_itr->bandwidth && lwb_idx == win_itr->frequency) {
							contribution += win_itr->whitespace;
							win_itr = temp_window_vec.erase(win_itr); //If a larger TS is found, remove
						}
						else win_itr++;
					}

					if (unique) {
						// Calculate Overlap
						for (int p = lwb_idx; p <= rwb_idx; p++){

							//if(ws_ptr[p] == 0) {
							min = ((ws_array[p] - overlap[p]) >= *itr) ? (*itr) : (ws_array[p] - overlap[p]);
							if (min < 0) min = 0;
							contribution += min;
							overlap[p] += min;
							//}
						}
						//std::cout << " BW: " << bw_count << " TS: " << *itr << " Con: " << contribution << std::endl << std::endl; //BW of ws is equal to e_idx-s_idx+1
						window temp = window(*itr, bw_count, lwb_idx, contribution, frame_num); //TS,BW,F,WS,F_n
						temp_window_vec.push_back(temp);
						contribution = 0;
					}
					//Increment to the next TS level
					//if(itr != win_vals_set.end()) itr++;
				}
			}

			//Save all the unique windows created by this detection phase
			for (win_itr = temp_window_vec.begin(); win_itr < temp_window_vec.end(); win_itr++){
				window_vec->push_back(*win_itr);
			}
			//Clear temporary window vector
			temp_window_vec.clear();

			//Increment the counter by the width of the total detected window.
			i = e_idx;
		}
	}

	//Once Detection is performed for the "previous" frame, update with current frame.
	for (int i = 0; i < WIN_SAMPS; i++) { //If whitespace does exist at this bin, increment appropriate entry 

		overlap[i] *= (int)ws_ptr[i]; //Mask overlap array appropriately

		if ((int)ws_ptr[i] != 0) {
			ws_array[i]++;
		}
		else ws_array[i] *= (int)ws_ptr[i];
	}

}