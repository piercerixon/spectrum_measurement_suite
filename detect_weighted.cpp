#include "process.h"
#include <complex>
#include <cmath>
#include <set>
#include <vector>
#include <iostream>


/////////INCOMPLETE !!!!!!

/*
The objective of this function is to scan through the spectrum and attribute windows based on weighted metrics. Each window is evaluated for its utility
and the highest utility set of windows will be allocated. Every whitespace will be counted, however only clusters of whitespace that is sufficient to meet the minimum
bound for a window will actually be 'allocated'.

Several variants of this function should exist, one that prioritises timescale and one that prioritises continuous bandwidth (i.e. largest windows possible) as
any particular bias will impact on the resulting spectrum utility and eventual whitespace quality indexation.

Perhaps a window is the wrong way to think about things and really only a relationship between neighbouring frequencies and predictions for the continuity of an
opportunity are the only truely 'objective' measures of whitespace quality.

Anything to do with a utility function, or anything of the like, is a subjective lense for which to view the spectrum, and thus will attempt to carve out slices of
spectrum that best suit it, however this is not really transferrable to other utility functions. A distribution and relationship model may be the answer to this problem
as the subjective 'lense' can be applied to the objective model, but not vice versa. It will provide a likelyness of support, or supportibility or the subjective measure.

Ok, what objective measures are there?

Associtivity graph? or assoc matrix, with probabilities? still haev to make a hard decision though when allocating the windows
so i suppose that really we just have to build the basic algo (lense) and then just generate the resulting window set, hopefully
it looks somewhat intelligible .... I imagine it would however, we still run up against the subjective vs objective problem.

These rants are fun arent they?

We should actually co-generate window sets.

One set will be the discrete window set, and the other will be a continuous window set. And a third, with realtime decision
making: i.e. greedy (initially) - all of these should still use a common weighting scheme also.

Do we want to do bandwidth splitting in the discrete set. For example, as this is somewhat a-priori, we have the flexibility
to perform decisions AFTER the fact. This would be very good to contrast with the greedy set.

I have now reconvinced myself of the same thing i thought of earlier this week. Good.

Lets do it then. After this tripple analysis is done, maybe then we can think about how to perform the general associaitivity relationships.


First algo will be greedy n.n

*/
void detect_greedy(float* ws_frame, const int num_wins, int(* ws_array)[2], std::vector<window>* window_vec, int* frame_num, double* count) {

	//std::cout << "Commence Whitespace Detection" << std::endl;

	//Process the ws
	float* ws_ptr = ws_frame;

	int s_idx, e_idx;
	int bw_count;
	int contribution = 0;
	int lwb_idx, rwb_idx;
	int min;

	int min_bw = 66; //at 131072 with 25MHz BW, 66 ~ 12.5khz
	int min_ts = 10;

	//Need to keep an index of currently active windows. Windows that are closed off, will be 'allocated' - removed from the active list and stored. 
	//There will be no assessment in this algorithm for whether or not the allocated window is the correct choice.

	//This is going to be fundamentally different from the original detection routine. The reason for this is to accommodate future modifications to the program
	//The future modifications will incorporate confidence information to assist the algorithm with determining the optimal window allocation. 

	//const int active_bw = 0;
	const int active_bw = WIN_SAMPS / 10; //To remove the roll off from the edges of the spectrum due to sampling.

	int current_bw = 0;
	int ts_count = 0;
	bool watch = false;
	int win_id = 1;
	int alloc_id = 0;
	int temp = 0;

	for (int m = 0; m < num_wins; m++) { //This will loop as many times as num_wins is specified

		if (m != 0) {
			ws_ptr += WIN_SAMPS;  //set ws_ptr to the beginning of the current whitespace frame (this is used for multiple frames in an array)
			(*frame_num)++;
		}

		temp = (m + 1) % 500;
		if (temp == 0 || m + 1 == num_wins) std::cout << "Detecting " << m + 1 << " of " << num_wins << std::endl;

		//could build this into the main loop.
		for (int i = 0 + active_bw; i < WIN_SAMPS - active_bw; i++) if (ws_ptr[i] == 1) (*count)++;

		//main loops
		for (int i = 0 + active_bw; i < WIN_SAMPS - active_bw; i++) {

			//find new windows
			if (current_bw == 0) s_idx = i;

			if (ws_array[i][0] >= min_ts && ws_array[i][1] == 0) current_bw++; //only count BW for areas of spectrum where there is not currently an observed window

			else if (current_bw >= min_bw) {//window needs to be reserved, as this is greedy it cannot be reallocated, only grown in time
				for (int j = s_idx; j < i; j++) ws_array[j][1] = win_id;
				win_id++;
				current_bw = 0;
			}
			else current_bw = 0;

			if (i == (WIN_SAMPS - active_bw - 1) && current_bw >= min_bw) {//window on the edge of the spectrum, hence the special case u.u
				for (int j = s_idx; j <= i; j++) ws_array[j][1] = win_id;
				win_id++;
				current_bw = 0;
			}
		}

		for (int i = 0 + active_bw; i < WIN_SAMPS - active_bw; i++) {

			//allocate termminated windows
			if (ws_ptr[i] == 0 && ws_array[i][1] != 0) {//allocate window, and update ws_array appropriately

				alloc_id = ws_array[i][1];
				ts_count = ws_array[i][0];
				lwb_idx = i;
				rwb_idx = i;

				while (lwb_idx > 0 + active_bw && ws_array[lwb_idx - 1][1] == alloc_id) lwb_idx--; //Find leftmost window bound
				while (rwb_idx + 1 < WIN_SAMPS - active_bw && ws_array[rwb_idx + 1][1] == alloc_id) rwb_idx++; //Find rightmost window bound

				//update ws_array
				for (int k = lwb_idx; k <= rwb_idx; k++){
				if (ws_array[k][0] < ts_count) ts_count = ws_array[k][0];
					ws_array[k][0] = 0;
					ws_array[k][1] = 0;
				}

				//Calculate BW
				bw_count = rwb_idx - lwb_idx + 1;
				//Calculate contribution (note THIS IS NOT WEIGHTED YET)
				contribution = ts_count * bw_count;

				//Save window
				window_vec->push_back(window(ts_count, bw_count, lwb_idx, contribution, *frame_num)); //TS,BW,F,WS,F_n

				contribution = 0;
				i = rwb_idx;
			}


			//Simultaneous windows will be unique in frequency - there is no overlapping. 
			//ws_array[bin identifier][current timescale,identified window id] could even expand this out with bin specific information, like confidence ...

			//check phase
			//new window identification
			//existing window allocation


		}
		//Once Detection is performed for the "previous" frame, update with current frame.
		for (int i = 0 + active_bw; i < WIN_SAMPS - active_bw; i++)  { //If whitespace does exist at this bin, increment appropriate entry 

			if ((int)ws_ptr[i] != 0) {
				ws_array[i][0]++;
			}
			else ws_array[i][0] = 0;
		}

	}

}


/*********************This is also known as MaxCover********************/