#include "process.h"
#include <iostream>
#include <set>
#include <complex>
#include <vector>

void test() {

	const int w_samps = 16;
	const int num_wins = 16;
	unsigned char ws_frame[num_wins][w_samps]; // = {
	//	{1,1,0,0,1,0,0,0,0,0,1,1,1,1,1,1},
	//	{1,0,1,0,1,0,0,1,0,0,1,0,0,1,1,0},
	//	{1,0,1,0,1,1,1,0,1,1,0,1,1,0,1,1},
	//	{1,0,1,0,0,1,1,1,1,1,1,0,0,1,0,0},
	//	{0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,0},
	//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}; 
	
	for(int i = 0; i < num_wins; i++) {
		for(int j = 0; j < w_samps; j++) {
			if(i < num_wins-1) ws_frame[i][j] = std::rand()%2;
			else ws_frame[i][j] = 0;
		}
	}

	int ws_array[w_samps];

	const int TEST_NUM = 2;
	
	std::cout << "\nPrinting Test Array: " << std::endl;
	std::cout << "   0 1 2 3 4 5 6 7 8 9 a b c d e f" << std::endl;
	
	for (int i = 0; i < num_wins; i++) {
		std::cout << i << ": ";
		for (int j = 0; j < w_samps; j++) {	
			std::cout << (int)ws_frame[i][j];
			if(j != w_samps-1) std::cout << ",";
		}
		std::cout << std::endl;
	}
	
	int overlap[w_samps];

	for(int i = 0; i < w_samps; i++) {
		ws_array[i] = (int)ws_frame[0][i];
		overlap[i] = 0;
	}

		if (TEST_NUM == 1) {
			std::vector<window> window_vec;
			detect_once(&ws_frame[0][0], num_wins, ws_array, overlap, &window_vec, 0);

			bar_plot(&window_vec);
		}

		if (TEST_NUM == 2) {
		std::cout << "\nWhitespace Summary Test 2: " << std::endl;
		
		//Process the ws
		unsigned char* ws_ptr = (unsigned char*)&ws_frame[1][0]; //This will only be single dimensional in the actual program
		int overlap[w_samps];
		//unsigned char check_mask[w_samps];

		//HACK - put the first whitespace frame into the whitesapce array
		for(int i = 0; i < w_samps; i++) {
			ws_array[i] = (int)ws_frame[0][i];
			overlap[i] = 0;
		}

		std::vector<int> zero_val_idx;
		std::vector<int> zero_mask;
		std::vector<int> temp_ws_array;
		
		std::set<int>::iterator itr;
		std::set<int> win_vals_set;

		std::vector<window> window_vec;

		int s_idx, e_idx, bw_idx;
		int bw_count, contribution;
		int lwb_idx, rwb_idx;
		int min;

		int maxBW = 0;
		int maxTS = 0;
		int maxWS = 0;

		for(int k = 1; k < num_wins; k++, ws_ptr += w_samps){ //Once a whole window is parsed correctly, increment the pointer by a whole window (to get the next one)

			bw_idx = -1;
			
			for(int i = 0; i < w_samps; i++) {
				
				//Scans through the next frame for a 0 (no ws) that preceeded a previous 1 (ws), upon finding a zero, the associated whitespace opportunity that ended is then analysed 
				if((int)ws_ptr[i] == 0 && ws_array[i] != 0 && i > bw_idx){
					
					s_idx = i;
					e_idx = i;
					temp_ws_array.clear();
					zero_val_idx.clear();
					win_vals_set.clear();
					zero_mask.clear();

					while(s_idx > 0 && ws_array[s_idx-1] != 0) s_idx--; //Find leftmost ws bound
					while(e_idx+1 < w_samps && ws_array[e_idx+1] != 0) e_idx++; //Find rightmost ws bound

					//temp_ws_array = (int*)realloc(temp_ws_array,sizeof(int)*((e_idx-s_idx)+1)); 
					
					for(int j = 0; j < e_idx-s_idx+1; j++) {
						temp_ws_array.push_back(ws_array[s_idx+j]); //Copy detected window to a temporary window
						

						zero_mask.push_back(ws_ptr[s_idx+j]); //Copy zero locations for appropriate window
						if(ws_ptr[s_idx+j] == 0) {
							win_vals_set.insert(ws_array[s_idx+j]);
							zero_val_idx.push_back(j); // Hunt for all zeroes under band - these will form the basis of the window requiring to be computed and are relative to the temp array. 
						}
					}

					for(int j = 0; j < zero_mask.size(); j++){
						itr = win_vals_set.begin();

						if(zero_mask[j] == 0) {
							//std::cout << "zero loc" << k << "," << e_idx << std::endl;
							while(temp_ws_array[j] >= *itr && itr != win_vals_set.end()) {
								
								lwb_idx = j;
								rwb_idx = j;

								//Find Window
								while(lwb_idx > 0 && temp_ws_array[lwb_idx-1] >= *itr) {
									lwb_idx--; //Find leftmost ws bound
									if(temp_ws_array[lwb_idx] == *itr) zero_mask[lwb_idx] = 1;
								}
								while(rwb_idx+1 < temp_ws_array.size() && temp_ws_array[rwb_idx+1] >= *itr) {
									rwb_idx++; //Find rightmost ws bound
									if(temp_ws_array[rwb_idx] == *itr) zero_mask[rwb_idx] = 1;
								}
								
								//Calculate BW
								bw_count = rwb_idx-lwb_idx+1;
								
								if(bw_count == 1){ //If BW is only 1, then the TS is obviously equal to whatever the corresponsing ws_array value is, and does not need to iterate through the set
									itr = win_vals_set.find(temp_ws_array[j]); 
								}

								// Calculate Overlap
								contribution = 0;

								for(int p = lwb_idx; p <= rwb_idx; p++){
									
									min = ((temp_ws_array[p] - overlap[p+s_idx]) >= *itr)?(*itr):(temp_ws_array[p] - overlap[p+s_idx]);
									if(min < 0) min = 0;
									contribution += min;
									overlap[p+s_idx] += min;
								}

								//Save and output result
								
								std::cout << " BW: " << bw_count << " TS: " << *itr << " Con: " << contribution << std::endl; //BW of ws is equal to e_idx-s_idx+1
								
								window temp = window(*itr, bw_count, s_idx, contribution, (k-1));

								if (*itr > maxTS) maxTS = *itr;
								if (bw_count > maxBW) maxBW = bw_count;
								if (contribution > maxWS) maxWS = contribution;

								window_vec.push_back(temp);

								if(bw_count == temp_ws_array.size()) {
									//std::cout << "GET RID OF THIS BRO " << *itr << std::endl;
									win_vals_set.erase(itr);
									itr = win_vals_set.begin(); //Can only do this because sets are ordered, and only the smallest value can possibly span the entire band.
								}else itr++;
							}
						}
					}

					bw_idx = e_idx; //Increment the counter by the width of the total detected window.
				}
			}

			for(int i = 0; i < w_samps; i++) { //If whitespace does exist at this bin, increment appropriate entry 

				overlap[i] *= (int)ws_ptr[i]; //Mask overlap array appropriately

				if((int)ws_ptr[i] != 0) {
					ws_array[i]++; 
				}
				else ws_array[i] *= (int)ws_ptr[i];
			}
		}

		std::cout << "MAX TS: " << maxTS << " MAX BW: " << maxBW << std::endl;

		std::vector<float> BW_s, BWWS_s, TS_s, TSWS_s; //Summaries
		std::vector<float>::iterator BWit, BWWSit, TSit, TSWSit; //Iterators

		std::vector<window>::iterator WINit; //Window iterator

		
		for(WINit = window_vec.begin(); WINit < window_vec.end(); WINit++) {
			//std::cout << "BW: " << WINit->bandwidth << " TS: " << WINit->timescale << " F: " << WINit->frequency << " WS: " << WINit->whitespace << " F_no: " << WINit->frame_no << std::endl;
			
			//Summary of Bandwidths

			BWit = BW_s.begin();
			BWWSit = BWWS_s.begin();

			if(BWit == BW_s.end()) {
				BW_s.push_back((float)WINit->bandwidth);
				BWWS_s.push_back((float)WINit->whitespace);
				
			} else

			while (BWit!=BW_s.end()) {
				if (*BWit == (float)WINit->bandwidth) {
					*BWWSit += (float)WINit->whitespace;
					break;
				} 
				else if (*BWit > (float)WINit->bandwidth) {
					BW_s.insert(BWit,(float)WINit->bandwidth);
					BWWS_s.insert(BWWSit,(float)WINit->whitespace);
					break;
				}
				else if ((BWit+1) == BW_s.end()) {
					BW_s.push_back((float)WINit->bandwidth);
					BWWS_s.push_back((float)WINit->whitespace);
					break;
				}
				++BWit;
				++BWWSit;
			}

			//Summary of Timescales

			TSit = TS_s.begin();
			TSWSit = TSWS_s.begin();

			if(TSit == TS_s.end()) {
				TS_s.push_back((float)WINit->timescale);
				TSWS_s.push_back((float)WINit->whitespace);
				
			} else

			while (TSit!=TS_s.end()) {
				if (*TSit == (float)WINit->timescale) {
					*TSWSit += (float)WINit->whitespace;
					break;
				} 
				else if (*TSit > (float)WINit->timescale) {
					TS_s.insert(TSit,(float)WINit->timescale);
					TSWS_s.insert(TSWSit,(float)WINit->whitespace);
					break;
				}
				else if ((TSit+1) == TS_s.end()) {
					TS_s.push_back((float)WINit->timescale);
					TSWS_s.push_back((float)WINit->whitespace);
					break;
				}
				++TSit;
				++TSWSit;
			}
		}

		std::cout << "BW Summary: ";
		print_vector(BW_s);
		std::cout << "BW WS Summary: ";
		print_vector(BWWS_s);

		std::cout << "TS Summary: ";
		print_vector(TS_s);
		std::cout << "TS WS Summary: ";
		print_vector(TSWS_s);

		mglData mglBW, mglBWws, mglTS, mglTSws;

		mglBW.Set(BW_s);
		mglBWws.Set(BWWS_s);
		mglTS.Set(TS_s);
		mglTSws.Set(TSWS_s);

		//std::vector<window>::iterator it;
		//for(it = window_vec.begin(); it < window_vec.end(); it++) {
		//	std::cout << "BW: " << it->bandwidth << " TS: " << it->timescale << " F: " << it->frequency << " WS: " << it->whitespace << " F_no: " << it->frame_no << std::endl;
			
			//mglBW.a[((it->bandwidth)*arrSize)/maxBW] += it->whitespace;
			//mglTS.a[((it->timescale)*arrSize)/maxTS] += it->whitespace;
			//it->whitespace;
		//}

		std::vector<float> BW, TS, WS;



		mglData BWdat, TSdat, WSdat;

		mglFLTK *gr = new mglFLTK("Whitespace Summary"); //Yes the window object has to be spawned and then referenced within this thread. 
		//mglData xx=gr->Hist((mglData)*mglBW->a,(mglData)*mglBWws->a), yy=gr->Hist((mglData)*mglTS->a,(mglData)*mglTSws->a);
		//mglData xx=gr->Hist(mglBW,mglBWws), yy=gr->Hist((mglData)*mglTS->a,(mglData)*mglTSws->a);	
		//xx.Norm(0,1); yy.Norm(0,1);

		//gr->MultiPlot(3,3,3,2,2,"");   gr->SetRanges(-1,1,-1,1,0,1);
		//gr->Axis("xy"); gr->Dots(x,y,z,"wyrRk"); gr->Box();

		/******
		gr->SetOrigin(0,0);
		gr->MultiPlot(3,3,0,2,1,"");   gr->SetRanges(0,maxBW+1,0,100);
		gr->Box(); gr->Axis("xy"); gr->Bars(mglBW,mglBWws);
		gr->MultiPlot(3,3,5,1,2,"");   gr->SetRanges(0,100,0,maxTS+1);
		gr->Box(); gr->Axis("xy"); gr->Barh(mglTSws,mglTS);
		gr->SubPlot(3,3,2);

		******/

		gr->SetOrigin(0,0);
		gr->Axis("xy");
		gr->Box();
		gr->SetRanges(0,maxTS+1,0,maxBW+1);
		gr->Dots(TSdat,BWdat,WSdat);
		//gr->Puts(mglPoint(0.5,0.5),"Hist and\nMultiPlot\nsample","a",-6);
		gr->Run();

	} // End Test 2
	
} // End Tests