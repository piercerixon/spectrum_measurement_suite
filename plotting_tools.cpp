#include "process.h"
#include <iostream>
#include <vector>

void bar_plot(std::vector<window>* window_vec) {
	std::vector<float> BW_s, BWWS_s, TS_s, TSWS_s; //Summaries
	std::vector<float>::iterator BWit, BWWSit, TSit, TSWSit; //Iterators

	int maxBW = 0, maxTS = 0, maxWS = 0;
	long long recorded_count = 0;

	std::vector<window>::iterator WINit; //Window iterator
	for(WINit = window_vec->begin(); WINit < window_vec->end(); WINit++) {
		//std::cout << "BW: " << WINit->bandwidth << " TS: " << WINit->timescale << " F: " << WINit->frequency << " WS: " << WINit->whitespace << " F_no: " << WINit->frame_no << std::endl;
		
		recorded_count += WINit->whitespace;	
		//Summary of Bandwidths

		BWit = BW_s.begin();
		BWWSit = BWWS_s.begin();

		if(BWit == BW_s.end()) {
			BW_s.push_back((float)WINit->bandwidth);
			BWWS_s.push_back((float)WINit->whitespace);
				
		} else

		while (BWit!=BW_s.end()) {

			if ((int)WINit->timescale > maxTS) maxTS = WINit->timescale;
			if ((int)WINit->bandwidth > maxBW) maxBW = WINit->bandwidth;
			if ((int)WINit->whitespace > maxWS) maxWS = WINit->whitespace;

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

	
	bool truecount = false;
	if(false == recorded_count) truecount = true;
	std::cout << "FFT ws_count: " << "WS COUNT SHOULD BE HERE, BUT IT ISNT, ITS BROKEN LOL" << " recorded windows: " << recorded_count << " equal? " << truecount << std::endl;
	//system("Pause");

	//std::cout << "BW Summary: ";
	//print_vector(BW_s);
	//std::cout << "BW WS Summary: ";
	//print_vector(BWWS_s);

	//std::cout << "TS Summary: ";
	//print_vector(TS_s);
	//std::cout << "TS WS Summary: ";
	//print_vector(TSWS_s);

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

	mglFLTK *gr = new mglFLTK("Power Spectrum"); //Yes the window object has to be spawned and then referenced within this thread. 
	//mglData xx=gr->Hist((mglData)*mglBW->a,(mglData)*mglBWws->a), yy=gr->Hist((mglData)*mglTS->a,(mglData)*mglTSws->a);
	//mglData xx=gr->Hist(mglBW,mglBWws), yy=gr->Hist((mglData)*mglTS->a,(mglData)*mglTSws->a);	
	//xx.Norm(0,1); yy.Norm(0,1);
	//gr->MultiPlot(3,3,3,2,2,"");   gr->SetRanges(-1,1,-1,1,0,1);
	//gr->Axis("xy"); gr->Dots(x,y,z,"wyrRk"); gr->Box();
	gr->SetOrigin(0,0);
	//gr->MultiPlot(3,3,0,3,1,"");   gr->SetRanges(0,maxBW,0,maxWS);
	//gr->Box(); gr->Axis("xy"); gr->Bars(mglBW,mglBWws);
	//gr->MultiPlot(3,3,5,1,2,"");   gr->SetRanges(0,maxWS,0,maxTS);
	//gr->Box(); gr->Axis("xy"); gr->Barh(mglTS,mglTSws);
	gr->MultiPlot(2,3,0,2,1,"");   gr->SetRanges(0,maxBW,0,maxWS);
	gr->Box(); gr->Axis("xy"); gr->Bars(mglBW,mglBWws);
	gr->Label('x',"Bandwidth",1); gr->Label('y', "Bandwidth Whitespace",0);
	gr->MultiPlot(2,3,2,2,1,"");
	gr->SetRanges(0,200,0,3e7);
	gr->Box(); gr->Axis("xy"); gr->Bars(mglTS,mglTSws);
	gr->Label('x',"Timescale",1); gr->Label('y', "Timescale Whitespace",0);
	gr->SubPlot(3,3,2);
	//gr->Puts(mglPoint(0.5,0.5),"Hist and\nMultiPlot\nsample","a",-6);
	gr->Run();
}
