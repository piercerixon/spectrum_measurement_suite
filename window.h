#ifndef WINDOW_H
#define WINDOW_H

//whitespace window storage object
class window {
	public:
		int timescale;
		int bandwidth;
		int frequency;
		int whitespace; //could be long long, will need to test experimentally, u_int should be enough
		int frame_no;

		window(int t, int bw, int f, int ws, int n); //constructor
		~window();
};

#endif