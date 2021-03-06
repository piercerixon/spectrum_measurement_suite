
/*
void fftf_proc (std::complex<float>* buff, int idx, mglData *mglY) {

boost::unique_lock<boost::mutex> lock(fft_mutex[idx]);
//Prepare your body for the FFT
//CONVENTION fftwf = floats, instead of doubles used before. AS we only have 14 bits of precision it doesnt make sense to use double point precision
// but im using doubles anyway :D


fftwf_complex *fft_buff;
fftwf_plan plan;
float win_power = 0;
float *w_array, *avg_buff;
int avg_count = 0;
const char* filename = "fftwf_plan";

int frame_count = 0;

int error_counter = 0;

const int NUM_WINS = 1;

int ws_array[WIN_SAMPS];
unsigned char ws_frame[WIN_SAMPS];
int overlap[WIN_SAMPS];
for(int i = 0; i < WIN_SAMPS; i++){
ws_array[i] = 0;
ws_frame[i] = 0;
overlap[i] = 0;
}
//long long ws_count = 0;
std::vector<window> window_vec;
std::vector<window>* w_vec_ptr = &window_vec;
int frame_number = 0;
bool init = false;

fft_buff = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * WIN_SAMPS); //buffer allocated for Output
w_array = (float*) malloc(sizeof(float) * WIN_SAMPS); //allocate memory for window coefficients

avg_buff = (float*) malloc(sizeof(float) * WIN_SAMPS); //buffer allocated for averaging FFT results

//Cast the std::complex<double> as a fftw_complex
if (!fftwf_import_wisdom_from_filename(filename)) {
std::cout << "should use wisdom next time :)" << std::endl;
plan = fftwf_plan_dft_1d(WIN_SAMPS, reinterpret_cast<fftwf_complex*>(&buff[0]), fft_buff, FFTW_FORWARD, FFTW_MEASURE);
std::cout << "FFTW PLAN: " << fftwf_export_wisdom_to_filename(filename) << std::endl;
}
//plan = fftwf_plan_dft_1d(WIN_SAMPS, reinterpret_cast<fftwf_complex*>(&buff[0]), fft_buff, FFTW_FORWARD, FFTW_PATIENT);
else plan = fftwf_plan_dft_1d(WIN_SAMPS, reinterpret_cast<fftwf_complex*>(&buff[0]), fft_buff, FFTW_FORWARD, FFTW_MEASURE);

//Create coefficient array and x axis index for plotting
for (int i = 0; i < WIN_SAMPS; i++) {
w_array[i] = 0.35875 - 0.48829*cos(2*pi*i/(WIN_SAMPS-1)) + 0.14128*cos(4*pi*i/(WIN_SAMPS-1)) - 0.01168*cos(6*pi*i/(WIN_SAMPS-1)); //blackmann harris window
win_power += (w_array[i]*w_array[i]); //this computes the total window power and normalises it to account for DC gain due to the window.
}
win_power /= WIN_SAMPS; //normalise the total window power across each sample.

//std::cout << -10*std::log10(win_power) << std::endl;
//double offset = (- 10*std::log10(win_power) //DC gain of window
//				 - (174 - 10*std::log10(bandwidth/samp_rate))); //Noise floor
float offset = - 10 - rx_gain; // - 10*std::log10(win_power); //-10 is the MAX power detected by the ADC and take into account the gain of the frontend.

std::cout << "FFT[" << idx << "]: Spawned" << std::endl;
if(LOG) logfile << "FFT[" << idx << "]: Initialised and Sleeping\n";

super_cond.notify_one();
fft_cond.wait(lock);

std::ofstream fftout;
fftout.open ("fftout.csv");

while(true) {
if(LOG) logfile << "FFT[" << idx << "]: Awake ";
boost::system_time now = boost::get_system_time();

//Apply window to samples
//if ((int)(*num_rx_samps) == WIN_SAMPS*NUM_THREADS) {
for (int i = 0; i < WIN_SAMPS; i++) {
buff[i] *= w_array[i];
}

//Spin the FFT yoooooooooo
fftwf_execute(plan);

//Keep a moving average of every FFT that has happened until the averaging says so :D
//Probably need to optimise this better. Pointer arithmetic is probably the go.
//#pragma omp parallel for

for (int i = 0; i < WIN_SAMPS; i++) {
avg_buff[i] += (
10*std::log10(std::abs(fft_buff[i][0]*fft_buff[i][0] + fft_buff[i][1]*fft_buff[i][1])/WIN_SAMPS) //DFT bin magnitude
);
}

//Count the number of FFT frames that have been processed thus far
avg_count++;

//Perform ws detection

if (avg_count%10 == 0 || endfile) {
//frame_count++;

for(int i = 0; i < WIN_SAMPS; i++) {

if(!endfile) {
if((avg_buff[((WIN_SAMPS/2)+i)%WIN_SAMPS]/avg_count + offset) > threshold) ws_frame[i] = 0; //whitespace does not exist
else {
ws_frame[i] = 1; //whitespace exists
ws_count++;
}
}
else ws_frame[i] = 0; //Set final frame to all 0's to assist with correct window detection.
}

if(COMPARE) {
//std::cout << "Comparing" << std::endl;
bool error = false;
int temp = 0;

for(int i = 0; i < WIN_SAMPS; i++) {
if(GA[frame_number][i] != ws_frame[i]) {
if(error) {
std::cout << "TEST: " << frame_number << "," << i;
temp = i;
}
error = false;
error_counter++;
if(GA[frame_number][i] == 1) fftout << 3 << ',';
else if(GA[frame_number][i] == 0) fftout << 4 << ',';
else fftout << 7 << ',';
}
else {
if(!error && temp > 0) std::cout << "TEST: Num Err: " << i - temp << std::endl;
fftout << (int)ws_frame[i] << ',';
error = true;
}
}
fftout << "\n";
//if(error) system("Pause");

} else {
for(int i = 0; i < WIN_SAMPS; i++) {
fftout << (int)ws_frame[i] << ',';
}
fftout << "\n";
}

if(init) detect_dev(ws_frame, NUM_WINS, ws_array, overlap, w_vec_ptr, frame_number);
else {
for(int i = 0; i < WIN_SAMPS; i++) {
ws_array[i] = (int)ws_frame[i]; //This is done because the first entry in ws_array has to be initialised with the first frame
}
init = true;
}

frame_number++;
//	for (int i = 0; i < WIN_SAMPS; i++) {
//		std::cout << (int)ws_frame[i] << ',';
//	}



//if(avg_count%10 == 0) {
// DISPLAY CODE, got to flip the buffer around correctly !!!!

//display the averaged samples
if (idx == 0 && display_mtx.try_lock() && DISPLAY) { //ENTER CRITICAL REGION, will not block the FFT thread.
if(LOG) logfile << "FFT[" << idx << "]: Displaying ";

std::cout << "Updating frame with averages: " << avg_count << std::endl;

int step = WIN_SAMPS/num_disp_pts;
float val_max, val_min = 0;
float temp = 0;
int index = 0;

for (int i = 0; i < num_disp_pts; i++) {
index = i*step;
val_max = -1000;
val_min = 0;
temp = -1000; //If you actually get a level lower than this, you have issues :)

for (int j = 0; j < step; j++) {//Grab the max and min values from each step set
temp = (avg_buff[index + j]/avg_count);

if(temp > val_max) {val_max = temp;} //Take the MAX value
else if(temp < val_min) {val_min = temp;} //Else Take the MIN value

//avg_buff[index + j] = 0;

//std::cout << "point: " << ((WIN_SAMPS/2)+i)%WIN_SAMPS << " val: " << (avg_buff[i] / avg_count) << std::endl;
}

val_max += offset;
mglY->a[((num_disp_pts/2)+i)%num_disp_pts] = val_max;
}

display_mtx.unlock();
disp_cond.notify_one();
}
//Clear the average once it has been processed, to prepare it for the next set.
for(int i = 0; i < WIN_SAMPS; i++) avg_buff[i] = 0;
avg_count = 0;
}

//This is here to provide the final window detection and outputting of the window vector to file.
if(endfile) {
std::cout << "\nFFT["<< idx << "]: Error Count: " << error_counter << std::endl;

std::ofstream window_dump;
window_dump.open ("Window_dump.csv");
window_dump << 	"timescale,frequency,bandwidth,whitespace,frame_no\n";
std::vector<window>::iterator WINit; //Window iterator
std::cout << "FFT["<< idx << "]: Outputting Whitespace Windows" << std::endl;
for(WINit = window_vec.begin(); WINit < window_vec.end(); WINit++) {
window_dump << WINit->timescale << "," << WINit->frequency << "," << WINit->bandwidth << "," << WINit->whitespace << "," << WINit->frame_no << "\n";
}
std::cout << "FFT["<< idx << "]: Output Complete" << std::endl;
window_dump.flush();
window_dump.close();
fftout.flush();
fftout.close();

bar_plot(&window_vec); //If the end of the data has been reached, then display the data
break;
}

//Once all done, go back to sleep
if(LOG) logfile << "FFT[" << idx << "]: Sleeping ";
super_cond.notify_one();
fft_cond.wait(lock);
}

std::cout << "FFT["<< idx << "]: Terminating" << std::endl;
//destroy fft plan when all the fun is over
fftwf_destroy_plan(plan);
fftwf_free(fft_buff);
fftwf_cleanup_threads();
free(w_array);
}
*/

/*
void call_py_plot() {

if (PyCallable_Check(pFunc)) {

npy_intp dims[1] = { 6 }; //this would be NUM_SAMPS
float x[131072] = {0,1,20,20,1,60};

std::cout << "Cast array\n";
pArg = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, reinterpret_cast<void*>(x));
pArray = reinterpret_cast<PyArrayObject*>(pArg);
std::cout << "Call Python\n";

PyObject *pTup, *pScale, *pId, *pTotal;

pScale = PyLong_FromDouble(5);
pId = PyLong_FromDouble(0);
pTotal = PyLong_FromDouble(26);

pTup = PyTuple_New(4);
PyTuple_SET_ITEM(pTup, 0, pArg);
PyTuple_SET_ITEM(pTup, 1, pScale);
PyTuple_SET_ITEM(pTup, 2, pId);
PyTuple_SET_ITEM(pTup, 3, pTotal);

PyGILState_STATE gstate = PyGILState_Ensure();

PyObject_CallObject(pFunc, pTup);

//Modifying parameters
x[3] = 50;
PyTuple_SET_ITEM(pTup,2,PyLong_FromDouble(1));
PyTuple_SET_ITEM(pTup, 3, PyLong_FromDouble(90));
PyObject_CallObject(pFunc, pTup);

PyErr_Print();
}
else {
std::cout << "Plotting function unavailable\n\nTerminating\n";
system("pause");
exit(0);
}


}
*/