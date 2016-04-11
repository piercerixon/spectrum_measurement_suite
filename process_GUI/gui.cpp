#include "gui.h"

gui::gui(QWidget *parent)
	: QMainWindow(parent)
{

	//wait for signal from process thread before calling ui.setup?

	ui.setupUi(this);
	initUi();
}

gui::~gui()
{
	thread.terminate(); //Could be more graceful
}

void gui::initUi() {

	QCustomPlot* fft_plot = ui.plot_0;

	fft_plot->addGraph();
	fft_plot->graph(0);
	fft_plot->graph(0)->setPen(QPen(Qt::blue));
	fft_plot->graph(0)->setBrush(QBrush(QColor(240, 255, 200)));
	fft_plot->graph(0)->setAntialiasedFill(false);

	fft_plot->addGraph();
	fft_plot->graph(1);
	fft_plot->graph(1)->setPen(QPen(Qt::red));
	//fft_plot->graph(1)->setBrush(QBrush(QColor(240, 255, 200)));
	fft_plot->graph(1)->setAntialiasedFill(false);

	fft_plot->yAxis->setScaleType(fft_plot->yAxis->stLogarithmic);
	fft_plot->yAxis->setScaleLogBase(10);
	fft_plot->yAxis->setRange(0, 1e8);

	//Connect worker thread to gui plotting function
	connect(&thread, SIGNAL(plotSignal(QVector<double>, QVector<double>)), this, SLOT(plotSlot(QVector<double>, QVector<double>)));
}

void gui::plotSlot(QVector<double> yAvg, QVector<double> yMax){
	QCustomPlot* fft_plot = ui.plot_0;

	QVector <double> x(yAvg.length());
	for (int i = 0; i < yAvg.length(); i++) x[i] = i+1;
	
	fft_plot->graph(0)->setData(x, yAvg);
	fft_plot->graph(1)->setData(x, yMax);
	fft_plot->replot();
}