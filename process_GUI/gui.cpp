#include "gui.h"
#include "keyPressFilter.h"
#include "arrowPressFilter.h"
#include <boost/lexical_cast.hpp>

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

	fft_plot->setFocusPolicy(Qt::ClickFocus);

	fft_plot->addGraph();
	fft_plot->graph(0);
	fft_plot->graph(0)->setPen(QPen(Qt::blue));
	//fft_plot->graph(0)->setBrush(QBrush(QColor(240, 255, 200)));
	fft_plot->graph(0)->setAntialiasedFill(false);

	fft_plot->addGraph();
	fft_plot->graph(1);
	fft_plot->graph(1)->setPen(QPen(Qt::red));
	//fft_plot->graph(1)->setBrush(QBrush(QColor(240, 255, 200)));
	fft_plot->graph(1)->setAntialiasedFill(false);

	//fft_plot->yAxis->setScaleType(fft_plot->yAxis->stLogarithmic);
	//fft_plot->yAxis->setScaleLogBase(10);
	fft_plot->yAxis->setRange(-140, 0);
	fft_plot->xAxis->setRange(0, 2048);


	arrowPressFilter* plotfilter = new arrowPressFilter(ui.pushButton, ui.textEdit);
	fft_plot->installEventFilter(plotfilter);

	keyPressFilter* filter = new keyPressFilter(ui.pushButton);
	ui.textEdit->installEventFilter(filter);


	//Connect worker thread to gui plotting function
	fft_plot->graph(0)->setData(plotMapAvg, false);
	fft_plot->graph(1)->setData(plotMapMax, false);

	qRegisterMetaType<QVector <double> >("QVector <double>");
	connect(&thread, SIGNAL(plotSignal(QVector<double>, QVector<double>)), this, SLOT(plotSlot(QVector<double>, QVector<double>)));
	connect(ui.pushButton, SIGNAL(clicked()), this, SLOT(scanSlot()));
}

void gui::plotSlot(QVector<double> yAvg, QVector<double> yMax){
	qDebug() << "signal received";
	QCustomPlot* fft_plot = ui.plot_0;

	for (int i = 0; i < yAvg.length(); i++) {
		plotMapAvg->insert((double)i, QCPData((double)(i + 1), yAvg[i]));
		plotMapMax->insert((double)i, QCPData((double)(i + 1), yMax[i]));
	}

	fft_plot->replot();
	qDebug() << "Plot updated!";
}

void gui::scanSlot() {

	QString frameStr = ui.textEdit->toPlainText();
	bool valid;

	int frameInt = frameStr.toInt(&valid, 10);

	if (valid){
		//qDebug() << frameStr;
		ui.textEdit->setText(QString::fromStdString(boost::lexical_cast<std::string>(frameInt)));
		thread.requestUpdate(frameInt);
	}
	else qDebug() << "Input is not a number";
	
	//This should then invoke an update

}