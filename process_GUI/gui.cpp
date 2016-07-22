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
	QCustomPlot* waterfall = ui.plot_1;

	fft_plot->setFocusPolicy(Qt::ClickFocus);
	waterfall->setFocusPolicy(Qt::ClickFocus);

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

	fft_plot->addGraph();
	fft_plot->graph(2);
	fft_plot->graph(2)->setPen(QPen(Qt::green));
	//fft_plot->graph(1)->setBrush(QBrush(QColor(240, 255, 200)));
	fft_plot->graph(2)->setAntialiasedFill(false);

	//fft_plot->yAxis->setScaleType(fft_plot->yAxis->stLogarithmic);
	//fft_plot->yAxis->setScaleLogBase(10);
	fft_plot->yAxis->setRange(-140, 0);
	fft_plot->xAxis->setRange(0, 2048);


	arrowPressFilter* plotfilter = new arrowPressFilter(ui.pushButton, ui.textEdit);
	fft_plot->installEventFilter(plotfilter);

	keyPressFilter* filter = new keyPressFilter(ui.pushButton);
	ui.textEdit->installEventFilter(filter);

	//Waterfall code
		waterfall->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom); // this will also allow rescaling the color scale by dragging/zooming
		waterfall->axisRect()->setupFullAxesBox(true);
		waterfall->xAxis->setLabel("Frequency");
		waterfall->yAxis->setLabel("Time");

		// set up the QCPColorMap:
		colorMap = new QCPColorMap(waterfall->xAxis, waterfall->yAxis);
		waterfall->addPlottable(colorMap);
		int dim = 384;
		colorMap->data()->setSize(dim, dim);
		colorMap->data()->setRange(QCPRange(0, dim), QCPRange(0, dim)); 
	
		// add a color scale:
		colorScale = new QCPColorScale(waterfall);
		waterfall->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
		colorScale->setDataRange(QCPRange(-1, 2));
		colorScale->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
		colorMap->setColorScale(colorScale); // associate the color map with the color scale
		colorScale->axis()->setLabel("Whitespace Density");
		waterfall->plotLayout()->insertRow(0); // inserts an empty row above the default axis rect
		waterfall->plotLayout()->addElement(0, 0, new QCPPlotTitle(waterfall, "Waterfall (131072) -121"));
		colorMap->setGradient(QCPColorGradient::gpHot);

		// make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
		QCPMarginGroup *marginGroup = new QCPMarginGroup(waterfall);
		waterfall->axisRect()->setMarginGroup(QCP::msBottom | QCP::msTop, marginGroup);
		colorScale->setMarginGroup(QCP::msBottom | QCP::msTop, marginGroup);

	//Connect worker thread to gui plotting function
	fft_plot->graph(0)->setData(plotMapAvg, false);
	fft_plot->graph(1)->setData(plotMapMax, false);
	fft_plot->graph(2)->setData(plotMapMin, false);

	qRegisterMetaType<QVector <double> >("QVector <double>");
	connect(&thread, SIGNAL(plotSignal(QVector<double>, QVector<double>, QVector<double>)), this, SLOT(plotSlot(QVector<double>, QVector<double>, QVector<double>)));
	connect(&thread, SIGNAL(fallSignal(QVector<double>, int)), this, SLOT(fallSlot(QVector<double>, int)));
	connect(ui.pushButton, SIGNAL(clicked()), this, SLOT(scanSlot()));
}

void gui::plotSlot(QVector<double> yAvg, QVector<double> yMax, QVector<double> yMin){
	qDebug() << "signal received";
	QCustomPlot* fft_plot = ui.plot_0;

	for (int i = 0; i < yAvg.length(); i++) {
		plotMapAvg->insert((double)i, QCPData((double)(i + 1), yAvg[i]));
		plotMapMax->insert((double)i, QCPData((double)(i + 1), yMax[i]));
		plotMapMin->insert((double)i, QCPData((double)(i + 1), yMin[i]));
	}

	fft_plot->replot();
	qDebug() << "Plot updated!";
}

void gui::fallSlot(QVector<double> vals, int depth) {
	int dim = 384;

	QCustomPlot* waterfall = ui.plot_1;

	for (int yIndex = 0; yIndex < dim; ++yIndex)
	{
		for (int xIndex = 0; xIndex < dim; ++xIndex)
		{
			colorMap->data()->setCell(xIndex, yIndex, vals[xIndex+yIndex*dim]);
		}
	}

	colorMap->rescaleDataRange();

	waterfall->rescaleAxes();
	waterfall->replot();
	QString fname = "Waterfall 131072 -121.png";
	waterfall->savePng(fname, 0, 0, 1, -1);
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