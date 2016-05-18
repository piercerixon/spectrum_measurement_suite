#ifndef GUI_H
#define GUI_H

#include <QtWidgets/QMainWindow>
#include "ui_gui.h"
#include "procThread.h"

class gui : public QMainWindow
{
	Q_OBJECT

public:
	gui(QWidget *parent = 0);
	~gui();

private slots:
	void plotSlot(QVector<double>, QVector<double>, QVector<double>);
	void scanSlot();
	void fallSlot(QVector<double>, int);

private:
	void initUi();
	Ui::guiClass ui;
	procThread thread;
	QCPDataMap* plotMapAvg = new QCPDataMap;
	QCPDataMap* plotMapMax = new QCPDataMap;
	QCPDataMap* plotMapMin = new QCPDataMap;

	QCPColorScale *colorScale;
	QCPColorMap *colorMap;
};

#endif // GUI_H
