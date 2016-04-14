#ifndef APF_H
#define APF_H

#include <QtGui>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTextEdit>

#include <boost/lexical_cast.hpp>

class arrowPressFilter : public QObject
{
	Q_OBJECT

public:
	arrowPressFilter(QPushButton* button, QTextEdit* box) :QObject(){
		scanButton = button;
		textBox = box;
	};
	~arrowPressFilter(){};

	bool eventFilter(QObject*, QEvent*);

private:
	QPushButton* scanButton;
	QTextEdit* textBox;
};

inline bool arrowPressFilter::eventFilter(QObject *obj, QEvent *event)
{

	QString frameStr = textBox->toPlainText();
	bool valid;

	int frameInt = frameStr.toInt(&valid, 10);

	if (event->type() == QEvent::KeyPress) {
		QKeyEvent *keyEvent = static_cast<QKeyEvent *>(event);
		//qDebug("Key press %d", keyEvent->key());
		if (keyEvent->key() == Qt::Key_Left){
			if (valid){
				textBox->setText(QString::fromStdString(boost::lexical_cast<std::string>(--frameInt)));
				scanButton->click();
			}
			return true;
		}
		else if (keyEvent->key() == Qt::Key_Right) {
			if (valid){
				textBox->setText(QString::fromStdString(boost::lexical_cast<std::string>(++frameInt)));
				scanButton->click();
			}
			return true;
		}
	}
	else {
		// standard event processing
		return QObject::eventFilter(obj, event);
	}
}

#endif