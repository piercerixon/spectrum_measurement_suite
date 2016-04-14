#ifndef KPF_H
#define KPF_H

#include <QtGui>
#include <QtWidgets/QPushButton>

class keyPressFilter : public QObject
{
	Q_OBJECT

public:
	keyPressFilter(QPushButton* button) :QObject(){
		scanButton = button;
	};
	~keyPressFilter(){};

	bool eventFilter(QObject*, QEvent*);

private:
	QPushButton* scanButton;
};

inline bool keyPressFilter::eventFilter(QObject *obj, QEvent *event)
{
	if (event->type() == QEvent::KeyPress) {
		QKeyEvent *keyEvent = static_cast<QKeyEvent *>(event);
		//qDebug("Key press %d", keyEvent->key());
		if (keyEvent->key() >= Qt::Key_0 && keyEvent->key() <= Qt::Key_9 ||
			keyEvent->key() == Qt::Key_Backspace || keyEvent->key() == Qt::Key_Delete ||
			keyEvent->key() == Qt::Key_Left || keyEvent->key() == Qt::Key_Right) {
			return QObject::eventFilter(obj, event);
		}
		else if (keyEvent->key() == Qt::Key_Enter || keyEvent->key() == Qt::Key_Return) {
			scanButton->click();
			return true;
		}
		return true;
	}
	else {
		// standard event processing
		return QObject::eventFilter(obj, event);
	}
}

#endif