#include "gui.h"

class KeyPressEvent : public QObject
{
	Q_OBJECT
		

protected:
	bool eventFilter(QObject *obj, QEvent *event);
};

bool KeyPressEvent::eventFilter(QObject *obj, QEvent *event)
{
	if (event->type() == QEvent::KeyPress) {
		QKeyEvent *keyEvent = static_cast<QKeyEvent *>(event);
		qDebug("Ate key press %d", keyEvent->key());
		return true;
	}
	else {
		// standard event processing
		return QObject::eventFilter(obj, event);
	}
}