/********************************************************************************
** Form generated from reading UI file 'gui.ui'
**
** Created by: Qt User Interface Compiler version 5.5.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GUI_H
#define UI_GUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>
#include "qcustomplot.h"

QT_BEGIN_NAMESPACE

class Ui_guiClass
{
public:
    QAction *actionExit;
    QWidget *centralWidget;
    QCustomPlot *plot_0;
    QMenuBar *menuBar;
    QMenu *menuOptions;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *guiClass)
    {
        if (guiClass->objectName().isEmpty())
            guiClass->setObjectName(QStringLiteral("guiClass"));
        guiClass->resize(600, 400);
        actionExit = new QAction(guiClass);
        actionExit->setObjectName(QStringLiteral("actionExit"));
        centralWidget = new QWidget(guiClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        plot_0 = new QCustomPlot(centralWidget);
        plot_0->setObjectName(QStringLiteral("plot_0"));
        plot_0->setGeometry(QRect(120, 40, 361, 251));
        guiClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(guiClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 600, 21));
        menuOptions = new QMenu(menuBar);
        menuOptions->setObjectName(QStringLiteral("menuOptions"));
        guiClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(guiClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        guiClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(guiClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        guiClass->setStatusBar(statusBar);

        menuBar->addAction(menuOptions->menuAction());
        menuOptions->addAction(actionExit);

        retranslateUi(guiClass);
        QObject::connect(actionExit, SIGNAL(triggered()), guiClass, SLOT(close()));

        QMetaObject::connectSlotsByName(guiClass);
    } // setupUi

    void retranslateUi(QMainWindow *guiClass)
    {
        guiClass->setWindowTitle(QApplication::translate("guiClass", "gui", 0));
        actionExit->setText(QApplication::translate("guiClass", "Exit", 0));
        menuOptions->setTitle(QApplication::translate("guiClass", "Options", 0));
    } // retranslateUi

};

namespace Ui {
    class guiClass: public Ui_guiClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GUI_H
