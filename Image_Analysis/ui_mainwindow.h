/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 6.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QLabel *title_info;
    QPushButton *loadButton;
    QLineEdit *filePathInfo;
    QGraphicsView *view1;
    QGraphicsView *view2;
    QLabel *image_w;
    QLabel *image_h;
    QLabel *image_c;
    QLabel *image_w_val;
    QLabel *image_h_val;
    QLabel *image_c_val;
    QSlider *Slider_screentone_size;
    QLabel *label;
    QRadioButton *radioBtn_0;
    QRadioButton *radioBtn_1;
    QPushButton *result;
    QCheckBox *option_check;
    QLabel *label_2;
    QSlider *Slider_screentone_gap;
    QLabel *screenton_size;
    QLabel *screenton_gap;
    QCheckBox *bwrev_btn;
    QLabel *label_4;
    QSlider *image_threshold;
    QLabel *image_threshold_val;
    QLabel *label_3;
    QRadioButton *radioBtn_2;
    QRadioButton *radioBtn_3;
    QProgressBar *progressBar;
    QLabel *label_5;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(970, 653);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        title_info = new QLabel(centralwidget);
        title_info->setObjectName("title_info");
        title_info->setGeometry(QRect(20, 20, 71, 31));
        title_info->setAlignment(Qt::AlignmentFlag::AlignCenter);
        loadButton = new QPushButton(centralwidget);
        loadButton->setObjectName("loadButton");
        loadButton->setGeometry(QRect(670, 10, 100, 51));
        filePathInfo = new QLineEdit(centralwidget);
        filePathInfo->setObjectName("filePathInfo");
        filePathInfo->setGeometry(QRect(130, 20, 521, 31));
        QFont font;
        font.setItalic(true);
        filePathInfo->setFont(font);
        view1 = new QGraphicsView(centralwidget);
        view1->setObjectName("view1");
        view1->setGeometry(QRect(20, 80, 371, 461));
        view2 = new QGraphicsView(centralwidget);
        view2->setObjectName("view2");
        view2->setGeometry(QRect(400, 80, 371, 461));
        image_w = new QLabel(centralwidget);
        image_w->setObjectName("image_w");
        image_w->setGeometry(QRect(800, 10, 58, 16));
        image_w->setAlignment(Qt::AlignmentFlag::AlignLeading|Qt::AlignmentFlag::AlignLeft|Qt::AlignmentFlag::AlignVCenter);
        image_h = new QLabel(centralwidget);
        image_h->setObjectName("image_h");
        image_h->setGeometry(QRect(800, 30, 58, 16));
        image_c = new QLabel(centralwidget);
        image_c->setObjectName("image_c");
        image_c->setGeometry(QRect(800, 50, 58, 16));
        image_w_val = new QLabel(centralwidget);
        image_w_val->setObjectName("image_w_val");
        image_w_val->setGeometry(QRect(860, 10, 58, 16));
        QFont font1;
        font1.setBold(true);
        image_w_val->setFont(font1);
        image_w_val->setAlignment(Qt::AlignmentFlag::AlignCenter);
        image_h_val = new QLabel(centralwidget);
        image_h_val->setObjectName("image_h_val");
        image_h_val->setGeometry(QRect(860, 30, 58, 16));
        image_h_val->setFont(font1);
        image_h_val->setAlignment(Qt::AlignmentFlag::AlignCenter);
        image_c_val = new QLabel(centralwidget);
        image_c_val->setObjectName("image_c_val");
        image_c_val->setGeometry(QRect(860, 50, 58, 16));
        image_c_val->setFont(font1);
        image_c_val->setAlignment(Qt::AlignmentFlag::AlignCenter);
        Slider_screentone_size = new QSlider(centralwidget);
        Slider_screentone_size->setObjectName("Slider_screentone_size");
        Slider_screentone_size->setGeometry(QRect(790, 170, 111, 25));
        Slider_screentone_size->setOrientation(Qt::Orientation::Horizontal);
        label = new QLabel(centralwidget);
        label->setObjectName("label");
        label->setGeometry(QRect(790, 140, 58, 16));
        QFont font2;
        font2.setPointSize(14);
        font2.setBold(true);
        font2.setUnderline(false);
        label->setFont(font2);
        radioBtn_0 = new QRadioButton(centralwidget);
        radioBtn_0->setObjectName("radioBtn_0");
        radioBtn_0->setGeometry(QRect(790, 80, 99, 20));
        radioBtn_1 = new QRadioButton(centralwidget);
        radioBtn_1->setObjectName("radioBtn_1");
        radioBtn_1->setGeometry(QRect(790, 110, 99, 20));
        result = new QPushButton(centralwidget);
        result->setObjectName("result");
        result->setGeometry(QRect(790, 510, 161, 61));
        option_check = new QCheckBox(centralwidget);
        option_check->setObjectName("option_check");
        option_check->setGeometry(QRect(790, 480, 151, 20));
        label_2 = new QLabel(centralwidget);
        label_2->setObjectName("label_2");
        label_2->setGeometry(QRect(800, 200, 58, 16));
        Slider_screentone_gap = new QSlider(centralwidget);
        Slider_screentone_gap->setObjectName("Slider_screentone_gap");
        Slider_screentone_gap->setGeometry(QRect(790, 230, 111, 25));
        Slider_screentone_gap->setOrientation(Qt::Orientation::Horizontal);
        screenton_size = new QLabel(centralwidget);
        screenton_size->setObjectName("screenton_size");
        screenton_size->setGeometry(QRect(910, 170, 41, 20));
        screenton_size->setFont(font1);
        screenton_size->setAlignment(Qt::AlignmentFlag::AlignCenter);
        screenton_gap = new QLabel(centralwidget);
        screenton_gap->setObjectName("screenton_gap");
        screenton_gap->setGeometry(QRect(910, 230, 41, 20));
        screenton_gap->setFont(font1);
        screenton_gap->setAlignment(Qt::AlignmentFlag::AlignCenter);
        bwrev_btn = new QCheckBox(centralwidget);
        bwrev_btn->setObjectName("bwrev_btn");
        bwrev_btn->setGeometry(QRect(790, 380, 151, 20));
        label_4 = new QLabel(centralwidget);
        label_4->setObjectName("label_4");
        label_4->setGeometry(QRect(790, 410, 58, 16));
        QFont font3;
        font3.setPointSize(14);
        font3.setBold(true);
        label_4->setFont(font3);
        image_threshold = new QSlider(centralwidget);
        image_threshold->setObjectName("image_threshold");
        image_threshold->setGeometry(QRect(790, 440, 111, 25));
        image_threshold->setOrientation(Qt::Orientation::Horizontal);
        image_threshold_val = new QLabel(centralwidget);
        image_threshold_val->setObjectName("image_threshold_val");
        image_threshold_val->setGeometry(QRect(910, 440, 41, 20));
        image_threshold_val->setFont(font1);
        image_threshold_val->setAlignment(Qt::AlignmentFlag::AlignCenter);
        label_3 = new QLabel(centralwidget);
        label_3->setObjectName("label_3");
        label_3->setGeometry(QRect(790, 260, 58, 16));
        radioBtn_2 = new QRadioButton(centralwidget);
        radioBtn_2->setObjectName("radioBtn_2");
        radioBtn_2->setGeometry(QRect(790, 280, 99, 20));
        radioBtn_3 = new QRadioButton(centralwidget);
        radioBtn_3->setObjectName("radioBtn_3");
        radioBtn_3->setGeometry(QRect(790, 310, 99, 20));
        progressBar = new QProgressBar(centralwidget);
        progressBar->setObjectName("progressBar");
        progressBar->setGeometry(QRect(20, 550, 751, 23));
        progressBar->setValue(24);
        label_5 = new QLabel(centralwidget);
        label_5->setObjectName("label_5");
        label_5->setGeometry(QRect(790, 360, 121, 16));
        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName("menubar");
        menubar->setGeometry(QRect(0, 0, 970, 43));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName("statusbar");
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        title_info->setText(QCoreApplication::translate("MainWindow", "\346\252\224\346\241\210\350\267\257\345\276\221", nullptr));
        loadButton->setText(QCoreApplication::translate("MainWindow", "\351\201\270\346\223\207\346\252\224\346\241\210", nullptr));
        image_w->setText(QCoreApplication::translate("MainWindow", "Width", nullptr));
        image_h->setText(QCoreApplication::translate("MainWindow", "Height", nullptr));
        image_c->setText(QCoreApplication::translate("MainWindow", "Type", nullptr));
        image_w_val->setText(QCoreApplication::translate("MainWindow", "-", nullptr));
        image_h_val->setText(QCoreApplication::translate("MainWindow", "-", nullptr));
        image_c_val->setText(QCoreApplication::translate("MainWindow", "-", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "\347\266\262\351\273\236\345\244\247\345\260\217", nullptr));
        radioBtn_0->setText(QCoreApplication::translate("MainWindow", "\345\234\223\345\275\242\347\266\262\351\273\236", nullptr));
        radioBtn_1->setText(QCoreApplication::translate("MainWindow", "\346\226\271\345\275\242\347\266\262\351\273\236", nullptr));
        result->setText(QCoreApplication::translate("MainWindow", "\345\237\267\350\241\214\351\201\213\347\256\227", nullptr));
        option_check->setText(QCoreApplication::translate("MainWindow", "\350\274\270\345\207\272\346\252\224\346\241\210", nullptr));
        label_2->setText(QCoreApplication::translate("MainWindow", "\347\266\262\351\273\236\351\226\223\351\232\224", nullptr));
        screenton_size->setText(QCoreApplication::translate("MainWindow", "1", nullptr));
        screenton_gap->setText(QCoreApplication::translate("MainWindow", "1", nullptr));
        bwrev_btn->setText(QCoreApplication::translate("MainWindow", "\350\207\252\345\213\225\351\273\221\347\231\275\345\217\215\350\275\211\346\216\247\345\210\266", nullptr));
        label_4->setText(QCoreApplication::translate("MainWindow", "\351\273\221\347\231\275\351\226\245\345\200\274", nullptr));
        image_threshold_val->setText(QCoreApplication::translate("MainWindow", "1", nullptr));
        label_3->setText(QCoreApplication::translate("MainWindow", "\347\266\262\351\273\236\351\241\217\350\211\262", nullptr));
        radioBtn_2->setText(QCoreApplication::translate("MainWindow", "\347\201\260\351\232\216", nullptr));
        radioBtn_3->setText(QCoreApplication::translate("MainWindow", "\346\274\270\345\261\244", nullptr));
        label_5->setText(QCoreApplication::translate("MainWindow", "\350\207\252\345\213\225\346\252\242\346\270\254\345\205\250\345\261\200", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
