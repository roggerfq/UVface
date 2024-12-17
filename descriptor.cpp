/***************************************************************************
**                                                                        **
**  This source code is part of UVface++, a system developed by           **
**  Roger Figueroa Quintero as part of his undergraduate thesis titled:   **
**  "Image Analysis System for Face Detection and Recognition"            **
**  at Universidad del Valle, Cali, Colombia.                             **
**                                                                        **
**  Copyright (C) 2016-2024 Roger Figueroa Quintero                       **
**                                                                        **
**  This software is licensed for non-commercial use only. Redistribution **
**  and use in source and binary forms, with or without modification, are **
**  permitted provided that the following conditions are met:             **
**                                                                        **
**  1. Redistributions of source code must retain the above copyright     **
**     notice, this list of conditions, and the following disclaimer.     **
**  2. Redistributions in binary form must reproduce the above copyright  **
**     notice, this list of conditions, and the following disclaimer in   **
**     the documentation and/or other materials provided with the         **
**     distribution.                                                      **
**  3. The name of the author may not be used to endorse or promote       **
**     products derived from this software without specific prior written **
**     permission.                                                        **
**  4. This software is to be used for non-commercial purposes only.      **
**                                                                        **
**  This software is distributed in the hope that it will be useful, but  **
**  WITHOUT ANY WARRANTY; without even the implied warranty of            **
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                  **
**                                                                        **
****************************************************************************
**           Author: Roger Figueroa Quintero                              **
**  Contact Email: roggerfq@hotmail.com                                   **
**          Date: First developed: November 2016                          **
**                Last modified: December 2024                            **
****************************************************************************/

#include "descriptor.h"
//STL
#include <iostream>
#include <stdio.h>
//OPENCV
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//QT
#include <QFile>
#include <QVBoxLayout>
#include <QLabel>
#include <QFormLayout>
#include <QLineEdit>
#include <QMessageBox>

ABSTRACT_DESCRIPTOR::~ABSTRACT_DESCRIPTOR() {};

//________________________________TEST DESCRIPTOR___________________________________________________//

DESCRIPTOR_TEST1::DESCRIPTOR_TEST1(int mi, int ni, int nri): m(mi), n(ni), nr(nri), sigma(2) {

  descriptorTemp = new cv::Mat(nr, m * n, CV_8UC1);
  filteredImage = new cv::Mat;
  imgTemp = new cv::Mat(m, n, CV_8UC1);
  descriptorTest = new Eigen::MatrixXf;

  lineEditNewWidth = new QLineEdit(QString::number(n));
  connect(lineEditNewWidth, SIGNAL(textChanged(const QString & )), this, SLOT(edit_n()));

  lineEditNewHeight = new QLineEdit(QString::number(m));
  connect(lineEditNewHeight, SIGNAL(textChanged(const QString & )), this, SLOT(edit_m()));

  QFormLayout * formLayout = new QFormLayout;
  formLayout -> addRow(tr("New Width:"), lineEditNewWidth);
  formLayout -> addRow(tr("New Height:"), lineEditNewHeight);

  setLayout(formLayout);

}

DESCRIPTOR_TEST1::~DESCRIPTOR_TEST1() {

  delete descriptorTemp;
  delete filteredImage;
  delete imgTemp;
  delete descriptorTest;

  std::cout << "Destructor in test descriptor\n"; // Standard output in C++
}

void DESCRIPTOR_TEST1::edit_n() {

  n = (lineEditNewWidth -> text()).toInt();
  qDebug() << "New value of n=" << (lineEditNewWidth -> text()) << "\n"; // Debugging message

  emit editingDescriptor();
}

void DESCRIPTOR_TEST1::edit_m() {

  m = (lineEditNewHeight -> text()).toInt();

  qDebug() << "New value of m=" << (lineEditNewHeight -> text()) << "\n"; // Debugging message

  emit editingDescriptor();
}

void DESCRIPTOR_TEST1::loadSettings(QString path) {
  //Concept: Load configuration from file - apply configuration

  //____________Reading configuration from hard drive____________________//
  QFile fileTemp(path + "/settingDescriptor.info");
  fileTemp.open(QIODevice::ReadOnly);
  fileTemp.read(reinterpret_cast < char * > ( & m), sizeof(int));
  fileTemp.read(reinterpret_cast < char * > ( & n), sizeof(int));
  fileTemp.close();
  //____________________________________________________________________//

  //_____________APPLYING CONFIGURATION______________________________//

  delete descriptorTemp;
  descriptorTemp = new cv::Mat(nr, m * n, CV_8UC1);
  delete imgTemp;
  imgTemp = new cv::Mat(m, n, CV_8UC1);

  //___________________________________________________________________//

  //________________UPDATING VIEW_____________________________//
  lineEditNewWidth -> setText(QString::number(n));
  lineEditNewHeight -> setText(QString::number(m));
  //__________________________________________________________________//

  qDebug() << "LOADING PREVIOUS CONFIGURATION FROM=" << path << "\n"; // Debugging message

}

bool DESCRIPTOR_TEST1::applySettings(QString path) {
  //Concept: Apply current configuration - save current configuration to file

  if ((nr <= 0) || (m <= 0) || (n <= 0)) {
    QMessageBox msgBox;
    msgBox.setText(QString::fromUtf8("Some configuration parameters are less than zero."));
    msgBox.setWindowTitle("ERROR");
    msgBox.exec();
    return false;
  }

  //____Applying current configuration______//
  delete descriptorTemp;
  descriptorTemp = new cv::Mat(nr, m * n, CV_8UC1);
  delete imgTemp;
  imgTemp = new cv::Mat(m, n, CV_8UC1);
  //________________________________________//

  //____________Saving configuration to hard drive_______________//
  QFile fileTemp(path + "/settingDescriptor.info");
  fileTemp.open(QIODevice::WriteOnly);
  fileTemp.write(reinterpret_cast < char * > ( & m), sizeof(int));
  fileTemp.write(reinterpret_cast < char * > ( & n), sizeof(int));
  fileTemp.close();
  //___________________________________________________________________//

  qDebug() << "APPLYING CONFIGURATION TO=" << path << "\n"; // Debugging message

  return true;
}

cv::Mat * DESCRIPTOR_TEST1::descriptor_base(const cv::Mat & img) {

  cv::Mat img2 = img.clone();

  if (img2.channels() > 1)
    cv::cvtColor(img2, img2, CV_BGR2GRAY);

  if ((img2.cols > n) && ((img2.rows > m)))
    GaussianBlur(img2, ( * filteredImage), cv::Size(sigma * 6 + 1, sigma * 6 + 1), sigma);
  cv::resize(( * filteredImage), ( * imgTemp), cv::Size(n, m), 0, 0, cv::INTER_LANCZOS4);

  for (int i = 0; i < nr; i++)
    imgTemp -> reshape(0, 1).copyTo(descriptorTemp -> row(i));

  descriptorTemp -> convertTo(( * descriptorTemp), CV_32F);

  return descriptorTemp;

}

void DESCRIPTOR_TEST1::post_processing(cv::Mat & descriptor_base, cv::Mat & descriptor_end, std::vector < int > & ithRows) {

  std::vector < int > newithRows;
  int rowTop = 0, rowBottom;

  for (int i = 0; i < ithRows.size(); i++) {

    int rowBottom = ithRows[i];

    cv::Mat M = cv::Mat::zeros(1, descriptor_base.cols, CV_32F);

    for (int j = rowTop; j < rowBottom; j++) {
      M = M + descriptor_base.row(j);
    }

    M = M / (rowBottom - rowTop);
    descriptor_end.push_back(M);
    newithRows.push_back(descriptor_end.rows);

    rowTop = rowBottom;
  }

  ithRows = newithRows;

  /*
  //A simulated post-processing where descriptor_base is multiplied by 2
  descriptor_end=2*descriptor_base.clone();
  descriptor_end.convertTo(descriptor_end,CV_32F);
  */

}

Eigen::MatrixXf * DESCRIPTOR_TEST1::test(const cv::Mat & img) {

  cv::Mat temp = ( * descriptor_base(img)).clone();
  temp.convertTo(temp, CV_32F);

  float * pf = temp.ptr < float > (0);
  Eigen::Map < Eigen::MatrixXf > mf(pf, temp.cols, temp.rows);
  ( * descriptorTest) = mf;

  return descriptorTest;

}

QString DESCRIPTOR_TEST1::nameDescriptor() const {
  QString name("Descriptor with post-processing");
  return name;
}

DESCRIPTOR_TEST2::DESCRIPTOR_TEST2(int mi, int ni, int nri): m(mi), n(ni), nr(nri), sigma(2) {

  descriptorTemp = new cv::Mat(nr, m * n, CV_8UC1);
  filteredImage = new cv::Mat;
  imgTemp = new cv::Mat(m, n, CV_8UC1);
  descriptorTest = new Eigen::MatrixXf;

  lineEditNewWidth = new QLineEdit(QString::number(n));
  connect(lineEditNewWidth, SIGNAL(textChanged(const QString & )), this, SLOT(edit_n()));

  lineEditNewHeight = new QLineEdit(QString::number(m));
  connect(lineEditNewHeight, SIGNAL(textChanged(const QString & )), this, SLOT(edit_m()));

  lineEditNumberOfRepetitions = new QLineEdit(QString::number(nr));
  connect(lineEditNumberOfRepetitions, SIGNAL(textChanged(const QString & )), this, SLOT(edit_nr()));

  QFormLayout * formLayout = new QFormLayout;
  formLayout -> addRow(tr("New Width:"), lineEditNewWidth);
  formLayout -> addRow(tr("New Height:"), lineEditNewHeight);
  formLayout -> addRow(tr("Number of repetitions:"), lineEditNumberOfRepetitions);

  setLayout(formLayout);

}

DESCRIPTOR_TEST2::~DESCRIPTOR_TEST2() {

  delete descriptorTemp;
  delete filteredImage;
  delete imgTemp;
  delete descriptorTest;

  std::cout << "Destructor in test descriptor\n"; // Standard output in C++
}

void DESCRIPTOR_TEST2::edit_n() {

  n = (lineEditNewWidth -> text()).toInt();

  qDebug() << "New value of n=" << (lineEditNewWidth -> text()) << "\n";

  emit editingDescriptor();

}

void DESCRIPTOR_TEST2::edit_m() {

  m = (lineEditNewHeight -> text()).toInt();

  qDebug() << "New value of m=" << (lineEditNewHeight -> text()) << "\n";

  emit editingDescriptor();

}

void DESCRIPTOR_TEST2::edit_nr() {

  nr = (lineEditNumberOfRepetitions -> text()).toInt();

  qDebug() << "New value of nr=" << (lineEditNumberOfRepetitions -> text()) << "\n";

  emit editingDescriptor();

}

void DESCRIPTOR_TEST2::loadSettings(QString path) {
  //Concept: Load configuration from file - apply configuration

  //____________Reading configuration from hard drive____________________//
  QFile fileTemp(path + "/settingDescriptor.info");
  fileTemp.open(QIODevice::ReadOnly);
  fileTemp.read(reinterpret_cast < char * > ( & m), sizeof(int));
  fileTemp.read(reinterpret_cast < char * > ( & n), sizeof(int));
  fileTemp.read(reinterpret_cast < char * > ( & nr), sizeof(int));
  fileTemp.close();
  //____________________________________________________________________//

  //_____________APPLYING THE CONFIGURATION____________________________//
  delete descriptorTemp;
  descriptorTemp = new cv::Mat(nr, m * n, CV_8UC1);
  delete imgTemp;
  imgTemp = new cv::Mat(m, n, CV_8UC1);

  //___________________________________________________________________//

  //________________UPDATING THE VIEW_____________________________//
  lineEditNewWidth -> setText(QString::number(n));
  lineEditNewHeight -> setText(QString::number(m));
  lineEditNumberOfRepetitions -> setText(QString::number(nr));
  //__________________________________________________________________//

  qDebug() << "LOADING PREVIOUS CONFIGURATION FROM=" << path << "\n";

}

bool DESCRIPTOR_TEST2::applySettings(QString path) {
  //Concept: Apply current configuration - save in current configuration file

  if ((nr <= 0) || (m <= 0) || (n <= 0)) {
    QMessageBox msgBox;
    msgBox.setText(QString::fromUtf8("Some configuration parameters are less than zero."));
    msgBox.setWindowTitle("ERROR");
    msgBox.exec();
    return false;
  }

  //____Applying current configuration______//
  delete descriptorTemp;
  descriptorTemp = new cv::Mat(nr, m * n, CV_8UC1);
  delete imgTemp;
  imgTemp = new cv::Mat(m, n, CV_8UC1);
  //________________________________________//

  //____________Saving the configuration to hard drive_______________//
  QFile fileTemp(path + "/settingDescriptor.info");
  fileTemp.open(QIODevice::WriteOnly);
  fileTemp.write(reinterpret_cast < char * > ( & m), sizeof(int));
  fileTemp.write(reinterpret_cast < char * > ( & n), sizeof(int));
  fileTemp.write(reinterpret_cast < char * > ( & nr), sizeof(int));
  fileTemp.close();
  //___________________________________________________________________//

  qDebug() << "APPLYING CONFIGURATION TO=" << path << "\n";

  return true;
}

cv::Mat * DESCRIPTOR_TEST2::descriptor_base(const cv::Mat & img) {

  cv::Mat img2 = img.clone();
  if (img2.channels() > 1)
    cv::cvtColor(img2, img2, CV_BGR2GRAY);

  if ((img2.cols > n) && ((img2.rows > m)))
    GaussianBlur(img2, ( * filteredImage), cv::Size(sigma * 6 + 1, sigma * 6 + 1), sigma);
  cv::resize(( * filteredImage), ( * imgTemp), cv::Size(n, m), 0, 0, cv::INTER_LANCZOS4);

  for (int i = 0; i < nr; i++)
    imgTemp -> reshape(0, 1).copyTo(descriptorTemp -> row(i));

  descriptorTemp -> convertTo(( * descriptorTemp), CV_32F);

  return descriptorTemp;

}

void DESCRIPTOR_TEST2::post_processing(cv::Mat & descriptor_base, cv::Mat & descriptor_end, std::vector < int > & ithRows) {

  //This is the only thing that should be done if no post-processing is applied
  descriptor_end = cv::Mat();

}

Eigen::MatrixXf * DESCRIPTOR_TEST2::DESCRIPTOR_TEST2::test(const cv::Mat & img) {

  cv::Mat temp = ( * descriptor_base(img)).clone();
  temp.convertTo(temp, CV_32F);

  float * pf = temp.ptr < float > (0);
  Eigen::Map < Eigen::MatrixXf > mf(pf, temp.cols, temp.rows);
  ( * descriptorTest) = mf;

  return descriptorTest;

}

QString DESCRIPTOR_TEST2::nameDescriptor() const {
  QString name("Descriptor without post processing");
  return name;
}
