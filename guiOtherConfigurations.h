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

#ifndef GUIOTHERCONFIGURATIONS_H
#define GUIOTHERCONFIGURATIONS_H
//The file guiOtherConfigurations.h and guiOtherConfigurations.cpp will construct classes and functions required for extra configurations

//openCV
#include <opencv2/opencv.hpp>
//QT
#include <QDialog>
#include <QMap>

//forward QT classes
class QLineEdit;
class QComboBox;
class QCheckBox;

class configPresentationRecognition: public QDialog {

  Q_OBJECT

  //QLineEdit
  QLineEdit * lineEditFontScale;
  QLineEdit * lineEditR;
  QLineEdit * lineEditG;
  QLineEdit * lineEditB;
  QLineEdit * lineEditThickness;

  //QComboBox
  QComboBox * comboxFontType;
  //QCheckBox
  QCheckBox * combinedWithItalic;

  QMap < int, int > allFontTypes;
  int * fontType;
  double * fontScale;
  cv::Scalar * color;
  int * thickness;
  int * lineType;

  public:
    configPresentationRecognition(int * fontType, double * fontScale, cv::Scalar * color, int * thickness, int * lineType);

  public slots:
    void changeFontType(int type);
    void changeCombinedWithItalic(int state);
    void changeFontScale(const QString & text);
    void changeColorR(const QString & text);
    void changeColorG(const QString & text);
    void changeColorB(const QString & text);
    void changeThickness(const QString & text);

};

#endif
