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

//Custom
#include "trackerWindows_gui.h"
#include "trackerWindows.h"
//QT 
#include <QFormLayout>
#include <QVBoxLayout>
#include <QPlastiqueStyle>
#include <QGroupBox>
#include <QLineEdit>
#include <QPushButton>
#include <QIntValidator>
#include <QDebug>

trackerWindows_gui::trackerWindows_gui(trackerWindows * myTrackerWindows): myTrackerWindows(myTrackerWindows) {

  lineEditInitiaLpunctuation = new QLineEdit(QString::number(myTrackerWindows -> getInitiaLpunctuation()));
  connect(lineEditInitiaLpunctuation, SIGNAL(textEdited(const QString & )), this, SLOT(changeInitiaLpunctuation(const QString & )));
  QRegExp reInitiaLpunctuation("([0-9][0-9]*)");
  QRegExpValidator * validatorInitiaLpunctuation = new QRegExpValidator(reInitiaLpunctuation, this);
  lineEditInitiaLpunctuation -> setValidator(validatorInitiaLpunctuation);
  lineEditInitiaLpunctuation -> setFixedWidth(40);

  lineEditMinimumPunctuationToRecognize = new QLineEdit(QString::number(myTrackerWindows -> getMinimumPunctuationToRecognize()));
  connect(lineEditMinimumPunctuationToRecognize, SIGNAL(textEdited(const QString & )), this, SLOT(changeMinimumPunctuationToRecognize(const QString & )));
  QRegExp reMinimumPunctuationToRecognize("([0-9][0-9]*)");
  QRegExpValidator * validatorMinimumPunctuationToRecognize = new QRegExpValidator(reMinimumPunctuationToRecognize, this);
  lineEditMinimumPunctuationToRecognize -> setValidator(validatorMinimumPunctuationToRecognize);
  lineEditMinimumPunctuationToRecognize -> setFixedWidth(40);

  lineEditEps = new QLineEdit(QString::number(myTrackerWindows -> getEps()));
  connect(lineEditEps, SIGNAL(textEdited(const QString & )), this, SLOT(changeEps(const QString & )));
  QRegExp reEps("([0-9]*[//.][0-9]*)");
  QRegExpValidator * validatorEps = new QRegExpValidator(reEps, this);
  lineEditEps -> setValidator(validatorEps);
  lineEditEps -> setFixedWidth(40);

  buttonSetDefaultValues = new QPushButton(QString::fromUtf8("Default values"));
  connect(buttonSetDefaultValues, SIGNAL(clicked()), this, SLOT(setDefaultValues()));
  buttonSetDefaultValues -> setAutoDefault(false);

  //ORGANIZING THE GRAPHICAL PART
  QFormLayout * formLayoutPrincipal = new QFormLayout;
  formLayoutPrincipal -> addRow(QString::fromUtf8("Score to eliminate:"), lineEditInitiaLpunctuation);
  formLayoutPrincipal -> addRow(QString::fromUtf8("Score to recognize:"), lineEditMinimumPunctuationToRecognize);
  formLayoutPrincipal -> addRow(tr("EPS:"), lineEditEps);
  formLayoutPrincipal -> addRow(buttonSetDefaultValues);

  QGroupBox * groupBoxPrincipal = new QGroupBox(QString::fromUtf8("Frame tracking configuration"));
  groupBoxPrincipal -> setLayout(formLayoutPrincipal);
  groupBoxPrincipal -> setStyle(new QPlastiqueStyle);

  QVBoxLayout * layoutGroupBox = new QVBoxLayout;
  layoutGroupBox -> addWidget(groupBoxPrincipal);

  setLayout(layoutGroupBox);
  setWindowTitle(QString::fromUtf8("TRACKER CONFIGURATION"));
  setMinimumWidth(320);

}

void trackerWindows_gui::changeInitiaLpunctuation(const QString & text) {
  myTrackerWindows -> setInitiaLpunctuation(text.toInt());
}

void trackerWindows_gui::changeMinimumPunctuationToRecognize(const QString & text) {
  myTrackerWindows -> setMinimumPunctuationToRecognize(text.toInt());
}

void trackerWindows_gui::changeEps(const QString & text) {
  myTrackerWindows -> setEps(text.toDouble());
}

void trackerWindows_gui::setDefaultValues() {

  myTrackerWindows -> setDefaultValues();

  lineEditInitiaLpunctuation -> setText(QString::number(myTrackerWindows -> getInitiaLpunctuation()));
  lineEditMinimumPunctuationToRecognize -> setText(QString::number(myTrackerWindows -> getMinimumPunctuationToRecognize()));
  lineEditEps -> setText(QString::number(myTrackerWindows -> getEps()));

}
