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

#include "guiOtherConfigurations.h"

//QT
#include <QLabel>
#include <QLineEdit>
#include <QComboBox>
#include <QCheckBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QPlastiqueStyle>
#include <QIntValidator>

configPresentationRecognition::configPresentationRecognition(int * fontType, double * fontScale, cv::Scalar * color, int * thickness, int * lineType): fontType(fontType), fontScale(fontScale), color(color), thickness(thickness), lineType(lineType) {

  //By default
  *fontType = cv::FONT_HERSHEY_COMPLEX_SMALL;
  *fontScale = 2;
  *color = cv::Scalar(255, 0, 0);
  *thickness = 4;
  *lineType = CV_AA;

  lineEditFontScale = new QLineEdit(QString::number(*fontScale));
  connect(lineEditFontScale, SIGNAL(textEdited(const QString &)), this, SLOT(changeFontScale(const QString &)));
  lineEditFontScale->setFixedWidth(40);
  QRegExp reFontScale("([0-9]*[//.][0-9]*)");
  QRegExpValidator *validatorFontScale = new QRegExpValidator(reFontScale, this);
  lineEditFontScale->setValidator(validatorFontScale);
  lineEditFontScale->setFixedWidth(40);

  lineEditThickness = new QLineEdit(QString::number(*thickness));
  connect(lineEditThickness, SIGNAL(textEdited(const QString &)), this, SLOT(changeThickness(const QString &)));
  lineEditThickness->setFixedWidth(40);

  //______________________about font type__________________________________//
  allFontTypes[0] = cv::FONT_HERSHEY_SIMPLEX;
  allFontTypes[1] = cv::FONT_HERSHEY_PLAIN;
  allFontTypes[2] = cv::FONT_HERSHEY_DUPLEX;
  allFontTypes[3] = cv::FONT_HERSHEY_COMPLEX;
  allFontTypes[4] = cv::FONT_HERSHEY_TRIPLEX;
  allFontTypes[5] = cv::FONT_HERSHEY_COMPLEX_SMALL;
  allFontTypes[6] = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
  allFontTypes[7] = cv::FONT_HERSHEY_SCRIPT_COMPLEX;

  comboxFontType = new QComboBox;
  comboxFontType->addItem("HERSHEY_SIMPLEX");
  comboxFontType->addItem("HERSHEY_PLAIN");
  comboxFontType->addItem("HERSHEY_DUPLEX");
  comboxFontType->addItem("HERSHEY_COMPLEX");
  comboxFontType->addItem("HERSHEY_TRIPLEX");
  comboxFontType->addItem("HERSHEY_COMPLEX_SMALL");
  comboxFontType->addItem("HERSHEY_SCRIPT_COMPLEX");
  connect(comboxFontType, SIGNAL(currentIndexChanged(int)), this, SLOT(changeFontType(int)));

  combinedWithItalic = new QCheckBox(QString::fromUtf8("combine with italic font"));
  connect(combinedWithItalic, SIGNAL(stateChanged(int)), this, SLOT(changeCombinedWithItalic(int)));

  //________________________________________________________________________________//

  lineEditR = new QLineEdit(QString::number(color->val[2]));
  connect(lineEditR, SIGNAL(textEdited(const QString &)), this, SLOT(changeColorR(const QString &)));
  lineEditR->setValidator(new QIntValidator(0, 255));
  lineEditR->setFixedWidth(40);
  lineEditG = new QLineEdit(QString::number(color->val[1]));
  connect(lineEditG, SIGNAL(textEdited(const QString &)), this, SLOT(changeColorG(const QString &)));
  lineEditG->setValidator(new QIntValidator(0, 255));
  lineEditG->setFixedWidth(40);
  lineEditB = new QLineEdit(QString::number(color->val[0]));
  connect(lineEditB, SIGNAL(textEdited(const QString &)), this, SLOT(changeColorB(const QString &)));
  lineEditB->setValidator(new QIntValidator(0, 255));
  lineEditB->setFixedWidth(40);

  //Arranging graphical part
  QGridLayout *layoutColor = new QGridLayout;
  layoutColor->addWidget(new QLabel(QString("<font color=red>R</font></h2>")), 0, 0, 1, 1);
  layoutColor->addWidget(lineEditR, 0, 1, 1, 1);
  layoutColor->addWidget(new QLabel(QString("<font color=green>G</font></h2>")), 0, 2, 1, 1);
  layoutColor->addWidget(lineEditG, 0, 3, 1, 1);
  layoutColor->addWidget(new QLabel(QString("<font color=blue>B</font></h2>")), 0, 4, 1, 1);
  layoutColor->addWidget(lineEditB, 0, 5, 1, 1);

  QGroupBox *groupBoxColor = new QGroupBox(tr("Color"));
  groupBoxColor->setLayout(layoutColor);
  groupBoxColor->setStyle(new QPlastiqueStyle);

  QGridLayout *layoutPrincipal = new QGridLayout;
  layoutPrincipal->addWidget(new QLabel("Scale"), 0, 0, 1, 1);
  layoutPrincipal->addWidget(lineEditFontScale, 0, 1, 1, 1);
  layoutPrincipal->addWidget(new QLabel("Thickness"), 0, 2, 1, 1);
  layoutPrincipal->addWidget(lineEditThickness, 0, 3, 1, 1);
  layoutPrincipal->addWidget(new QLabel("Font type"), 1, 0, 1, 2);
  layoutPrincipal->addWidget(comboxFontType, 1, 2, 1, 3);
  layoutPrincipal->addWidget(combinedWithItalic, 2, 0, 1, 4);
  layoutPrincipal->addWidget(groupBoxColor, 3, 0, 1, 4);

  setLayout(layoutPrincipal);
  setWindowTitle(QString::fromUtf8("Presentation Recognition"));
  setMinimumWidth(300);

}

void configPresentationRecognition::changeFontType(int type) {

  if (combinedWithItalic->checkState() == Qt::Checked)
    *fontType = allFontTypes[type] + cv::FONT_ITALIC;
  else
    *fontType = allFontTypes[type];

}

void configPresentationRecognition::changeCombinedWithItalic(int state) {

  if (state == Qt::Checked)
    *fontType = allFontTypes[comboxFontType->currentIndex()] + cv::FONT_ITALIC;
  else
    *fontType = allFontTypes[comboxFontType->currentIndex()];

}

void configPresentationRecognition::changeFontScale(const QString &text) {
  *fontScale = text.toInt();
}

void configPresentationRecognition::changeColorR(const QString &text) {
  color->val[2] = text.toInt();
}
void configPresentationRecognition::changeColorG(const QString &text) {
  color->val[1] = text.toInt();
}

void configPresentationRecognition::changeColorB(const QString &text) {
  color->val[0] = text.toInt();
}

void configPresentationRecognition::changeThickness(const QString &text) {
  *thickness = text.toInt();
}
