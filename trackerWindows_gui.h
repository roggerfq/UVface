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

#ifndef TRACKERWINDOWSGUI_H
#define TRACKERWINDOWSGUI_H

//QT
#include <QDialog>

//forward declarations for custom classes
class trackerWindows;

//forward declarations for QT classes
class QLineEdit;
class QPushButton;

class trackerWindows_gui: public QDialog {

  Q_OBJECT

  //QLineEdit
  QLineEdit * lineEditInitiaLpunctuation;
  QLineEdit * lineEditMinimumPunctuationToRecognize;
  QLineEdit * lineEditEps;

  //QPushButton
  QPushButton * buttonSetDefaultValues;

  trackerWindows * myTrackerWindows;
  public:
    trackerWindows_gui(trackerWindows * myTrackerWindows);

  public slots:
    void changeInitiaLpunctuation(const QString & text);
    void changeMinimumPunctuationToRecognize(const QString & text);
    void changeEps(const QString & text);
    void setDefaultValues();

};

#endif
