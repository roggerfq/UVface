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

//QT
#include <QApplication>
#include <QMetaType>
//Own classes
#include "interfazPrincipal.h"
#include "trackerWindows.h" //Window tracker
#include "plotSparseSolution.h"
//openCV
#include <opencv2/opencv.hpp> //Required for registering the Mat type
//STL
#include <vector> //Required for registering the vector type


//____________________REGISTERING NECESSARY TYPES___________________________//
Q_DECLARE_METATYPE(std::vector < int > )
Q_DECLARE_METATYPE(QList < int > )
Q_DECLARE_METATYPE(std::vector < std::string > )
Q_DECLARE_METATYPE(cv::Point)
Q_DECLARE_METATYPE(std::vector < cv::Point > )
Q_DECLARE_METATYPE(cv::Mat)
Q_DECLARE_METATYPE(std::vector < cv::Mat > )
Q_DECLARE_METATYPE(std::vector < cv::Rect > )
Q_DECLARE_METATYPE(std::vector < cv::RotatedRect > )
Q_DECLARE_METATYPE(imageTransaction)
Q_DECLARE_METATYPE(QList < imageTransaction > )
//___________________________________________________________________________//

int main(int argc, char * argv[]) {

  QApplication app(argc, argv);

  //____________________REGISTERING NECESSARY TYPES___________________________//
  qRegisterMetaType < std::vector < int > > ("std::vector<int>");
  qRegisterMetaType < QList < int > > ("QList<int>");
  qRegisterMetaType < std::vector < std::string > > ("std::vector<std::string>");
  qRegisterMetaType < cv::Point > ("cv::Point");
  qRegisterMetaType < std::vector < cv::Point > > ("std::vector<cv::Point>");
  qRegisterMetaType < cv::Mat > ("cv::Mat");
  qRegisterMetaType < std::vector < cv::Mat > > ("std::vector<cv::Mat>");
  qRegisterMetaType < std::vector < cv::Rect > > ("std::vector<cv::Rect>");
  qRegisterMetaType < std::vector < cv::RotatedRect > > ("std::vector<cv::RotatedRect>");
  qRegisterMetaType < imageTransaction > ("imageTransaction");
  qRegisterMetaType < QList < imageTransaction > > ("QList<imageTransaction>");
  //___________________________________________________________________________//

  /*
  PLOT_SPARSE_SOLUTION PL;
  PL.show();
  */

  interfaz interfazPrincipal; //Main interface
  interfazPrincipal.show();

  return app.exec();

  return 0;
}
