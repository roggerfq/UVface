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

#ifndef GUICONFIGDETECTOR_H
#define GUICONFIGDETECTOR_H

//QT
#include <QDialog>
#include <QThread>
#include <QMutex>
#include <QLabel>
//openCV
#include "opencv2/highgui/highgui.hpp"

//Forward QT classes
class QLineEdit;
class QCheckBox;
class QPushButton;
class QGroupBox;

//Forward custom classes
class CASCADE_CLASSIFIERS_EVALUATION;
class GUI_DETECTOR;

class threadDetector: public QThread {

  Q_OBJECT

  //Synchronization
  QMutex mutex;
  volatile bool stopped;

  //Operation
  int command;
  cv::VideoCapture cap; //Captures video from camera or video file
  //cv::VideoCapture cap2; //Captures video from camera or video file
  cv::Mat frame; //Frame from video to detect (camera)
  cv::Mat frameTemp; //Auxiliary matrix for the camera capture routine
  cv::Mat frame2; //Frame from video to detect (video)
  //std::vector<cv::Mat> listDetectedObjects;

  //List of rectangle coordinates for detection functions 
  //std::vector<cv::RotatedRect> coordinatesDetectedObjectsRotated; //When the angle is normalized
  //std::vector<cv::Rect> coordinatesDetectedObjects; //When the angle is not normalized

  cv::Mat currentImage; //Image to detect

  CASCADE_CLASSIFIERS_EVALUATION * objectDetector;

  //Flags
  bool detectorIsLoad;
  bool normalizeRotation;
  bool doubleList;
  bool groupingRectangles;

  public:
    threadDetector();
  ~threadDetector();
  void stop();

  void loadDetector(std::string fileName);
  bool setDevice(int i); //For camera
  bool setDevice(std::string videoFile); //For video file

  //Command functions for run
  int detectObjectVideoCamera(int device);
  int detectObjectVideoFile(std::string nameFileVideo);
  int detectObjectImageFile(std::string nameFileImage);

  //Run functions
  void startDetectObjectVideoCamera();
  void startDetectObjectVideoFile();
  void startDetectObjectImageFile();

  //Get functions
  bool detectorIsReady() const; //Informs if the detector is loaded and configured
  int getCommand() const;
  int getSizeMaxWindow() const;

  signals:
    void listCoordinatesAndDetectedObjects(std::vector < cv::Mat > listDetectedObjects, std::vector < cv::Rect > coordinatesDetectedObjects);
  void listCoordinatesAndDetectedObjectsRotated(std::vector < cv::Mat > listDetectedObjects, std::vector < cv::RotatedRect > coordinatesDetectedObjectsRotated);
  void listCoordinatesAndDetectedObjects_img(std::vector < cv::Mat > listDetectedObjects, std::vector < cv::Rect > coordinatesDetectedObjects);
  void listCoordinatesAndDetectedObjectsRotated_img(std::vector < cv::Mat > listDetectedObjects, std::vector < cv::RotatedRect > coordinatesDetectedObjectsRotated);
  void finishedDetection();
  void setSizeFrame(int width, int high);
  void imageReady(const cv::Mat & frame);
  void clearLabelVideo();
  void resetTrackerWindows();
  void stopRecognizer();
  void enableRecognition();

  friend class GUI_DETECTOR;

  protected:
    void run();

};

class WINDOW: public QLabel {
  Q_OBJECT

  QPushButton * buttonNext;
  QPushButton * buttonBack;
  QPushButton * buttonCapture;
  QPushButton * buttonLoadFiles;

  QLabel * labelMenu;

  public:
    WINDOW();

};

class GUI_DETECTOR: public QDialog {

  Q_OBJECT

  //QPushButton
  QPushButton * buttonLoadDetector;
  QPushButton * buttonApplySettings;

  //QLineEdit
  QLineEdit * lineEditDegreesDetections;
  QLineEdit * lineEditSizeBase;
  QLineEdit * lineEditFactorScaleWindow;
  QLineEdit * lineEditStepWindow;
  QLineEdit * lineEditSizeMaxWindow;
  QLineEdit * lineEditGroupThreshold;
  QLineEdit * lineEditEps;
  QLineEdit * lineEditNumberClassifiersUsed;
  QLineEdit * lineEditLineThicknessRectangles;
  QLineEdit * lineEditColorRectanglesR;
  QLineEdit * lineEditColorRectanglesG;
  QLineEdit * lineEditColorRectanglesB;

  //For color restrictions in the HSV space
  QLineEdit * lineEditHmin;
  QLineEdit * lineEditHmax;
  QLineEdit * lineEditSmin;
  QLineEdit * lineEditSmax;
  QLineEdit * lineEditVmin;
  QLineEdit * lineEditVmax;

  //QCheckBox
  QCheckBox * checkBoxFlagActivateSkinColor;
  QCheckBox * checkBoxNormalizeRotation;
  QCheckBox * checkBoxDoubleList;
  QCheckBox * checkBoxFlagExtractColorImages;

  //QLabel
  QLabel * textNumberStrongLearns;

  //QGroupBox
  QGroupBox * groupBoxExtraSettings;

  //Detector thread address
  threadDetector * myThreadDetector;
  int numberStrongLearns;

  //Flags
  bool flagEdition;

  public:
    GUI_DETECTOR(threadDetector * myThreadDetector);

  public slots:
    void setEnabledFalseAllConfig();
  void setEnabledTrueConfigDefault();
  void setdefaultConfig();
  void setEnabledHsvConfig(int flag);
  void setEnabledOptionNotGroup();
  void activeConfigHsv(int state);
  void normalizeRotation(int state);
  void loadDetector();
  void setConfig();
  void edition();
};

#endif
