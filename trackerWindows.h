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

#ifndef TRACKERWINDOWS_H
#define TRACKERWINDOWS_H

//openCV
#include "opencv2/highgui/highgui.hpp"//Contains the STL vector class in C++
//QT
#include <QObject>
#include <QTime>
#include <QMap>

class imageTransaction {

  public: imageTransaction(): id(-1) {}
  imageTransaction(cv::Mat img, QTime myBirthdate, int id): img(img),
  myBirthdate(myBirthdate),
  id(id) {}

  imageTransaction(const imageTransaction & otherimageTransaction) {

    img = otherimageTransaction.img;
    myBirthdate = otherimageTransaction.myBirthdate;
    id = otherimageTransaction.id;
    name = otherimageTransaction.name;
  }

  /*This class is used to allow the facial recognition algorithm to interact with the tracker class. It is only useful for packaging an image with its respective ID and the date it was detected*/
  cv::Mat img;
  QTime myBirthdate;
  int id;
  std::string name;

};

//This class will represent a detection rectangle
class rectangleDetection: public cv::Rect {

  public: rectangleDetection();
  rectangleDetection(const cv::Rect & rect, cv::Mat & img, int punctuationBeforeDeleting, QTime birthdate, int id = -1, bool isNew = true);
  rectangleDetection(const rectangleDetection & rectDetection);

  void setRect(const cv::Rect & rectangle);
  cv::Rect getRect() const;

  int id; //Numerical identifier
  QTime myBirthdate; //Stores when (h,m,s) the detection was created
  int punctuationStableWindow; //Score that will grow with the number of repetitions of this window
  int punctuationBeforeDeleting; //Score that, if it reaches zero, will delete the tracker
  bool isNew; //Indicates whether it was created recently or if it had been created previously
  bool empty; //Indicates if this detection hasn't been assigned values yet
  bool windowSentToRecognizer; //Indicates if this detection has already been sent to the facial recognizer
  bool isRecognized; //Indicates if the detection has already been recognized
  std::string name; //Name assigned to the detection
  cv::Mat img; //Image inside the rectangle
};

//This class will represent a rotated detection rectangle
class rotatedRectDetection: public cv::RotatedRect {

  public: rotatedRectDetection();
  rotatedRectDetection(const cv::RotatedRect & rotatedRect, cv::Mat & img, int punctuationBeforeDeleting, QTime birthdate, int id = -1, bool isNew = true);
  rotatedRectDetection(const rotatedRectDetection & rotatedDetection);

  void setRotatedRect(const cv::RotatedRect & rotatedRect);
  cv::RotatedRect getRotatedRect() const;

  int id; //Numerical identifier
  QTime myBirthdate; //Stores when (h,m,s) the detection was created
  int punctuationStableWindow; //Score that will grow with the number of repetitions of this window
  int punctuationBeforeDeleting; //Score that, if it reaches zero, will delete the tracker
  bool isNew; //Indicates whether it was created recently or if it had been created previously
  bool empty; //Indicates if this detection hasn't been assigned values yet
  bool windowSentToRecognizer; //Indicates if this detection has already been sent to the facial recognizer
  bool isRecognized; //Indicates if the detection has already been recognized
  std::string name; //Name assigned to the detection
  cv::Mat img; //Image inside the rectangle
};

class trackerWindows: public QObject {

  Q_OBJECT

  //Object representing the list of detections over time for zero-degree rectangular detections
  std::vector < rectangleDetection > detectedObjectsList;

  //Object representing the list of detections over time for possibly rotated rectangular detections
  std::vector < rotatedRectDetection > detectedRotatedObjectsList;

  //List of detections that need to be sent to the recognizer
  QList < imageTransaction > listToRecognize;

  //List that will store recognized images, their IDs, and the time they were sent to the recognizer
  QMap < int, imageTransaction > listRecognizedImages;

  //__________OBJECTS CONTROLLING ID ASSIGNMENT________________//
  int idNext;
  QList < int > idUnassigned;
  //_____________________________________________________________________//

  const int defaultInitiaLpunctuation;
  const int defaultMinimumPunctuationToRecognize;
  const double defaultEps;

  int initiaLpunctuation;
  int minimumPunctuationToRecognize;
  double eps;

  public:

    trackerWindows(QObject * parent = 0);

  int getId(); //Gets an available ID
  void freeId(int id); //Frees an ID

  //set functions
  void setDefaultValues();
  void setInitiaLpunctuation(int nFrames);
  void setMinimumPunctuationToRecognize(int nFrames);
  void setEps(double valueEps);

  //get functions
  int getInitiaLpunctuation() const;
  int getMinimumPunctuationToRecognize() const;
  double getEps() const;

  //Performs tracking on rectangular detections
  void groupRectsDetection(std::vector < rectangleDetection > & rectangleDetectionList);
  void checkRecognition();

  //Performs tracking on possibly rotated detections
  void groupRectRotatedDetection(std::vector < rotatedRectDetection > & rectRotatedDetectionList);
  void checkRecognitionRectRotated();

  public slots:
    void reset(); //Resets the variables involved in tracking
  void recognizedImage(imageTransaction newImageTransaction);
  void newGroupDetections(std::vector < cv::Mat > listDetectedObjects, std::vector < cv::Rect > coordinatesDetectedObjects);
  void newGroupDetections(std::vector < cv::Mat > listDetectedObjects, std::vector < cv::RotatedRect > coordinatesDetectedObjects);

  signals:
    void setTextInDetection(const std::vector < cv::Point > listPoints,
      const std::vector < std::string > listText);
  void recognizeImagesList(QList < imageTransaction > listToRecognize);

};

#endif
