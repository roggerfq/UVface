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

#ifndef RECOGNIZERFACIAL_H
#define RECOGNIZERFACIAL_H

//Forward declarations of custom classes
class DATA_BASE;
class ABSTRACT_DESCRIPTOR;
class DICTIONARY;
class imageTransaction; //Used for transaction of detection over time

//STL
#include <vector>
//openCV
#include "opencv2/opencv.hpp"
//EIGEN
#include <eigen/Eigen/Dense>
#include <eigen/Eigen/Sparse>
//QT
#include <QThread>
#include <QMutexLocker>
#include <QMutex>
#include <QMap>
#include <QStringList>
#include <QDebug>//DELETE

//Forward declarations of QT classes
class QStackedWidget;

class RECOGNIZER_FACIAL: public QObject {
  Q_OBJECT

  //These will be used in case a slot needs to be terminated without stopping the event loop prematurely, see bool calculateDescriptors()
  volatile bool stopped;
  QMutex mutex;

  DATA_BASE * dataBase; //Database
  QString pathDataBase;
  QStackedWidget * widgetsDescriptors; //Stack that will store the graphic part of each descriptor
  std::vector < ABSTRACT_DESCRIPTOR * > descriptors; //Stack that will store each descriptor
  ABSTRACT_DESCRIPTOR * p_descriptor; //Pointer used to handle each descriptor in descriptors
  DICTIONARY * myDictionary;

  cv::Mat * descriptorsBase; //Stores the base descriptors for a type 1 descriptor
  cv::Mat * descriptor_end; //Stores the final descriptors for a type 1 descriptor
  std::vector < int > ithRows; //Stores the upper limit of each row of descriptorsBase corresponding to each image group  
  std::vector < int > ithRowsId; //Stores the id corresponding to each section represented by ithRows
  QMap < int, QString > nameUsersListAndId; //This list associates the id with the username 
  Eigen::MatrixXf * descriptor_out; //Used to deliver the information to the sparse solution
  Eigen::MatrixXf * descriptorTemp; //Temporary used for recognition
  cv::Mat tempImg; //Temporary used for recognition
  Eigen::Map < Eigen::MatrixXf > * mf; //Used to transform from an openCV matrix to an EIGEN matrix
  float thresholdFaceRecognizer;
  int lengthStackImages;
  int newWidthImages;
  int newHighImages;
  bool flagResizeImages;
  cv::Mat imageToRecognize;

  //Values related to recognition
  float expectedPercentageDifference; //Score given to the recognition
  int id; //Numerical identifier for the recognition class
  double elapsedRecognition; //Computation time during recognition

  public:
    RECOGNIZER_FACIAL();
  ~RECOGNIZER_FACIAL();

  //Get functions
  DATA_BASE * getDataBase() const;
  QStackedWidget * getWidgetsDescriptors() const;

  //Sparse solution get functions
  bool dictionaryIsBuilt() const; //Returns true if the dictionary is already built, false otherwise
  DICTIONARY * getDictionary() const;
  QMap < int, QString > getNameUsersListAndId() const; //Returns a list associated with each valid id for the current dictionary
  QStringList getListNameUsers() const; //Returns the list of names only, without the ids
  int get_lm() const;
  int get_nc() const;
  int get_numberZeros() const;
  float get_ck() const;
  int get_m() const;
  int get_n() const;
  int get_numberDescriptors() const;
  float get_threshold();

  //Sparse solution set functions
  void set_lm(int new_lm); //Sets the number of descriptors to pass to the FAST FILTER
  void set_nc(int new_nc); //Sets the maximum number of descriptors that can be evaluated
  void set_numberZeros(int num); //Sets the maximum expected number of zeros in the sparse solution 
  void set_ck(float new_ck); //Sets the regularization factor in the equation min |x|1+ck(Dx-b)^2 (|x|1 = l1 norm of x)

  //Test configuration functions
  void setConfigurationTest(int width, int high, int lengthStack);
  void loadConfigTest();
  int get_newWidthImages() const;
  int get_newHighImages() const;
  int get_lengthStackImages() const;

  QString nameDescriptor(int index) const;
  QString currentNameDescriptor() const;
  void loadSettings(); //Sets the previously stored parameters for the current descriptor
  bool applySettings();
  void saveInfoSparseSolution(); //Here we save the sparse solution information and the association of names and ids
  void saveThreshold(float threshold); //Here we store the comparison threshold for recognition
  void loadInfoSparseSolution(); //Here we load the sparse solution information and the association of names and ids
  void setSparseSolution();
  bool calculateDescriptors();

  void getImgTest(QMap < QString, QStringList > & listDirAndImgTest);
  void recognizedImage(const cv::Mat img, QString & recognitionResult, double & score); //Added only to test algorithms
  cv::Mat corruptedPixelImg(cv::Mat & img, double percentage);

  public slots:
    void stop();
  void enableRecognition();
  void startLoadDataBase(QString namePathDataBase);
  void startCalculateDescriptors();
  void startRecognizeFaceImage(cv::Mat img);
  void recognizeImagesList(QList < imageTransaction > listToRecognize);
  void recognizedImage(std::vector < cv::Mat > listDetectedObjects, std::vector < cv::Rect > coordinatesDetectedObjects);
  void recognizedImage(std::vector < cv::Mat > listDetectedObjects, std::vector < cv::RotatedRect > coordinatesDetectedObjectsRotated);
  void generateTest();

  signals:
    void editingDescriptor(); //Informs if the selected descriptor has been edited
  void loadedDatabase(); //This signal should be emitted when the database has finished loading
  void calculationEndDescriptors(bool flag);
  /*This signal should be emitted when the user cancels the descriptor calculation,
  the flag should be true if the calculation completed successfully or false if the user canceled the operation*/
  void currentUserProcessing(int n);
  void currentUserProcessingInfo(const QString & info);
  void testResultImageInformation(const QString & infoText);
  void zeroDescriptorsOrZeroImages(const QString & infoText); //Emitted to inform that the number of descriptors or images is zero
  void enabledProgressbar(bool flag); //This signal enables or disables progressDialogDescriptorsCalculation
  void recognizedImage(imageTransaction newImageTransaction);
  void recognizedImage(const cv::Point point,
    const QString text, cv::Mat img); //For the result of the images
  void setPlotSparseSolution();
  void testWasGenerated();

};

#endif
