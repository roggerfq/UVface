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

#include "guiConfigDetector.h"
#include "detector.h"
#include <QLineEdit>
#include <QCheckBox>
#include <QPushButton>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QGroupBox>
#include <QPlastiqueStyle>
#include <QDir>
#include <QFileInfo>
#include <QFileDialog>
#include <QMessageBox>
#include <QIntValidator>

#define inTest 0

#if inTest == 1
void rotateImg(cv::Mat & img) {
  static cv::Size sz = img.size();
  cv::transpose(img, img);
  cv::flip(img, img, 1);
  resize(img, img, sz);
}
#endif

// Path of the last loaded detector
QString LAST_PATH_DETECTOR = QDir::homePath();

threadDetector::threadDetector() {
  objectDetector = NULL;
  detectorIsLoad = false;
  normalizeRotation = false;
  doubleList = false;
  command = 0; // Means it does nothing
}

threadDetector::~threadDetector() {
  stop();
  wait();
  if (objectDetector != NULL) {
    delete objectDetector;
    objectDetector = NULL;
  }
  std::cout << "THE DETECTOR THREAD HAS BEEN DESTROYED\n";
}

void threadDetector::stop() {
  std::cout << "STOP IN threadDetector::stop()\n";
  QMutexLocker locker(&mutex);
  stopped = true;
  command = 0;
  std::cout << "STOP FINISHED IN threadDetector::stop()\n";
}

int threadDetector::detectObjectVideoCamera(int device) {
  /*
  NOTE: Returns:
  -1 when everything went well
  0 when it cannot open the device
  greater than zero, when the frame size exceeds the maximum window size, it returns that value
  */

  if (isRunning()) { // In case there is an ongoing task
    stop();
    wait();
  }

  // cap.open(device);

  if (device == -1) {
    // Check DEVICE_URL
    const char * DEVICE_URL = std::getenv("DEVICE_URL");
    if (DEVICE_URL != NULL) {
      cap.open(DEVICE_URL);
      std::cout << "Opening: " << DEVICE_URL << std::endl;
    }
  } else {
    // Check other devices
    cap.open(device);
  }

  if (!cap.isOpened()) {
    emit clearLabelVideo(); // We leave the graphic label clean
    return 0; // In case opening the device fails
  }

  // Protection against images larger than allowed size
  int tempMaxSide = std::max(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
  if (tempMaxSide > getSizeMaxWindow()) {
    cap.release();
    emit clearLabelVideo(); // We leave the graphic label clean
    return tempMaxSide; // In case the search window size is smaller than the largest side of a frame
  }

  command = 1;
  stopped = false;
  start();
  return -1;
}

int threadDetector::detectObjectVideoFile(std::string nameFileVideo) {
  /*
  NOTE: Returns:
  -1 when everything went well
  0 when it cannot open the device
  greater than zero, when the frame size exceeds the maximum window size, it returns that value
  */

  if (isRunning()) { // In case there is an ongoing task
    stop();
    wait();
  }

  cap.open(nameFileVideo);
  if (!cap.isOpened()) {
    emit clearLabelVideo(); // We leave the graphic label clean
    return 0; // In case opening the device fails
  }

  // Protection against images larger than allowed size
  int tempMaxSide = std::max(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
  if (tempMaxSide > getSizeMaxWindow()) {
    cap.release();
    emit clearLabelVideo(); // We leave the graphic label clean
    return tempMaxSide;
  }

  command = 2;
  stopped = false;
  start();

  return -1;
}

int threadDetector::detectObjectImageFile(std::string nameFileImage) {
  /*
  NOTE: Returns:
  -1 when everything went well
  0 when it cannot open the device
  greater than zero, when the frame size exceeds the maximum window size, it returns that value
  */

  if (isRunning()) { // In case there is an ongoing task
    stop();
    wait();
  }

  currentImage = cv::imread(nameFileImage);
  if (!currentImage.data) {
    emit clearLabelVideo(); // We leave the graphic label clean
    return 0; // Protection against corrupted images
  }

  // Protection against images larger than allowed size
  int tempMaxSide = std::max(currentImage.rows, currentImage.cols);
  if (tempMaxSide > getSizeMaxWindow()) {
    emit clearLabelVideo(); // We leave the graphic label clean
    return tempMaxSide;
  }

  command = 3;
  stopped = false;
  start();

  return -1;
}

void threadDetector::startDetectObjectVideoCamera() {

  //______________________________________________________________________________________________//
  /*The 3 lines below are due to the fact that when recognizing from an image, the recognition cannot be canceled, otherwise, the recognizer will not have enough time to analyze the detections, and the result will not be visible. Therefore, the recognition is not canceled in threadDetector::startDetectObjectImageFile(), forcing it to be done here*/
  emit resetTrackerWindows(); // Only to ensure this is the last event the tracker thread will process (not needed here)
  emit stopRecognizer(); // Stop the recognition
  emit enableRecognition(); // Re-enable recognition
  //______________________________________________________________________________________________//

  std::cout << "Entering startDetectObjectVideoCamera()\n";

  // Since everything went well, we emit the width and height of the frame before starting
  emit setSizeFrame(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));

  if (!groupingRectangles) {

    while (true) {

      {
        QMutexLocker locker( & mutex);
        if (stopped) {
          stopped = false;
          break;
        }
      }

      cap >> frameTemp;

      cv::flip(frameTemp, frame, 1);
      objectDetector->detectObjectRectanglesUngrouped(frame);
      emit imageReady(frame.clone());

    }

  } else if (normalizeRotation) {

    //___________________________________________________________________________//
    while (true) {

      {
        QMutexLocker locker( & mutex);
        if (stopped) {
          stopped = false;
          break;
        }
      }

      cap >> frameTemp;
      cv::flip(frameTemp, frame, 1);
      std::vector < cv::RotatedRect > coordinatesDetectedObjectsRotated; // When the angle is normalized
      std::vector < cv::Mat > listDetectedObjects; // Objects that will be detected
      objectDetector->detectObjectRectanglesRotatedGrouped(frame, & listDetectedObjects, & coordinatesDetectedObjectsRotated);

      emit listCoordinatesAndDetectedObjectsRotated(listDetectedObjects, coordinatesDetectedObjectsRotated);

      emit imageReady(frame.clone());

    }
    //___________________________________________________________________________//

  } else {

    //____________________________________________________________________________//
    while (true) {

      {
        QMutexLocker locker( & mutex);
        if (stopped) {
          stopped = false;
          break;
        }
      }

      cap >> frameTemp;
      cv::flip(frameTemp, frame, 1);
      std::vector < cv::Rect > coordinatesDetectedObjects; // When the angle is not normalized
      std::vector < cv::Mat > listDetectedObjects; // Objects that will be detected
      objectDetector->detectObjectRectanglesGroupedZeroDegrees(frame, & listDetectedObjects, & coordinatesDetectedObjects, doubleList);

      emit listCoordinatesAndDetectedObjects(listDetectedObjects, coordinatesDetectedObjects);

      emit imageReady(frame.clone());

    }
    //___________________________________________________________________________//

  }

  cap.release(); // Close the device previously opened in detectObjectVideoCamera(int device)
  command = 0;
  emit resetTrackerWindows(); // Only to ensure this is the last event the tracker thread will process
  emit stopRecognizer(); // Stop the recognition
  emit enableRecognition(); // Re-enable recognition
  emit finishedDetection();
  // emit clearLabelVideo(); // Leave the graphic label clean (optional)

  std::cout << "Exiting startDetectObjectVideoCamera()\n";

}

void threadDetector::startDetectObjectVideoFile() {

  //______________________________________________________________________________________________//
  /*The 3 lines below are due to the fact that when recognizing from an image, the recognition cannot be canceled; otherwise, the recognizer will not have enough time to analyze the detections, and the result will not be visible. Therefore, the recognition is not canceled in threadDetector::startDetectObjectImageFile(), forcing it to be done here*/
  emit resetTrackerWindows(); // Only to ensure this is the last event the tracker thread will process (not needed here)
  emit stopRecognizer(); // Stop the recognition
  emit enableRecognition(); // Re-enable recognition
  //______________________________________________________________________________________________//

  std::cout << "Entering startDetectObjectVideoFile()\n";

  // Since everything went well, we emit the width and height of the frame before starting
  emit setSizeFrame(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));

  if (!groupingRectangles) {

    while (true) {

      {
        QMutexLocker locker( & mutex);
        if (stopped) {
          stopped = false;
          break;
        }
      }

      cap >> frame2;

      if (frame2.empty())
        break;

      #if inTest == 1
      rotateImg(frame2);
      #endif

      objectDetector->detectObjectRectanglesUngrouped(frame2);
      emit imageReady(frame2.clone());

    }

  } else if (normalizeRotation) {

    //______________________________________________________________________//
    while (true) {

      {
        QMutexLocker locker( & mutex);
        if (stopped) {
          stopped = false;
          break;
        }
      }

      cap >> frame2;

      if (frame2.empty())
        break;

      #if inTest == 1
      rotateImg(frame2);
      #endif

      std::vector < cv::RotatedRect > coordinatesDetectedObjectsRotated; // When the angle is normalized
      std::vector < cv::Mat > listDetectedObjects; // Objects that will be detected
      objectDetector->detectObjectRectanglesRotatedGrouped(frame2, & listDetectedObjects, & coordinatesDetectedObjectsRotated);

      emit listCoordinatesAndDetectedObjectsRotated(listDetectedObjects, coordinatesDetectedObjectsRotated);

      emit imageReady(frame2.clone());

    }
    //_________________________________________________________________________//

  } else {

    //__________________________________________________________________________//

    while (true) {

      {
        QMutexLocker locker( & mutex);
        if (stopped) {
          stopped = false;
          break;
        }
      }

      cap >> frame2;

      if (frame2.empty())
        break;

      #if inTest == 1
      rotateImg(frame2);
      #endif

      std::vector < cv::Rect > coordinatesDetectedObjects; // When the angle is not normalized
      std::vector < cv::Mat > listDetectedObjects; // Objects that will be detected
      objectDetector->detectObjectRectanglesGroupedZeroDegrees(frame2, & listDetectedObjects, & coordinatesDetectedObjects, doubleList);

      emit listCoordinatesAndDetectedObjects(listDetectedObjects, coordinatesDetectedObjects);

      emit imageReady(frame2.clone());

    }

    //__________________________________________________________________________//

  }

  cap.release();
  command = 0;
  emit resetTrackerWindows(); // Only to ensure this is the last event the tracker thread will process
  emit stopRecognizer(); // Stop the recognition
  emit enableRecognition(); // Re-enable recognition
  emit finishedDetection();
  // emit clearLabelVideo(); // Leave the graphic label clean (optional)

  std::cout << "Exiting startDetectObjectVideoFile()\n";

}

void threadDetector::startDetectObjectImageFile() {

  //______________________________________________________________________________________________//
  /*The 3 lines below are due to the fact that when recognizing from an image, the recognition cannot be canceled; otherwise, the recognizer will not have enough time to analyze the detections, and the result will not be visible. Therefore, the recognition is not canceled in threadDetector::startDetectObjectImageFile(), forcing it to be done here*/
  // emit resetTrackerWindows(); // Only to ensure this is the last event the tracker thread will process (not needed here)
  emit stopRecognizer(); // Stop the recognition
  emit enableRecognition(); // Re-enable recognition
  //______________________________________________________________________________________________//

  std::cout << "Entering startDetectObjectImageFile()\n";

  // Since everything went well, we emit the width and height of the frame before starting
  emit setSizeFrame(currentImage.cols, currentImage.rows);

  QMutexLocker locker( & mutex);

  if (!groupingRectangles) {

    objectDetector->detectObjectRectanglesUngrouped(currentImage);
    emit imageReady(currentImage.clone());

  } else if (normalizeRotation) {

    //_____________________________________________________________________________________//

    std::vector < cv::RotatedRect > coordinatesDetectedObjectsRotated; // When the angle is normalized
    std::vector < cv::Mat > listDetectedObjects; // Objects that will be detected
    objectDetector->detectObjectRectanglesRotatedGrouped(currentImage, & listDetectedObjects, & coordinatesDetectedObjectsRotated);

    emit listCoordinatesAndDetectedObjectsRotated_img(listDetectedObjects, coordinatesDetectedObjectsRotated);

    emit imageReady(currentImage.clone());
    //______________________________________________________________________________________//

  } else {

    //______________________________________________________________________________________//

    std::vector < cv::Rect > coordinatesDetectedObjects; // When the angle is not normalized
    std::vector < cv::Mat > listDetectedObjects; // Objects that will be detected
    objectDetector->detectObjectRectanglesGroupedZeroDegrees(currentImage, & listDetectedObjects, & coordinatesDetectedObjects, doubleList);

    emit listCoordinatesAndDetectedObjects_img(listDetectedObjects, coordinatesDetectedObjects);

    emit imageReady(currentImage.clone());
    //_______________________________________________________________________________________//

  }

  command = 0;
  emit finishedDetection();

}

void threadDetector::loadDetector(std::string fileName) {

  if (isRunning()) { // In case there is an ongoing task
    stop();
    wait();
  }

  QMutexLocker locker( & mutex);

  if (objectDetector != NULL) {
    delete objectDetector;
    objectDetector = NULL;
  }

  objectDetector = new CASCADE_CLASSIFIERS_EVALUATION(fileName);
  detectorIsLoad = true;

}

bool threadDetector::setDevice(int i) {

  if (cap.isOpened()) cap.release(); // In case a device was previously open
  cap.open(i);
  if (!cap.isOpened()) return false; // In case opening the device fails
  return true;

}

bool threadDetector::setDevice(std::string videoFile) {

  return false;
}

bool threadDetector::detectorIsReady() const {
  return detectorIsLoad;
}

int threadDetector::getCommand() const {
  return command;
}

int threadDetector::getSizeMaxWindow() const {
  return objectDetector -> getSizeMaxWindow();
}

void threadDetector::run() {

  switch (command) {
  case 1:
    startDetectObjectVideoCamera();
    break;
  case 2:
    startDetectObjectVideoFile();
    break;
  case 3:
    startDetectObjectImageFile();
    break;

  }

}

WINDOW::WINDOW() {

  setMinimumSize(640, 480);
  //setBackgroundRole(QPalette::Base);
  setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  setScaledContents(true);

  QPalette palette;
  palette.setColor(QPalette::Background, Qt::black);
  setPalette(palette);
  //setWindowFlags(Qt::Window | Qt::WindowTitleHint | Qt::CustomizeWindowHint);

  buttonNext = new QPushButton("next");
  buttonBack = new QPushButton("back");
  buttonCapture = new QPushButton("Capture");
  buttonLoadFiles = new QPushButton("Files");

  labelMenu = new QLabel;
  labelMenu -> hide();
  QHBoxLayout * layoutMenu = new QHBoxLayout;
  layoutMenu -> addWidget(buttonNext);
  layoutMenu -> addWidget(buttonBack);
  layoutMenu -> addWidget(buttonCapture);
  layoutMenu -> addWidget(buttonLoadFiles);
  labelMenu -> setLayout(layoutMenu);
  QHBoxLayout * layoutPrincipalMenu = new QHBoxLayout;
  layoutPrincipalMenu -> addWidget(labelMenu);
  setLayout(layoutPrincipalMenu);

}

GUI_DETECTOR::GUI_DETECTOR(threadDetector * myThreadDetector): myThreadDetector(myThreadDetector) {

  numberStrongLearns = 0;
  flagEdition = false;

  // QPushButton
  buttonLoadDetector = new QPushButton("Load a detector");
  buttonLoadDetector -> setFixedWidth(200);
  buttonLoadDetector -> setAutoDefault(false);
  connect(buttonLoadDetector, SIGNAL(clicked()), this, SLOT(loadDetector()));

  buttonApplySettings = new QPushButton("Apply");
  buttonApplySettings -> setAutoDefault(false);
  connect(buttonApplySettings, SIGNAL(clicked()), this, SLOT(setConfig()));

  // QLineEdit
  lineEditDegreesDetections = new QLineEdit;
  QRegExp reDegreesDetections("([-]?[\\d][0-9]*([\\s][-]?[\\d][0-9]*)*)");
  QRegExpValidator * validatorDegreesDetections = new QRegExpValidator(reDegreesDetections, this);
  lineEditDegreesDetections -> setValidator(validatorDegreesDetections);
  connect(lineEditDegreesDetections, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));

  lineEditSizeBase = new QLineEdit;
  QRegExp reSizeBase("([1-9][0-9]*)");
  QRegExpValidator * validatorSizeBase = new QRegExpValidator(reSizeBase, this);
  lineEditSizeBase -> setValidator(validatorSizeBase);
  lineEditSizeBase -> setFixedWidth(40);
  connect(lineEditSizeBase, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));

  lineEditFactorScaleWindow = new QLineEdit;
  QRegExp reFactorScaleWindow("([0-9]*[//.][0-9]*)");
  QRegExpValidator * validatorFactorScaleWindow = new QRegExpValidator(reFactorScaleWindow, this);
  lineEditFactorScaleWindow -> setValidator(validatorFactorScaleWindow);
  lineEditFactorScaleWindow -> setFixedWidth(40);
  connect(lineEditFactorScaleWindow, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));

  lineEditStepWindow = new QLineEdit;
  QRegExp reStepWindow("([0-9]*[//.][0-9]*)");
  QRegExpValidator * validatorStepWindow = new QRegExpValidator(reStepWindow, this);
  lineEditStepWindow -> setValidator(validatorStepWindow);
  lineEditStepWindow -> setFixedWidth(40);
  connect(lineEditStepWindow, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));

  lineEditSizeMaxWindow = new QLineEdit;
  QRegExp reSizeMaxWindow("([1-9][0-9]*)");
  QRegExpValidator * validatorSizeMaxWindow = new QRegExpValidator(reSizeMaxWindow, this);
  lineEditSizeMaxWindow -> setValidator(validatorSizeMaxWindow);
  lineEditSizeMaxWindow -> setFixedWidth(40);
  connect(lineEditSizeMaxWindow, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));

  lineEditGroupThreshold = new QLineEdit;
  QRegExp reGroupThreshold("([0-9][0-9]*)");
  QRegExpValidator * validatorGroupThreshold = new QRegExpValidator(reGroupThreshold, this);
  lineEditGroupThreshold -> setValidator(validatorGroupThreshold);
  lineEditGroupThreshold -> setFixedWidth(40);
  connect(lineEditGroupThreshold, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));
  connect(lineEditGroupThreshold, SIGNAL(textEdited(const QString & )), this, SLOT(setEnabledOptionNotGroup()));

  lineEditEps = new QLineEdit;
  QRegExp reEps("([0-9]*[//.][0-9]*)");
  QRegExpValidator * validatorEps = new QRegExpValidator(reEps, this);
  lineEditEps -> setValidator(validatorEps);
  lineEditEps -> setFixedWidth(40);
  connect(lineEditEps, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));

  lineEditNumberClassifiersUsed = new QLineEdit;
  QRegExp reNumberClassifiersUsed("([1-9][0-9]*)");
  QRegExpValidator * validatorNumberClassifiersUsed = new QRegExpValidator(reNumberClassifiersUsed, this);
  lineEditNumberClassifiersUsed -> setValidator(validatorNumberClassifiersUsed);
  lineEditNumberClassifiersUsed -> setFixedWidth(40);
  connect(lineEditNumberClassifiersUsed, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));

  lineEditLineThicknessRectangles = new QLineEdit;
  QRegExp reThicknessRectangles("([1-9][0-9]*)");
  QRegExpValidator * validatorThicknessRectangles = new QRegExpValidator(reThicknessRectangles, this);
  lineEditLineThicknessRectangles -> setValidator(validatorThicknessRectangles);
  lineEditLineThicknessRectangles -> setFixedWidth(40);
  connect(lineEditLineThicknessRectangles, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));

  lineEditColorRectanglesR = new QLineEdit;
  lineEditColorRectanglesR -> setValidator(new QIntValidator(0, 255));
  lineEditColorRectanglesR -> setFixedWidth(40);
  connect(lineEditColorRectanglesR, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));
  lineEditColorRectanglesG = new QLineEdit;
  lineEditColorRectanglesG -> setValidator(new QIntValidator(0, 255));
  lineEditColorRectanglesG -> setFixedWidth(40);
  connect(lineEditColorRectanglesG, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));
  lineEditColorRectanglesB = new QLineEdit;
  lineEditColorRectanglesB -> setValidator(new QIntValidator(0, 255));
  lineEditColorRectanglesB -> setFixedWidth(40);
  connect(lineEditColorRectanglesB, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));

  // QCheckBox
  checkBoxFlagActivateSkinColor = new QCheckBox("Activate skin color");
  connect(checkBoxFlagActivateSkinColor, SIGNAL(stateChanged(int)), this, SLOT(edition()));
  connect(checkBoxFlagActivateSkinColor, SIGNAL(stateChanged(int)), this, SLOT(activeConfigHsv(int)));

  checkBoxNormalizeRotation = new QCheckBox("Normalize rotations");
  connect(checkBoxNormalizeRotation, SIGNAL(stateChanged(int)), this, SLOT(normalizeRotation(int)));

  checkBoxDoubleList = new QCheckBox("Double list");
  connect(checkBoxDoubleList, SIGNAL(stateChanged(int)), this, SLOT(edition()));
  
  checkBoxFlagExtractColorImages = new QCheckBox("Extract color detections");
  connect(checkBoxFlagExtractColorImages, SIGNAL(stateChanged(int)), this, SLOT(edition()));

  //QLabel
  textNumberStrongLearns = new QLabel(QString("<font color=red>MAX strongLearns</font></h2>"));

  QGridLayout * layoutConfig = new QGridLayout;
  layoutConfig -> addWidget(new QLabel(QString::fromUtf8("Detection degrees")), 0, 0, 1, 1);
  layoutConfig -> addWidget(lineEditDegreesDetections, 0, 1, 1, 3);
  layoutConfig -> addWidget(new QLabel(QString::fromUtf8("Base size")), 1, 0, 1, 1);
  layoutConfig -> addWidget(lineEditSizeBase, 1, 1, 1, 1);
  layoutConfig -> addWidget(new QLabel(QString::fromUtf8("Maximum size")), 1, 2, 1, 1);
  layoutConfig -> addWidget(lineEditSizeMaxWindow, 1, 3, 1, 1);
  layoutConfig -> addWidget(new QLabel(QString::fromUtf8("Step factor")), 2, 0, 1, 1);
  layoutConfig -> addWidget(lineEditStepWindow, 2, 1, 1, 1);
  layoutConfig -> addWidget(new QLabel(QString::fromUtf8("Scale factor")), 2, 2, 1, 1);
  layoutConfig -> addWidget(lineEditFactorScaleWindow, 2, 3, 1, 1);
  layoutConfig -> addWidget(new QLabel(QString::fromUtf8("GroupThreshold")), 3, 0, 1, 1);
  layoutConfig -> addWidget(lineEditGroupThreshold, 3, 1, 1, 1);
  layoutConfig -> addWidget(new QLabel(QString::fromUtf8("EPS")), 3, 2, 1, 1);
  layoutConfig -> addWidget(lineEditEps, 3, 3, 1, 1);
  layoutConfig -> addWidget(new QLabel(QString::fromUtf8("# strongLearns")), 4, 0, 1, 1);
  layoutConfig -> addWidget(lineEditNumberClassifiersUsed, 4, 1, 1, 1);
  layoutConfig -> addWidget(textNumberStrongLearns, 4, 2, 1, 1);

  QGroupBox * groupBoxConfig = new QGroupBox(tr("Search window configurations"));
  groupBoxConfig -> setLayout(layoutConfig);
  groupBoxConfig -> setStyle(new QPlastiqueStyle);

  QGridLayout * layoutRectDetection = new QGridLayout;
  layoutRectDetection -> addWidget(new QLabel("Thickness"), 0, 0, 1, 1);
  layoutRectDetection -> addWidget(lineEditLineThicknessRectangles, 0, 1, 1, 1);
  layoutRectDetection -> addWidget(new QLabel(QString("<font color=red>R</font></h2>")), 0, 2, 1, 1);
  layoutRectDetection -> addWidget(lineEditColorRectanglesR, 0, 3, 1, 1);
  layoutRectDetection -> addWidget(new QLabel(QString("<font color=green>G</font></h2>")), 0, 4, 1, 1);
  layoutRectDetection -> addWidget(lineEditColorRectanglesG, 0, 5, 1, 1);
  layoutRectDetection -> addWidget(new QLabel(QString("<font color=blue>B</font></h2>")), 0, 6, 1, 1);
  layoutRectDetection -> addWidget(lineEditColorRectanglesB, 0, 7, 1, 1);

  QGroupBox * groupBoxRectDetection = new QGroupBox(tr("Shape and color of detection window"));
  groupBoxRectDetection -> setLayout(layoutRectDetection);
  groupBoxRectDetection -> setStyle(new QPlastiqueStyle);

  QGridLayout * layoutExtraSettings = new QGridLayout;
  layoutExtraSettings -> addWidget(checkBoxFlagActivateSkinColor, 0, 0, 1, 2, Qt::AlignLeft);
  layoutExtraSettings -> addWidget(checkBoxNormalizeRotation, 0, 2, 1, 2, Qt::AlignLeft);
  layoutExtraSettings -> addWidget(checkBoxDoubleList, 1, 0, 1, 2, Qt::AlignLeft);
  layoutExtraSettings -> addWidget(checkBoxFlagExtractColorImages, 1, 2, 1, 2, Qt::AlignRight);

  //________________________HSV CONFIGURATION GRAPH PANEL______________________________________//

  lineEditHmin = new QLineEdit;
  connect(lineEditHmin, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));
  lineEditHmin -> setFixedWidth(40);
  lineEditHmin -> setValidator(new QIntValidator(0, 255));
  layoutExtraSettings -> addWidget(new QLabel(QString::fromUtf8("Min H")), 2, 0, 1, 1, Qt::AlignLeft);
  layoutExtraSettings -> addWidget(lineEditHmin, 2, 1, 1, 1, Qt::AlignLeft);

  lineEditHmax = new QLineEdit;
  connect(lineEditHmax, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));
  lineEditHmax -> setFixedWidth(40);
  lineEditHmax -> setValidator(new QIntValidator(0, 255));
  layoutExtraSettings -> addWidget(new QLabel(QString::fromUtf8("Max H")), 2, 2, 1, 1, Qt::AlignRight);
  layoutExtraSettings -> addWidget(lineEditHmax, 2, 3, 1, 1, Qt::AlignRight);

  lineEditSmin = new QLineEdit;
  connect(lineEditSmin, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));
  lineEditSmin -> setFixedWidth(40);
  lineEditSmin -> setValidator(new QIntValidator(0, 255));
  layoutExtraSettings -> addWidget(new QLabel(QString::fromUtf8("Min S")), 3, 0, 1, 1, Qt::AlignLeft);
  layoutExtraSettings -> addWidget(lineEditSmin, 3, 1, 1, 1, Qt::AlignLeft);

  lineEditSmax = new QLineEdit;
  connect(lineEditSmax, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));
  lineEditSmax -> setFixedWidth(40);
  lineEditSmax -> setValidator(new QIntValidator(0, 255));
  layoutExtraSettings -> addWidget(new QLabel(QString::fromUtf8("Max S")), 3, 2, 1, 1, Qt::AlignRight);
  layoutExtraSettings -> addWidget(lineEditSmax, 3, 3, 1, 1, Qt::AlignRight);

  lineEditVmin = new QLineEdit;
  connect(lineEditVmin, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));
  lineEditVmin -> setFixedWidth(40);
  lineEditVmin -> setValidator(new QIntValidator(0, 255));
  layoutExtraSettings -> addWidget(new QLabel(QString::fromUtf8("Min V")), 4, 0, 1, 1, Qt::AlignLeft);
  layoutExtraSettings -> addWidget(lineEditVmin, 4, 1, 1, 1, Qt::AlignLeft);

  lineEditVmax = new QLineEdit;
  connect(lineEditVmax, SIGNAL(textEdited(const QString & )), this, SLOT(edition()));
  lineEditVmax -> setFixedWidth(40);
  lineEditVmax -> setValidator(new QIntValidator(0, 255));
  layoutExtraSettings -> addWidget(new QLabel(QString::fromUtf8("Max V")), 4, 2, 1, 1, Qt::AlignRight);
  layoutExtraSettings -> addWidget(lineEditVmax, 4, 3, 1, 1, Qt::AlignRight);

  setEnabledHsvConfig(false);

  //_____________________________________________________________________________________//

  groupBoxExtraSettings = new QGroupBox(tr("Extra configurations"));
  groupBoxExtraSettings -> setLayout(layoutExtraSettings);
  groupBoxExtraSettings -> setStyle(new QPlastiqueStyle);

  QGridLayout * layoutPrincipal = new QGridLayout;
  layoutPrincipal -> addWidget(buttonLoadDetector, 0, 2, 1, 1, Qt::AlignCenter);
  layoutPrincipal -> addWidget(groupBoxConfig, 1, 0, 4, 4);
  layoutPrincipal -> addWidget(groupBoxRectDetection, 5, 0, 2, 4);
  layoutPrincipal -> addWidget(groupBoxExtraSettings, 7, 0, 2, 4);
  layoutPrincipal -> addWidget(buttonApplySettings, 9, 3, 1, 1, Qt::AlignRight);

  setEnabledFalseAllConfig();
  setLayout(layoutPrincipal);

  setWindowTitle(QString::fromUtf8("FACE DETECTOR CONFIGURATION"));
}

void GUI_DETECTOR::setEnabledFalseAllConfig() {

  //QPushButton
  buttonApplySettings -> setEnabled(false);

  //QLineEdit
  lineEditDegreesDetections -> setEnabled(false);
  lineEditSizeBase -> setEnabled(false);
  lineEditFactorScaleWindow -> setEnabled(false);
  lineEditStepWindow -> setEnabled(false);
  lineEditSizeMaxWindow -> setEnabled(false);
  lineEditGroupThreshold -> setEnabled(false);
  lineEditEps -> setEnabled(false);
  lineEditNumberClassifiersUsed -> setEnabled(false);
  lineEditLineThicknessRectangles -> setEnabled(false);
  lineEditColorRectanglesR -> setEnabled(false);
  lineEditColorRectanglesG -> setEnabled(false);
  lineEditColorRectanglesB -> setEnabled(false);

  //QCheckBox 
  checkBoxFlagActivateSkinColor -> setEnabled(false);
  checkBoxNormalizeRotation -> setEnabled(false);
  checkBoxDoubleList -> setEnabled(false);
  checkBoxFlagExtractColorImages -> setEnabled(false);

}

void GUI_DETECTOR::setEnabledTrueConfigDefault() {

  //QPushButton
  buttonApplySettings -> setEnabled(false);

  //QLineEdit
  lineEditDegreesDetections -> setEnabled(true);
  lineEditSizeBase -> setEnabled(true);
  lineEditFactorScaleWindow -> setEnabled(true);
  lineEditStepWindow -> setEnabled(true);
  lineEditSizeMaxWindow -> setEnabled(true);
  lineEditGroupThreshold -> setEnabled(true);
  lineEditEps -> setEnabled(true);
  lineEditNumberClassifiersUsed -> setEnabled(true);
  lineEditLineThicknessRectangles -> setEnabled(true);
  lineEditColorRectanglesR -> setEnabled(true);
  lineEditColorRectanglesG -> setEnabled(true);
  lineEditColorRectanglesB -> setEnabled(true);

  //QCheckBox 
  checkBoxFlagActivateSkinColor -> setEnabled(true);
  checkBoxNormalizeRotation -> setEnabled(true);
  checkBoxDoubleList -> setEnabled(true);
  checkBoxFlagExtractColorImages -> setEnabled(true);

}

void GUI_DETECTOR::setdefaultConfig() {

  //QLineEdit
  lineEditDegreesDetections -> setText("0");
  lineEditSizeBase -> setText("40");
  lineEditFactorScaleWindow -> setText("1.2");
  lineEditStepWindow -> setText("0.1");
  lineEditSizeMaxWindow -> setText("2000");
  lineEditGroupThreshold -> setText("1");
  lineEditEps -> setText("0.5");

  numberStrongLearns = myThreadDetector -> objectDetector -> getNumberStrongLearns();
  textNumberStrongLearns -> setText("<font color=red>MAX strongLearns</font></h2>=<font color=blue>" + QString::number(numberStrongLearns) + "</font></h2>");
  lineEditNumberClassifiersUsed -> setText(QString::number(numberStrongLearns));

  lineEditLineThicknessRectangles -> setText("2");
  lineEditColorRectanglesR -> setText("0");
  lineEditColorRectanglesG -> setText("255");
  lineEditColorRectanglesB -> setText("0");

  //QCheckBox 
  checkBoxFlagActivateSkinColor -> setCheckState(Qt::Unchecked);
  checkBoxNormalizeRotation -> setCheckState(Qt::Unchecked);
  checkBoxDoubleList -> setCheckState(Qt::Checked);
  checkBoxFlagExtractColorImages -> setCheckState(Qt::Checked);

}

void GUI_DETECTOR::setEnabledHsvConfig(int flag) {
  lineEditHmin -> setEnabled(flag);
  lineEditHmax -> setEnabled(flag);
  lineEditSmin -> setEnabled(flag);
  lineEditSmax -> setEnabled(flag);
  lineEditVmin -> setEnabled(flag);
  lineEditVmax -> setEnabled(flag);

  if (flag) {
    lineEditHmin -> setText("0");
    lineEditHmax -> setText("20");
    lineEditSmin -> setText("10");
    lineEditSmax -> setText("150");
    lineEditVmin -> setText("60");
    lineEditVmax -> setText("255");
  } else {
    lineEditHmin -> clear();
    lineEditHmax -> clear();
    lineEditSmin -> clear();
    lineEditSmax -> clear();
    lineEditVmin -> clear();
    lineEditVmax -> clear();
  }

}

void GUI_DETECTOR::setEnabledOptionNotGroup() {

  if (lineEditGroupThreshold -> text() != QString("")) {
    if (lineEditGroupThreshold -> text().toInt() == 0) {

      checkBoxNormalizeRotation -> setEnabled(false);
      checkBoxDoubleList -> setEnabled(false);
      checkBoxFlagExtractColorImages -> setEnabled(false);
      lineEditEps -> setEnabled(false);

    } else {

      checkBoxNormalizeRotation -> setEnabled(true);
      if (checkBoxNormalizeRotation -> checkState() == Qt::Unchecked)
        checkBoxDoubleList -> setEnabled(true);

      checkBoxFlagExtractColorImages -> setEnabled(true);
      lineEditEps -> setEnabled(true);

    }
  }

}

void GUI_DETECTOR::activeConfigHsv(int state) {

  if (state == Qt::Checked)
    setEnabledHsvConfig(true);
  else
    setEnabledHsvConfig(false);

}

void GUI_DETECTOR::normalizeRotation(int state) {

  if (state == Qt::Checked) { //Disable the option to double the list
    checkBoxDoubleList -> setCheckState(Qt::Unchecked);
    checkBoxDoubleList -> setEnabled(false);
  } else if (state == Qt::Unchecked) {
    checkBoxDoubleList -> setEnabled(true);
    checkBoxDoubleList -> setCheckState(Qt::Checked);
  }

  edition();
}

void GUI_DETECTOR::loadDetector() {

  if (myThreadDetector -> detectorIsLoad) {

    QMessageBox::StandardButton question;
    question = QMessageBox::question(this, "Test", QString::fromUtf8("A detector file has already been loaded. If you load a new one, the current one will be deleted. Do you want to continue?"), QMessageBox::Yes | QMessageBox::No);

    if (question == QMessageBox::No)
      return;

  }

  //____Open the dialog to load a file____
  QString fileName = QFileDialog::getOpenFileName(this, tr("Open detector"), LAST_PATH_DETECTOR, tr("Detector files xml (*.xml)\n"));
  if (fileName == "") {
      return; //In case the user does not open anything.
  }else {
      QFileInfo fileInfo(fileName);
      LAST_PATH_DETECTOR = fileInfo.path();
  }
  //_________________________________________________

  myThreadDetector -> loadDetector(fileName.toStdString());
  setdefaultConfig();
  setEnabledTrueConfigDefault();
  setConfig();
}

void GUI_DETECTOR::setConfig() {

  // Check if any field is empty
  if ((lineEditDegreesDetections->text() == "") || (lineEditSizeMaxWindow->text() == "") || (lineEditSizeBase->text() == "") ||
    (lineEditFactorScaleWindow->text() == "") || (lineEditStepWindow->text() == "") || (lineEditGroupThreshold->text() == "") || 
    (lineEditEps->text() == "") || (lineEditNumberClassifiersUsed->text() == "") || (lineEditLineThicknessRectangles->text() == "") || 
    (lineEditColorRectanglesR->text() == "") || (lineEditColorRectanglesG->text() == "") || (lineEditColorRectanglesB->text() == "")) {
    QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("Some fields are missing."), QMessageBox::Ok);
    return;
  }

  // If skin color activation is checked, check HSV fields
  if (Qt::Checked == checkBoxFlagActivateSkinColor->checkState()) {
    if ((lineEditHmin->text() == "") || (lineEditHmax->text() == "") || (lineEditSmin->text() == "") || (lineEditSmax->text() == "") || 
        (lineEditVmin->text() == "") || (lineEditVmax->text() == "")) {
      QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("Some fields in the HSV color space restriction configuration are missing."), QMessageBox::Ok);
    }
  }

  // Setting degrees
  QString qstrDegrees = lineEditDegreesDetections->text();
  QStringList listQstrDegrees = qstrDegrees.split(" "); // Split by space

  std::vector < double > degreesDetection;
  for (int i = 0; i < listQstrDegrees.size(); i++)
    degreesDetection.push_back(listQstrDegrees[i].toDouble());

  int sizeMaxWindow = lineEditSizeMaxWindow->text().toInt();
  int sizeBase = lineEditSizeBase->text().toInt();
  if (sizeBase > sizeMaxWindow) {
    QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("The base size must be smaller than the maximum size"), QMessageBox::Ok);
    return;
  }

  double factorScaleWindow = lineEditFactorScaleWindow->text().toDouble();
  if (factorScaleWindow <= 0) {
    QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("The scale factor must be greater than zero"), QMessageBox::Ok);
    return;
  }

  double stepWindow = lineEditStepWindow->text().toDouble();
  if (stepWindow <= 0) {
    QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("The step factor must be greater than zero"), QMessageBox::Ok);
    return;
  }

  int groupThreshold = lineEditGroupThreshold->text().toInt();
  if (groupThreshold == 0)
    myThreadDetector->groupingRectangles = false;
  else
    myThreadDetector->groupingRectangles = true;

  double eps = lineEditEps->text().toDouble();

  int numberClassifiersUsed = lineEditNumberClassifiersUsed->text().toInt();
  if (numberClassifiersUsed > numberStrongLearns) {
    QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("The file only has ") + QString::number(numberStrongLearns) + QString(" classifiers"), QMessageBox::Ok);
    return;
  }

  int thicknessRectangles = lineEditLineThicknessRectangles->text().toInt();
  int colorRectanglesR = lineEditColorRectanglesR->text().toInt();
  int colorRectanglesG = lineEditColorRectanglesG->text().toInt();
  int colorRectanglesB = lineEditColorRectanglesB->text().toInt();

  //QCheckBox 
  bool flagActivateSkinColor = false;
  bool normalizeRotation = false;
  bool doubleList = false;
  bool flagExtractColorImages = false;

  if (checkBoxFlagActivateSkinColor->checkState() == Qt::Checked)
    flagActivateSkinColor = true;
  if (checkBoxNormalizeRotation->checkState() == Qt::Checked)
    normalizeRotation = true;
  if (checkBoxDoubleList->checkState() == Qt::Checked)
    doubleList = true;
  if (checkBoxFlagExtractColorImages->checkState() == Qt::Checked)
    flagExtractColorImages = true;

  myThreadDetector->objectDetector->setDegreesDetections(degreesDetection);

  myThreadDetector->objectDetector->setSizeBase(sizeBase);
  myThreadDetector->objectDetector->setSizeMaxWindow(sizeMaxWindow);

  myThreadDetector->objectDetector->setFactorScaleWindow(factorScaleWindow);
  myThreadDetector->objectDetector->setStepWindow(stepWindow);

  myThreadDetector->objectDetector->setGroupThreshold(groupThreshold);
  myThreadDetector->objectDetector->setEps(eps);

  myThreadDetector->objectDetector->setNumberClassifiersUsed(numberClassifiersUsed);

  myThreadDetector->objectDetector->setLineThicknessRectangles(thicknessRectangles);
  myThreadDetector->objectDetector->setColorRectangles(cv::Scalar(colorRectanglesB, colorRectanglesG, colorRectanglesR));

  myThreadDetector->objectDetector->setFlagActivateSkinColor(flagActivateSkinColor);
  if (flagActivateSkinColor) {
    int hmin, smin, vmin, hmax, smax, vmax;
    hmin = lineEditHmin->text().toInt();
    hmax = lineEditHmax->text().toInt();
    smin = lineEditSmin->text().toInt();
    smax = lineEditSmax->text().toInt();
    vmin = lineEditVmin->text().toInt();
    vmax = lineEditVmax->text().toInt();

    myThreadDetector->objectDetector->setHsvMin(cv::Scalar(hmin, smin, vmin));
    myThreadDetector->objectDetector->setHsvMax(cv::Scalar(hmax, smax, vmax));

  }

  myThreadDetector->normalizeRotation = normalizeRotation;
  myThreadDetector->doubleList = doubleList;

  myThreadDetector->objectDetector->setFlagExtractColorImages(flagExtractColorImages);

  myThreadDetector->objectDetector->initializeFeatures();

  buttonApplySettings->setEnabled(false);
  flagEdition = false;
}

void GUI_DETECTOR::edition() {
  flagEdition = true;
  buttonApplySettings->setEnabled(true);
}
