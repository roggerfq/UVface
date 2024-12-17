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

#ifndef INTERFAZPRINCIPAL_H
#define INTERFAZPRINCIPAL_H

//Own
#include "trackerWindows.h"
#include <QMainWindow>
#include <QThread>
#include <QTreeView>
//openCV
#include <opencv2/opencv.hpp>

//ADVANCED OWN CLASSES
class threadDetector; //Face detector thread
class GUI_DETECTOR; //GUI for face detector thread configuration
class RECOGNIZER_FACIAL; //Facial recognition thread
class GUI_FACE_RECOGNIZER; //Facial recognition GUI
class trackerWindows_gui; //Window tracker configuration
class configPresentationRecognition; //Presentation configuration for recognizer results

//ADVANCED QT CLASSES
class QLabel;
class QAction;
class QMenu;
class QToolBar;
class QPushButton;
class QLineEdit;
class QSlider;

//Pending
class QStandardItemModel;
class QStandardItem;
class QKeyEvent;
class QMouseEvent;

class viewerListImages: public QTreeView {
  Q_OBJECT

  QStandardItemModel * model;
  QStandardItem * rootNode;

  //QAction

  //Global menus
  QAction * seeFlowDetectorAction;
  QAction * seeFlowRecognizerAction;
  QAction * saveImagesAction;
  QAction * addImagesToUserAction;
  QAction * clearAction;
  QAction * configStackLengthImagesAction;

  //QMenu
  QMenu * contextMenuGeneral;
  QMenu * contextMenuItem;

  //Will store the images in view
  std::vector < cv::Mat > listCvImg;

  bool flagDisconnectFlowDetector;
  bool flagDisconnectFlowRecognizer;
  bool flagDetectorIsActive;

  int stackLengthImages;

  public:
    enum flowType {
      connectDetector,
      disconnectDetector,
      connectRecognizer,
      disconnectRecognizer
    };

  viewerListImages(QWidget * parent = 0);

  void createActions();
  void createContextMenu();

  public slots:
    void onCustomContextMenu(const QPoint & point);
  void setFlagDetectorIsActive(bool flag);
  void selectingFlowDetector(bool flag);
  void selectingFlowRecognizer(bool flag);
  void setHeaderName(QString headerName = QString::fromUtf8("Image Flow Viewer"));
  void addImagesList(std::vector < cv::Mat > listImages); //Interface for the detector
  void addImages(cv::Mat img, std::string name); //Interface for the recognizer
  void deleteRow(int i);
  void saveSelectedImages();
  void addImagesToUser();
  void clear(); //Clears the images in the view
  void configStackLengthImages();

  signals:
    void connectionType(viewerListImages::flowType type);
  void addImagesToDataBase(std::vector < cv::Mat > listImg, QString nameUser);
  void showImage(const cv::Mat & img);

  protected:
    void keyPressEvent(QKeyEvent * event);
  void mousePressEvent(QMouseEvent * event);
  void mouseDoubleClickEvent(QMouseEvent * event);

};

class interfaz: public QMainWindow {
  private: Q_OBJECT

  //_______OBJECTS RELATED TO THE VIDEO PANEL________________//
  int width_base; //This is just an initial standard size (it can change)
  int height_base; //This is just an initial standard size (it can change)
  double scale_videoWindow; /*This is a value by which the labelVideo will be enlarged while preserving the proportion given by width_base and height_base*/
  int width_videoWindow; //Width of labelVideo
  int height_videoWindow; //Height of labelVideo
  QLabel * labelVideo; //Here the video captured from some standard device will be shown
  cv::Mat imgCv; //Will store the image that should be shown in the graphical label
  cv::Mat textCv; //Will store the information to be displayed on the imgCv (text, numbers, etc.)
  //_________________________________________________________________//

  //QAction
  QAction * exitAction;
  QAction * configDetectorAction;
  QAction * configTrackerWindowsAction;
  QAction * configRecognizerAction;
  QAction * configPresentationRecognitionAction;
  QAction * enableRecognitionAction;

  QAction * nextAction;
  QAction * captureAction;
  QAction * backAction;
  QAction * loadFilesAction;

  QAction * generalInformationAction;
  QAction * manualFacialDetectorAction;
  QAction * manualTrackerWindowsAction;
  QAction * manualFacialRecognizerAction;

  //QMenu
  QMenu * fileMenu;
  QMenu * detectorMenu;
  QMenu * trackerMenu;
  QMenu * recognizerMenu;
  QMenu * helpMenu;

  QMenu * contextMenuGeneral;

  //QToolBar
  QToolBar * detectorToolBar;
  QToolBar * trackerToolBar;
  QToolBar * recognizerToolBar;
  QToolBar * acquisitionToolBar;

  //QPushButton

  //QLabel
  QLabel * labelInfoFilesLoads;
  QLabel * labelInfoPlayTime;

  //QListView
  viewerListImages * listImages;

  //QLineEdit
  QLineEdit * lineEditDevice;

  //QSlider
  QSlider * sliderPlayTime;

  //__________Variables_______________
  //The following two variables will store the total time of a video loaded from a file
  QString qstrTotalMinutes;
  QString qstrTotalSeconds;

  QStringList nameFiles; //Stores the temporary list of video and image file names
  int indexNameFile; //Stores the index of the file name currently being sent to the detector

  QStringList formatImages; //List of image formats
  QStringList formatVideos; //List of video formats
  QString qstrFormatImages; //QString for image format
  QString qstrFormatVideos; //QString for video format

  //These variables will store the form, color, and type of the text that will display the recognition results
  int fontTypeTextRecognition;
  double fontScaleTextRecognition;
  cv::Scalar colorTextRecognition;
  int thicknessTextRecognition;
  int lineTypeTextRecognition;

  //Algorithms
  threadDetector * detector;
  GUI_DETECTOR * guiDetector;
  trackerWindows myTrackerWindows;
  trackerWindows_gui * myTrackerWindows_gui;
  QThread threadTrackerWindows;
  RECOGNIZER_FACIAL * facialRecognizer;
  QThread threadRecognizerFacial;
  GUI_FACE_RECOGNIZER * guiFacialRecognizer;
  configPresentationRecognition * guiConfigPresentationRecognition;

  void setFormatImagesAndVideo();
  //Graphical configuration
  void setPanelVideo(); //Sets the video panel
  void setPanelItemsImages(); //Sets the panel where detected or recognized images will be shown optionally
  void createActions();
  void createMenus();
  void createContextMenu();
  void createToolBars();
  void setLayoutPrincipal(); //Sets all widgets in their respective places

  public: interfaz();
  ~interfaz();

  public slots: void onCustomContextMenu(const QPoint & point);
  void enableActionRecognition(bool flag);
  void enableRecognition(bool flag);
  void initializeSizeimgText(int width, int high); //Initializes the size of the textCv matrix
  void clearLabelVideo();
  void captureVideo();
  void setTextInDetection(const cv::Point point,
    const QString text);
  void setTextInDetection(const std::vector < cv::Point > listPoints,
    const std::vector < std::string > listText);
  void clearText();
  void showVideo(const cv::Mat & img);
  void showImage(const cv::Mat & img);
  void activeFlowsViewImages(viewerListImages::flowType type);
  void detectionImagesList(std::vector < cv::Mat > listDetectedObjects);
  void recognitionImagesList(imageTransaction myImageTransaction);
  void recognitionImagesList(const cv::Point,
    const QString text, cv::Mat img);
  void showNumberLoadFiles(int totalFiles, int currentFile = 0);
  void showPlayTimeTotal(int minutes, int seconds);
  void showPlayTimeRemaining(int minutes, int seconds);
  void configDetector();
  void configTrackerWindows();
  void configFacialRecognizer();
  void presentationRecognitionConfig();
  void loadFiles();
  void next();
  void back();
  void currentFile();
  void finishedDetection();

};

#endif
