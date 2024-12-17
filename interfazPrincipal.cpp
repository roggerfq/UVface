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

//_____OWN HEADER FILES__________________
#include "interfazPrincipal.h"
#include "guiConfigDetector.h" // Face detector + GUI
#include "trackerWindows_gui.h"/*Note: trackerWindows.h is declared in interfazPrincipal.h to avoid using it as a pointer, preventing issues with dynamically allocated memory since it will reside in another thread*/
#include <guiFaceRecognizer.h>
#include "recognizerFacial.h"
#include "guiOtherConfigurations.h"
//QT CLASSES
#include <QLabel>
#include <QImage>
#include <QGridLayout>
#include <QAction>
#include <QMenu>
#include <QMenuBar>
#include <QToolBar>
#include <QPushButton>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QSlider>
#include <QMessageBox>
#include <QFileDialog>
#include <QInputDialog>
#include <QDir>
#include <QFileInfo>
//In review
//#include <QTreeView>
#include <QStandardItemModel>
#include <QStandardItem>
#include <QKeyEvent>
#include <QMouseEvent>
//REMOVE
#include <QDebug>
#include <iostream>
#include <QHeaderView>

// Path of the last loaded files
QString LAST_PATH_FILES_TEST = QDir::homePath();


viewerListImages::viewerListImages(QWidget * parent): QTreeView(parent) {

  flagDisconnectFlowDetector = false;
  flagDisconnectFlowRecognizer = false;
  flagDetectorIsActive = false;

  stackLengthImages = 20; // Default length

  model = new QStandardItemModel;
  model -> setColumnCount(1);
  setModel(model);
  expandAll();
  model -> setHeaderData(0, Qt::Horizontal, "");
  rootNode = model -> invisibleRootItem(); // Only used for comparison in context menus

  createActions();
  createContextMenu();
  setHeaderName();

  setContextMenuPolicy(Qt::CustomContextMenu);
  connect(this, SIGNAL(customContextMenuRequested(const QPoint & )), this, SLOT(onCustomContextMenu(const QPoint & )));

  setSelectionMode(QAbstractItemView::ExtendedSelection);

  header() -> setStretchLastSection(false);

}

void viewerListImages::createActions() {

  // Global menus
  seeFlowDetectorAction = new QAction(QString::fromUtf8("View detector flow"), this);
  seeFlowDetectorAction -> setCheckable(true);
  connect(seeFlowDetectorAction, SIGNAL(triggered(bool)), this, SLOT(selectingFlowDetector(bool)));
  seeFlowRecognizerAction = new QAction(QString::fromUtf8("View recognizer flow"), this);
  seeFlowRecognizerAction -> setCheckable(true);
  connect(seeFlowRecognizerAction, SIGNAL(triggered(bool)), this, SLOT(selectingFlowRecognizer(bool)));

  saveImagesAction = new QAction(QString::fromUtf8("Save selected images"), this);
  connect(saveImagesAction, SIGNAL(triggered(bool)), this, SLOT(saveSelectedImages()));

  addImagesToUserAction = new QAction(QString::fromUtf8("Add images to database"), this);
  connect(addImagesToUserAction, SIGNAL(triggered(bool)), this, SLOT(addImagesToUser()));

  clearAction = new QAction(QString::fromUtf8("Clear view"), this);
  connect(clearAction, SIGNAL(triggered(bool)), this, SLOT(clear()));

  configStackLengthImagesAction = new QAction(QString::fromUtf8("Image stack length"), this);
  connect(configStackLengthImagesAction, SIGNAL(triggered(bool)), this, SLOT(configStackLengthImages()));

}

void viewerListImages::createContextMenu() {

  contextMenuGeneral = new QMenu(this);
  contextMenuGeneral -> addAction(seeFlowDetectorAction);
  contextMenuGeneral -> addAction(seeFlowRecognizerAction);
  contextMenuGeneral -> addAction(saveImagesAction);
  contextMenuGeneral -> addAction(addImagesToUserAction);
  contextMenuGeneral -> addAction(clearAction);
  contextMenuGeneral -> addAction(configStackLengthImagesAction);

}

void viewerListImages::onCustomContextMenu(const QPoint & point) {
  std::cout << "flagDetectorIsActive=" << flagDetectorIsActive << "\n";
  if ((model -> rowCount() > 0) && ((!flagDetectorIsActive) || ((!flagDisconnectFlowDetector) && (!flagDisconnectFlowRecognizer)))) {
    saveImagesAction -> setEnabled(true);
    addImagesToUserAction -> setEnabled(true);
    clearAction -> setEnabled(true);
  } else {
    saveImagesAction -> setEnabled(false);
    addImagesToUserAction -> setEnabled(false);
    clearAction -> setEnabled(false);
  }

  contextMenuGeneral -> exec(mapToGlobal(point));

  /*//Not implemented for this version
  QModelIndex index=indexAt(point);

  if(index.isValid()&&(index.parent()==rootNode->index()))//It's an item
  {
  //contextMenuItem->exec(mapToGlobal(point));
  }else
  { 
  contextMenuGeneral->exec(mapToGlobal(point));
  }
  */

}

void viewerListImages::setFlagDetectorIsActive(bool flag) {
  flagDetectorIsActive = flag;
}

void viewerListImages::selectingFlowDetector(bool flag) {

  if (flagDisconnectFlowDetector) {
    flagDisconnectFlowDetector = false;
    emit connectionType(disconnectDetector);
  }

  if (flag) {

    if (seeFlowRecognizerAction -> isChecked()) {
      seeFlowRecognizerAction -> setChecked(false);
      flagDisconnectFlowRecognizer = false;
      emit connectionType(disconnectRecognizer);
    }

    seeFlowDetectorAction -> setChecked(true);
    flagDisconnectFlowDetector = true;
    emit connectionType(connectDetector);

  }

}

void viewerListImages::selectingFlowRecognizer(bool flag) {

  if (flagDisconnectFlowRecognizer) {
    flagDisconnectFlowRecognizer = false;
    emit connectionType(disconnectRecognizer);
  }

  if (flag) {

    if (seeFlowDetectorAction -> isChecked()) {
      seeFlowDetectorAction -> setChecked(false);
      flagDisconnectFlowDetector = false;
      emit connectionType(disconnectDetector);
    }

    seeFlowRecognizerAction -> setChecked(true);
    flagDisconnectFlowRecognizer = true;
    emit connectionType(connectRecognizer);

  }

}

void viewerListImages::setHeaderName(QString headerName) {
  model -> setHeaderData(0, Qt::Horizontal, headerName);
}

void viewerListImages::addImagesList(std::vector < cv::Mat > listImages) {

  //___From here, the corresponding information about the number of images in the list can be updated_________//
  setHeaderName(QString::fromUtf8("Image Flow Viewer=") + QString::number(model -> rowCount()));

  while (model -> rowCount() >= stackLengthImages) {
    QList < QStandardItem * > tempItems = model -> takeRow(model -> rowCount() - 1);
    qDeleteAll(tempItems);
  }

  for (int i = 0; i < listImages.size(); i++) {
    QStandardItem * tempItem = new QStandardItem();
    cv::Mat tempImg = listImages[i];

    if (tempImg.channels() == 1) {
      cvtColor(tempImg, tempImg, CV_GRAY2BGR);
    }

    QImage image(tempImg.data, tempImg.cols, tempImg.rows, tempImg.step, QImage::Format_RGB888);
    tempItem -> setData(QVariant(QPixmap::fromImage(image.rgbSwapped())), Qt::DecorationRole);
    tempItem -> setText(QString::fromStdString("HeightxWidth=") + QString::number(tempImg.rows) + QString("x") + QString::number(tempImg.cols));
    model -> insertRow(i, tempItem);
  }

}

void viewerListImages::addImages(cv::Mat img, std::string name) {

  //___From here, the corresponding information about the number of images in the list can be updated_________//
  setHeaderName(QString::fromUtf8("Image Flow Viewer=") + QString::number(model -> rowCount()));

  while (model -> rowCount() >= stackLengthImages) {
    QList < QStandardItem * > tempItems = model -> takeRow(model -> rowCount() - 1);
    qDeleteAll(tempItems);
  }

  if (img.channels() == 1) {
    cvtColor(img, img, CV_GRAY2BGR);
  }

  QStandardItem * tempItem = new QStandardItem();
  QImage image(img.data, img.cols, img.rows, img.step, QImage::Format_RGB888);
  tempItem -> setData(QVariant(QPixmap::fromImage(image.rgbSwapped())), Qt::DecorationRole);
  //tempItem->setText(QString::fromStdString(name));
  tempItem -> setText(QString("Name=") + QString::fromStdString(name) + QString::fromStdString("\nHeightxWidth=") + QString::number(img.rows) + QString("x") + QString::number(img.cols));
  model -> insertRow(0, tempItem);

}

void viewerListImages::saveSelectedImages() {

  QList < QModelIndex > listSelectedItems = selectedIndexes();

  if (listSelectedItems.empty()) {
    QMessageBox::warning(this, QString::fromUtf8("INFORMATION"), QString::fromUtf8("No image selected"), QMessageBox::Ok);
    return;
  }

  QString tempPathDir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
    QDir::home().path());

  QString nameBaseImage = tempPathDir + QString("/image");

  std::vector < int > compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);

  cv::RNG rng(QDateTime::currentMSecsSinceEpoch());

  int from_to[] = {
    0,
    0,
    1,
    1,
    2,
    2
  };

  for (int i = 0; i < listSelectedItems.size(); i++) {

    QStandardItem * tempItem = model -> item(listSelectedItems[i].row());
    QVariant qvImg = tempItem -> data(Qt::DecorationRole);
    QImage tempImg = qvImg.value < QImage > ();
    cv::Mat imgCv4 = cv::Mat(tempImg.height(), tempImg.width(), CV_8UC4, const_cast < uchar * > (tempImg.bits()), tempImg.bytesPerLine());
    cv::Mat imgCv3(imgCv4.rows, imgCv4.cols, CV_8UC3);
    mixChannels( & imgCv4, 1, & imgCv3, 1, from_to, 3);

    QString nameFileImg = nameBaseImage + QString::number(rng.uniform(0, 2147483647)) + ".png";
    cv::imwrite(nameFileImg.toStdString().c_str(), imgCv3, compression_params);

  }

}

void viewerListImages::addImagesToUser() {

  QList < QModelIndex > listSelectedItems = selectedIndexes();

  if (listSelectedItems.empty()) {
    QMessageBox::warning(this, QString::fromUtf8("INFORMATION"), QString::fromUtf8("No image selected"), QMessageBox::Ok);
    return;
  }

  bool ok;
  QString nameUser = QInputDialog::getText(this, tr("QInputDialog::getText()"), QString::fromUtf8("Enter the username to which you wish to\nadd the selected images:"), QLineEdit::Normal, QString(), & ok);
  if (!ok) return;

  std::vector < cv::Mat > listImg;
  int from_to[] = {
    0,
    0,
    1,
    1,
    2,
    2
  };
  for (int i = 0; i < listSelectedItems.size(); i++) {

    QStandardItem * tempItem = model -> item(listSelectedItems[i].row());
    QVariant qvImg = tempItem -> data(Qt::DecorationRole);
    QImage tempImg = qvImg.value < QImage > ();
    cv::Mat imgCv4 = cv::Mat(tempImg.height(), tempImg.width(), CV_8UC4, const_cast < uchar * > (tempImg.bits()), tempImg.bytesPerLine());
    cv::Mat imgCv3(imgCv4.rows, imgCv4.cols, CV_8UC3);
    mixChannels( & imgCv4, 1, & imgCv3, 1, from_to, 3);
    listImg.push_back(imgCv3.clone());

  }

  emit addImagesToDataBase(listImg, nameUser);

}

void viewerListImages::clear() {

  while (model -> rowCount()) {
    QList < QStandardItem * > tempItems = model -> takeRow(model -> rowCount() - 1);
    qDeleteAll(tempItems);
  }

  setHeaderName(QString::fromUtf8("Image Flow Viewer"));

}

void viewerListImages::configStackLengthImages() {

  bool ok;
  int num = QInputDialog::getInt(this, QString::fromUtf8("Image Stack"), QString::fromUtf8("Length:"), 20, 1, 100000, 1, & ok);

  if (ok)
    stackLengthImages = num;

}

void viewerListImages::deleteRow(int i) {

  QList < QStandardItem * > tempItems = model -> takeRow(i);
  qDeleteAll(tempItems);

}

void viewerListImages::keyPressEvent(QKeyEvent * event) {

  if (event -> key() == Qt::Key_Delete) {

    QList < QModelIndex > listSelectedItems = selectedIndexes();

    std::vector < int > rows;
    for (int i = 0; i < listSelectedItems.size(); i++) {
      rows.push_back(listSelectedItems[i].row());
    }

    std::sort(rows.begin(), rows.end());
    std::reverse(rows.begin(), rows.end());

    for (int i = 0; i < rows.size(); i++)
      deleteRow(rows[i]);

    event -> accept();
  } else {
    QTreeView::keyPressEvent(event);
  }

  int temp = model -> rowCount();
  if (temp > 0)
    setHeaderName(QString::fromUtf8("Image Flow Viewer=") + QString::number(model -> rowCount()));
  else
    setHeaderName(QString::fromUtf8("Image Flow Viewer"));

}

void viewerListImages::mousePressEvent(QMouseEvent * event) {

  QModelIndex index = indexAt(event -> pos());

  if (!index.isValid())
    clearFocus();

  QTreeView::mousePressEvent(event);

}

void viewerListImages::mouseDoubleClickEvent(QMouseEvent * event) {

  QModelIndex index = indexAt(QPoint(event -> x(), event -> y()));

  if (index.isValid() && (index.parent() == rootNode -> index())) //It is an item
  {

    int from_to[] = {
      0,
      0,
      1,
      1,
      2,
      2
    };
    QStandardItem * tempItem = model -> item(index.row());
    QVariant qvImg = tempItem -> data(Qt::DecorationRole);
    QImage tempImg = qvImg.value < QImage > ();
    cv::Mat imgCv4 = cv::Mat(tempImg.height(), tempImg.width(), CV_8UC4, const_cast < uchar * > (tempImg.bits()), tempImg.bytesPerLine());
    cv::Mat imgCv3(imgCv4.rows, imgCv4.cols, CV_8UC3);
    mixChannels( & imgCv4, 1, & imgCv3, 1, from_to, 3);

    emit showImage(imgCv3);

  }

}

interfaz::interfaz(): detector(NULL), guiDetector(NULL), myTrackerWindows_gui(NULL), facialRecognizer(NULL), guiFacialRecognizer(NULL), guiConfigPresentationRecognition(NULL) {

  //___________INITIALIZATION OF VARIABLES_______________//
  imgCv = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0)); //Default
  textCv = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0)); //Default
  //_____________________________________________________//

  //___________ALGORITHM INITIALIZATION_________________//
  //Face detector
  detector = new threadDetector;
  guiDetector = new GUI_DETECTOR(detector);

  //Window tracker
  myTrackerWindows_gui = new trackerWindows_gui( & myTrackerWindows);
  myTrackerWindows.moveToThread( & threadTrackerWindows);

  //Facial recognizer
  facialRecognizer = new RECOGNIZER_FACIAL;
  facialRecognizer -> moveToThread( & threadRecognizerFacial);
  guiFacialRecognizer = new GUI_FACE_RECOGNIZER(facialRecognizer);

  threadRecognizerFacial.start(); /*This should be done after "guiFacialRecognizer=new GUI_FACE_RECOGNIZER(facialRecognizer)" because some signals are connected internally*/
  threadTrackerWindows.start(); //It is called here because necessary connections must be established first

  //_____________________________________________________//

  setFormatImagesAndVideo(); //Setting accepted image and video formats
  setPanelVideo(); //Sets the video panel
  setPanelItemsImages(); //Sets the panel where detected or recognized images will optionally be shown
  createActions();
  createMenus();
  createContextMenu();
  createToolBars();
  setLayoutPrincipal(); //Sets all widgets in their respective places
  showMaximized();
  setMinimumSize(750, 500);
  setWindowTitle(QString::fromUtf8("UVface"));

  //EXTRA CONFIGURATIONS
  guiConfigPresentationRecognition = new configPresentationRecognition( & fontTypeTextRecognition, & fontScaleTextRecognition, & colorTextRecognition, & thicknessTextRecognition, & lineTypeTextRecognition);

  //____________________________________________INITIAL CONNECTIONS______________________________________________________________

  //Basic connections between the detector and this scope
  connect(detector, SIGNAL(imageReady(const cv::Mat & )), this, SLOT(showVideo(const cv::Mat & )), Qt::QueuedConnection);
  connect(detector, SIGNAL(clearLabelVideo()), this, SLOT(clearLabelVideo()), Qt::QueuedConnection);
  connect(detector, SIGNAL(setSizeFrame(int, int)), this, SLOT(initializeSizeimgText(int, int)));
  connect(detector, SIGNAL(finishedDetection()), this, SLOT(finishedDetection()), Qt::QueuedConnection);

  //_____________________________________________________________________________________________________________//
  /*
  These connections ensure proper synchronization between the detector, tracker, and recognizer
  */
  connect(detector, SIGNAL(resetTrackerWindows()), & myTrackerWindows, SLOT(reset()), Qt::BlockingQueuedConnection);
  connect(detector, SIGNAL(stopRecognizer()), facialRecognizer, SLOT(stop()), Qt::DirectConnection);
  connect(detector, SIGNAL(enableRecognition()), facialRecognizer, SLOT(enableRecognition()), Qt::BlockingQueuedConnection);
  //______________________________________________________________________________________________________________//

  //_________________Here the window tracker is connected to the face recognizer________________________________________//
  connect( & myTrackerWindows, SIGNAL(recognizeImagesList(QList < imageTransaction > )), facialRecognizer, SLOT(recognizeImagesList(QList < imageTransaction > )), Qt::QueuedConnection);
  connect(facialRecognizer, SIGNAL(recognizedImage(imageTransaction)), & myTrackerWindows, SLOT(recognizedImage(imageTransaction)), Qt::BlockingQueuedConnection);
  //______________________________________________________________________________________________________________________________//

  //Result from the recognizer or tracker, depending on whether it's video or image
  connect( & myTrackerWindows, SIGNAL(setTextInDetection(const std::vector < cv::Point > ,
    const std::vector < std::string > )), this, SLOT(setTextInDetection(const std::vector < cv::Point > ,
    const std::vector < std::string > )), Qt::QueuedConnection);
  connect(facialRecognizer, SIGNAL(recognizedImage(const cv::Point,
    const QString, cv::Mat)), this, SLOT(setTextInDetection(const cv::Point,
    const QString)), Qt::QueuedConnection);

  //Useful connections to know the state of the database
  enableActionRecognition(false); //Must be called for the first time because no database has been loaded yet
  connect(guiFacialRecognizer, SIGNAL(recognizerIsReady(bool)), this, SLOT(enableActionRecognition(bool)));

  //Connection useful for controlling listImages
  connect(listImages, SIGNAL(addImagesToDataBase(std::vector < cv::Mat > , QString)), guiFacialRecognizer, SLOT(insertExternalImages(std::vector < cv::Mat > , QString)));

  //_______________________________________________________________________________________________________________________________

}

interfaz::~interfaz() {

  // Dynamic memory belonging to the detector and its graphical interface
  delete detector;
  delete guiDetector;

  // Closing the thread for the tracker
  threadTrackerWindows.quit();
  threadTrackerWindows.wait();
  delete myTrackerWindows_gui;

  // Closing the thread for the Facial Recognizer
  facialRecognizer -> stop(); // Just in case it is processing something
  threadRecognizerFacial.quit();
  threadRecognizerFacial.wait();

  // Dynamic memory belonging to the Facial Recognizer and its graphical interface
  delete facialRecognizer;
  delete guiFacialRecognizer;

  // Deleting dynamic memory for extra configurations
  delete guiConfigPresentationRecognition;

  std::cout << "Destroying the main interface\n";

}

void interfaz::setFormatImagesAndVideo() {

  // Image format list
  formatImages.push_back("png");
  formatImages.push_back("xpm");
  formatImages.push_back("jpg");
  formatImages.push_back("pgm");
  formatImages.push_back("jpeg");
  formatImages.push_back("JPEG");

  // Video format list
  formatVideos.push_back("avi");
  formatVideos.push_back("3gp");
  formatVideos.push_back("wmv");
  formatVideos.push_back("mp4");
  formatVideos.push_back("mkv");

  // Building accepted formats list as a string
  qstrFormatImages = "";
  for (int i = 0; i < formatImages.size(); i++)
    qstrFormatImages = qstrFormatImages + QString("*.") + formatImages[i] + " ";

  qstrFormatVideos = "";
  for (int i = 0; i < formatVideos.size(); i++)
    qstrFormatVideos = qstrFormatVideos + QString("*.") + formatVideos[i] + " ";

}

void interfaz::setPanelVideo() {

  width_base = 640;
  height_base = 480;
  scale_videoWindow = 1.43; //1.43
  width_videoWindow = scale_videoWindow * width_base; //1050
  height_videoWindow = scale_videoWindow * height_base; //732
  labelVideo = new QLabel;
  labelVideo -> setFixedSize(width_videoWindow, height_videoWindow);
  /*_________________VERY IMPORTANT FOR FULL IMAGE DISPLAY_______________________*/
  labelVideo -> setBackgroundRole(QPalette::Base);
  labelVideo -> setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  labelVideo -> setScaledContents(true);
  /*__________________________________________________________________________*/

  //__________About the context menus__________________//
  labelVideo -> setContextMenuPolicy(Qt::CustomContextMenu);
  connect(labelVideo, SIGNAL(customContextMenuRequested(const QPoint & )), this, SLOT(onCustomContextMenu(const QPoint & )));
  //___________________________________________________//

  clearLabelVideo(); //Initially clear the video window so it appears black

}

void interfaz::setPanelItemsImages() {

  listImages = new viewerListImages;
  listImages -> header() -> setMinimumSectionSize(500);
  listImages -> header() -> setDefaultSectionSize(500);
  listImages -> header() -> setStretchLastSection(false);

  connect(listImages, SIGNAL(connectionType(viewerListImages::flowType)), this, SLOT(activeFlowsViewImages(viewerListImages::flowType)));
  connect(listImages, SIGNAL(showImage(const cv::Mat & )), this, SLOT(showImage(const cv::Mat & )));

}

void interfaz::createActions() {

  exitAction = new QAction(tr("Exit"), this);
  exitAction -> setIcon(QIcon("./application_images/Actions/Exit.png"));
  exitAction -> setIconVisibleInMenu(true);
  exitAction -> setStatusTip(tr("Exit the application"));
  exitAction -> setShortcut(tr("Ctrl+Q")); // Non-standard shortcut

  configDetectorAction = new QAction(tr("Configure detector"), this);
  connect(configDetectorAction, SIGNAL(triggered()), this, SLOT(configDetector()));
  configDetectorAction -> setIcon(QIcon("./application_images/Actions/Detector.png"));
  configDetectorAction -> setIconVisibleInMenu(true);
  configDetectorAction -> setStatusTip(QString::fromUtf8("Configure detection algorithm"));
  configDetectorAction -> setShortcut(tr("Ctrl+D")); // Non-standard shortcut

  configTrackerWindowsAction = new QAction(tr("Configure Window Tracker"), this);
  connect(configTrackerWindowsAction, SIGNAL(triggered()), this, SLOT(configTrackerWindows()));
  configTrackerWindowsAction -> setIcon(QIcon("./application_images/Actions/Tracker.png"));
  configTrackerWindowsAction -> setIconVisibleInMenu(true);
  configTrackerWindowsAction -> setStatusTip(QString::fromUtf8("Configure window tracking algorithm"));
  configTrackerWindowsAction -> setShortcut(tr("Ctrl+S")); // Non-standard shortcut

  configRecognizerAction = new QAction(tr("Configure Facial Recognizer"), this);
  connect(configRecognizerAction, SIGNAL(triggered()), this, SLOT(configFacialRecognizer()));
  configRecognizerAction -> setIcon(QIcon("./application_images/Actions/Recognizer.png"));
  configRecognizerAction -> setIconVisibleInMenu(true);
  configRecognizerAction -> setStatusTip(QString::fromUtf8("Configure facial recognition algorithm"));
  configRecognizerAction -> setShortcut(tr("Ctrl+R")); // Non-standard shortcut

  configPresentationRecognitionAction = new QAction(QString::fromUtf8("Facial Recognizer Presentation"), this);
  connect(configPresentationRecognitionAction, SIGNAL(triggered()), this, SLOT(presentationRecognitionConfig()));

  enableRecognitionAction = new QAction(tr("Enable recognition"), this);
  enableRecognitionAction -> setCheckable(true);
  connect(enableRecognitionAction, SIGNAL(toggled(bool)), this, SLOT(enableRecognition(bool)));

  //______________________About Capture Options___________________________________//
  backAction = new QAction(QString::fromUtf8("Back"), this);
  connect(backAction, SIGNAL(triggered()), this, SLOT(back()));
  backAction -> setIcon(QIcon("./application_images/Actions/Back.png"));
  backAction -> setIconVisibleInMenu(true);
  backAction -> setShortcut(tr("Ctrl+S")); // Non-standard shortcut

  captureAction = new QAction(tr("Capture"), this);
  connect(captureAction, SIGNAL(triggered()), this, SLOT(captureVideo()));
  captureAction -> setIcon(QIcon("./application_images/Actions/Capture.png"));
  captureAction -> setIconVisibleInMenu(true);
  captureAction -> setShortcut(tr("Ctrl+R")); // Non-standard shortcut

  nextAction = new QAction(tr("Next"), this);
  connect(nextAction, SIGNAL(triggered()), this, SLOT(next()));
  nextAction -> setIcon(QIcon("./application_images/Actions/Next.png"));
  nextAction -> setIconVisibleInMenu(true);
  nextAction -> setShortcut(tr("Ctrl+R")); // Non-standard shortcut

  loadFilesAction = new QAction(tr("Load from file"), this);
  connect(loadFilesAction, SIGNAL(triggered()), this, SLOT(loadFiles()));
  loadFilesAction -> setIcon(QIcon("./application_images/Actions/Files.png"));
  loadFilesAction -> setIconVisibleInMenu(true);
  loadFilesAction -> setShortcut(tr("Ctrl+R")); // Non-standard shortcut
  //______________________________________________________________________________//

  // About the help menu for the user
  generalInformationAction = new QAction(QString::fromUtf8("General program information"), this);
  manualFacialDetectorAction = new QAction(tr("Facial detector manual"), this);
  manualTrackerWindowsAction = new QAction(tr("Window tracker manual"), this);
  manualFacialRecognizerAction = new QAction(tr("Facial recognizer manual"), this);

}

void interfaz::createMenus() {

  fileMenu = menuBar() -> addMenu(tr("&File"));
  fileMenu -> addAction(exitAction);

  detectorMenu = menuBar() -> addMenu(tr("&Detector"));
  detectorMenu -> addAction(configDetectorAction);

  trackerMenu = menuBar() -> addMenu(tr("&Tracker"));
  trackerMenu -> addAction(configTrackerWindowsAction);

  recognizerMenu = menuBar() -> addMenu(tr("&Recognizer"));
  recognizerMenu -> addAction(configRecognizerAction);
  recognizerMenu -> addAction(configPresentationRecognitionAction);

  helpMenu = menuBar() -> addMenu(tr("&Help"));
  helpMenu -> addAction(generalInformationAction);
  helpMenu -> addAction(manualFacialDetectorAction);
  helpMenu -> addAction(manualTrackerWindowsAction);
  helpMenu -> addAction(manualFacialRecognizerAction);

}

void interfaz::createContextMenu() {

  contextMenuGeneral = new QMenu(labelVideo);

  contextMenuGeneral -> addAction(configDetectorAction);
  contextMenuGeneral -> addAction(configTrackerWindowsAction);
  contextMenuGeneral -> addAction(configRecognizerAction);
  contextMenuGeneral -> addAction(enableRecognitionAction);

}

void interfaz::createToolBars() {

  detectorToolBar = addToolBar(tr("&Detector"));
  detectorToolBar -> addAction(configDetectorAction);
  detectorToolBar -> setStyleSheet("QToolBar { background:gray; }");
  detectorToolBar -> setMovable(true);
  detectorToolBar -> setAllowedAreas(Qt::TopToolBarArea | Qt::BottomToolBarArea);
  addToolBar(Qt::TopToolBarArea, detectorToolBar);
  detectorToolBar -> move(detectorToolBar -> pos().x() + 20, detectorToolBar -> pos().y());

  trackerToolBar = addToolBar(tr("&Tracker"));
  trackerToolBar -> addAction(configTrackerWindowsAction);
  trackerToolBar -> setStyleSheet("QToolBar { background:gray; }");
  trackerToolBar -> setMovable(true);
  trackerToolBar -> setAllowedAreas(Qt::TopToolBarArea | Qt::BottomToolBarArea);
  addToolBar(Qt::TopToolBarArea, trackerToolBar);

  recognizerToolBar = addToolBar(tr("&Recognizer"));
  recognizerToolBar -> addAction(configRecognizerAction);
  recognizerToolBar -> setStyleSheet("QToolBar { background:gray; }");
  recognizerToolBar -> setMovable(true);
  recognizerToolBar -> setAllowedAreas(Qt::TopToolBarArea | Qt::BottomToolBarArea);
  addToolBar(Qt::TopToolBarArea, recognizerToolBar);

  //________________CAPTURE TOOLS___________________________// 
  acquisitionToolBar = addToolBar(tr("&Image acquisition"));

  //Empty widget to create a separation space
  QWidget * empty = new QWidget();
  empty -> setFixedWidth(30);
  acquisitionToolBar -> addWidget(empty);

  acquisitionToolBar -> addWidget(new QLabel("<font color=white>" + QString::fromUtf8("Camera:"))); //Camera notice
  lineEditDevice = new QLineEdit("0");
  lineEditDevice -> setFixedWidth(30);
  lineEditDevice -> setFocusPolicy(Qt::ClickFocus);
  acquisitionToolBar -> addWidget(lineEditDevice); //Attach optional device input

  acquisitionToolBar -> addAction(loadFilesAction); //Attach option to load files from hard drive

  labelInfoFilesLoads = new QLabel;
  labelInfoFilesLoads -> setFixedWidth(150);
  labelInfoFilesLoads -> setFrameShape(QFrame::StyledPanel);
  showNumberLoadFiles(0);
  acquisitionToolBar -> addWidget(labelInfoFilesLoads);

  acquisitionToolBar -> addAction(backAction);
  acquisitionToolBar -> addAction(captureAction);
  acquisitionToolBar -> addAction(nextAction);

  //NOTE: The slider and QLabel showing video file progress are left for future improvements
  /*
  labelInfoPlayTime=new QLabel;
  showPlayTimeTotal(0,0);//Default
  showPlayTimeRemaining(0,0);//Default
  acquisitionToolBar->addWidget(labelInfoPlayTime);//Attach label showing video playtime
  sliderPlayTime=new QSlider(Qt::Horizontal);
  sliderPlayTime->setFixedWidth(90);
  sliderPlayTime->setEnabled(false);
  acquisitionToolBar->addWidget(sliderPlayTime);//Attach slider showing video playtime
  */

  acquisitionToolBar -> setStyleSheet("QToolBar { background:gray; }");
  acquisitionToolBar -> setMovable(true);
  acquisitionToolBar -> setAllowedAreas(Qt::TopToolBarArea | Qt::BottomToolBarArea);
  addToolBar(Qt::TopToolBarArea, acquisitionToolBar);
  acquisitionToolBar -> setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Maximum);

  //_______________________________________________________//

}

void interfaz::setLayoutPrincipal() {

  QGridLayout * layoutPrincipal = new QGridLayout;
  layoutPrincipal -> addWidget(labelVideo, 0, 0, 5, 6);
  layoutPrincipal -> addWidget(listImages, 0, 6, 5, 2);

  QLabel * labelPrincipal = new QLabel;
  labelPrincipal -> setLayout(layoutPrincipal);

  setCentralWidget(labelPrincipal);

}

//_____________________________________________SLOTS____________________________________________________//

void interfaz::onCustomContextMenu(const QPoint & point) {
  contextMenuGeneral -> exec(labelVideo -> mapToGlobal(point));
}

void interfaz::enableActionRecognition(bool flag) {

  if (flag)
    enableRecognitionAction -> setEnabled(true);
  else
    enableRecognitionAction -> setEnabled(false);

  enableRecognitionAction -> setChecked(false);

}

void interfaz::enableRecognition(bool flag) {

  if (enableRecognitionAction -> isChecked()) {
    connect(detector, SIGNAL(listCoordinatesAndDetectedObjects(std::vector < cv::Mat > , std::vector < cv::Rect > )), & myTrackerWindows, SLOT(newGroupDetections(std::vector < cv::Mat > , std::vector < cv::Rect > )), Qt::QueuedConnection);
    connect(detector, SIGNAL(listCoordinatesAndDetectedObjectsRotated(std::vector < cv::Mat > , std::vector < cv::RotatedRect > )), & myTrackerWindows, SLOT(newGroupDetections(std::vector < cv::Mat > , std::vector < cv::RotatedRect > )), Qt::QueuedConnection);

    connect(detector, SIGNAL(listCoordinatesAndDetectedObjects_img(std::vector < cv::Mat > , std::vector < cv::Rect > )), facialRecognizer, SLOT(recognizedImage(std::vector < cv::Mat > , std::vector < cv::Rect > )), Qt::QueuedConnection);
    connect(detector, SIGNAL(listCoordinatesAndDetectedObjectsRotated_img(std::vector < cv::Mat > , std::vector < cv::RotatedRect > )), facialRecognizer, SLOT(recognizedImage(std::vector < cv::Mat > , std::vector < cv::RotatedRect > )), Qt::QueuedConnection);

  } else {

    disconnect(detector, SIGNAL(listCoordinatesAndDetectedObjects(std::vector < cv::Mat > , std::vector < cv::Rect > )), & myTrackerWindows, SLOT(newGroupDetections(std::vector < cv::Mat > , std::vector < cv::Rect > )));
    disconnect(detector, SIGNAL(listCoordinatesAndDetectedObjectsRotated(std::vector < cv::Mat > , std::vector < cv::RotatedRect > )), & myTrackerWindows, SLOT(newGroupDetections(std::vector < cv::Mat > , std::vector < cv::RotatedRect > )));

    disconnect(detector, SIGNAL(listCoordinatesAndDetectedObjects_img(std::vector < cv::Mat > , std::vector < cv::Rect > )), facialRecognizer, SLOT(recognizedImage(std::vector < cv::Mat > , std::vector < cv::Rect > )));
    disconnect(detector, SIGNAL(listCoordinatesAndDetectedObjectsRotated_img(std::vector < cv::Mat > , std::vector < cv::RotatedRect > )), facialRecognizer, SLOT(recognizedImage(std::vector < cv::Mat > , std::vector < cv::RotatedRect > )));

    QMetaObject::invokeMethod( & myTrackerWindows, "reset", Qt::BlockingQueuedConnection);
    QMetaObject::invokeMethod(facialRecognizer, "stop", Qt::DirectConnection);
    QMetaObject::invokeMethod(facialRecognizer, "enableRecognition", Qt::BlockingQueuedConnection);
    clearText(); //To clear any name residues in the video

    std::cout << "Finishing disabling recognition in interfaz::enableRecognition\n";

  }

}

void interfaz::initializeSizeimgText(int width, int high) {
  textCv = cv::Mat::zeros(high, width, CV_8UC3);
}

void interfaz::clearLabelVideo() {
  textCv = cv::Mat::zeros(textCv.rows, textCv.cols, CV_8UC3); //Assumed previous initialization
  imgCv = cv::Mat::zeros(imgCv.rows, imgCv.cols, CV_8UC3); //Assumed previous initialization
  QImage imgBlack(width_videoWindow, height_videoWindow, QImage::Format_RGB32);
  imgBlack.fill(QColor(0, 0, 0));
  labelVideo -> setPixmap(QPixmap::fromImage(imgBlack));
}

void interfaz::captureVideo() {

  if ((detector -> isRunning()) && (detector -> getCommand() == 1)) { //Camera capture has priority

    detector -> stop();
    detector -> wait();

  } else {

    if (!detector -> detectorIsReady()) { //The detector must be loaded
      QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("The face detector is not configured yet."), QMessageBox::Ok);
      return;
    }

    int tempReturn = detector -> detectObjectVideoCamera(lineEditDevice -> text().toInt());
    if (tempReturn != -1) {
      /*In case the device opening fails or if the frames produced by the device exceed the maximum search window size*/

      if (tempReturn == 0) {
        QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("Could not open the device ") + lineEditDevice -> text(), QMessageBox::Ok);
      } else {
        QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("Could not analyze the frames of the device ") + lineEditDevice -> text() + QString::fromUtf8(" because its largest side measures ") + QString::number(tempReturn) + QString::fromUtf8(" pixels and the maximum search window size is ") + QString::number(detector -> getSizeMaxWindow()) + QString::fromUtf8(" pixels, please set a size equal to or greater than the search window size and try again "), QMessageBox::Ok);
      }

    } else {
      listImages -> setFlagDetectorIsActive(true);
    }

  }

}

void interfaz::setTextInDetection(const cv::Point point,
  const QString text) //For the image result
{

  std::cout << "Entering text\n";
  textCv = cv::Mat::zeros(textCv.rows, textCv.cols, CV_8UC3); //Assumed previous initialization
  cv::putText(textCv, text.toStdString(), point, fontTypeTextRecognition, fontScaleTextRecognition, colorTextRecognition, thicknessTextRecognition, lineTypeTextRecognition);
  showVideo(imgCv); //Display the result
  std::cout << "Exiting text\n";

}

void interfaz::setTextInDetection(const std::vector < cv::Point > listPoints,
  const std::vector < std::string > listText) {

  textCv = cv::Mat::zeros(textCv.rows, textCv.cols, CV_8UC3); //Assumed previous initialization
  for (int i = 0; i < listPoints.size(); i++)
    cv::putText(textCv, listText[i], listPoints[i], fontTypeTextRecognition, fontScaleTextRecognition, colorTextRecognition, thicknessTextRecognition, lineTypeTextRecognition);

  //showVideo(imgCv);//Display the result

}

void interfaz::clearText() {
  textCv = cv::Mat::zeros(textCv.rows, textCv.cols, CV_8UC3); //Assumed previous initialization
  showVideo(imgCv); //Display the result
}

void interfaz::showVideo(const cv::Mat & img) {

  imgCv = textCv + img;
  QImage image(imgCv.data, imgCv.cols, imgCv.rows, imgCv.step, QImage::Format_RGB888);
  labelVideo -> setPixmap(QPixmap::fromImage(image.rgbSwapped()));

}

void interfaz::showImage(const cv::Mat & img) {

  QImage image(img.data, img.cols, img.rows, img.step, QImage::Format_RGB888);
  labelVideo -> setPixmap(QPixmap::fromImage(image.rgbSwapped()));

}

void interfaz::activeFlowsViewImages(viewerListImages::flowType type) {

  switch (type) {

  case viewerListImages::connectDetector:

    connect(detector, SIGNAL(listCoordinatesAndDetectedObjects(std::vector < cv::Mat > , std::vector < cv::Rect > )), this, SLOT(detectionImagesList(std::vector < cv::Mat > )));
    connect(detector, SIGNAL(listCoordinatesAndDetectedObjectsRotated(std::vector < cv::Mat > , std::vector < cv::RotatedRect > )), this, SLOT(detectionImagesList(std::vector < cv::Mat > )));

    connect(detector, SIGNAL(listCoordinatesAndDetectedObjects_img(std::vector < cv::Mat > , std::vector < cv::Rect > )), this, SLOT(detectionImagesList(std::vector < cv::Mat > )));
    connect(detector, SIGNAL(listCoordinatesAndDetectedObjectsRotated_img(std::vector < cv::Mat > , std::vector < cv::RotatedRect > )), this, SLOT(detectionImagesList(std::vector < cv::Mat > )));

    listImages -> clear();
    break;
  case viewerListImages::disconnectDetector:

    disconnect(detector, SIGNAL(listCoordinatesAndDetectedObjects(std::vector < cv::Mat > , std::vector < cv::Rect > )), this, SLOT(detectionImagesList(std::vector < cv::Mat > )));
    disconnect(detector, SIGNAL(listCoordinatesAndDetectedObjectsRotated(std::vector < cv::Mat > , std::vector < cv::RotatedRect > )), this, SLOT(detectionImagesList(std::vector < cv::Mat > )));

    disconnect(detector, SIGNAL(listCoordinatesAndDetectedObjects_img(std::vector < cv::Mat > , std::vector < cv::Rect > )), this, SLOT(detectionImagesList(std::vector < cv::Mat > )));
    disconnect(detector, SIGNAL(listCoordinatesAndDetectedObjectsRotated_img(std::vector < cv::Mat > , std::vector < cv::RotatedRect > )), this, SLOT(detectionImagesList(std::vector < cv::Mat > )));

    break;
  case viewerListImages::connectRecognizer:
    connect(facialRecognizer, SIGNAL(recognizedImage(imageTransaction)), this, SLOT(recognitionImagesList(imageTransaction)), Qt::QueuedConnection);
    connect(facialRecognizer, SIGNAL(recognizedImage(const cv::Point,
      const QString, cv::Mat)), this, SLOT(recognitionImagesList(const cv::Point,
      const QString, cv::Mat)), Qt::QueuedConnection);
    listImages -> clear();
    break;
  case viewerListImages::disconnectRecognizer:
    disconnect(facialRecognizer, SIGNAL(recognizedImage(imageTransaction)), this, SLOT(recognitionImagesList(imageTransaction)));
    disconnect(facialRecognizer, SIGNAL(recognizedImage(const cv::Point,
      const QString, cv::Mat)), this, SLOT(recognitionImagesList(const cv::Point,
      const QString, cv::Mat)));
    break;

  }

}

void interfaz::detectionImagesList(std::vector < cv::Mat > listDetectedObjects) {
  listImages -> addImagesList(listDetectedObjects);
}

void interfaz::recognitionImagesList(imageTransaction myImageTransaction) {

  listImages -> addImages(myImageTransaction.img, myImageTransaction.name);

}

void interfaz::recognitionImagesList(const cv::Point,
  const QString text, cv::Mat img) {

  listImages -> addImages(img, text.toStdString());

}

void interfaz::showNumberLoadFiles(int totalFiles, int currentFile) {

  if (totalFiles == 0)
    labelInfoFilesLoads -> setText("<font color=white>" + QString::fromUtf8("0 files loaded"));
  else
    labelInfoFilesLoads -> setText("<font color=white>" + QString::number(currentFile) + QString(" of ") + QString::number(totalFiles) + QString(" in total"));

}

void interfaz::showPlayTimeTotal(int minutes, int seconds) {
  qstrTotalSeconds = QString("%1").arg(seconds, 2, 10, QChar('0'));
  qstrTotalMinutes = QString::number(minutes);
}

void interfaz::showPlayTimeRemaining(int minutes, int seconds) {
  QString qstrActualSeconds = QString("%1").arg(seconds, 2, 10, QChar('0'));
  QString qstrActualMinutes = QString::number(minutes);
  labelInfoPlayTime -> setText("<font color=white>" + QString("-") + qstrActualMinutes + QString(":") + qstrActualSeconds + QString("/") + qstrTotalMinutes + QString(":") + qstrTotalSeconds);
}

void interfaz::configDetector() {

  int tempCommand = detector -> getCommand();
  if (detector -> isRunning()) {

    detector -> stop();
    detector -> wait();
  }

  guiDetector -> exec();

  if (tempCommand == 1) // Was capturing video from camera
    captureVideo();
  else if (!nameFiles.empty()) // From file if images or videos are loaded (internally the function checks)
    currentFile();

}

void interfaz::configTrackerWindows() {

  int tempCommand = detector -> getCommand();
  if (detector -> isRunning()) {

    detector -> stop();
    detector -> wait();
  }

  myTrackerWindows_gui -> exec();

  if (tempCommand == 1) // Was capturing video from camera
    captureVideo();
  else if (!nameFiles.empty()) // From file if images or videos are loaded (internally the function checks)
    currentFile();

}

void interfaz::configFacialRecognizer() {

  int tempCommand = detector -> getCommand();
  if (detector -> isRunning()) {

    detector -> stop();
    detector -> wait();
  }

  guiFacialRecognizer -> exec();

  if (tempCommand == 1) // Was capturing video from camera
    captureVideo();
  else if (!nameFiles.empty()) // From file if images or videos are loaded (internally the function checks)
    currentFile();

}

void interfaz::presentationRecognitionConfig() {
  guiConfigPresentationRecognition -> exec();
}

void interfaz::loadFiles() {

  QStringList tempNameFiles = QFileDialog::getOpenFileNames(
    this,
    QString::fromUtf8("Select one or more image or video files"),
    LAST_PATH_FILES_TEST,
    QString::fromUtf8("Images or video") + QString("(") + qstrFormatImages + qstrFormatVideos + QString(")"));

  if (tempNameFiles.empty()) { 
      return;
  } else {
      QFileInfo fileInfo(tempNameFiles[0]);
      LAST_PATH_FILES_TEST = fileInfo.path();
  }
  

  nameFiles = tempNameFiles;

  indexNameFile = -1;
  showNumberLoadFiles(nameFiles.size(), 0);
  next();

}

void interfaz::next() {

  if (nameFiles.empty()) return;

  if ((detector -> isRunning()) && (detector -> getCommand() == 1)) return; // Camera capture has priority

  if (!detector -> detectorIsReady()) { // The detector must be loaded
    QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("The face detector is not configured yet."), QMessageBox::Ok);
    return;
  }

  indexNameFile++;
  if (indexNameFile >= nameFiles.size()) indexNameFile = 0;

  currentFile();

}

void interfaz::back() {

  if (nameFiles.empty()) return;

  if ((detector -> isRunning()) && (detector -> getCommand() == 1)) return; // Camera capture has priority

  if (!detector -> detectorIsReady()) { // The detector must be loaded
    QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("The face detector is not configured yet."), QMessageBox::Ok);
    return;
  }

  indexNameFile--;
  if (indexNameFile < 0) indexNameFile = nameFiles.size() - 1;

  currentFile();

}

void interfaz::currentFile() {

  std::cout << "The reading device will be sent to the detector next\n";

  showNumberLoadFiles(nameFiles.size(), indexNameFile + 1);

  // Checking the image format
  QString tempNameFile = nameFiles[indexNameFile];
  QString tempFormat = (tempNameFile.split("."))[1];

  if (formatVideos.contains(tempFormat)) // It's a video
  {

    int tempReturn = detector -> detectObjectVideoFile(tempNameFile.toStdString());

    if (tempReturn != -1) {
      /* In case the device cannot be opened or if the frames produced by the device are larger than the maximum window size */

      if (tempReturn == 0) {
        QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("Could not open the video ") + tempNameFile, QMessageBox::Ok);
      } else {
        QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("Could not analyze the frames of the video ") + tempNameFile + QString::fromUtf8(" because its largest side is ") + QString::number(tempReturn) + QString::fromUtf8(" pixels and the maximum window size is ") + QString::number(detector -> getSizeMaxWindow()) + QString::fromUtf8(" pixels, please set a size equal to or larger for the search window and try again"), QMessageBox::Ok);
      }

    } else {
      listImages -> setFlagDetectorIsActive(true);
    }

  } else // It's an image
  {

    int tempReturn = detector -> detectObjectImageFile(tempNameFile.toStdString());

    if (tempReturn != -1) {
      /* In case the device cannot be opened or if the frames produced by the device are larger than the maximum window size */

      if (tempReturn == 0) {
        QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("Could not open the image ") + tempNameFile, QMessageBox::Ok);
      } else {
        QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("Could not analyze the frames of the image ") + tempNameFile + QString::fromUtf8(" because its largest side is ") + QString::number(tempReturn) + QString::fromUtf8(" pixels and the maximum window size is ") + QString::number(detector -> getSizeMaxWindow()) + QString::fromUtf8(" pixels, please set a size equal to or larger for the search window and try again"), QMessageBox::Ok);
      }

    } else {
      listImages -> setFlagDetectorIsActive(true);
    }

  }

}

void interfaz::finishedDetection() {
  listImages -> setFlagDetectorIsActive(false);
}
