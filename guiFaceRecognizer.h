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

#ifndef GUIFACERECOGNIZER_H
#define GUIFACERECOGNIZER_H

//QT
#include <QDialog>
#include <QMessageBox>
#include <QImage>
#include <QThread>
#include <QMutexLocker>
#include <QMutex>
#include <QLabel>
#include <QThread>
//openCV
#include "opencv2/opencv.hpp"

//Custom forward declarations
class RECOGNIZER_FACIAL;
class DATA_BASE;
class USER;
class PLOT_SPARSE_SOLUTION;

//Qt forward declarations
class QLineEdit;
class QCheckBox;
class QPushButton;
class QLabel;
class QStackedWidget;
class QVBoxLayout;
class QGridLayout;
class QGroupBox;
class QHBoxLayout;
class QTabWidget;
class QCompleter;
class QMouseEvent;
class QMenu;
class QAction;
class QComboBox;
class QProgressDialog;
class QCloseEvent;

class videoStream: public QThread {
  Q_OBJECT

  volatile bool stopped;
  QMutex mutex;

  cv::VideoCapture cap;
  cv::Mat frame;

  public:
    videoStream();
  ~videoStream();
  bool setDevice(int device);
  void stop();
  cv::Mat getFrame() const;

  signals:
    void emitFrame(QImage qframe);
  void stoppedeExecuted();
  protected:
    void run();

};

class labelRubberBand: public QLabel {
  Q_OBJECT

  QRect rubberBandRect;
  bool rubberBandIsShown;
  QPoint origin;
  QPoint endPos;

  QPixmap pixmap;

  //QMENU
  QMenu * contextMenuImage;

  //QACTION
  QAction * actionCropImage;

  bool flagCrop; //Indicates whether the image has been cropped (true) or not (false) when the image changes (pushImageToCrop(cv::Mat *img))

  public:
    cv::Mat * imgToCrop; //This will store the address of the image to crop

  public:
    labelRubberBand();
  ~labelRubberBand();
  void setPixmap(QPixmap image);
  void setPixmap(cv::Mat & img);
  void clearImageToCrop();

  bool imageIsCrop() const;

  public slots:
    void cropImage();
  void pushImageToCrop(cv::Mat * img);

  signals:
    void imageIscrop();

  protected:
    void mousePressEvent(QMouseEvent * event);
  void mouseMoveEvent(QMouseEvent * event);
  void mouseReleaseEvent(QMouseEvent * event);
  void paintEvent(QPaintEvent * event);
};


class GUI_FACE_RECOGNIZER: public QDialog {

  Q_OBJECT

  //QLabel
  int widthLabelGraphic;
  int highLabelGraphic;
  QImage imageShow;

  //RubberBand
  labelRubberBand * graphicAndRubberBand;

  QLabel * infoGraphic;
  QStackedWidget * stackInfoGraphic;

  QLabel * labelConfigDataBase;
  QLabel * labelConfigDescriptor;
  QLabel * labelConfigSparseSolution;
  QLabel * labelTestFaceRecognizer;

  QLabel * labelCurrentWidthImage;
  QLabel * labelHighCurrentImage;
  QLabel * labelNumberImages;
  QLabel * labelNumberUsers;
  QLabel * labelNumberCurrenImage;

  QLabel * labelStateDataBase; // Will show the information about the database (whether it has been edited or if all descriptors have been calculated)
  QLabel * labelStateDescriptor; // Will show the information about the descriptor (whether it has been changed or not)

  QLabel * labelShowRecognitionResult; // Will show the result of the recognition for the test phase

  //QPushButton
  QPushButton * buttonbackImageDataBase;
  QPushButton * buttonnextImageDataBase;
  QPushButton * buttonBurstCapture;

  QPushButton * buttonAddNewUser;
  QPushButton * buttonDeleteUser;

  QPushButton * buttonAddImagesFile;

  QPushButton * buttonDeleteImage;
  QPushButton * buttonApplyImage;
  QPushButton * buttonApplyUser;
  QPushButton * buttonApplyDataBase;

  QPushButton * buttonAddImageToUser;

  QPushButton * buttonLoadDataBase;
  QPushButton * buttonApplyTotalConfig;

  QPushButton * buttonPreviousConfiguration;

  QPushButton * buttonApplyConfigDictionary;

  //Belonging to the test label
  QPushButton * buttonApplyConfigTest;
  QPushButton * buttonLoadImageFromFileToRecognize;
  QPushButton * buttonGraphicDispersedSolution;
  QPushButton * buttonGenerateTest;
  QPushButton * buttonbackImageTest;
  QPushButton * buttonnextImageTest;
  QPushButton * buttonBurstCaptureTest;

  //QLineEdit
  QLineEdit * lineEditDevice;
  QLineEdit * lineEditNameUser;

  QLineEdit * lineEditWidthImages;
  QLineEdit * lineEditHighImages;

  //For the sparse solution configuration
  QLineEdit * lineEdit_lm;
  QLineEdit * lineEdit_nc;
  QLineEdit * lineEditNumberZeros;
  QLineEdit * lineEdit_ck;
  QLineEdit * lineEditRecognitionThreshold;

  //Belonging to the test label
  QLineEdit * lineEditMaximumStackImages;
  QLineEdit * lineEditWidthTestImage;
  QLineEdit * lineEditHighTestImage;
  QLineEdit * lineEditDeviceTest;

  //QCheckBox;
  QCheckBox * checkBoxCaptureCam;
  QCheckBox * checkBoxMirrorImage;

  //QVBoxLayout
  QVBoxLayout * layoutPrincipal;
  QVBoxLayout * layoutGraphicPrincipal;
  QVBoxLayout * layoutStackAndButtonApplyTotalConfig;

  //QHBoxLayout
  QHBoxLayout * layoutStackInfoGraphic;

  QHBoxLayout * layoutAddDeleteUser;

  //QGridLayout
  QGridLayout * layoutinfoGraphic1;

  QGridLayout * layoutConfigGetImages;
  QGridLayout * layoutConfigResizeImages;
  QGridLayout * layoutPrincipalConfigDataBase;

  QGridLayout * layoutConfigDescriptors;

  //QGroupBox
  QGroupBox * groupBoxInfoGraphic1;
  QGroupBox * groupBoxInfoGraphic2;

  QGroupBox * groupBoxConfigGetImages;
  QGroupBox * groupBoxConfigResizeImages;

  //QTabWidget
  QTabWidget * tabWidgetConfig;

  //QStackedWidget
  QStackedWidget * widgetsDescriptors;

  //QComboBox
  QComboBox * comboBoxDescriptors;

  //QProgressDialog
  QProgressDialog * progressDialogDescriptorsCalculation;

  //QMessageBox
  QMessageBox messageGeneratingTest;

  //VIDEO
  videoStream videoThread;

  //PLOT_SPARSE_SOLUTION
  PLOT_SPARSE_SOLUTION * plotSparseSolution;

  //________Objects not related to the graphics_____________//
  RECOGNIZER_FACIAL * recognizerFacial; // Facial recognition thread
  DATA_BASE * dataBase; // Database
  QCompleter * completer;
  USER * currentUser; // Represents the current user being operated on, if no user exists, it should point to NULL
  cv::Mat imageEditing;
  bool flagGenerateImageMirror;
  //____These variables are required for tracking descriptor edits____//
  QString previousDescriptorName; // Descriptor currently used for calculated descriptors (nameDescriptor.txt)
  bool flagDescriptorEditing; // Indicates whether the descriptor has been edited (true) or not (false)
  bool flagChangeDescriptor; // Indicates whether there has been a change in the descriptor
  bool flagGenerateTest; // Indicates whether a test is being generated on a database
  //_______________________________________________________________________________________//
  //These variables are used in the test phase
  cv::Mat imageEditingTest;
  QStringList listFileNameImagesTest;
  int indexImagesTest;
  //_____________________________________________________________//
  
  public:
    GUI_FACE_RECOGNIZER(RECOGNIZER_FACIAL * myrecognizerFacial);
  ~GUI_FACE_RECOGNIZER();

  void setLabelGraphic();
  void setLabelInfoGraphic1();
  void setLabelInfoGraphic2();
  void setStackInfoGraphic();
  void setTabWidgetConfig();
  void setMyLayouts();

  // Database configuration functions
  void setAddDeleteUserConfig();
  void setSourceImagesConfig();
  void setEditImagesConfig();
  void setLayoutPrincipalDataBaseConfig();

  void setNumberUsersInfo(int i = -1);
  void setNumberImagesInfo(int i = -1);
  void setNumberCurrentImageInfo(int i = -1);
  void setWidthCurrentImageInfo(int i = -1);
  void setHighCurrentImageInfo(int i = -1);

  void setEnabledPanelAddDeleteUserConfig(bool flagEnabled);
  void setEnabledPanelAcquisitionImages(bool flagEnabled);
  void setEnabledPanelEditionImages(bool flagEnabled);
  void setEnabledPanelConfigDataBase(bool flagEnabled);

  void setCompleterNameUsers();

  // Descriptor configuration functions
  void setLayoutPrincipalDescriptorConfig();
  void editedDescriptor(bool flag); // Informs if the descriptor has been edited

  // Related to descriptor configuration
  QString getPreviousDescriptorName(); /* Retrieves the descriptor name currently used to calculate descriptors from disk, returns an empty QString if it doesn't exist */
  void saveNameCurrentDescriptor(); // Saves the current selected descriptor name to disk

  // Sparse solution configuration functions
  void setLayoutPrincipalSparseSolution();
  void readStateDictionary(); // Reads the state of the dictionary and shows it in labelConfigSparseSolution

  // Test configuration functions
  void setLayoutPrincipalTest();

  public slots:
    void showImage(QImage image);
  void clearLabelGraphic(); // slot
  void currentIndexTabWidget(int index);
  void message_errorDescriptors(const QString & infoText); // Error message indicating that descriptors or images do not exist
  void enabledProgressbar(bool flag); // Enables or disables progressDialogDescriptorsCalculation

  // Related to global configuration
  void applyTotalConfig(); // slot
  void loadedDatabase(); // This slot should be called when the database has finished loading
  void calculationEndDescriptors(bool flag); /* This slot should be called when descriptors finish calculating, flag=true if completed correctly, or flag=false if the operation was canceled */

  // Database configuration slot
  void editedDatabase(bool flag); // Informs if the database has been edited (slot)
  void editingNameUserFinished(); // slot
  void editingNameUserTextChanged(); // slot
  void captureImages(); // slot
  void loadDataBase(); // slot
  void addUser(); // slot
  void deleteUser(); // slot
  void activeCaptureFromCam(int state); // slot
  void setGenerateImageMirror(int state); // slot
  void addImagenCaptured(); // This attaches the captured image (slot)
  void loadImagesFromFile(); // slot
  void insertExternalImages(std::vector < cv::Mat > listImages, QString nameUser);
  void seeNextImage(); // slot
  void seeBackImage(); // slot
  void resizeImage(); /* Applies scaling to the current image in editing, if the values in lineEditWidthImages and lineEditHighImages are zero or an empty QString, no scaling is applied and only the crop operation is considered (slot) */
  void resizeAllImage(); /* Applies equal scaling to all images of the user (plus a crop operation to the current image in editing if it happened), if the values in lineEditWidthImages and lineEditHighImages are zero or an empty QString, no scaling is applied and nothing happens (including the crop operation of the current image in editing, if it took place) (slot) */
  void resizeAllDataBase(); /* Applies equal scaling to all images in the database (plus a crop operation to the current image in editing if it happened), if the values in lineEditWidthImages and lineEditHighImages are zero or an empty QString, no scaling is applied and nothing happens (including the crop operation of the current image in editing, if it took place) (slot) */
  void deleteImage(); // Deletes the current image in editing (slot)

  // Descriptor configuration slot
  void setDescriptor();
  void changeDescriptor(int index); // slot
  void editingDescriptor(); // This slot will be called if any descriptor has been edited (slot)
  void previousConfiguration(); // slot

  // Sparse solution configuration slots
  void editedConfigSparseSolution();
  void ApplyConfigSparseSolution();

  // Test configuration slots
  void setConfigurationTest();
  void editedConfigTest();
  void ApplyConfigTest();
  void captureImagesTest();
  void cropImageTest();
  void loadImagesToTest();
  void seeNextImageTest();
  void seeBackImageTest();
  void setTestResultImageInfo(const QString & infoText);
  void generateTest();
  void testWasGenerated();

  // Related to plotSparseSolution
  void setPlotSparseSolution();
  void resultsSparseSolution();

  signals:
    void startLoadDataBase(QString namePathDataBase);
  void startCalculateDescriptors();
    void startRecognizeFaceImage(cv::Mat img);
  void recognizerIsReady(bool flag);

  protected:
    void closeEvent(QCloseEvent * event);

};

#endif
