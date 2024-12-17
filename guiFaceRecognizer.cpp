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


//Own
#include "guiFaceRecognizer.h"
#include "dataBaseImages.h"
#include "recognizerFacial.h"
#include "plotSparseSolution.h"
//C Standard
#include <stdio.h>
//QT
#include <QLineEdit>
#include <QCheckBox>
#include <QPushButton>
#include <QLabel>
#include <QStackedWidget>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QPlastiqueStyle>
#include <QTabWidget>
#include <QCompleter>
#include <QFileDialog>
#include <QToolTip>
#include <QMouseEvent>
#include <QStylePainter>
#include <QPainter>
#include <QMenu>
#include <QAction>
#include <QComboBox>
#include <QFormLayout>
#include <QProgressDialog>
#include <QCloseEvent>
#include <QDir>
#include <QFileInfo>
//Debug
#include <QDebug>

//path of the last image loaded into the database
QString LAST_PATH_IMAGE = QDir::homePath();
//path of the last test image loaded
QString LAST_PATH_IMAGE_TEST = QDir::homePath();


//________________________VIDEO SERVER CLASS_________________________________________//

videoStream::videoStream() {

  stopped = true;

}

videoStream::~videoStream() {
  stop();
  wait();
  qDebug() << "videoStream stopped\n"; //Debug message
}

void videoStream::stop() {

  mutex.lock();
  stopped = true;
  mutex.unlock();

}

cv::Mat videoStream::getFrame() const {
  return frame.clone();
}

bool videoStream::setDevice(int device) {

  if (isRunning()) return false; //Cannot select another device if capturing frames

  //cap.open(device);

  //____________________________________________________//

  if (device == -1) {
    //check DEVICE_URL
    const char * DEVICE_URL = std::getenv("DEVICE_URL");
    if (DEVICE_URL != NULL) {
      cap.open(DEVICE_URL);
      std::cout << "opening: " << DEVICE_URL << std::endl; //Console output
    }
  } else {
    //check other devices
    cap.open(device);
  }

  //____________________________________________________//

  if (!cap.isOpened()) return false; //If the device doesn't exist

  return true;

}

void videoStream::run() {

  if (!cap.isOpened()) return; //In case no device was selected

  stopped = false;

  while (true) {

    //________THREAD CONTROL_____________
    {
      QMutexLocker locker( & mutex);
      if (stopped) {
        stopped = false;
        break;
      }
    }
    //_______________________________________

    //_________________CAPTURE_______________
    cap >> frame;
    QImage image(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
    emitFrame(image);
    //_______________________________________

  }

  cap.release();

}

labelRubberBand::labelRubberBand() {

  rubberBandIsShown = false;
  imgToCrop = NULL;

  setFrameShape(QFrame::StyledPanel);
  setBackgroundRole(QPalette::Base);

  //_______________MENUS AND ACTIONS____________________________//
  setContextMenuPolicy(Qt::CustomContextMenu);
  contextMenuImage = new QMenu(this);
  actionCropImage = new QAction(QString::fromUtf8("Crop image"), this); //Translated message
  contextMenuImage -> addAction(actionCropImage);

  connect(actionCropImage, SIGNAL(triggered()), this, SLOT(cropImage()));

  //__________________________________________________________//

}

labelRubberBand::~labelRubberBand() {

}

void labelRubberBand::setPixmap(QPixmap image) {
  pixmap = image.scaled(size());
  update();
}

void labelRubberBand::setPixmap(cv::Mat & img) {

  flagCrop = false;

  imgToCrop = & img;

  QImage image(imgToCrop -> data, imgToCrop -> cols, imgToCrop -> rows, imgToCrop -> step, QImage::Format_RGB888);
  setPixmap(QPixmap::fromImage(image.rgbSwapped()));

}

void labelRubberBand::clearImageToCrop() {
  imgToCrop = NULL;
}

bool labelRubberBand::imageIsCrop() const {
  return flagCrop;
}

void labelRubberBand::cropImage() {

  if (imgToCrop == NULL) return;

  //_________________CALCULATE NEW COORDINATES_______________________//
  int rowi = (imgToCrop -> rows) * (double(origin.y()) / height());
  int rowf = (imgToCrop -> rows) * (double(endPos.y()) / height());

  int coli = (imgToCrop -> cols) * (double(origin.x()) / width());
  int colf = (imgToCrop -> cols) * (double(endPos.x()) / width());
  //_____________________________________________________________________//

  if ((rowf > rowi) && (colf > coli)) {

    //Image cropped
    ( * imgToCrop) = ( * imgToCrop)(cv::Range(rowi, rowf), cv::Range(coli, colf)).clone();

    //Display the resulting image
    QImage image(imgToCrop -> data, imgToCrop -> cols, imgToCrop -> rows, imgToCrop -> step, QImage::Format_RGB888);
    setPixmap(QPixmap::fromImage(image.rgbSwapped()));

  }

  flagCrop = true;

  emit imageIscrop();
}

void labelRubberBand::pushImageToCrop(cv::Mat * img) {

  flagCrop = false;
  imgToCrop = img;

}

void labelRubberBand::mousePressEvent(QMouseEvent * event) {

  origin = event -> pos();
  rubberBandRect = QRect(0, 0, 0, 0);
  update();

}

void labelRubberBand::mouseMoveEvent(QMouseEvent * event) {

  QPoint newPos = event -> pos();

  if ((newPos.x() > 0) && (newPos.y() > 0) && (newPos.x() < width()) && (newPos.y() < height())) {

    endPos = newPos;
    rubberBandRect = QRect(origin, newPos);

    QToolTip::showText(event -> globalPos(), QString("%1,%2")
      .arg(rubberBandRect.size().width())
      .arg(rubberBandRect.size().height()), this); //Tooltip for coordinates

    update();

  }

}

void labelRubberBand::mouseReleaseEvent(QMouseEvent * event) {

  if ((rubberBandRect.width() > 5) && (rubberBandRect.height() > 5)) {
    contextMenuImage -> exec(mapToGlobal(endPos));
  }

  rubberBandRect = QRect(0, 0, 0, 0);
  update();

}

void labelRubberBand::paintEvent(QPaintEvent * event) {

  QStylePainter painter(this);

  painter.drawPixmap(0, 0, pixmap);

  if ((rubberBandRect.width() > 5) && (rubberBandRect.height() > 5)) {
    QPainterPath path;
    path.addRect(rubberBandRect);
    painter.fillPath(path, QColor(0, 0, 60, 60));
    painter.setPen(QPen(Qt::blue, 2, Qt::DashDotLine));
    painter.drawRect(rubberBandRect.normalized().adjusted(0, 0, -1, -1));
  }

  QLabel::paintEvent(event);

}

//__________________________________________________________________________________________//
GUI_FACE_RECOGNIZER::GUI_FACE_RECOGNIZER(RECOGNIZER_FACIAL * myrecognizerFacial): recognizerFacial(myrecognizerFacial) {

  //____________Initial logical states________________________
  completer = NULL;
  //dataBase=mydataBase; //Database
  dataBase = recognizerFacial -> getDataBase();
  currentUser = NULL;
  flagGenerateImageMirror = false;
  flagGenerateTest = false;
  //______________________________________________________

  //______CONFIGURATION OF INFORMATION MESSAGE WHILE TEST IS BEING GENERATED________//

  messageGeneratingTest.setText(QString::fromUtf8("Generating test on the database"));
  messageGeneratingTest.setWindowTitle("Generating Test");
  messageGeneratingTest.setStandardButtons(QMessageBox::NoButton);
  //________________________________________________________________________________________//

  //________Pre-configuration of progressDialogDescriptorsCalculation_____________//
  progressDialogDescriptorsCalculation = new QProgressDialog(this);
  progressDialogDescriptorsCalculation -> setMinimumDuration(0);
  progressDialogDescriptorsCalculation -> setMinimum(1);
  progressDialogDescriptorsCalculation -> setWindowTitle("Descriptor calculation in progress");
  progressDialogDescriptorsCalculation -> setFixedWidth(600);
  progressDialogDescriptorsCalculation -> setWindowFlags(Qt::Window | Qt::WindowTitleHint | Qt::CustomizeWindowHint);
  progressDialogDescriptorsCalculation -> move(geometry().center());
  connect(recognizerFacial, SIGNAL(currentUserProcessing(int)), progressDialogDescriptorsCalculation, SLOT(setValue(int)));
  connect(recognizerFacial, SIGNAL(currentUserProcessingInfo(const QString & )), progressDialogDescriptorsCalculation, SLOT(setLabelText(const QString & )));
  connect(progressDialogDescriptorsCalculation, SIGNAL(canceled()), recognizerFacial, SLOT(stop()), Qt::DirectConnection);

  //_________________________________________________________________________________//

  buttonLoadDataBase = new QPushButton("Load database");
  buttonLoadDataBase -> setAutoDefault(false);
  connect(buttonLoadDataBase, SIGNAL(clicked()), this, SLOT(loadDataBase()));

  buttonApplyTotalConfig = new QPushButton("Apply"); //Used to apply the global configuration
  connect(buttonApplyTotalConfig, SIGNAL(clicked()), this, SLOT(applyTotalConfig()));
  buttonApplyTotalConfig -> setAutoDefault(false);
  buttonApplyTotalConfig -> setEnabled(false);

  //Set the graph for the sparse solution results
  plotSparseSolution = new PLOT_SPARSE_SOLUTION;

  setLabelGraphic();
  setLabelInfoGraphic1();
  setLabelInfoGraphic2();
  setStackInfoGraphic();
  setTabWidgetConfig();

  //____Setting up the database configuration panel_______//
  setAddDeleteUserConfig();
  setSourceImagesConfig();
  setEditImagesConfig();
  setLayoutPrincipalDataBaseConfig();
  setLayoutPrincipalDescriptorConfig();
  setLayoutPrincipalSparseSolution();
  setLayoutPrincipalTest();
  //___________________________________________________________________//

  setMyLayouts();
  setWindowTitle(QString::fromUtf8("FACE RECOGNIZER CONFIGURATION"));

  //____________Initial graphic states________________________
  setEnabledPanelConfigDataBase(false);
  //______________________________________________________

  //INITIAL CONNECTIONS
  connect(tabWidgetConfig, SIGNAL(currentChanged(int)), this, SLOT(currentIndexTabWidget(int)));
  connect( & videoThread, SIGNAL(emitFrame(QImage)), this, SLOT(showImage(QImage)), Qt::QueuedConnection);
  connect(dataBase, SIGNAL(haveBeenEdited(bool)), this, SLOT(editedDatabase(bool)));
  connect(recognizerFacial, SIGNAL(editingDescriptor()), this, SLOT(editingDescriptor()));
  connect(recognizerFacial, SIGNAL(loadedDatabase()), this, SLOT(loadedDatabase()));
  connect(recognizerFacial, SIGNAL(calculationEndDescriptors(bool)), this, SLOT(calculationEndDescriptors(bool)));
  connect(recognizerFacial, SIGNAL(testResultImageInformation(const QString & )), this, SLOT(setTestResultImageInfo(const QString & )));
  connect(recognizerFacial, SIGNAL(zeroDescriptorsOrZeroImages(const QString & )), this, SLOT(message_errorDescriptors(const QString & )));
  connect(recognizerFacial, SIGNAL(enabledProgressbar(bool)), this, SLOT(enabledProgressbar(bool)));
  connect(recognizerFacial, SIGNAL(setPlotSparseSolution()), this, SLOT(setPlotSparseSolution()), Qt::QueuedConnection);
  connect(recognizerFacial, SIGNAL(testWasGenerated()), this, SLOT(testWasGenerated()), Qt::QueuedConnection);

  connect(this, SIGNAL(startLoadDataBase(QString)), recognizerFacial, SLOT(startLoadDataBase(QString)), Qt::QueuedConnection);
  connect(this, SIGNAL(startCalculateDescriptors()), recognizerFacial, SLOT(startCalculateDescriptors()), Qt::QueuedConnection);
  connect(this, SIGNAL(startRecognizeFaceImage(cv::Mat)), recognizerFacial, SLOT(startRecognizeFaceImage(cv::Mat)));

}

GUI_FACE_RECOGNIZER::~GUI_FACE_RECOGNIZER() {

  delete plotSparseSolution;

}

void GUI_FACE_RECOGNIZER::setLabelGraphic() {

  widthLabelGraphic = 640 / 1.5;
  highLabelGraphic = 480 / 1.5;

  //______________________________________________________________________________________________________
  graphicAndRubberBand = new labelRubberBand;
  graphicAndRubberBand -> setFixedSize(widthLabelGraphic, highLabelGraphic);

  clearLabelGraphic();
  //______________________________________________________________________________________________________

}

void GUI_FACE_RECOGNIZER::setLabelInfoGraphic1() {

  //_____________Info graphic 1________________________
  buttonbackImageDataBase = new QPushButton(QIcon(QPixmap("./application_images/Buttons/back.png")), "");
  connect(buttonbackImageDataBase, SIGNAL(clicked()), this, SLOT(seeBackImage()));
  buttonbackImageDataBase -> setIconSize(QSize(30, 25));
  buttonbackImageDataBase -> setFixedWidth(70);
  buttonbackImageDataBase -> setAutoDefault(false);
  buttonbackImageDataBase -> setEnabled(false);
  buttonnextImageDataBase = new QPushButton(QIcon(QPixmap("./application_images/Buttons/next.png")), "");
  connect(buttonnextImageDataBase, SIGNAL(clicked()), this, SLOT(seeNextImage()));
  buttonnextImageDataBase -> setIconSize(QSize(30, 25));
  buttonnextImageDataBase -> setFixedWidth(70);
  buttonnextImageDataBase -> setAutoDefault(false);
  buttonnextImageDataBase -> setEnabled(false);

  //____________________________________________________//
  buttonBurstCapture = new QPushButton(QIcon(QPixmap("./application_images/Buttons/BUTTON-REC.png")), ""); //Capture
  connect(buttonBurstCapture, SIGNAL(clicked()), this, SLOT(captureImages()));
  buttonBurstCapture -> setAutoDefault(false);
  buttonBurstCapture -> setFixedHeight(70);
  buttonBurstCapture -> setFixedWidth(70);
  buttonBurstCapture -> setFlat(false);
  buttonBurstCapture -> setEnabled(false);

  QRegion * region = new QRegion( * (new QRect(buttonBurstCapture -> x() + 5, buttonBurstCapture -> y() + 5, 60, 60)), QRegion::Ellipse);
  buttonBurstCapture -> setMask( * region);

  //____________________________________________________//

  layoutinfoGraphic1 = new QGridLayout;
  layoutinfoGraphic1 -> addWidget(buttonbackImageDataBase, 0, 2, 1, 1);
  layoutinfoGraphic1 -> addWidget(buttonBurstCapture, 0, 3, 1, 1);
  layoutinfoGraphic1 -> addWidget(buttonnextImageDataBase, 0, 4, 1, 1);

  groupBoxInfoGraphic1 = new QGroupBox(tr("Editing database images"));
  groupBoxInfoGraphic1 -> setLayout(layoutinfoGraphic1);
  groupBoxInfoGraphic1 -> setStyle(new QWindowsStyle);
  //______________________________________________________

}

void GUI_FACE_RECOGNIZER::setLabelInfoGraphic2() {

  //_____________Info graphic 2________________________
  buttonbackImageTest = new QPushButton(QIcon(QPixmap("./application_images/Buttons/back.png")), "");
  connect(buttonbackImageTest, SIGNAL(clicked()), this, SLOT(seeBackImageTest()));
  buttonbackImageTest -> setIconSize(QSize(30, 25));
  buttonbackImageTest -> setFixedWidth(70);
  buttonbackImageTest -> setAutoDefault(false);
  buttonbackImageTest -> setEnabled(false);
  buttonnextImageTest = new QPushButton(QIcon(QPixmap("./application_images/Buttons/next.png")), "");
  connect(buttonnextImageTest, SIGNAL(clicked()), this, SLOT(seeNextImageTest()));
  buttonnextImageTest -> setIconSize(QSize(30, 25));
  buttonnextImageTest -> setFixedWidth(70);
  buttonnextImageTest -> setAutoDefault(false);
  buttonnextImageTest -> setEnabled(false);

  //____________________________________________________//
  buttonBurstCaptureTest = new QPushButton(QIcon(QPixmap("./application_images/Buttons/BUTTON-REC.png")), ""); //Capture
  connect(buttonBurstCaptureTest, SIGNAL(clicked()), this, SLOT(captureImagesTest()));
  buttonBurstCaptureTest -> setAutoDefault(false);
  buttonBurstCaptureTest -> setFixedHeight(70);
  buttonBurstCaptureTest -> setFixedWidth(70);
  buttonBurstCaptureTest -> setFlat(false);
  buttonBurstCaptureTest -> setEnabled(false);

  QRegion * region = new QRegion( * (new QRect(buttonBurstCaptureTest -> x() + 5, buttonBurstCaptureTest -> y() + 5, 60, 60)), QRegion::Ellipse);
  buttonBurstCaptureTest -> setMask( * region);

  //____________________________________________________//

  QGridLayout * layoutinfoGraphic2 = new QGridLayout;
  layoutinfoGraphic2 -> addWidget(buttonbackImageTest, 0, 2, 1, 1);
  layoutinfoGraphic2 -> addWidget(buttonBurstCaptureTest, 0, 3, 1, 1);
  layoutinfoGraphic2 -> addWidget(buttonnextImageTest, 0, 4, 1, 1);

  groupBoxInfoGraphic2 = new QGroupBox(tr("Facial recognition test"));
  groupBoxInfoGraphic2 -> setLayout(layoutinfoGraphic2);
  groupBoxInfoGraphic2 -> setStyle(new QWindowsStyle);
  //______________________________________________________

}

void GUI_FACE_RECOGNIZER::setStackInfoGraphic() {

  infoGraphic = new QLabel;
  infoGraphic -> setStyle(new QWindowsStyle);
  infoGraphic -> setFrameShape(QFrame::StyledPanel);
  infoGraphic -> setFixedSize(widthLabelGraphic, 130);

  stackInfoGraphic = new QStackedWidget;
  stackInfoGraphic -> addWidget(groupBoxInfoGraphic1);
  stackInfoGraphic -> addWidget(groupBoxInfoGraphic2);

  layoutStackInfoGraphic = new QHBoxLayout;
  layoutStackInfoGraphic -> addWidget(stackInfoGraphic);
  infoGraphic -> setLayout(layoutStackInfoGraphic);

}

void GUI_FACE_RECOGNIZER::setTabWidgetConfig() {

  labelConfigDataBase = new QLabel;
  labelConfigDescriptor = new QLabel;
  labelConfigSparseSolution = new QLabel;
  labelTestFaceRecognizer = new QLabel;

  tabWidgetConfig = new QTabWidget;
  tabWidgetConfig -> addTab(labelConfigDataBase, "Database");
  tabWidgetConfig -> addTab(labelConfigDescriptor, "Descriptor");
  tabWidgetConfig -> addTab(labelConfigSparseSolution, "SRC");
  tabWidgetConfig -> addTab(labelTestFaceRecognizer, "Test");

  labelTestFaceRecognizer -> setFixedSize(400, 450);

}

void GUI_FACE_RECOGNIZER::showImage(QImage image) {
  graphicAndRubberBand -> setPixmap(QPixmap::fromImage(image.rgbSwapped()));
}

void GUI_FACE_RECOGNIZER::setAddDeleteUserConfig() {

  //_______________Add or remove user_____________________
  lineEditNameUser = new QLineEdit;
  connect(lineEditNameUser, SIGNAL(editingFinished()), this, SLOT(editingNameUserFinished()));
  connect(lineEditNameUser, SIGNAL(textChanged(const QString & )), this, SLOT(editingNameUserTextChanged()));

  buttonAddNewUser = new QPushButton("Add user");
  connect(buttonAddNewUser, SIGNAL(clicked()), this, SLOT(addUser()));
  buttonAddNewUser -> setAutoDefault(false);

  buttonDeleteUser = new QPushButton("Delete user");
  connect(buttonDeleteUser, SIGNAL(clicked()), this, SLOT(deleteUser()));
  buttonDeleteUser -> setAutoDefault(false);

  layoutAddDeleteUser = new QHBoxLayout;
  layoutAddDeleteUser -> addWidget(buttonAddNewUser);
  layoutAddDeleteUser -> addWidget(lineEditNameUser);
  layoutAddDeleteUser -> addWidget(buttonDeleteUser);

  //___________________________________________________________

}

void GUI_FACE_RECOGNIZER::setSourceImagesConfig() {

  //____________Decision about the source of the user's images__________
  checkBoxMirrorImage = new QCheckBox("Generate mirror image");
  connect(checkBoxMirrorImage, SIGNAL(stateChanged(int)), this, SLOT(setGenerateImageMirror(int)));
  checkBoxMirrorImage -> setEnabled(false);

  checkBoxCaptureCam = new QCheckBox("Camera");
  connect(checkBoxCaptureCam, SIGNAL(stateChanged(int)), this, SLOT(activeCaptureFromCam(int)));

  buttonAddImagesFile = new QPushButton("File");
  connect(buttonAddImagesFile, SIGNAL(clicked()), this, SLOT(loadImagesFromFile()));
  buttonAddImagesFile -> setAutoDefault(false);

  lineEditDevice = new QLineEdit("0");
  lineEditDevice -> setFixedWidth(40);

  buttonAddImageToUser = new QPushButton("Add image");
  connect(buttonAddImageToUser, SIGNAL(clicked()), this, SLOT(addImagenCaptured()));
  buttonAddImageToUser -> setAutoDefault(false);
  buttonAddImageToUser -> setEnabled(false);

  layoutConfigGetImages = new QGridLayout;

  layoutConfigGetImages -> addWidget(checkBoxMirrorImage, 0, 0, 1, 3, Qt::AlignLeft);
  layoutConfigGetImages -> addWidget(buttonAddImagesFile, 1, 0, 1, 1, Qt::AlignLeft);
  layoutConfigGetImages -> addWidget(checkBoxCaptureCam, 2, 0, 1, 1, Qt::AlignLeft);
  layoutConfigGetImages -> addWidget(new QLabel("Device"), 2, 2, 1, 1, Qt::AlignLeft);
  layoutConfigGetImages -> addWidget(lineEditDevice, 2, 3, 1, 1, Qt::AlignLeft);
  layoutConfigGetImages -> addWidget(buttonAddImageToUser, 2, 4, 1, 1, Qt::AlignLeft);

  groupBoxConfigGetImages = new QGroupBox(tr("Image Acquisition"));
  groupBoxConfigGetImages -> setLayout(layoutConfigGetImages);
  groupBoxConfigGetImages -> setStyle(new QWindowsStyle);

  //_____________________________________________________________________

}

void GUI_FACE_RECOGNIZER::setEditImagesConfig() {

  //___________________Edit images_________________________
  labelNumberImages = new QLabel;
  setNumberImagesInfo();
  {

    labelNumberImages -> setAutoFillBackground(true); // IMPORTANT!
    QPalette pal = labelNumberImages -> palette();
    pal.setColor(QPalette::Window, QColor(230, 230, 230));
    labelNumberImages -> setPalette(pal);

    QFont serifFont("Times", 12, QFont::Bold);
    labelNumberImages -> setFont(serifFont);
  }
  //____________________________________________________________//

  //_____________________________________________________________________//
  labelNumberCurrenImage = new QLabel;
  setNumberCurrentImageInfo();
  {

    labelNumberCurrenImage -> setAutoFillBackground(true); // IMPORTANT!
    QPalette pal = labelNumberCurrenImage -> palette();
    pal.setColor(QPalette::Window, QColor(230, 230, 230));
    labelNumberCurrenImage -> setPalette(pal);

    QFont serifFont("Times", 12, QFont::Bold);
    labelNumberCurrenImage -> setFont(serifFont);
  }
  //____________________________________________________________________//

  //____________________________________________________________________//
  labelCurrentWidthImage = new QLabel;
  setWidthCurrentImageInfo();
  {

    labelCurrentWidthImage -> setAutoFillBackground(true); // IMPORTANT!
    QPalette pal = labelCurrentWidthImage -> palette();
    pal.setColor(QPalette::Window, QColor(230, 230, 230));
    labelCurrentWidthImage -> setPalette(pal);

    QFont serifFont("Times", 12, QFont::Bold);
    labelCurrentWidthImage -> setFont(serifFont);
  }
  //____________________________________________________________________//

  //____________________________________________________________________//
  labelHighCurrentImage = new QLabel("Current height=");
  setHighCurrentImageInfo();
  {

    labelHighCurrentImage -> setAutoFillBackground(true); // IMPORTANT!
    QPalette pal = labelHighCurrentImage -> palette();
    pal.setColor(QPalette::Window, QColor(230, 230, 230));
    labelHighCurrentImage -> setPalette(pal);

    QFont serifFont("Times", 12, QFont::Bold);
    labelHighCurrentImage -> setFont(serifFont);
  }
  //____________________________________________________________________//

  lineEditWidthImages = new QLineEdit;
  lineEditWidthImages -> setFixedWidth(50);
  lineEditHighImages = new QLineEdit;
  lineEditHighImages -> setFixedWidth(50);

  buttonDeleteImage = new QPushButton("Delete image");
  connect(buttonDeleteImage, SIGNAL(clicked()), this, SLOT(deleteImage()));
  buttonDeleteImage -> setAutoDefault(false);
  buttonApplyImage = new QPushButton("Apply to image");
  connect(buttonApplyImage, SIGNAL(clicked()), this, SLOT(resizeImage()));
  buttonApplyImage -> setAutoDefault(false);
  buttonApplyUser = new QPushButton("Apply to user");
  connect(buttonApplyUser, SIGNAL(clicked()), this, SLOT(resizeAllImage()));
  buttonApplyUser -> setAutoDefault(false);

  buttonApplyDataBase = new QPushButton("Apply to database");
  connect(buttonApplyDataBase, SIGNAL(clicked()), this, SLOT(resizeAllDataBase()));
  buttonApplyDataBase -> setAutoDefault(false);

  layoutConfigResizeImages = new QGridLayout;
  layoutConfigResizeImages -> addWidget(buttonDeleteImage, 0, 0, 1, 1);
  layoutConfigResizeImages -> addWidget(labelNumberImages, 0, 1, 1, 1);
  layoutConfigResizeImages -> addWidget(labelNumberCurrenImage, 1, 0, 1, 2);
  layoutConfigResizeImages -> addWidget(labelCurrentWidthImage, 2, 0, 1, 1);
  layoutConfigResizeImages -> addWidget(labelHighCurrentImage, 2, 1, 1, 1);
  layoutConfigResizeImages -> addWidget(new QLabel("Rescale width"), 3, 0, 1, 1);
  layoutConfigResizeImages -> addWidget(lineEditWidthImages, 3, 1, 1, 1);
  layoutConfigResizeImages -> addWidget(new QLabel("Rescale height"), 4, 0, 1, 1);
  layoutConfigResizeImages -> addWidget(lineEditHighImages, 4, 1, 1, 1);
  layoutConfigResizeImages -> addWidget(buttonApplyImage, 5, 0, 1, 1);
  layoutConfigResizeImages -> addWidget(buttonApplyUser, 5, 1, 1, 1);
  layoutConfigResizeImages -> addWidget(buttonApplyDataBase, 6, 0, 1, 2);

  groupBoxConfigResizeImages = new QGroupBox(tr("Image Editing"));
  groupBoxConfigResizeImages -> setLayout(layoutConfigResizeImages);
  groupBoxConfigResizeImages -> setStyle(new QWindowsStyle);

  //___________________________________________________________________________________

}

void GUI_FACE_RECOGNIZER::setLayoutPrincipalDataBaseConfig() {

  //__________________________________________________________________________________//
  labelNumberUsers = new QLabel;

  labelStateDataBase = new QLabel;
  labelStateDataBase -> setFixedSize(20, 20);

  QHBoxLayout * layoutLabelNumberUsersAndLabelStateDataBase = new QHBoxLayout;
  layoutLabelNumberUsersAndLabelStateDataBase -> addWidget(labelNumberUsers);
  layoutLabelNumberUsersAndLabelStateDataBase -> addWidget(labelStateDataBase);

  setNumberUsersInfo();
  {

    labelNumberUsers -> setAutoFillBackground(true); // IMPORTANT!
    QPalette pal = labelNumberUsers -> palette();
    pal.setColor(QPalette::Window, QColor(230, 230, 230));
    labelNumberUsers -> setPalette(pal);

    QFont serifFont("Times", 12, QFont::Bold);
    labelNumberUsers -> setFont(serifFont);
  }

  //__________________________________________________________________________________//

  //___________Setting the main layout for database configuration___________
  layoutPrincipalConfigDataBase = new QGridLayout;
  layoutPrincipalConfigDataBase -> addLayout(layoutAddDeleteUser, 0, 0);

  layoutPrincipalConfigDataBase -> addLayout(layoutLabelNumberUsersAndLabelStateDataBase, 1, 0, 1, 2);
  layoutPrincipalConfigDataBase -> addWidget(groupBoxConfigGetImages, 2, 0);
  layoutPrincipalConfigDataBase -> addWidget(groupBoxConfigResizeImages, 3, 0, 2, 1);

  labelConfigDataBase -> setLayout(layoutPrincipalConfigDataBase);
  //____________________________________________________________________________________

}

void GUI_FACE_RECOGNIZER::setLayoutPrincipalDescriptorConfig() {

  widgetsDescriptors = recognizerFacial -> getWidgetsDescriptors();
  comboBoxDescriptors = new QComboBox;

  int numberDescriptors = widgetsDescriptors -> count();

  for (int i = 0; i < numberDescriptors; i++)
    comboBoxDescriptors -> addItem(recognizerFacial -> nameDescriptor(i));

  connect(comboBoxDescriptors, SIGNAL(currentIndexChanged(int)), this, SLOT(changeDescriptor(int)));

  //_________EDITING WARNING______________//
  labelStateDescriptor = new QLabel;
  labelStateDescriptor -> setFixedSize(20, 20);
  //_______________________________________//

  //_________PREVIOUS CONFIGURATION BUTTON_____//
  buttonPreviousConfiguration = new QPushButton(QString::fromUtf8("Previous\nconfiguration"));
  connect(buttonPreviousConfiguration, SIGNAL(clicked()), this, SLOT(previousConfiguration()));
  buttonPreviousConfiguration -> setAutoDefault(false);
  buttonPreviousConfiguration -> setEnabled(false);
  //___________________________________________//

  QHBoxLayout * layoutComboBoxDescriptors = new QHBoxLayout;
  layoutComboBoxDescriptors -> addWidget(comboBoxDescriptors);
  layoutComboBoxDescriptors -> addWidget(labelStateDescriptor);
  QGroupBox * groupBoxSelectDescriptor = new QGroupBox(QString::fromUtf8("Select a descriptor"));
  groupBoxSelectDescriptor -> setLayout(layoutComboBoxDescriptors);
  groupBoxSelectDescriptor -> setStyle(new QWindowsStyle);

  QVBoxLayout * layoutWidgetsDescriptors = new QVBoxLayout;
  layoutWidgetsDescriptors -> addWidget(widgetsDescriptors);
  QGroupBox * groupBoxWidgetsDescriptors = new QGroupBox(QString::fromUtf8("Descriptor configuration"));
  groupBoxWidgetsDescriptors -> setLayout(layoutWidgetsDescriptors);
  groupBoxWidgetsDescriptors -> setStyle(new QWindowsStyle);

  layoutConfigDescriptors = new QGridLayout;

  layoutConfigDescriptors -> addWidget(groupBoxSelectDescriptor, 0, 0, 1, 6);
  layoutConfigDescriptors -> addWidget(buttonPreviousConfiguration, 0, 6, 1, 4, Qt::AlignRight);
  layoutConfigDescriptors -> addWidget(groupBoxWidgetsDescriptors, 1, 0, 10, 10);

  labelConfigDescriptor -> setLayout(layoutConfigDescriptors);

  labelConfigDescriptor -> setEnabled(false);

}

void GUI_FACE_RECOGNIZER::setLayoutPrincipalSparseSolution() {

  lineEdit_lm = new QLineEdit;
  lineEdit_lm -> setFixedWidth(50);

  lineEdit_nc = new QLineEdit;
  lineEdit_nc -> setFixedWidth(50);

  lineEditNumberZeros = new QLineEdit;
  lineEditNumberZeros -> setFixedWidth(50);

  lineEdit_ck = new QLineEdit;
  lineEdit_ck -> setFixedWidth(50);

  lineEditRecognitionThreshold = new QLineEdit;
  lineEditRecognitionThreshold -> setFixedWidth(50);

  //CONNECTIONS
  connect(lineEdit_lm, SIGNAL(textChanged(const QString & )), this, SLOT(editedConfigSparseSolution()));
  connect(lineEdit_nc, SIGNAL(textChanged(const QString & )), this, SLOT(editedConfigSparseSolution()));
  connect(lineEditNumberZeros, SIGNAL(textChanged(const QString & )), this, SLOT(editedConfigSparseSolution()));
  connect(lineEdit_ck, SIGNAL(textChanged(const QString & )), this, SLOT(editedConfigSparseSolution()));
  connect(lineEditRecognitionThreshold, SIGNAL(textChanged(const QString & )), this, SLOT(editedConfigSparseSolution()));

  buttonApplyConfigDictionary = new QPushButton(QString::fromUtf8("Apply\nconfiguration"));
  connect(buttonApplyConfigDictionary, SIGNAL(clicked()), this, SLOT(ApplyConfigSparseSolution()));
  buttonApplyConfigDictionary -> setAutoDefault(false);
  buttonApplyConfigDictionary -> setEnabled(false);

  QFormLayout * formLayoutSparseSolution = new QFormLayout;
  formLayoutSparseSolution -> addRow(QString::fromUtf8("Fast filtering:"), lineEdit_lm);
  formLayoutSparseSolution -> addRow(QString::fromUtf8("Maximum number of descriptors:"), lineEdit_nc);
  formLayoutSparseSolution -> addRow(QString::fromUtf8("Maximum number of zeros:"), lineEditNumberZeros);
  formLayoutSparseSolution -> addRow(QString::fromUtf8("Shrink factor:"), lineEdit_ck);
  formLayoutSparseSolution -> addRow(QString::fromUtf8("Recognition threshold:"), lineEditRecognitionThreshold);
  formLayoutSparseSolution -> addRow("        ", buttonApplyConfigDictionary);

  QGroupBox * groupBoxSparseSolution = new QGroupBox(QString::fromUtf8("Sparse Solution Configuration"));
  groupBoxSparseSolution -> setLayout(formLayoutSparseSolution);
  groupBoxSparseSolution -> setStyle(new QWindowsStyle);

  QVBoxLayout * layoutPrincipalSparseSolution = new QVBoxLayout;
  layoutPrincipalSparseSolution -> addWidget(groupBoxSparseSolution);

  labelConfigSparseSolution -> setLayout(layoutPrincipalSparseSolution);

  readStateDictionary(); //Set the view for the dictionary configuration

}

void GUI_FACE_RECOGNIZER::setLayoutPrincipalTest() {

  labelShowRecognitionResult = new QLabel;
  {

    labelShowRecognitionResult -> setAutoFillBackground(true); // IMPORTANT!
    QPalette pal = labelShowRecognitionResult -> palette();
    pal.setColor(QPalette::Window, QColor(230, 230, 230));
    labelShowRecognitionResult -> setPalette(pal);

    QFont serifFont("Times", 12, QFont::Bold);
    labelShowRecognitionResult -> setFont(serifFont);

  }

  buttonLoadImageFromFileToRecognize = new QPushButton("File");
  connect(buttonLoadImageFromFileToRecognize, SIGNAL(clicked()), this, SLOT(loadImagesToTest()));
  buttonLoadImageFromFileToRecognize -> setAutoDefault(false);

  buttonGenerateTest = new QPushButton("Generate test");
  connect(buttonGenerateTest, SIGNAL(clicked()), this, SLOT(generateTest()));
  buttonGenerateTest -> setAutoDefault(false);

  lineEditDeviceTest = new QLineEdit;
  lineEditDeviceTest -> setFixedWidth(30);
  lineEditDeviceTest -> setText("0");

  buttonApplyConfigTest = new QPushButton(QString::fromUtf8("Apply configuration"));
  connect(buttonApplyConfigTest, SIGNAL(clicked()), this, SLOT(ApplyConfigTest()));
  buttonApplyConfigTest -> setAutoDefault(false);

  lineEditMaximumStackImages = new QLineEdit;
  connect(lineEditMaximumStackImages, SIGNAL(textChanged(const QString & )), this, SLOT(editedConfigTest()));
  lineEditMaximumStackImages -> setFixedWidth(50);
  lineEditWidthTestImage = new QLineEdit;
  connect(lineEditWidthTestImage, SIGNAL(textChanged(const QString & )), this, SLOT(editedConfigTest()));
  lineEditWidthTestImage -> setFixedWidth(50);
  lineEditHighTestImage = new QLineEdit;
  connect(lineEditHighTestImage, SIGNAL(textChanged(const QString & )), this, SLOT(editedConfigTest()));
  lineEditHighTestImage -> setFixedWidth(50);

  buttonGraphicDispersedSolution = new QPushButton("Plot sparse solutions");
  connect(buttonGraphicDispersedSolution, SIGNAL(clicked()), this, SLOT(resultsSparseSolution()));
  buttonGraphicDispersedSolution -> setAutoDefault(false);
  buttonGraphicDispersedSolution -> setEnabled(false);

  QGridLayout * layoutResizeImagesTest = new QGridLayout;
  layoutResizeImagesTest -> addWidget(new QLabel("Rescale width"), 0, 0, 1, 1);
  layoutResizeImagesTest -> addWidget(lineEditWidthTestImage, 0, 1, 1, 1);
  layoutResizeImagesTest -> addWidget(new QLabel("Rescale height"), 0, 2, 1, 1);
  layoutResizeImagesTest -> addWidget(lineEditHighTestImage, 0, 3, 1, 1);
  //layoutResizeImagesTest->addWidget(new QLabel(QString::fromUtf8("Stack length")),1,0,1,1);//Not implemented
  //layoutResizeImagesTest->addWidget(lineEditMaximumStackImages,1,1,1,1);//Not implemented
  layoutResizeImagesTest -> addWidget(buttonApplyConfigTest, 1, 2, 1, 2);

  QGroupBox * groupBoxResizeImagesTest = new QGroupBox(QString::fromUtf8("Extra configuration"));
  groupBoxResizeImagesTest -> setLayout(layoutResizeImagesTest);
  groupBoxResizeImagesTest -> setStyle(new QWindowsStyle);

  QHBoxLayout * layoutButtonsLoadImages = new QHBoxLayout;
  layoutButtonsLoadImages -> addWidget(buttonLoadImageFromFileToRecognize);
  layoutButtonsLoadImages -> addWidget(new QLabel("Device"));
  layoutButtonsLoadImages -> addWidget(lineEditDeviceTest);
  layoutButtonsLoadImages -> addWidget(buttonGenerateTest);
  QGroupBox * groupBoxTestImagesFiles = new QGroupBox(QString::fromUtf8("Image acquisition"));
  groupBoxTestImagesFiles -> setLayout(layoutButtonsLoadImages);
  groupBoxTestImagesFiles -> setStyle(new QWindowsStyle);

  QHBoxLayout * layoutShowRecognitionResult = new QHBoxLayout;
  layoutShowRecognitionResult -> addWidget(labelShowRecognitionResult);
  QGroupBox * groupBoxShowRecognitionResult = new QGroupBox(QString::fromUtf8("Results per image"));
  groupBoxShowRecognitionResult -> setLayout(layoutShowRecognitionResult);
  groupBoxShowRecognitionResult -> setStyle(new QWindowsStyle);

  QGridLayout * layoutPrincipalConfigTest = new QGridLayout;
  layoutPrincipalConfigTest -> addWidget(groupBoxResizeImagesTest, 0, 0, 1, 4);
  layoutPrincipalConfigTest -> addWidget(groupBoxTestImagesFiles, 1, 0, 1, 4);
  layoutPrincipalConfigTest -> addWidget(groupBoxShowRecognitionResult, 2, 0, 4, 4);
  layoutPrincipalConfigTest -> addWidget(buttonGraphicDispersedSolution, 6, 2, 1, 2);

  labelTestFaceRecognizer -> setLayout(layoutPrincipalConfigTest);
  labelTestFaceRecognizer -> setEnabled(false);

  /*
  labelShowRecognitionResult->setText(

  "SELECTED USER=NATALIA\nSCORE=0.32432\nCURRENT THRESHOLD=0.2\nRECOGNIZED USER=NATALIA\nRECOGNIZED IMAGE IN=0.23 s\nNUMBER OF DESCRIPTORS=100\nDIMENSION OF DESCRIPTORS=120\nDESCRIPTOR NAME=GTP"

  );
  */

}

void GUI_FACE_RECOGNIZER::readStateDictionary() {

  if (recognizerFacial -> dictionaryIsBuilt()) {

    lineEdit_lm -> setText(QString::number(recognizerFacial -> get_lm()));
    lineEdit_nc -> setText(QString::number(recognizerFacial -> get_nc()));
    lineEditNumberZeros -> setText(QString::number(recognizerFacial -> get_numberZeros()));
    lineEdit_ck -> setText(QString::number(recognizerFacial -> get_ck()));
    lineEditRecognitionThreshold -> setText(QString::number(recognizerFacial -> get_threshold()));

    labelConfigSparseSolution -> setEnabled(true);
    emit recognizerIsReady(true); // Notifies that the recognizer is ready

  } else {

    lineEdit_lm -> clear();
    lineEdit_nc -> clear();
    lineEditNumberZeros -> clear();
    lineEdit_ck -> clear();
    lineEditRecognitionThreshold -> clear();

    labelConfigSparseSolution -> setEnabled(false);
    emit recognizerIsReady(false); // Notifies that the recognizer is NOT ready

  }

  buttonApplyConfigDictionary -> setEnabled(false); // Since no editing has been done

}

void GUI_FACE_RECOGNIZER::editedConfigSparseSolution() {

  buttonApplyConfigDictionary -> setEnabled(true); // Enables it so the new configuration can be applied

}

void GUI_FACE_RECOGNIZER::ApplyConfigSparseSolution() {

  buttonApplyConfigDictionary -> setEnabled(false); // Disables it since the configuration has been applied

  // Extract the current values
  int m = recognizerFacial -> get_m(); // Not editable 
  int n = recognizerFacial -> get_n(); // Not editable 
  int lm = lineEdit_lm -> text().toInt();
  int nc = lineEdit_nc -> text().toInt();
  int numberDescriptors = recognizerFacial -> get_numberDescriptors(); // Not editable 
  int numberZeros = lineEditNumberZeros -> text().toInt();

  // Apply the values
  recognizerFacial -> set_lm(lm);
  recognizerFacial -> set_nc(nc);
  recognizerFacial -> set_numberZeros(numberZeros);

  // Store the values
  std::string nameFile = ((dataBase -> getPath()) + QString("/Dictionary.info")).toStdString();
  FILE * fileDataBase;
  if ((fileDataBase = fopen(nameFile.c_str(), "rb+")) == NULL) return;

  int parameters[6];
  parameters[0] = m;
  parameters[1] = n;
  parameters[2] = lm;
  parameters[3] = nc;
  parameters[4] = numberDescriptors;
  parameters[5] = numberZeros;

  fseek(fileDataBase, 0, SEEK_SET);
  fwrite(parameters, sizeof(int), 6, fileDataBase);

  float ck = lineEdit_ck -> text().toFloat();
  recognizerFacial -> set_ck(ck);
  fwrite( & ck, sizeof(float), 1, fileDataBase);

  fclose(fileDataBase);

  // Store the threshold
  float threshold = lineEditRecognitionThreshold -> text().toFloat();
  recognizerFacial -> saveThreshold(threshold);

}

void GUI_FACE_RECOGNIZER::setConfigurationTest() {

  if (recognizerFacial -> dictionaryIsBuilt()) {

    lineEditMaximumStackImages -> setText(QString::number(recognizerFacial -> get_lengthStackImages()));
    lineEditWidthTestImage -> setText(QString::number(recognizerFacial -> get_newWidthImages()));
    lineEditHighTestImage -> setText(QString::number(recognizerFacial -> get_newHighImages()));
    lineEditDeviceTest -> setText("0");

    labelTestFaceRecognizer -> setEnabled(true);
    buttonBurstCaptureTest -> setEnabled(true);

    setTestResultImageInfo("");
  } else {

    lineEditMaximumStackImages -> clear();
    lineEditWidthTestImage -> clear();
    lineEditHighTestImage -> clear();

    labelTestFaceRecognizer -> setEnabled(false);

    buttonbackImageTest -> setEnabled(false);
    buttonnextImageTest -> setEnabled(false);
    buttonBurstCaptureTest -> setEnabled(false);

    imageEditingTest = cv::Mat(); // Set an empty image by default
    if (stackInfoGraphic -> currentIndex() == 1) // Clear the displayed graph if the current image corresponds to the test phase
      clearLabelGraphic();

    setTestResultImageInfo("");

  }

  buttonApplyConfigTest -> setEnabled(false);

}

void GUI_FACE_RECOGNIZER::editedConfigTest() {
  buttonApplyConfigTest -> setEnabled(true);
}

void GUI_FACE_RECOGNIZER::ApplyConfigTest() {

  buttonApplyConfigTest -> setEnabled(false);

  int newWidthImages = lineEditWidthTestImage -> text().toInt();
  int newHighImages = lineEditHighTestImage -> text().toInt();
  int lengthStackImages = lineEditMaximumStackImages -> text().toInt();

  recognizerFacial -> setConfigurationTest(newWidthImages, newHighImages, lengthStackImages);

}

void GUI_FACE_RECOGNIZER::setNumberUsersInfo(int i) {

  QString tempText;
  if (i == -1)
    tempText = QString::fromUtf8("Number of users=");
  else
    tempText = QString::fromUtf8("Number of users=") + QString("<FONT COLOR=red>") + QString::number(i) + QString("</FONT>");

  labelNumberUsers -> setText(tempText);

}

void GUI_FACE_RECOGNIZER::setNumberImagesInfo(int i) {

  QString tempText;
  if (i == -1)
    tempText = QString::fromUtf8("No. of Images=");
  else
    tempText = QString::fromUtf8("No. of Images=") + QString("<FONT COLOR=red>") + QString::number(i) + QString("</FONT>");

  labelNumberImages -> setText(tempText);

}

void GUI_FACE_RECOGNIZER::setNumberCurrentImageInfo(int i) {

  QString tempText;
  if (i == -1)
    tempText = QString::fromUtf8("Current image being edited number=");
  else
    tempText = QString::fromUtf8("Current image being edited number=") + QString("<FONT COLOR=red>") + QString::number(i) + QString("</FONT>");

  labelNumberCurrenImage -> setText(tempText);

}

void GUI_FACE_RECOGNIZER::setWidthCurrentImageInfo(int i) {

  QString tempText;
  if (i == -1)
    tempText = QString::fromUtf8("Current width=");
  else
    tempText = QString::fromUtf8("Current width=") + QString("<FONT COLOR=red>") + QString::number(i) + QString("</FONT>");

  labelCurrentWidthImage -> setText(tempText);

}

void GUI_FACE_RECOGNIZER::setHighCurrentImageInfo(int i) {

  QString tempText;
  if (i == -1)
    tempText = QString::fromUtf8("Current height=");
  else
    tempText = QString::fromUtf8("Current height=") + QString("<FONT COLOR=red>") + QString::number(i) + QString("</FONT>");

  labelHighCurrentImage -> setText(tempText);

}

void GUI_FACE_RECOGNIZER::setEnabledPanelAddDeleteUserConfig(bool flagEnabled) {

  buttonAddNewUser -> setEnabled(flagEnabled);
  buttonDeleteUser -> setEnabled(flagEnabled);
  lineEditNameUser -> setEnabled(flagEnabled);

}

void GUI_FACE_RECOGNIZER::setEnabledPanelAcquisitionImages(bool flagEnabled) {

  buttonAddImagesFile -> setEnabled(flagEnabled);
  checkBoxMirrorImage -> setEnabled(flagEnabled);
  checkBoxCaptureCam -> setEnabled(flagEnabled);
  lineEditDevice -> setEnabled(flagEnabled);
  buttonAddImageToUser -> setEnabled(flagEnabled);

}

void GUI_FACE_RECOGNIZER::setEnabledPanelEditionImages(bool flagEnabled) {

  buttonDeleteImage -> setEnabled(flagEnabled);
  buttonApplyImage -> setEnabled(flagEnabled);
  buttonApplyUser -> setEnabled(flagEnabled);
  buttonApplyDataBase -> setEnabled(flagEnabled);

  lineEditWidthImages -> setEnabled(flagEnabled);
  lineEditHighImages -> setEnabled(flagEnabled);

}

void GUI_FACE_RECOGNIZER::setEnabledPanelConfigDataBase(bool flagEnabled) {

  setEnabledPanelAddDeleteUserConfig(flagEnabled);
  setEnabledPanelAcquisitionImages(flagEnabled);
  setEnabledPanelEditionImages(flagEnabled);

}

void GUI_FACE_RECOGNIZER::editedDatabase(bool flag) {

  if (flag == true) {
    labelStateDataBase -> setPixmap(QPixmap("./application_images/Labels/modified_icon.png").scaled(labelStateDataBase -> width(), labelStateDataBase -> height(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
  } else {
    labelStateDataBase -> setPixmap(QPixmap("./application_images/Labels/ok.png").scaled(labelStateDataBase -> width(), labelStateDataBase -> height(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
  }

  if (flag)
    buttonApplyTotalConfig -> setEnabled(true);
  else
    buttonApplyTotalConfig -> setEnabled(false);

}

void GUI_FACE_RECOGNIZER::editingNameUserFinished() {

  lineEditNameUser -> clearFocus(); // Clears the focus

  if (lineEditNameUser -> text() == "") return;

  if (dataBase -> existUser(lineEditNameUser -> text())) // If the user exists, enable adding images
  {

    checkBoxMirrorImage -> setEnabled(true);
    checkBoxCaptureCam -> setEnabled(true);
    buttonAddImagesFile -> setEnabled(true);

    // Here we attach the user to be worked on
    currentUser = dataBase -> getUser(lineEditNameUser -> text());

    //________________If the user has images, show the first one on screen__________________
    currentUser -> resetIndexCurrentImageEditing(); // Indicating that we want to see the first image
    currentUser -> getImageEditingNext(imageEditing);

    if (imageEditing.empty()) {
      clearLabelGraphic();
      setNumberImagesInfo(0);
    } else {
      graphicAndRubberBand -> setPixmap(imageEditing);

      buttonbackImageDataBase -> setEnabled(true);
      buttonnextImageDataBase -> setEnabled(true);
      setEnabledPanelEditionImages(true);

      setNumberImagesInfo(currentUser -> getNumberImages());
      setNumberCurrentImageInfo(1);
      setWidthCurrentImageInfo(imageEditing.cols);
      setHighCurrentImageInfo(imageEditing.rows);
    }

    //_______________________________________________________________________________________________

  } else {

    currentUser = NULL;

    checkBoxMirrorImage -> setEnabled(false);
    checkBoxCaptureCam -> setEnabled(false);
    buttonAddImagesFile -> setEnabled(false);

    setEnabledPanelEditionImages(false); // Disables editing

    buttonbackImageDataBase -> setEnabled(false);
    buttonnextImageDataBase -> setEnabled(false);

    setNumberImagesInfo();
    setNumberCurrentImageInfo();
    setWidthCurrentImageInfo();
    setHighCurrentImageInfo();

    graphicAndRubberBand -> clearImageToCrop();
  }

}

void GUI_FACE_RECOGNIZER::editingNameUserTextChanged() {

  currentUser = NULL;

  checkBoxMirrorImage -> setEnabled(false);
  checkBoxCaptureCam -> setCheckState(Qt::Unchecked);
  checkBoxCaptureCam -> setEnabled(false);

  setEnabledPanelAcquisitionImages(false);
  setEnabledPanelEditionImages(false); // Deactivates editing

  // Deactivates the view
  buttonbackImageDataBase -> setEnabled(false);
  buttonnextImageDataBase -> setEnabled(false);

  clearLabelGraphic();

  setNumberImagesInfo();
  setNumberCurrentImageInfo();
  setWidthCurrentImageInfo();
  setHighCurrentImageInfo();

  graphicAndRubberBand -> clearImageToCrop();
}

void GUI_FACE_RECOGNIZER::captureImages() {

  static bool flagApplyTotalConfig;

  if (videoThread.isRunning()) {

    videoThread.stop();
    imageEditing = videoThread.getFrame();
    graphicAndRubberBand -> pushImageToCrop( & imageEditing);
    buttonAddImageToUser -> setEnabled(true); // Enables the copy of the captured image

    tabWidgetConfig -> setEnabled(true); // Replaces the commented lines in this scope

    buttonLoadDataBase -> setEnabled(true);
    if (flagApplyTotalConfig) buttonApplyTotalConfig -> setEnabled(true);

    if (currentUser -> getNumberImages() > 0) { // If the user has images, enables the image view
      buttonbackImageDataBase -> setEnabled(true);
      buttonnextImageDataBase -> setEnabled(true);
    }

  } else {

    /*
    tabWidgetConfig->setEnabled(false);

    buttonLoadDataBase->setEnabled(false);
    flagApplyTotalConfig=buttonApplyTotalConfig->isEnabled();
    buttonApplyTotalConfig->setEnabled(false);


    buttonbackImageDataBase->setEnabled(false);
    buttonnextImageDataBase->setEnabled(false);
    */

    int device = 0;
    QString strDevice = lineEditDevice -> text();
    if (strDevice != "") // Default is camera 0
      device = strDevice.toInt();

    if (!videoThread.setDevice(device)) {
      QMessageBox::warning(this, QString::fromUtf8("INFORMATION"), QString::fromUtf8("The video device is not available"), QMessageBox::Ok);
      return;
    }

    videoThread.start();
  }

}

void GUI_FACE_RECOGNIZER::setDescriptor() {

  int indexDescriptor = comboBoxDescriptors -> findText(previousDescriptorName); // Retrieves the index
  if ((previousDescriptorName != "") && (indexDescriptor != -1)) { // If such file exists, set the corresponding descriptor with its configuration

    comboBoxDescriptors -> setCurrentIndex(indexDescriptor); // Also calls void changeDescriptor(int index)
    recognizerFacial -> loadSettings(); // Loads the previous settings of the currently selected descriptor

    flagDescriptorEditing = false; // No editing has been done yet 
    flagChangeDescriptor = false; // No descriptor has been changed yet
    labelConfigDescriptor -> setEnabled(true); // Enables the choice of a descriptor
    buttonPreviousConfiguration -> setEnabled(false);
    editedDescriptor(false); // Inform that the descriptor configuration has not been edited

  } else {
    // The initial descriptor will be the first in the list
    previousDescriptorName = "";
    comboBoxDescriptors -> setCurrentIndex(0); // Default descriptor is set
    flagDescriptorEditing = true; // Inform that editing has occurred since descriptors need to be constructed
    flagChangeDescriptor = false; // Conceptually, no descriptor change has happened
    labelConfigDescriptor -> setEnabled(true); // Enables the choice of a descriptor
    editedDescriptor(true); // Inform that the descriptor configuration has been edited (which means it has not been applied)
  }

}

void GUI_FACE_RECOGNIZER::changeDescriptor(int index) {

  if ((previousDescriptorName != "") && (!flagDescriptorEditing)) { // Controls editing states

    if (previousDescriptorName == comboBoxDescriptors -> itemText(index)) {
      editedDescriptor(false);
      buttonPreviousConfiguration -> setEnabled(false);
      flagChangeDescriptor = false;

    } else {
      editedDescriptor(true);
      buttonPreviousConfiguration -> setEnabled(true);
      flagChangeDescriptor = true;

    }

  }

  widgetsDescriptors -> setCurrentIndex(index);

}

void GUI_FACE_RECOGNIZER::editedDescriptor(bool flag) {

  if (flag == true) {
    labelStateDescriptor -> setPixmap(QPixmap("./application_images/Labels/modified_icon.png").scaled(labelStateDescriptor -> width(), labelStateDescriptor -> height(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
  } else {
    labelStateDescriptor -> setPixmap(QPixmap("./application_images/Labels/ok.png").scaled(labelStateDescriptor -> width(), labelStateDescriptor -> height(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
  }

  if (!(dataBase -> dataBaseIsEdited())) {
    if (flag == true)
      buttonApplyTotalConfig -> setEnabled(true);
    else
      buttonApplyTotalConfig -> setEnabled(false);
  }

}

void GUI_FACE_RECOGNIZER::editingDescriptor() {

  qDebug() << "EDIT REQUEST\n";
  qDebug() << "comboBoxDescriptors->currentText()=" << comboBoxDescriptors -> currentText() << "  previousDescriptorName=" << previousDescriptorName << "\n";

  qDebug() << "Comparison=" << (comboBoxDescriptors -> currentText() == previousDescriptorName) << "\n";

  if ((comboBoxDescriptors -> currentText() == previousDescriptorName)) { // Only considers editing of the currently selected descriptor
    flagDescriptorEditing = true;
    editedDescriptor(true);
    buttonPreviousConfiguration -> setEnabled(true);

  }

}

QString GUI_FACE_RECOGNIZER::getPreviousDescriptorName() {

  QString tempPreviousDescriptorName("");

  if (QFile::exists((dataBase -> getPath()) + "/nameDescriptor.txt")) { // If such a file exists, set the corresponding descriptor with its configuration
    // Read the descriptor name
    QFile fileTemp(((dataBase -> getPath()) + "/nameDescriptor.txt"));
    if (!fileTemp.open(QIODevice::ReadOnly | QIODevice::Text))
      return tempPreviousDescriptorName;
    QTextStream in ( & fileTemp);
    tempPreviousDescriptorName = in.readLine();
    fileTemp.close();
    //_______________________________________
  }

  return tempPreviousDescriptorName;

}

void GUI_FACE_RECOGNIZER::previousConfiguration() {

  int indexDescriptor = comboBoxDescriptors -> findText(previousDescriptorName); // Retrieves the index
  comboBoxDescriptors -> setCurrentIndex(indexDescriptor); // Also calls void changeDescriptor(int index)
  recognizerFacial -> loadSettings(); // Loads the previous settings of the currently selected descriptor

  flagDescriptorEditing = false; // No editing has been done yet 
  flagChangeDescriptor = false; // No descriptor has been changed yet
  labelConfigDescriptor -> setEnabled(true); // Enables the choice of a descriptor
  buttonPreviousConfiguration -> setEnabled(false);
  editedDescriptor(false); // Inform that the descriptor configuration has not been edited

}

void GUI_FACE_RECOGNIZER::saveNameCurrentDescriptor() {

  QFile fileTemp((dataBase -> getPath()) + "/nameDescriptor.txt"); // The file is created again
  if (!fileTemp.open(QIODevice::WriteOnly | QIODevice::Text))
    return;
  QTextStream out( & fileTemp);
  out << (recognizerFacial -> currentNameDescriptor()) << "\n";
  fileTemp.close();

}

void GUI_FACE_RECOGNIZER::setMyLayouts() {

  layoutGraphicPrincipal = new QVBoxLayout;
  layoutGraphicPrincipal -> addWidget(graphicAndRubberBand);
  layoutGraphicPrincipal -> addWidget(infoGraphic);

  QHBoxLayout * layoutPrincipal = new QHBoxLayout;
  layoutPrincipal -> addLayout(layoutGraphicPrincipal);

  //_____________________________________________________
  layoutStackAndButtonApplyTotalConfig = new QVBoxLayout;
  layoutStackAndButtonApplyTotalConfig -> addWidget(buttonLoadDataBase);
  layoutStackAndButtonApplyTotalConfig -> addWidget(tabWidgetConfig);
  layoutStackAndButtonApplyTotalConfig -> addWidget(buttonApplyTotalConfig);
  //_____________________________________________________

  layoutPrincipal -> addLayout(layoutStackAndButtonApplyTotalConfig);

  setLayout(layoutPrincipal);

}

void GUI_FACE_RECOGNIZER::clearLabelGraphic() {
  QImage black(widthLabelGraphic, highLabelGraphic, QImage::Format_RGB888);
  black.fill(QColor(0, 0, 0));
  graphicAndRubberBand -> setPixmap(QPixmap::fromImage((black)));
}

void GUI_FACE_RECOGNIZER::currentIndexTabWidget(int index) {

  qDebug() << "Actual index=" << index << "\n";

  if (index == 0) {

    graphicAndRubberBand -> clearImageToCrop();
    groupBoxInfoGraphic1 -> setEnabled(true);
    stackInfoGraphic -> setCurrentIndex(0);

    if (currentUser != NULL) {
      clearLabelGraphic();
      if (!imageEditing.empty())
        graphicAndRubberBand -> setPixmap(imageEditing); //NOTE here the handled image changes
    } else {
      clearLabelGraphic();
    }

  } else if (index == 3) {

    graphicAndRubberBand -> clearImageToCrop();
    groupBoxInfoGraphic2 -> setEnabled(true);
    stackInfoGraphic -> setCurrentIndex(1);
    clearLabelGraphic();
    if (!imageEditingTest.empty())
      graphicAndRubberBand -> setPixmap(imageEditingTest); //NOTE here the handled image changes

  } else {

    groupBoxInfoGraphic1 -> setEnabled(false);
    groupBoxInfoGraphic2 -> setEnabled(false);

  }

  // A static variable is created here because no better solution was found
  static bool flag1Connection = false;
  if ((index == 3) && (!flag1Connection)) {
    connect(graphicAndRubberBand, SIGNAL(imageIscrop()), this, SLOT(cropImageTest()));
    flag1Connection = true;
  } else if ((index == 0) && (flag1Connection)) {
    disconnect(graphicAndRubberBand, SIGNAL(imageIscrop()), this, SLOT(cropImageTest()));
    flag1Connection = false;
  }

  if (index != 0)
    buttonLoadDataBase -> setEnabled(false);
  else
    buttonLoadDataBase -> setEnabled(true);

}

void GUI_FACE_RECOGNIZER::message_errorDescriptors(const QString & infoText) {

  QMessageBox msgBox;
  msgBox.setText(infoText);
  msgBox.setWindowTitle("Error in descriptor calculation");
  msgBox.exec();

}

void GUI_FACE_RECOGNIZER::enabledProgressbar(bool flag) {

  if (flag)
    progressDialogDescriptorsCalculation -> setEnabled(true);
  else
    progressDialogDescriptorsCalculation -> setEnabled(false);

}

//slots corresponding to the database configuration
void GUI_FACE_RECOGNIZER::loadDataBase() {

  QString tempPathDir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
    QDir::homePath(),
    QFileDialog::ShowDirsOnly |
    QFileDialog::DontResolveSymlinks);

  if (tempPathDir == "") return;
  

  //________INITIAL GRAPHIC STATES____________
  //NOTE: These should be the initial states if a database is loaded again, having previously loaded one
  setEnabledPanelConfigDataBase(false);
  buttonbackImageDataBase -> setEnabled(false);
  buttonnextImageDataBase -> setEnabled(false);
  buttonBurstCapture -> setEnabled(false);
  lineEditDevice -> setText("0");
  lineEditNameUser -> clear();
  lineEditWidthImages -> clear();
  lineEditHighImages -> clear();
  checkBoxCaptureCam -> setCheckState(Qt::Unchecked);
  checkBoxMirrorImage -> setCheckState(Qt::Unchecked);
  //______________________________________________

  emit startLoadDataBase(tempPathDir); // The database and sparse solution are loaded

  setEnabled(false); // The entire interface is disabled while the database is loading

  buttonGraphicDispersedSolution -> setEnabled(false);
  plotSparseSolution -> close(); // If we reached here, we close plotSparseSolution

}

void GUI_FACE_RECOGNIZER::setCompleterNameUsers() {

  if (completer != NULL) {
    lineEditNameUser -> setCompleter(0); // The current QCompleter is removed
    delete completer; // The current QCompleter is deleted
    completer = NULL;
  }

  completer = new QCompleter(dataBase -> getListNameUsers(), this);
  completer -> setCaseSensitivity(Qt::CaseInsensitive);
  lineEditNameUser -> setCompleter(completer);

}

void GUI_FACE_RECOGNIZER::addUser() {

  QString tempNameUser = lineEditNameUser -> text();

  if (tempNameUser != "") {

    if (!dataBase -> createUser(tempNameUser)) {
      QMessageBox::warning(this, QString::fromUtf8("INFORMATION"), QString::fromUtf8("The username already exists, please enter another"), QMessageBox::Ok);
    } else {

      setCompleterNameUsers();

      setNumberUsersInfo(dataBase -> getNumberUsers());
      checkBoxMirrorImage -> setEnabled(true);
      checkBoxCaptureCam -> setEnabled(true);
      checkBoxCaptureCam -> setCheckState(Qt::Unchecked);
      buttonAddImagesFile -> setEnabled(true);

      // Here we set the user on which we will work
      currentUser = dataBase -> getUser(tempNameUser);
      clearLabelGraphic();

    }

  }

}

void GUI_FACE_RECOGNIZER::deleteUser() {

  QString tempNameUser = lineEditNameUser -> text();

  if (tempNameUser != "") {

    if (!dataBase -> deleteUser(tempNameUser)) {
      QMessageBox::warning(this, QString::fromUtf8("INFORMATION"), QString::fromUtf8("The username cannot be deleted as it does not exist in the database"), QMessageBox::Ok);
    } else {
      setCompleterNameUsers();

      setNumberUsersInfo(dataBase -> getNumberUsers());

      checkBoxMirrorImage -> setEnabled(false);
      checkBoxCaptureCam -> setCheckState(Qt::Unchecked);
      checkBoxCaptureCam -> setEnabled(false);
      buttonAddImagesFile -> setEnabled(false);

      buttonbackImageDataBase -> setEnabled(false);
      buttonnextImageDataBase -> setEnabled(false);

      // Here we make the pointer currentUser point to NULL since it should not point to any user
      currentUser = NULL;
      lineEditNameUser -> setText(""); // We clear QLineEdit

      setEnabledPanelEditionImages(false); // The editing is disabled

      clearLabelGraphic();

      // Update the user information
      setNumberUsersInfo(dataBase -> getNumberUsers());
      setNumberImagesInfo();
      setNumberCurrentImageInfo();
      setWidthCurrentImageInfo();
      setHighCurrentImageInfo();

    }

  }

}

void GUI_FACE_RECOGNIZER::activeCaptureFromCam(int state) {

  if (state == Qt::Checked) {

    buttonAddImagesFile -> setEnabled(false);
    lineEditDevice -> setEnabled(true);
    buttonBurstCapture -> setEnabled(true);

  } else if (state == Qt::Unchecked) {

    buttonAddImagesFile -> setEnabled(true);
    lineEditDevice -> setEnabled(false);
    buttonBurstCapture -> setEnabled(false);
    buttonAddImageToUser -> setEnabled(false);

  }

}

void GUI_FACE_RECOGNIZER::setGenerateImageMirror(int state) {

  if (state == Qt::Checked)
    flagGenerateImageMirror = true;
  else if (state == Qt::Unchecked)
    flagGenerateImageMirror = false;

}

void GUI_FACE_RECOGNIZER::addImagenCaptured() {

  if (currentUser == NULL) return;

  if (flagGenerateImageMirror == true) //If the mirror image option is enabled
  {

    std::vector < cv::Mat > imagesTemp;
    cv::Mat temp;
    cv::flip(imageEditing, temp, 1); //Mirror image
    imagesTemp.push_back(temp.clone());
    imagesTemp.push_back(imageEditing.clone());

    currentUser -> addImages(imagesTemp);

  } else
    currentUser -> addImages(imageEditing);

  //We use the buttonApplyUser as a flag
  if (!buttonApplyUser -> isEnabled())
    setEnabledPanelEditionImages(true);
  //Enable image viewing
  buttonbackImageDataBase -> setEnabled(true);
  buttonnextImageDataBase -> setEnabled(true);

  graphicAndRubberBand -> setPixmap(imageEditing); //Important to set the flag flagCrop of graphicAndRubberBand to false
  //Update the image information
  setNumberImagesInfo(currentUser -> getNumberImages());
  int index = currentUser -> getIndexCurrentImageEditing();
  setNumberCurrentImageInfo(index + 1); //We chose to start from 1
  setWidthCurrentImageInfo(imageEditing.cols);
  setHighCurrentImageInfo(imageEditing.rows);

}

void GUI_FACE_RECOGNIZER::loadImagesFromFile() {

  if (currentUser == NULL) return;

  //__________________________Searching for images________________________//
  QStringList listPathImages = QFileDialog::getOpenFileNames(this,
    "Select one or more files to open",
    LAST_PATH_IMAGE,
    "Images (*.png *.jpg *.jpeg *.pgm *.JPEG)");
  //_______________________________________________________________________//

  if (listPathImages.empty()) {
	  return;
  }else {
	  QFileInfo fileInfo(listPathImages[0]);
	  LAST_PATH_IMAGE = fileInfo.path();
  }

  std::vector < cv::Mat > images;

  if (flagGenerateImageMirror == true) //If the mirror image option is enabled
  {

    for (int i = 0; i < listPathImages.size(); i++) {

      cv::Mat temp = cv::imread(listPathImages[i].toStdString());
      if (temp.data) //Ensure the image was loaded
      {
        cv::Mat tempFlip;
        cv::flip(temp, tempFlip, 1); //Mirror image
        images.push_back(tempFlip.clone()); //Store mirror images
        images.push_back(temp.clone()); //Store original images
      }

    }

  } else {

    for (int i = 0; i < listPathImages.size(); i++) {

      cv::Mat temp = cv::imread(listPathImages[i].toStdString());
      if (temp.data) //Ensure the image was loaded
        images.push_back(temp.clone()); //Store images

    }

  }

  currentUser -> addImages(images); //Add the images to the user

  //Show the last image in the list
  imageEditing = images.back();
  graphicAndRubberBand -> setPixmap(imageEditing);

  //We use the buttonApplyUser as a flag
  if (!buttonApplyUser -> isEnabled())
    setEnabledPanelEditionImages(true);

  buttonbackImageDataBase -> setEnabled(true);
  buttonnextImageDataBase -> setEnabled(true);

  //Update the image information
  setNumberImagesInfo(currentUser -> getNumberImages());
  int index = currentUser -> getIndexCurrentImageEditing();
  setNumberCurrentImageInfo(index + 1); //We chose to start from 1
  setWidthCurrentImageInfo(imageEditing.cols);
  setHighCurrentImageInfo(imageEditing.rows);

}

void GUI_FACE_RECOGNIZER::insertExternalImages(std::vector < cv::Mat > listImages, QString nameUser) {

  if (listImages.empty()) {
    //Empty image list
    QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("The input image list is empty."), QMessageBox::Ok);
    return;
  }

  if (!dataBase -> databaseHasBeenLoaded()) {
    //The database has not been built or loaded yet
    QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("There is no database built or loaded yet."), QMessageBox::Ok);
    return;
  }

  if (!dataBase -> existUser(nameUser)) {
    //The user does not exist
    QMessageBox::warning(this, tr("WARNING"), QString::fromUtf8("The user you entered does not exist in the database."), QMessageBox::Ok);
    return;
  }

  USER * tempUser = dataBase -> getUser(nameUser);
  tempUser -> addImages(listImages); //Add the images to the user

  if (tempUser == currentUser) {
    //Show the last image in the list
    imageEditing = listImages.back();
    graphicAndRubberBand -> setPixmap(imageEditing);

    //We use the buttonApplyUser as a flag
    if (!buttonApplyUser -> isEnabled())
      setEnabledPanelEditionImages(true);

    buttonbackImageDataBase -> setEnabled(true);
    buttonnextImageDataBase -> setEnabled(true);

    //Update the image information
    setNumberImagesInfo(currentUser -> getNumberImages());
    int index = currentUser -> getIndexCurrentImageEditing();
    setNumberCurrentImageInfo(index + 1); //We chose to start from 1
    setWidthCurrentImageInfo(imageEditing.cols);
    setHighCurrentImageInfo(imageEditing.rows);
  }

}

void GUI_FACE_RECOGNIZER::seeNextImage() {

  if (buttonAddImageToUser -> isEnabled())
    buttonAddImageToUser -> setEnabled(false);

  //We use the buttonApplyUser as a flag
  if (!buttonApplyUser -> isEnabled())
    setEnabledPanelEditionImages(true);

  int index = currentUser -> getImageEditingNext(imageEditing);
  graphicAndRubberBand -> setPixmap(imageEditing);

  //Update the image information
  setNumberCurrentImageInfo(index + 1); //We chose to start from 1
  setWidthCurrentImageInfo(imageEditing.cols);
  setHighCurrentImageInfo(imageEditing.rows);

}

void GUI_FACE_RECOGNIZER::seeBackImage() {

  if (buttonAddImageToUser -> isEnabled())
    buttonAddImageToUser -> setEnabled(false);

  //We use the buttonApplyUser as a flag
  if (!buttonApplyUser -> isEnabled())
    setEnabledPanelEditionImages(true);

  int index = currentUser -> getImageEditingBack(imageEditing);
  graphicAndRubberBand -> setPixmap(imageEditing);

  //Update the image information
  setNumberCurrentImageInfo(index + 1); //We chose to start from 1
  setWidthCurrentImageInfo(imageEditing.cols);
  setHighCurrentImageInfo(imageEditing.rows);

}

//NOTE: The following function was not used
void resizeAndGausianBlur(cv::Mat & img, cv::Size newSize) {

  static double F = 0.97;
  static cv::Mat imgTemp;

  double factorWidth = double(newSize.width) / img.cols;
  double factorHeight = double(newSize.height) / img.rows;

  double sigmaX = std::sqrt((-8 * std::log(F)) / (factorWidth * factorWidth));
  double sigmaY = std::sqrt((-8 * std::log(F)) / (factorHeight * factorHeight));

  int kx = 6 * sigmaX;
  if (!(kx % 2)) kx++;
  int ky = 6 * sigmaY;
  if (!(ky % 2)) ky++;

  cv::GaussianBlur(img, imgTemp, cv::Size(kx, ky), sigmaX, sigmaY);
  cv::resize(imgTemp, img, cv::Size(), factorWidth, factorHeight, CV_INTER_AREA); //Image resizing

}

void GUI_FACE_RECOGNIZER::resizeImage() {

  if (buttonAddImageToUser -> isEnabled())
    buttonAddImageToUser -> setEnabled(false); /*If an edit is applied, the image addition must be canceled until a new capture is taken*/

  QString qstrWidth = lineEditWidthImages -> text();
  QString qstrHigh = lineEditHighImages -> text();

  if ((qstrWidth == "") || (qstrHigh == "") || (qstrWidth.toInt() == 0) || (qstrHigh.toInt() == 0)) { //Only cropping operation is applied

    if (graphicAndRubberBand -> imageIsCrop()) {

      int tempIndex = currentUser -> getIndexCurrentImageEditing();
      currentUser -> saveImageInIndex(imageEditing, tempIndex);

      graphicAndRubberBand -> setPixmap(imageEditing); //Update the image so that the flagCrop flag of graphicAndRubberBand is updated

      //Update image information
      setWidthCurrentImageInfo(imageEditing.cols);
      setHighCurrentImageInfo(imageEditing.rows);

    }

  } else { //Apply resizing and cropping

    int newWidth = qstrWidth.toInt();
    int newHigh = qstrHigh.toInt();

    if ((imageEditing.rows != newHigh) || (imageEditing.cols != newWidth)) { //Resizing and cropping

      cv::resize(imageEditing, imageEditing, cv::Size(newWidth, newHigh), 0, 0, CV_INTER_AREA); //Image resizing

      int tempIndex = currentUser -> getIndexCurrentImageEditing();
      currentUser -> saveImageInIndex(imageEditing, tempIndex);

      graphicAndRubberBand -> setPixmap(imageEditing); //Update the image so that the flagCrop flag of graphicAndRubberBand is updated

      //Update image information
      setWidthCurrentImageInfo(imageEditing.cols);
      setHighCurrentImageInfo(imageEditing.rows);

    }

  }

}

void GUI_FACE_RECOGNIZER::resizeAllImage() {

  if (buttonAddImageToUser -> isEnabled())
    buttonAddImageToUser -> setEnabled(false); /*If an edit is applied, the image addition must be canceled until a new capture is taken*/

  QString qstrWidth = lineEditWidthImages -> text();
  QString qstrHigh = lineEditHighImages -> text();

  if ((qstrWidth == "") || (qstrHigh == "") || (qstrWidth.toInt() == 0) || (qstrHigh.toInt() == 0))
    return; //DO NOTHING

  //Now check if the currently edited image has had any cropping applied
  int tempIndex = currentUser -> getIndexCurrentImageEditing();
  if (graphicAndRubberBand -> imageIsCrop())
    currentUser -> saveImageInIndex(imageEditing, tempIndex);

  //Next, resize all the user's images that differ from the established width and height

  int newWidth = qstrWidth.toInt();
  int newHigh = qstrHigh.toInt();
  cv::Mat tempImg;

  int tempNumberImages = currentUser -> getNumberImages();
  for (int i = 0; i < tempNumberImages; i++) {

    currentUser -> getImage(tempImg, i);
    if ((tempImg.rows != newHigh) || (tempImg.cols != newWidth)) {

      cv::resize(tempImg, tempImg, cv::Size(newWidth, newHigh), 0, 0, CV_INTER_AREA); //Image resizing
      currentUser -> saveImageInIndex(tempImg, i);

    }

  }

  currentUser -> getImage(imageEditing, tempIndex); //Load the current image back into editing
  graphicAndRubberBand -> setPixmap(imageEditing); //Update the image so that the flagCrop flag of graphicAndRubberBand is updated
  //Update image information
  setWidthCurrentImageInfo(imageEditing.cols);
  setHighCurrentImageInfo(imageEditing.rows);

}

void GUI_FACE_RECOGNIZER::resizeAllDataBase() {

  if (buttonAddImageToUser -> isEnabled())
    buttonAddImageToUser -> setEnabled(false); /*If an edit is applied, the image addition must be canceled until a new capture is taken*/

  QString qstrWidth = lineEditWidthImages -> text();
  QString qstrHigh = lineEditHighImages -> text();

  if ((qstrWidth == "") || (qstrHigh == "") || (qstrWidth.toInt() == 0) || (qstrHigh.toInt() == 0))
    return; //DO NOTHING

  //Now check if the currently edited image has had any cropping applied
  int tempIndex = currentUser -> getIndexCurrentImageEditing();
  if (graphicAndRubberBand -> imageIsCrop())
    currentUser -> saveImageInIndex(imageEditing, tempIndex);

  //Next, resize all the images of the user that differ from the established width and height

  int newWidth = qstrWidth.toInt();
  int newHigh = qstrHigh.toInt();
  cv::Mat tempImg;

  QList < USER * > tempUsers = dataBase -> getAllUsers();

  for (int j = 0; j < tempUsers.size(); j++) {

    int tempNumberImages = tempUsers[j] -> getNumberImages();
    for (int i = 0; i < tempNumberImages; i++) {

      tempUsers[j] -> getImage(tempImg, i);
      if ((tempImg.rows != newHigh) || (tempImg.cols != newWidth)) {

        cv::resize(tempImg, tempImg, cv::Size(newWidth, newHigh), 0, 0, CV_INTER_AREA); //Image resizing
        tempUsers[j] -> saveImageInIndex(tempImg, i);

      }

    }

  }

  currentUser -> getImage(imageEditing, tempIndex); //Load the current image back into editing
  graphicAndRubberBand -> setPixmap(imageEditing); //Update the image so that the flagCrop flag of graphicAndRubberBand is updated
  //Update image information
  setWidthCurrentImageInfo(imageEditing.cols);
  setHighCurrentImageInfo(imageEditing.rows);

}

void GUI_FACE_RECOGNIZER::deleteImage() {

  //Delete the image
  int tempIndex = currentUser -> getIndexCurrentImageEditing();
  currentUser -> deleteImage(tempIndex);

  //Show the immediately previous image
  tempIndex = currentUser -> getIndexCurrentImageEditing();
  currentUser -> getImage(imageEditing, tempIndex);

  if (!imageEditing.empty()) {
    graphicAndRubberBand -> setPixmap(imageEditing); //Update the image so that the flagCrop flag of graphicAndRubberBand is updated

    //Update image information
    setNumberImagesInfo(currentUser -> getNumberImages());
    setNumberCurrentImageInfo(tempIndex + 1); //Chose to start from 1
    setWidthCurrentImageInfo(imageEditing.cols);
    setHighCurrentImageInfo(imageEditing.rows);

  } else {

    setEnabledPanelEditionImages(false);
    buttonbackImageDataBase -> setEnabled(false);
    buttonnextImageDataBase -> setEnabled(false);

    setNumberImagesInfo(0);
    setNumberCurrentImageInfo(); //Chose to start from 1
    setWidthCurrentImageInfo();
    setHighCurrentImageInfo();

    clearLabelGraphic();

  }

}

void GUI_FACE_RECOGNIZER::applyTotalConfig() {

  bool tempSomeEditingDescriptor = false;

  if (flagChangeDescriptor || flagDescriptorEditing) { //The descriptor has changed

    if (!recognizerFacial -> applySettings()) //Apply the current settings corresponding to the current descriptor
      return;

    saveNameCurrentDescriptor(); //Save the current descriptor's name to disk
    previousDescriptorName = getPreviousDescriptorName(); //Get the current descriptor's name
    flagChangeDescriptor = false; //Reset this variable
    flagDescriptorEditing = false; //Reset this variable
    editedDescriptor(false); //Notify that a new configuration has been applied (no editing from the descriptor)
    buttonPreviousConfiguration -> setEnabled(false);
    dataBase -> resetListFlagCalculatedDescriptor(); //Set all calculated descriptor flags to false for each user
    tempSomeEditingDescriptor = true;
  }

  if ((tempSomeEditingDescriptor) || (dataBase -> dataBaseIsEdited())) { //Then descriptor calculation is performed

    emit startCalculateDescriptors();
    setEnabled(false); //Disable the interface

    //______________Here we prepare progressDialogDescriptorsCalculation for the process____________________________
    int tempUserNumbers = dataBase -> getNumberUsers();
    progressDialogDescriptorsCalculation -> setMaximum(tempUserNumbers + 3);
    /*This is just an estimated time
    that may not always match the reality, but it works well in most cases. The time will be distributed as follows:
    tempUserNumbers steps for descriptor calculation, 1 for postprocessing, 1 for establishing the sparse solution, and 1 for sending the completion message*/
    progressDialogDescriptorsCalculation -> setEnabled(true); //This is to prevent it from being canceled due to the previous line
    progressDialogDescriptorsCalculation -> show();
    //_______________________________________________________________________________________________________________

  }

  buttonGraphicDispersedSolution -> setEnabled(false);
  plotSparseSolution -> close(); //If we reach here, then plotSparseSolution is closed

}

void GUI_FACE_RECOGNIZER::loadedDatabase() {

  setEnabled(true); //Re-enable the interface

  //___________CORRESPONDING TO THE DATABASE CONFIGURATION___________________//
  setCompleterNameUsers();
  setNumberUsersInfo(dataBase -> getNumberUsers());

  //enable-disable panels
  setEnabledPanelAddDeleteUserConfig(true);

  //___________________________________________________________________________________//

  //___________CORRESPONDING TO THE DESCRIPTOR CONFIGURATION______________________//

  previousDescriptorName = getPreviousDescriptorName(); //Reads the descriptor name from disk for the database
  setDescriptor();
  /*Sets the descriptor with its current configuration. If such descriptor doesn't exist in the system,
  descriptor zero will be loaded by default*/

  //_____________________________________________________________________________________//

  //___________CORRESPONDING TO THE SPARSE SOLUTION______________________//
  /*NOTE: If we reach this point, myDictionary is different from NULL if there was a copy on the hard drive, or NULL if there was no such copy. Regardless, the function readStateDictionary() sets the correct states on the configuration panel labelConfigSparseSolution based on whether myDictionary is NULL or not*/
  readStateDictionary();
  setConfigurationTest();
  //_______________________________________________________________________//

}

void GUI_FACE_RECOGNIZER::calculationEndDescriptors(bool flag) {

  setEnabled(true); //Re-enable the interface

  /*
  IMPORTANT NOTE: If the user presses cancel, some descriptors may not be calculated, postprocessing may not occur, or the sparse solution dictionary may not be set, which conceptually is an edit of the database. Therefore, the following sets the database to be edited if flag is false, regardless of whether it was edited before starting the descriptor calculation
  */

  if (flag)
    dataBase -> doneEditing(false);
  else
    dataBase -> doneEditing(true);

  /*Regardless of whether the dictionary was set (descriptor calculation finished successfully) or not (the user pressed cancel), the function readStateDictionary() will set the correct state based on whether the value of myDictionary is still NULL or not*/
  readStateDictionary();
  setConfigurationTest();

}

void GUI_FACE_RECOGNIZER::captureImagesTest() {

  static bool flagApplyTotalConfig;

  if (videoThread.isRunning()) {

    videoThread.stop();
    imageEditingTest = videoThread.getFrame();
    graphicAndRubberBand -> pushImageToCrop( & imageEditingTest);

    tabWidgetConfig -> setEnabled(true);

    if (flagApplyTotalConfig) buttonApplyTotalConfig -> setEnabled(true);

    emit startRecognizeFaceImage(imageEditingTest);

  } else {

    int device = 0;
    QString strDevice = lineEditDeviceTest -> text();
    if (strDevice != "") //By default, camera 0 is selected
      device = strDevice.toInt();

    if (!videoThread.setDevice(device)) {
      QMessageBox::warning(this, QString::fromUtf8("INFORMATION"), QString::fromUtf8("The video device is not available"), QMessageBox::Ok);
      return;
    }

    tabWidgetConfig -> setEnabled(false);

    flagApplyTotalConfig = buttonApplyTotalConfig -> isEnabled();
    buttonApplyTotalConfig -> setEnabled(false);

    videoThread.start();
  }

}

void GUI_FACE_RECOGNIZER::cropImageTest() {

  qDebug() << "Test image cropped\n";

  emit startRecognizeFaceImage(imageEditingTest);

}

void GUI_FACE_RECOGNIZER::loadImagesToTest() {

  listFileNameImagesTest.clear(); //Clear the previous list
  listFileNameImagesTest = QFileDialog::getOpenFileNames(this, tr("Open File"),
    LAST_PATH_IMAGE_TEST,
    tr("Images (*.png *.pgm *.jpg *.jpeg *.JPEG)"));

  indexImagesTest = -1;
  if (!listFileNameImagesTest.empty()) {
    QFileInfo fileInfo(listFileNameImagesTest[0]);
    LAST_PATH_IMAGE_TEST = fileInfo.path();  
    buttonbackImageTest -> setEnabled(true);
    buttonnextImageTest -> setEnabled(true);
    seeNextImageTest();
  }

}

void GUI_FACE_RECOGNIZER::seeNextImageTest() {

  indexImagesTest++;
  if (indexImagesTest == listFileNameImagesTest.size()) indexImagesTest = 0;

  imageEditingTest = cv::imread(listFileNameImagesTest[indexImagesTest].toStdString());
  graphicAndRubberBand -> setPixmap(imageEditingTest);
  emit startRecognizeFaceImage(imageEditingTest);

}

void GUI_FACE_RECOGNIZER::seeBackImageTest() {

  indexImagesTest--;
  if (indexImagesTest == -1) indexImagesTest = listFileNameImagesTest.size() - 1;

  imageEditingTest = cv::imread(listFileNameImagesTest[indexImagesTest].toStdString());
  graphicAndRubberBand -> setPixmap(imageEditingTest);
  emit startRecognizeFaceImage(imageEditingTest);

}

void GUI_FACE_RECOGNIZER::setTestResultImageInfo(const QString & infoText) {

  labelShowRecognitionResult -> setText(infoText);

}

void GUI_FACE_RECOGNIZER::generateTest() {

  qDebug() << "Calling RECOGNIZER_FACIAL::generateTest()\n";
  flagGenerateTest = true;
  buttonGraphicDispersedSolution -> setEnabled(false);
  plotSparseSolution -> close(); //If we reach here, then plotSparseSolution is closed
  setEnabled(false);

  messageGeneratingTest.show();

  QMetaObject::invokeMethod(recognizerFacial, "generateTest", Qt::QueuedConnection);

}

void GUI_FACE_RECOGNIZER::testWasGenerated() {

  qDebug() << "Returning from RECOGNIZER_FACIAL::generateTest()\n";
  flagGenerateTest = false;
  setEnabled(true);

  messageGeneratingTest.hide();

}

void GUI_FACE_RECOGNIZER::setPlotSparseSolution() {

  buttonGraphicDispersedSolution -> setEnabled(true);

  if (plotSparseSolution -> isVisible()) {
    plotSparseSolution -> setNameUsersListAndId(recognizerFacial -> getNameUsersListAndId());
    plotSparseSolution -> setNameList(recognizerFacial -> getListNameUsers());
    plotSparseSolution -> setDictionary(recognizerFacial -> getDictionary());
    plotSparseSolution -> plotFirstResult(); //To show the result of the first descriptor
  } else {
    plotSparseSolution -> setNameUsersListAndId(recognizerFacial -> getNameUsersListAndId());
    plotSparseSolution -> setNameList(recognizerFacial -> getListNameUsers());
    plotSparseSolution -> setDictionary(recognizerFacial -> getDictionary());
  }

}

void GUI_FACE_RECOGNIZER::resultsSparseSolution() {
  plotSparseSolution -> plotFirstResult(); //To show the result of the first descriptor
  plotSparseSolution -> show();
}

void GUI_FACE_RECOGNIZER::closeEvent(QCloseEvent * event) {

  if ((flagGenerateTest) || (videoThread.isRunning()) || (progressDialogDescriptorsCalculation -> isVisible())) {
    event -> ignore();
  } else {
    plotSparseSolution -> close();
    event -> accept();
  }

}
