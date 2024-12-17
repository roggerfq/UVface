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

#include "recognizerFacial.h"
#include "dataBaseImages.h" // Database manager
#include "trackerWindows.h" // Window tracker
//openCV
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "descriptor.h" // Base descriptor from which all descriptors must derive
//_____________INCLUDE YOUR DESCRIPTORS HERE_______________________//
#include "gtp2.h"
//_________________________________________________________________//
#include "dictionary.h" // Sparse solution NOTE: It must be declared after the opencv library definitions
//QT
#include <QStackedWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDebug>
#include <QTime>

RECOGNIZER_FACIAL::RECOGNIZER_FACIAL() {

  //__________Initial values___________________//
  dataBase = new DATA_BASE;
  myDictionary = NULL;
  mf = NULL;
  descriptorsBase = NULL;
  descriptor_end = NULL;
  descriptor_out = new Eigen::MatrixXf;
  //______________________________________________//

  DESCRIPTOR_TEST1 * des1 = new DESCRIPTOR_TEST1(12, 10, 1);
  DESCRIPTOR_TEST2 * des2 = new DESCRIPTOR_TEST2(12, 10, 1);
  GTP * gtpDescriptor = new GTP;

  descriptors.push_back(des1);
  descriptors.push_back(des2);
  descriptors.push_back(gtpDescriptor);

  for (int i = 0; i < descriptors.size(); i++)
    connect(descriptors[i], SIGNAL(editingDescriptor()), this, SIGNAL(editingDescriptor()));

  widgetsDescriptors = new QStackedWidget;
  for (int i = 0; i < descriptors.size(); i++)
    widgetsDescriptors -> addWidget(descriptors[i]);

}

RECOGNIZER_FACIAL::~RECOGNIZER_FACIAL() {
  /*NOTE: The destructor of std::vector<ABSTRACT_DESCRIPTOR  *> descriptors is not called
     because ABSTRACT_DESCRIPTOR is derived from QDialog, 
     so QT calls the virtual destructors of the derived classes when deleting the widgets*/

  delete dataBase;

  if (descriptor_out != NULL) {
    delete descriptor_out;
    descriptor_out = NULL;
  }

  if (myDictionary != NULL) {
    delete myDictionary;
    myDictionary = NULL;
  }

}

DATA_BASE * RECOGNIZER_FACIAL::getDataBase() const {
  return dataBase;
}

QStackedWidget * RECOGNIZER_FACIAL::getWidgetsDescriptors() const {
  return widgetsDescriptors;
}

bool RECOGNIZER_FACIAL::dictionaryIsBuilt() const {

  if (myDictionary)
    return true;
  else
    return false;

}

DICTIONARY * RECOGNIZER_FACIAL::getDictionary() const {
  return myDictionary;
}

QMap < int, QString > RECOGNIZER_FACIAL::getNameUsersListAndId() const {
  return nameUsersListAndId;
}

QStringList RECOGNIZER_FACIAL::getListNameUsers() const {
  return nameUsersListAndId.values();
}

int RECOGNIZER_FACIAL::get_lm() const {
  return myDictionary -> get_lm();
}

int RECOGNIZER_FACIAL::get_nc() const {
  return myDictionary -> get_nc();
}

int RECOGNIZER_FACIAL::get_numberZeros() const {
  return myDictionary -> get_numberZeros();
}

float RECOGNIZER_FACIAL::get_ck() const {
  return myDictionary -> get_ck();
}

int RECOGNIZER_FACIAL::get_m() const {
  return myDictionary -> get_m();
}

int RECOGNIZER_FACIAL::get_n() const {
  return myDictionary -> get_n();
}

int RECOGNIZER_FACIAL::get_numberDescriptors() const {
  return myDictionary -> get_numberDescriptors();
}

float RECOGNIZER_FACIAL::get_threshold() {

  if (QFile::exists(dataBase -> getPath() + QString("/thresholdInfo"))) {
    QFile fileTemp(dataBase -> getPath() + QString("/thresholdInfo"));
    fileTemp.open(QIODevice::ReadOnly);
    fileTemp.read(reinterpret_cast < char * > ( & thresholdFaceRecognizer), sizeof(float));
    fileTemp.close();
  } else {
    thresholdFaceRecognizer = 0.2; // This value is set by default
  }

  return thresholdFaceRecognizer;
}

void RECOGNIZER_FACIAL::set_lm(int new_lm) {
  myDictionary -> set_lm(new_lm);
}

void RECOGNIZER_FACIAL::set_nc(int new_nc) {
  myDictionary -> set_nc(new_nc);
}

void RECOGNIZER_FACIAL::set_numberZeros(int num) {
  myDictionary -> set_numberZeros(num);
}

void RECOGNIZER_FACIAL::set_ck(float new_ck) {
  myDictionary -> set_ck(new_ck);
}

void RECOGNIZER_FACIAL::setConfigurationTest(int width, int high, int lengthStack) {

  newWidthImages = width;
  newHighImages = high;
  lengthStackImages = lengthStack;

  if ((newWidthImages <= 0) || (newHighImages <= 0))
    flagResizeImages = false;
  else
    flagResizeImages = true;

  //____________Store the test configuration__________________
  QFile fileTemp(dataBase -> getPath() + QString("/configTest.info"));
  fileTemp.open(QIODevice::WriteOnly);
  fileTemp.write(reinterpret_cast < char * > ( & newWidthImages), sizeof(int));
  fileTemp.write(reinterpret_cast < char * > ( & newHighImages), sizeof(int));
  fileTemp.write(reinterpret_cast < char * > ( & lengthStackImages), sizeof(int));
  fileTemp.close();
  //____________________________________________________________________

}

void RECOGNIZER_FACIAL::loadConfigTest() {

  if (QFile::exists(dataBase -> getPath() + QString("/configTest.info"))) {
    QFile fileTemp(dataBase -> getPath() + QString("/configTest.info"));
    fileTemp.open(QIODevice::ReadOnly);
    fileTemp.read(reinterpret_cast < char * > ( & newWidthImages), sizeof(int));
    fileTemp.read(reinterpret_cast < char * > ( & newHighImages), sizeof(int));
    fileTemp.read(reinterpret_cast < char * > ( & lengthStackImages), sizeof(int));
    fileTemp.close();

    if ((newWidthImages <= 0) || (newHighImages <= 0))
      flagResizeImages = false;
    else
      flagResizeImages = true;

  } else {

    // Means there is no default configuration set yet 
    lengthStackImages = -1;
    newWidthImages = -1;
    newHighImages = -1;
    flagResizeImages = false;
  }

}

int RECOGNIZER_FACIAL::get_newWidthImages() const {
  return newWidthImages;
}

int RECOGNIZER_FACIAL::get_newHighImages() const {
  return newHighImages;
}

int RECOGNIZER_FACIAL::get_lengthStackImages() const {
  return lengthStackImages;
}

QString RECOGNIZER_FACIAL::nameDescriptor(int index) const {
  return (descriptors[index]) -> nameDescriptor();
}

QString RECOGNIZER_FACIAL::currentNameDescriptor() const {
  int index = widgetsDescriptors -> currentIndex(); // Get the current index of the selected descriptor
  return (descriptors[index]) -> nameDescriptor();
}

void RECOGNIZER_FACIAL::loadSettings() {

  int index = widgetsDescriptors -> currentIndex(); // Get the current index of the selected descriptor
  p_descriptor = descriptors[index]; // Choose the descriptor during loading
  p_descriptor -> loadSettings(dataBase -> getPath());

}

bool RECOGNIZER_FACIAL::applySettings() {

  int index = widgetsDescriptors -> currentIndex(); // Get the current index of the selected descriptor
  return (descriptors[index]) -> applySettings(dataBase -> getPath());

}

void RECOGNIZER_FACIAL::saveInfoSparseSolution() { // Here we save the information of the sparse solution and the name-ID association

  // Here we serialize the names and IDs
  QFile fileTemp(((dataBase -> getPath()) + QString("/namesAndId.info")));
  fileTemp.open(QIODevice::WriteOnly);
  QDataStream out( & fileTemp);
  out << nameUsersListAndId;
  fileTemp.close();

  // Next, we save the sparse solution data
  myDictionary -> saveDataBase(((dataBase -> getPath()) + QString("/Dictionary.info")).toStdString());

}

void RECOGNIZER_FACIAL::saveThreshold(float threshold) {

  thresholdFaceRecognizer = threshold;

  //____Store a file with its comparison threshold for facial recognition_________
  QFile fileTemp(dataBase -> getPath() + QString("/thresholdInfo"));
  fileTemp.open(QIODevice::WriteOnly);
  fileTemp.write(reinterpret_cast < char * > ( & threshold), sizeof(float));
  fileTemp.close();
  //_______________________________________________________________________________________________

}

void RECOGNIZER_FACIAL::loadInfoSparseSolution() {

  if ((QFile::exists((dataBase -> getPath()) + QString("/namesAndId.info"))) && (QFile::exists((dataBase -> getPath()) + QString("/Dictionary.info")))) {

    nameUsersListAndId.clear(); //In case a database was previously loaded
    QFile fileTemp((dataBase -> getPath()) + QString("/namesAndId.info"));
    fileTemp.open(QIODevice::ReadOnly);
    QDataStream in ( & fileTemp);
    in >> nameUsersListAndId;
    fileTemp.close();

    if (myDictionary != NULL) {
      delete myDictionary;
      myDictionary = NULL;
    }

    myDictionary = new DICTIONARY(((dataBase -> getPath()) + QString("/Dictionary.info")).toStdString());

    //myDictionary->seeInfo();
    //qDebug()<<nameUsersListAndId<<"\n";

  } else { //In case another database is loaded after one has been loaded previously

    if (myDictionary != NULL) {
      delete myDictionary;
      myDictionary = NULL;
      nameUsersListAndId.clear();
    }

  }

  if (myDictionary != NULL)
    std::cout << "Number of descriptors in the database=" << myDictionary -> get_numberDescriptors() << "\n"; //Output the number of descriptors in the database

}

void RECOGNIZER_FACIAL::setSparseSolution() {

  cv::Mat * pDescriptorTemp = NULL;

  if (descriptor_end -> empty()) //Means there was no post-processing
  {
    qDebug() << "Descriptor detected without post-processing \n";
    pDescriptorTemp = descriptorsBase;
  } else {
    qDebug() << "Descriptor detected with post-processing \n";
    pDescriptorTemp = descriptor_end;
  }

  /*
  std::cout<<"Final descriptor total\n\n";
  std::cout<<(*pDescriptorTemp)<<"\n";
  */

  if (myDictionary != NULL) {
    delete myDictionary;
    myDictionary = NULL;
  }

  int m = pDescriptorTemp -> cols; //Inverted because the returned descriptors are also inverted
  int n = pDescriptorTemp -> rows; //Inverted because the returned descriptors are also inverted
  int nc = 500; //Default
  int lm = 100; //Default
  if (n < lm)
    lm = n; //Default

  myDictionary = new DICTIONARY(m, n, nc, lm);

  float * pf = pDescriptorTemp -> ptr < float > (0);

  if (mf != NULL) {
    delete mf;
    mf = NULL;
  }

  mf = new Eigen::Map < Eigen::MatrixXf > (pf, pDescriptorTemp -> cols, pDescriptorTemp -> rows);

  int rowAnt = 0, cl = 0;
  while (!ithRows.empty()) {

    ( * descriptor_out) = mf -> middleCols(rowAnt, ithRows[0] - rowAnt);
    rowAnt = ithRows[0];
    ithRows.erase(ithRows.begin());

    myDictionary -> eigenPush(( * descriptor_out), ithRowsId[cl]);
    cl++;

  }

  //myDictionary->seeInfo();

  delete mf;
  mf = NULL;

}

bool RECOGNIZER_FACIAL::calculateDescriptors() {

  QTime timeTestRecognition;
  timeTestRecognition.start();

  //_____________INITIALIZE SOME NECESSARY VARIABLES_____________//
  descriptorsBase = new cv::Mat;
  descriptor_end = new cv::Mat;
  ithRows.clear();
  ithRowsId.clear();

  //NEW: The following lines were added to avoid ambiguity when the user presses cancel
  //_____________________________________________//

  if (myDictionary != NULL) {
    QFile::remove(dataBase -> getPath() + QString("/thresholdInfo"));
    QFile::remove(dataBase -> getPath() + QString("/namesAndId.info"));
    QFile::remove(dataBase -> getPath() + QString("/Dictionary.info"));
    QFile::remove(dataBase -> getPath() + QString("/configTest.info"));

    delete myDictionary;
    myDictionary = NULL;
  }
  //______________________________________________//
  //______________________________________________________//

  int index = widgetsDescriptors -> currentIndex(); //Get the current index of the selected descriptor

  p_descriptor = descriptors[index]; //Interface for the currently selected descriptor

  QStringList listNameUsers = dataBase -> getListNameUsers();

  //________THREAD CONTROL_____________
  {
    QMutexLocker locker( & mutex);
    if (stopped) {
      stopped = false;
      return false; //Indicates that the calculation was stopped by the user
    }
  }
  //_______________________________________

  for (int i = 0; i < listNameUsers.size(); i++) {

    //________THREAD CONTROL_____________
    {
      QMutexLocker locker( & mutex);
      if (stopped) {
        stopped = false;
        return false; //Indicates that the calculation was stopped by the user
      }
    }
    //_______________________________________

    USER * tempUser = dataBase -> getUser(listNameUsers[i]); //Get the user
    if (tempUser -> calculeDescriptors(p_descriptor, descriptorsBase)) //Calculate the base descriptor and stack it
    { //Users without images return false and are not considered
      emit currentUserProcessing(i + 1);
      currentUserProcessingInfo("Calculating descriptors for " + tempUser -> getNameUser());

      ithRows.push_back(descriptorsBase -> rows); //Record the number of rows so far
      ithRowsId.push_back(tempUser -> getId()); //Record the corresponding id
    } else {

      //If this happens, it means there are no images or the existing images did not produce descriptors
      if (tempUser -> getNumberImages() == 0) {
        QString infoText = QString::fromUtf8("User ") + tempUser -> getNameUser() + QString::fromUtf8(" has no images");
        emit zeroDescriptorsOrZeroImages(infoText);
      } else {

        //______Here we look for the image causing the error______________________
        QList < bool > listFlagCalculatedDescriptor = tempUser -> getListFlagCalculatedDescriptor();
        int indexImageError;
        for (int i = 0; i < listFlagCalculatedDescriptor.size(); i++) {
          if (listFlagCalculatedDescriptor[i] == false) {
            indexImageError = i;
            break;
          }

        }
        //__________________________________________________________________________________

        //NOTE: Remember that the index in the presentation (view) was chosen to start from 1
        QString infoText = QString::fromUtf8("Image number ") + QString::number(indexImageError + 1) + QString::fromUtf8(" belonging to user ") + tempUser -> getNameUser() + QString::fromUtf8(" does not produce descriptors, please delete it");
        emit zeroDescriptorsOrZeroImages(infoText);
      }
      return false;

    }

  }

  emit enabledProgressbar(false); //Disable the cancel button of the QProgressDialog

  int tempUserNumbers = dataBase -> getNumberUsers();

  //_________Apply post-processing to the base descriptors__________________________________//
  emit currentUserProcessing(tempUserNumbers + 1);
  currentUserProcessingInfo(QString::fromUtf8("Applying post-processing to the descriptors"));
  p_descriptor -> post_processing(( * descriptorsBase), ( * descriptor_end), ithRows);
  //_____________________________________________________________________________________________________//

  //__________________Setting dictionary for the sparse solution algorithm___________________________
  emit currentUserProcessing(tempUserNumbers + 2);
  currentUserProcessingInfo(QString::fromUtf8("Setting dictionary for the sparse solution algorithm"));

  setSparseSolution(); //Configure the sparse solution
  nameUsersListAndId = dataBase -> getListNameUsersAndId();
  saveInfoSparseSolution(); //Store the sparse solution and nameUsersListAndId to disk
  saveThreshold(0.2); //Set the comparison threshold to 0.2 by default
  setConfigurationTest(100, 120, 15); //Default values

  emit currentUserProcessing(tempUserNumbers + 3);
  currentUserProcessingInfo(QString::fromUtf8("The process finished successfully"));
  //_____________________________________________________________________________________________________________

  //qDebug()<<"List\n";
  //qDebug()<<nameUsersListAndId<<"\n";

  //______FREE DYNAMIC MEMORY THAT IS NO LONGER NEEDED___________//
  delete descriptorsBase;
  delete descriptor_end;
  descriptorsBase = NULL;
  descriptor_end = NULL;

  //______________________________________________________________________//

  calculationEndDescriptors(true); //Emit the signal indicating that descriptors have been calculated correctly

  emit enabledProgressbar(true); //Re-enable the cancel button of the QProgressDialog

  qDebug() << "Time spent on training=" << timeTestRecognition.elapsed() << "\n";

  return true; //Indicates that the descriptor calculation finished successfully

}

void RECOGNIZER_FACIAL::startLoadDataBase(QString namePathDataBase) {

  pathDataBase = namePathDataBase;

  dataBase -> loadDataBase(pathDataBase);
  loadInfoSparseSolution();
  loadConfigTest();

  emit loadedDatabase(); //Notify that the database has been loaded

}

void RECOGNIZER_FACIAL::startCalculateDescriptors() {

  stopped = false;

  if (!calculateDescriptors()) //Calculate descriptors for the database
  { //HERE WE RESET SOME VARIABLES AND DELETE THE DYNAMIC MEMORY CREATED IN calculateDescriptors()

    if (!descriptorsBase) {
      delete descriptorsBase;
      descriptorsBase = NULL;
    }

    if (!descriptor_end) {
      delete descriptor_end;
      descriptor_end = NULL;
    }
    p_descriptor = NULL;

    /*The variables ithRows and ithRowsId are reset at the beginning of the calculateDescriptors() function*/
    int tempUserNumbers = dataBase -> getNumberUsers();
    emit currentUserProcessing(tempUserNumbers + 3);
    currentUserProcessingInfo(QString::fromUtf8("The process was cancelled by the user"));

    emit calculationEndDescriptors(false); /*Emit the signal indicating that descriptor calculation was cancelled*/

    qDebug() << "DELETING DYNAMIC MEMORY IN ERROR HANDLING OF calculateDescriptors() FUNCTION";
  }

}

void RECOGNIZER_FACIAL::startRecognizeFaceImage(cv::Mat img) {

  imageToRecognize = img.clone();
  std::cout << "Image rows=" << imageToRecognize.rows << "  image columns=" << imageToRecognize.cols << "\n";

  QTime timeTestRecognition;
  timeTestRecognition.start();

  if (flagResizeImages && ((imageToRecognize.rows > newHighImages) || (imageToRecognize.cols > newWidthImages))) {
    cv::resize(imageToRecognize, tempImg, cv::Size(newWidthImages, newHighImages), 0, 0, cv::INTER_LANCZOS4);
    std::cout << "Resized to rows=" << tempImg.rows << "  columns=" << tempImg.cols << "\n";
    descriptorTemp = p_descriptor -> test(tempImg);
  } else {
    descriptorTemp = p_descriptor -> test(imageToRecognize);
  }

  if (descriptorTemp != NULL) {

    myDictionary -> dispersedSolution(( * descriptorTemp)); // Sparse solution
    id = myDictionary -> estimateCluster(expectedPercentageDifference);
    elapsedRecognition = timeTestRecognition.elapsed();

    QString recognitionResult;
    if (expectedPercentageDifference >= thresholdFaceRecognizer)
      recognitionResult = nameUsersListAndId[id];
    else
      recognitionResult = "Unknown";

    QString infoText = "Descriptor name=" + p_descriptor -> nameDescriptor() + "\nMost probable user=" + nameUsersListAndId[id] + "\nScore=" + QString::number(expectedPercentageDifference) + "\nThreshold set=" + QString::number(thresholdFaceRecognizer) + "\nResult=" + recognitionResult + "\nNumber of calculated descriptors=" + QString::number(descriptorTemp -> cols()) + "\nDescriptor dimensions=" + QString::number(descriptorTemp -> rows()) + "\nComputation time=" + QString::number(elapsedRecognition) + " ms";

    emit testResultImageInformation(infoText);
    //myDictionary->sendToMatlab();
    emit setPlotSparseSolution();

  }

}

void RECOGNIZER_FACIAL::recognizeImagesList(QList < imageTransaction > listToRecognize) {

  if (stopped) {
    std::cout << "Event not processed\n";
    return;
  }

  std::cout << "\n\nA list of size=" << listToRecognize.size() << " arrived\n";

  for (int i = 0; i < listToRecognize.size(); i++) {

    //________THREAD CONTROL_____________
    {
      QMutexLocker locker( & mutex);
      if (stopped) {
        std::cout << "Exiting mutex in RECOGNIZER_FACIAL::recognizeImagesList\n";
        return; // Informs that the calculation was stopped by the user
      }
    }
    //_____________________________________

    //_______________________RECOGNITION ROUTINE SHOULD BE WRITTEN NEXT_____________________________________________//

    std::cout << "Recognizing image with id=" << listToRecognize[i].id << "\n";
    imageToRecognize = listToRecognize[i].img;

    std::cout << "Input image rows=" << imageToRecognize.rows << "  columns=" << imageToRecognize.cols << "\n";

    if (flagResizeImages && ((imageToRecognize.rows > newHighImages) || (imageToRecognize.cols > newWidthImages))) {
      cv::resize(imageToRecognize, tempImg, cv::Size(newWidthImages, newHighImages), 0, 0, cv::INTER_LANCZOS4);
      std::cout << "Resized to rows=" << tempImg.rows << "  columns=" << tempImg.cols << "\n";
      descriptorTemp = p_descriptor -> test(tempImg);
    } else {
      descriptorTemp = p_descriptor -> test(imageToRecognize);
    }

    if (descriptorTemp != NULL) {

      myDictionary -> dispersedSolution(( * descriptorTemp)); // Sparse solution
      id = myDictionary -> estimateCluster(expectedPercentageDifference);

      QString recognitionResult;
      if (expectedPercentageDifference >= thresholdFaceRecognizer)
        recognitionResult = nameUsersListAndId[id];
      else
        recognitionResult = "Unknown";

      listToRecognize[i].name = recognitionResult.toStdString();

      emit recognizedImage(listToRecognize[i]);

    }

    //______________________________________________________________________________________________________________________________//

  }

  std::cout << "Exiting RECOGNIZER_FACIAL::recognizeImagesList\n";
}

void RECOGNIZER_FACIAL::recognizedImage(std::vector < cv::Mat > listDetectedObjects, std::vector < cv::Rect > coordinatesDetectedObjects) {

  if (stopped) {
    std::cout << "Event not processed\n";
    return;
  }

  std::cout << "\n\nA list of size=" << listDetectedObjects.size() << " arrived\n";

  for (int i = 0; i < listDetectedObjects.size(); i++) {

    //________THREAD CONTROL_____________
    {
      QMutexLocker locker( & mutex);
      if (stopped) {
        std::cout << "Exiting mutex in RECOGNIZER_FACIAL::recognizedImage\n";
        return; // Informs that the calculation was stopped by the user
      }
    }
    //_____________________________________

    //_______________________RECOGNITION ROUTINE SHOULD BE WRITTEN NEXT_____________________________________________//

    imageToRecognize = listDetectedObjects[i];
    std::cout << "Image rows=" << imageToRecognize.rows << "  columns=" << imageToRecognize.cols << "\n";

    if (flagResizeImages && ((imageToRecognize.rows > newHighImages) || (imageToRecognize.cols > newWidthImages))) {
      cv::resize(imageToRecognize, tempImg, cv::Size(newWidthImages, newHighImages), 0, 0, cv::INTER_LANCZOS4);
      std::cout << "Resized to rows=" << tempImg.rows << "  columns=" << tempImg.cols << "\n";
      descriptorTemp = p_descriptor -> test(tempImg);
    } else {
      descriptorTemp = p_descriptor -> test(imageToRecognize);
    }

    if (descriptorTemp != NULL) {

      myDictionary -> dispersedSolution(( * descriptorTemp)); // Sparse solution
      id = myDictionary -> estimateCluster(expectedPercentageDifference);

      QString recognitionResult;
      if (expectedPercentageDifference >= thresholdFaceRecognizer)
        recognitionResult = nameUsersListAndId[id];
      else
        recognitionResult = "Unknown";

      {
        QMutexLocker locker( & mutex);
        if (!stopped)
          emit recognizedImage(coordinatesDetectedObjects[i].tl(), recognitionResult, imageToRecognize.clone());
      }

    }

    //_________________________________________________________________________________________//

  }

  std::cout << "Exiting RECOGNIZER_FACIAL::recognizedImage\n";

}

void RECOGNIZER_FACIAL::recognizedImage(std::vector < cv::Mat > listDetectedObjects, std::vector < cv::RotatedRect > coordinatesDetectedObjectsRotated) {

  if (stopped) {
    std::cout << "Event not processed\n";
    return;
  }

  std::cout << "\n\nA list of size=" << listDetectedObjects.size() << " arrived\n";

  for (int i = 0; i < listDetectedObjects.size(); i++) {

    //________THREAD CONTROL_____________
    {
      QMutexLocker locker( & mutex);
      if (stopped) {
        std::cout << "Exited mutex in RECOGNIZER_FACIAL::recognizedImage\n";
        return; //Informs that the calculation was stopped by the user
      }
    }
    //___________________________________

    //_______________________RECOGNITION ROUTINE SHOULD BE WRITTEN HERE_____________________________________________//

    imageToRecognize = listDetectedObjects[i];
    std::cout << "Rows of the image=" << imageToRecognize.rows << "  columns of the image=" << imageToRecognize.cols << "\n";

    if (flagResizeImages && ((imageToRecognize.rows > newHighImages) || (imageToRecognize.cols > newWidthImages))) {
      cv::resize(imageToRecognize, tempImg, cv::Size(newWidthImages, newHighImages), 0, 0, cv::INTER_LANCZOS4);
      std::cout << "Resized to rows=" << tempImg.rows << "  columns=" << tempImg.cols << "\n";
      descriptorTemp = p_descriptor -> test(tempImg);
    } else {
      descriptorTemp = p_descriptor -> test(imageToRecognize);
    }

    if (descriptorTemp != NULL) {

      myDictionary -> dispersedSolution(( * descriptorTemp)); //Sparse solution
      id = myDictionary -> estimateCluster(expectedPercentageDifference);

      QString recognitionResult;
      if (expectedPercentageDifference >= thresholdFaceRecognizer)
        recognitionResult = nameUsersListAndId[id];
      else
        recognitionResult = "Unknown";

      {
        QMutexLocker locker( & mutex);
        if (!stopped)
          emit recognizedImage(coordinatesDetectedObjectsRotated[i].boundingRect().tl(), recognitionResult, imageToRecognize.clone());
      }

    }

  }

  std::cout << "Exited RECOGNIZER_FACIAL::recognizedImage\n";

}

void RECOGNIZER_FACIAL::stop() {

  qDebug() << "ENTERED void RECOGNIZER_FACIAL::stop() \n";
  mutex.lock();
  stopped = true;
  mutex.unlock();
  qDebug() << "EXITED void RECOGNIZER_FACIAL::stop()\n";

}

void RECOGNIZER_FACIAL::enableRecognition() {
  /*
  This method must be called after stop is invoked while using an external recognition function,
  because these leave the stopped variable set to false, and if left in this state, some methods synchronized with this flag will not work properly.
  */
  stopped = false;
  std::cout << "Recognition re-enabled\n";
}

//__________________________________TEST SLOT DECLARATION BELOW______________________________________//

void RECOGNIZER_FACIAL::getImgTest(QMap < QString, QStringList > & listDirAndImgTest) {

  QString pathRootDataBaseSet("");
  {
    QStringList temp = dataBase -> getPath().split("/");
    for (int i = 1; i < temp.size() - 1; i++)
      pathRootDataBaseSet = pathRootDataBaseSet + QString("/") + temp[i];

    QString nameTemp = temp.last();
    temp = nameTemp.split("_");
    nameTemp = "";
    for (int i = 0; i < temp.size() - 1; i++)
      nameTemp = nameTemp + temp[i] + QString("_");

    pathRootDataBaseSet = pathRootDataBaseSet + QString("/") + nameTemp + QString("test");

  }

  qDebug() << "Reading test set from the directory=" << pathRootDataBaseSet << "\n";

  QDir dirTest = QDir(pathRootDataBaseSet);
  qDebug() << dirTest.entryList() << "\n";

  dirTest.setFilter(QDir::AllDirs | QDir::NoDotAndDotDot); //Only directories will be read
  QStringList subDirTest = dirTest.entryList();
  QStringList filtersSubDirTest;
  filtersSubDirTest << "*.jpg" << "*.png" << "*.pgm" << "*.JPEG";

  for (int i = 0; i < subDirTest.size(); i++) {

    QDir tempDir = QDir(dirTest.path() + QString("/") + subDirTest[i]);
    tempDir.setNameFilters(filtersSubDirTest);
    QStringList nameImg = tempDir.entryList();

    for (int j = 0; j < nameImg.size(); j++) {
      nameImg[j] = tempDir.path() + QString("/") + nameImg[j];
    }

    listDirAndImgTest.insert(subDirTest[i], nameImg);

  }

}

void RECOGNIZER_FACIAL::recognizedImage(const cv::Mat img, QString & recognitionResult, double & score) {

  imageToRecognize = img.clone();
  if (flagResizeImages && ((imageToRecognize.rows > newHighImages) || (imageToRecognize.cols > newWidthImages))) {
    cv::resize(imageToRecognize, tempImg, cv::Size(newWidthImages, newHighImages), 0, 0, cv::INTER_LANCZOS4);
    descriptorTemp = p_descriptor -> test(tempImg);
  } else {
    descriptorTemp = p_descriptor -> test(imageToRecognize);
  }

  if (descriptorTemp != NULL) {

    myDictionary -> dispersedSolution(( * descriptorTemp)); //Sparse solution
    id = myDictionary -> estimateCluster(expectedPercentageDifference);

    score = expectedPercentageDifference;
    recognitionResult = nameUsersListAndId[id];

  } else {
    score = 0;
    recognitionResult = ""; //Changed to empty because there might be a folder in the test directory named NN
  }

}

cv::Mat RECOGNIZER_FACIAL::corruptedPixelImg(cv::Mat & img, double percentage) {

  cv::Mat corruptedImg;

  cv::cvtColor(img, corruptedImg, CV_BGR2GRAY);

  srand((QTime::currentTime()).msec()); //Initial seed

  int numberPixel = corruptedImg.rows * corruptedImg.cols;
  QList < int > listAllPixel;
  listAllPixel.reserve(numberPixel);
  for (int i = 0; i < numberPixel; i++)
    listAllPixel.push_back(i);

  if (percentage > 100) percentage = 100;
  if (percentage < 0) percentage = 0;
  int numberPixelstoCorrupt = numberPixel * (percentage / 100);

  std::random_shuffle(listAllPixel.begin(), listAllPixel.end()); //Shuffling randomly

  uchar * pImg = corruptedImg.ptr < uchar > (0);
  for (int i = 0; i < numberPixelstoCorrupt; i++)
    pImg[listAllPixel[i]] = rand() % 256;

  return corruptedImg;

}

void RECOGNIZER_FACIAL::generateTest() {

  qDebug() << "STARTING TEST IN generateTest()\n"; //

  //__________CODE SHOULD GO HERE_________________//

  //Test based on the article Face Recognition: Comparative Study between Linear and non-Linear Dimensionality Reduction Methods
  //__________________________________________________________________________//

  //________________With the orl_faces database______________________________________//
  QMap < QString, QStringList > listDirAndImgTest;
  getImgTest(listDirAndImgTest);
  QMap < QString, QStringList > ::iterator iteratorList;

  QTime timeTestRecognition;
  timeTestRecognition.start();

  //_______________________________________________________________//

  double accuracy = 0;
  for (iteratorList = listDirAndImgTest.begin(); iteratorList != listDirAndImgTest.end(); ++iteratorList) {

    QString tempNameUser = iteratorList.key();
    QStringList listImg = iteratorList.value();

    double pi = 0;
    for (int i = 0; i < listImg.size(); i++) {
      cv::Mat img = cv::imread(listImg[i].toStdString());
      QString recognitionResult;
      double score;
      recognizedImage(img, recognitionResult, score);

      if (recognitionResult == tempNameUser)
        pi++;

    }

    pi = pi / listImg.size();
    accuracy = accuracy + pi;

  }

  accuracy = accuracy / listDirAndImgTest.size();
  qDebug() << "Accuracy=" << accuracy << "\n";
  qDebug() << "\nTime spent on the test=" << timeTestRecognition.elapsed() << "\n";

  //________________________________________________________________//

  qDebug() << "FINISHING TEST IN generateTest()\n";
  emit testWasGenerated(); //Please do not delete this line, as it informs UVface that the test has finished. 

}
