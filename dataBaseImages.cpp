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

#include <dataBaseImages.h>
#include "descriptor.h"
// Qt
#include <QDebug>
#include <QTextStream>
// OpenCV
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// This global variable controls the storage format of descriptors: 0 for XML and 1 for DES
const int FLAG_FORMAT = 1;

// Define the two possible extensions for storing descriptor files
static QString descriptorFormat[2] = {
    ".xml",
    ".des"
};

// The following function is useful for saving the descriptor in .des format
void saveMatInFileDes(cv::Mat* temp, QString path) {
    /*
     * Numeric types in cv::Mat:
     * 
     * CV_8U  - 8-bit unsigned integers (0..255)
     * CV_8S  - 8-bit signed integers (-128..127)
     * CV_16U - 16-bit unsigned integers (0..65535)
     * CV_16S - 16-bit signed integers (-32768..32767)
     * CV_32S - 32-bit signed integers (-2147483648..2147483647)
     * CV_32F - 32-bit floating-point numbers (-FLT_MAX..FLT_MAX, INF, NAN)
     * CV_64F - 64-bit floating-point numbers (-DBL_MAX..DBL_MAX, INF, NAN)
     */

    int depth = temp->depth();

    switch (depth) {
        case CV_8U: {
            unsigned char* pTemp = temp->ptr<unsigned char>(0);
            QFile descriptorFile(path);
            descriptorFile.open(QIODevice::WriteOnly);

            int rows = temp->rows;
            int cols = temp->cols;

            descriptorFile.write(reinterpret_cast<char*>(&depth), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(&rows), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(&cols), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(pTemp), cols * rows * sizeof(unsigned char));
            descriptorFile.close();
        } break;

        case CV_8S: {
            char* pTemp = temp->ptr<char>(0);
            QFile descriptorFile(path);
            descriptorFile.open(QIODevice::WriteOnly);

            int rows = temp->rows;
            int cols = temp->cols;

            descriptorFile.write(reinterpret_cast<char*>(&depth), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(&rows), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(&cols), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(pTemp), cols * rows * sizeof(char));
            descriptorFile.close();
        } break;

        case CV_16U: {
            unsigned short int* pTemp = temp->ptr<unsigned short int>(0);
            QFile descriptorFile(path);
            descriptorFile.open(QIODevice::WriteOnly);

            int rows = temp->rows;
            int cols = temp->cols;

            descriptorFile.write(reinterpret_cast<char*>(&depth), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(&rows), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(&cols), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(pTemp), cols * rows * sizeof(unsigned short int));
            descriptorFile.close();
        } break;

        case CV_16S: {
            short int* pTemp = temp->ptr<short int>(0);
            QFile descriptorFile(path);
            descriptorFile.open(QIODevice::WriteOnly);

            int rows = temp->rows;
            int cols = temp->cols;

            descriptorFile.write(reinterpret_cast<char*>(&depth), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(&rows), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(&cols), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(pTemp), cols * rows * sizeof(short int));
            descriptorFile.close();
        } break;

        case CV_32S: {
            int* pTemp = temp->ptr<int>(0);
            QFile descriptorFile(path);
            descriptorFile.open(QIODevice::WriteOnly);

            int rows = temp->rows;
            int cols = temp->cols;

            descriptorFile.write(reinterpret_cast<char*>(&depth), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(&rows), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(&cols), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(pTemp), cols * rows * sizeof(int));
            descriptorFile.close();
        } break;

        case CV_32F: {
            float* pTemp = temp->ptr<float>(0);
            QFile descriptorFile(path);
            descriptorFile.open(QIODevice::WriteOnly);

            int rows = temp->rows;
            int cols = temp->cols;

            descriptorFile.write(reinterpret_cast<char*>(&depth), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(&rows), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(&cols), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(pTemp), cols * rows * sizeof(float));
            descriptorFile.close();
        } break;

        case CV_64F: {
            double* pTemp = temp->ptr<double>(0);
            QFile descriptorFile(path);
            descriptorFile.open(QIODevice::WriteOnly);

            int rows = temp->rows;
            int cols = temp->cols;

            descriptorFile.write(reinterpret_cast<char*>(&depth), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(&rows), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(&cols), sizeof(int));
            descriptorFile.write(reinterpret_cast<char*>(pTemp), cols * rows * sizeof(double));
            descriptorFile.close();
        } break;
    }
}

//The following function is useful for loading the descriptor in .des format
void loadMatInFileDes(cv::Mat * descriptorsBase, QString path) {

  /*
  Numeric types of cv::Mat

  CV_8U - 8-bit unsigned integers ( 0..255 )
  CV_8S - 8-bit signed integers ( -128..127 )
  CV_16U - 16-bit unsigned integers ( 0..65535 )
  CV_16S - 16-bit signed integers ( -32768..32767 )
  CV_32S - 32-bit signed integers ( -2147483648..2147483647 )
  CV_32F - 32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )
  CV_64F - 64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )
  */

  QFile descriptorFile(path);
  descriptorFile.open(QIODevice::ReadOnly);
  int depth;
  int rows;
  int cols;
  descriptorFile.read(reinterpret_cast < char * > ( & depth), sizeof(int));
  descriptorFile.read(reinterpret_cast < char * > ( & rows), sizeof(int));
  descriptorFile.read(reinterpret_cast < char * > ( & cols), sizeof(int));

  switch (depth) {

  case CV_8U: {
    unsigned char * pTemp = new unsigned char[cols * rows];
    descriptorFile.read(reinterpret_cast < char * > (pTemp), cols * rows * sizeof(unsigned char));
    cv::Mat temp(rows, cols, CV_8U, pTemp);
    descriptorsBase -> push_back(temp); //Stacking the descriptor   
    delete[] pTemp;
  }
  break;
  case CV_8S: {
    char * pTemp = new char[cols * rows];
    descriptorFile.read(reinterpret_cast < char * > (pTemp), cols * rows * sizeof(char));
    cv::Mat temp(rows, cols, CV_8S, pTemp);
    descriptorsBase -> push_back(temp); //Stacking the descriptor   
    delete[] pTemp;
  }
  break;
  case CV_16U: {
    unsigned short int * pTemp = new unsigned short int[cols * rows];
    descriptorFile.read(reinterpret_cast < char * > (pTemp), cols * rows * sizeof(unsigned short int));
    cv::Mat temp(rows, cols, CV_16U, pTemp);
    descriptorsBase -> push_back(temp); //Stacking the descriptor   
    delete[] pTemp;
  }
  break;
  case CV_16S: {
    short int * pTemp = new short int[cols * rows];
    descriptorFile.read(reinterpret_cast < char * > (pTemp), cols * rows * sizeof(short int));
    cv::Mat temp(rows, cols, CV_16S, pTemp);
    descriptorsBase -> push_back(temp); //Stacking the descriptor   
    delete[] pTemp;
  }
  break;
  case CV_32S: {
    int * pTemp = new int[cols * rows];
    descriptorFile.read(reinterpret_cast < char * > (pTemp), cols * rows * sizeof(int));
    cv::Mat temp(rows, cols, CV_32S, pTemp);
    descriptorsBase -> push_back(temp); //Stacking the descriptor   
    delete[] pTemp;
  }
  break;
  case CV_32F: {
    float * pTemp = new float[cols * rows];
    descriptorFile.read(reinterpret_cast < char * > (pTemp), cols * rows * sizeof(float));
    cv::Mat temp(rows, cols, CV_32F, pTemp);
    descriptorsBase -> push_back(temp); //Stacking the descriptor   
    delete[] pTemp;
  }
  break;
  case CV_64F: {
    double * pTemp = new double[cols * rows];
    descriptorFile.read(reinterpret_cast < char * > (pTemp), cols * rows * sizeof(double));
    cv::Mat temp(rows, cols, CV_64F, pTemp);
    descriptorsBase -> push_back(temp); //Stacking the descriptor   
    delete[] pTemp;
  }
  break;
  }

  descriptorFile.close();

}

//____________________Global variables for the file______________
std::vector < int > generateCompressionParams() {

  std::vector < int > compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(100); //The highest quality
  return compression_params;

}
static std::vector < int > G_COMPRESSION_PARAMS = generateCompressionParams(); //Image writing parameters

//_______________________________________________________________

USER::USER(QDir myDir, QString nameUser, int id, DATA_BASE * myDataBase, bool createDir): nameUser(nameUser), id(id), myDataBase(myDataBase) {

  if (createDir == true) {
    myDir.mkdir(nameUser); //Creating the folder

    dir = QDir(myDir.path() + "/" + nameUser); //Associating the folder with this class's directory
    QStringList filters;
    filters << "*.jpg" << "*.png" << "*.pgm" << "*.JPEG";
    dir.setNameFilters(filters);

    numeratorNext = 0; //Since there are no images
  } else {

    dir = QDir(myDir.path() + "/" + nameUser); //Associating the folder with this class's directory
    QStringList filters;
    filters << "*.jpg" << "*.png" << "*.pgm" << "*.JPEG";
    dir.setNameFilters(filters);

    QFile::remove(dir.path() + "/info.txt"); //Remove it so a new file can be created if there is a previous one
    loadListImages(); //Here we load the list of image names in the folder

    setNumeratorName(); //Update image numbering

    myDataBase -> haveBeenEdited(this); //Add to the editing list

  }

  saveId(); //Store the id on disk
  indexCurrentImageEditing = -1;

}

USER::USER(QDir myDir, QString nameUser, DATA_BASE * myDataBase): nameUser(nameUser), myDataBase(myDataBase) {

  dir = QDir(myDir.path() + "/" + nameUser); // Associating the folder with the directory of this class
  QStringList filters;
  filters << "*.jpg" << "*.png" << "*.pgm" << "*.JPEG";
  dir.setNameFilters(filters);

  readId(); // Reading the disk ID
  loadListImages(); // Loading the list of image names in the folder
  setNumeratorName(); // Updating the image numbering

  indexCurrentImageEditing = -1;

  if (listNameImages.empty())
    myDataBase -> haveBeenEdited(this); // Adding to the editing list

}

USER::~USER() {
  saveInfo();
  qDebug() << "The user with name=" << nameUser << " and id=" << id << " has been deleted.\n"; // Debug message for user deletion
}

void USER::readId() {
  // Reading the disk ID here

  QFile idFile(dir.path() + "/id");
  idFile.open(QIODevice::ReadOnly);
  idFile.read(reinterpret_cast < char * > ( & id), sizeof(int));
  idFile.close();
}

void USER::saveId() {

  //____Storing a file with its ID_____
  QFile idFile(dir.path() + "/id");
  idFile.open(QIODevice::WriteOnly);
  idFile.write(reinterpret_cast < char * > ( & id), sizeof(int));
  idFile.close();
  //_________________________________________
}

void USER::setNumeratorName() {

  // This function looks for missing indices to name the images with those indices

  numeratorNext = 0; // Starting the check from zero
  QList < int > assignedNumbers;

  for (int i = 0; i < listNameImages.size(); i++) {

    QString tempNameImage = listNameImages[i];
    QStringList listTemp = tempNameImage.split("_"); /* Splitting the name and the index (only considered if the name was assigned by the system, in which case the canonical name will be image_x.jpg where x is the index or numerator) */
    if (listTemp.size() == 2) // If the image name matches the numbering (example: nameImage_121.jpg)
    {
      //_______________________________________________________
      QString temp1 = listTemp[1];
      QStringList listTemp2 = temp1.split("."); // Example: 121.jpg
      if (listTemp2.size() == 2) {
        bool ok;
        int num = listTemp2[0].toInt( & ok);
        if (ok)
          assignedNumbers.push_back(num);
      }
      //_______________________________________________________
    }

  }

  if (!assignedNumbers.empty()) // If the list is not empty
  {

    int numberMax = * std::max_element(assignedNumbers.begin(), assignedNumbers.end()); // Extracting the maximum assigned index
    numeratorNext = numberMax + 1; // This will be the next maximum index to be assigned

    for (int i = 0; i < numeratorNext; i++)
      numeratorNotAssigned.push_back(i); // Filling the list with all indices up to numeratorNext-1

    for (int i = 0; i < assignedNumbers.size(); i++)
      numeratorNotAssigned.removeOne(assignedNumbers[i]); // Removing all assigned indices

  }

}

int USER::getIndex() {

  int id;

  if (numeratorNotAssigned.empty()) {
    id = numeratorNext;
    numeratorNext++;
  } else {
    id = numeratorNotAssigned.takeLast();
  }

  return id;
}

void USER::loadListImages() {

  if (dir.exists("info.txt")) {

    QFile fileTemp(dir.path() + "/info.txt");

    if (!fileTemp.open(QIODevice::ReadOnly | QIODevice::Text))
      return;

    QTextStream in ( & fileTemp);

    while (!in.atEnd()) {

      QString line = in.readLine();

      /* List of images plus the flag indicating if a descriptor has been calculated for the image */
      QStringList listNameImageAndFlag = line.split("="); // Images cannot have names that include the '=' character
      QString tempNameImage = listNameImageAndFlag[0];
      QStringList nameDescriptorTemp = tempNameImage.split("."); // Split by dot to extract only the name (example: image_1.jpg)
      bool tempFlag = static_cast < bool > (listNameImageAndFlag[1].toInt());

      if (dir.exists(tempNameImage)) // Check if the image still exists (it may have been deleted manually)
      { // If the image exists, it should be added normally

        listNameImages.push_back(tempNameImage); // If the image exists, add it to the list
        listNameDescriptorsImages.push_back(nameDescriptorTemp[0] + descriptorFormat[FLAG_FORMAT]);
        listFlagCalculatedDescriptor.push_back(tempFlag);

        if (tempFlag == false)
          myDataBase -> haveBeenEdited(this); // Added to the editing list       

      } else { // If the image does not exist, its associated .des file should be deleted
        dir.remove(nameDescriptorTemp[0] + descriptorFormat[FLAG_FORMAT]);
        myDataBase -> haveBeenEdited(this); // Added to the editing list
      }

    }

    // Read also the new images
    QStringList listNameImagesTotal = dir.entryList(); // List of image names on disk
    for (int i = 0; i < listNameImages.size(); i++)
      listNameImagesTotal.removeOne(listNameImages[i]);

    if (!listNameImagesTotal.empty()) myDataBase -> haveBeenEdited(this); // Added to the editing list

    for (int i = 0; i < listNameImagesTotal.size(); i++) { // Insert the new list
      listNameImages.push_back(listNameImagesTotal[i]);
      QStringList nameDescriptorTemp = listNameImagesTotal[i].split(".");
      listNameDescriptorsImages.push_back(nameDescriptorTemp[0] + descriptorFormat[FLAG_FORMAT]);
      listFlagCalculatedDescriptor.push_back(false);
    }

  } else { // Build from scratch

    // Read the list of images
    listNameImages = dir.entryList(); // List of image names on disk
    for (int i = 0; i < listNameImages.size(); i++)
      listFlagCalculatedDescriptor.push_back(false); // Descriptors have not been calculated yet

    //_____ Build the list of descriptor names for each image _______//
    for (int i = 0; i < listNameImages.size(); i++) {

      QString nameImagenTemp = listNameImages[i];
      QStringList nameDescriptorTemp = nameImagenTemp.split("."); /* NOTE: image names cannot have intermediate dots, example: imagexrdd.fdfdfd.jpg */
      listNameDescriptorsImages.push_back(nameDescriptorTemp[0] + descriptorFormat[FLAG_FORMAT]);

      // This was done to clean the folder of descriptor files that will no longer be used, so it doesn't have to be done manually
      dir.remove(nameDescriptorTemp[0] + descriptorFormat[0]); //.xml
      dir.remove(nameDescriptorTemp[0] + descriptorFormat[1]); //.des

    }
    //_____________________________________________________________//

    // This is because if an old user had their info.txt file manually deleted, we would end up here
    myDataBase -> haveBeenEdited(this); // Added to the editing list

  }

}

void USER::addImages(const cv::Mat & image) {

  QString nameBase = "image_" + QString::number(getIndex());
  QString nameBaseImage = nameBase + QString(".jpg");
  QString nameBaseDescriptor = nameBase + descriptorFormat[FLAG_FORMAT];

  QString nameFileImageWrite = dir.path() + "/" + nameBaseImage; // Organizing the name 

  cv::imwrite(nameFileImageWrite.toStdString(), image, G_COMPRESSION_PARAMS); // Write the image to disk

  listNameImages.push_back(nameBaseImage); // Store the image name
  listNameDescriptorsImages.push_back(nameBaseDescriptor);
  listFlagCalculatedDescriptor.push_back(false); // Mark that the image's base descriptor has not been calculated yet

  indexCurrentImageEditing = listNameImages.size() - 1; // Update the index of the current image being edited

  myDataBase -> haveBeenEdited(this); // Added to the editing list 

}

void USER::addImages(const std::vector < cv::Mat > & images) {

  for (int i = 0; i < images.size(); i++)
    addImages(images[i]);

}

void USER::deleteImage(int index) {

  if ((index >= 0) && (index < listNameImages.size())) {

    // Delete from physical memory
    dir.remove(listNameImages[index]);
    dir.remove(listNameDescriptorsImages[index]);

    // Delete from virtual memory
    listNameImages.removeAt(index);
    listNameDescriptorsImages.removeAt(index);
    listFlagCalculatedDescriptor.removeAt(index);

    if (indexCurrentImageEditing == 0)
      indexCurrentImageEditing = listNameImages.size() - 1;
    else
      indexCurrentImageEditing--;

    myDataBase -> haveBeenEdited(this); // Added to the editing list
  }

}

bool USER::calculeDescriptors(ABSTRACT_DESCRIPTOR * p_descriptor, cv::Mat * descriptorsBase) {

  /* NOTE: Issues that can occur here include a lack of images or if the descriptor calculation does not produce vectors, in which case it will return false immediately to notify the user of this problem */

  if (listNameImages.empty()) return false; // No images for the user

  cv::Mat tempImage;

  for (int i = 0; i < listNameImages.size(); i++) {

    if ((!listFlagCalculatedDescriptor[i]) || (!QFile::exists(dir.path() + "/" + listNameDescriptorsImages[i]))) { // Descriptor is only calculated if it hasn't been calculated before, or if the previously calculated file was deleted

      getImage(tempImage, listNameImages[i]); // Get the image

      cv::Mat * temp = p_descriptor -> descriptor_base(tempImage); // Calculate the descriptor

      if (temp == NULL) return false; // The image did not produce descriptors

      // Means the .xml format was chosen for descriptor files
      if (FLAG_FORMAT == 0) {

        //_________ STORE THE BASE DESCRIPTOR FOR EACH IMAGE HERE _____________________//
        cv::FileStorage fs((dir.path() + "/" + listNameDescriptorsImages[i]).toStdString(), cv::FileStorage::WRITE);
        fs << "descriptor" << ( * temp);
        fs.release();
        //__________________________________________________________________________//

        // Means the .des format was chosen for descriptor files
      } else {
        saveMatInFileDes(temp, (dir.path() + "/" + listNameDescriptorsImages[i]));
      }

      descriptorsBase -> push_back(( * temp)); // Stack the descriptor
      listFlagCalculatedDescriptor[i] = true; // Mark it as calculated regardless of whether descriptors were produced or not

    } else { // If the descriptor for the image is already calculated, we just load it

      // Means the .xml format was chosen for descriptor files
      if (FLAG_FORMAT == 0) {

        //_________ READ THE BASE DESCRIPTOR FOR EACH IMAGE HERE _____________________//

        cv::Mat temp;
        cv::FileStorage fs((dir.path() + "/" + listNameDescriptorsImages[i]).toStdString(), cv::FileStorage::READ);
        fs["descriptor"] >> temp;
        fs.release();
        descriptorsBase -> push_back(temp); // Stack the descriptor   

        //__________________________________________________________________________//

        // Means the .des format was chosen for descriptor files
      } else {
        loadMatInFileDes(descriptorsBase, (dir.path() + "/" + listNameDescriptorsImages[i]));
      }

    }

  }

  return true; // Means the user must be added as they have descriptors

}

void USER::saveInfo() {

  {
    // Saving the list of folders with their corresponding ids
    QFile::remove(dir.path() + "/info.txt");

    QFile fileTemp(dir.path() + "/info.txt");
    if (!fileTemp.open(QIODevice::WriteOnly | QIODevice::Text))
      return;

    QTextStream out( & fileTemp);

    for (int i = 0; i < listNameImages.size(); i++)
      out << (listNameImages[i]) << "=" << (listFlagCalculatedDescriptor[i]) << "\n"; /* Here we store each image name with its descriptor flag */

    fileTemp.close();
  }

}

void USER::resetListFlagCalculatedDescriptor() {

  for (int i = 0; i < listFlagCalculatedDescriptor.size(); i++)
    listFlagCalculatedDescriptor[i] = false; // Remember, it is not necessary to modify the files

}

QString USER::getNameUser() const {
  return nameUser;
}

int USER::getId() const {
  return id;
}

int USER::getNumberImages() const {
  return listNameImages.size();
}

int USER::getIndexCurrentImageEditing() const {
  return indexCurrentImageEditing;
}

void USER::getImage(cv::Mat & image, QString nameImage) {
  image = cv::imread((dir.path() + "/" + nameImage).toStdString());
}

void USER::getImage(cv::Mat & image, int index) {

  if ((index >= 0) && (index < listNameImages.size()))
    image = cv::imread((dir.path() + "/" + listNameImages[index]).toStdString());
  else
    image = cv::Mat();

}

void USER::saveImageInIndex(cv::Mat & image, int index) {
  cv::imwrite((dir.path() + "/" + listNameImages[index]).toStdString(), image, G_COMPRESSION_PARAMS);
  listFlagCalculatedDescriptor[index] = false; // Remember that it is not necessary to modify the files
  myDataBase -> haveBeenEdited(this); // Added to the editing list 
}

void USER::resetIndexCurrentImageEditing() {
  indexCurrentImageEditing = -1;
}

int USER::getImageEditingNext(cv::Mat & image) {

  if (listNameImages.empty()) {
    image = cv::Mat();
    return -1;
  }

  indexCurrentImageEditing++;
  if (indexCurrentImageEditing > listNameImages.size() - 1)
    indexCurrentImageEditing = 0;

  image = cv::imread((dir.path() + "/" + listNameImages[indexCurrentImageEditing]).toStdString());

  return indexCurrentImageEditing;

}

int USER::getImageEditingBack(cv::Mat & image) {

  if (listNameImages.empty()) {
    image = cv::Mat();
    return -1;
  }

  indexCurrentImageEditing--;
  if (indexCurrentImageEditing < 0)
    indexCurrentImageEditing = listNameImages.size() - 1;

  image = cv::imread((dir.path() + "/" + listNameImages[indexCurrentImageEditing]).toStdString());

  return indexCurrentImageEditing;

}

QList < bool > USER::getListFlagCalculatedDescriptor() const {
  return listFlagCalculatedDescriptor;
}

DATA_BASE::DATA_BASE() {
  dataBaseIsLoad = false;
  flagOldDeletedUsers = false;
  flagNewUsersDeleted = false;
}

DATA_BASE::DATA_BASE(QString path) {

  if (QDir(path).exists()) {

    dir = QDir(path); // Assign the database directory
    dir.setFilter(QDir::AllDirs | QDir::NoDotAndDotDot); // Only directories will be read

    loadFlagEditing(); // Load information on whether there are users lacking descriptors in the database

    flagOldDeletedUsers = false;
    flagNewUsersDeleted = false;

    readIdNextAndIdnotAssigned();

    readOldUsers();
    createOldList();

    readNewUsers();
    createNewList();

    if (flagOldDeletedUsers) // If old users have been deleted, the database has been edited
      doneEditing(true);

    dataBaseIsLoad = true;
  }

}

DATA_BASE::~DATA_BASE() {

  saveInfo(); // Save the names and ids of corresponding users

  QList < USER * > tempListUsers = usersList.values();
  for (int i = 0; i < tempListUsers.size(); i++)
    delete tempListUsers[i];
  usersList.clear();
  editedUserList.clear();

}

bool DATA_BASE::loadDataBase(QString path) {

  if (QDir(path).exists()) {

    if (dataBaseIsLoad == true)
      clearDataBase();

    dir = QDir(path); // Assign the database directory
    dir.setFilter(QDir::AllDirs | QDir::NoDotAndDotDot); // Only directories will be read

    loadFlagEditing(); // Load information on whether there are users lacking descriptors in the database

    flagOldDeletedUsers = false;
    flagNewUsersDeleted = false;

    readIdNextAndIdnotAssigned();

    readOldUsers();
    createOldList();

    readNewUsers();
    createNewList();

    if (flagOldDeletedUsers) // If old users have been deleted, the database has been edited
      doneEditing(true);

    dataBaseIsLoad = true;

  }

}

void DATA_BASE::clearDataBase() {

  saveInfo(); // Save the names and ids of corresponding users

  dir = QDir();

  oldIdList.clear();
  oldUsersList.clear();
  newUsersList.clear();

  QList < USER * > tempListUsers = usersList.values();
  for (int i = 0; i < tempListUsers.size(); i++)
    delete tempListUsers[i];
  usersList.clear();
  nameUsersList.clear();
  editedUserList.clear();

  idNext = 0;
  idnotAssigned.clear();

  // Flags
  flagOldDeletedUsers = false;
  flagNewUsersDeleted = false;

  dataBaseIsLoad = false;

}

int DATA_BASE::getIdCorrect() {

  int id;
  if (idnotAssigned.empty()) {

    id = idNext; // If the list is empty, assign the next id to the highest assigned id
    idNext++;

  } else { // Assign ids of users that have been deleted
    id = idnotAssigned.takeLast();
  }

  return id;

}

void DATA_BASE::releaseId(int id) {

  // Prepare idNext and idnotAssigned
  if (idNext == id + 1) // Means the last user was deleted
  {

    while ((idnotAssigned.removeOne(id - 1)) && (id > 0))
      id = id - 1;

    idNext = id; // Assign the highest available id

  } else if (id + 1 < idNext)
    idnotAssigned.push_back(id);

}

QStringList DATA_BASE::getListNameUsers() const {
  return nameUsersList.values();
}

QMap < int, QString > DATA_BASE::getListNameUsersAndId() const {
  return nameUsersList;
}

int DATA_BASE::getNumberUsers() const {
  return nameUsersList.size();
}

int DATA_BASE::getIdOfUser(QString nameUser) {

  //______________Here we read the user's disk id_____________
  int id;
  QFile fileTemp(dir.path() + "/" + nameUser + "/id");
  fileTemp.open(QIODevice::ReadOnly);
  fileTemp.read(reinterpret_cast < char * > ( & id), sizeof(int));
  fileTemp.close();
  //___________________________________________________________

  return id;

}

USER * DATA_BASE::getUser(QString nameUser) {
  int id = getIdOfUser(nameUser); // Get the user's id
  return usersList[id]; // Return the address of the user
}

QList < USER * > DATA_BASE::getAllUsers() {
  return usersList.values();
}

QString DATA_BASE::getPath() const {
  return dir.path();
}

bool DATA_BASE::existUser(const QString nameUser) const {
  return dir.exists(nameUser);
}

bool DATA_BASE::dataBaseIsEdited() const {
  return flagDatabaseHasBeenModified;
}

bool DATA_BASE::databaseHasBeenLoaded() const {
  return dataBaseIsLoad;
}

void DATA_BASE::readOldUsers() {

  QFile fileTemp(dir.path() + "/infoOldList.txt");

  if (!fileTemp.open(QIODevice::ReadOnly | QIODevice::Text))
    return;

  QTextStream in ( & fileTemp);

  while (!in.atEnd()) {

    QString line = in.readLine();

    QStringList listNameAndId = line.split("="); // List of id and user name
    QString tempNameUser = listNameAndId[0];
    int tempId = listNameAndId[1].toInt();

    oldIdList.push_back(tempId);
    oldUsersList.push_back(tempNameUser);

  }

}

void DATA_BASE::createOldList() {

  for (int i = 0; i < oldUsersList.size(); i++) {

    //__________________________________________________________________________
    if (dir.exists(oldUsersList[i])) { // If the directory still exists
      USER * tempUser = new USER(dir, oldUsersList[i], this); // Create the old user
      usersList.insert(tempUser -> getId(), tempUser); // Associate the list with its own id
      nameUsersList.insert(tempUser -> getId(), tempUser -> getNameUser());
    } else {
      int id = oldIdList[i]; // Extract the old id
      releaseId(id); // Release the old id
      flagOldDeletedUsers = true;
    }

    //___________________________________________________________________________

  }

}

void DATA_BASE::readNewUsers() {

  newUsersList = dir.entryList();
  for (int i = 0; i < oldUsersList.size(); i++)
    newUsersList.removeOne(oldUsersList[i]);

}

void DATA_BASE::createNewList() {

  for (int i = 0; i < newUsersList.size(); i++) {

    int id = getIdCorrect();
    USER * tempUser = new USER(dir, newUsersList[i], id, this, false); // Create the new user and specify not to create the folder
    usersList.insert(id, tempUser); // Associate the list with its own id
    nameUsersList.insert(tempUser -> getId(), tempUser -> getNameUser());

  }

}

bool DATA_BASE::createUser(QString nameUser) {

  if (dir.exists(nameUser)) return false; // The user already exists

  int id = getIdCorrect();
  USER * tempUser = new USER(dir, nameUser, id, this); // Create the new user
  usersList.insert(id, tempUser); // Associate the list with its own id
  nameUsersList.insert(tempUser -> getId(), tempUser -> getNameUser());

  doneEditing(true); // EDITING

  return true; // The user was created successfully

}

bool DATA_BASE::deleteUser(QString nameUser) {

  if (!dir.exists(nameUser)) return false; // The user does not exist

  int id = getIdOfUser(nameUser); // Here we read the disk id of the user to delete

  //____________________________________//
  doneEditing(true); // EDITING
  editedUserList.remove(id);
  //____________________________________//

  USER * tempUser = usersList[id];
  delete tempUser; // First delete from dynamic memory
  usersList.remove(id); // Remove the user from the associative list
  nameUsersList.remove(id);

  //___Remove all files____________________________
  QDir tempDir = QDir(dir.path() + QString("/") + nameUser);
  QStringList tempListFiles = tempDir.entryList(QDir::AllEntries | QDir::NoDotAndDotDot);

  for (int i = 0; i < tempListFiles.size(); i++)
    tempDir.remove(tempListFiles[i]);
  //_____________________________________________________________

  dir.rmdir(nameUser); // Remove the folder

  releaseId(id); // Update the id list

  flagNewUsersDeleted = true;

  return true; // The user was deleted successfully

}

void DATA_BASE::readIdNextAndIdnotAssigned() {

  //____Reading idNext and idnotAssigned_________
  if (QFile::exists(dir.path() + "/infoIdNextAndIdnotAssigned")) {

    QFile fileTemp(dir.path() + "/infoIdNextAndIdnotAssigned");
    fileTemp.open(QIODevice::ReadOnly);

    fileTemp.read(reinterpret_cast < char * > ( & idNext), sizeof(int));

    bool flagTemp = false;
    fileTemp.read(reinterpret_cast < char * > ( & flagTemp), sizeof(bool));

    if (flagTemp == true) {

      int tempSz;
      fileTemp.read(reinterpret_cast < char * > ( & tempSz), sizeof(int));
      int * p_idnotAssigned = new int[tempSz];
      fileTemp.read(reinterpret_cast < char * > (p_idnotAssigned), tempSz * sizeof(int));

      for (int i = 0; i < tempSz; i++)
        idnotAssigned.push_back(p_idnotAssigned[i]);

      delete p_idnotAssigned;

    }

    fileTemp.close();

  } else {
    idNext = 0;
  }
  //______________________________________________

}

void DATA_BASE::saveInfo() {

  if (dir == QDir()) return; // Directory not assigned

  {
    // Saving the list of folders with their corresponding id assigned
    QFile::remove(dir.path() + "/infoOldList.txt");

    QFile fileTemp(dir.path() + "/infoOldList.txt");
    if (!fileTemp.open(QIODevice::WriteOnly | QIODevice::Text))
      return;

    QTextStream out( & fileTemp);

    QMap < int, USER * > ::iterator it;
    for (it = usersList.begin(); it != usersList.end(); ++it)
      out << (( * it) -> getNameUser()) << "=" << ( * it) -> getId() << "\n"; // Store each folder name here

    fileTemp.close();
  }

  //____Storing idNext and idnotAssigned_______
  {

    qDebug() << "Saving idNext=" << idNext << "\n";
    qDebug() << "Saving list=" << idnotAssigned << "\n";

    QFile::remove(dir.path() + "/infoIdNextAndIdnotAssigned");

    QFile fileTemp(dir.path() + "/infoIdNextAndIdnotAssigned");
    fileTemp.open(QIODevice::WriteOnly);

    fileTemp.write(reinterpret_cast < char * > ( & idNext), sizeof(int));
    if (!idnotAssigned.empty()) {
      bool flagTemp = true;
      int tempSz = idnotAssigned.size();
      fileTemp.write(reinterpret_cast < char * > ( & flagTemp), sizeof(bool));
      fileTemp.write(reinterpret_cast < char * > ( & tempSz), sizeof(int));

      int * p_idnotAssigned = new int[tempSz];
      for (int i = 0; i < tempSz; i++)
        p_idnotAssigned[i] = idnotAssigned[i];

      fileTemp.write(reinterpret_cast < char * > (p_idnotAssigned), tempSz * sizeof(int));

      delete p_idnotAssigned;

    } else {
      bool flagTemp = false;
      fileTemp.write(reinterpret_cast < char * > ( & flagTemp), sizeof(bool));
    }

    fileTemp.close();

  }
  //______________________________________________

}

void DATA_BASE::loadFlagEditing() {

  if (QFile::exists(dir.path() + "/flagEditing")) {

    QFile fileTemp(dir.path() + "/flagEditing");
    fileTemp.open(QIODevice::ReadOnly);

    bool flagTemp;
    fileTemp.read(reinterpret_cast < char * > ( & flagTemp), sizeof(bool));

    flagDatabaseHasBeenModified = flagTemp;
    emit haveBeenEdited(flagTemp);
  } else {
    flagDatabaseHasBeenModified = false; // To ensure an initial state
    doneEditing(true);
  }

}

void DATA_BASE::doneEditing(bool flag) {

  if ((flag == true) && (flagDatabaseHasBeenModified == false)) {
    QFile::remove(dir.path() + "/flagEditing");
    QFile fileTemp(dir.path() + "/flagEditing");
    fileTemp.open(QIODevice::WriteOnly);
    bool flagTemp = true;
    fileTemp.write(reinterpret_cast < char * > ( & flagTemp), sizeof(bool));

    fileTemp.close();
    flagDatabaseHasBeenModified = true;
    emit haveBeenEdited(true);
  } else if ((flag == false) && (flagDatabaseHasBeenModified == true)) {

    QFile::remove(dir.path() + "/flagEditing");
    QFile fileTemp(dir.path() + "/flagEditing");
    fileTemp.open(QIODevice::WriteOnly);
    bool flagTemp = false;
    fileTemp.write(reinterpret_cast < char * > ( & flagTemp), sizeof(bool));

    fileTemp.close();
    flagDatabaseHasBeenModified = false;

    emit haveBeenEdited(false);

  }

}

void DATA_BASE::haveBeenEdited(USER * user) {

  // ALTERNATIVE CODE TO THE COMMENTED ONE BELOW
  if (flagDatabaseHasBeenModified == false)
    doneEditing(true); // EDITING

  //_________________NOT USED IN CURRENT IMPLEMENTATION______________________//
  /*
  NOTE: The reason for not using the following code to track modified users is because deleting a user required relocating the D matrix in the sparse solution, which did not provide any advantages when loading modified users and rather complicated the code.
  */
  /*
  if(!editedUserList.contains(user->id))
  {
  qDebug()<<"EDITING on "<<user->nameUser<<"\n";
  editedUserList.insert(user->id,user);
  if(flagDatabaseHasBeenModified==false)
  doneEditing(true);//EDITING
  }else if(flagDatabaseHasBeenModified==false){
  doneEditing(true);//EDITING
  }
  */
  //_______________________________________________________________________//

}

void DATA_BASE::resetListFlagCalculatedDescriptor() {

  QList < USER * > tempListUsers = usersList.values();

  for (int i = 0; i < tempListUsers.size(); i++)
    (tempListUsers[i]) -> resetListFlagCalculatedDescriptor();

}
