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

#ifndef DATABASEIMAGES_H
#define DATABASEIMAGES_H

//STL
#include <vector>
#include <iostream> //Delete
//OpenCV
namespace cv {
  class Mat;
};

//QT
#include <QDir>
#include <QStringList>
#include <QMap>


class DATA_BASE;
class ABSTRACT_DESCRIPTOR;

class USER {

  QDir dir; //Directory where the information belonging to this user is stored
  QString nameUser; //User's name
  int id; //Numeric identifier of the user

  QStringList listNameImages; //List of image names stored on disk
  QStringList listNameDescriptorsImages; //List of image descriptor names stored on disk (chosen extension .des)
  QList < bool > listFlagCalculatedDescriptor; //Indicates whether the base descriptor of the image has been calculated (true) or not (false)

  int numeratorNext; //Highest unassigned image numerator
  QList < int > numeratorNotAssigned; //Stack of numerators less than numeratorNext that are unassigned

  int indexCurrentImageEditing;

  DATA_BASE * myDataBase;

  //Private functions
  void getImage(cv::Mat & image, QString nameImage);

  public:
    USER(QDir myDir, QString nameUser, int id, DATA_BASE * myDataBase = NULL, bool createDir = true); //For new users
  USER(QDir myDir, QString nameUser, DATA_BASE * myDataBase = NULL); //For users whose database has already assigned an id
  ~USER();
  void readId();
  void saveId();
  void setNumeratorName(); //Sets the number from which images will be enumerated
  int getIndex();

  void loadListImages();
  void addImages(const cv::Mat & image);
  void addImages(const std::vector < cv::Mat > & images);
  void deleteImage(int index);
  bool calculeDescriptors(ABSTRACT_DESCRIPTOR * p_descriptor, cv::Mat * descriptorsBase);
  void saveInfo();
  void resetListFlagCalculatedDescriptor(); //Sets the descriptor as not calculated for each image

  //Get functions
  QString getNameUser() const;
  int getId() const;
  int getNumberImages() const;
  int getIndexCurrentImageEditing() const;
  void getImage(cv::Mat & image, int index);
  void saveImageInIndex(cv::Mat & image, int index);
  void resetIndexCurrentImageEditing();
  int getImageEditingNext(cv::Mat & image); /*Returns the index of the current image being edited, if the user has no images, the return is -1 and the image is set to empty*/
  int getImageEditingBack(cv::Mat & image); /*Returns the index of the current image being edited, if the user has no images, the return is -1 and the image is set to empty*/
  QList < bool > getListFlagCalculatedDescriptor() const; //Returns the descriptor calculation flags

  friend class DATA_BASE;

};

class DATA_BASE: public QObject {
  Q_OBJECT

  QDir dir; //Directory where the information belonging to this database is stored

  QList < int > oldIdList; //List of old ids
  QStringList oldUsersList; //List of old folders
  QStringList newUsersList; //List of new folders

  QMap < int, USER * > usersList; //Here, users will be accumulated and associated with their own id
  QMap < int, QString > nameUsersList;
  int idNext; //The id that follows the maximum assigned id
  QList < int > idnotAssigned; //Ids less than the maximum assigned id

  QMap < int, USER * > editedUserList;
  /*Stores the list of users that have been edited, the following defines what is considered an edit:
  -A user has at least one image without its respective descriptor
  -Images are added to the user (either manually or via the program)
  -Images are deleted from the user (either manually or via the program)
  -An image is modified from the program (rescaled or cropped)
  -The creation of a new user (either manually or via the program)
  -The deletion of a user (either manually or via the program)
  */

  //Flags
  bool dataBaseIsLoad;
  bool flagOldDeletedUsers;
  bool flagNewUsersDeleted;
  bool flagDatabaseHasBeenModified; //Should be true when there has been an edit, see above to know when an edit should exist.

  public:
    DATA_BASE();
  DATA_BASE(QString path);
  ~DATA_BASE();
  bool loadDataBase(QString path);
  void clearDataBase();
  //_____Functions controlling ids___
  int getIdCorrect();
  void releaseId(int id);
  //______________________________________

  //____________Get functions_____________
  QStringList getListNameUsers() const;
  QMap < int, QString > getListNameUsersAndId() const;
  int getNumberUsers() const;
  int getIdOfUser(QString nameUser);
  USER * getUser(QString nameUser);
  QList < USER * > getAllUsers();
  QString getPath() const;
  bool existUser(const QString nameUser) const;
  bool dataBaseIsEdited() const;
  bool databaseHasBeenLoaded() const;
  //______________________________________

  void readOldUsers();
  void createOldList();
  void readNewUsers();
  void createNewList();
  bool createUser(QString nameUser);
  bool deleteUser(QString nameUser);
  void readIdNextAndIdnotAssigned();
  void saveInfo();
  void loadFlagEditing(); //Loads the editing flag status and sets it in flagDatabaseHasBeenModified
  void doneEditing(bool flag); //Stores in a file true if there has been an edit or false if not.
  void haveBeenEdited(USER * user); //This function is used to store the users who are pending descriptor calculation
  void resetListFlagCalculatedDescriptor(); //Sets the descriptor as not calculated for all images of all users
  signals:
    void haveBeenEdited(bool flag);

};

#endif
