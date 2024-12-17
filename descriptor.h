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

#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

//stl
#include <vector>
//openCV
namespace cv {
  class Mat;
};
//EIGEN
#include <eigen/Eigen/Dense>
#include <eigen/Eigen/Sparse>
//QT
#include <QDialog>
#include <QDebug>


class ABSTRACT_DESCRIPTOR: public QDialog {
  Q_OBJECT

  public:

    //______________________________________OPENCV INPUT INTERFACE________________________________________________________________________//
    /*This class does not assume an underlying data type, the implementation of the virtual functions will handle the data type of cv::Mat*/
    virtual cv::Mat * descriptor_base(const cv::Mat & img) = 0; /*Calculates the base descriptor, the returned descriptors should be in rows. If no descriptor is returned, NULL should be returned*/
  virtual void post_processing(cv::Mat & descriptor_base, cv::Mat & descriptor_end, std::vector < int > & ithRows) = 0; /*Applies a transformation (e.g., PCA, LDA, ICA, FOURIER, etc.) to a set of base descriptors. It also defines the data type of descriptor_end (only float is supported in the current implementation)*/
  //_______________________________________________________________________________________________________________________________________//

  //______________________________________EIGEN RETURN INTERFACE_________________________________________________________________________//
  virtual Eigen::MatrixXf * test(const cv::Mat & img) = 0; //Calculates the "final descriptor" (The term final descriptor refers to a base descriptor to which post-processing has been applied). 

  /*NOTES:
  -The return interface is due to the fact that the sparse solution algorithm is mostly implemented with the EIGEN library
  -The choice of float was made because precision is not as important for classification via sparse solution, and many algorithms can generate an excessive number of descriptors, so memory optimization is necessary*/
  //_______________________________________________________________________________________________________________________________________//

  virtual QString nameDescriptor() const = 0;

  ABSTRACT_DESCRIPTOR(): constructionWasSuccessful(false) {};
  virtual~ABSTRACT_DESCRIPTOR() = 0;

  virtual void loadSettings(QString path) = 0;
  virtual bool applySettings(QString path) = 0;

  bool constructionWasSuccessful; //Created for exception handling, not used in this implementation 

  signals:
    void editingDescriptor();

};

//The following classes are two simple test descriptors

class QLineEdit;

//Descriptor with post-processing
class DESCRIPTOR_TEST1: public ABSTRACT_DESCRIPTOR {

  Q_OBJECT
  //________________Graphical part____________________

  //QLineEdit
  QLineEdit * lineEditNewWidth;
  QLineEdit * lineEditNewHeight;

  //_________________________________________________

  int m;
  int n;
  cv::Mat * imgTemp;
  cv::Mat * filteredImage;
  const double sigma;
  cv::Mat * descriptorTemp;
  Eigen::MatrixXf * descriptorTest;
  int nr; //Useful parameter to simulate multiple descriptors
  public:
    DESCRIPTOR_TEST1(int mi, int ni, int nri);
  ~DESCRIPTOR_TEST1();

  //____________________________________________________Reimplementation of virtual functions___________________________________________//

  cv::Mat * descriptor_base(const cv::Mat & img);
  void post_processing(cv::Mat & descriptor_base, cv::Mat & descriptor_end, std::vector < int > & ithRows);
  Eigen::MatrixXf * test(const cv::Mat & img);
  QString nameDescriptor() const;
  void loadSettings(QString path);
  bool applySettings(QString path);
  //______________________________________________________________________________________________________________________________________//

  public slots:
    void edit_n();
  void edit_m();

};

//Descriptor without post-processing
class DESCRIPTOR_TEST2: public ABSTRACT_DESCRIPTOR {

  Q_OBJECT
  //________________Graphical part____________________

  //QLineEdit
  QLineEdit * lineEditNewWidth;
  QLineEdit * lineEditNewHeight;
  QLineEdit * lineEditNumberOfRepetitions;

  //_________________________________________________

  int m;
  int n;
  cv::Mat * imgTemp;
  cv::Mat * filteredImage;
  const double sigma;
  cv::Mat * descriptorTemp;
  Eigen::MatrixXf * descriptorTest;
  int nr; //Useful parameter to simulate multiple descriptors
  public:
    DESCRIPTOR_TEST2(int mi, int ni, int nri);
  ~DESCRIPTOR_TEST2();

  //____________________________________________________Reimplementation of virtual functions___________________________________________//

  cv::Mat * descriptor_base(const cv::Mat & img);
  void post_processing(cv::Mat & descriptor_base, cv::Mat & descriptor_end, std::vector < int > & ithRows);
  Eigen::MatrixXf * test(const cv::Mat & img);
  QString nameDescriptor() const;
  void loadSettings(QString path);
  bool applySettings(QString path);
  //______________________________________________________________________________________________________________________________________//

  public slots:
    void edit_n();
  void edit_m();
  void edit_nr();

};

#endif
