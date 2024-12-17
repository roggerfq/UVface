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

#ifndef GTP_H
#define GTP_H

//Custom
#include "descriptor.h" //ABSTRACT DESCRIPTOR
//STL
#include <fstream>      // std::ifstream
//openCV
#include <opencv2/opencv.hpp>

class QCheckBox;

class GTP: public ABSTRACT_DESCRIPTOR {

  const int maxFeatures; //Maximum number of expected features
  const int szRect; //Length of the sides of the rectangle for the resulting images from usable ellipses
  const int szHistlow; //Length of the dimension to which the imgHist data will be reduced
  QString pathDataBase; //Path corresponding to the database

  std::ifstream fp; //File containing the ellipses

  double ** vec_dp; //Here the parameters for each ellipse will be stored
  int ARect; //Area of the rectangle for the resulting images from usable ellipses
  double rc; //Radius of the circle after applying the affine transformation to each ellipse
  int centerRect; //Center of a rectangle with sides szRect.
  cv::Mat * imagesRect; //Will store the images resulting from the ellipses
  cv::Mat * imgZscore; //Will store the normalization (zScoreNormalization(cv::Mat *imageSrc,cv::Mat *imageDest)) of images stored in imagesRect
  cv::Mat * imgGtp; /*Stores each image after convolving with the Gabor filter at different angles and combining them in a ternary pattern*/
  int dHist; //Histogram dimension
  cv::Mat * imgHist; /*Will store the set of histograms calculated for each image*/
  cv::Mat * imgHIstTemp; //Matrix returned during each iteration (variable size)
  Eigen::MatrixXf * descriptorTest; //Matrix returned for a test image (To be analyzed using sparse solution)
  //cv::Mat *imgHistlow; //Stores projections of imgHist over the first szHistlow principal components

  double minimumSizeAxisEllipses; //Minimum acceptable size for the axes of each ellipse
  int numberUsefulFeatures;

  //The following functions are part of the private implementation and are used to complete the GTP descriptor procedures
  void zScoreNormalization(const cv::Mat * const imageSrc, cv::Mat * imageDest);

  //___________Variables related to the Gabor filter and ternary pattern________________________//
  int szGk; //Kernel size
  int numberOrientations;
  int * pow3; //Stores the first numberOrientations powers of 3 starting from 0
  cv::Mat * gaborKernel; //Vector for the kernel at each different orientation: 0°, 45°, 90°, and 135°
  void constructGaborKernels();
  double ut; //Threshold for the ternary pattern 
  void gtp(cv::Mat * img, cv::Mat * ltp); //Convolves the imaginary part of the Gabor kernel with img at the corresponding angles and combines the results in a ternary pattern
  //_____________________________________________________________________________________________________//
  void histogram(cv::Mat * img, int row); //Builds a histogram by partitioning each image imgGtp into subcells
  cv::PCA * pca; //This class will compute and store the principal components

  public:
    GTP();
  ~GTP();

  void setGui(); //Sets the graphical interface for GTP

  //____________________________________________________Reimplementation of virtual functions___________________________________________//

  cv::Mat * descriptor_base(const cv::Mat & img);
  void post_processing(cv::Mat & descriptor_base, cv::Mat & descriptor_end, std::vector < int > & ithRows);
  Eigen::MatrixXf * test(const cv::Mat & img);
  QString nameDescriptor() const;
  void loadSettings(QString path);
  bool applySettings(QString path);
  //______________________________________________________________________________________________________________________________________//

  //These two functions allow us to store and retrieve the previously calculated PCA
  void savePca(); //Stores all information related to PCA
  void LoadPca(); //Retrieves all information related to PCA

  void drawEllipses(cv::Mat & img);
  //Setter functions
  void set_minimumSizeAxisEllipses(const double & minAxis);

  //Getter functions
  double get_minimumSizeAxisEllipses() const;
  int getNumberUsefulFeatures() const;

  //Testing functions
  void seeImagesRect();

  //__________________________GRAPHIC PART_______________________________//
  QCheckBox * checkBox_PCA;

};

#endif
