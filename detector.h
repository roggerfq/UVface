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

#ifndef DETECTOR_H
#define DETECTOR_H
//stl
#include <iostream>
#include<vector>
//________________OPENCV LIBRARIES___________________
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//______________________________________________________________________________________

/*
If the following flag is set to 1, the functions and classes related to evaluating a face detector 
using the software provided at http://vis-www.cs.umass.edu/fddb/ will be compiled.
*/
#define EVALUATION_FDDB 1

/*Forward declarations*/
class NODE_EVALUATION;
class TREE_TRAINING_EVALUATION;
class STRONG_LEARN_EVALUATION;
class CASCADE_CLASSIFIERS_EVALUATION;

typedef double(NODE_EVALUATION:: * PointerToEvaluationNode_Evaluation)(cv::Mat & , int scale, int orderDegrees);
class NODE_EVALUATION {

  CASCADE_CLASSIFIERS_EVALUATION * parentClassifier; //Through this pointer, we can access some data

  class NODE_EVALUATION * parent; //Points to the parent node, if it's the initial node its value should be zero
  class NODE_EVALUATION * nodeLeft; //Points to the left node, if there is no derived node, it should point to zero
  class NODE_EVALUATION * nodeRight; //Points to the right node, if there is no derived node, it should point to zero

  bool nodeIsTerminal;
  double threshold;
  int numFeature;
  cv::Point2i * feature;
  std::vector < std::vector < int * > > stackFeatures;
  double yt;

  PointerToEvaluationNode_Evaluation pointerToFunctionEvaluation; /*Pointer to the evaluation function*/

  void setNodeAsTerminal();
  public:
    /*NOTE: The NULL parameter in parentClassifier is because in the constructor of TREE_TRAINING_EVALUATION a pointer
    to this class is initially created*/
    NODE_EVALUATION(NODE_EVALUATION * myParent = NULL, CASCADE_CLASSIFIERS_EVALUATION * parentClassifier = NULL): parent(myParent), parentClassifier(parentClassifier), nodeLeft(NULL), nodeRight(NULL), feature(NULL), nodeIsTerminal(false) {
      pointerToFunctionEvaluation = & NODE_EVALUATION::evaluateNodeNoTerminal;
    }
    ~NODE_EVALUATION();
  void loadNode(cv::FileNode nodeRootFile);
  void initializeFeatures();
  double evaluateNode(cv::Mat & image, int scale, int orderDegrees);
  double evaluateNodeNoTerminal(cv::Mat & image, int scale, int orderDegrees); /*Evaluation function for non-terminal NODE*/
  double evaluationNodeTerminal(cv::Mat & image, int scale, int orderDegrees); /*Evaluation function for terminal node*/

};

class TREE_TRAINING_EVALUATION {

  CASCADE_CLASSIFIERS_EVALUATION * parentClassifier; //Through this pointer, we can access some data

  NODE_EVALUATION * nodeRoot;

  public:
    TREE_TRAINING_EVALUATION(CASCADE_CLASSIFIERS_EVALUATION * parentClassifier): nodeRoot(NULL), parentClassifier(parentClassifier) {}
    ~TREE_TRAINING_EVALUATION();
  void loadWeakLearn(cv::FileNode weakLearnsTrees, int num);
  void initializeFeatures();
  double evaluateTree(cv::Mat & image, int scale, int orderDegrees);

};

class STRONG_LEARN_EVALUATION {

  CASCADE_CLASSIFIERS_EVALUATION * parentClassifier; //Through this pointer, we can access some data

  std::vector < TREE_TRAINING_EVALUATION * > weakLearns;
  double threshold;
  public:
    STRONG_LEARN_EVALUATION(CASCADE_CLASSIFIERS_EVALUATION * parentClassifier): parentClassifier(parentClassifier) {};
  ~STRONG_LEARN_EVALUATION();
  void loadStrongLearn(cv::FileNode fileStrongLearn, int stage);
  double evaluateStrongLearnWithZeroThreshold(cv::Mat & image, int scale, int orderDegrees);
  bool evaluateStrongLearn(cv::Mat & image, int scale, int orderDegrees); /*Evaluation using threshold*/
  void initializeFeatures();

  #if EVALUATION_FDDB == 1
  double scoreDetection; //scoreDetection is the confidence of the detection
  bool FDDB_evaluateStrongLearn(cv::Mat & image, int scale, int orderDegrees);
  #endif

};

class CASCADE_CLASSIFIERS_EVALUATION {
  std::vector < STRONG_LEARN_EVALUATION * > strongLearnsEvaluation; /*Stores each of the strong classifiers in the cascade*/
  int numberClassifiersUsed; /*Number of classifiers to use, its value should vary between 1 and strongLearnsEvaluation.size(), by default its value is strongLearnsEvaluation.size()*/
  int widthImages;
  int highImages;
  cv::Point2i * NPD;
  std::vector < double > degrees; //These are the detection degrees, by default 0 degrees
  int sizeBaseEvaluation; //By default, its size is the smallest side of the image size
  double factorScaleWindow; //Factor by which the window will widen
  double stepWindow; //Factor by which the window will move according to its size
  double sizeMaxWindow; //Maximum size of the search window

  //Rectangle parameters for displaying detection
  int lineThicknessRectangles; //Thickness of the lines drawing the rectangles (default is 1)
  cv::Scalar colorRectangles; //By default, the color is green
  int groupThreshold; //Minimum threshold after which a group of similar rectangles is not removed
  double eps; //Relative minimum difference in the sides of two rectangles, beyond which they are grouped
  bool flagActivateSkinColor;
  /*If this variable is true, a simple skin color algorithm is activated, by default it is false.
  Keep in mind that if you are detecting something other than a human face, or if the detector will face very varying lighting conditions, or if the input image is in grayscale, the detection will be affected by skin color as areas outside the human skin color range will be removed.*/
  bool flagExtractColorImages; /*Controls whether the detected images are returned in color (from the input image) or in grayscale, by default it is false, meaning the images are returned in grayscale*/

  /*Important variables during execution*/
  std::vector < cv::RotatedRect > windowsCandidatesRotated;
  std::vector < cv::Rect > windowsCandidates;
  cv::Mat ImageBackground; //Extra background image to avoid errors when evaluating features that go beyond image borders
  int zsBackground; //Width that the image to analyze must have to avoid evaluating features at non-existent coordinates
  cv::Size szImg; //Width of the last analyzed image
  cv::Mat imageAndBackgroundColor; //Only useful for detectObjectRectanglesRotatedGrouped function
  cv::Mat imageAndBackgroundGray; //Only useful for detectObjectRectanglesRotatedGrouped function
  cv::Mat imageGray; /*This matrix will be used by each high-level detection function (including those belonging to the FDDB database evaluation) to embed the much smaller input image (see zsBackground variable) to avoid memory access errors when analyzing rectangles (sliding window) at the image edges*/
  cv::Mat hsv; //Skin Color
  cv::Mat bw; //Skin Color
  cv::Mat integralBw; //Integral image of bw
  cv::Scalar hsvMin; //Minimum value in the hsv space accepted as skin color
  cv::Scalar hsvMax; //Maximum value in the hsv space accepted as skin color

  cv::FileStorage * fileCascadeClassifier;
  void generateFeatures();
  void loadCascadeClasifier();
  public:
    CASCADE_CLASSIFIERS_EVALUATION(std::string nameFile);
  ~CASCADE_CLASSIFIERS_EVALUATION();

  void initializeFeatures();
  //___If any of the following functions are called, initializeFeatures() should be called to make the change take effect____
  void setDegreesDetections(std::vector < double > myDegrees); //Sets the window search degrees
  void setSizeBase(int sizeBase); //Sets the smallest size for the search window
  void setFactorScaleWindow(double factorScale); //Sets the factor by which the search window will widen
  void setStepWindow(double factorStep); //Sets the factor by which the window will move according to its size
  void setSizeMaxWindow(double maxSize); //Sets the maximum size of the search window
  //______________________________________________________________________________________________________________________________
  void setLineThicknessRectangles(int thicknessRectangles);
  void setColorRectangles(const cv::Scalar & color);

  /*
  In general, the following should be considered when setting the parameters groupThreshold and eps:
  1-If the detector shows few adjacent detections, i.e., there are few neighboring detections over the same object, 
  groupThreshold should be set to a small value to ensure that detections with only one rectangle are not removed. 
  The minimum value that groupThreshold can take is 1. Due to this, in some detection functions, the list is internally doubled.
  2-The eps parameter generally works well for values between 0.2 and 0.5. It controls the relative difference in the sides of two rectangles beyond which they are declared as neighbors.
  */
  void setGroupThreshold(int threshold);
  void setEps(double myEps);

  void setFlagActivateSkinColor(bool activateSkinColor);
  void setFlagExtractColorImages(bool extractColorImages);
  void setNumberClassifiersUsed(int number); //Sets the number of classifiers to use
  void setHsvMin(const cv::Scalar & hsv);
  void setHsvMax(const cv::Scalar & hsv);

  //________Get functions_______________//
  int getNumberStrongLearns() const; //Returns the total number of strong classifiers in the cascade
  double getSizeMaxWindow() const; //Returns the maximum size of the search window
  cv::Scalar getHsvMin() const;
  cv::Scalar getHsvMax() const;

  //____________________________________//

  //The following functions are the lowest-level detection functions
  bool evaluateClassifier(cv::Mat & image, int scale, int orderDegrees);
  bool evaluateClassifier(cv::Mat & image, int scale, int orderDegrees, int begin, int end);

  //______________Here are some useful detection functions___________________//
  /*The following function detects an object (with the option for skin color) but does not group similar rectangles*/
  void detectObjectRectanglesUngrouped(cv::Mat & image);
  /*The following function has the option to double the number of detections in an internal list before grouping the rectangles. This is because the OpenCV algorithm cv::groupRectangles removes rectangles whose group contains fewer or equal to groupThreshold. This is an issue when the classifier has a low false positive rate, as detections are very localized and tend to have few neighbors. This problem may also occur when detection is done from a single angle, so doubling the list prevents many correct detections from being removed by the algorithm.*/
  void detectObjectRectanglesGroupedZeroDegrees(cv::Mat & image, std::vector < cv::Mat > * listDetectedObjects = NULL, std::vector < cv::Rect > * coordinatesDetectedObjects = NULL, bool doubleDetectedList = true, bool paintDetections = true);
  /*The following function detects the object at the set angles and may optionally extract those regions from the image, normalize them to the standard training angle, and return them in the list std::vector<cv::Mat> listDetectedObjects */
  void detectObjectRectanglesRotatedGrouped(cv::Mat & image, std::vector < cv::Mat > * listDetectedObjects = NULL, std::vector < cv::RotatedRect > * coordinatesDetectedObjects = NULL, bool paintDetections = true);
  //_______________________________________________________________________________________________________________//

  /*The following methods and variables are only useful for my undergraduate thesis. These methods are responsible for extracting rectangles with the score of each detection and will be used by the software provided by the FDDB database (http://vis-www.cs.umass.edu/fddb/). For clarity, each function or class related to the following functions and variables will start with the prefix FDDB.*/
  #if EVALUATION_FDDB == 1
  //The following function is used for the FDDB database evaluation
  double scoreDetection; //scoreDetection is the confidence of the detection
  bool FDDB_evaluateClassifier(cv::Mat & image, int scale, int orderDegrees);
  void FDDB_detectObjectRectanglesGroupedZeroDegrees(cv::Mat & image, std::vector < cv::Rect > & rectanglesDetected, std::vector < double > & scores, bool doubleDetectedList = true);
  #endif

  friend class STRONG_LEARN_EVALUATION;
  friend class TREE_TRAINING_EVALUATION;
  friend class NODE_EVALUATION;

};

#endif
