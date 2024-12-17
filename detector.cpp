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

#include <detector.h>

/*
//________________OPEN CV LIBRARIES___________________
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp" 
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//___________________________________________________
*/

//________________________GLOBAL VARIABLES FOR THE FILE______________________________________________//

static
const int pixelResolution = 256; // Pixel resolution, this depends on the possible values for evaluating an NPD

static float NPD_VALUES_FOR_EVALUATION[pixelResolution][pixelResolution]; //LOCK-TABLE suitable for evaluation

bool generateLockTableEvaluationDetector() // Here we fill NPD_VALUES_FOR_EVALUATION
{

  float npd_temp;
  for (int x = 0; x < pixelResolution; x++) {
    for (int y = 0; y < pixelResolution; y++) {
      npd_temp = (float(x - y)) / (x + y);
      if (npd_temp != npd_temp) npd_temp = 0; // detecting NaN values
      NPD_VALUES_FOR_EVALUATION[x][y] = npd_temp;
    }
  }

  return true;
}

static bool flagFill = generateLockTableEvaluationDetector(); // This ensures that NPD_VALUES_FOR_EVALUATION is filled when the program starts

//____________________________________________________________________________________________________//

class WINDOWING {
  cv::Point2f center;
  double degree;
  int scale;
  public:
    WINDOWING(cv::Point2f c, int s, double degree): center(c), scale(s), degree(degree) {}

};

NODE_EVALUATION::~NODE_EVALUATION() {

  if (!nodeIsTerminal) {

    //______________Stack where rotated and scaled features are stored_______________//
    if (!stackFeatures.empty()) {
      for (int i = 0; i < stackFeatures.size(); i++) {

        for (int j = 0; j < stackFeatures[i].size(); j++) {
          delete[] stackFeatures[i][j];
        }

      }
      stackFeatures.clear();
    }
    //____________________________________________________________________________________________//

    delete feature;
    delete nodeLeft;
    delete nodeRight;
  }

}

void NODE_EVALUATION::setNodeAsTerminal() {
  pointerToFunctionEvaluation = & NODE_EVALUATION::evaluationNodeTerminal;
  nodeIsTerminal = true;
}

void NODE_EVALUATION::loadNode(cv::FileNode nodeRootFile) {

  cv::FileNode information = nodeRootFile["information"];
  nodeIsTerminal = (bool)(int) information["nodeIsTerminal"];
  threshold = (double) information["threshold"];
  numFeature = (int) information["numFeature"];

  if (!nodeIsTerminal) {
    feature = new cv::Point2i;
    * feature = parentClassifier -> NPD[numFeature];
  }

  if (nodeIsTerminal) setNodeAsTerminal(); /*This must be called if the node is terminal*/

  yt = (double) information["yt"];

  /*_________________APPLYING RECURSION__________________________*/
  if (!nodeIsTerminal) {

    cv::FileNode nodeLeftFile = nodeRootFile["nodeLeft"];
    nodeLeft = new NODE_EVALUATION(this, parentClassifier);
    nodeLeft -> loadNode(nodeLeftFile);

    cv::FileNode nodeRightFile = nodeRootFile["nodeRight"];
    nodeRight = new NODE_EVALUATION(this, parentClassifier);
    nodeRight -> loadNode(nodeRightFile);

  }
  /*____________________________________________________________________*/

}

void NODE_EVALUATION::initializeFeatures() {

  if (!nodeIsTerminal) {

    //______________Clear the stack, this is in case the established parameters need to be changed_______________//
    if (!stackFeatures.empty()) {
      for (int i = 0; i < stackFeatures.size(); i++) {

        for (int j = 0; j < stackFeatures[i].size(); j++) {
          delete[] stackFeatures[i][j];
        }

      }
      stackFeatures.clear();
    }
    stackFeatures.reserve(parentClassifier -> degrees.size()); // Introduced on October 5, 2015
    //________________________________________________________________________________________________________________//

    int x1 = feature -> x % parentClassifier -> widthImages;
    int y1 = feature -> x / parentClassifier -> widthImages;

    int x2 = feature -> y % parentClassifier -> widthImages;
    int y2 = feature -> y / parentClassifier -> widthImages;

    for (int i = 0; i < parentClassifier -> degrees.size(); i++) {

      /*____________ROTATION MATRIX__________________*/
      cv::Mat MR(2, 2, cv::DataType < double > ::type);
      MR.at < double > (0, 0) = std::cos(parentClassifier -> degrees[i] * (M_PI / 180));
      MR.at < double > (0, 1) = std::sin(parentClassifier -> degrees[i] * (M_PI / 180));
      MR.at < double > (1, 0) = -std::sin(parentClassifier -> degrees[i] * (M_PI / 180));
      MR.at < double > (1, 1) = std::cos(parentClassifier -> degrees[i] * (M_PI / 180));
      /*________________________________________________*/

      cv::Mat V1(2, 1, cv::DataType < double > ::type);
      V1.at < double > (0, 0) = (x1 - (parentClassifier -> widthImages / 2));
      V1.at < double > (1, 0) = ((parentClassifier -> highImages / 2) - y1);

      cv::Mat V2(2, 1, cv::DataType < double > ::type);
      V2.at < double > (0, 0) = (x2 - (parentClassifier -> widthImages / 2));
      V2.at < double > (1, 0) = ((parentClassifier -> highImages / 2) - y2);

      V1 = MR.t() * V1;
      V2 = MR.t() * V2;
      /*____________0.5 is for numerical error________________*/
      V1.at < double > (0, 0) = (int)(V1.at < double > (0, 0) + (parentClassifier -> widthImages / 2) + 0.5);
      V1.at < double > (1, 0) = (int)((parentClassifier -> highImages / 2) - V1.at < double > (1, 0) + 0.5);
      V2.at < double > (0, 0) = (int)(V2.at < double > (0, 0) + (parentClassifier -> widthImages / 2) + 0.5);
      V2.at < double > (1, 0) = (int)((parentClassifier -> highImages / 2) - V2.at < double > (1, 0) + 0.5);
      /*____________________________________________________________*/

      std::vector < int * > stackTemp;
      //___________________Introduced on October 5, 2015____________________________//
      int count = 0;
      for (int sizeBase = parentClassifier -> sizeBaseEvaluation; sizeBase <= parentClassifier -> sizeMaxWindow; sizeBase = (parentClassifier -> factorScaleWindow) * sizeBase) {
        count++;
      }
      //____________________________________________________________________________________//
      stackTemp.reserve(count);

      for (int sizeBase = parentClassifier -> sizeBaseEvaluation; sizeBase <= parentClassifier -> sizeMaxWindow; sizeBase = (parentClassifier -> factorScaleWindow) * sizeBase) {

        int ky = (sizeBase / parentClassifier -> highImages), kx = (sizeBase / parentClassifier -> widthImages);

        int * vecTemp = new int[4];
        vecTemp[0] = ky * V1.at < double > (1, 0);
        vecTemp[1] = kx * V1.at < double > (0, 0);
        vecTemp[2] = ky * V2.at < double > (1, 0);
        vecTemp[3] = kx * V2.at < double > (0, 0);

        stackTemp.push_back(vecTemp);

      }

      stackFeatures.push_back(stackTemp);

    }

    nodeLeft -> initializeFeatures();
    nodeRight -> initializeFeatures();
  }

}

double NODE_EVALUATION::evaluateNode(cv::Mat & image, int scale, int orderDegrees) {
  return (this->*pointerToFunctionEvaluation)(image, scale, orderDegrees);
}

double NODE_EVALUATION::evaluateNodeNoTerminal(cv::Mat & image, int scale, int orderDegrees) {

  int * vecTemp = (stackFeatures[orderDegrees])[scale];

  int p1 = image.at < uchar > (vecTemp[0], vecTemp[1]);
  int p2 = image.at < uchar > (vecTemp[2], vecTemp[3]);

  /*
  int delta=1;
  int p1=(image.at<uchar>(vecTemp[0],vecTemp[1])+image.at<uchar>(vecTemp[0]+delta*scale,vecTemp[1])+image.at<uchar>(vecTemp[0],vecTemp[1]+delta*scale)+image.at<uchar>(vecTemp[0]-delta*scale,vecTemp[1])+image.at<uchar>(vecTemp[0],vecTemp[1]-delta*scale))/5;
  int p2=(image.at<uchar>(vecTemp[2],vecTemp[3])+image.at<uchar>(vecTemp[2]+delta*scale,vecTemp[3])+image.at<uchar>(vecTemp[2],vecTemp[3]+delta*scale)+image.at<uchar>(vecTemp[2]-delta*scale,vecTemp[3])+image.at<uchar>(vecTemp[2],vecTemp[3]-delta*scale))/5;
  */

  if (NPD_VALUES_FOR_EVALUATION[p1][p2] > threshold) return nodeRight -> evaluateNode(image, scale, orderDegrees); /*Send to the right node*/
  return nodeLeft -> evaluateNode(image, scale, orderDegrees); /*Send to the left node*/

}

double NODE_EVALUATION::evaluationNodeTerminal(cv::Mat & image, int scale, int orderDegrees) {
  //std::cout << "At node=" << yt << "\n"; // Debug message to show the value of yt at this node
  return yt;
}

TREE_TRAINING_EVALUATION::~TREE_TRAINING_EVALUATION() {

  delete nodeRoot; // The root node is deleted, which will call the destructors of its child nodes in sequence
}

void TREE_TRAINING_EVALUATION::initializeFeatures() {
  nodeRoot -> initializeFeatures();
}

void TREE_TRAINING_EVALUATION::loadWeakLearn(cv::FileNode weakLearnsTrees, int num) {

  std::string str_tree;
  std::stringstream sstm;
  sstm << "tree_" << num;
  str_tree = sstm.str();

  cv::FileNode tree = weakLearnsTrees[str_tree];
  cv::FileNode information = tree["information"];

  cv::FileNode nodeRootFile = tree["nodeRoot"];
  nodeRoot = new NODE_EVALUATION(NULL, parentClassifier);
  nodeRoot -> loadNode(nodeRootFile);

}

double TREE_TRAINING_EVALUATION::evaluateTree(cv::Mat & image, int scale, int orderDegrees) {

  return nodeRoot -> evaluateNode(image, scale, orderDegrees);
}

STRONG_LEARN_EVALUATION::~STRONG_LEARN_EVALUATION() {

  for (int i = 0; i < weakLearns.size(); i++)
    delete weakLearns[i];
  weakLearns.clear();

}

void STRONG_LEARN_EVALUATION::initializeFeatures() {

  for (int i = 0; i < weakLearns.size(); i++)
    weakLearns[i] -> initializeFeatures();

}

void STRONG_LEARN_EVALUATION::loadStrongLearn(cv::FileNode fileStrongLearn, int stage) {

  std::string str_stage;
  std::stringstream sstm;
  sstm << "stage_" << stage;
  str_stage = sstm.str();

  cv::FileNode strongLearn = fileStrongLearn[str_stage];
  cv::FileNode information = strongLearn["information"];
  threshold = (double) information["threshold"];

  /*Next, each weakLearn or tree is loaded*/
  cv::FileNode weakLearnsTrees = strongLearn["weakLearns"];
  weakLearns.reserve(weakLearnsTrees.size());

  for (int i = 0; i < weakLearnsTrees.size(); i++) {
    TREE_TRAINING_EVALUATION * treeAux = new TREE_TRAINING_EVALUATION(parentClassifier); // Linked to the parent classifier
    treeAux -> loadWeakLearn(weakLearnsTrees, i); /*Here each classifier is loaded*/
    weakLearns.push_back(treeAux);
  }

  std::cout << "Strong classifier number=" << stage << "\n"; // Debug message showing the current strong classifier number
}

double STRONG_LEARN_EVALUATION::evaluateStrongLearnWithZeroThreshold(cv::Mat & image, int scale, int orderDegrees) {

  double evaluation = 0;

  for (int i = 0; i < weakLearns.size(); i++)
    evaluation = evaluation + weakLearns[i] -> evaluateTree(image, scale, orderDegrees);

  return evaluation;
}

bool STRONG_LEARN_EVALUATION::evaluateStrongLearn(cv::Mat & image, int scale, int orderDegrees) {

  if (evaluateStrongLearnWithZeroThreshold(image, scale, orderDegrees) >= threshold)
    return true; /*Classified as positive*/
  else
    return false; /*Classified as negative*/

}

#if EVALUATION_FDDB == 1

bool STRONG_LEARN_EVALUATION::FDDB_evaluateStrongLearn(cv::Mat & image, int scale, int orderDegrees) {

  scoreDetection = evaluateStrongLearnWithZeroThreshold(image, scale, orderDegrees);

  if (scoreDetection >= threshold)
    return true; /*Classified as positive*/
  else
    return false; /*Classified as negative*/

}

#endif

CASCADE_CLASSIFIERS_EVALUATION::CASCADE_CLASSIFIERS_EVALUATION(std::string nameFile): NPD(NULL) {

  fileCascadeClassifier = new cv::FileStorage(nameFile, cv::FileStorage::READ);
  loadCascadeClasifier();

  //______________THIS SHOULD BE DONE AFTER LOADING THE FULL CLASSIFIER_____________________
  degrees.push_back(0); // Zero degrees
  factorScaleWindow = 1.2; // 1.2 was the value that worked best in tests
  stepWindow = 0.2; // 0.2 was the value that worked best in tests
  sizeMaxWindow = 640; // Generally, images won't exceed this size; for larger sizes, modify it
  initializeFeatures(); // Features are initialized with the previous parameters
  //_______________________________________________________________________________________________

  //______Other default parameters__________//
  lineThicknessRectangles = 1;
  colorRectangles = cv::Scalar(0, 255, 0); // Default color is green
  groupThreshold = 1;
  eps = 0.2;
  flagActivateSkinColor = false; // By default, skin color is not activated
  hsvMin = cv::Scalar(0, 10, 60); // Minimum value, works very well
  hsvMax = cv::Scalar(20, 150, 255); // Maximum value, works very well
  flagExtractColorImages = false; // By default, detected regions are returned in grayscale

  numberClassifiersUsed = strongLearnsEvaluation.size();

  //____________________________________________//

}

CASCADE_CLASSIFIERS_EVALUATION::~CASCADE_CLASSIFIERS_EVALUATION() {

  // We delete the dynamic memory for the vector that stores the NPD
  if (NPD != NULL) {
    delete[] NPD;
    NPD = NULL;
  }

  // We delete the dynamic memory for the strongLearns
  for (int i = 0; i < strongLearnsEvaluation.size(); i++)
    delete strongLearnsEvaluation[i];
  strongLearnsEvaluation.clear();

}

void CASCADE_CLASSIFIERS_EVALUATION::initializeFeatures() {

  for (int i = 0; i < strongLearnsEvaluation.size(); i++)
    strongLearnsEvaluation[i] -> initializeFeatures();

  zsBackground = 0.8 * double(sizeMaxWindow);
  szImg = cv::Size(0, 0);

}

void CASCADE_CLASSIFIERS_EVALUATION::setDegreesDetections(std::vector < double > myDegrees) {
  degrees = myDegrees;
}

void CASCADE_CLASSIFIERS_EVALUATION::setSizeBase(int sizeBase) {
  sizeBaseEvaluation = sizeBase;
}

void CASCADE_CLASSIFIERS_EVALUATION::setFactorScaleWindow(double factorScale) {
  factorScaleWindow = factorScale;
}

void CASCADE_CLASSIFIERS_EVALUATION::setStepWindow(double factorStep) {
  stepWindow = factorStep;
}

void CASCADE_CLASSIFIERS_EVALUATION::setSizeMaxWindow(double maxSize) {
  sizeMaxWindow = maxSize;
}

void CASCADE_CLASSIFIERS_EVALUATION::setLineThicknessRectangles(int thicknessRectangles) {

  lineThicknessRectangles = thicknessRectangles;

}

void CASCADE_CLASSIFIERS_EVALUATION::setColorRectangles(const cv::Scalar & color) {

  colorRectangles = color;

}

void CASCADE_CLASSIFIERS_EVALUATION::setGroupThreshold(int threshold) {
  groupThreshold = threshold;
}

void CASCADE_CLASSIFIERS_EVALUATION::setEps(double myEps) {
  eps = myEps;
}

void CASCADE_CLASSIFIERS_EVALUATION::setFlagActivateSkinColor(bool activateSkinColor) {
  flagActivateSkinColor = activateSkinColor;
}

void CASCADE_CLASSIFIERS_EVALUATION::setFlagExtractColorImages(bool extractColorImages) {
  flagExtractColorImages = extractColorImages;
}

void CASCADE_CLASSIFIERS_EVALUATION::setNumberClassifiersUsed(int number) {

  if (number < 1)
    numberClassifiersUsed = 1;
  else if (number > strongLearnsEvaluation.size())
    numberClassifiersUsed = strongLearnsEvaluation.size();
  else
    numberClassifiersUsed = number;

}

void CASCADE_CLASSIFIERS_EVALUATION::setHsvMin(const cv::Scalar & hsv) {
  hsvMin = hsv;
}

void CASCADE_CLASSIFIERS_EVALUATION::setHsvMax(const cv::Scalar & hsv) {
  hsvMax = hsv;
}

int CASCADE_CLASSIFIERS_EVALUATION::getNumberStrongLearns() const {
  return strongLearnsEvaluation.size();
}

double CASCADE_CLASSIFIERS_EVALUATION::getSizeMaxWindow() const {
  return sizeMaxWindow;
}

cv::Scalar CASCADE_CLASSIFIERS_EVALUATION::getHsvMin() const {
  return hsvMin;
}

cv::Scalar CASCADE_CLASSIFIERS_EVALUATION::getHsvMax() const {
  return hsvMax;
}

void CASCADE_CLASSIFIERS_EVALUATION::generateFeatures() {

  int p = widthImages * highImages;
  int numberFeature = p * (p - 1) / 2; // This is the result of summing (p-1)+(p-2)....+1

  if (NPD != NULL) {
    delete[] NPD; /* This will free memory if a double call occurs, because the base width and height might change */
    NPD = NULL;
  }

  NPD = new cv::Point2i[numberFeature]; // Requesting space to store features
  // ____ Filling with the values of each coordinate ___
  int count = 0;
  for (int i = 0; i < p; i++) {
    for (int j = i + 1; j < p; j++) {
      NPD[count] = cv::Point2i(i, j); /* For the calculation, apply (xi-xj)/(xi+xj), where xi and xj are the pixels at coordinates i and j respectively. */
      count++;
    }
  }

}

void CASCADE_CLASSIFIERS_EVALUATION::loadCascadeClasifier() {

  /* ____________________ Here begins the classifier reading ________________________ */

  cv::FileNode information = (*fileCascadeClassifier)["information"];

  /*__________ HERE IMPORTANT VARIABLES ARE UPDATED __________*/

  widthImages = (int) information["G_WIDTH_IMAGE"];
  highImages = (int) information["G_HEIGHT_IMAGE"];
  sizeBaseEvaluation = std::min(widthImages, highImages); // By default, its size is the smaller side of the image size
  generateFeatures(); // Must be called after widthImages and highImages are assigned

  /*_______________________________________________________________*/

  /* Next, each strong learn is loaded */
  cv::FileNode cascade_classifiers = (*fileCascadeClassifier)["cascade_classifiers"];
  strongLearnsEvaluation.reserve(cascade_classifiers.size());

  for (int i = 0; i < cascade_classifiers.size(); i++) {
    STRONG_LEARN_EVALUATION* strongLearnAux = new STRONG_LEARN_EVALUATION(this);
    strongLearnAux->loadStrongLearn(cascade_classifiers, i); /* Here each classifier is loaded */
    strongLearnsEvaluation.push_back(strongLearnAux);
  }

  fileCascadeClassifier->release();
  if (fileCascadeClassifier != NULL) delete fileCascadeClassifier;

}

bool CASCADE_CLASSIFIERS_EVALUATION::evaluateClassifier(cv::Mat& image, int scale, int orderDegrees) {

  for (int i = 0; i < numberClassifiersUsed; i++)
    if (!strongLearnsEvaluation[i]->evaluateStrongLearn(image, scale, orderDegrees)) return false; /* Classified as negative label */

  return true; /* Classified as positive label */

}

bool CASCADE_CLASSIFIERS_EVALUATION::evaluateClassifier(cv::Mat& image, int scale, int orderDegrees, int begin, int end) {

  for (int i = begin; i < end; i++)
    if (!strongLearnsEvaluation[i]->evaluateStrongLearn(image, scale, orderDegrees)) return false; /* Classified as negative label */

  scoreDetection = strongLearnsEvaluation[end - 1]->scoreDetection;

  return true; /* Classified as positive label */

}

// __________________________________ DECLARING DIFFERENT DETECTION FUNCTIONS __________________________________

void CASCADE_CLASSIFIERS_EVALUATION::detectObjectRectanglesUngrouped(cv::Mat& image) {

  if (szImg != image.size()) {
    ImageBackground = cv::Mat::zeros(sizeMaxWindow + 2 * zsBackground, sizeMaxWindow + 2 * zsBackground, CV_8UC1);
    imageGray = ImageBackground(cv::Range(zsBackground, zsBackground + image.rows), cv::Range(zsBackground, zsBackground + image.cols));
    szImg = image.size();
  }

  cvtColor(image, imageGray, CV_BGR2GRAY); // Converting to grayscale

  if (flagActivateSkinColor) {
    cvtColor(image, hsv, CV_BGR2HSV);
    inRange(hsv, hsvMin, hsvMax, bw);
    integral(bw, integralBw, CV_32S);
  }

  int idx = 0;

  if (flagActivateSkinColor) {
    //______________________________________________________________________________________________________________________//
    for (int sizeBase = sizeBaseEvaluation; sizeBase <= std::min(image.rows, image.cols); sizeBase = factorScaleWindow * sizeBase, idx++) {
      int step_x = sizeBase * stepWindow; /* In this factor, the search window will move in rows */
      int step_y = sizeBase * stepWindow; /* In this factor, the search window will move in columns */
      for (int i = 0; i <= image.rows - sizeBase; i = i + step_x) {
        for (int j = 0; j <= image.cols - sizeBase; j = j + step_y) {

          /*__________ CALCULATING WINDOW POSITION __________*/
          int topLeft_x = i;
          int topLeft_y = j;
          int bottomRight_x = sizeBase + i - 1;
          int bottomRight_y = sizeBase + j - 1;
          /*____________________________________________________________*/

          cv::Mat window;
          window = imageGray(cv::Range(topLeft_x, bottomRight_x + 1), cv::Range(topLeft_y, bottomRight_y + 1));

          bool flagDetected = false;
          double myDegree = 0;
          int num = 0;

          double sum2 = integralBw.at<int>(bottomRight_x + 1, bottomRight_y + 1) + integralBw.at<int>(topLeft_x, topLeft_y) - (integralBw.at<int>(bottomRight_x + 1, topLeft_y) + integralBw.at<int>(topLeft_x, bottomRight_y + 1)); // Integral image evaluation

          if (sum2 > 76.5 * sizeBase * sizeBase) { // 76.5 = 255*0.3

            for (int orderDegrees = 0; orderDegrees < degrees.size(); orderDegrees++) {
              if (evaluateClassifier(window, idx, orderDegrees)) {

                myDegree = myDegree + degrees[orderDegrees];
                flagDetected = true;
                num++;

              }
            }

          }

          if (flagDetected == true) {

            myDegree = myDegree / num;
            cv::RotatedRect rectRotate(cv::Point2f(j + (sizeBase / 2), i + (sizeBase / 2)), cv::Size2f(sizeBase, sizeBase), -myDegree);
            cv::Point2f vertices[4];
            rectRotate.points(vertices);
            for (int i = 0; i < 4; i++)
              cv::line(image, vertices[i], vertices[(i + 1) % 4], colorRectangles, lineThicknessRectangles);

          }

        }
      }

    }
    //_______________________________________________________________________________________________________________________//
  } else {
    //_______________________________________________________________________________________________________________________//
    for (int sizeBase = sizeBaseEvaluation; sizeBase <= std::min(image.rows, image.cols); sizeBase = factorScaleWindow * sizeBase, idx++) {
      int step_x = sizeBase * stepWindow; /* In this factor, the search window will move in rows */
      int step_y = sizeBase * stepWindow; /* In this factor, the search window will move in columns */
      for (int i = 0; i <= image.rows - sizeBase; i = i + step_x) {
        for (int j = 0; j <= image.cols - sizeBase; j = j + step_y) {

          /*__________ CALCULATING WINDOW POSITION __________*/
          int topLeft_x = i;
          int topLeft_y = j;
          int bottomRight_x = sizeBase + i - 1;
          int bottomRight_y = sizeBase + j - 1;
          /*____________________________________________________________*/

          cv::Mat window;
          window = imageGray(cv::Range(topLeft_x, bottomRight_x + 1), cv::Range(topLeft_y, bottomRight_y + 1));

          bool flagDetected = false;
          double myDegree = 0;
          int num = 0;

          for (int orderDegrees = 0; orderDegrees < degrees.size(); orderDegrees++) {
            if (evaluateClassifier(window, idx, orderDegrees)) {

              myDegree = myDegree + degrees[orderDegrees];
              flagDetected = true;
              num++;

            }
          }

          if (flagDetected == true) {

            myDegree = myDegree / num;
            cv::RotatedRect rectRotate(cv::Point2f(j + (sizeBase / 2), i + (sizeBase / 2)), cv::Size2f(sizeBase, sizeBase), -myDegree);
            cv::Point2f vertices[4];
            rectRotate.points(vertices);
            for (int i = 0; i < 4; i++)
              cv::line(image, vertices[i], vertices[(i + 1) % 4], colorRectangles, lineThicknessRectangles);

          }

        }
      }

    }
    //___________________________________________________________________________________________________//
  }

}

void CASCADE_CLASSIFIERS_EVALUATION::detectObjectRectanglesGroupedZeroDegrees(cv::Mat & image, std::vector < cv::Mat > * listDetectedObjects, std::vector < cv::Rect > * coordinatesDetectedObjects, bool doubleDetectedList, bool paintDetections) {

  if (szImg != image.size()) {
    ImageBackground = cv::Mat::zeros(sizeMaxWindow + 2 * zsBackground, sizeMaxWindow + 2 * zsBackground, CV_8UC1);
    imageGray = ImageBackground(cv::Range(zsBackground, zsBackground + image.rows), cv::Range(zsBackground, zsBackground + image.cols));
    szImg = image.size();
  }

  cvtColor(image, imageGray, CV_BGR2GRAY); // Converting to grayscale

  if (flagActivateSkinColor) {
    cvtColor(image, hsv, CV_BGR2HSV);
    inRange(hsv, hsvMin, hsvMax, bw);
    integral(bw, integralBw, CV_32S);
  }

  int idx = 0;

  if (flagActivateSkinColor) {
    //_______________________________________________________________________________________________________________________//
    for (int sizeBase = sizeBaseEvaluation; sizeBase <= std::min(image.rows, image.cols); sizeBase = factorScaleWindow * sizeBase, idx++) {
      int step_x = sizeBase * stepWindow; /* In this factor, the search window will move in rows */
      int step_y = sizeBase * stepWindow; /* In this factor, the search window will move in columns */
      for (int i = 0; i <= image.rows - sizeBase; i = i + step_x) {
        for (int j = 0; j <= image.cols - sizeBase; j = j + step_y) {

          /*__________ CALCULATING THE WINDOW POSITION __________*/
          int topLeft_x = i;
          int topLeft_y = j;
          int bottomRight_x = sizeBase + i - 1;
          int bottomRight_y = sizeBase + j - 1;
          /*____________________________________________________________*/

          cv::Mat window;
          window = imageGray(cv::Range(topLeft_x, bottomRight_x + 1), cv::Range(topLeft_y, bottomRight_y + 1));

          //___________________________________________________________________________________________//
          double sum2 = integralBw.at < int > (bottomRight_x + 1, bottomRight_y + 1) + integralBw.at < int > (topLeft_x, topLeft_y) - (integralBw.at < int > (bottomRight_x + 1, topLeft_y) + integralBw.at < int > (topLeft_x, bottomRight_y + 1)); // Integral image evaluation

          if (sum2 > 76.5 * sizeBase * sizeBase) { // 76.5 = 255 * 0.3

            for (int orderDegrees = 0; orderDegrees < degrees.size(); orderDegrees++) {
              if (evaluateClassifier(window, idx, orderDegrees)) {

                cv::Rect rectTemp(j, i, sizeBase, sizeBase);
                windowsCandidates.push_back(rectTemp);

              }
            }

          }
          //___________________________________________________________________________________________//

        }
      }

    }
    //_____________________________________________________________________________________________________________________//

  } else {

    //_______________________________________________________________________________________________________________________//
    for (int sizeBase = sizeBaseEvaluation; sizeBase <= std::min(image.rows, image.cols); sizeBase = factorScaleWindow * sizeBase, idx++) {
      int step_x = sizeBase * stepWindow; /* In this factor, the search window will move in rows */
      int step_y = sizeBase * stepWindow; /* In this factor, the search window will move in columns */
      for (int i = 0; i <= image.rows - sizeBase; i = i + step_x) {
        for (int j = 0; j <= image.cols - sizeBase; j = j + step_y) {

          /*__________ CALCULATING THE WINDOW POSITION __________*/
          int topLeft_x = i;
          int topLeft_y = j;
          int bottomRight_x = sizeBase + i - 1;
          int bottomRight_y = sizeBase + j - 1;
          /*____________________________________________________________*/

          cv::Mat window;
          window = imageGray(cv::Range(topLeft_x, bottomRight_x + 1), cv::Range(topLeft_y, bottomRight_y + 1));

          //___________________________________________________________________________________________//

          for (int orderDegrees = 0; orderDegrees < degrees.size(); orderDegrees++) {
            if (evaluateClassifier(window, idx, orderDegrees)) {

              cv::Rect rectTemp(j, i, sizeBase, sizeBase);
              windowsCandidates.push_back(rectTemp);

            }
          }
          //___________________________________________________________________________________________//

        }
      }

    }
    //_____________________________________________________________________________________________________________________//
  }

  if (doubleDetectedList == true) {
    /* This is done because the cv::groupRectangles function will remove the group that contains only one rectangle, so we double the list to avoid these being removed, as they may be sparse when the false negative rate is very low */
    //________________________________________________________________//
    int sz = windowsCandidates.size();
    for (int i = 0; i < sz; i++)
      windowsCandidates.push_back(cv::Rect(windowsCandidates[i]));
    //________________________________________________________________//
  }

  // Grouping similar rectangles
  cv::groupRectangles(windowsCandidates, groupThreshold, eps);

  /* NOTE: We extract the rectangles first because if we do it after drawing the rectangles, the extracted image will also be colored with the rectangle lines */
  //_________ Here we extract the detected images _________________
  if (listDetectedObjects != NULL) {

    if (flagExtractColorImages == true) {

      for (int i = 0; i < windowsCandidates.size(); i++)
        listDetectedObjects->push_back(image(windowsCandidates[i]).clone());

    } else {

      for (int i = 0; i < windowsCandidates.size(); i++)
        listDetectedObjects->push_back(imageGray(windowsCandidates[i]).clone());

    }

  }
  //_________________________________________________________________

  if (paintDetections) {
    // Painting the rectangles
    for (int i = 0; i < windowsCandidates.size(); i++)
      cv::rectangle(image, windowsCandidates[i], colorRectangles, lineThicknessRectangles);
  }

  if (coordinatesDetectedObjects != NULL)
    (*coordinatesDetectedObjects) = windowsCandidates; /* Coordinates of the detected rectangles relative to the input image coordinates */

  windowsCandidates.clear();

}

//______________________________________________________________________________________________________________//

// The class below is a modification of the SimilarRects class from openCV
class SimilarRectsRotated {
  public: SimilarRectsRotated(double _eps): eps(_eps) {}
  inline bool operator()(const cv::RotatedRect & r1,
    const cv::RotatedRect & r2) const {
    double delta = eps * (std::min(r1.size.width, r2.size.width) + std::min(r1.size.height, r2.size.height)) * 0.5;

    return std::abs(r1.center.x - r2.center.x) <= delta &&
      std::abs(r1.center.y - r2.center.y) <= delta &&
      std::abs(r1.size.width - r2.size.width) <= delta &&
      std::abs(r1.size.height - r2.size.height) <= delta;
  }
  double eps;
};

// The following function is a modification of the groupRectangle function from openCV
void groupRectanglesRotated(std::vector < cv::RotatedRect > & rectList, int groupThreshold, double eps) {
  if (groupThreshold <= 0 || rectList.empty()) {
    return;
  }

  std::vector < int > labels;
  int nclasses = partition(rectList, labels, SimilarRectsRotated(eps));

  std::vector < cv::RotatedRect > rrects(nclasses);
  std::vector < int > rweights(nclasses, 0);
  int i, j, nlabels = (int) labels.size();
  for (i = 0; i < nlabels; i++) {
    int cls = labels[i];
    rrects[cls].center.x += rectList[i].center.x;
    rrects[cls].center.y += rectList[i].center.y;
    rrects[cls].size.width += rectList[i].size.width;
    rrects[cls].size.height += rectList[i].size.height;
    rrects[cls].angle += rectList[i].angle;
    rweights[cls]++;
  }

  for (i = 0; i < nclasses; i++) {
    cv::RotatedRect r = rrects[i];
    float s = 1.f / rweights[i];
    rrects[i] = cv::RotatedRect(cv::Point2f(s * rrects[i].center.x, s * rrects[i].center.y), cv::Size2f(s * rrects[i].size.width, s * rrects[i].size.height), s * rrects[i].angle);
  }

  rectList.clear();

  for (i = 0; i < nclasses; i++) {
    cv::RotatedRect r1 = rrects[i];
    int n1 = rweights[i];
    // filter out rectangles which don't have enough similar rectangles
    if (n1 <= groupThreshold)
      continue;
    // filter out small face rectangles inside large rectangles
    for (j = 0; j < nclasses; j++) {
      int n2 = rweights[j];

      if (j == i || n2 <= groupThreshold)
        continue;
      cv::RotatedRect r2 = rrects[j];

      int dx = cv::saturate_cast < int > (r2.size.width * eps);
      int dy = cv::saturate_cast < int > (r2.size.height * eps);

      if (i != j &&
        r1.center.x >= r2.center.x - dx &&
        r1.center.y >= r2.center.y - dy &&
        r1.center.x + r1.size.width <= r2.center.x + r2.size.width + dx &&
        r1.center.y + r1.size.height <= r2.center.y + r2.size.height + dy &&
        (n2 > std::max(3, n1) || n1 < 3))
        break;
    }

    if (j == nclasses) {
      rectList.push_back(r1);
    }
  }
}

void CASCADE_CLASSIFIERS_EVALUATION::detectObjectRectanglesRotatedGrouped(cv::Mat & image, std::vector < cv::Mat > * listDetectedObjects, std::vector < cv::RotatedRect > * coordinatesDetectedObjects, bool paintDetections) {

  if (szImg != image.size()) {
    ImageBackground = cv::Mat::zeros(sizeMaxWindow + 2 * zsBackground, sizeMaxWindow + 2 * zsBackground, CV_8UC1);
    imageGray = ImageBackground(cv::Range(zsBackground, zsBackground + image.rows), cv::Range(zsBackground, zsBackground + image.cols));

    imageAndBackgroundColor = cv::Mat::zeros(3 * image.rows, 3 * image.cols, CV_8UC3);
    imageAndBackgroundGray = cv::Mat::zeros(3 * image.rows, 3 * image.cols, CV_8UC1);

    szImg = image.size();
  }

  cvtColor(image, imageGray, CV_BGR2GRAY); // Converting to grayscale

  if (flagActivateSkinColor) {
    cvtColor(image, hsv, CV_BGR2HSV);
    inRange(hsv, hsvMin, hsvMax, bw);
    integral(bw, integralBw, CV_32S);
  }

  int idx = 0;

  if (flagActivateSkinColor) {

    //__________________________________________________________________________________________________________________________//
    for (int sizeBase = sizeBaseEvaluation; sizeBase <= std::min(image.rows, image.cols); sizeBase = factorScaleWindow * sizeBase, idx++) {
      int step_x = sizeBase * stepWindow; /* In this factor, the search window will move along rows */
      int step_y = sizeBase * stepWindow; /* In this factor, the search window will move along columns */
      for (int i = 0; i <= image.rows - sizeBase; i = i + step_x) {
        for (int j = 0; j <= image.cols - sizeBase; j = j + step_y) {

          /*__________CALCULATING WINDOW POSITION______________*/
          int topLeft_x = i;
          int topLeft_y = j;
          int bottomRight_x = sizeBase + i - 1;
          int bottomRight_y = sizeBase + j - 1;
          /*____________________________________________________________*/

          cv::Mat window;
          window = imageGray(cv::Range(topLeft_x, bottomRight_x + 1), cv::Range(topLeft_y, bottomRight_y + 1));

          //______________________________________________________________________________//
          double sum2 = integralBw.at < int > (bottomRight_x + 1, bottomRight_y + 1) + integralBw.at < int > (topLeft_x, topLeft_y) - (integralBw.at < int > (bottomRight_x + 1, topLeft_y) + integralBw.at < int > (topLeft_x, bottomRight_y + 1)); // Integral image evaluation

          if (sum2 > 76.5 * sizeBase * sizeBase) { // 76.5=255*0.3

            for (int orderDegrees = 0; orderDegrees < degrees.size(); orderDegrees++) {
              if (evaluateClassifier(window, idx, orderDegrees)) {

                cv::RotatedRect windowTemp(cv::Point2f(j + (sizeBase / 2), i + (sizeBase / 2)), cv::Size2f(sizeBase, sizeBase), -degrees[orderDegrees]);
                windowsCandidatesRotated.push_back(windowTemp);

              }
            }

          }
          //______________________________________________________________________________//

        }
      }

    }
    //_____________________________________________________________________________________________________________________//

  } else {
    //__________________________________________________________________________________________________________________________//
    for (int sizeBase = sizeBaseEvaluation; sizeBase <= std::min(image.rows, image.cols); sizeBase = factorScaleWindow * sizeBase, idx++) {
      int step_x = sizeBase * stepWindow; /* In this factor, the search window will move along rows */
      int step_y = sizeBase * stepWindow; /* In this factor, the search window will move along columns */
      for (int i = 0; i <= image.rows - sizeBase; i = i + step_x) {
        for (int j = 0; j <= image.cols - sizeBase; j = j + step_y) {

          /*__________CALCULATING WINDOW POSITION______________*/
          int topLeft_x = i;
          int topLeft_y = j;
          int bottomRight_x = sizeBase + i - 1;
          int bottomRight_y = sizeBase + j - 1;
          /*____________________________________________________________*/

          cv::Mat window;
          window = imageGray(cv::Range(topLeft_x, bottomRight_x + 1), cv::Range(topLeft_y, bottomRight_y + 1));

          //______________________________________________________________________________//
          for (int orderDegrees = 0; orderDegrees < degrees.size(); orderDegrees++) {
            if (evaluateClassifier(window, idx, orderDegrees)) {

              cv::RotatedRect windowTemp(cv::Point2f(j + (sizeBase / 2), i + (sizeBase / 2)), cv::Size2f(sizeBase, sizeBase), -degrees[orderDegrees]);
              windowsCandidatesRotated.push_back(windowTemp);

            }
          }
          //______________________________________________________________________________//

        }
      }

    }
    //_____________________________________________________________________________________________________________________//
  }

  groupRectanglesRotated(windowsCandidatesRotated, groupThreshold, eps);

  // Here we extract the rectangles of the detected objects
  if (listDetectedObjects != NULL) {

    if (flagExtractColorImages == true) {

      image.copyTo(imageAndBackgroundColor(cv::Range(image.rows, 2 * image.rows), cv::Range(image.cols, 2 * image.cols)));

      for (int i = 0; i < windowsCandidatesRotated.size(); i++) {
        cv::Rect tempBoundingRect = windowsCandidatesRotated[i].boundingRect(); // Bounding rectangle of the rotated rectangle
        cv::Mat imagesBoundingRect = imageAndBackgroundColor(tempBoundingRect + cv::Point(image.cols, image.rows));
        cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(tempBoundingRect.width / 2, tempBoundingRect.height / 2), windowsCandidatesRotated[i].angle, 1.0);

        cv::Mat R;
        cv::warpAffine(imagesBoundingRect, R, M, imagesBoundingRect.size(), cv::INTER_CUBIC);

        getRectSubPix(R, windowsCandidatesRotated[i].size, cv::Point2f(tempBoundingRect.width / 2, tempBoundingRect.height / 2), R);

        listDetectedObjects -> push_back(R);
      }

    } else {

      imageGray.copyTo(imageAndBackgroundGray(cv::Range(image.rows, 2 * image.rows), cv::Range(image.cols, 2 * image.cols)));

      for (int i = 0; i < windowsCandidatesRotated.size(); i++) {
        cv::Rect tempBoundingRect = windowsCandidatesRotated[i].boundingRect(); // Bounding rectangle of the rotated rectangle
        cv::Mat imagesBoundingRect = imageAndBackgroundGray(tempBoundingRect + cv::Point(image.cols, image.rows));
        cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(tempBoundingRect.width / 2, tempBoundingRect.height / 2), windowsCandidatesRotated[i].angle, 1.0);

        cv::Mat R;
        cv::warpAffine(imagesBoundingRect, R, M, imagesBoundingRect.size(), cv::INTER_CUBIC);

        getRectSubPix(R, windowsCandidatesRotated[i].size, cv::Point2f(tempBoundingRect.width / 2, tempBoundingRect.height / 2), R);

        listDetectedObjects -> push_back(R);
      }

    }

  }

  if (paintDetections) {
    // Drawing the rectangles
    for (int i = 0; i < windowsCandidatesRotated.size(); i++) {

      cv::RotatedRect rectRotate = windowsCandidatesRotated[i];
      cv::Point2f vertices[4];
      rectRotate.points(vertices);
      for (int i = 0; i < 4; i++)
        cv::line(image, vertices[i], vertices[(i + 1) % 4], colorRectangles, lineThicknessRectangles);

    }

  }

  if (coordinatesDetectedObjects != NULL)
    ( * coordinatesDetectedObjects) = windowsCandidatesRotated; /* Coordinates of the detected rectangles with respect to the input image coordinates */

  windowsCandidatesRotated.clear();

}

// THE METHODS, CLASSES, AND FUNCTIONS DECLARED BELOW ARE ONLY USEFUL FOR THE FDDB DATABASE EVALUATION
#if EVALUATION_FDDB == 1

bool CASCADE_CLASSIFIERS_EVALUATION::FDDB_evaluateClassifier(cv::Mat & image, int scale, int orderDegrees) {

  for (int i = 0; i < numberClassifiersUsed; i++)
    if (!strongLearnsEvaluation[i] -> FDDB_evaluateStrongLearn(image, scale, orderDegrees)) return false; /*Classified as negative label*/

  scoreDetection = strongLearnsEvaluation[numberClassifiersUsed - 1] -> scoreDetection; // Detection score is taken

  return true; /*Classified as positive label*/

}

// This class will store the detection rectangle with its respective score 
class FDDB_RECT_AND_SCORES: public cv::Rect {
  public: FDDB_RECT_AND_SCORES() {};
  FDDB_RECT_AND_SCORES(int x, int y, int width, int height, double score = 0): cv::Rect(x, y, width, height),
  score(score) {}

  FDDB_RECT_AND_SCORES(const FDDB_RECT_AND_SCORES & rect) {
    x = rect.x;
    y = rect.y;
    width = rect.width;
    height = rect.height;
    score = rect.score;
  }

  cv::Rect getRect() const {
    return cv::Rect(x, y, width, height);
  }

  double score;
};

// This class is a modification of the openCV SimilarRects class, used to check if two rectangles are neighbors
class FDDB_SimilarRects {
  public: FDDB_SimilarRects(double _eps): eps(_eps) {}
  inline bool operator()(const FDDB_RECT_AND_SCORES & r1,
    const FDDB_RECT_AND_SCORES & r2) const {
    double delta = eps * (std::min(r1.width, r2.width) + std::min(r1.height, r2.height)) * 0.5;
    return std::abs(r1.x - r2.x) <= delta &&
      std::abs(r1.y - r2.y) <= delta &&
      std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
      std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
  }
  double eps;
};

// This function is a modification of the openCV groupRectangles function, used to group neighboring detection rectangles
void FDDB_groupRectangles(std::vector < FDDB_RECT_AND_SCORES > & rectList, int groupThreshold, double eps,
  std::vector < int > * weights = NULL, std::vector < double > * levelWeights = NULL) {
  if (groupThreshold <= 0 || rectList.empty()) {
    if (weights) {
      size_t i, sz = rectList.size();
      weights -> resize(sz);
      for (i = 0; i < sz; i++)
        ( * weights)[i] = 1;
    }
    return;
  }

  std::vector < int > labels;
  int nclasses = partition(rectList, labels, FDDB_SimilarRects(eps));

  std::vector < FDDB_RECT_AND_SCORES > rrects(nclasses);
  std::vector < int > rweights(nclasses, 0);
  std::vector < int > rejectLevels(nclasses, 0);
  std::vector < double > rejectWeights(nclasses, DBL_MIN);
  int i, j, nlabels = (int) labels.size();
  for (i = 0; i < nlabels; i++) {
    int cls = labels[i];
    rrects[cls].x += rectList[i].x;
    rrects[cls].y += rectList[i].y;
    rrects[cls].width += rectList[i].width;
    rrects[cls].height += rectList[i].height;
    rrects[cls].score += rectList[i].score;
    rweights[cls]++;
  }

  bool useDefaultWeights = false;

  if (levelWeights && weights && !weights -> empty() && !levelWeights -> empty()) {
    for (i = 0; i < nlabels; i++) {
      int cls = labels[i];
      if (( * weights)[i] > rejectLevels[cls]) {
        rejectLevels[cls] = ( * weights)[i];
        rejectWeights[cls] = ( * levelWeights)[i];
      } else if ((( * weights)[i] == rejectLevels[cls]) && (( * levelWeights)[i] > rejectWeights[cls]))
        rejectWeights[cls] = ( * levelWeights)[i];
    }
  } else
    useDefaultWeights = true;

  for (i = 0; i < nclasses; i++) {
    FDDB_RECT_AND_SCORES r = rrects[i];
    float s = 1.f / rweights[i];
    rrects[i] = FDDB_RECT_AND_SCORES(cv::saturate_cast < int > (r.x * s),
      cv::saturate_cast < int > (r.y * s),
      cv::saturate_cast < int > (r.width * s),
      cv::saturate_cast < int > (r.height * s),
      s * r.score);
  }

  rectList.clear();
  if (weights)
    weights -> clear();
  if (levelWeights)
    levelWeights -> clear();

  for (i = 0; i < nclasses; i++) {
    FDDB_RECT_AND_SCORES r1 = rrects[i];
    int n1 = rweights[i];
    double w1 = rejectWeights[i];
    int l1 = rejectLevels[i];

    // filter out rectangles which don't have enough similar rectangles
    if (n1 <= groupThreshold)
      continue;
    // filter out small face rectangles inside large rectangles
    for (j = 0; j < nclasses; j++) {
      int n2 = rweights[j];

      if (j == i || n2 <= groupThreshold)
        continue;
      FDDB_RECT_AND_SCORES r2 = rrects[j];

      int dx = cv::saturate_cast < int > (r2.width * eps);
      int dy = cv::saturate_cast < int > (r2.height * eps);

      if (i != j &&
        r1.x >= r2.x - dx &&
        r1.y >= r2.y - dy &&
        r1.x + r1.width <= r2.x + r2.width + dx &&
        r1.y + r1.height <= r2.y + r2.height + dy &&
        (n2 > std::max(3, n1) || n1 < 3))
        break;
    }

    if (j == nclasses) {
      rectList.push_back(r1);
      if (weights)
        weights -> push_back(useDefaultWeights ? n1 : l1);
      if (levelWeights)
        levelWeights -> push_back(w1);
    }
  }
}

/* This function detects faces in an image and returns the rectangles of these coordinates and the corresponding scores for each detection, as required by the software provided at http://vis-www.cs.umass.edu/fddb/ */
void CASCADE_CLASSIFIERS_EVALUATION::FDDB_detectObjectRectanglesGroupedZeroDegrees(cv::Mat & image, std::vector < cv::Rect > & rectanglesDetected, std::vector < double > & scores, bool doubleDetectedList) {

  std::vector < FDDB_RECT_AND_SCORES > fddbWindowsCandidates;

  if (szImg != image.size()) {
    ImageBackground = cv::Mat::zeros(sizeMaxWindow + 2 * zsBackground, sizeMaxWindow + 2 * zsBackground, CV_8UC1);
    imageGray = ImageBackground(cv::Range(zsBackground, zsBackground + image.rows), cv::Range(zsBackground, zsBackground + image.cols));
    szImg = image.size();
  }

  cvtColor(image, imageGray, CV_BGR2GRAY); // Converting to grayscale

  if (flagActivateSkinColor) {
    cvtColor(image, hsv, CV_BGR2HSV);
    inRange(hsv, hsvMin, hsvMax, bw);
    integral(bw, integralBw, CV_32S);
  }

  int idx = 0;

  if (flagActivateSkinColor) {

    //_______________________________________________________________________________________________________________________//
    for (int sizeBase = sizeBaseEvaluation; sizeBase <= std::min(image.rows, image.cols); sizeBase = factorScaleWindow * sizeBase, idx++) {
      int step_x = sizeBase * stepWindow; /* This factor moves the search window in rows */
      int step_y = sizeBase * stepWindow; /* This factor moves the search window in columns */
      for (int i = 0; i <= image.rows - sizeBase; i = i + step_x) {
        for (int j = 0; j <= image.cols - sizeBase; j = j + step_y) {

          /*__________CALCULATION OF WINDOW POSITION______________*/
          int topLeft_x = i;
          int topLeft_y = j;
          int bottomRight_x = sizeBase + i - 1;
          int bottomRight_y = sizeBase + j - 1;
          /*____________________________________________________________*/

          cv::Mat window;
          window = imageGray(cv::Range(topLeft_x, bottomRight_x + 1), cv::Range(topLeft_y, bottomRight_y + 1));

          //___________________________________________________________________________________________//
          double sum2 = integralBw.at < int > (bottomRight_x + 1, bottomRight_y + 1) + integralBw.at < int > (topLeft_x, topLeft_y) - (integralBw.at < int > (bottomRight_x + 1, topLeft_y) + integralBw.at < int > (topLeft_x, bottomRight_y + 1)); // Integral image evaluation

          if (sum2 > 76.5 * sizeBase * sizeBase) { //76.5=255*0.3

            for (int orderDegrees = 0; orderDegrees < degrees.size(); orderDegrees++) {
              if (FDDB_evaluateClassifier(window, idx, orderDegrees)) {

                FDDB_RECT_AND_SCORES rectTemp(j, i, sizeBase, sizeBase, scoreDetection);
                fddbWindowsCandidates.push_back(rectTemp);

              }
            }

          }
          //___________________________________________________________________________________________//

        }
      }

    }
    //_______________________________________________________________________________________________________________________//

  } else {

    //_______________________________________________________________________________________________________________________//
    for (int sizeBase = sizeBaseEvaluation; sizeBase <= std::min(image.rows, image.cols); sizeBase = factorScaleWindow * sizeBase, idx++) {
      int step_x = sizeBase * stepWindow; /* This factor moves the search window in rows */
      int step_y = sizeBase * stepWindow; /* This factor moves the search window in columns */
      for (int i = 0; i <= image.rows - sizeBase; i = i + step_x) {
        for (int j = 0; j <= image.cols - sizeBase; j = j + step_y) {

          /*__________CALCULATION OF WINDOW POSITION______________*/
          int topLeft_x = i;
          int topLeft_y = j;
          int bottomRight_x = sizeBase + i - 1;
          int bottomRight_y = sizeBase + j - 1;
          /*____________________________________________________________*/

          cv::Mat window;
          window = imageGray(cv::Range(topLeft_x, bottomRight_x + 1), cv::Range(topLeft_y, bottomRight_y + 1));

          //___________________________________________________________________________________________//

          for (int orderDegrees = 0; orderDegrees < degrees.size(); orderDegrees++) {
            if (FDDB_evaluateClassifier(window, idx, orderDegrees)) {

              FDDB_RECT_AND_SCORES rectTemp(j, i, sizeBase, sizeBase, scoreDetection);
              fddbWindowsCandidates.push_back(rectTemp);

            }
          }
          //___________________________________________________________________________________________//

        }
      }

    }
    //_______________________________________________________________________________________________________________________//

  }

  if (doubleDetectedList == true) {
    /* This is done because the function cv::groupRectangles will remove the group that only has one rectangle, so we double the list to avoid these from being removed as they may be sparse when the false negative rate is very low */
    //________________________________________________________________//
    int sz = fddbWindowsCandidates.size();
    for (int i = 0; i < sz; i++)
      fddbWindowsCandidates.push_back(FDDB_RECT_AND_SCORES(fddbWindowsCandidates[i]));
    //________________________________________________________________//
  }

  // Grouping similar rectangles
  FDDB_groupRectangles(fddbWindowsCandidates, groupThreshold, eps);

  for (int i = 0; i < fddbWindowsCandidates.size(); i++) {
    rectanglesDetected.push_back(fddbWindowsCandidates[i].getRect());
    scores.push_back(fddbWindowsCandidates[i].score);
  }

  // Drawing rectangles
  for (int i = 0; i < fddbWindowsCandidates.size(); i++)
    cv::rectangle(image, fddbWindowsCandidates[i].getRect(), colorRectangles, lineThicknessRectangles);

}
#endif
