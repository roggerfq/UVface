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

#include "trackerWindows.h"
#include <iostream>

// This class is a modification of the SimilarRects class from OpenCV. It determines if two rectangleDetection objects are neighbors.
class SimilarRectsDetection {
public:
    SimilarRectsDetection(double _eps): eps(_eps) {}
    inline bool operator()(const rectangleDetection & r1,
                           const rectangleDetection & r2) const {
        double delta = eps * (std::min(r1.width, r2.width) + std::min(r1.height, r2.height)) * 0.5;
        return std::abs(r1.x - r2.x) <= delta &&
               std::abs(r1.y - r2.y) <= delta &&
               std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
               std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
    }
    double eps;
};

// The following class is a modification of the SimilarRects class from OpenCV.
class SimilarRectsRotated {
public:
    SimilarRectsRotated(double _eps): eps(_eps) {}
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

rectangleDetection::rectangleDetection(): punctuationStableWindow(0), id(-1), isNew(true), empty(true), windowSentToRecognizer(false), isRecognized(false) {}

rectangleDetection::rectangleDetection(const cv::Rect & rect, cv::Mat & img, int punctuationBeforeDeleting, QTime birthdate, int id, bool isNew): img(img), punctuationStableWindow(0), punctuationBeforeDeleting(punctuationBeforeDeleting), id(id), isNew(isNew), empty(false), windowSentToRecognizer(false), isRecognized(false) {

    x = rect.x;
    y = rect.y;
    width = rect.width;
    height = rect.height;

    myBirthdate = birthdate;
}

rectangleDetection::rectangleDetection(const rectangleDetection & rectDetection) {

    x = rectDetection.x;
    y = rectDetection.y;
    width = rectDetection.width;
    height = rectDetection.height;

    id = rectDetection.id;
    myBirthdate = rectDetection.myBirthdate;
    punctuationStableWindow = rectDetection.punctuationStableWindow;
    punctuationBeforeDeleting = rectDetection.punctuationBeforeDeleting;
    isNew = rectDetection.isNew;
    empty = rectDetection.empty;
    windowSentToRecognizer = rectDetection.windowSentToRecognizer;
    isRecognized = rectDetection.isRecognized;
    name = rectDetection.name;
    img = rectDetection.img;
}

void rectangleDetection::setRect(const cv::Rect & rectangle) {
    x = rectangle.x;
    y = rectangle.y;
    width = rectangle.width;
    height = rectangle.height;
}

cv::Rect rectangleDetection::getRect() const {
    return cv::Rect(x, y, width, height);
}

rotatedRectDetection::rotatedRectDetection(): punctuationStableWindow(0), id(-1), isNew(true), empty(true), windowSentToRecognizer(false), isRecognized(false) {}

rotatedRectDetection::rotatedRectDetection(const cv::RotatedRect & rotatedRect, cv::Mat & img, int punctuationBeforeDeleting, QTime birthdate, int id, bool isNew): img(img), punctuationStableWindow(0), punctuationBeforeDeleting(punctuationBeforeDeleting), id(id), isNew(isNew), empty(false), windowSentToRecognizer(false), isRecognized(false) {

    center = rotatedRect.center;
    size = rotatedRect.size;
    angle = rotatedRect.angle;

    myBirthdate = birthdate;
}

rotatedRectDetection::rotatedRectDetection(const rotatedRectDetection & rotatedDetection) {

    center = rotatedDetection.center;
    size = rotatedDetection.size;
    angle = rotatedDetection.angle;

    id = rotatedDetection.id;
    myBirthdate = rotatedDetection.myBirthdate;
    punctuationStableWindow = rotatedDetection.punctuationStableWindow;
    punctuationBeforeDeleting = rotatedDetection.punctuationBeforeDeleting;
    isNew = rotatedDetection.isNew;
    empty = rotatedDetection.empty;
    windowSentToRecognizer = rotatedDetection.windowSentToRecognizer;
    isRecognized = rotatedDetection.isRecognized;
    name = rotatedDetection.name;
    img = rotatedDetection.img;
}

void rotatedRectDetection::setRotatedRect(const cv::RotatedRect & rotatedRect) {
    center = rotatedRect.center;
    size = rotatedRect.size;
    angle = rotatedRect.angle;
}

cv::RotatedRect rotatedRectDetection::getRotatedRect() const {
    return cv::RotatedRect(center, size, angle);
}

trackerWindows::trackerWindows(QObject * parent): QObject(parent), defaultInitiaLpunctuation(10), defaultMinimumPunctuationToRecognize(1), defaultEps(1) {

    // Mandatory default parameter
    idNext = 0;

    setDefaultValues();
}

int trackerWindows::getId() {

    int id;

    if (idUnassigned.empty()) {
        id = idNext;
        idNext++;
    } else {
        id = idUnassigned.takeLast();
    }

    return id;
}

void trackerWindows::freeId(int id) {
    if (id >= 0)
        idUnassigned.push_back(id);
}

// Set functions

void trackerWindows::setDefaultValues() {

    // Default parameters
    initiaLpunctuation = defaultInitiaLpunctuation;
    minimumPunctuationToRecognize = defaultMinimumPunctuationToRecognize;
    eps = defaultEps;
}

void trackerWindows::setInitiaLpunctuation(int nFrames) {
    initiaLpunctuation = nFrames;
}

void trackerWindows::setMinimumPunctuationToRecognize(int nFrames) {
    minimumPunctuationToRecognize = nFrames;
}

void trackerWindows::setEps(double valueEps) {
    eps = valueEps;
}

// Get functions
int trackerWindows::getInitiaLpunctuation() const {
    return initiaLpunctuation;
}

int trackerWindows::getMinimumPunctuationToRecognize() const {
    return minimumPunctuationToRecognize;
}

double trackerWindows::getEps() const {
    return eps;
}

void trackerWindows::groupRectsDetection(std::vector<rectangleDetection>& rectangleDetectionList) {

    std::vector<int> labels;
    int nclasses = partition(rectangleDetectionList, labels, SimilarRectsDetection(eps));
    std::vector<rectangleDetection> rrects(nclasses);
    int nlabels = (int)labels.size();
    for (int i = 0; i < nlabels; i++) {

        int cls = labels[i];

        if (rrects[cls].empty) { // If no detection rectangle has been assigned at index cls, then simply insert it.
            rrects[cls] = rectangleDetectionList[i];
            if (rrects[cls].id == -1) rrects[cls].id = getId(); // If the detection rectangle is new, it won't have an ID, so we assign one.

        } else { // If there is already an assignment at position cls, do the following:

            if ((rrects[cls].isNew) && (!rectangleDetectionList[i].isNew)) {
                /*
                Update the ID, birthdate, punctuationStableWindow, windowSentToRecognizer, isRecognized, and img. 
                Also, free the ID previously assigned to rrects[cls].
                */
                freeId(rrects[cls].id); // The function internally checks if ID is equal to -1.
                rrects[cls].id = rectangleDetectionList[i].id;
                rrects[cls].myBirthdate = rectangleDetectionList[i].myBirthdate;
                rrects[cls].punctuationStableWindow = rectangleDetectionList[i].punctuationStableWindow;
                rrects[cls].windowSentToRecognizer = rectangleDetectionList[i].windowSentToRecognizer;
                rrects[cls].isRecognized = rectangleDetectionList[i].isRecognized;
                rrects[cls].name = rectangleDetectionList[i].name;
                // rrects[cls].img = rectangleDetectionList[i].img;
            } else if ((!rrects[cls].isNew) && (rectangleDetectionList[i].isNew)) {
                // Only update the coordinates and punctuationBeforeDeleting, and free the previous ID.
                freeId(rectangleDetectionList[i].id); // The function internally checks if ID is equal to -1.
                rrects[cls].setRect(rectangleDetectionList[i].getRect());
                rrects[cls].punctuationBeforeDeleting = rectangleDetectionList[i].punctuationBeforeDeleting;
                rrects[cls].img = rectangleDetectionList[i].img; // Priority is given to the most recent image.
            } else {
                // In the case of two old detections, keep the previously assigned one and free the new ID.
                // In the case of two new detections, keep the previously assigned one and free the new ID.
                freeId(rectangleDetectionList[i].id); // The function internally checks if ID is equal to -1.
            }

        }

    }

    // There is a possibility of no grouping; in that case, work with the same input rectangleDetectionList.
    if (rrects.empty()) rrects = rectangleDetectionList; // For optimal memory organization, assign the result at the end.
    rectangleDetectionList.clear();

    for (int i = 0; i < rrects.size(); i++) {

        if (rrects[i].isNew) {
            rrects[i].isNew = false; // Mark as old for the next iteration.
            rectangleDetectionList.push_back(rrects[i]);
        } else {

            rrects[i].punctuationBeforeDeleting--;
            if (rrects[i].punctuationBeforeDeleting > 0) {

                // Update the scores.
                if (!rrects[i].windowSentToRecognizer) {
                    rrects[i].punctuationStableWindow++;

                    if (rrects[i].punctuationStableWindow > minimumPunctuationToRecognize) {
                        imageTransaction tempimageTransaction(rrects[i].img, rrects[i].myBirthdate, rrects[i].id);
                        listToRecognize.push_back(tempimageTransaction);
                        rrects[i].windowSentToRecognizer = true;
                    }

                }

                rectangleDetectionList.push_back(rrects[i]); // If punctuationBeforeDeleting is still greater than zero, keep the detection.

            } else
                freeId(rrects[i].id); // Otherwise, free the ID and do not keep the detection.

        }

    }

}

void trackerWindows::checkRecognition() {

    std::vector<cv::Point> listPoints;
    std::vector<std::string> listText;

    if (listRecognizedImages.empty()) {

        for (int i = 0; i < detectedObjectsList.size(); i++) {

            if (detectedObjectsList[i].isRecognized) {
                listText.push_back(detectedObjectsList[i].name);
                listPoints.push_back(detectedObjectsList[i].tl());
            } else {
                /* listText.push_back((QString::number(detectedObjectsList[i].id) + QString(" ") + detectedObjectsList[i].myBirthdate.toString()).toStdString()); */
                listText.push_back(QString(" ").toStdString());
                listPoints.push_back(detectedObjectsList[i].tl());
            }

        }

    } else {

        for (int i = 0; i < detectedObjectsList.size(); i++) {

            if (listRecognizedImages.contains(detectedObjectsList[i].id)) {
                imageTransaction tempTransaction = listRecognizedImages.take(detectedObjectsList[i].id);
                if (tempTransaction.myBirthdate == detectedObjectsList[i].myBirthdate) {
                    detectedObjectsList[i].name = tempTransaction.name;
                    detectedObjectsList[i].isRecognized = true;
                }
            }

            if (detectedObjectsList[i].isRecognized) {
                listText.push_back(detectedObjectsList[i].name);
                listPoints.push_back(detectedObjectsList[i].tl());
            } else {
                /* listText.push_back((QString::number(detectedObjectsList[i].id) + QString(" ") + detectedObjectsList[i].myBirthdate.toString()).toStdString()); */
                listText.push_back(QString(" ").toStdString());
                listPoints.push_back(detectedObjectsList[i].tl());
            }

        }

    }

    emit setTextInDetection(listPoints, listText);
}

void trackerWindows::groupRectRotatedDetection(std::vector<rotatedRectDetection>& rectRotatedDetectionList) {

    std::vector<int> labels;
    int nclasses = partition(rectRotatedDetectionList, labels, SimilarRectsRotated(eps));
    std::vector<rotatedRectDetection> rrects(nclasses);
    int nlabels = (int)labels.size();
    for (int i = 0; i < nlabels; i++) {

        int cls = labels[i];

        if (rrects[cls].empty) { // If no detection rectangle has been assigned at index cls before, insert it
            rrects[cls] = rectRotatedDetectionList[i];
            if (rrects[cls].id == -1) rrects[cls].id = getId(); // If the detection rectangle is new, it won't have an ID, so assign one

        } else { // If there is already an assignment at position cls, perform the following

            if ((rrects[cls].isNew) && (!rectRotatedDetectionList[i].isNew)) {
                /*
                The ID, birthdate, punctuationStableWindow, windowSentToRecognizer, isRecognized, and img should be updated. 
                The previously assigned ID to rrects[cls] must also be released.
                */
                freeId(rrects[cls].id); // The function internally checks if id equals -1
                rrects[cls].id = rectRotatedDetectionList[i].id;
                rrects[cls].myBirthdate = rectRotatedDetectionList[i].myBirthdate;
                rrects[cls].punctuationStableWindow = rectRotatedDetectionList[i].punctuationStableWindow;
                rrects[cls].windowSentToRecognizer = rectRotatedDetectionList[i].windowSentToRecognizer;
                rrects[cls].isRecognized = rectRotatedDetectionList[i].isRecognized;
                rrects[cls].name = rectRotatedDetectionList[i].name;
                // rrects[cls].img=rectRotatedDetectionList[i].img;
            } else if ((!rrects[cls].isNew) && (rectRotatedDetectionList[i].isNew)) { // Only coordinates and punctuationBeforeDeleting should be updated, releasing the previous ID
                freeId(rectRotatedDetectionList[i].id); // The function internally checks if id equals -1
                rrects[cls].setRotatedRect(rectRotatedDetectionList[i].getRotatedRect());
                rrects[cls].punctuationBeforeDeleting = rectRotatedDetectionList[i].punctuationBeforeDeleting;
                rrects[cls].img = rectRotatedDetectionList[i].img; // Prioritize the most recent image
            } else {
                // In the case of two old detections, keep the previously assigned one and release the new one's ID
                // In the case of two new detections, keep the previously assigned one and release the new one's ID
                freeId(rectRotatedDetectionList[i].id); // The function internally checks if id equals -1
            }

        }

    }

    // If there is no grouping, work with the same input rectRotatedDetectionList
    if (rrects.empty()) rrects = rectRotatedDetectionList; // For optimal memory organization, assign the result last
    rectRotatedDetectionList.clear();

    for (int i = 0; i < rrects.size(); i++) {

        if (rrects[i].isNew) {
            rrects[i].isNew = false; // Mark as old for the next iteration
            rectRotatedDetectionList.push_back(rrects[i]);
        } else {

            rrects[i].punctuationBeforeDeleting--;
            if (rrects[i].punctuationBeforeDeleting > 0) {

                // Update scores
                if (!rrects[i].windowSentToRecognizer) {
                    rrects[i].punctuationStableWindow++;

                    if (rrects[i].punctuationStableWindow > minimumPunctuationToRecognize) {
                        imageTransaction tempimageTransaction(rrects[i].img, rrects[i].myBirthdate, rrects[i].id);
                        listToRecognize.push_back(tempimageTransaction);
                        rrects[i].windowSentToRecognizer = true;
                    }

                }

                rectRotatedDetectionList.push_back(rrects[i]); // If punctuationBeforeDeleting is still greater than zero, keep the detection

            } else
                freeId(rrects[i].id); // Otherwise, release the ID and do not keep the detection

        }

    }

}

void trackerWindows::checkRecognitionRectRotated() {

    std::vector<cv::Point> listPoints;
    std::vector<std::string> listText;

    if (listRecognizedImages.empty()) {

        for (int i = 0; i < detectedRotatedObjectsList.size(); i++) {

            if (detectedRotatedObjectsList[i].isRecognized) {
                listText.push_back(detectedRotatedObjectsList[i].name);
                listPoints.push_back(detectedRotatedObjectsList[i].boundingRect().tl());
            } else {
                /* listText.push_back((QString::number(detectedRotatedObjectsList[i].id)+QString(" ")+detectedRotatedObjectsList[i].myBirthdate.toString()).toStdString());*/
                listText.push_back(QString(" ").toStdString());
                listPoints.push_back(detectedRotatedObjectsList[i].boundingRect().tl());
            }

        }

    } else {

        for (int i = 0; i < detectedRotatedObjectsList.size(); i++) {

            if (listRecognizedImages.contains(detectedRotatedObjectsList[i].id)) {
                imageTransaction tempTransaction = listRecognizedImages.take(detectedRotatedObjectsList[i].id);
                if (tempTransaction.myBirthdate == detectedRotatedObjectsList[i].myBirthdate) {
                    detectedRotatedObjectsList[i].name = tempTransaction.name;
                    detectedRotatedObjectsList[i].isRecognized = true;
                }
            }

            if (detectedRotatedObjectsList[i].isRecognized) {
                listText.push_back(detectedRotatedObjectsList[i].name);
                listPoints.push_back(detectedRotatedObjectsList[i].boundingRect().tl());
            } else {
                /*listText.push_back((QString::number(detectedRotatedObjectsList[i].id)+QString(" ")+detectedRotatedObjectsList[i].myBirthdate.toString()).toStdString());*/
                listText.push_back(QString(" ").toStdString());
                listPoints.push_back(detectedRotatedObjectsList[i].boundingRect().tl());
            }

        }

    }

    emit setTextInDetection(listPoints, listText);

}

void trackerWindows::reset() {

    detectedObjectsList.clear();
    detectedRotatedObjectsList.clear();
    listToRecognize.clear();
    listRecognizedImages.clear();

    idNext = 0;
    idUnassigned.clear();

    std::cout << "RESET in trackerWindows::reset()\n";

}

void trackerWindows::recognizedImage(imageTransaction newImageTransaction) {

    listRecognizedImages.insert(newImageTransaction.id, newImageTransaction);

}

void trackerWindows::newGroupDetections(std::vector<cv::Mat> listDetectedObjects, std::vector<cv::Rect> coordinatesDetectedObjects) {

  if (detectedObjectsList.empty()) { // Initially, the list will be empty

    listRecognizedImages.clear();

    for (int i = 0; i < listDetectedObjects.size(); i++) {

      cv::Mat tempImg = listDetectedObjects[i]; // Detected image
      cv::Rect tempRect = coordinatesDetectedObjects[i]; // Rectangle enclosing the detected image
      rectangleDetection tempRectangleDetection(tempRect, tempImg, initiaLpunctuation, QTime::currentTime(), getId(), false);

      detectedObjectsList.push_back(tempRectangleDetection);

    }

  } else { // When the list is no longer empty

    for (int i = 0; i < listDetectedObjects.size(); i++) {

      cv::Mat tempImg = listDetectedObjects[i]; // Detected image
      cv::Rect tempRect = coordinatesDetectedObjects[i]; // Rectangle enclosing the detected image
      rectangleDetection tempRectangleDetection(tempRect, tempImg, initiaLpunctuation, QTime::currentTime(), -1, true);

      detectedObjectsList.push_back(tempRectangleDetection);

    }

    groupRectsDetection(detectedObjectsList); // Sent to grouping

    // Next, the list is sent for recognition
    if (!listToRecognize.empty()) {
      emit recognizeImagesList(listToRecognize);
      std::cout << "List emitted from trackerWindows with size=" << listToRecognize.size() << "\n";
      listToRecognize.clear(); // Items already sent for recognition are removed from the list
    }

    checkRecognition(); // The recognized detections are updated in the view

  }

}

void trackerWindows::newGroupDetections(std::vector<cv::Mat> listDetectedObjects, std::vector<cv::RotatedRect> coordinatesDetectedObjects) {

  if (detectedRotatedObjectsList.empty()) { // Initially, the list will be empty

    listRecognizedImages.clear();

    for (int i = 0; i < listDetectedObjects.size(); i++) {

      cv::Mat tempImg = listDetectedObjects[i]; // Detected image
      cv::RotatedRect tempRotatedRect = coordinatesDetectedObjects[i]; // Rectangle enclosing the detected image
      rotatedRectDetection tempRotatedRectDetection(tempRotatedRect, tempImg, initiaLpunctuation, QTime::currentTime(), getId(), false);

      detectedRotatedObjectsList.push_back(tempRotatedRectDetection);

    }

  } else { // When the list is no longer empty

    for (int i = 0; i < listDetectedObjects.size(); i++) {

      cv::Mat tempImg = listDetectedObjects[i]; // Detected image
      cv::RotatedRect tempRotatedRect = coordinatesDetectedObjects[i]; // Rectangle enclosing the detected image
      rotatedRectDetection tempRotatedRectDetection(tempRotatedRect, tempImg, initiaLpunctuation, QTime::currentTime(), -1, true);

      detectedRotatedObjectsList.push_back(tempRotatedRectDetection);

    }

    groupRectRotatedDetection(detectedRotatedObjectsList); // Sent to grouping

    // Next, the list is sent for recognition
    if (!listToRecognize.empty()) {
      emit recognizeImagesList(listToRecognize);
      std::cout << "List emitted from trackerWindows with size=" << listToRecognize.size() << "\n";
      listToRecognize.clear(); // Items already sent for recognition are removed from the list
    }

    checkRecognitionRectRotated(); // The recognized detections are updated in the view

  }

}
