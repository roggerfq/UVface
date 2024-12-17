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

#ifndef PLOTSPARSESOLUTION_H
#define PLOTSPARSESOLUTION_H

//QT Classes
#include "qcustomplot.h" //Third-party libraries
#include <QDialog>
#include <QMap>

//Forward declarations
class DICTIONARY;

//Forward QT classes
class QLabel;
class QTabWidget;
class QToolBar;
class QAction;
class QMenu;
class QAction;
class QWheelEvent;

class PLOT_SPARSE_SOLUTION: public QDialog {

  Q_OBJECT

  QCustomPlot * plotApproximation; //Here the input vector will be plotted along with its respective sparse solution approximation
  QCustomPlot * plotComponentsApproximation; //Here the components contributing to the sparse solution approximation will be plotted
  QCustomPlot * plotClustersDistance; //Here the distances of the possible classes relative to the input descriptor will be plotted
  QCustomPlot * plotSave; //Used as a temporary variable

  //QLabel
  QLabel * labelInfoNumDescriptor;

  //QToolBar
  QToolBar * toolBar;

  //QTabWidget
  QTabWidget * tabWidgetPlots;

  //QAction
  QAction * nextFeatureAction;
  QAction * backFeatureAction;

  //QMenu
  QMenu * contextMenuGraphic;

  QAction * saveSelectedGraphicAction;
  QAction * sendGraphicsToMatlabAction;

  DICTIONARY * myDictionary; //Methods for the sparse solution
  QVector < QString > listNameLabels;
  QMap < int, QString > myNameUsersListAndId;
  int indexFeature;
  int maxNumberFeatures;

  public:
    PLOT_SPARSE_SOLUTION();

  void setPlotApproximation();
  void setPlotComponentsApproximation();
  void setPlotClustersDistance();
  void setLayoutPrincipal();
  void setActions();
  void createContextMenu();
  void setToolBar();

  public slots:

    void graphApproximation(QVector < double > originalVector, QVector < double > vectorApproximation, double error);
    void graphComponentsApproximation(QVector < double > contribution, QVector < int > cl, QVector < QString > listNameLabels);
    void graphClustersDistance();
    void zoomPlotApproximation(QWheelEvent * event);
    void zoomAxisXPlotComponentsApproximation(QWheelEvent * event);
    void zoomPlotClustersDistance(QWheelEvent * event);
    void setDictionary(DICTIONARY * dictionary);
    void setNameList(const QStringList list);
    void setNameUsersListAndId(const QMap < int, QString > & nameUsersListAndId);
    void plotFirstResult();
    void nextFeature();
    void backFeature();
    void plotResultsSparseSolution(int index);
    void showInfoNumDescriptor(int index);
    void onCustomContextMenu(const QPoint & point);
    void saveSelectedGraphic();
    void sendGraphicsToMatlab();

};

#endif
