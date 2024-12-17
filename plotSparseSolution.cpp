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

// Custom Headers
#include "plotSparseSolution.h"
#include <iostream> // To avoid issues with dictionary.h
// OpenCV
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "dictionary.h" // OpenCV libraries must be declared first
#include <QDebug>
// Qt
#include <QLabel>
#include <QFileDialog>
#include <QTabWidget>
#include <QAction>
#include <QMenu>
#include <QToolBar>
#include <QWheelEvent>
#include <QVBoxLayout>

PLOT_SPARSE_SOLUTION::PLOT_SPARSE_SOLUTION() {

  setPlotApproximation();
  setPlotComponentsApproximation();
  setPlotClustersDistance();
  setLayoutPrincipal();
  setActions();
  createContextMenu();
  setToolBar(); // Must be called after setting the main layout

  plotApproximation->setContextMenuPolicy(Qt::CustomContextMenu);
  plotComponentsApproximation->setContextMenuPolicy(Qt::CustomContextMenu);
  plotClustersDistance->setContextMenuPolicy(Qt::CustomContextMenu);
  connect(plotApproximation, SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(onCustomContextMenu(const QPoint &)));
  connect(plotComponentsApproximation, SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(onCustomContextMenu(const QPoint &)));
  connect(plotClustersDistance, SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(onCustomContextMenu(const QPoint &)));
}

void PLOT_SPARSE_SOLUTION::setActions() {

  backFeatureAction = new QAction(QString::fromUtf8("Previous descriptor"), this);
  connect(backFeatureAction, SIGNAL(triggered()), this, SLOT(backFeature()));
  backFeatureAction->setIcon(QIcon("./application_images/Actions/Back_plot.png"));
  backFeatureAction->setIconVisibleInMenu(true);

  nextFeatureAction = new QAction(tr("Next descriptor"), this);
  connect(nextFeatureAction, SIGNAL(triggered()), this, SLOT(nextFeature()));
  nextFeatureAction->setIcon(QIcon("./application_images/Actions/Next_plot.png"));
  nextFeatureAction->setIconVisibleInMenu(true);

  saveSelectedGraphicAction = new QAction(QString::fromUtf8("Save graphic"), this);
  connect(saveSelectedGraphicAction, SIGNAL(triggered()), this, SLOT(saveSelectedGraphic()));
  saveSelectedGraphicAction->setIcon(QIcon("./application_images/Actions/saveGraphic.png"));
  saveSelectedGraphicAction->setIconVisibleInMenu(true);

  sendGraphicsToMatlabAction = new QAction(QString::fromUtf8("Send data to Matlab"), this);
  connect(sendGraphicsToMatlabAction, SIGNAL(triggered()), this, SLOT(sendGraphicsToMatlab()));
  sendGraphicsToMatlabAction->setIcon(QIcon("./application_images/Actions/iconMatlab.png"));
  sendGraphicsToMatlabAction->setIconVisibleInMenu(true);
}

void PLOT_SPARSE_SOLUTION::createContextMenu() {

  contextMenuGraphic = new QMenu(this);
  contextMenuGraphic->addAction(saveSelectedGraphicAction);
  contextMenuGraphic->addAction(sendGraphicsToMatlabAction);
}

void PLOT_SPARSE_SOLUTION::setToolBar() {

  toolBar = new QToolBar;
  toolBar->setStyleSheet("QToolBar { background:gray; }");
  toolBar->setFixedHeight(50);
  layout()->setMenuBar(toolBar);

  toolBar->addAction(backFeatureAction);
  toolBar->addAction(nextFeatureAction);

  // Add the label to display the result of the currently plotted descriptor
  labelInfoNumDescriptor = new QLabel;
  labelInfoNumDescriptor->setFixedWidth(280);
  labelInfoNumDescriptor->setFrameShape(QFrame::StyledPanel);
  toolBar->addWidget(labelInfoNumDescriptor);
}

void PLOT_SPARSE_SOLUTION::setPlotApproximation() {

  plotApproximation = new QCustomPlot;

  plotApproximation->legend->setVisible(true);
  plotApproximation->legend->setFont(QFont("Helvetica", 9));
  plotApproximation->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignTop | Qt::AlignRight);
  plotApproximation->yAxis->grid()->setSubGridVisible(true);
  plotApproximation->xAxis->grid()->setSubGridVisible(true);

  //____________Main Title_______________________________//
  plotApproximation->plotLayout()->insertRow(0);
  plotApproximation->plotLayout()->addElement(0, 0, new QCPPlotTitle(plotApproximation, QString::fromUtf8("Approximation")));
  //____________Axis Titles___________________________//
  plotApproximation->xAxis->setLabel("X");
  plotApproximation->yAxis->setLabel("Y");

  plotApproximation->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes | QCP::iSelectLegend | QCP::iSelectPlottables);

  connect(plotApproximation, SIGNAL(mouseWheel(QWheelEvent *)), this, SLOT(zoomPlotApproximation(QWheelEvent *)));
}

void PLOT_SPARSE_SOLUTION::setPlotComponentsApproximation() {

  plotComponentsApproximation = new QCustomPlot;

  //____________Main Title_______________________________//
  plotComponentsApproximation->plotLayout()->insertRow(0);
  plotComponentsApproximation->plotLayout()->addElement(0, 0, new QCPPlotTitle(plotComponentsApproximation, QString::fromUtf8("Sparse Solution")));
  //____________Axis Titles___________________________//
  plotComponentsApproximation->xAxis->setLabel("Descriptors");
  plotComponentsApproximation->yAxis->setLabel("Contribution of each descriptor");

  plotComponentsApproximation->yAxis->grid()->setSubGridVisible(true);
  plotComponentsApproximation->xAxis->grid()->setSubGridVisible(true);
  plotComponentsApproximation->xAxis->setAutoTicks(false);
  plotComponentsApproximation->xAxis->setAutoTickLabels(false);

  plotComponentsApproximation->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes | QCP::iSelectLegend | QCP::iSelectPlottables);

  connect(plotComponentsApproximation, SIGNAL(mouseWheel(QWheelEvent *)), this, SLOT(zoomAxisXPlotComponentsApproximation(QWheelEvent *)));
}

void PLOT_SPARSE_SOLUTION::setPlotClustersDistance() {

  plotClustersDistance = new QCustomPlot;
  //____________Main Title_______________________________//
  plotClustersDistance->plotLayout()->insertRow(0);
  plotClustersDistance->plotLayout()->addElement(0, 0, new QCPPlotTitle(plotClustersDistance, QString::fromUtf8("L2 norm between possible classes and the input descriptor")));
  //____________Axis Titles___________________________//
  plotClustersDistance->xAxis->setLabel("Possible Classes");
  plotClustersDistance->yAxis->setLabel("Reconstruction Error");

  plotClustersDistance->yAxis->grid()->setSubGridVisible(true);
  plotClustersDistance->xAxis->grid()->setSubGridVisible(true);
  plotClustersDistance->xAxis->setAutoTicks(false);
  plotClustersDistance->xAxis->setAutoTickLabels(false);

  plotClustersDistance->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes | QCP::iSelectLegend | QCP::iSelectPlottables);

  connect(plotClustersDistance, SIGNAL(mouseWheel(QWheelEvent *)), this, SLOT(zoomPlotClustersDistance(QWheelEvent *)));
}

void PLOT_SPARSE_SOLUTION::setLayoutPrincipal() {

  tabWidgetPlots = new QTabWidget;
  tabWidgetPlots -> addTab(plotApproximation, QString::fromUtf8("Approximation"));
  tabWidgetPlots -> addTab(plotComponentsApproximation, QString::fromUtf8("SRC"));
  tabWidgetPlots -> addTab(plotClustersDistance, QString::fromUtf8("Possible Classes"));

  QVBoxLayout * layoutComponentsApproximation = new QVBoxLayout;
  layoutComponentsApproximation -> addWidget(tabWidgetPlots);

  setLayout(layoutComponentsApproximation);
  setMinimumSize(750, 500);
  setWindowTitle("SRC Plot");

  setWindowModality(Qt::WindowModal); // Important to prevent the calling event loop from blocking

}

void PLOT_SPARSE_SOLUTION::graphApproximation(QVector < double > originalVector, QVector < double > vectorApproximation, double error) {

  plotApproximation -> clearGraphs(); // Clear the graph

  QVector < double > x(originalVector.size());
  for (int i = 0; i < originalVector.size(); i++)
    x[i] = i;

  plotApproximation -> addGraph(plotApproximation -> xAxis, plotApproximation -> yAxis);
  plotApproximation -> graph(0) -> setPen(QPen(Qt::blue));
  plotApproximation -> graph(0) -> setSelectedPen(QPen(Qt::green));
  plotApproximation -> graph(0) -> setName("Original Descriptor");
  plotApproximation -> graph(0) -> setLineStyle(QCPGraph::lsLine);

  plotApproximation -> addGraph(plotApproximation -> xAxis, plotApproximation -> yAxis);
  plotApproximation -> graph(1) -> setPen(QPen(Qt::red));
  plotApproximation -> graph(1) -> setSelectedPen(QPen(Qt::green));
  plotApproximation -> graph(1) -> setName(QString::fromUtf8("Approximation"));
  plotApproximation -> graph(1) -> setLineStyle(QCPGraph::lsLine);

  plotApproximation -> graph(0) -> setData(x, originalVector);
  plotApproximation -> graph(1) -> setData(x, vectorApproximation);

  double minOriginalVector = * std::min_element(originalVector.begin(), originalVector.end());
  double maxOriginalVector = * std::max_element(originalVector.begin(), originalVector.end());
  double minVectorApproximation = * std::min_element(vectorApproximation.begin(), vectorApproximation.end());
  double maxVectorApproximation = * std::max_element(vectorApproximation.begin(), vectorApproximation.end());

  double min = std::min(minOriginalVector, minVectorApproximation);
  double max = std::max(maxOriginalVector, maxVectorApproximation);

  plotApproximation -> xAxis -> setRange(-1, originalVector.size() - 1); // Leave a -1 space between limits 0 and size-1
  plotApproximation -> yAxis -> setRange(min, max);

  plotApproximation -> replot();

}

void PLOT_SPARSE_SOLUTION::graphComponentsApproximation(QVector < double > contribution, QVector < int > cl, QVector < QString > listNameLabels) {

  plotComponentsApproximation -> clearGraphs(); // Clear the graph

  int sz = cl.size(); // Number of descriptors

  //______________Extract descriptor groups_______________
  QVector < int > colClusters;
  int val = cl[0];
  for (int i = 0; i < sz; i++) {

    if (val != cl[i]) {
      val = cl[i];
      colClusters.push_back(i);
    }

  }
  colClusters.push_back(sz);
  //__________________________________________________________________

  QVector < double > tickVector;
  int colAnt = 0;
  for (int i = 0; i < colClusters.size(); i++) {

    QVector < double > components(colClusters[i] - colAnt), values(colClusters[i] - colAnt);
    int n = 0;
    for (int j = colAnt; j < colClusters[i]; j++, n++) {

      components[n] = j;
      values[n] = contribution[j];

    }

    tickVector.push_back((components.last() + colAnt) / 2);

    //_____________________________GRAPH________________________________//
    plotComponentsApproximation -> addGraph(plotComponentsApproximation -> xAxis, plotComponentsApproximation -> yAxis); // Add a new graph

    plotComponentsApproximation -> graph(i) -> setSelectedPen(QPen(Qt::green));

    if (i % 2 == 0)
      plotComponentsApproximation -> graph(i) -> setPen(QPen(Qt::blue));
    else
      plotComponentsApproximation -> graph(i) -> setPen(QPen(Qt::red));

    plotComponentsApproximation -> graph(i) -> setLineStyle(QCPGraph::lsImpulse);
    plotComponentsApproximation -> graph(i) -> setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, 5));
    plotComponentsApproximation -> graph(i) -> setData(components, values);
    //____________________________________________________________________//

    colAnt = colClusters[i];

  }

  plotComponentsApproximation -> xAxis -> setTickVector(tickVector);
  plotComponentsApproximation -> xAxis -> setTickVectorLabels(listNameLabels);
  plotComponentsApproximation -> xAxis -> setTickLabelRotation(60);
  plotComponentsApproximation -> xAxis -> setSubTickCount(0);

  double minValueContribution = * std::min_element(contribution.begin(), contribution.end());
  double maxValueContribution = * std::max_element(contribution.begin(), contribution.end());

  //________________________________________________________________________________________________//
  /*
  std::cout<<"Maximum match "<<maxValueContribution<<" at "<<(std::distance(contribution.begin(),std::max_element(contribution.begin(),contribution.end()))+1)<<"\n";
  */

  double k = contribution.size();
  double absSum = 0;
  for (int i = 0; i < k; i++)
    absSum = absSum + std::abs(contribution[i]);

  std::cout << "Sum=" << absSum << "\n";
  double SCI = (((k * maxValueContribution) / absSum) - 1) / (k - 1);

  std::cout << "Image " << (std::distance(contribution.begin(), std::max_element(contribution.begin(), contribution.end())) + 1) << ", SCI=" << SCI << " \n";

  //________________________________________________________________________________________________//

  plotComponentsApproximation -> xAxis -> setRange(0, contribution.size()); // Leave a 1 space between limits 1 and contribution.size()-1
  plotComponentsApproximation -> yAxis -> setRange(minValueContribution, maxValueContribution);

  plotComponentsApproximation -> replot(); // Repaint the graph

}

void PLOT_SPARSE_SOLUTION::graphClustersDistance() {

  plotClustersDistance->clearGraphs(); // Clear the graph

  // Remember that the size of clt and clustersDistance is the same
  std::vector<int> clt = myDictionary->get_clt();
  std::vector<float> clustersDistance = myDictionary->get_clustersDistance();

  if (clt.empty()) return;

  QVector<QString> possibleLabels(clt.size());
  QVector<double> distance(clustersDistance.size());

  for (int i = 0; i < clt.size(); i++) {
    possibleLabels[i] = myNameUsersListAndId[clt[i]];
    distance[i] = clustersDistance[i];
    // qDebug() << "name=" << possibleLabels[i] << "  distance=" << distance[i] << "\n";
  }

  plotClustersDistance->addGraph(plotClustersDistance->xAxis, plotClustersDistance->yAxis);
  plotClustersDistance->graph(0)->setPen(QPen(Qt::blue));
  plotClustersDistance->graph(0)->setLineStyle(QCPGraph::lsLine);
  plotClustersDistance->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, 5));

  QVector<double> tickVector(distance.size()), X(distance.size());
  for (int i = 1; i <= distance.size(); i++)
    tickVector[i - 1] = i;

  plotClustersDistance->graph(0)->setData(tickVector, distance);
  plotClustersDistance->xAxis->setTickVector(tickVector);
  plotClustersDistance->xAxis->setTickVectorLabels(possibleLabels);
  plotClustersDistance->xAxis->setTickLabelRotation(60);
  plotClustersDistance->xAxis->setSubTickCount(0);

  double minDistance = *std::min_element(distance.begin(), distance.end());
  double maxDistance = *std::max_element(distance.begin(), distance.end());

  plotClustersDistance->xAxis->setRange(0, distance.size() + 1);
  plotClustersDistance->yAxis->setRange(minDistance, maxDistance);
  plotClustersDistance->replot(); // Redraw the graph
}

void PLOT_SPARSE_SOLUTION::zoomPlotApproximation(QWheelEvent* event) {

  if (plotApproximation->xAxis->selectedParts().testFlag(QCPAxis::spAxis))
    plotApproximation->axisRect()->setRangeZoom(plotApproximation->xAxis->orientation());
  else if (plotApproximation->yAxis->selectedParts().testFlag(QCPAxis::spAxis))
    plotApproximation->axisRect()->setRangeZoom(plotApproximation->yAxis->orientation());
  else
    plotApproximation->axisRect()->setRangeZoom(Qt::Horizontal | Qt::Vertical);
}

void PLOT_SPARSE_SOLUTION::zoomAxisXPlotComponentsApproximation(QWheelEvent* event) {

  if (plotComponentsApproximation->xAxis->selectedParts().testFlag(QCPAxis::spAxis))
    plotComponentsApproximation->axisRect()->setRangeZoom(plotComponentsApproximation->xAxis->orientation());
  else if (plotComponentsApproximation->yAxis->selectedParts().testFlag(QCPAxis::spAxis))
    plotComponentsApproximation->axisRect()->setRangeZoom(plotComponentsApproximation->yAxis->orientation());
  else
    plotComponentsApproximation->axisRect()->setRangeZoom(Qt::Horizontal | Qt::Vertical);
}

void PLOT_SPARSE_SOLUTION::zoomPlotClustersDistance(QWheelEvent* event) {

  if (plotClustersDistance->xAxis->selectedParts().testFlag(QCPAxis::spAxis))
    plotClustersDistance->axisRect()->setRangeZoom(plotClustersDistance->xAxis->orientation());
  else if (plotClustersDistance->yAxis->selectedParts().testFlag(QCPAxis::spAxis))
    plotClustersDistance->axisRect()->setRangeZoom(plotClustersDistance->yAxis->orientation());
  else
    plotClustersDistance->axisRect()->setRangeZoom(Qt::Horizontal | Qt::Vertical);
}

void PLOT_SPARSE_SOLUTION::setDictionary(DICTIONARY* dictionary) {

  myDictionary = dictionary;
  indexFeature = -1;
  maxNumberFeatures = myDictionary->get_nct() - 1; // Start indexing from zero
}

void PLOT_SPARSE_SOLUTION::setNameList(const QStringList list) {

  listNameLabels.clear();
  for (int i = 0; i < list.size(); i++)
    listNameLabels.push_back(list[i]);
}

void PLOT_SPARSE_SOLUTION::setNameUsersListAndId(const QMap<int, QString>& nameUsersListAndId) {
  myNameUsersListAndId = nameUsersListAndId;
}

void PLOT_SPARSE_SOLUTION::plotFirstResult() {

  indexFeature = -1;
  graphClustersDistance();
  nextFeature();
}

void PLOT_SPARSE_SOLUTION::nextFeature() {

  indexFeature++;
  if (indexFeature >= maxNumberFeatures) indexFeature = maxNumberFeatures;

  plotResultsSparseSolution(indexFeature);
}

void PLOT_SPARSE_SOLUTION::backFeature() {

  indexFeature--;
  if (indexFeature <= 0) indexFeature = 0;

  plotResultsSparseSolution(indexFeature);
}

void PLOT_SPARSE_SOLUTION::plotResultsSparseSolution(int index) {

  showInfoNumDescriptor(index);

  Eigen::MatrixXf XS, XA;
  myDictionary->get_completeSolution(XS, XA, index);

  Eigen::MatrixXf b = myDictionary->bDescriptor(index);

  // Remember that XA and b should have the same number of rows
  QVector<double> originalVector(b.rows()), vectorApproximation(b.rows());
  double meansquareError = 0;

  for (int i = 0; i < b.rows(); i++) {
    originalVector[i] = b(i, 0);
    vectorApproximation[i] = XA(i, 0);
    meansquareError = meansquareError + std::pow((originalVector[i] - vectorApproximation[i]), 2);
  }

  meansquareError = std::sqrt(meansquareError / b.rows());

  QVector<double> contribution(XS.rows());
  for (int i = 0; i < XS.rows(); i++)
    contribution[i] = XS(i, 0);

  int numberDescriptors = myDictionary->get_numberDescriptors();
  int* CL = myDictionary->get_CL();

  QVector<int> cl(numberDescriptors);
  for (int i = 0; i < numberDescriptors; i++)
    cl[i] = CL[i];

  graphApproximation(originalVector, vectorApproximation, meansquareError);
  graphComponentsApproximation(contribution, cl, listNameLabels);
}

void PLOT_SPARSE_SOLUTION::showInfoNumDescriptor(int index) {

  // Choose to number the descriptors between 1 and the total

  QFont myFont;
  QString str = QString("Descriptor number ") + QString::number(index + 1) + QString(" of ") + QString::number(maxNumberFeatures + 1) +
    QString(" in total    ");

  QFontMetrics fm(myFont);
  int width = fm.width(str);
  labelInfoNumDescriptor->setText("<font color=white>" + str);
  labelInfoNumDescriptor->setFixedWidth(width);
}

void PLOT_SPARSE_SOLUTION::onCustomContextMenu(const QPoint& point) {

  QWidget* wg = childAt(point);
  if (wg == static_cast<QCustomPlot*>(plotApproximation)) {

    plotSave = dynamic_cast<QCustomPlot*>(wg);
    contextMenuGraphic->exec(mapToGlobal(point));

  } else if (wg == static_cast<QCustomPlot*>(plotComponentsApproximation)) {

    plotSave = dynamic_cast<QCustomPlot*>(wg);
    contextMenuGraphic->exec(mapToGlobal(point));

  } else if (wg == static_cast<QCustomPlot*>(plotClustersDistance)) {

    plotSave = dynamic_cast<QCustomPlot*>(wg);
    contextMenuGraphic->exec(mapToGlobal(point));
  }
}

void PLOT_SPARSE_SOLUTION::saveSelectedGraphic() {

  QString fileName = QFileDialog::getSaveFileName(this, QString::fromUtf8("Save Graphic"),
    QDir::homePath() + QString("/untitled.png"),
    tr("Images (*.png)"));

  if (fileName == "") {
    qDebug() << QString::fromUtf8("Error, you must enter a name for the graphic to save") << "\n";
    return; // Does NOT save anything
  }

  if ((fileName.split(".")).size() == 1) {
    qDebug() << QString::fromUtf8("Error, the file name format must be imageName.png") << "\n";
    return;
  }

  plotSave -> savePng(fileName);

}

void PLOT_SPARSE_SOLUTION::sendGraphicsToMatlab() {

  myDictionary -> sendToMatlab();
  qDebug() << QString::fromUtf8("The dictionary data was written in the current directory, use this data from Matlab with the help of the readData.m file") << "\n";

}
