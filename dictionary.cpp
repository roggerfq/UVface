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

/*_____________DEBUGGING________________________*/
#include <iostream>
#include <stdio.h>
#include <iomanip> // std::setprecision, for now used for debugging
#include <QTime>//Delete at the end (used in test time functions)

QTime time_test;
void tic() {
  std::cout << "Start\n";
  time_test.start();
}

void toc() {
  std::cout << "Time=" << time_test.elapsed() << "\n";
}
/*_______________________________________________*/

/*________STL_________*/
#include <cstdlib>
#include <ctime>
#include <cerrno>
#include <cstring>
#include <clocale>

/*___________________*/

/*________openCV_________*/
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

/*_______________________*/

/*________Own________*/
#include "dictionary.h"
/*_______________________*/

/*_______openMP__________*/
#include <omp.h>
/*_______________________*/

/*_________________Constants and global variables______________________________________*/

int G_MAX_NUMBER_THREADS = omp_get_max_threads();

/*_______________________________________________________________________________________*/

#if FLAG_PRECISION == CV_64F

DICTIONARY::DICTIONARY(int m, int n, int nc, int lm, double ck): m(m), n(n), nc(nc), lm(lm), ck(ck) {

  nct = nc;
  numberZeros = m / 2; //Default value, can be changed
  numberDescriptors = 0; //At the start, this must be the value as the matrix is assumed to be empty
  nl = 0; //At the start, this must be the value as the matrix is assumed to be empty
  nr = 0; //At the start, this must be the value as the matrix is assumed to be empty
  maxNumberIterations = 2 * numberZeros; //Maximum number of iterations in sparse solution search
  fck = 1.0 / (2 * ck); //Criterion for the SHRINK function

  D = new Eigen::MatrixXd(m, n);
  CL = new int[n]; //Will store in each index the class corresponding to that column index in matrix D
  intermediateConstructor();

}

DICTIONARY::DICTIONARY(std::string nameFile) {

  FILE * fileDataBase;
  if ((fileDataBase = fopen(nameFile.c_str(), "rb")) == NULL) return;

  int parameters[6];

  fread(parameters, sizeof(int), 6, fileDataBase);
  m = parameters[0];
  n = parameters[1];
  lm = parameters[2];
  nc = parameters[3];
  numberDescriptors = parameters[4];
  numberZeros = parameters[5];

  fread( & ck, sizeof(double), 1, fileDataBase);

  //_______________________________________________default values_________________________________________//
  nct = nc;
  nl = 0; //At the start, this must be the value as the matrix is assumed to be empty
  nr = 0; //At the start, this must be the value as the matrix is assumed to be empty
  maxNumberIterations = 2 * numberZeros; //Maximum number of iterations in sparse solution search
  fck = 1.0 / (2 * ck); //Criterion for the SHRINK function
  //__________________________________________________________________________________________________________//

  D = new Eigen::MatrixXd(m, n);
  fread(D -> data(), sizeof(double), m * numberDescriptors, fileDataBase);

  CL = new int[n]; //Will store in each index the class corresponding to that column index in matrix D
  fread(CL, sizeof(int), numberDescriptors, fileDataBase);

  intermediateConstructor();
  normsWD(0, numberDescriptors);

  fclose(fileDataBase);

}

void DICTIONARY::intermediateConstructor() {

  //D=new Eigen::MatrixXd(m,n);
  //CL=new int[n];//Will store in each index the class corresponding to that column index in matrix D
  clt.reserve(n);

  arraycl = new std::vector < int > * [n];

  for (int i = 0; i < n; i++) {
    arraycl[i] = new std::vector < int > ;

  }
  //______________________________________________
  /*The reason for choosing nc matrix spaces is due to the way the extraction of submatrices from D was implemented, so that it took advantage of the multi-core*/
  DML = new Eigen::MatrixXd[nc];
  for (int i = 0; i < nc; i++)
    DML[i] = Eigen::MatrixXd::Zero(m, lm);
  //______________________________________________
  DN = new Eigen::MatrixXd(lm, lm * nc);

  WD = new Eigen::VectorXd(n);
  WD2 = new Eigen::VectorXd(n);
  WD2I = new Eigen::MatrixXd(lm, nc);
  Dsub2 = new Eigen::MatrixXd(lm, lm * nc);
  b = new Eigen::MatrixXd(m, nc);
  Wb = new Eigen::VectorXd(nc);
  Btemp = new Eigen::MatrixXd(n, nc);
  B = new Eigen::MatrixXd(n, nc);
  B_data = new DATA_ORDER(B, lm); //This links B with B_data

  BN = new Eigen::MatrixXd(lm, nc);
  BK_1 = new Eigen::MatrixXd(lm, nc);

  x = new Eigen::MatrixXd(lm, nc);
  xn = new Eigen::MatrixXd(lm, nc);
  u = new Eigen::MatrixXd(lm, nc);
  AE = new Eigen::MatrixXd(lm, nc);
  jn = new Eigen::VectorXi(nc);
  arr = new Eigen::VectorXi;
  ( * arr) = lm * Eigen::VectorXi::LinSpaced(nc, 0, nc - 1);
  jn_mod = new Eigen::VectorXi(nc);
  noZeros = new Eigen::RowVectorXi(nc);

}

void DICTIONARY::seeInfo() {

  std::cout << "\n\n\n";
  std::cout << "nzeros=" << numberZeros << " m=" << m << " n=" << n << " lm=" << lm << " nc=" << nc << " nct=" << nct << "\n";
  std::cout << "nl=" << nl << "  nr=" << nr << "\n";
  std::cout << "Number of descriptors=" << numberDescriptors << " ck=" << ck << "\n";

  std::cout << "x:\n" << ( * x) << "\n";
  std::cout << "DN:\n" << ( * DN) << "\n";
  std::cout << "BN:\n" << ( * BN) << "\n";
  std::cout << "ind:\n" << (B_data -> ind) << "\n";

  std::cout << "D:\n" << ( * D) << "\n";

  std::cout << "WD:\n" << (WD -> transpose()) << "\n";
  std::cout << "WD2:\n" << (WD2 -> transpose()) << "\n";

  std::cout << "CLASSES\n";
  for (int i = 0; i < n; i++)
    std::cout << CL[i] << " ";
  std::cout << "\n";

}

void DICTIONARY::clean() {
  ( * x) = Eigen::MatrixXd::Zero(lm, nc);
  ( * b) = Eigen::MatrixXd::Zero(m, nc);
}

void DICTIONARY::sendToMatlab() {

  int info[4];
  info[0] = nl;
  info[1] = nr;
  info[2] = m;
  info[3] = nct;

  FILE * file_info;
  if ((file_info = fopen("info", "wb")) == NULL) return;
  fwrite(info, sizeof(int), 4, file_info);

  Eigen::MatrixXd XS = Eigen::MatrixXd::Zero(n, nct);

  for (int j = 0; j < nct; j++) {
    for (int i = nl; i < nl + lm; i++) {
      XS(B_data -> ind(i, j), j) = ( * x)(i - nl, j);
    }
  }

  FILE * file_D;
  if ((file_D = fopen("matriz_D", "wb")) == NULL) return;
  FILE * file_b;
  if ((file_b = fopen("matriz_b", "wb")) == NULL) return;
  FILE * file_XS;
  if ((file_XS = fopen("matriz_XS", "wb")) == NULL) return;

  fwrite( & (( * D)(0, nl)), sizeof(double), m * (nr - nl + 1), file_D);
  fwrite(b -> data(), sizeof(double), m * nct, file_b);

  Eigen::MatrixXd XN = XS.block(nl, 0, (nr - nl + 1), nct);
  fwrite(XN.data(), sizeof(double), (nr - nl + 1) * nct, file_XS);

  fclose(file_info);
  fclose(file_D);
  fclose(file_b);
  fclose(file_XS);

}

void DICTIONARY::saveDataBase(std::string nameFile) {

  FILE * fileDataBase;
  if ((fileDataBase = fopen(nameFile.c_str(), "wb")) == NULL) return;

  int parameters[6];
  parameters[0] = m;
  parameters[1] = n;
  parameters[2] = lm;
  parameters[3] = nc;
  parameters[4] = numberDescriptors;
  parameters[5] = numberZeros;

  fwrite(parameters, sizeof(int), 6, fileDataBase);
  fwrite( & ck, sizeof(double), 1, fileDataBase);

  fwrite(D -> data(), sizeof(double), m * numberDescriptors, fileDataBase);

  fwrite(CL, sizeof(int), numberDescriptors, fileDataBase);

  fclose(fileDataBase);

}

void DICTIONARY::eigenPush(const Eigen::MatrixXd & M, int cl) {
  /*Use this method to insert a group of descriptors from an EIGEN library matrix*/

  if (M.rows() != m) return; //The matrix must have the same number of rows as the descriptor matrix D

  if ((M.cols() + numberDescriptors) <= n) {
    D -> middleCols(numberDescriptors, M.cols()) = M;
    normsWD(numberDescriptors, M.cols()); //Update the norm and squared norm of each column

    for (int i = numberDescriptors; i < numberDescriptors + M.cols(); i++) /*Here the class identification is updated*/
      CL[i] = cl;

    numberDescriptors = numberDescriptors + M.cols(); //Update numberDescriptors so the next push works properly
  } else {

    int aux = n * int((M.cols() / n) + 1); /*This value will always be greater than M.cols() and is enough to allocate a new memory block*/

    resizeDictionary(aux);

    D -> middleCols(numberDescriptors, M.cols()) = M;

    for (int i = numberDescriptors; i < numberDescriptors + M.cols(); i++) /*Here the class identification is updated*/
      CL[i] = cl;

    normsWD(numberDescriptors, M.cols()); //Update the norm and squared norm of each column
    numberDescriptors = numberDescriptors + M.cols(); //Update numberDescriptors so the next push works properly
  }
}

void DICTIONARY::opencvPush(cv::Mat & M, int cl) {
  /*Use this method to insert a group of descriptors from an OpenCV library matrix*/

  double * p = M.ptr < double > (0);
  Eigen::Map < Eigen::MatrixXd > newM(p, M.cols, M.rows);

  eigenPush(newM.transpose(), cl);

}

void DICTIONARY::pointerPush(double * pM, int cols, int cl) {
  /*Use this method to insert a group of descriptors from a double array*/
  /*NOTE: Assume that the memory addresses increase row by row*/

  Eigen::Map < Eigen::MatrixXd > newM(pM, m, cols);

  eigenPush(newM, cl);

}

//___________________________________________________________________________________________________________________

//_____________INTERFACE for eigen____________________//
void DICTIONARY::dispersedSolution(const Eigen::MatrixXd & Mb) {

  if ((Mb.rows() != m) || (Mb.cols() > nc)) return;

  b -> leftCols(Mb.cols()).noalias() = Mb;
  dispersedSolution_lowLevel(Mb.cols(), 0, numberDescriptors - 1);
}

void DICTIONARY::dispersedSolution(const Eigen::MatrixXd & Mb, int ncol) {

  if ((Mb.rows() != m) || (Mb.cols() > nc) || (Mb.cols() < ncol)) return;

  b -> leftCols(ncol).noalias() = Mb.leftCols(ncol);
  dispersedSolution_lowLevel(ncol, 0, numberDescriptors - 1);
}

void DICTIONARY::dispersedSolution(const Eigen::MatrixXd & Mb, int ncol, int vnr, int vnl) {

  if ((Mb.rows() != m) || (Mb.cols() > nc) || (Mb.cols() < ncol) || (vnr - vnl + 1 < lm)) return;

  b -> leftCols(ncol).noalias() = Mb.leftCols(ncol);
  dispersedSolution_lowLevel(ncol, vnl, vnr);

}

//____________INTERFACE FOR OPENCV____________________//

void DICTIONARY::dispersedSolution(cv::Mat & Mb) {

  double * p = Mb.ptr < double > (0);
  Eigen::Map < Eigen::MatrixXd > newMb(p, Mb.cols, Mb.rows);

  dispersedSolution(newMb.transpose());

}

void DICTIONARY::dispersedSolution(cv::Mat & Mb, int ncol) {

  double * p = Mb.ptr < double > (0);
  Eigen::Map < Eigen::MatrixXd > newMb(p, Mb.cols, Mb.rows);

  dispersedSolution(newMb.transpose(), ncol);

}

void DICTIONARY::dispersedSolution(cv::Mat & Mb, int ncol, int vnr, int vnl) {

  double * p = Mb.ptr < double > (0);
  Eigen::Map < Eigen::MatrixXd > newMb(p, Mb.cols, Mb.rows);

  dispersedSolution(newMb.transpose(), ncol, vnr, vnl);

}

//___________LOW-LEVEL INTERFACE TO AVOID COPYING VALUES____________

double * DICTIONARY::ptr() {
  return b -> data();
}

void DICTIONARY::dispersedSolution() {
  dispersedSolution_lowLevel(nc, 0, numberDescriptors - 1);
}

void DICTIONARY::dispersedSolution(int ncol) {
  dispersedSolution_lowLevel(ncol, 0, numberDescriptors - 1);
}

void DICTIONARY::dispersedSolution(int ncol, int vnr, int vnl) {
  dispersedSolution_lowLevel(ncol, vnl, vnr);
}

//_______________________________________________________________________________________________________________________________________

void DICTIONARY::set_lm(int new_lm) {
  /*This function will set the new lm, which cannot be greater than n, nor less than 1*/
  if ((new_lm < 1) || (new_lm > n)) return; //The function has no effect in these cases

  /*_____________All memory locations affected by lm must be resized___________________*/

  /*The reason for choosing nc matrix spaces is due to the way the extraction of submatrices from D was implemented, so that multi-core processing could be exploited*/

  lm = new_lm; //HERE lm is updated

  //______________________________________________
  delete[] DML; /*For efficiency, it is completely deleted to avoid memory leaks in each vector position due to the internal resize call if another zero matrix was assigned instead of deleting it*/
  DML = new Eigen::MatrixXd[nc];
  for (int i = 0; i < nc; i++)
    DML[i] = Eigen::MatrixXd::Zero(m, lm);
  //______________________________________________
  delete DN;
  DN = new Eigen::MatrixXd(lm, lm * nc);

  delete WD2I;
  WD2I = new Eigen::MatrixXd(lm, nc);
  delete Dsub2;
  Dsub2 = new Eigen::MatrixXd(lm, lm * nc);
  delete B_data;
  B_data = new DATA_ORDER(B, lm); //Here B is linked with B_data
  delete BN;
  BN = new Eigen::MatrixXd(lm, nc);
  delete BK_1;
  BK_1 = new Eigen::MatrixXd(lm, nc);
  delete x;
  x = new Eigen::MatrixXd(lm, nc);
  delete xn;
  xn = new Eigen::MatrixXd(lm, nc);
  delete u;
  u = new Eigen::MatrixXd(lm, nc);
  delete AE;
  AE = new Eigen::MatrixXd(lm, nc);
  ( * arr) = lm * Eigen::VectorXi::LinSpaced(nc, 0, nc - 1);

}

void DICTIONARY::set_nc(int new_nc) {
  /*This function pre-sets the maximum number of descriptors to be solved, the function has no effect if new_nc is less than 1*/

  if (new_nc < 1) return; /*The function has no effect*/

  /*The following must update all the variables dependent on nc*/

  nc = new_nc; //HERE nc is updated

  //______________________________________________
  delete[] DML;
  DML = new Eigen::MatrixXd[nc];
  for (int i = 0; i < nc; i++)
    DML[i] = Eigen::MatrixXd::Zero(m, lm);
  //______________________________________________
  delete DN;
  DN = new Eigen::MatrixXd(lm, lm * nc);
  delete WD2I;
  WD2I = new Eigen::MatrixXd(lm, nc);
  delete Dsub2;
  Dsub2 = new Eigen::MatrixXd(lm, lm * nc);
  delete b;
  b = new Eigen::MatrixXd(m, nc);
  delete Wb;
  Wb = new Eigen::VectorXd(nc);
  delete Btemp;
  Btemp = new Eigen::MatrixXd(n, nc);
  delete B;
  B = new Eigen::MatrixXd(n, nc);
  delete B_data;
  B_data = new DATA_ORDER(B, lm); //Here B is linked with B_data again
  delete BN;
  BN = new Eigen::MatrixXd(lm, nc);
  delete BK_1;
  BK_1 = new Eigen::MatrixXd(lm, nc);
  delete x;
  x = new Eigen::MatrixXd(lm, nc);
  delete xn;
  xn = new Eigen::MatrixXd(lm, nc);
  delete u;
  u = new Eigen::MatrixXd(lm, nc);
  delete AE;
  AE = new Eigen::MatrixXd(lm, nc);
  delete jn;
  jn = new Eigen::VectorXi(nc);
  delete arr;
  arr = new Eigen::VectorXi;
  ( * arr) = lm * Eigen::VectorXi::LinSpaced(nc, 0, nc - 1);
  delete jn_mod;
  jn_mod = new Eigen::VectorXi(nc);
  delete noZeros;
  noZeros = new Eigen::RowVectorXi(nc);

}

void DICTIONARY::set_numberZeros(int num) {
  numberZeros = num;
  maxNumberIterations = 2 * numberZeros;
}

void DICTIONARY::set_ck(double new_ck) {
  ck = new_ck;
  fck = 1.0 / (2 * ck); // Criterion for the SHRINK function
}

int DICTIONARY::get_lm() {
  return lm;
}

int DICTIONARY::get_nc() {
  return nc;
}

int DICTIONARY::get_nct() {
  return nct;
}

int DICTIONARY::get_numberZeros() {
  return numberZeros;
}

double DICTIONARY::get_ck() {
  return ck;
}

int DICTIONARY::get_m() {
  return m;
}

int DICTIONARY::get_n() {
  return n;
}

int DICTIONARY::get_numberDescriptors() {
  return numberDescriptors;
}

void DICTIONARY::get_completeSolution(Eigen::MatrixXd & XS, Eigen::MatrixXd & XA, int index) {
  XS = Eigen::MatrixXd::Zero(numberDescriptors, 1);
  XA = Eigen::MatrixXd::Zero(m, 1);

  for (int i = nl; i < nl + lm; i++) {
    XS(B_data -> ind(i, index), 0) = ( * x)(i - nl, index);
    XA = XA + XS(B_data -> ind(i, index), 0) * (DML[index]).col(i - nl);
  }

  /*
  XS=Eigen::MatrixXd::Zero(numberDescriptors,1);

  for(int i=nl;i<nl+lm;i++){
    XS(B_data->ind(i,index),0)=(*x)(i-nl,index);
  }

  XA.noalias()=D->middleCols(0,numberDescriptors)*XS;
  */
}

Eigen::MatrixXd DICTIONARY::bDescriptor(int index) const {
  return b -> col(index);
}

int * DICTIONARY::get_CL() const {
  return CL;
}

std::vector < int > DICTIONARY::get_clt() const {
  return clt;
}

std::vector < double > DICTIONARY::get_clustersDistance() const;
{
  return clustersDistance;
}

void DICTIONARY::normsWD(int c, int sz) {
  WD2 -> middleRows(c, sz).noalias() = (D -> middleCols(c, sz).colwise().squaredNorm()); /* Square norm calculation for each column of matrix D */

  WD -> middleRows(c, sz).noalias() = WD2 -> middleRows(c, sz).array().sqrt().matrix(); /* Norm calculation for each column of matrix D */
}

void DICTIONARY::resizeDictionary(int sz) {
  /* This function will handle increasing memory once the initial limit is reached,
  the size will double for all the involved variables */

  int * aux = new int[n + sz];
  for (int i = 0; i < n; i++)
    aux[i] = CL[i];
  delete CL;
  CL = aux;

  //_______________________________
  for (int i = 0; i < n; i++) {
    delete arraycl[i];
  }

  delete[] arraycl;
  //________________________________
  D -> conservativeResize(Eigen::NoChange, n + sz);
  WD -> conservativeResize(n + sz);
  WD2 -> conservativeResize(n + sz);
  n = D -> cols(); // n should be updated
  clt.resize(n);
  clt.clear();
  //________________________________
  arraycl = new std::vector < int > * [n];

  for (int i = 0; i < n; i++) {
    arraycl[i] = new std::vector < int > ;
  }
  //________________________________

  Btemp -> resize(n, Eigen::NoChange); // Remember that n has changed    
  B -> resize(n, Eigen::NoChange); // Remember that n has changed
  delete B_data;
  B_data = new DATA_ORDER(B, lm); // Here it is linked with B again
}

void DICTIONARY::dispersedSolution_lowLevel(int vnct, int vnl, int vnr) {

  //_________
  nct = vnct;
  nl = vnl;
  nr = vnr;
  //_________

  Btemp -> block(nl, 0, nr - nl + 1, nct).noalias() = (D -> middleCols(nl, nr - nl + 1).transpose() * (b -> leftCols(nct))); // Matrix D'*b multiplication
  Wb -> topRows(nct).noalias() = b -> leftCols(nct).colwise().norm().transpose(); // Calculation of the norm for each column of matrix b

  /*
  std::cout<<"Btemp:\n"<<(*Btemp)<<"\n";
  std::cout<<"Wb:\n"<<(*Wb)<<"\n";
  getchar();
  */
  /* Correlation */

  B -> block(nl, 0, nr - nl + 1, nct).noalias() = (((Btemp -> block(nl, 0, nr - nl + 1, nct).array().rowwise()) / (Wb -> topRows(nct).transpose().array())).colwise() / (WD -> middleRows(nl, nr - nl + 1).array())).matrix();

  /*
  std::cout<<"Wb:\n"<<(Wb->transpose())<<"\n";
  std::cout<<"WD:\n"<<(WD->transpose())<<"\n";
  std::cout<<"WD:\n"<<(WD->middleRows(nl,nr-nl+1))<<"\n";
  std::cout<<"B:\n"<<(*B)<<"\n";
  getchar();
  */

  B_data -> reset_ind(nct, nl, nr); // Indices must be reset for each new solution to be found
  B_data -> nth_element(nct, nl, nr); // Sorting the correlation to choose the highest lm values of each B column 
  //B_data->show_ind();

  /*
  std::cout<<"ind:\n"<<(B_data->ind.block(nl,0,lm,nct))<<"\n";
  getchar();
  */

  // Extracting values from Btemp according to the highest values of each column in B
  for (int j = 0; j < nct; j++) {
    /*time=0*/
    for (int i = 0; i < lm; i++) {
      (BN -> data())[i + lm * j] = ( * Btemp)(B_data -> p_ind[n * j + i + nl], j);
    }
  }

  //std::cout<<"BN:\n"<<(*BN)<<"\n";
  //B_data->show_ind();

  // GENERATING SUBMATRICES OF D'*D TO SOLVE AND THE WD2I (SQUARE NORM CORRESPONDING TO THE ITERATION)
  #pragma omp parallel for schedule(dynamic)
  for (int k = 0; k < nct; k++) {
    for (int i = 0; i < lm; i++) {
      (DML[k]).col(i).noalias() = D -> col(B_data -> ind(i + nl, k));
      ( * WD2I)(i, k) = ( * WD2)(B_data -> ind(i + nl, k));
    }
    DN -> middleCols(k * lm, lm).noalias() = (DML[k]).transpose() * (DML[k]);

    for (int j = 0; j < lm; j++)
      ( * DN)(j, j + k * lm) = 0;
  }

  /*
  std::cout<<"D"<<(*D)<<"\n";
  for(int i=0;i<nc;i++)
  std::cout<<"DML["<<i<<"]:\n"<<DML[i]<<"\n";
  std::cout<<"DN:\n"<<(*DN)<<"\n";
  std::cout<<"WD2I:\n"<<(*WD2I)<<"\n";
  */

  //__________________Before iteration________________
  x -> leftCols(nct).noalias() = Eigen::MatrixXd::Zero(lm, nct);
  expectedReadyVectors = 0;
  ith = 0;
  //maxNumberIterations=2*numberZeros;
  while ((expectedReadyVectors < nct) && (ith < maxNumberIterations)) {

    u -> leftCols(nct) = (BN -> leftCols(nct).array() > fck).select((BN -> leftCols(nct).array() - fck), (BN -> leftCols(nct).array() < -fck).select((BN -> leftCols(nct).array() + fck), 0));

    u -> leftCols(nct).array() = (u -> leftCols(nct).array() / WD2I -> leftCols(nct).array());

    AE -> leftCols(nct).array() = ck * ((x -> leftCols(nct)) - (u -> leftCols(nct))).array() * ((WD2I -> leftCols(nct).array() * ((u -> leftCols(nct)) + (x -> leftCols(nct))).array()) - 2 * BN -> leftCols(nct).array()) + x -> leftCols(nct).array().abs() - u -> leftCols(nct).array().abs();

    for (int i = 0; i < nct; i++) {
      for (int j = 0; j < lm; j++) {
        Btemp -> data()[j + lm * i] = 0;
      }
    }

    /*
    if(ith%10==0){
    std::cout<<"X:\n"<<(*x)<<"\n";
    std::cout<<"U:\n"<<(*u)<<"\n";
    std::cout<<"AE:\n"<<(*AE)<<"\n";
    }
    getchar();
    */
    expectedReadyVectors = Btemp ->size();

    ith++;
  }
}

int DICTIONARY::estimateCluster(double & expectedPercentageDifference) {

  /*
  std::cout<<"Here we are "<<nl<<" "<<nr<<"  "<<nct<<"\n";

  std::cout<<"ind:\n"<<(B_data->ind)<<"\n";
  getchar();

  std::cout<<"\nx:\n"<<(*x)<<"\n";

  std::cout<<"Selected indices:\n";


  for(int i=nl;i<nl+lm;i++) //Iterate over rows where the sparse solution was previously found 
  {
  for(int j=0;j<nct;j++) //Iterate over columns where the sparse solution is found 
    {
       
        std::cout<<B_data->ind(i,j)<<" ";
        

        }    
       std::cout<<"\n";
     }

  std::cout<<"Cluster before:\n";
  for(int i=0;i<clt.size();i++)
  std::cout<<clt[i]<<" ";
  std::cout<<"\n";

  Eigen::MatrixXd MM(lm,nct);
  std::cout<<"Chosen classes:\n";
   */

  clt.clear(); //Cleared initially to allow data extraction, modified on October 19, 2015

  for (int j = 0; j < nct; j++) //Iterate over columns where the sparse solution is found 
  {

    int iaux = 0, col = lm * j, colr = n * j;
    int * p = & (B_data -> ind(0, j));

    for (int i = nl; i < nl + lm; i++, iaux++) //Iterate over rows where the sparse solution was previously found 
    {

      //MM(i-nl,j)=CL[p[i]];
      int aux_cl = CL[p[i]];
      clt.push_back(aux_cl);
      (arraycl[aux_cl]) -> push_back(iaux + col);

    }

  }

  /*
  std::cout<<"Classes:\n";
  for(int i=0;i<n;i++)
  std::cout<<CL[i]<<" ";
  std::cout<<"\n";
  */

  /*
  std::cout<<"Cluster after:\n";
  for(int i=0;i<clt.size();i++)
  std::cout<<clt[i]<<" ";
  std::cout<<"\n";
  */

  std::sort(clt.begin(), clt.end());
  clt.erase(std::unique(clt.begin(), clt.end()), clt.end());

  /*
  std::cout<<"Class matrix:\n"<<MM<<"\n";
  std::cout<<"Classes:\n";
  for(int i=0;i<clt.size();i++)
  std::cout<<clt[i]<<" ";
  std::cout<<"\n";
  */

  /*
  std::cout<<"Table:\n";
  for(int i=0;i<clt.size();i++){
   int sz=(arraycl[clt[i]])->size();
    for(int j=0;j<sz;j++){
    std::cout<<(*arraycl[clt[i]])[j]<<" ";
    }
    std::cout<<"\n";
  }
  */

  //Eigen::MatrixXd x_aux;
  double * score_cl = new double[clt.size()];
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < clt.size(); i++) {
    Eigen::MatrixXd x_aux = Eigen::MatrixXd::Zero(lm, nct);
    for (int j = 0; j < (arraycl[clt[i]]) -> size(); j++) {
      (x_aux.data())[( * arraycl[clt[i]])[j]] = (x -> data())[( * arraycl[clt[i]])[j]];
    }

    double temp = 0;
    for (int ii = 0; ii < nct; ii++) {

      /*    
          std::cout<<"Column="<<ii<<" matrix="<<ii<<"\n";
          std::cout<<"x_aux:\n"<<x_aux<<"\n";
          std::cout<<"x_aux:\n"<<x_aux.col(ii)<<"\n";
          std::cout<<"DML:\n"<<DML[ii]<<"\n";
          std::cout<<"b:\n"<<(*b)<<"\n";
          std::cout<<"Decision="<<x_aux.col(ii).isZero(0)<<"\n";
       */

      if (x_aux.col(ii).isZero(0))
        temp = temp + b -> col(ii).norm();
      else {
        Eigen::MatrixXd rs = Eigen::MatrixXd::Zero(m, 1);
        Eigen::MatrixXd x_temp = x_aux.col(ii);
        //std::cout<<"x_temp:\n"<<x_temp<<"\n";
        for (int j = 0; j < lm; j++) {
          if (x_temp(j) != 0) {
            rs = rs + (x_temp(j)) * ((DML[ii]).col(j));
            /*
                std::cout<<"Temp values="<<x_temp(j)<<"\n";
                std::cout<<"((DML[ii]).col(j)):\n"<<((DML[ii]).col(j))<<"\n";
                std::cout<<"Entered:\n"<<rs<<"\n";
                getchar();
            */
          }

        }

        temp = temp + (b -> col(ii) - rs).norm();

      }

      //std::cout<<"Final temp="<<temp<<"\n";

      /*std::cout<<"ii="<<ii<<"\n";
      std::cout<<"x_aux:\n"<<x_aux<<"\n";
      std::cout<<"DML:\n"<<DML[ii]<<"\n";
      std::cout<<"b:\n"<<(*b)<<"\n";
      std::cout<<"temp="<<temp<<"\n";*/
      //getchar();

    }

    score_cl[i] = temp;

  }

  /*
  for(int i=0;i<clt.size();i++)
  std::cout<<"Class="<<clt[i]<<" value="<<score_cl[i]<<"\n";
  */

  double * min_score = std::min_element(score_cl, score_cl + clt.size());
  int indMinScore = std::distance(score_cl, min_score);
  int classChosen = clt[indMinScore];

  //____________SCORE CALCULATION________________________

  /*

  Initially, the score_cl function is inverted relative to the maximum point, meaning the maximum becomes zero and the minimum becomes the maximum. Then, the mean is calculated without considering the maximum value in this inverted function, and this value is compared with the isolated maximum value to calculate the mean. The comparison is done using the percentage difference, with a weight k applied to the subtracted value. The formula is shown below:

  expectedPercentageDifference=(max-k*(m))/max=1-k*(m/max)

  m is the mean and max is the isolated maximum value for calculating the mean in the inverted function. 

  */

  /*
  std::cout<<"score_cl=[";
  for(int i=0;i<clt.size();i++)
  std::cout<<score_cl[i]<<" ";
  std::cout<<"]\n\n";
  */

  //This vector will be useful for future analysis
  clustersDistance.clear();
  clustersDistance.reserve(clt.size());
  for (int i = 0; i < clt.size(); i++)
    clustersDistance.push_back(score_cl[i]);

  double * max_score = std::max_element(score_cl, score_cl + clt.size());
  double maxNscores = ( * max_score) - ( * min_score);
  double sScores = 0;
  for (int i = 0; i < clt.size(); i++)
    sScores = sScores + score_cl[i];

  double meanScores = ( * max_score) + ((( * min_score) - sScores) / (clt.size() - 1));

  expectedPercentageDifference = 1 - 2 * (meanScores / (maxNscores));

  //____________________________________________________________

  delete score_cl;
  //_____________CLEANING clt AND USED POSITIONS IN arraycl____________//
  for (int i = 0; i < clt.size(); i++) {
    for (int j = 0; j < (arraycl[clt[i]]) -> size(); j++) {
      (arraycl[clt[i]]) -> clear();
    }
  }

  //clt.clear();
  //___________________________________________________________________________//

  return classChosen; //Return the estimated class to which the group of descriptors in b possibly belongs

}

DICTIONARY::~DICTIONARY() {

  delete D;
  delete CL;

  //_______________________________
  for (int i = 0; i < n; i++) {
    delete arraycl[i];

  }

  delete[] arraycl;
  //________________________________

  delete[] DML;
  delete DN;
  delete WD;
  delete WD2;
  delete WD2I;
  delete Dsub2;
  delete b;
  delete B;
  delete B_data;
  delete BN;
  delete BK_1;
  delete x;
  delete xn;
  delete u;
  delete AE;
  delete jn;
  delete arr;
  delete jn_mod;
  delete noZeros;

  //Missing on September 21, 2015
  delete Wb;
  delete Btemp;

  std::cout << "Dictionary deleted\n";

}

#else

DICTIONARY::DICTIONARY(int m, int n, int nc, int lm, float ck): m(m), n(n), nc(nc), lm(lm), ck(ck) {

  nct = nc;
  numberZeros = m / 2; //Default value with the option to change
  numberDescriptors = 0; //At the beginning, this should be the value since the matrix is assumed to be empty
  nl = 0; //At the beginning, this should be the value since the matrix is assumed to be empty
  nr = 0; //At the beginning, this should be the value since the matrix is assumed to be empty
  //maxNumberIterations=2*numberZeros;//Maximum number of iterations for sparse solution search
  fck = 1.0 / (2 * ck); //Criterion for the SHRINK function

  D = new Eigen::MatrixXf(m, n);
  CL = new int[n]; //Will store the class corresponding to each index column for matrix D
  intermediateConstructor();

}

DICTIONARY::DICTIONARY(std::string nameFile) {

  FILE * fileDataBase;
  if ((fileDataBase = fopen(nameFile.c_str(), "rb")) == NULL) return;

  int parameters[6];

  fread(parameters, sizeof(int), 6, fileDataBase);
  m = parameters[0];
  n = parameters[1];
  lm = parameters[2];
  nc = parameters[3];
  numberDescriptors = parameters[4];
  numberZeros = parameters[5];

  fread( & ck, sizeof(float), 1, fileDataBase);

  //_______________________________________________Default values_________________________________________//
  nct = nc;
  nl = 0; //When starting, this should be the value since the matrix is assumed to be empty
  nr = 0; //When starting, this should be the value since the matrix is assumed to be empty
  maxNumberIterations = 2 * numberZeros; //Maximum number of iterations in the search for sparse solutions
  fck = 1.0 / (2 * ck); //Criterion for the SHRINK function
  //__________________________________________________________________________________________________//

  D = new Eigen::MatrixXf(m, n);
  fread(D -> data(), sizeof(float), m * numberDescriptors, fileDataBase);

  CL = new int[n]; //Will store the class corresponding to each column index for the matrix D
  fread(CL, sizeof(int), numberDescriptors, fileDataBase);

  intermediateConstructor();
  normsWD(0, numberDescriptors);

  fclose(fileDataBase);

}

void DICTIONARY::intermediateConstructor() {

  //D=new Eigen::MatrixXf(m,n);
  //CL=new int[n]; //Will store the class corresponding to each column index for the matrix D
  clt.reserve(n);

  arraycl = new std::vector < int > * [n];

  for (int i = 0; i < n; i++) {
    arraycl[i] = new std::vector < int > ;

  }
  //______________________________________________
  /*The reason for choosing nc matrix spaces is due to the way the extraction of submatrices from D was implemented, to leverage the multi-core system*/
  DML = new Eigen::MatrixXf[nc];
  for (int i = 0; i < nc; i++)
    DML[i] = Eigen::MatrixXf::Zero(m, lm);
  //______________________________________________
  DN = new Eigen::MatrixXf(lm, lm * nc);

  WD = new Eigen::VectorXf(n);
  WD2 = new Eigen::VectorXf(n);
  WD2I = new Eigen::MatrixXf(lm, nc);
  Dsub2 = new Eigen::MatrixXf(lm, lm * nc);
  b = new Eigen::MatrixXf(m, nc);
  Wb = new Eigen::VectorXf(nc);
  Btemp = new Eigen::MatrixXf(n, nc);
  B = new Eigen::MatrixXf(n, nc);
  B_data = new DATA_ORDER(B, lm); //Here, B is linked with B_data

  BN = new Eigen::MatrixXf(lm, nc);
  BK_1 = new Eigen::MatrixXf(lm, nc);

  x = new Eigen::MatrixXf(lm, nc);
  xn = new Eigen::MatrixXf(lm, nc);
  u = new Eigen::MatrixXf(lm, nc);
  AE = new Eigen::MatrixXf(lm, nc);
  jn = new Eigen::VectorXi(nc);
  arr = new Eigen::VectorXi;
  (*arr) = lm * Eigen::VectorXi::LinSpaced(nc, 0, nc - 1);
  jn_mod = new Eigen::VectorXi(nc);
  noZeros = new Eigen::RowVectorXi(nc);

}

void DICTIONARY::seeInfo() {

  /*
  std::cout<<"\n\n\n";
  std::cout<<"nzeros="<<numberZeros<<" m="<<m<<" n="<<n<<" lm="<<lm<<" nc="<<nc<<" nct="<<nct<<"\n";
  std::cout<<"nl="<<nl<<"  nr="<<nr<<"\n";
  std::cout<<"Number of descriptors="<<numberDescriptors<<" ck="<<ck<<"\n";


  std::cout<<"x:\n"<<(*x)<<"\n";
  std::cout<<"DN:\n"<<(*DN)<<"\n";
  std::cout<<"BN:\n"<<(*BN)<<"\n";
  std::cout<<"ind:\n"<<(B_data->ind)<<"\n";
  */
  std::cout << "ind:\n" << (B_data -> ind) << "\n";
  std::cout << "x:\n" << ( * x) << "\n";
  std::cout << "D:\n" << ( * D) << "\n";

  std::cout << "WD:\n" << (WD -> transpose()) << "\n";
  std::cout << "WD2:\n" << (WD2 -> transpose()) << "\n";

  std::cout << "CLASSES\n";
  for (int i = 0; i < n; i++)
    std::cout << CL[i] << " ";
  std::cout << "\n";

}

void DICTIONARY::clean() {
  ( * x) = Eigen::MatrixXf::Zero(lm, nc);
  ( * b) = Eigen::MatrixXf::Zero(m, nc);
}

void DICTIONARY::sendToMatlab() {

  int info[4];
  info[0] = nl;
  info[1] = nr;
  info[2] = m;
  info[3] = nct;

  FILE * file_info;
  if ((file_info = fopen("info", "wb")) == NULL) return;
  fwrite(info, sizeof(int), 4, file_info);

  Eigen::MatrixXf XS = Eigen::MatrixXf::Zero(n, nct);

  for (int j = 0; j < nct; j++) {
    for (int i = nl; i < nl + lm; i++) {
      XS(B_data -> ind(i, j), j) = ( * x)(i - nl, j);
    }
  }

  FILE * file_D;
  if ((file_D = fopen("matrix_D", "wb")) == NULL) return;
  FILE * file_b;
  if ((file_b = fopen("matrix_b", "wb")) == NULL) return;
  FILE * file_XS;
  if ((file_XS = fopen("matrix_XS", "wb")) == NULL) return;

  fwrite( & (( * D)(0, nl)), sizeof(float), m * (nr - nl + 1), file_D);
  fwrite(b -> data(), sizeof(float), m * nct, file_b);

  Eigen::MatrixXf XN = XS.block(nl, 0, (nr - nl + 1), nct);
  fwrite(XN.data(), sizeof(float), (nr - nl + 1) * nct, file_XS);

  fclose(file_info);
  fclose(file_D);
  fclose(file_b);
  fclose(file_XS);

}

void DICTIONARY::saveDataBase(std::string nameFile) {

  FILE * fileDataBase;
  if ((fileDataBase = fopen(nameFile.c_str(), "wb")) == NULL) return;

  int parameters[6];
  parameters[0] = m;
  parameters[1] = n;
  parameters[2] = lm;
  parameters[3] = nc;
  parameters[4] = numberDescriptors;
  parameters[5] = numberZeros;

  fwrite(parameters, sizeof(int), 6, fileDataBase);
  fwrite( & ck, sizeof(float), 1, fileDataBase);

  fwrite(D -> data(), sizeof(float), m * numberDescriptors, fileDataBase);

  fwrite(CL, sizeof(int), numberDescriptors, fileDataBase);

  fclose(fileDataBase);

}

void DICTIONARY::eigenPush(const Eigen::MatrixXf & M, int cl) {
  /*Use this method to insert a group of descriptors from an EIGEN matrix*/

  if (M.rows() != m) return; //The matrix must have the same number of rows as the descriptor matrix D

  if ((M.cols() + numberDescriptors) <= n) {
    D -> middleCols(numberDescriptors, M.cols()) = M;
    normsWD(numberDescriptors, M.cols()); //Update the norm and square norm of each column

    for (int i = numberDescriptors; i < numberDescriptors + M.cols(); i++) /*Here, the class identification is updated*/
      CL[i] = cl;

    numberDescriptors = numberDescriptors + M.cols(); //The numberDescriptors must be updated so the next push works well
  } else {

    int aux = n * int((M.cols() / n) + 1); /*This value will always be greater than M.cols() and is enough to leave a new memory block ready*/

    resizeDictionary(aux);

    D -> middleCols(numberDescriptors, M.cols()) = M;

    for (int i = numberDescriptors; i < numberDescriptors + M.cols(); i++) /*Here, the class identification is updated*/
      CL[i] = cl;

    normsWD(numberDescriptors, M.cols()); //Update the norm and square norm of each column
    numberDescriptors = numberDescriptors + M.cols(); //The numberDescriptors must be updated so the next push works well
  }

}

void DICTIONARY::opencvPush(cv::Mat & M, int cl) {
  /*Use this method to insert a group of descriptors from an openCV matrix*/

  float * p = M.ptr < float > (0);
  Eigen::Map < Eigen::MatrixXf > newM(p, M.cols, M.rows);

  eigenPush(newM.transpose(), cl);

}

void DICTIONARY::pointerPush(float * pM, int cols, int cl) {
  /* Use this method to insert a group of descriptors from a float vector */
  /* NOTE: Assume that the increasing order of addresses goes by rows */

  Eigen::Map < Eigen::MatrixXf > newM(pM, m, cols);

  eigenPush(newM, cl);

}

//___________________________________________________________________________________________________________________

//_____________INTERFACE for eigen____________________//
void DICTIONARY::dispersedSolution(const Eigen::MatrixXf & Mb) {

  if ((Mb.rows() != m) || (Mb.cols() > nc)) return;

  b -> leftCols(Mb.cols()).noalias() = Mb;
  dispersedSolution_lowLevel(Mb.cols(), 0, numberDescriptors - 1);
}

void DICTIONARY::dispersedSolution(const Eigen::MatrixXf & Mb, int ncol) {

  if ((Mb.rows() != m) || (Mb.cols() > nc) || (Mb.cols() < ncol)) return;

  b -> leftCols(ncol).noalias() = Mb.leftCols(ncol);
  dispersedSolution_lowLevel(ncol, 0, numberDescriptors - 1);
}

void DICTIONARY::dispersedSolution(const Eigen::MatrixXf & Mb, int ncol, int vnr, int vnl) {

  if ((Mb.rows() != m) || (Mb.cols() > nc) || (Mb.cols() < ncol) || (vnr - vnl + 1 < lm)) return;

  b -> leftCols(ncol).noalias() = Mb.leftCols(ncol);
  dispersedSolution_lowLevel(ncol, vnl, vnr);

}

//____________INTERFACE FOR OPENCV____________________//

void DICTIONARY::dispersedSolution(cv::Mat & Mb) {

  float * p = Mb.ptr < float > (0);
  Eigen::Map < Eigen::MatrixXf > newMb(p, Mb.cols, Mb.rows);

  dispersedSolution(newMb.transpose());

}

void DICTIONARY::dispersedSolution(cv::Mat & Mb, int ncol) {

  float * p = Mb.ptr < float > (0);
  Eigen::Map < Eigen::MatrixXf > newMb(p, Mb.cols, Mb.rows);

  dispersedSolution(newMb.transpose(), ncol);

}

void DICTIONARY::dispersedSolution(cv::Mat & Mb, int ncol, int vnr, int vnl) {

  float * p = Mb.ptr < float > (0);
  Eigen::Map < Eigen::MatrixXf > newMb(p, Mb.cols, Mb.rows);

  dispersedSolution(newMb.transpose(), ncol, vnr, vnl);

}

//___________LOW-LEVEL INTERFACE TO AVOID COPYING VALUES____________

float * DICTIONARY::ptr() {
  return b -> data();
}

void DICTIONARY::dispersedSolution() {
  dispersedSolution_lowLevel(nc, 0, numberDescriptors - 1);
}

void DICTIONARY::dispersedSolution(int ncol) {
  dispersedSolution_lowLevel(ncol, 0, numberDescriptors - 1);
}

void DICTIONARY::dispersedSolution(int ncol, int vnr, int vnl) {
  dispersedSolution_lowLevel(ncol, vnl, vnr);
}

//_______________________________________________________________________________________________________________________________________

void DICTIONARY::set_lm(int new_lm) {
  /* This function will set the new lm, which cannot be greater than n, nor less than 1 */
  if ((new_lm < 1) || (new_lm > n)) return; // The function has no effect in these cases

  /*_____________All memory locations affected by lm should be adjusted___________________*/

  /* The reason for choosing nc matrix spaces is due to the way the submatrix extraction from D was implemented, taking advantage of multi-core processing */

  lm = new_lm; // HERE lm is updated

  //______________________________________________
  delete[] DML; /* For efficiency, it is completely deleted to avoid memory leaks in each position of the vector due to the internal resize call if another zero matrix were assigned instead of deleting */
  DML = new Eigen::MatrixXf[nc];
  for (int i = 0; i < nc; i++)
    DML[i] = Eigen::MatrixXf::Zero(m, lm);
  //______________________________________________
  delete DN;
  DN = new Eigen::MatrixXf(lm, lm * nc);

  delete WD2I;
  WD2I = new Eigen::MatrixXf(lm, nc);
  delete Dsub2;
  Dsub2 = new Eigen::MatrixXf(lm, lm * nc);
  delete B_data;
  B_data = new DATA_ORDER(B, lm); // Here B is linked with B_data
  delete BN;
  BN = new Eigen::MatrixXf(lm, nc);
  delete BK_1;
  BK_1 = new Eigen::MatrixXf(lm, nc);
  delete x;
  x = new Eigen::MatrixXf(lm, nc);
  delete xn;
  xn = new Eigen::MatrixXf(lm, nc);
  delete u;
  u = new Eigen::MatrixXf(lm, nc);
  delete AE;
  AE = new Eigen::MatrixXf(lm, nc);
  (*arr) = lm * Eigen::VectorXi::LinSpaced(nc, 0, nc - 1);
}

void DICTIONARY::set_nc(int new_nc) {
  /* This function pre-sets the maximum number of possible descriptors to be solved, the function has no effect if new_nc is less than 1 */

  if (new_nc < 1) return; /* The function has no effect */

  /* The following should update all variables that depend on nc */

  nc = new_nc; // HERE nc is updated

  //______________________________________________
  delete[] DML;
  DML = new Eigen::MatrixXf[nc];
  for (int i = 0; i < nc; i++)
    DML[i] = Eigen::MatrixXf::Zero(m, lm);
  //______________________________________________
  delete DN;
  DN = new Eigen::MatrixXf(lm, lm * nc);
  delete WD2I;
  WD2I = new Eigen::MatrixXf(lm, nc);
  delete Dsub2;
  Dsub2 = new Eigen::MatrixXf(lm, lm * nc);
  delete b;
  b = new Eigen::MatrixXf(m, nc);
  delete Wb;
  Wb = new Eigen::VectorXf(nc);
  delete Btemp;
  Btemp = new Eigen::MatrixXf(n, nc);
  delete B;
  B = new Eigen::MatrixXf(n, nc);
  delete B_data;
  B_data = new DATA_ORDER(B, lm); // Here B is linked with B_data again
  delete BN;
  BN = new Eigen::MatrixXf(lm, nc);
  delete BK_1;
  BK_1 = new Eigen::MatrixXf(lm, nc);
  delete x;
  x = new Eigen::MatrixXf(lm, nc);
  delete xn;
  xn = new Eigen::MatrixXf(lm, nc);
  delete u;
  u = new Eigen::MatrixXf(lm, nc);
  delete AE;
  AE = new Eigen::MatrixXf(lm, nc);
  delete jn;
  jn = new Eigen::VectorXi(nc);
  delete arr;
  arr = new Eigen::VectorXi;
  (*arr) = lm * Eigen::VectorXi::LinSpaced(nc, 0, nc - 1);
  delete jn_mod;
  jn_mod = new Eigen::VectorXi(nc);
  delete noZeros;
  noZeros = new Eigen::RowVectorXi(nc);
}

void DICTIONARY::set_numberZeros(int num) {
  numberZeros = num;
  maxNumberIterations = 2 * numberZeros;
}

void DICTIONARY::set_ck(float new_ck) {
  ck = new_ck;
  fck = 1.0 / (2 * ck); // Criterion for the SHRINK function
}

int DICTIONARY::get_lm() {
  return lm;
}

int DICTIONARY::get_nc() {
  return nc;
}

int DICTIONARY::get_nct() {
  return nct;
}

int DICTIONARY::get_numberZeros() {
  return numberZeros;
}

float DICTIONARY::get_ck() {
  return ck;
}

int DICTIONARY::get_m() {
  return m;
}

int DICTIONARY::get_n() {
  return n;
}

int DICTIONARY::get_numberDescriptors() {
  return numberDescriptors;
}

void DICTIONARY::get_completeSolution(Eigen::MatrixXf & XS, Eigen::MatrixXf & XA, int index) {

  XS = Eigen::MatrixXf::Zero(numberDescriptors, 1);
  XA = Eigen::MatrixXf::Zero(m, 1);

  for (int i = nl; i < nl + lm; i++) {
    XS(B_data -> ind(i, index), 0) = (*x)(i - nl, index);
    XA = XA + XS(B_data -> ind(i, index), 0) * (DML[index]).col(i - nl);
  }
}

Eigen::MatrixXf DICTIONARY::bDescriptor(int index) const {
  return b -> col(index);
}

int * DICTIONARY::get_CL() const {
  return CL;
}

std::vector < int > DICTIONARY::get_clt() const {
  return clt;
}

std::vector < float > DICTIONARY::get_clustersDistance() const {
  return clustersDistance;
}

void DICTIONARY::normsWD(int c, int sz) {

  WD2 -> middleRows(c, sz).noalias() = (D -> middleCols(c, sz).colwise().squaredNorm()); /* Square norm calculation of each column of the D matrix */

  WD -> middleRows(c, sz).noalias() = WD2 -> middleRows(c, sz).array().sqrt().matrix(); /* Norm calculation of each column of the D matrix */
}

void DICTIONARY::resizeDictionary(int sz) {
  /*This function will handle increasing memory when the initial limit has been reached,
  the size will be doubled for all the involved variables*/

  int * aux = new int[n + sz];
  for (int i = 0; i < n; i++)
    aux[i] = CL[i];
  delete CL;
  CL = aux;

  //_______________________________
  for (int i = 0; i < n; i++) {
    delete arraycl[i];
  }

  delete[] arraycl;
  //________________________________
  D -> conservativeResize(Eigen::NoChange, n + sz);
  WD -> conservativeResize(n + sz);
  WD2 -> conservativeResize(n + sz);
  n = D -> cols(); //n MUST BE UPDATED
  clt.resize(n);
  clt.clear();
  //________________________________
  arraycl = new std::vector < int > * [n];

  for (int i = 0; i < n; i++) {
    arraycl[i] = new std::vector < int > ;
  }
  //________________________________

  Btemp -> resize(n, Eigen::NoChange); //Remember that n changed    
  B -> resize(n, Eigen::NoChange); //Remember that n changed
  delete B_data;
  B_data = new DATA_ORDER(B, lm); //Here it is linked to B again
}

void DICTIONARY::dispersedSolution_lowLevel(int vnct, int vnl, int vnr) {

  //_________
  nct = vnct;
  nl = vnl;
  nr = vnr;
  //_________

  Btemp -> block(nl, 0, nr - nl + 1, nct).noalias() = (D -> middleCols(nl, nr - nl + 1).transpose() * (b -> leftCols(nct))); //Multiplying D'*b
  Wb -> topRows(nct).noalias() = b -> leftCols(nct).colwise().norm().transpose(); //Calculating the norm of each column of matrix b

  /*
  std::cout<<"Btemp:\n"<<(*Btemp)<<"\n";
  std::cout<<"Wb:\n"<<(*Wb)<<"\n";
  getchar();
  */
  /*Correlation*/

  B -> block(nl, 0, nr - nl + 1, nct).noalias() = (((Btemp -> block(nl, 0, nr - nl + 1, nct).array().rowwise()) / (Wb -> topRows(nct).transpose().array())).colwise() / (WD -> middleRows(nl, nr - nl + 1).array())).matrix();

  /*
  std::cout<<"Wb:\n"<<(Wb->transpose())<<"\n";
  std::cout<<"WD:\n"<<(WD->transpose())<<"\n";
  std::cout<<"WD:\n"<<(WD->middleRows(nl,nr-nl+1))<<"\n";
  std::cout<<"B:\n"<<(*B)<<"\n";
  getchar();
  */

  //(B_data->p_ind)[0]=20;
  //(B_data->p_ind)[1]=-50;
  //B_data->show_ind();

  /*
  (*B)(8,0)=9;
  (*B)(9,0)=13;
  (*B)(8,1)=9;
  (*B)(8,2)=9;
  (*B)(8,3)=9;
  */
  //std::cout<<"B:\n"<<(*B)<<"\n";
  //std::cout<<"Btemp:\n"<<(*Btemp)<<"\n";

  B_data -> reset_ind(nct, nl, nr); //Indexes must be reset for each new solution to be found
  B_data -> nth_element(nct, nl, nr); //Sorting the correlation to choose the lm highest values for each column of B 
  //B_data->show_ind();

  /*
  std::cout<<"ind:\n"<<(B_data->ind.block(nl,0,lm,nct))<<"\n";
  getchar();
  */

  //Extracting Btemp values according to the highest values in each column of B
  for (int j = 0; j < nct; j++) {
    /*time=0*/
    for (int i = 0; i < lm; i++) {
      (BN -> data())[i + lm * j] = ( * Btemp)(B_data -> p_ind[n * j + i + nl], j);
    }
  }

  //std::cout<<"BN:\n"<<(*BN)<<"\n";
  //B_data->show_ind();

  //GENERATING THE D'*D SUBMATRICES TO BE SOLVED AND THE WD2I(SQUARED NORM CORRESPONDING TO THE ITERATION)
  #pragma omp parallel for schedule(dynamic)
  for (int k = 0; k < nct; k++) {
    for (int i = 0; i < lm; i++) {
      (DML[k]).col(i).noalias() = D -> col(B_data -> ind(i + nl, k));
      ( * WD2I)(i, k) = ( * WD2)(B_data -> ind(i + nl, k));
    }
    DN -> middleCols(k * lm, lm).noalias() = (DML[k]).transpose() * (DML[k]);

    for (int j = 0; j < lm; j++)
      ( * DN)(j, j + k * lm) = 0;

  }

  /*
  std::cout<<"D"<<(*D)<<"\n";
  for(int i=0;i<nc;i++)
  std::cout<<"DML["<<i<<"]:\n"<<DML[i]<<"\n";
  std::cout<<"DN:\n"<<(*DN)<<"\n";
  std::cout<<"WD2I:\n"<<(*WD2I)<<"\n";
  */

  //__________________Before the iteration________________
  x -> leftCols(nct).noalias() = Eigen::MatrixXf::Zero(lm, nct);
  expectedReadyVectors = 0;
  ith = 0;
  maxNumberIterations = 2 * numberZeros;
  while ((expectedReadyVectors < nct) && (ith < maxNumberIterations)) {

    u -> leftCols(nct) = (BN -> leftCols(nct).array() > fck).select((BN -> leftCols(nct).array() - fck), (BN -> leftCols(nct).array() < -fck).select((BN -> leftCols(nct).array() + fck), 0));

    u -> leftCols(nct).array() = (u -> leftCols(nct).array() / WD2I -> leftCols(nct).array());

    AE -> leftCols(nct).array() = ck * ((x -> leftCols(nct)) - (u -> leftCols(nct))).array() * ((WD2I -> leftCols(nct).array() * ((u -> leftCols(nct)) + (x -> leftCols(nct))).array()) - 2 * BN -> leftCols(nct).array()) + x -> leftCols(nct).array().abs() - u -> leftCols(nct).array().abs();

    for (int i = 0; i < nct; ++i)
      AE -> col(i).maxCoeff(jn -> data() + i);

    xn -> leftCols(nct).noalias() = x -> leftCols(nct);

    /*
    std::cout<<"jn_mod:\n"<<(*jn_mod)<<"\n";
    std::cout<<"jn:\n"<<(*jn)<<"\n";
    std::cout<<"arr:\n"<<(*arr)<<"\n";
    */

    jn_mod -> topRows(nct).noalias() = (jn -> topRows(nct)) + (arr -> topRows(nct));

    //std::cout<<"jn_mod:\n"<<(*jn_mod)<<"\n";

    /*
    std::cout<<"AE:\n"<<(*AE)<<"\n";
    std::cout<<"jn:\n"<<(jn->transpose())<<"\n";
    std::cout<<"arr:\n"<<(arr->transpose())<<"\n";
    std::cout<<"jn_mod:\n"<<(jn_mod->transpose())<<"\n";
    */

    for (int i = 0; i < nct; i++) {
      int jn_mod_i = ( * jn_mod)(i);
      float temp = ( * u)(jn_mod_i);
      ( * xn)(jn_mod_i) = temp;
      BK_1 -> col(i).noalias() = BN -> col(i) + (( * x)(jn_mod_i) - temp) * DN -> col(jn_mod_i);
    }

    BN -> leftCols(nct).noalias() = BK_1 -> leftCols(nct);
    x -> leftCols(nct).noalias() = xn -> leftCols(nct);

    noZeros -> head(nct) = (x -> leftCols(nct).array() != 0).cast < int > ().colwise().sum();
    expectedReadyVectors = ((noZeros -> head(nct).array() >= numberZeros).cast < int > ().sum());

    /*
    std::cout<<"x:\n"<<(*x)<<"\n";
    std::cout<<"noZeros:\n"<<(*noZeros)<<"\n";
    std::cout<<"expectedReadyVectors: "<<(expectedReadyVectors)<<"\n";
    getchar();
    */

    ith++;
  }

}

int DICTIONARY::estimateCluster(float & expectedPercentageDifference) {

 
  clt.clear(); // Clearing was established at the beginning to allow data extraction, modification date: October 19, 2015

  for (int j = 0; j < nct; j++) // Iterating over the columns where the sparse solution is found 
  {

    int iaux = 0, col = lm * j, colr = n * j;
    int * p = & (B_data -> ind(0, j));

    for (int i = nl; i < nl + lm; i++, iaux++) // Iterating over the rows where the sparse solution was previously found 
    {
      //MM(i-nl,j)=CL[p[i]];
      int aux_cl = CL[p[i]];
      clt.push_back(aux_cl);
      (arraycl[aux_cl]) -> push_back(iaux + col);

    }

  }

  std::sort(clt.begin(), clt.end());
  clt.erase(std::unique(clt.begin(), clt.end()), clt.end());

  //Eigen::MatrixXf x_aux;
  float * score_cl = new float[clt.size()];
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < clt.size(); i++) {
    Eigen::MatrixXf x_aux = Eigen::MatrixXf::Zero(lm, nct);
    for (int j = 0; j < (arraycl[clt[i]]) -> size(); j++) {
      (x_aux.data())[( * arraycl[clt[i]])[j]] = (x -> data())[( * arraycl[clt[i]])[j]];
    }

    float temp = 0;
    for (int ii = 0; ii < nct; ii++) {
      if (x_aux.col(ii).isZero(0))
        temp = temp + b -> col(ii).norm();
      else {
        Eigen::MatrixXf rs = Eigen::MatrixXf::Zero(m, 1);
        Eigen::MatrixXf x_temp = x_aux.col(ii);
        //std::cout<<"x_temp:\n"<<x_temp<<"\n";
        for (int j = 0; j < lm; j++) {
          if (x_temp(j) != 0) {
            rs = rs + (x_temp(j)) * ((DML[ii]).col(j));
          }

        }

        temp = temp + (b -> col(ii) - rs).norm();

      }
    }

    score_cl[i] = temp;

  }

  /*
  for(int i=0;i<clt.size();i++)
  std::cout<<"Class="<<clt[i]<<" value="<<score_cl[i]<<"\n";
  */

  float * min_score = std::min_element(score_cl, score_cl + clt.size());
  int indMinScore = std::distance(score_cl, min_score);
  int classChosen = clt[indMinScore];

  //____________HERE THE SCORE IS CALCULATED________________________

  /*

  Initially, the score_cl function is inverted with respect to the maximum point, meaning the maximum becomes zero and the minimum becomes the maximum, then the average is calculated without taking the maximum value of this function, and this value is compared with the isolated maximum value to calculate the average. The comparison is done by the percentage difference, giving a weight k to the subtracted value. The formula is shown below:

  expectedPercentageDifference=(max-k*(m))/max=1-k*(m/max)

  m is the mean and max is the isolated maximum value used to calculate the mean in the inverted function. 

  */

  //This vector will be useful for future analysis
  clustersDistance.clear();
  clustersDistance.reserve(clt.size());
  for (int i = 0; i < clt.size(); i++)
    clustersDistance.push_back(score_cl[i]);

  float * max_score = std::max_element(score_cl, score_cl + clt.size());
  float maxNscores = ( * max_score) - ( * min_score);
  float sScores = 0;
  for (int i = 0; i < clt.size(); i++)
    sScores = sScores + score_cl[i];

  float meanScores = ( * max_score) + ((( * min_score) - sScores) / (clt.size() - 1));

  expectedPercentageDifference = 1 - 2 * (meanScores / (maxNscores));

  //____________________________________________________________

  delete score_cl;
  //_____________CLEANING clt AND THE USED POSITIONS IN arraycl____________//
  for (int i = 0; i < clt.size(); i++) {
    for (int j = 0; j < (arraycl[clt[i]]) -> size(); j++) {
      (arraycl[clt[i]]) -> clear();
    }
  }

  //clt.clear();
  //___________________________________________________________________________//

  return classChosen; //Returning the estimated class that the descriptor group in b most likely belongs to

}

DICTIONARY::~DICTIONARY() {

  delete D;
  delete CL;

  //_______________________________
  for (int i = 0; i < n; i++) {
    delete arraycl[i];

  }

  delete[] arraycl;
  //________________________________

  delete[] DML;
  delete DN;
  delete WD;
  delete WD2;
  delete WD2I;
  delete Dsub2;
  delete b;
  delete B;
  delete B_data;
  delete BN;
  delete BK_1;
  delete x;
  delete xn;
  delete u;
  delete AE;
  delete jn;
  delete arr;
  delete jn_mod;
  delete noZeros;

  //Missing as of September 21, 2015
  delete Wb;
  delete Btemp;

  std::cout << "Dictionary deleted\n";
}

#endif


