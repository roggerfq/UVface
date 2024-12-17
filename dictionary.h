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

#ifndef DICTIONARY_H
#define DICTIONARY_H

#include <eigen/Eigen/Dense>
#include <eigen/Eigen/Sparse>
#include <omp.h>
//#define FLAG_PRECISION CV_64F /*Uncomment this line to use double precision (double 64 bits)*/
#define FLAG_PRECISION CV_32F /*Uncomment this line to use 32-bit floating point*/

//_______________________________HIDDEN OPENCV CLASSES__________________________________
namespace cv {
  class Mat;
};
//______________________________________________________________________________________

/*____________DEBUGGING::Delete at the end_______________*/

void tic();
void toc();
/*______________________________________________________*/

/*_________________Constants and global variables______________________________________*/
extern
const int G_DEPTH;
void tic();
void toc();
/*_______________________________________________________________________________________*/

#if FLAG_PRECISION == CV_64F

static int G_COLUMN_FOR_THREAD; /*Since no other alternative was found, this variable is defined as global, which controls the column to sort M for the respective thread of the DATA_ORDER class*/
#pragma omp threadprivate(G_COLUMN_FOR_THREAD) /*Indicates to the compiler that the variable G_COLUMN_FOR_THREAD is global in each thread*/
class DICTIONARY {
  //________________CLASS ATTRIBUTES__________________________
  int m; //Number of rows in the dictionary matrix
  int n; //Number of columns in the dictionary matrix
  int numberDescriptors; //Keeps track of the number of stored descriptors
  int nl; //Stores the value of the left column from which to start searching in the matrix
  int nr; //Stores the value of the right column up to where to search in the matrix
  int nc; //Maximum number of allowed descriptors
  int nct; //Number of descriptors to evaluate
  int lm; //Number of columns in the resulting matrix after applying Fast Filtering through correlation
  int numberZeros; //Expected number of non-zero values in the sparse solution

  double ck; //Regularization factor for the equation min(|x|1 + ck(b-Ax)^2)
  double fck; //Factor related to the SHRINK function, it should be equal to 1.0/(2*ck)

  Eigen::MatrixXd * D; //Dictionary matrix
  Eigen::MatrixXd * DML; //Pointer to submatrices of size D of size mxlm
  Eigen::MatrixXd * DN; //Will store the multiplication of each submatrix extracted from D by its transpose
  Eigen::VectorXd * WD; //Will store the norm of each column of D
  Eigen::VectorXd * WD2; //Will store the squared norm of each column of D
  Eigen::MatrixXd * WD2I; //Will store the squared norm of each selected column of D according to the correlation
  Eigen::MatrixXd * Dsub2; //Selected submatrices of D multiplied by their corresponding transposes (d*d')
  Eigen::MatrixXd * b; //This matrix will store descriptors less than or equal to nc in its columns
  Eigen::VectorXd * Wb; //This vector will store the norm of the nct descriptors of b in its rows
  Eigen::MatrixXd * Btemp; //Will store the multiplication D'*b
  Eigen::MatrixXd * B; //Will store the correlation between the columns of D and the columns of b
  Eigen::MatrixXd * BN; //Will store the lm rows of D'*b with the highest correlation in B
  Eigen::MatrixXd * BK_1; //Will store each subsequent BN in the iteration

  Eigen::MatrixXd * x; //Will store a representation of the previous sparse solution (useful in the iteration) 
  Eigen::MatrixXd * xn; //Will store a representation of the subsequent sparse solution (useful in the iteration) 
  Eigen::MatrixXd * u; //Will store the possible new values of each coordinate for the sparse vector x update
  Eigen::MatrixXd * AE; //Will store the variation in the energy function between iterations
  Eigen::VectorXi * jn; //Will store the indices with the highest scores in the energy matrix AE
  Eigen::VectorXi * arr; //Offset that will always be added to jn to construct jn_mod
  Eigen::VectorXi * jn_mod; //Will store the modified indices for use in the iteration
  Eigen::RowVectorXi * noZeros; //Will store the number of non-zero values in each column of x
  int expectedReadyVectors; //Will store the number of columns of x that have already met the numberZeros requirement
  int maxNumberIterations; //Stores the maximum number of iterations
  int ith; //Stores the iteration number in the iteration cycle
  //__________________________Useful structure for sorting_______________________________________

  struct DATA_ORDER {
    private: double ** matrix_values;
    int cols;
    int rows;
    int lm;
    int * ofset1;
    int * ofset2;

    Eigen::MatrixXi ind_k; //Will store the values 1,2,3,4,5...n in each of its columns

    //_________________________THE STRUCTURE BELOW IS USEFUL FOR SORTING_____________________
    struct sortFunctor {
      DATA_ORDER * DO;
      sortFunctor() {}
      void setFather(DATA_ORDER * DO) {
        this -> DO = DO;
      }
      bool operator()(const int & a,
        const int & b) {

        return (DO -> matrix_values[G_COLUMN_FOR_THREAD][a] > DO -> matrix_values[G_COLUMN_FOR_THREAD][b]);

      }

    };
    sortFunctor SF; //This functor is useful to avoid using a static method call in the std::nth_element function
    //______________________________________________________________________________________________________

    public:

      DATA_ORDER(Eigen::MatrixXd * M, int lme) {

        //___________HERE THE MATRIX TO BE SORTED (M) IS LINKED_____________
        lm = lme;
        cols = M -> cols();
        rows = M -> rows();
        matrix_values = new double * [cols];
        double * temp = M -> data();

        for (int i = 0; i < cols; i++) {
          matrix_values[i] = temp + rows * i;
        }
        //_______________________________________________________________

        SF.setFather(this); //Here this class is linked with sortFunctor

        //________Generating constant index matrix_________
        ind_k = Eigen::MatrixXi::Zero(rows, cols);
        Eigen::VectorXi vl = Eigen::VectorXi::LinSpaced(rows, 0, rows - 1);
        ind_k.colwise() += vl;
        //______________________________________________________
        ind = Eigen::MatrixXi::Zero(rows, cols); //Pre-allocate memory
        p_ind = ind.data();
        ofset1 = p_ind + lm;;
        ofset2 = p_ind + rows;
        reset_ind();

      }

      ~DATA_ORDER() {

        delete[] matrix_values;

      }

    //_______________________________HERE THE ORDERING OF THE lm HIGHEST VALUES IS DONE_________________________   

    void nth_element(int nct, int nl, int nr) {

      p_ind = ind.data() + nl;
      ofset1 = p_ind + lm;
      ofset2 = p_ind + (nr - nl) + 1;
      #pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < nct; i++) {
        G_COLUMN_FOR_THREAD = i;
        int ofset3 = i * rows;
        std::nth_element(p_ind + ofset3, ofset1 + ofset3, ofset2 + ofset3, SF);
      }

      //Resetting default values
      p_ind = ind.data();
      ofset1 = p_ind + lm;;
      ofset2 = p_ind + rows;

    }

    //__________________________________________________________________________________________________________________

    void reset_ind() {
      ind.noalias() = ind_k; //Reset the indices
    }

    void reset_ind(int nct, int nl, int nr) {
      ind.block(nl, 0, nr - nl + 1, nct).noalias() = ind_k.block(nl, 0, nr - nl + 1, nct); //Reset the indices
    }

    void show_ind() {
      std::cout << "ind:\n" << ind << "\n";
    }

    //PUBLIC ATTRIBUTES
    Eigen::MatrixXi ind;
    int * p_ind;

  };

  //____________________________________________________________________________________________________________________

  DATA_ORDER * B_data; // Alias for the matrix B, useful for sorting

  int * CL; /* This vector will be responsible for associating each column number in matrix D with its class number, classes must have an identifier between 1 and n. If no identifier is given, the class will default to zero, meaning no particular class has been assigned */

  std::vector < int > clt; // Useful for the estimateCluster() function, stores possible classes.
  std::vector < int > ** arraycl; // Stores the reference coordinates of nr
  std::vector < double > clustersDistance;

  // Private Functions                                                                                                 
  void normsWD(int c, int sz);
  void resizeDictionary(int sz);
  void dispersedSolution_lowLevel(int vnct, int vnl, int vnr);
  /* This function calculates the sparse solution for nct descriptors in the
  columns of matrix b, for a FAST FILTER of the matrix between columns nl and nr of D (nl and nr are counted from column zero) */

  void intermediateConstructor(); // Used to complete the construction of the class with constructor parameters

  public:
    DICTIONARY(int m, int n, int nc, int lm, double ck = 500); // Use this constructor to create a new descriptor database
  DICTIONARY(std::string nameFile); // Use this constructor to load a previously created database
  ~DICTIONARY();

  //_____________INTERFACE for Eigen____________________//
  void dispersedSolution(const Eigen::MatrixXd & Mb); /* Calculates the solution for the entire Mb matrix using D from 0 to nr */
  void dispersedSolution(const Eigen::MatrixXd & Mb, int ncol); /* Calculates the solution for the first ncol columns of the Mb matrix using D from 0 to nr */
  void dispersedSolution(const Eigen::MatrixXd & Mb, int ncol, int vnr, int vnl = 0); /* Calculates the solution for the first ncol columns of the Mb matrix using D from vnl to vnr */

  //____________INTERFACE FOR OPENCV____________________//

  void dispersedSolution(cv::Mat & Mb); /* Calculates the solution for the entire Mb matrix using D from 0 to nr */
  void dispersedSolution(cv::Mat & Mb, int ncol); /* Calculates the solution for the first ncol columns of the Mb matrix using D from 0 to nr */
  void dispersedSolution(cv::Mat & Mb, int ncol, int vnr, int vnl = 0); /* Calculates the solution for the first ncol columns of the Mb matrix using D from vnl to vnr */

  //___________LOW LEVEL INTERFACE TO AVOID COPYING VALUES INTO MATRIX b____________
  // It is assumed that values are in b */
  double * ptr(); // Pointer to b->data()
  void dispersedSolution(); // Calculates the sparse solution for all values in b, using D from 0 to nr
  void dispersedSolution(int ncol); /* Calculates the solution for the first ncol columns of matrix b using D from 0 to nr */
  void dispersedSolution(int ncol, int vnr, int vnl = 0); /* Calculates the solution for the first ncol columns of matrix b using D from vnl to vnr */

  //____________THE FOLLOWING FUNCTION ESTIMATES THE CLASS BELONGING TO THE CURRENT b SOLUTION IN MEMORY___________
  int estimateCluster(double & expectedPercentageDifference);

  //_________DATA INSERTION FUNCTIONS___________
  void eigenPush(const Eigen::MatrixXd & M, int cl = 0);
  void opencvPush(cv::Mat & M, int cl = 0);
  void pointerPush(double * pM, int cols, int cl = 0); // NOTE: Assume that the increasing order of addresses is by rows
  // Functions to remove data

  //_________PARAMETER CHANGE FUNCTIONS_________//
  void set_lm(int new_lm); // Sets the number of descriptors to pass to FAST FILTER
  void set_nc(int new_nc); // Sets the maximum number of descriptors that can be evaluated
  void set_numberZeros(int num); // Sets the expected maximum number of zeros in the sparse solution
  void set_ck(double new_ck); // Sets the regularization factor in the equation min |x|1+ck(Dx-b)^2 (|x|1 = l1 norm of x)
  //_________PARAMETER QUERY FUNCTIONS_______________//
  int get_lm();
  int get_nc();
  int get_nct();
  int get_numberZeros();
  double get_ck();
  int get_m();
  int get_n();
  int get_numberDescriptors();
  void get_completeSolution(Eigen::MatrixXd & XS, Eigen::MatrixXd & XA, int index); /* Obtains the complete solution vector and the approximation for the corresponding input response vector at the same index */
  Eigen::MatrixXd bDescriptor(int index) const;
  int * get_CL() const;
  std::vector < int > get_clt() const;
  std::vector < double > get_clustersDistance() const;

  //________DEBUGGING FUNCTIONS____________________
  void seeInfo();
  void clean();
  void sendToMatlab(); // Use this class to communicate with Matlab (NOTE: place the readData.m file in the build folder)
  void saveDataBase(std::string nameFile); // This function will store matrix D with the fundamental data in the nameFile directory

};

#else

static int G_COLUMN_FOR_THREAD; /* Due to the lack of other alternatives, this variable was defined globally, which controls the column to sort M for the respective thread in the DATA_ORDER class */
#pragma omp threadprivate(G_COLUMN_FOR_THREAD) /* Informs the compiler that the variable G_COLUMN_FOR_THREAD is global in each thread */
class DICTIONARY {
  //________________CLASS ATTRIBUTES__________________________
  int m; // Number of rows in the dictionary matrix
  int n; // Number of columns in the dictionary matrix
  int numberDescriptors; // Keeps track of the number of descriptors stored
  int nl; // Stores the value of the left column from where searching in the matrix will begin
  int nr; // Stores the value of the right column until where searching will be performed in the matrix
  int nc; // Maximum number of descriptors allowed
  int nct; // Number of descriptors to evaluate
  int lm; // Number of columns in the resulting matrix after applying Fast Filtering through correlation
  int numberZeros; // Expected number of non-zero values in the sparse solution

  float ck; // Regularization factor in the equation min(|x|1+ck(b-Ax)^2)
  float fck; // Factor related to the SHRINK function, must be equal to 1.0/(2*ck)

  Eigen::MatrixXf * D; // Dictionary matrix
  Eigen::MatrixXf * DML; // Pointer to submatrices of size D of size mxlm 
  Eigen::MatrixXf * DN; // Will store the multiplication of each submatrix extracted from D with its transpose
  Eigen::VectorXf * WD; // Will store the norm of each column of D
  Eigen::VectorXf * WD2; // Will store the square norm of each column of D
  Eigen::MatrixXf * WD2I; // Will store the square norm of each selected column of D according to correlation
  Eigen::MatrixXf * Dsub2; // Submatrices of selected D multiplied by their corresponding transposes (d*d')
  Eigen::MatrixXf * b; // This matrix will store in its columns a number of descriptors less than or equal to nc
  Eigen::VectorXf * Wb; // This vector will store in its rows the respective norm of the nct descriptors of b
  Eigen::MatrixXf * Btemp; // Will store the multiplication D'*b
  Eigen::MatrixXf * B; // Will store the correlation between the columns of D and the columns of b
  Eigen::MatrixXf * BN; // Will store the lm rows of D'*b that score highest in correlation B
  Eigen::MatrixXf * BK_1; // Will store each subsequent BN in the iteration

  Eigen::MatrixXf * x; // Will store a representation of the previous sparse solution (useful in iteration) 
  Eigen::MatrixXf * xn; // Will store a representation of the subsequent sparse solution (useful in iteration) 
  Eigen::MatrixXf * u; // Will store the possible new values of each coordinate for updating the sparse vector x
  Eigen::MatrixXf * AE; // Will store the variation in the energy function between iterations
  Eigen::VectorXi * jn; // Will store the indices that score highest in the energy matrix AE
  Eigen::VectorXi * arr; // Offset that will always be added to jn to construct jn_mod
  Eigen::VectorXi * jn_mod; // Will store the modified indices to use in the iteration
  Eigen::RowVectorXi * noZeros; // Will store the number of non-zero values in each column of x
  int expectedReadyVectors; // Will store the number of columns in x that have met the numberZeros requirement
  int maxNumberIterations; // Stores the maximum number of iterations
  int ith; // Stores the number of iterations in the iteration cycle
  //__________________________Structure useful for sorting_______________________________________

  struct DATA_ORDER {
    private: float ** matrix_values;
    int cols;
    int rows;
    int lm;
    int * ofset1;
    int * ofset2;

    Eigen::MatrixXi ind_k; // Will store the values 1,2,3,4,5...n in each of its columns

    //_________________________THE FOLLOWING STRUCTURE IS USEFUL FOR SORTING_____________________
    struct sortFunctor {
      DATA_ORDER * DO;
      sortFunctor() {}
      void setFather(DATA_ORDER * DO) {
        this -> DO = DO;
      }
      bool operator()(const int & a,
        const int & b) {

        return (DO -> matrix_values[G_COLUMN_FOR_THREAD][a] > DO -> matrix_values[G_COLUMN_FOR_THREAD][b]);

      }

    };
    sortFunctor SF; // This functor is useful to avoid using a static method call in the std::nth_element function
    //______________________________________________________________________________________________________

    public:

      DATA_ORDER(Eigen::MatrixXf * M, int lme) {

        //___________HERE THE MATRIX TO BE SORTED (M) IS LINKED_____________
        lm = lme;
        cols = M -> cols();
        rows = M -> rows();
        matrix_values = new float * [cols];
        float * temp = M -> data();

        for (int i = 0; i < cols; i++) {
          matrix_values[i] = temp + rows * i;
        }
        //_______________________________________________________________

        SF.setFather(this); // Here this class is linked with sortFunctor

        //________Generating constant index matrix_________
        ind_k = Eigen::MatrixXi::Zero(rows, cols);
        Eigen::VectorXi vl = Eigen::VectorXi::LinSpaced(rows, 0, rows - 1);
        ind_k.colwise() += vl;
        //______________________________________________________
        ind = Eigen::MatrixXi::Zero(rows, cols); // Pre-allocates memory
        p_ind = ind.data();
        ofset1 = p_ind + lm;;
        ofset2 = p_ind + rows;
        reset_ind();

      }

      ~DATA_ORDER() {

        delete[] matrix_values;

      }

    //_______________________________HERE THE ORDERING OF THE lm HIGHEST VALUES IS PERFORMED_________________________

    void nth_element(int nct, int nl, int nr) {

      p_ind = ind.data() + nl;
      ofset1 = p_ind + lm;
      ofset2 = p_ind + (nr - nl) + 1;
      #pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < nct; i++) {
        G_COLUMN_FOR_THREAD = i;
        int ofset3 = i * rows;
        std::nth_element(p_ind + ofset3, ofset1 + ofset3, ofset2 + ofset3, SF);
      }

      // Resetting default values
      p_ind = ind.data();
      ofset1 = p_ind + lm;
      ofset2 = p_ind + rows;

    }

    //__________________________________________________________________________________________________________________

    void reset_ind() {
      ind.noalias() = ind_k; // Resetting the indices
    }

    void reset_ind(int nct, int nl, int nr) {
      ind.block(nl, 0, nr - nl + 1, nct).noalias() = ind_k.block(nl, 0, nr - nl + 1, nct); // Resetting the indices
    }

    void show_ind() {
      std::cout << "ind:\n" << ind << "\n"; // Print the matrix "ind"
    }

    //PUBLIC ATTRIBUTES
    Eigen::MatrixXi ind;
    int * p_ind;

  };

  //____________________________________________________________________________________________________________________

  DATA_ORDER * B_data; // Alias for matrix B, useful for ordering

  int * CL; /* This vector will associate each column number of matrix D with its class number, classes must have an identifier between 1 and n, if no identifier is given, the class will default to zero, meaning no particular class has been assigned */

  std::vector < int > clt; // Useful for the function estimateCluster(), it will store the possible classes.
  std::vector < int > ** arraycl; // Will store the reference coordinates for nr
  std::vector < float > clustersDistance;

  //Private Functions
  void normsWD(int c, int sz);
  void resizeDictionary(int sz);
  void dispersedSolution_lowLevel(int vnct, int vnl, int vnr);
  /* This function calculates the sparse solution of nct descriptors in the
  columns of matrix b, for a FAST FILTER of the matrix between the columns nl and nr of D (nl and nr are 0-indexed) */

  void intermediateConstructor(); // Used to complete the construction of the class with the parameters from the constructor

  public:
    DICTIONARY(int m, int n, int nc, int lm, float ck = 500); // Use this constructor to create a new descriptor database
  DICTIONARY(std::string nameFile); // Use this constructor to load a previously created database
  ~DICTIONARY();

  //_____________INTERFACE for eigen____________________//
  void dispersedSolution(const Eigen::MatrixXf & Mb); /* Calculates the solution for the entire matrix Mb using D from 0 to nr */
  void dispersedSolution(const Eigen::MatrixXf & Mb, int ncol); /* Calculates the solution for the first ncol columns of the matrix Mb using D from 0 to nr */
  void dispersedSolution(const Eigen::MatrixXf & Mb, int ncol, int vnr, int vnl = 0); /* Calculates the solution for the first ncol columns of the matrix Mb using D from vnl to vnr */

  //____________INTERFACE FOR OPENCV____________________//

  void dispersedSolution(cv::Mat & Mb); /* Calculates the solution for the entire matrix Mb using D from 0 to nr */
  void dispersedSolution(cv::Mat & Mb, int ncol); /* Calculates the solution for the first ncol columns of the matrix Mb using D from 0 to nr */
  void dispersedSolution(cv::Mat & Mb, int ncol, int vnr, int vnl = 0); /* Calculates the solution for the first ncol columns of the matrix Mb using D from vnl to vnr */

  //___________LOW-LEVEL INTERFACE TO AVOID COPYING VALUES TO MATRIX b____________
  // It is assumed that values are already in b*/
  float * ptr(); // Pointer to b->data()
  void dispersedSolution(); // Calculates the sparse solution for all values in b, using D from 0 to nr
  void dispersedSolution(int ncol); /* Calculates the solution for the first ncol columns of matrix b using D from 0 to nr */
  void dispersedSolution(int ncol, int vnr, int vnl = 0); /* Calculates the solution for the first ncol columns of matrix b using D from vnl to vnr */

  //____________THE FOLLOWING FUNCTION ESTIMATES THE CLASS BELONGING TO THE CURRENT SOLUTION OF b IN MEMORY___________
  int estimateCluster(float & expectedPercentageDifference);

  //_________DATA INSERTION FUNCTIONS___________
  void eigenPush(const Eigen::MatrixXf & M, int cl = 0);
  void opencvPush(cv::Mat & M, int cl = 0);
  void pointerPush(float * pM, int cols, int cl = 0); // NOTE: Assume that the increasing order of addresses is row-major

  //_________PARAMETER CHANGING FUNCTIONS_________//
  void set_lm(int new_lm); // Sets the number of descriptors to be passed to the FAST FILTER
  void set_nc(int new_nc); // Sets the maximum number of descriptors to evaluate
  void set_numberZeros(int num); // Sets the expected number of zeros in the sparse solution
  void set_ck(float new_ck); // Sets the regularization factor in the equation min |x|1+ck(Dx-b)^2 (|x|1 = l1 norm of x)
  //_________PARAMETER QUERY FUNCTIONS_______________//
  int get_lm();
  int get_nc();
  int get_nct();
  int get_numberZeros();
  float get_ck();
  int get_m();
  int get_n();
  int get_numberDescriptors();
  void get_completeSolution(Eigen::MatrixXf & XS, Eigen::MatrixXf & XA, int index); /* Gets the complete solution vector and the approximation for the input response vector corresponding to the same index */
  Eigen::MatrixXf bDescriptor(int index) const;
  int * get_CL() const;
  std::vector < int > get_clt() const;
  std::vector < float > get_clustersDistance() const;

  //________DEBUG FUNCTIONS____________________
  void seeInfo();
  void clean();
  void sendToMatlab(); // Use this method to communicate with Matlab (NOTE: Place the file readData.m in the build folder)
  void saveDataBase(std::string nameFile); // This function will store the matrix D with the fundamental data at the location nameFile

};

#endif

#endif