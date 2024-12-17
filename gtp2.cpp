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
//read file
#include <fstream>
#include <string>
#include <sstream>
/*___________________*/
/*________openCV_________*/
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
/*_______________________*/
/*_______openMP__________*/
#include <omp.h>
/*_______________________*/
#include "gtp2.h"
/*_______QT_______________*/
#include <QCheckBox>
#include <QFormLayout>
/*________________________*/
//________Added to use the std::exit(int exit_code) function as exception handling___________//
#include <cstdlib>
//For reading directories
#include <sys/types.h>
#include <dirent.h>
#include <sys/mount.h> // mount

/*
bool mountRamdisk() {

  int VAL = 0;

  DIR * dir = opendir("/ramdisk_UVface");
  if (dir) {

    closedir(dir);

    VAL = system("cp ./ramdisk_inicio/extract_features_64bit.ln /ramdisk_UVface/extract_features_64bit.ln");
    VAL = system("cp ./ramdisk_inicio/imagen.pgm.sedgelap /ramdisk_UVface/imagen.pgm.sedgelap");
    //VAL=system("cp ./ramdisk_inicio/imagen.pgm /ramdisk_UVface/imagen.pgm");

    std::cout << "The necessary startup files for the proper functioning of the GTP descriptor were successfully mounted in the /ramdisk_UVface folder\n";
    return true;
  } else {
    std::cout << "The ramdisk_UVface folder, required for mounting some startup files necessary for the proper execution of the GTP descriptor, does not exist or does not have the necessary write permissions. Therefore, the program cannot run.\n";

    return false;
  }

}
*/

//Problem: if there is no imagen.pgm.sedgelap file 
void createSedgelapFile() {

  const std::string filepath = "/ramdisk_UVface/imagen.pgm.sedgelap";

  // Try to open the file in read mode
  std::ifstream fp(filepath.c_str(), std::ifstream::in); // Use .c_str()

  // If the file cannot be opened (doesn't exist), create it in write mode
  if (!fp.is_open()) {
    std::cout << "The file does not exist. Creating the file...\n";
    std::ofstream createFile(filepath.c_str()); // Use .c_str()
    if (createFile.is_open()) {
      std::cout << "File created successfully: " << filepath << "\n";
    } else {
      std::cerr << "Error creating the file.\n";
      return;
    }

    // Now try to reopen the file in read mode
    fp.open(filepath.c_str(), std::ifstream::in); // Use .c_str()
    if (!fp.is_open()) {
      std::cerr << "Error opening the file after creating it.\n";
      return;
    }
  }

  // Close the file
  fp.close();

}

/*__________________this funtion are used to mount the ramdisk________________*/
void ensureDirectoryExists(const std::string& path) {
    std::string command = "mkdir -p " + path;
    if (system(command.c_str()) == 0) {
        std::cout << "Directory created: " << path << std::endl;
    } else {
        std::cerr << "Error al crear el directorio: " << path << std::endl;
        exit(EXIT_FAILURE);
    }
}

void mountTmpfs(const std::string& path) {
    if (mount("tmpfs", path.c_str(), "tmpfs", 0, "size=512M,mode=777") == 0) {
        std::cout << "tmpfs mounted on: " << path << std::endl;
    } else {
        perror("Error mounting tmpfs");
        exit(EXIT_FAILURE);
    }
}

void copyFile(const std::string& source, const std::string& destination) {
    std::string command = "cp " + source + " " + destination;
    if (system(command.c_str()) == 0) {
        std::cout << "File copied from " << source << " to " << destination << std::endl;
    } else {
        std::cerr << "Error copying the file: " << source << std::endl;
        exit(EXIT_FAILURE);
    }
}

/*_____________________________________________________________________________*/


GTP::GTP(): maxFeatures(8000), szRect(40), szHistlow(128) {

  //__________Simple exception handling for the case where ramdisk_UVface was not mounted properly_______ 
  //if (!mountRamdisk()) std::exit(EXIT_FAILURE);
  //_______________________________________________________________________________________________________________

  //rows=szHistlow;//THIS VARIABLE BELONGS TO THE ABSTRACT BASE CLASS

  constructGaborKernels(); //Here we construct the Gabor filters to be used

  /*Here we open the file where the ellipse parameters obtained from the extract_features_64bit.ln software will be located 
  It is assumed that an emulated folder in RAM named ramdisk_UVface has been created*/

  //fp.open("/ramdisk_UVface/imagen.pgm.sedgelap",std::ifstream::in);
  std::string ramdiskPath = "/ramdisk_UVface";
  std::string sourceFile = "../extract_features_64bit.ln";
  std::string destinationFile = ramdiskPath + "/extract_features_64bit.ln";
  //___mounting ramdisk_______//
  // Paso 1: Asegurar que el directorio existe
  ensureDirectoryExists(ramdiskPath);
  // Paso 2: Montar tmpfs
  mountTmpfs(ramdiskPath);
  // Paso 3: Copiar el archivo
  copyFile(sourceFile, destinationFile);
  //__________________________//
  createSedgelapFile();
  
  fp.open("/ramdisk_UVface/imagen.pgm.sedgelap", std::ifstream::in);

  vec_dp = new double * [maxFeatures];
  /*The number 1000 is due to the fact that images smaller than 200x200, which are expected to be used, will probably not 
  generate more than 1000 features*/
  for (int i = 0; i < maxFeatures; ++i)
    vec_dp[i] = new double[5]; //5 are the columns of the file "/ramdisk_UVface/imagen.pgm.sedgelap"

  /*rc represents the radius that the resulting circle should have after applying the affine transformation to each ellipse 
  so that it covers a square with sides of szRectxszRect*/
  ARect = szRect * szRect;
  rc = sqrt(2 * (ARect)) / 2;
  centerRect = szRect / 2;

  imagesRect = new cv::Mat[maxFeatures]; //1000 is chosen because it is expected to be the maximum number of useful features
  imgZscore = new cv::Mat[maxFeatures];
  imgGtp = new cv::Mat[maxFeatures];
  for (int i = 0; i < maxFeatures; i++) {
    imagesRect[i].create(szRect, szRect, CV_8UC1); //Initializing the memory
    imgZscore[i].create(szRect, szRect, CV_32F);
    imgGtp[i].create(szRect, szRect, CV_8UC1);
  }

  minimumSizeAxisEllipses = 10; //10 is chosen as the default minimum size for each ellipse

  dHist = 4 * 4 * 81;
  imgHist = new cv::Mat;
  imgHist -> create(maxFeatures, dHist, CV_32F);
  imgHIstTemp = new cv::Mat;
  //imgHistlow=new cv::Mat;
  //imgHistlow->create(maxFeatures,szHistlow,CV_32F);
  pca = new cv::PCA; //The construction is done by default until the call in the reimplemented virtual function post_processing
  descriptorTest = new Eigen::MatrixXf;

  setGui(); //Setting up the graphical interface

}

GTP::~GTP() {

  fp.close();

  //________Deleting dynamic memory_______

  for (int i = 0; i < maxFeatures; ++i)
    delete[] vec_dp[i];

  delete[] vec_dp;
  delete[] imagesRect;
  delete[] imgZscore;
  delete[] imgGtp;
  delete[] gaborKernel;
  delete[] pow3;
  delete imgHist;
  delete imgHIstTemp;
  //delete imgHistlow;
  delete pca;
  delete descriptorTest;
  //_________________________________________

}

void GTP::setGui() {

  std::cout << "Setting up the graphical interface\n";

  checkBox_PCA = new QCheckBox;

  QFormLayout * formLayout = new QFormLayout;
  formLayout -> addRow(tr("Calculate PCA:"), checkBox_PCA);

  setLayout(formLayout);

}

void GTP::zScoreNormalization(const cv::Mat *
  const imageSrc, cv::Mat * imageDest) {

  cv::Mat u, s;
  cv::Mat srcTemp;
  imageSrc -> assignTo(srcTemp, CV_32F);

  cv::meanStdDev(srcTemp, u, s);
  u.assignTo(u, CV_32F);
  s.assignTo(s, CV_32F);

  ( * imageDest) = (srcTemp - (u.at < float > (0, 0)) + 3 * (s.at < float > (0, 0))) / (6 * s.at < float > (0, 0));

  float * p_imageDest = imageDest -> ptr < float > (0);

  for (int j = 0; j < ARect; j++) {

    if (p_imageDest[j] < 0) p_imageDest[j] = 0;
    else if (p_imageDest[j] > 1) p_imageDest[j] = 1;

  }

  /*
  std::cout<<"img="<<(*imageSrc)<<"\n";
  std::cout<<"dest="<<(*imageDest)<<"\n";


  std::cout<<"Mean="<<u<<"\n";
  std::cout<<"StdDev="<<s<<"\n";

  cv::imshow("V",(*imageSrc));
  cv::waitKey(0);
  */

}

void GTP::constructGaborKernels() {

  /*NOTE: We worked with 32-bit precision because the openCV filter2D function requires a CV_32F type kernel.
  It is also clarified that this method was not optimized because it is only called once in the constructor and the calculations required are insignificant*/

  numberOrientations = 4; //Number of orientations
  double orientations[] = {
    0,
    45,
    90,
    135
  }; //Orientation vector in degrees for the filter
  double sigma = 1;
  double Kv = M_PI / 2;
  szGk = 6 * sigma + 1; //Added 1 to make the kernel odd

  /*Creating the evaluation domain*/
  /*This part of the code mimics the MATLAB meshgrid function*/
  cv::Mat X(szGk, szGk, cv::DataType < int > ::type);
  cv::Mat Y(szGk, szGk, cv::DataType < int > ::type);

  int lim = -szGk / 2;
  for (int i = 0; i < szGk; i++, lim++)
    X.col(i) = lim;
  Y = X.clone().t();
  /*________________________________*/

  cv::Mat Z = (X.mul(X) + Y.mul(Y));
  Z.convertTo(Z, CV_32F);

  //__________CALCULATING THE GAUSSIAN__________//
  cv::Mat GAUSIAN;
  cv::exp(-((Kv * Kv) / (2 * sigma * sigma)) * Z, GAUSIAN);
  GAUSIAN = ((Kv * Kv) / (sigma * sigma)) * GAUSIAN;

  //std::cout<<"Gaussian type="<<GAUSIAN.type()<<"\n";

  //_______Here we remove values caused by numerical noise_____________________
  float max_GAUSIAN = * std::max_element(GAUSIAN.begin < float > (), GAUSIAN.end < float > ());
  float EPS = std::numeric_limits < float > ::epsilon();
  float aux2 = EPS * max_GAUSIAN;

  for (int i = 0; i < GAUSIAN.rows; i++) {
    for (int j = 0; j < GAUSIAN.cols; j++) {
      if (GAUSIAN.at < float > (i, j) < aux2) {
        GAUSIAN.at < float > (i, j) = 0;
      }
    }
  }
  //_________________________________________________________________________________

  //___________Here we normalize the Gaussian so that its sum equals 1__________________
  float sum_GAUSIAN = cv::sum(GAUSIAN)[0];
  if (sum_GAUSIAN != 0)
    GAUSIAN = GAUSIAN / sum_GAUSIAN;
  //_________________________________________________________________________________

  //__________________________________________//

  gaborKernel = new cv::Mat[numberOrientations]; //Array for the kernel for each different orientation
  //Convert to float because subsequent operations are with floating-point matrices
  X.convertTo(X, CV_32F);
  Y.convertTo(Y, CV_32F);

  for (int n = 0; n < numberOrientations; n++) {

    double alpha = (orientations[n]) * M_PI / 180;
    cv::Mat Xp = Kv * cos(alpha) * X.clone();
    cv::Mat Yp = Kv * sin(alpha) * Y.clone();

    //_________FIRST CONSTRUCTING THE SIN 2D FUNCTION_________//
    cv::Mat sin2D(szGk, szGk, CV_32F);
    for (int i = 0; i < szGk; i++) {
      for (int j = 0; j < szGk; j++) {
        sin2D.at < float > (i, j) = sin(Xp.at < float > (i, j) + Yp.at < float > (i, j));
      }
    }

    //________________________________________________________//

    //Now store each filter
    gaborKernel[n] = GAUSIAN.mul(sin2D);

    /*
    std::cout<<"Angle="<<orientations[n]<<" in radians="<<alpha<<"\n";
    std::cout<<"G="<<gaborKernel[n]<<"\n";
    getchar();
    */

  }

  //Initialize the pow3 array to be used in the gtp(cv::Mat *img) function
  pow3 = new int[numberOrientations];
  for (int i = 0; i < numberOrientations; i++)
    pow3[i] = std::pow(3, i);

  //Initialize the threshold used in the ternary pattern (gtp(cv::Mat *img))
  ut = 0.007; //0.03 is the standard, apparently 0.01 worked better
  //ut=0.005;//Previous value
}

void GTP::gtp(cv::Mat * img, cv::Mat * ltp) {

  cv::Mat temp;
  ( * ltp) = cv::Mat::zeros(szRect, szRect, CV_8UC1);

  for (int i = 0; i < numberOrientations; i++) {
    /*NOTE: Matlab and openCV may show slight differences at the edges of the resulting image temp.
    Also, remember that the result of filtering is a correlation, so it gives the negative of the convolution (the dilemma arises: should I choose the positive, i.e., the negative of the filter2D result, which would be equivalent to convolution, or the positive, i.e., just filter2D?), since what is going to be built is a ternary encoding, the sign doesn't matter as the encoding simply gets rearranged*/

    cv::filter2D( * img, temp, CV_32F, gaborKernel[i]);
    //ltp=ltp+(pow3[i]*((temp<-ut)+2*(temp>ut))/255);
    ( * ltp) = ( * ltp) + (pow3[i] * ((temp < -ut) + 2 * (temp > ut)) / 255);

    /*
    std::cout<<"\n\nimg="<<(*img)<<"\n\n";
    std::cout<<"temp="<<temp<<"\n\n";
    std::cout<<"comp="<<(pow3[i]*((temp<-ut)+2*(temp>ut))/255)<<"\n\n";
    std::cout<<"ltp="<<(*ltp)<<"\n\n";
    std::cout<<"kernel="<<gaborKernel[i]<<"\n\n";
    std::cout<<"kernel number="<<i<<"\n";
    getchar();
    */

  }

}

//___________________________________________________________________________________________________
namespace Gh { //This namespace is created to simplify the use of the calcHist function

  //Number of bins
  int histSize[] = {
    81
  };

  //Range of histogram values
  float range[] = {
    0,
    81
  }; //Remember, the upper limit is exclusive
  const float * histRange[] = {
    range
  };

  //Settings (see openCV documentation)
  bool uniform = true;
  bool accumulate = false;

  //Channels to analyze
  int channels[] = {
    0
  };
}

void GTP::histogram(cv::Mat * img, int row) {

  /*This part is specifically for images img of size 40x40*/
  cv::Mat temp, hist;

  /*
  std::cout<<"img="<<(*img)<<"\n";
  getchar();
  */

  int nh = 0;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {

      temp = (( * img)(cv::Range(10 * i, 10 * i + 10), cv::Range(10 * j, 10 * j + 10))).clone();
      //Now we extract the histogram of the temp image
      calcHist( & temp, 1, 0, cv::Mat(), hist, 1, Gh::histSize, Gh::histRange, Gh::uniform, Gh::accumulate);
      //Now concatenate the histogram with the previously stored ones
      imgHist -> row(row).colRange(nh, nh + 81) = hist.t();
      nh = nh + 81;

      /*
      std::cout<<"HIST="<<imgHist->row(row)<<"\n";
      std::cout<<"temp="<<temp<<"\n";
      std::cout<<"hist="<<hist.t()<<"\n";
      getchar();
      */

    }
  }

  //imgHist->row(row)=imgHist->row(row)/100;//Here the histogram is normalized (remember each of the 16 cells is 10x10=100)

  float normRow = cv::norm(imgHist -> row(row));
  /*
  std::cout<<"H="<<imgHist->row(row)<<"\n";
  std::cout<<"norm="<<normRow<<"\n";
  getchar();
  */

  if (normRow != 0)
    imgHist -> row(row) = (imgHist -> row(row)) / normRow;

  float * prow = imgHist -> row(row).ptr < float > (0);
  for (int i = 0; i < 1296; i++)
    prow[i] = tanh(20 * prow[i]);

}

//_______________________________________________________________________________________________________

//__________________read number key points______________________

// Convert the std::string to const char* for compatibility
int readSecondLineIfPositive(const std::string & filePath) {
  std::ifstream file(filePath.c_str());
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file at " << filePath << std::endl;  // Error message when the file can't be opened
    return 0;
  }

  std::string line;
  int lineCount = 0;

  while (std::getline(file, line)) {
    ++lineCount;
    if (lineCount == 2) { // Check if it's the second line
      std::istringstream iss(line);
      int number;
      if (iss >> number && number > 0) { // Check if the line is an integer greater than zero
        return number;
      } else {
        return 0; // If not a valid positive integer, return 0
      }
    }
  }

  // If the file has less than 2 lines, return 0
  return 0;
}

cv::Mat * GTP::descriptor_base(const cv::Mat & myImage) {

  cv::Mat img;

  if (myImage.channels() > 1)
    cv::cvtColor(myImage, img, CV_BGR2GRAY);
  else
    img = myImage;

  cv::imwrite("/ramdisk_UVface/imagen.pgm", img); // Here the image is written to be analyzed by extract_features_64bit.ln

  // Here we call extract_features_64bit.ln 

  /*__________________Here we extract the number of generated features____________________________*/
  /*NOTE: The code is somewhat complicated because it needs to search for the position in the output stream of extract_features_64bit.ln
  the number of features extracted from the file /ramdisk_UVface/imagen.pgm*/

  // Command to execute
  const char * command = "/ramdisk_UVface/extract_features_64bit.ln -sedgelap -noangle -i /ramdisk_UVface/imagen.pgm";

  // Open the pipe
  FILE * pipe = popen(command, "r");

  // Read the command output
  char bufferInfoPipe[128];
  while (fgets(bufferInfoPipe, sizeof(bufferInfoPipe), pipe) != 0) {
    std::cout << bufferInfoPipe; // Print each line of the output
  }

  pclose(pipe);

  int numberKeyPoints = readSecondLineIfPositive("/ramdisk_UVface/imagen.pgm.sedgelap");
  std::cout << "numberKeyPoints: " << numberKeyPoints << std::endl;
  
    /*Only if the number of features extracted by extract_features_64bit.ln is greater than zero is the file "/ramdisk_UVface/imagen.pgm.sedgelap" read */
  if (numberKeyPoints > 0) {

    fp.seekg(std::ios_base::beg);
    double dp;
    int num;
    fp >> dp; /*This is done only for protocol, to remove the first number*/
    fp >> num; /*The number of features is found in the second line of the file*/

    if (numberKeyPoints > maxFeatures) { //Protection for when the number of features exceeds the maximum supported in memory
      std::cout << "The number of features exceeded the maximum set of " << maxFeatures << ", returning null by default" << "\n";
      return NULL;
    }

    /*________Here the ellipse parameters are stored in the vec_dp array______*/
    for (int i = 0; i < numberKeyPoints; i++) {
      for (int j = 0; j < 5; j++) {
        fp >> (vec_dp[i][j]);
      }
    }
    /*_________________________________________________________________________________*/

    numberUsefulFeatures = 0; //Reset at each cycle
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numberKeyPoints; i++) {

      cv::Mat MQ(2, 2, CV_64F);
      MQ.at < double > (0, 0) = vec_dp[i][2];
      MQ.at < double > (0, 1) = vec_dp[i][3];
      MQ.at < double > (1, 0) = vec_dp[i][3];
      MQ.at < double > (1, 1) = vec_dp[i][4];

      cv::Mat EIG_val(2, 1, CV_64F);
      cv::Mat EIG_vec(2, 2, CV_64F);

      //std::cout<<MQ<<"\n";
      eigen(MQ, EIG_val, EIG_vec); //Here the eigenvectors and eigenvalues of MQ are calculated
      double l1 = 1.0 / sqrt(EIG_val.at < double > (1, 0)); //Major axis, as it is divided by the smaller eigenvector
      double l2 = 1.0 / sqrt(EIG_val.at < double > (0, 0)); //Minor axis, as it is divided by the larger eigenvector

      double alpha = atan2(EIG_vec.at < double > (1, 1), EIG_vec.at < double > (1, 0)); //Major axis is taken as reference

      //std::cout<<"l1="<<l1<<" l2="<<l2<<" alpha="<<alpha<<"\n";

      double c = vec_dp[i][0];
      double f = vec_dp[i][1];


      //_____coordinates enclosing the ellipse____________________//
      double sc = vec_dp[i][3] / sqrt(vec_dp[i][2] * vec_dp[i][4] - vec_dp[i][3] * vec_dp[i][3]);

      double xsi = sc * sqrt(1 / vec_dp[i][2]);
      double yyi = -(vec_dp[i][2] / vec_dp[i][3]) * xsi;
      xsi = xsi + c;
      yyi = yyi + f;

      double xsf = -sc * sqrt(1 / vec_dp[i][2]);
      double yyf = -(vec_dp[i][2] / vec_dp[i][3]) * xsf;
      xsf = xsf + c;
      yyf = yyf + f;

      double ysi = sc * sqrt(1 / vec_dp[i][4]);
      double xxi = -(vec_dp[i][4] / vec_dp[i][3]) * ysi;
      xxi = xxi + c;
      ysi = ysi + f;

      double ysf = -sc * sqrt(1 / vec_dp[i][4]);
      double xxf = -(vec_dp[i][4] / vec_dp[i][3]) * ysf;
      xxf = xxf + c;
      ysf = ysf + f;

      cv::Rect brect = cv::Rect(cv::Point(xxi, yyi), cv::Point(xxf, yyf));
      //______________________________________________________________//

      //if((brect.x>=0)&&(brect.y>=0)&&(brect.x+brect.width<img.cols)&&(brect.y+brect.height<img.rows)&&(l1>minimumSizeAxisEllipses)&&(l2>minimumSizeAxisEllipses)){

      //if((brect.x>=0)&&(brect.y>=0)&&(brect.x+brect.width<img.cols)&&(brect.y+brect.height<img.rows)&&(brect.width<0.4*double(img.cols))&&(brect.height<0.4*double(img.rows))){ //best approximation 0.4 is better

      //if((brect.x>=0)&&(brect.y>=0)&&(brect.x+brect.width<img.cols)&&(brect.y+brect.height<img.rows)){// best 
      if ((xxi >= 0) && (yyi >= 0) && (xxf < img.cols) && (yyf < img.rows)) {

        cv::Mat * imgRect;
        cv::Mat * imgRectZscore;
        cv::Mat * imgRectGtp;
        int row;
        #pragma omp critical
		{
          numberUsefulFeatures++; //Counting useful features
          imgRect = & (imagesRect[numberUsefulFeatures - 1]);
          imgRectZscore = & (imgZscore[numberUsefulFeatures - 1]);
          imgRectGtp = & (imgGtp[numberUsefulFeatures - 1]);
          row = numberUsefulFeatures - 1;
        }

        cv::Mat Rimg = img(brect);


        cv::Mat e(2, 2, CV_64F);
        e.at < double > (1, 0) = 0;
        e.at < double > (0, 1) = 0;

        cv::Mat pc(2, 1, CV_64F);
        cv::Mat pcn(2, 1, CV_64F);
        cv::Mat iM1(2, 2, CV_64F);
        cv::Mat iM2(2, 3, CV_64F);

        /*Next, we calculate the affine transformation that converts an ellipse into a circle of radius rc*/
        e.at < double > (0, 0) = std::sqrt(EIG_val.at < double > (0, 0));
        e.at < double > (1, 1) = std::sqrt(EIG_val.at < double > (1, 0));
        iM1 = rc * EIG_vec.t() * e * EIG_vec; //Inverse affine transformation, such that the ellipse becomes a circle (MQ=iM1*iM1)

        //Center of the image before transformation
        pc.at < double > (0, 0) = Rimg.cols / 2;
        pc.at < double > (1, 0) = Rimg.rows / 2;

        /*We place the first row of iM1 in the corresponding positions in iM2*/
        iM2.at < double > (0, 0) = iM1.at < double > (0, 0);
        iM2.at < double > (0, 1) = iM1.at < double > (0, 1);

        pcn = centerRect - (iM1 * pc); /*Here we calculate the necessary translation to center the image, (*iM1)*(*pc) represents the new center of the image, and centerRect is the center of the szRectxszRect image, such that centerRect-(*iM1)*(*pc) is the vector that will translate the image transformed by the affine transformation iM1 so that it is centered with respect to the szRectxszRect rectangle*/

        iM2.at < double > (0, 2) = pcn.at < double > (0, 0);

        iM2.at < double > (1, 0) = iM1.at < double > (1, 0);
        iM2.at < double > (1, 1) = iM1.at < double > (1, 1);
        iM2.at < double > (1, 2) = pcn.at < double > (1, 0);

        cv::warpAffine(Rimg, *imgRect, iM2, cv::Size(szRect, szRect));

      

        //_____________________________________________________________________________________

        /*For speed, this part of the code assumes that the images in imagesRect are of type CV_8UC1 (i.e., img must be of type CV_8UC1 since warpAffine returns the same type as the source image to imagesRect), in general, everything works regardless of the type, but the following part will need this to save calculations in type conversion*/

        //Next, the illumination of each image in the imagesRect vector is normalized to cv::Mat imagesRect with the zScoreNormalization method
        zScoreNormalization(imgRect, imgRectZscore);
        /*Proceed to convolve imgRectZscore with the Gabor kernels for the corresponding angles (imaginary part) and combine the result in a ternary pattern*/
        gtp(imgRectZscore, imgRectGtp); //Remember the return type is CV_8UC1

        histogram(imgRectGtp, row); /*Proceed to calculate the histogram for each image and store it in the row of imgHist*/

        //_____________________________________________________________________________________

      }

    }

    if (numberUsefulFeatures > 0) {
      * imgHIstTemp = (( * imgHist)(cv::Range(0, numberUsefulFeatures), cv::Range::all())).clone(); //Select the rows of the matrix that contain the data from the iteration
      std::cout << "Number of useful features=" << numberUsefulFeatures << "  image address=" << & img << "\n";
      return imgHIstTemp;
    }

  }

  std::cout << "Number of useful features=" << 0 << "  image address=" << & img << "\n";
  return NULL;

}

void GTP::post_processing(cv::Mat & descriptor_base, cv::Mat & descriptor_end, std::vector < int > & ithRows) {

  if (Qt::Checked == checkBox_PCA -> checkState()) {

    (* pca)(descriptor_base, cv::Mat(), CV_PCA_DATA_AS_ROW, szHistlow); //Remember that the samples are stored by rows
    //Next, we project descriptor_base onto the first szHistlow principal components
    pca -> project(descriptor_base, descriptor_end);

    savePca(); //We store the PCA information

  } else {
    std::cout << "PCA was not calculated\n";
	std::cout << "Loading matrix PCA\n";
	LoadPca();
    pca -> project(descriptor_base, descriptor_end);
  }

}

Eigen::MatrixXf * GTP::test(const cv::Mat & img) {

  cv::Mat * p_temp = descriptor_base(img); //First, we calculate the initial base descriptor of the image
  if (p_temp != NULL) {
    cv::Mat mtemp;
    pca -> project((* p_temp), mtemp);

    float * pf = mtemp.ptr < float > (0);
    Eigen::Map < Eigen::MatrixXf > mf(pf, mtemp.cols, mtemp.rows);
    (* descriptorTest) = mf;

    //std::cout << "number of rows=" << descriptorTest->rows() << " number of columns=" << descriptorTest->cols() << "\n";

    return descriptorTest;
  }

  return NULL;
}

QString GTP::nameDescriptor() const {
  QString name("GTP");
  return name;
}

void GTP::loadSettings(QString path) {

  pathDataBase = path; //We update the path that will be used to store the PCA
  LoadPca(); //Here, we load the PCA from the previous directory
  std::cout << "Loading information\n";

}

bool GTP::applySettings(QString path) {

  pathDataBase = path; //We update the path that will be used to store the PCA

  return true;
}

void GTP::savePca() {

  cv::FileStorage fs((pathDataBase + QString("/pca.xml")).toStdString(), cv::FileStorage::WRITE);
  fs << "mean" << pca -> mean;
  fs << "e_vectors" << pca -> eigenvectors;
  fs << "e_values" << pca -> eigenvalues;
  fs.release();

}

void GTP::LoadPca() {

  cv::FileStorage fs((pathDataBase + QString("/pca.xml")).toStdString(), cv::FileStorage::READ);

  if (fs.isOpened()) { //In case there is no previous file yet
    fs["mean"] >> pca -> mean;
    fs["e_vectors"] >> pca -> eigenvectors;
    fs["e_values"] >> pca -> eigenvalues;
    fs.release();
  }

}

void GTP::drawEllipses(cv::Mat & img) {

  cv::resize(img, img, cv::Size(int(84 / 1.5), int(96 / 1.5)));

  cv::imwrite("/ramdisk_UVface/imagen.pgm", img); //Here, the image is written to be analyzed by extract_features_64bit.ln

  //Here, extract_features_64bit.ln is called
  FILE * pipe = popen("/ramdisk_UVface/extract_features_64bit.ln -sedgelap -noangle -i /ramdisk_UVface/imagen.pgm", "r");

  /*__________________Here we extract the number of features generated____________________________*/
  /*NOTE: The code is somewhat complicated because we need to find the position in the output stream of extract_features_64bit.ln
  to extract the number of features written to the file /ramdisk_UVface/imagen.pgm*/

  std::string result = "";
  char buffer[60];
  int cl = 0;
  while (!feof(pipe)) {
    if (fgets(buffer, 60, pipe) != NULL) {
      //std::cout << buffer << "\n";
      //getchar();
      if (cl == 4) {
        result += buffer + 16;
        break;
      }
      cl++;
    }
  }

  /*____________________________________________________________________________________________________*/

  fclose(pipe);

  /*Only if the number of features extracted by extract_features_64bit.ln is greater than zero do we read the file "/ramdisk_UVface/imagen.pgm.sedgelap" */
  if ((atoi(result.c_str()) > 0)) {

    fp.seekg(std::ios_base::beg);
    double dp;
    int num;
    fp >> dp; /*This is done only for protocol, to remove the first number*/
    fp >> num; /*In the second line of the file, we find the number of features*/

    std::cout << "number1=" << num << "\n";
    std::cout << "number2=" << atoi(result.c_str()) << "\n";

    /*________Here we store the ellipse parameters in the vec_dp array______*/
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < 5; j++)
        fp >> (vec_dp[i][j]);
    }
    /*_________________________________________________________________________________*/

    //tic();
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num; i++) {

      //std::cout << vec_dp[i][0] << " " << vec_dp[i][1] << " " << vec_dp[i][2] << " " << vec_dp[i][3] << " " << vec_dp[i][4] << "\n";

      cv::Mat MQ(2, 2, CV_64F);
      MQ.at < double > (0, 0) = vec_dp[i][2];
      MQ.at < double > (0, 1) = vec_dp[i][3];
      MQ.at < double > (1, 0) = vec_dp[i][3];
      MQ.at < double > (1, 1) = vec_dp[i][4];

      cv::Mat EIG_val(2, 1, CV_64F);
      cv::Mat EIG_vec(2, 2, CV_64F);

      //std::cout << MQ << "\n";
      eigen(MQ, EIG_val, EIG_vec); //Here, we calculate the eigenvectors and eigenvalues of MQ
      double l1 = 1.0 / sqrt(EIG_val.at < double > (1, 0)); //Major axis, since it is divided by the smaller eigenvector
      double l2 = 1.0 / sqrt(EIG_val.at < double > (0, 0)); //Minor axis, since it is divided by the larger eigenvector

      double alpha = atan2(EIG_vec.at < double > (1, 1), EIG_vec.at < double > (1, 0)); //The major axis is taken as reference

      //std::cout << "l1=" << l1 << " l2=" << l2 << " alpha=" << alpha << "\n";

      double c = vec_dp[i][0];
      double f = vec_dp[i][1];
      //std::cout << " c=" << c << " f=" << f << "\n";
      //getchar();

      cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f(c, f), cv::Size2f(2 * l1, 2 * l2), alpha * (180 / M_PI));
      cv::Rect brect = rRect.boundingRect();
      if ((brect.x >= 0) && (brect.y >= 0) && (brect.x + brect.width < img.cols) && (brect.y + brect.height < img.rows)) {
        //if((l1>minimumSizeAxisEllipses)&&(l2>minimumSizeAxisEllipses)){

        #pragma omp critical
		{
          ellipse(img, cv::Point(c, f), cv::Size(l1, l2), alpha * (180 / M_PI), 0, 360, cv::Scalar(255, 0, 0));
        }

      }

    }

    //toc();
  }

}

void GTP::set_minimumSizeAxisEllipses(const double & minAxis) {
  if ((minAxis <= 0)) return;
  minimumSizeAxisEllipses = minAxis;
}

double GTP::get_minimumSizeAxisEllipses() const {
  return minimumSizeAxisEllipses;
}

int GTP::getNumberUsefulFeatures() const {
  return numberUsefulFeatures;
}

void GTP::seeImagesRect() {
  cv::namedWindow("affine transformation + cropping", CV_WINDOW_NORMAL);

  for (int i = 0; i < numberUsefulFeatures; i++) {
    std::cout << "image number=" << i << "\n";
    //cv::imshow("affine transformation + cropping", imagesRect[i]);
    cv::imshow("affine transformation + cropping", imgGtp[i]);
    cv::waitKey(0);
  }

  cv::destroyWindow("affine transformation + cropping");
}
