#pragma once
#include "SVMfilter.h"

bool LoadFeat(const string &featFile, cv::Mat featData, const int featDim);

bool LoadFeat(const string & filename, cv::Mat &featMat);

bool IsNumber(float x);

bool WipeIllegalData(cv::Mat &featMat);

bool WipeIllegalChar(string &path);

/**
*   get the normalization coefficients of the photo features
*
*   @param featData: the Features of all photos
*   @param maxDim: the max value in each column of the feature mat
*   @param minDim: the min value in each column of the feature mat
*/
bool MaxMinInRow(cv::Mat &featData, vector<float> &maxDim, vector<float> &minDim);

/**
*   compute the AP of the photo rank
*
*   @param scores: the SVM predict score of all photos
*   @param badscores: the SVM predict score of bad photos
*/
float ComputeAP(vector<float> &scores, vector<float> &badscores);

bool MatPCA(cv::Mat &featData, cv::Mat &Mean, cv::Mat &EigVals, cv::Mat &EigVecs, const int Dim);

bool CheckDir(const string &dirPath);

bool ComputePreRec(vector<int> &predLabels, vector<int> &trueLabels, float &precision,
	float &recall, float &Fscore);

bool ComputePreRec(vector<int> &predLabels, vector<int> &trueLabels, float &precision,
	float &recall, float &Fscore, ofstream &out_PreRec);

bool ComputePreRec(vector<int> &predLabels, vector<int> &trueLabels, float &precision,
	float &recall, float &Fscore, const  string &imgPathFile, ofstream &out_PreRec);

void MatNorm(cv::Mat& featData,vector<float> &maxCoeffs,vector<float> &minCoeffs);

bool LoadImgPath(const string & imgPathFile, vector<string> &imgPath);