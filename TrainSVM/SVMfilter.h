#pragma once
#include <iostream>
#include <opencv2\opencv.hpp>
#include <vector>
#include <Windows.h>
#include <fstream>
#include <sstream>
#include <ctime>
#include <functional>
#include <cmath>
#include "io.h"
#include <numeric>

using namespace std;


class SVMmodel{

public:

	SVMmodel(const string &trainFeatFile, const string &trainLabelFile,
		const string &testFeatFile, const string &testLabelFile);

	SVMmodel(const string &trainFeatFile, const string &trainLabelFile,
		const string &testFeatFile, const string &testLabelFile,
		const float NegWeight,const float PosWeight);

	SVMmodel(const int dataSetIdx,const int featDim1,const int featDim2,
		const string trainImgPathFile);

	SVMmodel(const int dataSetIdx, const int featDim1, const int featDim2);

public:

	bool TrainAndTest(ofstream &out_PreRec,const float threshold);

	bool CrossValidAndTest(ofstream &out_PreRec, const float threshold,const int k);

private:

	bool chooseThreshold();

	bool TrainModel(cv::Mat &trData, cv::Mat & trLabel);

	bool TestModel(string &SVMfile, string &imgPathFile,
		ofstream &out_PreRec, const float &Threshold);

	bool TestModel(string &SVMfile, ofstream &out_PreRec, const float &Threshold);

    int CheckDarkOrExpo(cv::Mat &featData);

    bool CheckDarkOrExpo(const string &imgPathFile,cv::Mat &featMat,cv::Mat &labelMat);

    bool ChooseSubFeat(cv::Mat &featMat,vector<int> &colMark);
    

private:

	cv::Mat trainData;
	cv::Mat testData;
	cv::Mat trainLabel;
	cv::Mat testLabel;

	vector<float> maxCoeffs;
	vector<float> minCoeffs;

	float bestGamma;
	float bestC;
	float NegWeight;
	float PosWeight;
};