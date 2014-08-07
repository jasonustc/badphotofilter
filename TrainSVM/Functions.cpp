#pragma once
#include "Functions.h"

bool LoadFeat(const string &featFile,cv::Mat featData,const int featDim){
	ifstream loadFeature(featFile.c_str());
	if (!loadFeature){
		fprintf(stderr, "check the path:\t%s\n", featFile);
		exit(0);
	}
	//load train data.
	float feat = 0;
	int featNum = 0;
	while (loadFeature >> feat){
		featData.push_back(feat);
		featNum++;
	}
	if (featData.rows == 0 || featData.cols == 0){
		fprintf(stderr, "load feature in %s failed!\n", featFile);
		exit(0);
	}
	if (featNum%featDim != 0){
		fprintf(stderr, "check your feature dim in %s.\n",featFile);
		exit(0);
	}
	int sampleNum = featNum / featDim;
	featData.reshape(1, sampleNum);
	loadFeature.close();
	return true;
}

bool LoadFeat(const string & filename, cv::Mat &featMat)
{
	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		cout << "load data file " << filename << " failed!" << "\n";
		exit(0);
	}
	fs["vocabulary"] >> featMat;
	if (featMat.rows == 0 || featMat.cols == 0){
		fprintf(stderr, "empty data in %s\n", filename);
		exit(0);
	}
	fs.release();
	return true;
}

bool LoadImgPath(const string & imgPathFile, vector<string> &imgPath)
{
	ifstream  inImgPath(imgPathFile.c_str());
	string imgName;
	while (inImgPath.good())
	{
		getline(inImgPath, imgName);
		if (imgName.length() == 0)
		{
			break;
		}
		imgPath.push_back(imgName);
	}
	if (imgPath.size() == 0)
	{
		printf("Load image file %s failed!\n", imgPath);
		return false;
	}
	return true;
}

bool IsNumber(float x)
{
	return (x == x);
}

bool WipeIllegalData(cv::Mat &featMat)
{
	for (int i = 0; i<featMat.rows; i++)
	{
		if (!IsNumber(featMat.at<float>(i, 0)))
		{
			for (int j = 0; j<featMat.cols; j++)
			{
				featMat.at<float>(i, j) = 0;
			}
		}
	}
	return true;
}

bool WipeIllegalChar(string &path)
{
	//delte the ":" in the string.
	replace(path.begin(), path.end(), ':', '-');
	replace(path.begin(), path.end(), ' ', '-');
	return true;
}

float ComputeAP(vector<float> &scores, vector<float> &badscores)
{
	sort(scores.begin(), scores.end());
	sort(badscores.begin(), badscores.end());
	float AP = 0;
	for (int i = 0; i < badscores.size(); i++)
		//AP=sum(i/n):i: location in bad photo list. n: location in all photo list.
	{
		vector<float>::iterator result = find(scores.begin(), scores.end(), badscores[i]);
		int loc = int(result - scores.begin());
		if (result != scores.end())
		{
			AP += float(i + 1) / (loc + 1);
			cout << loc << "\t";
		}
	}
	AP = AP / badscores.size();
	return AP;
}

bool MatPCA(cv::Mat &featData, cv::Mat &Mean, cv::Mat &EigVals, cv::Mat &EigVecs, const int Dim)
{
	cv::Mat pMean = cvCreateMat(1, featData.cols, CV_32FC1);
	cv::Mat pEigVals =cvCreateMat(1, min(featData.rows, featData.cols), CV_32FC1);
	cv::Mat pEigVecs = cvCreateMat(min(featData.rows, featData.cols), featData.cols, CV_32FC1);
	IplImage pcaFeatData = featData;
	IplImage pcaMean = pMean;
	IplImage pcaEigVals = pEigVals;
	IplImage pcaEigVecs = pEigVecs;
	cvCalcPCA(&pcaFeatData, &pcaMean, &pcaEigVals, &pcaEigVecs, CV_PCA_DATA_AS_ROW);
	cv::Mat pResult = cvCreateMat(featData.rows, Dim, CV_32FC1);
	IplImage pcaResult = pResult;
	cvProjectPCA(&pcaFeatData, &pcaMean, &pcaEigVecs, &pcaResult);
	Mean = cv::Mat(&pcaMean, false);
	EigVecs = cv::Mat(&pcaEigVecs, false);
	return true;
}

bool MaxMinInRow(cv::Mat &featData, vector<float> &maxDim, vector<float> &minDim)
{
	maxDim.clear();
	minDim.clear();
	maxDim.resize(featData.cols, -10000);
	minDim.resize(featData.cols, FLT_MAX);
	for (int i = 0; i<featData.rows; i++)
	{
		for (int j = 0; j<featData.cols; j++)
		{
			if (featData.at<float>(i, j)>maxDim[j])
			{
				maxDim[j] = featData.at<float>(i, j);
			}
			if (featData.at<float>(i, j)<minDim[j])
			{
				minDim[j] = featData.at<float>(i, j);
			}
		}
	}
	return true;
}

bool CheckDir(const string &dirPath)
{
	if (_access(dirPath.c_str(), 0) != -1)
	{
		string cmd = "rd /s /q " + dirPath;
		system(cmd.c_str());
	}
	if (_access(dirPath.c_str(), 0) == -1)
	{
		string cmd = "md " + dirPath;
		system(cmd.c_str());
	}
	if (_access(dirPath.c_str(), 0) == -1)
	{
		cout << "Create folder " << dirPath << "failed!" << "\n";
		return false;
	}
	return true;
}

bool ComputePreRec(vector<int> &predLabels, vector<int> &trueLabels, float &precision,
	float &recall, float &Fscore)
{
	//ofstream out_PreRec("PreRec.txt");
	float truePtrue = 0, truePfalse = 0;
	float falsePfalse = 0, falsePtrue = 0;
	for (int i = 0; i < predLabels.size(); i++)
	{
		if (predLabels[i] == 1 && trueLabels[i] == 1)
		{
			truePtrue++;
		}
		if (predLabels[i] == -1 && trueLabels[i] == 1)
		{
			truePfalse++;
		}
		if (predLabels[i] == 1 && trueLabels[i] == -1)
		{
			falsePtrue++;
		}
		if (predLabels[i] == -1 && trueLabels[i] == -1)
		{
			falsePfalse++;
		}
	}
	precision = falsePfalse / (falsePfalse + truePfalse);
	recall = falsePfalse / (falsePfalse + falsePtrue);
	Fscore = 2 * precision*recall / (precision + recall);
	return true;
}

bool ComputePreRec(vector<int> &predLabels, vector<int> &trueLabels, float &precision,
	float &recall, float &Fscore, const  string &imgPathFile, ofstream &out_PreRec)
{
	bool hr;
	vector<string> imgPath;
	hr = LoadImgPath(imgPathFile, imgPath);
	if (!hr)
	{
		return false;
	}
	float truePtrue = 0, truePfalse = 0;
	float falsePfalse = 0, falsePtrue = 0;
	string truePosPath = "truePos";
	CheckDir(truePosPath);
	string falsePosPath = "faslePos";
	CheckDir(falsePosPath);
	string trueNegPath = "trueNeg";
	CheckDir(trueNegPath);
	string falseNegPath = "falseNeg";
	CheckDir(falseNegPath);
	for (int i = 0; i < predLabels.size(); i++)
	{
		if (predLabels[i] == 1 && trueLabels[i] == 1)
		{
            //cout<<imgPath[i]<<"\n";
			truePtrue++;
			int xx = imgPath[i].rfind("\\");
			string outImgPath = imgPath[i].substr(xx);
			outImgPath = truePosPath + outImgPath;
            //cout<<outImgPath<<"\n";
			CopyFileA(imgPath[i].c_str(), outImgPath.c_str(), FALSE);
		}
		else if (predLabels[i] == -1 && trueLabels[i] == 1)
		{
			truePfalse++;
			int xx = imgPath[i].rfind("\\");
			string outImgPath = imgPath[i].substr(xx);
			outImgPath = falsePosPath + outImgPath;
			CopyFileA(imgPath[i].c_str(), outImgPath.c_str(), FALSE);
		}
		else if (predLabels[i] == 1 && trueLabels[i] == -1)
		{
			falsePtrue++;
			int xx = imgPath[i].rfind("\\");
			string outImgPath = imgPath[i].substr(xx);
			outImgPath = falseNegPath + outImgPath;
			CopyFileA(imgPath[i].c_str(), outImgPath.c_str(), FALSE);
		}
		else if (predLabels[i] == -1 && trueLabels[i] == -1)
		{
			falsePfalse++;
			int xx = imgPath[i].rfind("\\");
			string outImgPath = imgPath[i].substr(xx);
			outImgPath = trueNegPath + outImgPath;
			CopyFileA(imgPath[i].c_str(), outImgPath.c_str(), FALSE);
		}
	}
	cout << "\ntrue, predict true:  " << truePtrue << endl;
	cout << "true, predict false:  " << truePfalse << endl;
	cout << "false, predict true:  " << falsePtrue << endl;
	cout << "false, predict false:  " << falsePfalse << endl;
	precision = falsePfalse / (falsePfalse + truePfalse);
	recall = falsePfalse / (falsePfalse + falsePtrue);
	Fscore = 2 * precision*recall / (precision + recall);
	out_PreRec << truePtrue << "\t" << truePfalse << "\t" << falsePtrue << "\t" << falsePfalse << "\t";
	out_PreRec << precision << "\t" << recall << "\t" << Fscore << "\n";
	return true;
}

bool ComputePreRec(vector<int> &predLabels, vector<int> &trueLabels, float &precision,
	float &recall, float &Fscore, ofstream &out_PreRec)
{
	float truePtrue = 0, truePfalse = 0;
	float falsePfalse = 0, falsePtrue = 0;
	for (int i = 0; i < predLabels.size(); i++)
	{
		if (predLabels[i] == 1 && trueLabels[i] == 1)
		{
			truePtrue++;
		}
		else if (predLabels[i] == -1 && trueLabels[i] == 1)
		{
			truePfalse++;
		}
		else if (predLabels[i] == 1 && trueLabels[i] == -1)
		{
			falsePtrue++;
		}
		else if (predLabels[i] == -1 && trueLabels[i] == -1)
		{
			falsePfalse++;
		}
	}
	cout << "\ntrue, predict true:  " << truePtrue << endl;
	cout << "true, predict false:  " << truePfalse << endl;
	cout << "false, predict true:  " << falsePtrue << endl;
	cout << "false, predict false:  " << falsePfalse << endl;
	precision = falsePfalse / (falsePfalse + truePfalse);
	recall = falsePfalse / (falsePfalse + falsePtrue);
	Fscore = 2 * precision*recall / (precision + recall);
	out_PreRec << truePtrue << "\t" << truePfalse << "\t" << falsePtrue << "\t" << falsePfalse << "\t";
	out_PreRec << precision << "\t" << recall << "\t" << Fscore << "\n";
	return true;
}

void MatNorm(cv::Mat& featData,vector<float> &maxCoeffs,vector<float> &minCoeffs)
{
	//////////////////
	//////////////////
	//normalization
	for (int i = 0; i<featData.rows; i++)
	{
		for (int j = 0; j<featData.cols; j++)
		{
			if (minCoeffs[j] != maxCoeffs[j])
			{
				featData.at<float>(i, j) = (featData.at<float>(i, j) - minCoeffs[j]) 
					/ (maxCoeffs[j] - minCoeffs[j]);
			}
			else{
                featData.at<float>(i,j)=0;
			}	
		}
	}
}