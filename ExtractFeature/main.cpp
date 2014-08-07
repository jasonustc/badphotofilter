#include "calcBlur.h"
#include "calcBrightness.h"
#include "Global.h"
#include "myFeatures.h"
#include <opencv2/opencv.hpp>
#include <Windows.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

#define ALL_FEATURE_DIM 43
#define DEEP_FEATURE_DIM  1024
#define LOCAL_CONTRAST_FEATURE_DIM 12 
#define GLOBAL_FEATURE_DIM 11
#define SUB_FEATURE_DIM 9 
#define HSV_FEATURE_DIM  11
//resize the image into MAX_IMAGE_WIDTH x something
#define MAX_IMAGE_WIDTH 1024

using namespace std;
using namespace cv;

int IsGoodImage(const string& imgPath);
bool LoadDataSet(const string& inFileName, vector<string> &dataSet, vector<int> &dataSetIsGoodVec);
bool LoadImgFeature(const string& fileName, const vector<string> &dataSet, const vector<int> &dataSetIsGoodVec,Mat &dataFeatMat);
bool SplitDataSetAndSave(const vector<int> &dataSetIsGoodVec, Mat &dataFeatMat, const vector<string> &dataSet);
bool SaveImgFeature(int k, const Mat &trainData, const Mat& testData, const Mat& trainLabel, const Mat& testLabel);
bool LoadSingleImgFeature(const string &imgPath, vector<float> &featData);
bool LoadSVMmodel(CvSVM &badPhotoClassifer, vector<float> &featData, Mat &testFeature);
bool outResult(string &imgPath, const int label, const float score,vector<float> &featData);

int main(int argc, char** argv)
{
	//	cout<<"Please enter the image path file that you want to assess:  "<<endl;
	//////////////////////////////////////////////////////////////////////////////////////////
    string imgPath="imgPath.txt";
	bool SVM = TRUE;
	string imgSuffix = imgPath;
	imgSuffix.replace(imgSuffix.begin(), imgSuffix.end() - 3, "");
	if ("txt" == imgSuffix)
	{
		vector<string> dataSetPath;
		vector<int> dataSetIsGoodVec;
		LoadDataSet(imgPath, dataSetPath,dataSetIsGoodVec);
		if(SVM){
			//load image feature.
			Mat dataFeatMat;
			LoadImgFeature(imgPath, dataSetPath, dataSetIsGoodVec,dataFeatMat);
            //split the dataset into training set and test set and save them into xml.
			SplitDataSetAndSave(dataSetIsGoodVec, dataFeatMat,dataSetPath);
		}
		else{
			ofstream outfeats("features.txt");
			for (unsigned int i = 0; i < dataSetPath.size(); i++)
			{
				//load image feature.
				vector<float> featData;
				bool hr = LoadSingleImgFeature(dataSetPath[i], featData);
				cout << dataSetPath[i] << "\n";
				outfeats << dataSetPath[i] << "\t";
				for (unsigned int j = 0; j < featData.size(); j++)
				{
					outfeats << featData[j] << "\t";
				}
				outfeats << "\n";
			}
			outfeats.close();
		}
		return 0;
	}
	else if ("jpg" == imgSuffix)
	{
        //load image feature.
		vector<float> featData;
		bool hr=LoadSingleImgFeature(imgPath, featData);
		if (!hr)
		{
			return -2;
		}
        //load SVM model
		CvSVM badPhotoClassifer;
		Mat testFeature=Mat::zeros(1, featData.size(), CV_32FC1);
		LoadSVMmodel(badPhotoClassifer, featData, testFeature);
        //predict.		
		float decision_value = badPhotoClassifer.predict(testFeature, true);
		int label = badPhotoClassifer.predict(testFeature);
		decision_value = 1.0 / (1.0 + exp(decision_value));
		//output result.
		outResult(imgPath, label, decision_value, featData);
		return label;
	}
	else{
		printf("%s is not an jpg image or an image path file!\n",imgPath);
		//system("pause");
		return -2;
	}
	//system("pause");
}
// to judge if a photo is good or bad photo.
int IsGoodImage(const string& imgPath){
	int imgFileNamePos = imgPath.rfind('\\');
	char flagChar = imgPath[imgFileNamePos + 1];
	if (flagChar == 'G'){
		return 1;
	}
	else if (flagChar == 'B'){
		return -1;
	}
	else{
		string errMsg = "Error: unrecognized image path name!";
		errMsg.append(imgPath);
		throw runtime_error(errMsg);
	}

}
// load paths of all the photos
bool LoadDataSet(const string& inFileName, vector<string> &dataSet, 
	vector<int> &dataSetIsGoodVec){
	dataSet.clear();
	dataSetIsGoodVec.clear();

	ifstream in(inFileName.c_str());
	string line;
	while (in.good())
	{
		getline(in, line);
		if (line.length() == 0)
		{
			break;
		}
		dataSet.push_back(line);
	}
	random_shuffle(dataSet.begin(), dataSet.end());

	int goodImgNum = 0;
	int badImgNum = 0;
	for (auto iter = dataSet.begin(); iter != dataSet.end(); iter++){
		int isGood = IsGoodImage(*iter);
		dataSetIsGoodVec.push_back(isGood);
		if (isGood == 1){
			goodImgNum++;
		}
		else{
			badImgNum++;
		}
	}

	in.close();

	cout << "There are " << goodImgNum << " good photos in this set" << endl;
	cout << "There are " << badImgNum << " bad photos in this set" << endl;

	return true;
}

inline bool SaveMatToXml(const string& fileName, const Mat &data){
	FileStorage fs(fileName.c_str(), FileStorage::WRITE);
	fs << "vocabulary" << data;
	fs.release();

	return true;
}

//save data into opencv xml.
bool SaveImgFeature(int k, const Mat &trainData, const Mat& testData, const Mat& trainLabel, const Mat& testLabel){
	ostringstream oss;
	oss << k;

	int trainNum = trainData.rows;
	int testNum = testData.rows;

	//save label
	string trainLabelfile = "trainLabel" + oss.str() + ".xml";
	SaveMatToXml(trainLabelfile, trainLabel);

//	string validLabelfile = "validLabel" + oss.str() + ".xml";
//	SaveMatToXml(validLabelfile, validLabel);

	string testLabelfile = "testLabel" + oss.str() + ".xml";
	SaveMatToXml(testLabelfile, testLabel);

	//save all feature
	string trainDatafile = "trainDataAll" + oss.str() + ".xml";
	SaveMatToXml(trainDatafile, trainData);
//	string validDatafile = "validDataAll" + oss.str() + ".xml";
//	SaveMatToXml(validDatafile, validData);
	string testDatafile = "testDataAll" + oss.str() + ".xml";
	SaveMatToXml(testDatafile, testData);

	return true;
}

bool resizeImg(Mat &srcImg);
// load image features by given path
bool LoadImgFeature(const string& fileName, const vector<string> &dataSet, const vector<int> &dataSetIsGoodVec,Mat &dataFeatMat){
	int imgNum = dataSet.size();
	int testImgNum = imgNum / 5;
	int trainImgNum = imgNum - testImgNum;
	cout << "total train:\t" << trainImgNum << "\n";
	cout << "total test:\t" << testImgNum << "\n";
	dataFeatMat = Mat::zeros(imgNum, ALL_FEATURE_DIM, CV_32FC1);
	for (int i = 0; i < imgNum; i++)
	{
        int ErrCode;
		cout << dataSet[i] << "\n";
		Mat testImg = imread(dataSet[i]);
		resizeImg(testImg);
		if (0 == testImg.rows || 0 == testImg.cols)
		{
			cerr << "number " << i << " : something wrong when loading the image" << endl;
			continue;
		}
		if (testImg.channels() == 1)
		{
			cvtColor(testImg, testImg, CV_GRAY2BGR);
		}
        //local feature 12 dims
        localFeatExtractor mylocalFeature(testImg);
        ErrCode=mylocalFeature.computeLocalFeat();
        if(ErrCode){
            return ErrCode;
		}
        //global feature: 8+1+2+11+9 
        globalFeatExtractor myglobalFeature(testImg);
        ErrCode=myglobalFeature.computeGlobalFeat();
        if(ErrCode){
            return ErrCode;
		}
        if(mylocalFeature.localFeat.size()+myglobalFeature.globalFeat.size() != ALL_FEATURE_DIM){
            printf("Check your feature dim!\n");
            system("pause");
            return false;
		}
        float* pRow=dataFeatMat.ptr<float>(i);
        for(int j=0;j<mylocalFeature.localFeat.size();j++){
            pRow[j]=mylocalFeature.localFeat[j];
		}
        for(int j=0;j<myglobalFeature.globalFeat.size();j++){
            pRow[j+mylocalFeature.localFeat.size()]=myglobalFeature.globalFeat[j];
		}
	}
	return true;
}
bool LoadSingleImgFeature(const string &imgPath, vector<float> &featData)
{
    int ErrCode;
	Mat testImg = imread(imgPath);
	Mat resizedImg;
	//resizedImg = testImg;
	resizeImg(testImg);
	if (0 == testImg.rows || 0 == testImg.cols)
	{
		printf("load image %s failed!\n", imgPath.c_str());
		return false;
	}
	if (testImg.channels() == 1)
	{
		cvtColor(testImg, testImg, CV_GRAY2BGR);
	}
	//extract feature.
	localFeatExtractor mylocalFeature(testImg);
	ErrCode=mylocalFeature.computeLocalFeat();
	if(ErrCode){
		return ErrCode;
	}
	for(int j=0;j<mylocalFeature.localFeat.size();j++){
		featData.push_back(mylocalFeature.localFeat[j]);
	}
	//blur,entropy,brightness,blurdiff,sharpness,contrast,edge.
	globalFeatExtractor myglobalFeature(testImg);
	ErrCode=myglobalFeature.computeGlobalFeat();
	if(ErrCode){
		return ErrCode;
	}
	for(int j=0;j<myglobalFeature.globalFeat.size();j++){
		featData.push_back(myglobalFeature.globalFeat[j]);
	}
	return true;
}

bool LoadSVMmodel(CvSVM &badPhotoClassifer,vector<float> &featData,Mat &testFeature)
{
	Mat normParams;//normalization param.
	string classifierPath = "SVM_model.xml";
	badPhotoClassifer.load(classifierPath.c_str());
	normParams = Mat::zeros(2, badPhotoClassifer.get_var_count(), CV_32FC1);
	vector<float> tempMax, tempMin;
	FileStorage fs(classifierPath, FileStorage::READ);
	fs["maxDimensions"] >> tempMax;
	fs["minDimensions"] >> tempMin;
	fs.release();
	for (unsigned int i = 0; i < tempMax.size(); i++)
	{
		normParams.at<float>(0, i) = tempMax[i];
		normParams.at<float>(1, i) = tempMin[i];
	}
	for (unsigned int i = 0; i < featData.size(); i++)
	{
		if (normParams.at<float>(0, i) != normParams.at<float>(1, i))
		{
			testFeature.at<float>(i) = (featData[i] - normParams.at<float>(1, i))
				/ (normParams.at<float>(0, i) - normParams.at<float>(1, i));
		}
	}
	return true;
}
bool SaveImgPath(const vector<string> &ImgPath, const string &fileName,int k)
{
	ostringstream oss;
	oss << k;
	string filePath = fileName + oss.str()+".txt";
	ofstream outImgPath(filePath.c_str());
	if (!outImgPath.is_open())
	{
		printf("Open image path file %s failed!", filePath);
		return false;
	}
	for (unsigned int i = 0; i < ImgPath.size(); i++){
		outImgPath << ImgPath[i] << "\n";
	}
	outImgPath.close();
	oss.clear();
	return true;
}
// split the dataset into 4/5 training and 1/5 testing.
bool SplitDataSetAndSave(const vector<int> &dataSetIsGoodVec, Mat &dataFeatMat,const vector<string> &dataSet)
{
	int imgNum = dataFeatMat.rows;
	vector<string> trainImg;
    vector<string> testImg;
	for (int k = 0; k < 5; k++)//split the dataset to 5 parts, 4 for training, 1 for testing.
	{
		cout << "iteration: " << k << "\n";
		Mat trainData,testData;
		Mat trainLabel,testLabel;
		trainImg.clear();
		testImg.clear();
		int trainNum = 0;
		int testNum = 0;
		for (int i = 0; i < imgNum; i++)// split photos
		{
			float *pFeatRow = dataFeatMat.ptr<float>(i);
			if (i % 5 == k)// test
			{
				for (unsigned int j = 0; j < ALL_FEATURE_DIM; j++)
				{
					testData.push_back(pFeatRow[j]);
				}
				testLabel.push_back(dataSetIsGoodVec[i]);
				testImg.push_back(dataSet[i]);
				testNum++;
			}
			else //  train
			{
				for (unsigned int j = 0; j < ALL_FEATURE_DIM; j++)
				{
					trainData.push_back(pFeatRow[j]);
				}
				trainLabel.push_back(dataSetIsGoodVec[i]);
				trainImg.push_back(dataSet[i]);
				trainNum++;
			}

		}

		trainData = trainData.reshape(1, trainNum);
		testData = testData.reshape(1, testNum);
		trainLabel = trainLabel.reshape(1, trainNum);
		testLabel = testLabel.reshape(1, testNum);
		SaveImgFeature(k, trainData, testData, trainLabel,testLabel);

		string trainfile = "trainImg";
		string testfile = "testImg";
		SaveImgPath(trainImg,trainfile,k);
		SaveImgPath(testImg, testfile,k);
	}
	return true;
}
bool outResult(string &imgPath,const int label,const float score,vector<float> &featData)
{
	int xx = imgPath.rfind(".");
	string filename = imgPath.substr(0, xx);
	filename += "_result.txt";
	ofstream out_prediction(filename.c_str());
	if (label == 1)
	{
		cout << imgPath << "\tis a good photo." << "\n";
		cout << "And the quality score is:\t" << score << "\n";
		out_prediction << "Label: good.\n";
		out_prediction << "Score:\t" << score << "\n";
	}
	else if (label == -1)
	{
		cout << imgPath << "\tis a bad photo." << "\n";
		cout << "And the quality score is:\t" << score << "\n";
		out_prediction << "Label: bad.\n";
		out_prediction << "Score:\t" << score << "\n";
	}
	out_prediction <<"Blur:\t"<< 1-featData[12] << "\n";
	out_prediction <<"Sharpness:\t"<< featData[15] << "\n";
	out_prediction <<"Contrast:\t"<< featData[16] << "\n";
	out_prediction <<"Brightness:\t"<< 1-featData[17] << "\n";
	out_prediction.close();
	return true;
}
bool resizeImg(Mat &srcImg)
{
	if (srcImg.cols == 0 && srcImg.rows == 0)
	{
		printf("error in resizing the image.\n");
		return false;
	}
	if (srcImg.cols > MAX_IMAGE_WIDTH)
	{
		float scale = MAX_IMAGE_WIDTH/ double(srcImg.cols);
		Size size = Size(srcImg.rows*scale,MAX_IMAGE_WIDTH);
		resize(srcImg, srcImg, size);
	}
	return true;
}