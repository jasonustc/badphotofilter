#include "UsefulHead.h"
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "calcBlur.h"
#include "ErrorCode.h"
#include "calcBrightness.h"
#include "time.h"

class localFeatExtractor
{
public:
	localFeatExtractor(Mat &srcImg);
	int computeLocalFeat();
	int computeLocalFeat2();
public:
	vector<float> localFeat;
private:
	float Sharpness(Mat &srcImg);
	float BlurDiff(Mat &srcImg);
	float pixContrast(Mat &srcImg);
	float freContrast(Mat &srcImg);
	float laplacePixEn(Mat &srcImg);
	float pixSimp(Mat &srcImg);
	float CalHighSpec(Mat &img);
    int CalcDyrAndInc(Mat &srcImg,float &dyRan, float &inCon);
	Mat imgData;
};

class globalFeatExtractor{
public:
	globalFeatExtractor(Mat &srcImg);
	int computeGlobalFeat();

public:
	vector<float> globalFeat;

private:
	float calcHighSpec(Mat &srcImg);
	float calcSharpness();
	float calcBlurDiff();
	float calcContrast();
	float calclaplacePixEn();
	float calcDarkness();
	float calcBrightness();
	int calcSubblockInfo(vector<float>& subareaInfo);
	int calcHistHSV(vector<float> &histFeat);
    int calcHistHSV2(vector<float> &histFeat);
	int calcBlurSharp(float& flBlur,float &flSharp);
    int CalcDofFeat(float &dofFeat1,float &dofFeat2);
    int CalcDyrAndInc(Mat &srcImg,float &dyRan,float &inCon);
	float calcSimp();

private:
	Mat imgData;
};