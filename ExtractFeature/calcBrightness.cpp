
#include "calcBrightness.h"

float CalcBrightness(const IplImage *pSrcImg)
{
	float flBrightness = 0;

	if (pSrcImg == NULL)
	{
		printf("\nError: in CalcBrightness, input image error!\n");
		exit(0);
	}

	int iWidth = pSrcImg->width;
	int iHeight = pSrcImg->height;
	IplImage *pGryImg = cvCreateImage(cvSize(iWidth, iHeight), IPL_DEPTH_8U, 1);
	cvZero(pGryImg);

	if (pSrcImg->nChannels == 3)	// 3-component
	{
		// cvCvtColor( pSrcImg, pGryImg, CV_BGR2GRAY );
		IplImage *pHSVImg = cvCreateImage(cvSize(iWidth, iHeight), IPL_DEPTH_8U, 3);
		cvCvtColor(pSrcImg, pHSVImg, CV_BGR2HSV);
		cvSplit(pHSVImg, NULL, NULL, pGryImg, NULL);
		cvReleaseImage(&pHSVImg);
	}
	else							// 1-component
	{
		cvCopy(pSrcImg, pGryImg, NULL);
	}

	cvSmooth(pGryImg, pGryImg, CV_GAUSSIAN, 3, 3);

	int iMaxVal = 0, iMinVal = 255, iPixVal = 0;
	int ix = 0, iy = 0, iCnt = 0;

	for (ix = 0; ix < iWidth; ix++)
	{
		for (iy = 0; iy < iHeight; iy++)
		{
			iPixVal = ((uchar*)(pGryImg->imageData + pGryImg->widthStep * iy))[ix];

			if (iPixVal >= iMaxVal) iMaxVal = iPixVal;		// for contrast
			if (iPixVal <= iMinVal) iMinVal = iPixVal;		// for contrast
		}
	}

	//int iThresh = iMaxVal * BRIGHTNESS_ACCEPT_RATIO;
	int iThresh = 17;
	for (ix = 0; ix < iWidth; ix++)
	{
		for (iy = 0; iy < iHeight; iy++)
		{
			iPixVal = ((uchar*)(pGryImg->imageData + pGryImg->widthStep * iy))[ix];
			if (iPixVal >= iThresh) iCnt++;				// for brightness 
		}
	}

	// calculate brightness
    
	float flFraction = float(iCnt) / float(iWidth * iHeight);
	float flRatio = float(1.0) / float(BRIGHTNESS_ACCEPT_FRACTION);

	if (flFraction >= BRIGHTNESS_ACCEPT_FRACTION)
	{
		flBrightness = 0.0;
	}
	else
	{
		flBrightness = -flRatio * flFraction + 1.0;
	}
	flBrightness = 1.0 < flBrightness ? 1.0 : flBrightness;
	flBrightness = 0.0 > flBrightness ? 0.0 : flBrightness;

	cvReleaseImage(&pGryImg);
	return flBrightness;
}


float CalcDarkness(const IplImage *pSrcImg)
{
	float flDarkness = 0;

	if (pSrcImg == NULL)
	{
		printf("\nError: in CalcBrightness, input image error!\n");
		exit(0);
	}

	int iWidth = pSrcImg->width;
	int iHeight = pSrcImg->height;
	IplImage *pGryImg = cvCreateImage(cvSize(iWidth, iHeight), IPL_DEPTH_8U, 1);
	cvZero(pGryImg);

	if (pSrcImg->nChannels == 3)	// 3-component
	{
		// cvCvtColor( pSrcImg, pGryImg, CV_BGR2GRAY );
		IplImage *pHSVImg = cvCreateImage(cvSize(iWidth, iHeight), IPL_DEPTH_8U, 3);
		cvCvtColor(pSrcImg, pHSVImg, CV_BGR2HSV);
		cvSplit(pHSVImg, NULL, NULL, pGryImg, NULL);
		cvReleaseImage(&pHSVImg);
	}
	else							// 1-component
	{
		cvCopy(pSrcImg, pGryImg, NULL);
	}

	cvSmooth(pGryImg, pGryImg, CV_GAUSSIAN, 3, 3);

	int iMaxVal = 0, iMinVal = 255, iPixVal = 0;
	int ix = 0, iy = 0, iCnt = 0;

	for (ix = 0; ix < iWidth; ix++)
	{
		for (iy = 0; iy < iHeight; iy++)
		{
			iPixVal = ((uchar*)(pGryImg->imageData + pGryImg->widthStep * iy))[ix];

			if (iPixVal >= iMaxVal) iMaxVal = iPixVal;		// for contrast
			if (iPixVal <= iMinVal) iMinVal = iPixVal;		// for contrast
		}
	}

	//int iThresh = iMinVal * DARKNESS_ACCEPT_RATIO;
	int iThresh = 240;
	for (ix = 0; ix < iWidth; ix++)
	{
		for (iy = 0; iy < iHeight; iy++)
		{
			iPixVal = ((uchar*)(pGryImg->imageData + pGryImg->widthStep * iy))[ix];
			if (iPixVal <= iThresh) iCnt++;				// for brightness 
		}
	}

	// calculate brightness

	float flFraction = float(iCnt) / float(iWidth * iHeight);
	float flRatio = float(1.0) / float(DARKNESS_ACCEPT_FRACTION);

	if (flFraction >= DARKNESS_ACCEPT_FRACTION)
	{
		flDarkness = 0.0;
	}
	else
	{
		flDarkness = -flRatio * flFraction + 1.0;
		//flDarkness = flFraction;
	}
	flDarkness = 1.0 < flDarkness ? 1.0 : flDarkness;
	flDarkness = 0.0 > flDarkness ? 0.0 : flDarkness;

	cvReleaseImage(&pGryImg);
	return flDarkness;
}