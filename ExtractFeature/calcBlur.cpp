#include "calcBlur.h"


int CalcBlur(const IplImage *pSrcImg,float &flBlur,float &flSharp)
{
//	float flBlur = 0, flSharp = 0;

	if (pSrcImg == NULL)
	{
		printf("\nError: in CalcBlur, input image error!\n");
        return EMPTY_IMAGE_DATA;
	}

	int iWidth = pSrcImg->width;
	int iHeight = pSrcImg->height;
	IplImage *pGryImg = cvCreateImage(cvSize(iWidth, iHeight), IPL_DEPTH_8U, 1);
	cvZero(pGryImg);

	if (pSrcImg->nChannels == 3)	// 3-component
	{
		cvCvtColor(pSrcImg, pGryImg, CV_BGR2GRAY);
	}
	else							// 1-component
	{
		cvCopy(pSrcImg, pGryImg, NULL);
	}

	int i = 0, j = 0;
	float *pImgData = new float[iHeight * iWidth];
	for (i = 0; i < iHeight; i++)
	{
		for (j = 0; j < iWidth; j++)
		{
			pImgData[iWidth * i + j] = (float)(((uchar*)(pGryImg->imageData + pGryImg->widthStep * i))[j]);
		}
	}

	CBlurDetector BlurDetector;
	HRESULT hr = BlurDetector.Detect(pImgData, iWidth, iHeight, flBlur,flSharp);

	DEL_ARRAY(pImgData);
	cvReleaseImage(&pGryImg);
	return FILTER_OK;
}

int CalcBlur2(const IplImage *pSrcImg,float &flBlur,float &flSharp)
{
//	float flBlur = 0, flSharp = 0;

	if (pSrcImg == NULL)
	{
		printf("\nError: in CalcBlur, input image error!\n");
        return EMPTY_IMAGE_DATA;
	}

	int iWidth = pSrcImg->width;
	int iHeight = pSrcImg->height;
	IplImage *pGryImg = cvCreateImage(cvSize(iWidth, iHeight), IPL_DEPTH_8U, 1);
	cvZero(pGryImg);

	if (pSrcImg->nChannels == 3)	// 3-component
	{
		cvCvtColor(pSrcImg, pGryImg, CV_BGR2GRAY);
	}
	else							// 1-component
	{
		cvCopy(pSrcImg, pGryImg, NULL);
	}

	int i = 0, j = 0;
	float *pImgData = new float[iHeight * iWidth];
	for (i = 0; i < iHeight; i++)
	{
		for (j = 0; j < iWidth; j++)
		{
			pImgData[iWidth * i + j] = (float)(((uchar*)(pGryImg->imageData + pGryImg->widthStep * i))[j]);
		}
	}

	CBlurDetector BlurDetector;
	HRESULT hr = BlurDetector.Detect2(pImgData, iWidth, iHeight, flBlur,flSharp);

	DEL_ARRAY(pImgData);
	cvReleaseImage(&pGryImg);
	return FILTER_OK;
}