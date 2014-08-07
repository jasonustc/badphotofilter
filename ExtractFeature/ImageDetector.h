//-----------------------------------------------------------------------------
// MSR China Media Computing Group
//
// Copyright 1999-2002 Microsoft Corporation. All Rights Reserved.
//
// File: ImageDetector.h -	the definition of Class CImageDetector for:
//							Image Orientation Detection, 
//							Indoor/Outdoor, 
//							City/Landscape.
// 
// Owner:	Yanfeng Sun  (yfsun@microsoft.com)
//			Lei Zhang
// 
// Last Modified: Feb. 28, 2002 by Yanfeng Sun
//				  April 12, 2002 by Lei Zhang, replace SVM with AdaBoost
//				  April 07, 2005 by Tao Mei, f-tmei.
//-----------------------------------------------------------------------------

#pragma once

#include <Windows.h>

//#include "DetectionModel.h"
//#include "Feature.h"
class CDetectionModel;


#ifndef LENGTH_EdgeDirHistogram
#define LENGTH_EdgeDirHistogram 13
#endif

//The definition for detection result
#define	DETECTION_UNKNOWN		-1
#define DETECTION_INDOOR		1
#define DETECTION_OUTDOOR		2
#define DETECTION_CITY			1
#define DETECTION_LANDSCAPE		2

#ifdef IMAGEDETECTOR
#define IDEXTERN _declspec(dllexport)
#else
#define IDEXTERN _declspec(dllimport)
#endif

#if _DEBUG
#pragma comment(lib,"CImageDetectorD.lib")
#else
#pragma comment(lib,"CImageDetector.lib")
#endif

//=============================================================================================
//struct for detection result
typedef struct
{
	int iOrientation;			// the orientation result, the result is in 0, 90, 180 and 270
	float fConfOrientation;		// the confidence level for orientation result, between 0 and 1

	int iIO;					// The result for indoor/outdoor. The value should be in DETECTION_INDOOR, 
								// DETECTION_OUTDOOR or DETECTION_UNKNOWN. Other value is meaningless. 
	float fConfIO;				// The confidence level for the result of indoor/outdoor detection.

	int iCL;					// The result for city/landscape. The value should be in DETECTION_CITY,
								// DETECTION_LANDSCAPE or DETECTION_UNKNOWN. Other value is meaningless.
	
	float fConfCL;				// The confidence level for the result of city/landscape detection.

} DETECTRESULT, * LPDETECTRESULT;

//////////////////////////////////////////////////////////////////////////
// Add by Tao Mei, 2004-12-19
typedef struct 
{
	float fEntropy0 ;			// edge direction entropy of 0 degree
	float fEntropy90 ;			// edge direction entropy of 90 degree
	float fEntropy180 ;			// edge direction entropy of 180 degree
	float fEntropy270 ;			// edge direction entropy of 270 degree

	float pflEdge0[LENGTH_EdgeDirHistogram];	// edge direction histogram of 0 degree
	float pflEdge90[LENGTH_EdgeDirHistogram];	// edge direction histogram of 90 degree
	float pflEdge180[LENGTH_EdgeDirHistogram];	// edge direction histogram of 180 degree
	float pflEdge270[LENGTH_EdgeDirHistogram];	// edge direction histogram of 270 degree

} DETECTEDGEHIST, * LPDETECTEDGEHIST ;
// Add End
//////////////////////////////////////////////////////////////////////////

//=============================================================================================
//Wrapper class for three image detectors
class IDEXTERN CImageDetector
{
private:
	CDetectionModel *m_ModelOrientation;		//model for orientation detection
	CDetectionModel *m_ModelIO;				//model for indoor/outdoor classificaiton
	CDetectionModel *m_ModelCL;				//model for city/landscape classification

	HRESULT GetCategory(const float *	pf1, 
						const int		iLength1,
						const float *	pf2,
						const int		iLength2,
						int *			piClass, 
						float *			pfConfidence, 
						CDetectionModel *pModel);
	void SetBitmapInfoHeader(const int iWidth, const int iHeight, BITMAPINFOHEADER *pbmi);
public:

	//
	//constructor and destructor
	//
	CImageDetector();		
	~CImageDetector();

	//
	//General functions 
	//
	HRESULT LoadOrientationModel( HMODULE hModule, LPCTSTR lpResType, LPCTSTR lpResCMEDH);		//Resource names for CM_EDH model
	HRESULT LoadIndoorOutdoorModel( HMODULE hModule, LPCTSTR lpResType, LPCTSTR lpResCMEDH);		//Resource names for CM_EDH model
	HRESULT LoadCityLandscapeModel( HMODULE hModule, LPCTSTR lpResType, LPCTSTR lpResCMEDH);	//Resource name for CM_EDH model

	HRESULT DetectImage(const BYTE *pbImage,	//pointer to 24 bits image buffer, 32bits aligned
						const int	iWidth,		//Image width
						const int	iHeight,	//Image height
						const BOOL bTopDown,	//TRUE, the scan lines is from top down; FALSE, from bottom up
						const int	iStride,	//the length in byte for each scan line, 0 for dword align as BITMAP
						const BOOL bIgnore180,	//Ignore 180 rotation, it is for digital camera mode
						LPDETECTRESULT pResult,			//Detection result
						LPDETECTEDGEHIST pEdgeHist);			//Edge entropy result 

	HRESULT DetectImage(const BITMAPINFOHEADER *pbmi,	//the bitmap info header, 24 bits image only
						const BYTE *pbImage,			//image buffer
						const BOOL bTopDown,			//TRUE, the scan lines is from top down; FALSE, from bottom up
						const int	iStride,			//the length in byte for each scan line, 0 for dword align as BITMAP
						const BOOL bIgnore180,			//Ignore 180 rotation, it is for digital camera mode
						LPDETECTRESULT pResult,			//Detection result
						LPDETECTEDGEHIST pEdgeHist);			//Edge entropy result 
	//
	//Advanced functions
	//
	HRESULT LoadOrientationModel( const LPVOID lpCMEDH,		//pointer to the buffer for CMEDH model data
								  const DWORD dwBufSizeCMEDH);//buffer size

	HRESULT LoadIndoorOutdoorModel( const LPVOID lpCMEDH,		//pointer to the buffer for CMEDH model data
									const DWORD dwBufSizeCMEDH);//buffer size

	HRESULT LoadCityLandscapeModel( const LPVOID lpCMEDH,		//pointer to the buffer for CMEDH model data
									const DWORD dwBufSizeCMEDH);//buffer size

	HRESULT ReadOrientationModel(const char* lpFileNameCMEDH);
	HRESULT ReadIndoorOutdoorModel(const char* lpFileNameCMEDH);
	HRESULT ReadCityLandscapeModel(const char* lpFileNameCMEDH);

	void RejectThresholdOrientation(const float t);
	void RejectThresholdIO(const float t);
	void RejectThresholdCL(const float t);

	void FreeModels();

	HRESULT ExtractFeatureCM(float *pCM, int *pCount, const BITMAPINFOHEADER *pbmi, const BYTE *pbImage, const BOOL bTopDown, const int iStride);
	HRESULT ExtractFeatureEDH(float *pEDH, int *pCount, const BITMAPINFOHEADER *pbmi, const BYTE *pbImage, const BOOL bTopDown, const int iStride);

	HRESULT ExtractFeatureCM(float *pCM, int *pCount, const BYTE *pbImage, const int iWidth, const int iHeight, const BOOL bTopDown,const int iStride);
	HRESULT ExtractFeatureEDH(float *pEDH, int *pCount, const BYTE *pbImage, const int iWidth, const int iHeight, const BOOL bTopDown,const int iStride);

	HRESULT GetOrientation( IN const float *pCM, 
							IN const float *pEDH, 
							OUT int *piOrientation, 
							OUT float *pfConfidence, 
							OUT LPDETECTEDGEHIST pEdgeHist,
							IN const BOOL bIgnore180 = FALSE) ; 
	
	HRESULT GetCategoryIO(	const float*	pf1,
							const int		iLength1,
							const float*	pf2,
							const int		iLength2,
							int *			piClass,
							float *			pfConfidence);

	HRESULT GetCategoryCL(	const float *	pCM, 
							const int		iLength1,
							const float *	pEDH, 
							const int		iLength2,
							int *			piClass, 
							float *			pfConfidence);
	HRESULT FinalDecision( LPDETECTRESULT lpResult);
};