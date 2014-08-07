// Global.h: interface for the CGlobal class.
//
//////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;

#define DEL_POINTER(x) if(NULL!=x) {delete x; x=NULL;}
#define DEL_ARRAY(x) if(NULL!=x) {delete x; x=NULL;}
#define DEL_FILE(x) if(NULL!=x) {fclose(x); x=NULL;}
#define DEL_IPLIMG(x) if(NULL!=x) {cvReleaseImage(&x); x=NULL;}

#ifndef FRAME_INTERVAL
#define FRAME_INTERVAL 5
#endif

#ifndef MAX_PATH
#define MAX_PATH 260
#endif

enum SmoothMethod {GAUSSIAN_FILTER, MEDIAN_FILTER};
enum FrameMotion {STA, PAN, TIT, ZOM, ROT, OBJ};

struct _SubShotInfo
{
	int iShotID;
	int iSubShotID;
	int iIntentID;
	long lBgnNo;
	long lEndNo;
	int iCameraMotion;
};

struct _ShotInfo 
{
	int iShotID;
	long lBgnNo;
	long lEndNo;
	std::vector<_SubShotInfo> vecSubShot;
};

struct _VideoInfo
{
	string szName;
	int iWidth;
	int iHeight;
	double dRate;
	double dDuration;
	long lFrameNum;
	long sampledFrameNum;
	long lShotNum;
	long lSubShotNum;
	std::vector<_ShotInfo> vecShot;
};


struct _CameraMotion {
	long frameID;
	float pan;
	float tit;
	float zom;
	float rot;
	float hyp;
	float err;
}; 

/************************************************************************/
/*	(u, v) is the corresponding optical flow in point (x, y)
/*	x' = x + u
/*	y' = y + v
/************************************************************************/

/************************************************************************/
/*	u = a1 + a2 * x + a3 * y
/*	v = a4 + a5 * x + a6 * y
/************************************************************************/
struct _AffineMotion {
	float a1 ;
	float a2 ;
	float a3 ;
	float a4 ;
	float a5 ;
	float a6 ;
	float Je ;		// motion error
};

/************************************************************************/
/*	u = a1 + a2 * x + a3 * y + a7 * x * x + a8 * x * y
/*	v = a4 + a5 * x + a6 * y + a7 * x * y + a8 * y * y
/************************************************************************/
struct _PerspectiveMotion {
	float a1 ;
	float a2 ;
	float a3 ;
	float a4 ;
	float a5 ;
	float a6 ;
	float a7 ;
	float a8 ;
	float Je ;		// motion error
};


static char *g_banner_start="\n\
=========================================\n\
= Video Unsuitability Score Calculation =\n\
=========================================\n\n";

static char *g_banner_end="\n\
=========================================\n\
= (C) COPYRIGHT, USTC & MSRA,  Tao Mei =\n\
=========================================\n\n";
