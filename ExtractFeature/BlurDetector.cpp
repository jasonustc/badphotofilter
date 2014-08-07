// BlurDetector.cpp: implementation of the CBlurDetector class.
//
//////////////////////////////////////////////////////////////////////

#include "BlurDetector.h"
#include <vector>
#include <math.h>

//#include <gdiplus.h>

//using namespace Gdiplus;


CBlurDetector::CBlurDetector()
{
//    m_gdiplusToken = NULL;
//    GdiplusStartupInput gdiplusStartupInput;
//    GdiplusStartup(&m_gdiplusToken, &gdiplusStartupInput, NULL);

	m_fBlurRatio = 0.0f;
	m_fSharpRatio = 0.0f;
    m_flSharpRatio = 0.5f; // Parameter: if sharp edge points are more than 5%, no blur
    m_flThreshold = 25.0f;  // Threshold to determine edge point, i.e. min edge strength
}

CBlurDetector::~CBlurDetector()
{
//    if (m_gdiplusToken)
//    {
//       GdiplusShutdown(m_gdiplusToken);
//    }
}


void CBlurDetector::DownSampleMatrix( CMatrix<float>& Img, 
									  int nWidth, 
									  int nHeight, 
									  int nScale, 
									  CMatrix<float>& Down)
{
    nWidth /= nScale;
    nHeight /= nScale;

	for(int i = 0; i < nHeight; i ++)
	{
		for(int j = 0; j < nWidth; j++)
		{
			float flMax = 0;
            for (int m = 0; m < nScale; m ++)
            {
                for (int n = 0; n < nScale; n ++)
                {
                    if (flMax < Img[i * nScale + m][j * nScale + n])
                    {
                        flMax = Img[i * nScale + m][j * nScale + n];
                    }
                }
            }

            Down[i][j] = flMax;//max-pooling.
        }
    }
}

void CBlurDetector::HWTEdgeDetect( CMatrix<float>& Img, 
								   int nWidth, 
								   int nHeight, 
								   CMatrix<float>& Map)
{
    std::vector<float> coeffs;
    for (int i = 0; i < nHeight / 2; i ++)
    {
        for (int j = 0; j < nWidth / 2; j ++)
        {
            float flLH = (Img[2 * i][2 * j + 1] - Img[2 * i][2 * j] 
                        + Img[2 * i + 1][2 * j + 1] - Img[2 * i + 1][2 * j]) / 4;

            float flHL = (Img[2 * i + 1][2 * j] + Img[2 * i + 1][2 * j + 1]
                        - Img[2 * i][2 * j] - Img[2 * i][2 * j + 1]) / 4;

            float flHH = (Img[2 * i + 1][2 * j + 1] + Img[2 * i][2 * j]
                        - Img[2 * i][2 * j + 1] - Img[2 * i + 1][2 * j]) / 4;

            Img[i][j] = (Img[2 * i][2 * j] + Img[2 * i][2 * j + 1]
                        + Img[2 * i + 1][2 * j] + Img[2 * i + 1][2 * j + 1]) / 4;//down sample.

            Map[i][j] = (float)sqrt(flLH * flLH + flHL * flHL + flHH * flHH);//edge map.
        }
    }
}

bool CBlurDetector::Detect( CMatrix<float>& Img, 
						    int nWidth, 
							int nHeight, 
							float& flBlur,
							float& flSharp)
{
    CMatrix<float> Emap(nHeight / 2, nWidth / 2);
    CMatrix<float> Emap1(nHeight / 8, nWidth / 8);
    CMatrix<float> Emap2(nHeight / 8, nWidth / 8);
    CMatrix<float> Emap3(nHeight / 8, nWidth / 8);

    if (!Emap || !Emap1 || !Emap2 || !Emap3)
    {
        return false;
    }

    HWTEdgeDetect(Img, nWidth, nHeight, Emap);
    DownSampleMatrix(Emap, nWidth / 2, nHeight / 2, 4, Emap1);//downsample the edgemap.

    HWTEdgeDetect(Img, nWidth / 2, nHeight / 2, Emap);
    DownSampleMatrix(Emap, nWidth / 4, nHeight / 4, 2, Emap2);//downsample the edgemap.

    HWTEdgeDetect(Img, nWidth / 4, nHeight / 4, Emap3);

    int nEdge = 0, nSharpEdge = 0, nBlurEdge = 0;
    for (int i = 0; i < nHeight / 8; i ++)
    {
        //for (int j = 0; j < nHeight / 8; j ++)
		for (int j = 0; j < nWidth / 8; j ++)
        {
            if (Emap1[i][j] >= m_flThreshold || Emap2[i][j] >= m_flThreshold || Emap3[i][j] >= m_flThreshold)
            {
                nEdge ++;
                if (Emap1[i][j] >= Emap2[i][j] && Emap2[i][j] >= Emap3[i][j])
                {
                    nSharpEdge ++;
                }
                else if (Emap1[i][j] < m_flThreshold)
                {
                    nBlurEdge ++;
                }
            }
        }
    }

    if (!nEdge)
    {
        flBlur = 1.0f;
		flSharp = 0;
    }
    else if (nSharpEdge >= (int)(nEdge * m_flSharpRatio))
    {
        flBlur = 0;
		flSharp = 1.0f;
    }
    else
    {
        flBlur = (float)nBlurEdge / nEdge;//blur extent.
	    flSharp = (float)nSharpEdge / nEdge;
    }

	m_fBlurRatio = (float)nBlurEdge / nEdge;//the ratio of bluredges.
	m_fSharpRatio = (float)nSharpEdge / nEdge;//the ratio of sharp edges.

    return true;
}

bool CBlurDetector::Detect2( CMatrix<float>& Img, 
						    int nWidth, 
							int nHeight, 
							float& flBlur,
							float& flSharp)
{
    CMatrix<float> Emap(nHeight / 2, nWidth / 2);
    CMatrix<float> Emap1(nHeight / 8, nWidth / 8);
    CMatrix<float> Emap2(nHeight / 8, nWidth / 8);
    CMatrix<float> Emap3(nHeight / 8, nWidth / 8);

    if (!Emap || !Emap1 || !Emap2 || !Emap3)
    {
        return false;
    }

    HWTEdgeDetect(Img, nWidth, nHeight, Emap);
    DownSampleMatrix(Emap, nWidth / 2, nHeight / 2, 4, Emap1);//downsample the edgemap.

    HWTEdgeDetect(Img, nWidth / 2, nHeight / 2, Emap);
    DownSampleMatrix(Emap, nWidth / 4, nHeight / 4, 2, Emap2);//downsample the edgemap.

    HWTEdgeDetect(Img, nWidth / 4, nHeight / 4, Emap3);

    int nEdge = 0, nDa = 0, nRg = 0,nBrg=0;
    for (int i = 0; i < nHeight / 8; i ++)
    {
        //for (int j = 0; j < nHeight / 8; j ++)
		for (int j = 0; j < nWidth / 8; j ++)
        {
            if (Emap1[i][j] >= m_flThreshold || Emap2[i][j] >= m_flThreshold || Emap3[i][j] >= m_flThreshold)
            {
                nEdge ++;
                if (Emap1[i][j] >= Emap2[i][j] && Emap2[i][j] >= Emap3[i][j])
                {
                    nDa++;
                }
				else if(Emap1[i][j]<=Emap2[i][j]&& Emap2[i][j]<=Emap3[i][j])
				{
                    nRg++;
				}
				else if(Emap2[i][j]>=Emap1[i][j] && Emap2[i][j]>= Emap3[i][j])
				{
                    nRg++;
				}
                else if (Emap1[i][j] < m_flThreshold)
                {
                    nBrg ++;
                }
            }
        }
    }

    if (!nEdge || !nRg)
    {
        flBlur = 1.0f;
		flSharp = 0;
    }
    else if (nBrg >= (int)(nEdge * m_flSharpRatio))
    {
        flBlur = 0;
		flSharp = 1.0f;
    }
    else
    {
        flBlur = (float)nBrg / nRg;//blur extent.
	    flSharp = (float)nDa / nEdge;
    }

	m_fBlurRatio = (float)nBrg / nRg;//the ratio of bluredges.
	m_fSharpRatio = (float)nDa / nEdge;//the ratio of sharp edges.

    return true;
}

HRESULT CBlurDetector::Detect(float* pflImg, int nWidth, int nHeight, float& flBlur,float &flSharp)
{
    flBlur = 0,flSharp=0;

    if (!pflImg || nWidth < 8 || nHeight < 8)
    {
        return E_INVALIDARG;
    }

    CMatrix<float> Img(nHeight, nWidth);
    if (!Img)
    {
        return E_OUTOFMEMORY;
    }

    for (int i = 0; i < nHeight; i ++)
    {
        CopyMemory(Img[i], pflImg + i * nWidth, sizeof(float) * nWidth);
    }

    if (Detect(Img, nWidth & -8, nHeight & -8, flBlur,flSharp) == true)
    {
        return S_OK;
    }
    return E_FAIL;
}

HRESULT CBlurDetector::Detect2(float* pflImg, int nWidth, int nHeight, float& flBlur,float &flSharp)
{
    flBlur = 0,flSharp=0;

    if (!pflImg || nWidth < 8 || nHeight < 8)
    {
        return E_INVALIDARG;
    }

    CMatrix<float> Img(nHeight, nWidth);
    if (!Img)
    {
        return E_OUTOFMEMORY;
    }

    for (int i = 0; i < nHeight; i ++)
    {
        CopyMemory(Img[i], pflImg + i * nWidth, sizeof(float) * nWidth);
    }

    if (Detect2(Img, nWidth & -8, nHeight & -8, flBlur,flSharp) == true)
    {
        return S_OK;
    }
    return E_FAIL;
}