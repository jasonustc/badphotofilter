// BlurDetector.h: interface for the CBlurDetector class.
//
//////////////////////////////////////////////////////////////////////

#pragma once
#include <Windows.h>
#include <opencv2\opencv.hpp>

template<class T>
class CMatrix
{
public:
    CMatrix(int m, int n)
    {
        m_pData = NULL;
        m_nRow = m_nCol = 0;
        if (m > 0 && n > 0)
        {
            m_pData = new T[m * n];
            if (m_pData != NULL)
            {
                ZeroMemory(m_pData, sizeof(T) * m * n);
                m_nRow = m;
                m_nCol = n;
            }
        }
    }
    ~CMatrix()
    {
        delete m_pData;
        m_pData = NULL;
        m_nRow = m_nCol = 0;
    }
 
public:
	int GetRows() 
	{
		return m_nRow;
	}
	int GetCols()
	{
		return m_nCol;
	}
    bool operator!()
    {
        return !m_pData;				 // Check whether the data is valid, call this before other operation
    }
    T* operator[](int i)
    {
        return m_pData + i * m_nCol;    // Get the pointer of a row
    }
	
protected:
    T*  m_pData;
    int m_nRow;
    int m_nCol;
};


class CBlurDetector  
{
public:
	CBlurDetector();
    ~CBlurDetector();

public:
	float m_fBlurRatio;
	float m_fSharpRatio;

    HRESULT Detect(BYTE* pfImg, int nWidth, int nHeight, float& flBlur);
    HRESULT Detect(float* pflImg, int nWidth, int nHeight, float& flBlur,float& flSharp);
    HRESULT Detect2(float* pflImg, int nWidth, int nHeight, float& flBlur,float &flSharp);
//    HRESULT Detect(TCHAR* pszFileName, float& flBlur);

protected:
    void DownSampleMatrix(CMatrix<float>& Img, int nWidth, int nHeight, int nScale, CMatrix<float>& Down);
    void HWTEdgeDetect(CMatrix<float>& Img, int nWidth, int nHeight, CMatrix<float>& Map);
    bool Detect(CMatrix<float>& Img, int nWidth, int nHeight, float& flBlur,float &flSharp);
    bool Detect2(CMatrix<float>& Img, int nWidth, int nHeight, float& flBlur,float &flSharp);

protected:
    float m_flSharpRatio;   // Min sharp edge point percentage
    float m_flThreshold;    // Min edge strength

private:
//    ULONG_PTR m_gdiplusToken;
};

