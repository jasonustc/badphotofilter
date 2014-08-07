#include "myFeatures.h"
#define cvQueryHistValue_1D( hist, idx0 ) \
    cvGetReal1D( (hist)->bins, (idx0) )
#define BLOCKS_PER_LINE 6
#define FRE_HIST_BIN 10

localFeatExtractor::localFeatExtractor(Mat &srcImg)
{
	if (0 == srcImg.rows || 0 == srcImg.cols){
		this->imgData = NULL;
	}
	else if (srcImg.channels() == 1){
		cvtColor(srcImg, this->imgData, CV_GRAY2BGR);
	}
	else{
		this->imgData = srcImg;
	}
	this->localFeat.clear();
}

float localFeatExtractor::CalHighSpec(Mat &img) {
	Mat padded, complexI, magCal;
	float dof = 0;
	int m = getOptimalDFTSize(img.rows);
	int n = getOptimalDFTSize(img.cols);
    //add the border for convolutiion kernel.
	copyMakeBorder(img, padded, 0,m-img.rows,0 ,n - img.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	merge(planes, 2, complexI);//Add to the expanded another plane with zeros
	dft(complexI, complexI);//the result may fit in the source matrix
    //compute the magnitude and switch to logarithmic scale
    //=> log(1+sqrt(Re(DFT(I))^2+Im(DFT(I)^2))
	split(complexI, planes);//planes[0]=Re(DFT(I),planes[1]=Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);//planes[0]=magnitude
	magCal = planes[0];
	magCal += Scalar::all(1);//switch to logarithmic scale
	log(magCal, magCal);
    //crop the spectrum, if it has an odd number of rows or columns
	magCal = magCal(Rect(0, 0, magCal.cols & -2, magCal.rows & -2));
	for (int i = 0; i < magCal.rows; i++)
	{
		float* pRow = magCal.ptr<float>(i);
		for (int j = 0; j < magCal.cols; j++)
		{
			float tempData = pRow[j];
			if (tempData > 5){//larger than 8 is considered as high spec.
				dof += 1;
			}
		}
	}
	dof /= (magCal.rows*magCal.cols);
	return dof;
}

float localFeatExtractor::Sharpness(Mat &srcImg)
{
	Mat I;
	float sharpness;
	if (srcImg.channels() == 3)
	{
		cvtColor(srcImg, I, CV_BGR2GRAY);
	}
	sharpness = CalHighSpec(I);
	return sharpness;
}

float  localFeatExtractor::BlurDiff(Mat &srcImg)
{
	Mat postImg;
	float diff;
	GaussianBlur(srcImg, postImg, Size(25, 25), 0, 0);
	Mat graySource,grayBlur;
	cvtColor(srcImg, graySource, CV_BGR2GRAY);
	cvtColor(postImg, grayBlur, CV_BGR2GRAY);
	Mat diffImg;
	diffImg = graySource-grayBlur;
	CvMat img = diffImg;
	cvMul(&img, &img, &img,1.0);
	CvScalar mse;
	diffImg = Mat::Mat(&img, true);
	mse=sum(diffImg)/(float(diffImg.cols*diffImg.rows));
	diff = mse.val[0];
	return diff;
}

float  localFeatExtractor::pixContrast(Mat &srcImg)
{
	vector<float> pixCount(256, 0);
	float pixDiff;
	for (int i = 0; i < srcImg.rows; i++)
	{
		for (int j = 0; j < srcImg.cols; j++)
		{
			int pixb = srcImg.at<Vec3b>(i, j)[0];
			pixCount[pixb]++;
			int pixg = srcImg.at<Vec3b>(i, j)[1];
			pixCount[pixg]++;
            int pixr = srcImg.at<Vec3b>(i, j)[2];
			pixCount[pixr]++;
		}
	}
	CvScalar sumPix = sum(pixCount);
	float sumCount=float(sumPix.val[0]);
	float threHis = 0.01*sumCount;
	int hisi = 0, hisj = 255;
	float txhis1 = 0, txhis2 = 0;
	for (int i = 0; i < 256; i++)
	{
		txhis1 += pixCount[i];
		txhis2 += pixCount[255 - i];
		if (txhis1 < threHis){
			hisi++;
		}
		if (txhis2 < threHis){
			hisj--;
		}
		if (txhis1 >= threHis&&txhis2 >= threHis){
			break;
		}
	}
	hisj = hisi >= hisj ? hisi : hisj;
	pixDiff = float(hisj - hisi);//pixel gap of high fraction pixels to low fraction pixels.
	pixDiff /= 256;
	return pixDiff;
}

//simplicity of the photo.
float localFeatExtractor::pixSimp(Mat &srcImg)
{
    if(srcImg.channels()==1){
        cvtColor(srcImg,srcImg,CV_GRAY2BGR);
	}
    vector<int> hist(4096,0);
    for(int i=0;i<srcImg.rows;i++){
        for(int j=0;j<srcImg.cols;j++){
            int binB=srcImg.at<Vec3b>(i,j)[0]/16;
            int binG=srcImg.at<Vec3b>(i,j)[1]/16;
            int binR=srcImg.at<Vec3b>(i,j)[2]/16;
            int binIdx=binB+binG*16+binR*16*16;
            hist[binIdx]++;
		}
	}
    int maxCount=*max_element(hist.begin(),hist.end());
    float binThre=maxCount*0.01;
    int S=0;
    for(unsigned int i=0;i<hist.size();i++){
        if(float(hist[i])>binThre){
            S++;
		}
	}
    float simplicity=float(S)/4096;
    return simplicity;
}

int localFeatExtractor::CalcDyrAndInc(Mat &srcImg,float &dyRan,float &inCon)
{
    vector<float> pixCount(256,0);
    float dynRan,IntenCon;
    for(int i=0;i<srcImg.rows;i++){
        for(int j=0;j<srcImg.cols;j++){
            int pixb=srcImg.at<Vec3b>(i,j)[0];
            pixCount[pixb]++;
            int pixg=srcImg.at<Vec3b>(i,j)[1];
			pixCount[pixg]++;
            int pixr=srcImg.at<Vec3b>(i,j)[2];
            pixCount[pixr]++;
		}
	}
    //normalization
    CvScalar sumPix=sum(pixCount);
    float sumCount=float(sumPix.val[0]);
    for(int i=0;i<256;i++){
        pixCount[i] /= sumCount;
	}
    int threL=0,threM=0,threH=255;
    float cumHisL=0,cumHisM=0,cumHisH=0;
    for(int i=0;i<256;i++){
        cumHisL += pixCount[i];
        cumHisM += pixCount[i];
        cumHisH += pixCount[255-i];
        if(cumHisL<0.05){
            threL++;
		}
        if(cumHisH<0.05){
            threH--;
		}
        if(cumHisM<0.5){
            threM++;
		}
        if(cumHisL>=0.05&&cumHisH>=0.05 && cumHisM>=0.5){
            break;
		}
	}
    threH = threH >= threL ? threH: threL;
    dyRan=1-float(threH-threL)/255;
    float sumH=0,sumL=0;
    for(int i=threL;i<=threM;i++){
        sumL += i*pixCount[i]/0.45;
	}
    for(int i=threM;i<=threH;i++){
        sumH += i*pixCount[i]/0.45;
	}
    sumH = sumH >= sumL ? sumH : sumL;
    inCon=1-(sumH-sumL)/255;
    return 0;
}

int localFeatExtractor::computeLocalFeat()
{
	this->localFeat.clear();
	Mat localImg;
	if (this->imgData.data==NULL||this->imgData.rows==0||this->imgData.cols==0){
		return EMPTY_IMAGE_DATA;
	}
	int blockWidth = this->imgData.cols / BLOCKS_PER_LINE;
	int blockHeight =this->imgData.rows / BLOCKS_PER_LINE;
	if (blockWidth <=2  || blockHeight <=2)
	{
		return TOO_SMALL_IMAGE_SIZE;
	}
	vector<float> blurs, sharps, complexes, flSharpnesses,blurChanges,contrasts;
	IplImage smallImg;
	float flBlur, flSharp,flSharpness,blurChange,contrast,flComplexity;
	localImg = this->imgData(Range(2 * blockHeight, 4 * blockHeight),
		Range(2 * blockWidth, 4 * blockWidth));
	smallImg = localImg;
    int errMeg;
	errMeg= CalcBlur(&smallImg, flBlur, flSharp);
    if(errMeg){
        return errMeg;
	}
	flSharpness= Sharpness(localImg);
	blurChange = BlurDiff(localImg);
	contrast = pixContrast(localImg);
	flComplexity = pixSimp(localImg);
	complexes.push_back(flComplexity);
	contrasts.push_back(contrast);
	blurChanges.push_back(blurChange);
	blurs.push_back(flBlur);
	sharps.push_back(flSharp);
	flSharpnesses.push_back(flSharpness);

	localImg = this->imgData(Range(blockHeight, 3 * blockHeight),
		Range(blockWidth, 3 * blockWidth));
	smallImg = localImg;
	errMeg=CalcBlur(&smallImg,flBlur,flSharp);
	if (errMeg){
		return errMeg;
	}
	flSharpness= Sharpness(localImg);
	blurChange = BlurDiff(localImg);
	contrast = pixContrast(localImg);
	flComplexity = pixSimp(localImg);
	complexes.push_back(flComplexity);
	contrasts.push_back(contrast);
	blurChanges.push_back(blurChange);
	blurs.push_back(flBlur);
	sharps.push_back(flSharp);
	flSharpnesses.push_back(flSharpness);
    
	localImg = this->imgData(Range(3 * blockHeight, 5 * blockHeight),
		Range(blockWidth, 3 * blockWidth));
	smallImg = localImg;
	errMeg=CalcBlur(&smallImg,flBlur,flSharp);
	if (errMeg){
		return errMeg;
	}
	flSharpness= Sharpness(localImg);
	blurChange = BlurDiff(localImg);
	contrast = pixContrast(localImg);
	flComplexity = pixSimp(localImg);
	complexes.push_back(flComplexity);
	contrasts.push_back(contrast);
	blurChanges.push_back(blurChange);
	blurs.push_back(flBlur);
	sharps.push_back(flSharp);
	flSharpnesses.push_back(flSharpness);

	localImg = this->imgData(Range(blockHeight, 3 * blockHeight),
		Range(3 * blockWidth, 5 * blockWidth));
	smallImg = localImg;
	errMeg=CalcBlur(&smallImg,flBlur,flSharp);
	if (errMeg){
		return errMeg;
	}
	flSharpness= Sharpness(localImg);
	blurChange = BlurDiff(localImg);
	contrast = pixContrast(localImg);
	flComplexity = pixSimp(localImg);
	complexes.push_back(flComplexity);
	contrasts.push_back(contrast);
	blurChanges.push_back(blurChange);
	blurs.push_back(flBlur);
	sharps.push_back(flSharp);
	flSharpnesses.push_back(flSharpness);

	localImg = this->imgData(Range(3 * blockHeight, 5 * blockHeight),
		Range(3 * blockWidth, 5 * blockWidth));
	smallImg = localImg;
	errMeg=CalcBlur(&smallImg,flBlur,flSharp);
	if (errMeg){
		return errMeg;
	}
	flSharpness= Sharpness(localImg);
	blurChange = BlurDiff(localImg);
	contrast = pixContrast(localImg);
	flComplexity = pixSimp(localImg);
	complexes.push_back(flComplexity);
	contrasts.push_back(contrast);
	blurChanges.push_back(blurChange);
	blurs.push_back(flBlur);
	sharps.push_back(flSharp);
	flSharpnesses.push_back(flSharpness);

	//take the best block into account
	int bestIndex = int(max_element(contrasts.begin(), 
		contrasts.end()) - contrasts.begin());
	int badIndex = int(min_element(contrasts.begin(), 
		contrasts.end()) - contrasts.begin());
    //local feature of best block.
	this->localFeat.push_back(blurs[bestIndex]);
	this->localFeat.push_back(sharps[bestIndex]);
	this->localFeat.push_back(flSharpnesses[bestIndex]);
	this->localFeat.push_back(contrasts[bestIndex]);
	this->localFeat.push_back(blurChanges[bestIndex]);
	this->localFeat.push_back(complexes[bestIndex]);
    //local feature of the worst block.
	this->localFeat.push_back(blurs[badIndex]);
	this->localFeat.push_back(sharps[badIndex]);
	this->localFeat.push_back(flSharpnesses[badIndex]);
	this->localFeat.push_back(contrasts[badIndex]);
	this->localFeat.push_back(blurChanges[badIndex]);
	this->localFeat.push_back(complexes[badIndex]);
	return FILTER_OK;
}

int localFeatExtractor::computeLocalFeat2()
{
	this->localFeat.clear();
	int cx = this->imgData.cols/2;
	int cy = this->imgData.rows/2;      
	Mat SegTL(this->imgData,Rect(0,0,cx,cy));
	Mat SegTR(this->imgData,Rect(cx,0,cx,cy));
	Mat SegBL(this->imgData,Rect(0,cy,cx,cy));
	Mat SegBR(this->imgData,Rect(cx,cy,cx,cy)); 
	Mat SegCenter(this->imgData,Rect(cx/2,cy/2,cx,cy));
	vector<float> blurs, sharps, complexes, flSharpnesses,blurChanges,contrasts;
	IplImage smallImg;
	float flBlur, flSharp,flSharpness,blurChange,contrast,flComplexity;
    //top left block
	smallImg = SegTL;
    int errMeg;
	errMeg= CalcBlur(&smallImg, flBlur, flSharp);
    if(errMeg){
        return errMeg;
	}
	flSharpness= Sharpness(SegTL);
	blurChange = BlurDiff(SegTL);
	contrast = pixContrast(SegTL);
	flComplexity = pixSimp(SegTL);
	complexes.push_back(flComplexity);
	contrasts.push_back(contrast);
	blurChanges.push_back(blurChange);
	blurs.push_back(flBlur);
	sharps.push_back(flSharp);
	flSharpnesses.push_back(flSharpness);

    //top right block
	smallImg = SegTR;
	errMeg=CalcBlur(&smallImg,flBlur,flSharp);
	if (errMeg){
		return errMeg;
	}
	flSharpness= Sharpness(SegTR);
	blurChange = BlurDiff(SegTR);
	contrast = pixContrast(SegTR);
	flComplexity = pixSimp(SegTR);
	complexes.push_back(flComplexity);
	contrasts.push_back(contrast);
	blurChanges.push_back(blurChange);
	blurs.push_back(flBlur);
	sharps.push_back(flSharp);
	flSharpnesses.push_back(flSharpness);
    
    //bottom left block
	smallImg = SegBL;
	errMeg=CalcBlur(&smallImg,flBlur,flSharp);
	if (errMeg){
		return errMeg;
	}
	flSharpness= Sharpness(SegBL);
	blurChange = BlurDiff(SegBL);
	contrast = pixContrast(SegBL);
	flComplexity = pixSimp(SegBL);
	complexes.push_back(flComplexity);
	contrasts.push_back(contrast);
	blurChanges.push_back(blurChange);
	blurs.push_back(flBlur);
	sharps.push_back(flSharp);
	flSharpnesses.push_back(flSharpness);

    //bottom right block
	smallImg = SegBR;
	errMeg=CalcBlur(&smallImg,flBlur,flSharp);
	if (errMeg){
		return errMeg;
	}
	flSharpness= Sharpness(SegBR);
	blurChange = BlurDiff(SegBR);
	contrast = pixContrast(SegBR);
	flComplexity = pixSimp(SegBR);
	complexes.push_back(flComplexity);
	contrasts.push_back(contrast);
	blurChanges.push_back(blurChange);
	blurs.push_back(flBlur);
	sharps.push_back(flSharp);
	flSharpnesses.push_back(flSharpness);

    //centre block
	smallImg =  SegCenter;
	errMeg=CalcBlur(&smallImg,flBlur,flSharp);
	if (errMeg){
		return errMeg;
	}
	flSharpness= Sharpness(SegCenter);
	blurChange = BlurDiff(SegCenter);
	contrast = pixContrast(SegCenter);
	flComplexity = pixSimp(SegCenter);
	complexes.push_back(flComplexity);
	contrasts.push_back(contrast);
	blurChanges.push_back(blurChange);
	blurs.push_back(flBlur);
	sharps.push_back(flSharp);
	flSharpnesses.push_back(flSharpness);

	//take the best block into account
	int bestIndex = int(max_element(contrasts.begin(), 
		contrasts.end()) - contrasts.begin());
	int badIndex = int(min_element(contrasts.begin(), 
		contrasts.end()) - contrasts.begin());
    //local feature of best block.
	this->localFeat.push_back(blurs[bestIndex]);
	this->localFeat.push_back(sharps[bestIndex]);
	this->localFeat.push_back(flSharpnesses[bestIndex]);
	this->localFeat.push_back(contrasts[bestIndex]);
	this->localFeat.push_back(blurChanges[bestIndex]);
	this->localFeat.push_back(complexes[bestIndex]);
    //local feature of the worst block.
	this->localFeat.push_back(blurs[badIndex]);
	this->localFeat.push_back(sharps[badIndex]);
	this->localFeat.push_back(flSharpnesses[badIndex]);
	this->localFeat.push_back(contrasts[badIndex]);
	this->localFeat.push_back(blurChanges[badIndex]);
	this->localFeat.push_back(complexes[badIndex]);
	return FILTER_OK;
}
//global feature extractor
globalFeatExtractor::globalFeatExtractor(Mat &srcImg)
{
	if (0 == srcImg.rows || 0 == srcImg.cols){
		this->imgData = NULL;
	}
	else if (srcImg.channels() == 1){
		cvtColor(srcImg, this->imgData, CV_GRAY2BGR);
	}
	else{
		this->imgData = srcImg;
	}
	this->globalFeat.clear();
}

int globalFeatExtractor::computeGlobalFeat()
{
    //8 dims: blur,sharp,blur difference,
	//sharpness,dynamic range,intensity contrast,brightness,darkness.
	int errMsg = FILTER_OK;
	float flBlur = 0, flSharp = 0;
	errMsg=calcBlurSharp(flBlur, flSharp);
	this->globalFeat.push_back(flBlur);
	this->globalFeat.push_back(flSharp);
	this->globalFeat.push_back(calcBlurDiff());
	this->globalFeat.push_back(calcSharpness());
	//this->globalFeat.push_back(calcContrast());
    float dynRan,intenCon;
    CalcDyrAndInc(this->imgData,dynRan,intenCon);
    this->globalFeat.push_back(dynRan);
    this->globalFeat.push_back(intenCon);
	this->globalFeat.push_back(calcBrightness());
	this->globalFeat.push_back(calcDarkness());
    //1 dim: simplicity.
	this->globalFeat.push_back(calcSimp());
    //11 dim: histogram of HSV.
	vector<float> hsvFeat;
	calcHistHSV2(hsvFeat);
	for (unsigned int i = 0; i < hsvFeat.size(); i++)
	{
		this->globalFeat.push_back(hsvFeat[i]);
	}
    //2 dim: the depth of field feature
    float dofFeat1,dofFeat2;
    CalcDofFeat(dofFeat1,dofFeat2);
    this->globalFeat.push_back(dofFeat1);
    this->globalFeat.push_back(dofFeat2);
    //9 dim: sub block feat
    vector<float> subBlockFeat;
    calcSubblockInfo(subBlockFeat);
	for (unsigned int i = 0; i < subBlockFeat.size(); i++)
	{
		this->globalFeat.push_back(subBlockFeat[i]);
	}
	return FILTER_OK;
}

float globalFeatExtractor::calcHighSpec(Mat &img) {
	Mat padded, complexI, magCal;
	float dof = 0;
	int m = getOptimalDFTSize(img.rows);
	int n = getOptimalDFTSize(img.cols);
	//add the border for convolutiion kernel.
	copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	merge(planes, 2, complexI);//Add to the expanded another plane with zeros
	dft(complexI, complexI);//the result may fit in the source matrix
	//compute the magnitude and switch to logarithmic scale
	//=> log(1+sqrt(Re(DFT(I))^2+Im(DFT(I)^2))
	split(complexI, planes);//planes[0]=Re(DFT(I),planes[1]=Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);//planes[0]=magnitude
	magCal = planes[0];
	magCal += Scalar::all(1);//switch to logarithmic scale
	log(magCal, magCal);
	//crop the spectrum, if it has an odd number of rows or columns
	magCal = magCal(Rect(0, 0, magCal.cols & -2, magCal.rows & -2));
	for (int i = 0; i < magCal.rows; i++)
	{
		float* pRow = magCal.ptr<float>(i);
		for (int j = 0; j < magCal.cols; j++)
		{
			float tempData = pRow[j];
			if (tempData > 5){//larger than 8 is considered as high spec.
				dof += 1;
			}
		}
	}
	dof /= (magCal.rows*magCal.cols);
	return dof;
}

float globalFeatExtractor::calcSharpness()
{
	Mat I;
	float sharpness;
	if (this->imgData.channels() == 3)
	{
		cvtColor(this->imgData, I, CV_BGR2GRAY);
	}
	sharpness = calcHighSpec(I);
	return sharpness;
}

float  globalFeatExtractor::calcBlurDiff()
{
	Mat postImg;
	float diff;
	GaussianBlur(this->imgData, postImg, Size(25, 25), 0, 0);
	Mat graySource, grayBlur;
	cvtColor(this->imgData, graySource, CV_BGR2GRAY);
	cvtColor(postImg, grayBlur, CV_BGR2GRAY);
	Mat diffImg;
	diffImg = graySource - grayBlur;
	CvMat img = diffImg;
	cvMul(&img, &img, &img, 1.0);
	CvScalar mse;
	diffImg = Mat::Mat(&img, true);
	mse = sum(diffImg) / (float(diffImg.cols*diffImg.rows));
	diff = mse.val[0];
	return diff;
}

float  globalFeatExtractor::calcContrast()
{
	vector<float> pixCount(256, 0);
	float pixDiff;
	for (int i = 0; i < this->imgData.rows; i++)
	{
		for (int j = 0; j < this->imgData.cols; j++)
		{
			int pixb = this->imgData.at<Vec3b>(i, j)[0];
			pixCount[pixb]++;
			int pixg = this->imgData.at<Vec3b>(i, j)[1];
			pixCount[pixb]++;
			int pixr = this->imgData.at<Vec3b>(i, j)[2];
			pixCount[pixb]++;
		}
	}
	CvScalar sumPix = sum(pixCount);
	float sumCount = float(sumPix.val[0]);
	float threHis = 0.01*sumCount;
	int hisi = 0, hisj = 255;
	float txhis1 = 0, txhis2 = 0;
	for (int i = 0; i < 256; i++)
	{
		txhis1 += pixCount[i];
		txhis2 += pixCount[255 - i];
		if (txhis1 < threHis){
			hisi++;
		}
		if (txhis2 < threHis){
			hisj--;
		}
		if (txhis1 >= threHis&&txhis2 >= threHis){
			break;
		}
	}
	hisj = hisi >= hisj ? hisi : hisj;
	pixDiff = float(hisj - hisi);//pixel gap of high fraction pixels to low fraction pixels.
	pixDiff /= 256;
	return pixDiff;
}
//simplicity of the photo.
float globalFeatExtractor::calcSimp()
{
   if(this->imgData.channels()==1){
        cvtColor(this->imgData,this->imgData,CV_GRAY2BGR);
	}
    vector<int> hist(4096,0);
    for(int i=0;i<this->imgData.rows;i++){
        for(int j=0;j<this->imgData.cols;j++){
            int binB=this->imgData.at<Vec3b>(i,j)[0]/16;
            int binG=this->imgData.at<Vec3b>(i,j)[1]/16;
            int binR=this->imgData.at<Vec3b>(i,j)[2]/16;
            int binIdx=binB+binG*16+binR*16*16;
            hist[binIdx]++;
		}
	}
    int maxCount=*max_element(hist.begin(),hist.end());
    float binThre=maxCount*0.01;
    int S=0;
    for(unsigned int i=0;i<hist.size();i++){
        if(float(hist[i])>binThre){
            S++;
		}
	}
    float simplicity=float(S)/4096;
    return simplicity;
}

float globalFeatExtractor::calcBrightness()
{
	IplImage SrcImg = this->imgData;
	IplImage* pSrcImg = &SrcImg;
	float flBrightness = 0;

	if (pSrcImg == NULL)
	{
		printf("\nError: in CalcBrightness, input image error!\n");
		return EMPTY_IMAGE_DATA;
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
		cvSplit(pHSVImg, NULL, NULL, pGryImg, NULL);// here is the value of each pixel
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
	//flBrightness=float(iCnt)/(iWidth*iHeight);

	cvReleaseImage(&pGryImg);
	return flBrightness;
}

float globalFeatExtractor::calcDarkness()
{
	float flDarkness = 0;
	IplImage SrcImg = this->imgData;
	IplImage* pSrcImg = &SrcImg;
	if (pSrcImg == NULL)
	{
		printf("\nError: in CalcBrightness, input image error!\n");
		return EMPTY_IMAGE_DATA;
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

int globalFeatExtractor::calcBlurSharp(float &flBlur,float &flSharp)
{
	int errMsg = FILTER_OK;
	IplImage srcImg =this->imgData;
	errMsg=CalcBlur(&srcImg, flBlur, flSharp);
	return errMsg;
}

int globalFeatExtractor::calcSubblockInfo(vector<float> &subareaFeat)
{
    clock_t start,finish;
    double duration;
//    start=clock();
	int errMsg = 0;
	subareaFeat.resize(9, 0);
	Mat img, img2;
	this->imgData.copyTo(img);
	this->imgData.copyTo(img2);

	if (img.channels() == 1)
	{
		cvtColor(img, img, CV_GRAY2BGR);
		cvtColor(img2, img2, CV_GRAY2BGR);
	}

	cvtColor(img, img, CV_BGR2GRAY);
	GaussianBlur(img, img, Size(5, 5), 0, 0, BORDER_DEFAULT);

	Canny(img, img, 80, 200, 3);
	img.convertTo(img, CV_32FC1);
//    imwrite("edgeImage.jpg",img);

	int nl = img.rows;
	int nc = img.cols;

	float aldata = 0;
	vector<float> rowdata(nl, 0), coldata(nc, 0);
	vector<float*> datap(nl);

	//compute the border line
	for (int i = 0; i<nl; i++)
	{
		datap[i] = img.ptr<float>(i);

		for (int j = 0; j<nc; j++)
		{
			rowdata[i] += datap[i][j];
			aldata += datap[i][j];
		}
	}
	for (int j = 0; j<nc; j++)
	{
		for (int i = 0; i<nl; i++)
		{
			coldata[j] += datap[i][j];
		}
	}

	float thresdata = 0.03*aldata;
	int rowi = 0, rowj = nl - 1;
    //the box contrains 0.94 of all edge energy as subject area 
	float tx1 = 0, tx2 = 0;
	for (int i = 0; i<nl; i++)
	{
		tx1 += rowdata[i]; tx2 += rowdata[nl - 1 - i];
		if (tx1<thresdata)
		{
			rowi++;
		}
		if (tx2<thresdata)
		{
			rowj--;
		}
		if (tx1 >= thresdata&&tx2 >= thresdata)
		{
			break;
		}
	}
	if (rowi >= rowj)
	{
		rowj = rowi;
	}

    //the box contains 0.94 of all edge energy as subject area
	int coli = 0, colj = nc - 1;
	tx1 = 0; tx2 = 0;
	for (int i = 0; i<nc; i++)
	{
		tx1 += coldata[i]; tx2 += coldata[nc - 1 - i];
		if (tx1<thresdata)
		{
			coli++;
		}
		if (tx2<thresdata)
		{
			colj--;
		}
		if (tx1 >= thresdata&&tx2 >= thresdata)
		{
			break;
		}
	}
	if (coli >= colj)
	{
		colj = coli;
	}

	//fraction of the subject area.
	subareaFeat[0] = (float)(rowj - rowi)*(colj - coli) / (nl*nc);
//    finish=clock();
//    duration=(double)(finish-start)/CLOCKS_PER_SEC;
//    printf("Extract sub area time: %f\n",duration);
//    start=clock();
	//local features of the subject area.
	if (rowj - rowi <= 2 || colj - coli <= 2)
	{
		for (int i = 1; i < 9; i++)
		{
			subareaFeat[i] = 0;
		}
	}
	else
	{
		//average of the edge energy in subject area
		subareaFeat[1]= 0.94*aldata/((float)(rowj - rowi)*(colj - coli));
		float flBlur, flSharp;
		Mat localImg(img2, Rect(coli, rowi, colj - coli, rowj - rowi));
		IplImage subImg = localImg;
		errMsg=CalcBlur(&subImg, flBlur, flSharp);
		if (errMsg){
			return errMsg;
		}
		subareaFeat[2] = flBlur;
		subareaFeat[3] = flSharp;
	}
 //   finish=clock();
 //   duration=(double)(finish-start)/CLOCKS_PER_SEC;
 //   printf("Compute blur,sharp %f\n",duration);
 //   start=clock();
	//hue and value features.
	img2.convertTo(img2, CV_32FC3);
	img2 /= 255;
	cvtColor(img2, img2, CV_BGR2HSV);
	Mat edgeSeg(img2, Rect(coli, rowi, colj - coli, rowj - rowi));
	int nlx = edgeSeg.rows, ncx = edgeSeg.cols;
	double ligcon = 0, huecon = 0;
	if (nlx == 0 || ncx == 0){
		for (int i = 4; i <8 ; i++){
			subareaFeat[i] = 0;
		}
	}
	else{
		for (int i = 0; i < nlx; i++)
		{
			for (int j = 0; j < ncx; j++)
			{
                //value and hue distribution of the subject area
				subareaFeat[4] += edgeSeg.at<Vec3f>(i, j)[2];
				float pix = edgeSeg.at<Vec3f>(i, j)[0];

				if (pix>320 || pix < 45)
				{
					subareaFeat[5]++;
				}
				else if (pix < 140)
				{
					subareaFeat[6]++;
				}
				else if (pix < 255)
				{
					subareaFeat[7]++;
				}
				else
				{
					subareaFeat[8]++;
				}
			}
		}
		for (int i = 4; i < 9; i++)
		{
			subareaFeat[i] /= (nlx*ncx);
		}
	}
//    finish=clock();
//    duration=(double)(finish-start)/CLOCKS_PER_SEC;
//    printf("computer HSV: %f\n",duration);
	return FILTER_OK;
}

int globalFeatExtractor::calcHistHSV(vector<float> &histFeat)
{
	Mat im;
	this->imgData.convertTo(im, CV_32FC3);
	im = im / 255;
	cvtColor(im, im, CV_BGR2HSV);

	histFeat.resize(12, 0);
	float avgsat = 0;

	for (int i = 0; i<im.rows; i++)
	{
		for (int j = 0; j<im.cols; j++)
		{
			float pix = im.at<Vec3f>(i, j)[0], pix2 = im.at<Vec3f>(i, j)[2];

			if (pix<10 || pix>320)
			{
				histFeat[0]++;
			}
			else if (pix<45)
			{
				histFeat[1]++;
			}
			else if (pix<80)
			{
				histFeat[2]++;
			}
			else if (pix<140)
			{
				histFeat[3]++;
			}
			else if (pix<190)
			{
				histFeat[4]++;
			}
			else if (pix<255)
			{
				histFeat[5]++;
			}
			else if (pix<275)
			{
				histFeat[6]++;
			}
			else
			{
				histFeat[7]++;
			}


			if (pix2<0.15)
			{
				histFeat[8]++;
			}
			else if (pix2<0.4)
			{
				histFeat[9]++;
			}
			else if (pix2<0.75)
			{
				histFeat[10]++;
			}
			else
				histFeat[11]++;
		}
	}


	for (int ii = 0; ii<histFeat.size(); ii++)
	{
		histFeat[ii] = histFeat[ii] / (im.rows*im.cols);
	}
	return FILTER_OK;
}

int globalFeatExtractor::CalcDofFeat(float &dofFeat1,float &dofFeat2)
{
   	Mat I;
	if (this->imgData.channels() == 3)
	{
		cvtColor(this->imgData, I, CV_BGR2GRAY);
	}
	int cx = I.cols/2;
	int cy = I.rows/2;      

	Mat SegTL(I,Rect(0,0,cx,cy));
	Mat SegTR(I,Rect(cx,0,cx,cy));
	Mat SegBL(I,Rect(0,cy,cx,cy));
	Mat SegBR(I,Rect(cx,cy,cx,cy)); 
	Mat SegCenter(I,Rect(cx/2,cy/2,cx,cy));

	vector<float> SpectrumCal;

	SpectrumCal.push_back(calcHighSpec(SegTL));
	SpectrumCal.push_back(calcHighSpec(SegTR));
	SpectrumCal.push_back(calcHighSpec(SegBL));
	SpectrumCal.push_back(calcHighSpec(SegBR));
	SpectrumCal.push_back(calcHighSpec(SegCenter));

	float allSpec;
	allSpec=SpectrumCal[0]+SpectrumCal[1]+SpectrumCal[2]+SpectrumCal[3];
	SpectrumCal[0]=SpectrumCal[0]/allSpec;
	SpectrumCal[1]=SpectrumCal[1]/allSpec;
	SpectrumCal[2]=SpectrumCal[2]/allSpec;
	SpectrumCal[3]=SpectrumCal[3]/allSpec;
	SpectrumCal[4]=SpectrumCal[4]/allSpec;

	allSpec=SpectrumCal[0]+SpectrumCal[1]+SpectrumCal[2]+SpectrumCal[3];
	float ave=(allSpec+SpectrumCal[4])/5;

	dofFeat1=0;dofFeat2=0;
	for (int i=0;i<SpectrumCal.size();i++)
	{
		if (dofFeat1<SpectrumCal[i])
		{
			dofFeat1=SpectrumCal[i];
		}
		dofFeat2+=(SpectrumCal[i]-ave)*(SpectrumCal[i]-ave);
	}
    return FILTER_OK;
}

int globalFeatExtractor::calcHistHSV2(vector<float> &histFeat)
{
	Mat im;
	this->imgData.convertTo(im, CV_32FC3);
	im = im / 255;
	cvtColor(im, im, CV_BGR2HSV);

	histFeat.resize(11, 0);
	for (int i = 0; i<im.rows; i++)
	{
		for (int j = 0; j<im.cols; j++)
		{
			float pix = im.at<Vec3f>(i, j)[0], pix2 = im.at<Vec3f>(i, j)[2];
            //hist of H
			if (pix<20 || pix>316)
			{
				histFeat[0]++;
			}
			else if (pix<40)
			{
				histFeat[1]++;
			}
			else if (pix<75)
			{
				histFeat[2]++;
			}
			else if (pix<155)
			{
				histFeat[3]++;
			}
			else if (pix<190)
			{
				histFeat[4]++;
			}
			else if (pix<270)
			{
				histFeat[5]++;
			}
			else if (pix<295)
			{
				histFeat[6]++;
			}
			else
			{
				histFeat[7]++;
			}

            //hist of V
			if (pix2<0.2)
			{
				histFeat[8]++;
			}
			else if (pix2<0.7)
			{
				histFeat[9]++;
			}
			else
				histFeat[10]++;
		}
	}

	for (int ii = 0; ii<histFeat.size(); ii++)
	{
		histFeat[ii] = histFeat[ii] / (im.rows*im.cols);
	}
	return FILTER_OK;
}

int globalFeatExtractor::CalcDyrAndInc(Mat &srcImg,float &dyRan,float &inCon)
{
    vector<float> pixCount(256,0);
    float dynRan,IntenCon;
    for(int i=0;i<srcImg.rows;i++){
        for(int j=0;j<srcImg.cols;j++){
            int pixb=srcImg.at<Vec3b>(i,j)[0];
            pixCount[pixb]++;
            int pixg=srcImg.at<Vec3b>(i,j)[1];
			pixCount[pixg]++;
            int pixr=srcImg.at<Vec3b>(i,j)[2];
            pixCount[pixr]++;
		}
	}
    //normalization
    CvScalar sumPix=sum(pixCount);
    float sumCount=float(sumPix.val[0]);
    for(int i=0;i<256;i++){
        pixCount[i] /= sumCount;
	}
    int threL=0,threM=0,threH=255;
    float cumHisL=0,cumHisM=0,cumHisH=0;
    for(int i=0;i<256;i++){
        cumHisL += pixCount[i];
        cumHisM += pixCount[i];
        cumHisH += pixCount[255-i];
        if(cumHisL<0.05){
            threL++;
		}
        if(cumHisH<0.05){
            threH--;
		}
        if(cumHisM<0.5){
            threM++;
		}
        if(cumHisL>=0.05&&cumHisH>=0.05 && cumHisM>=0.5){
            break;
		}
	}
    threH = threH >= threL ? threH: threL;
    dyRan=1-float(threH-threL)/255;
    float sumH=0,sumL=0;
    for(int i=threL;i<=threM;i++){
        sumL += i*pixCount[i]/0.45;
	}
    for(int i=threM;i<=threH;i++){
        sumH += i*pixCount[i]/0.45;
	}
    sumH = sumH >= sumL ? sumH : sumL;
    inCon=1-(sumH-sumL)/255;
    return 0;
}
