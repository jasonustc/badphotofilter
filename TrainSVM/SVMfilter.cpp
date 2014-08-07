#include "SVMfilter.h"
#include "Functions.h"

using namespace std;
using namespace cv;

#define Jnums 77 
#define ParaInter  7
#define DEEP_FEATURE_DIM  1024
//the threshold to determine the over exposure photos
#define BRIGHT_THRESHOLD 0.95
//the threshold to determine the too dark photos
#define DARK_THRESHOLD 0.75
//the index of the darkness feature
#define DARK_INDEX 18
//the index of the brightness feature
#define BRIGHT_INDEX 19

SVMmodel::SVMmodel(const string &trainFeatFile,const string &trainLabelFile,
	const string &testFeatFile,const string &testLabelFile)
{
    //load train features
	int loc = trainFeatFile.rfind(".");
	string extension = trainFeatFile.substr(loc+1);
	if (extension == ".txt"){
		LoadFeat(trainFeatFile, this->trainData,DEEP_FEATURE_DIM);
	}
	else if (extension == ".xml"){
		LoadFeat(trainFeatFile, this->trainData);
	}
	else{
		fprintf(stderr, "%s is not a txt or a xml..\n", trainFeatFile);
	}
    //load test features
	loc = testFeatFile.rfind(".");
	extension = testFeatFile.substr(loc+1);
	if (extension == ".txt"){
		LoadFeat(testFeatFile, this->testData,DEEP_FEATURE_DIM);
	}
	else if (extension == ".xml"){
		LoadFeat(testFeatFile, this->testData);
	}
	else{
		fprintf(stderr, "%s is not a txt or a xml..\n", testFeatFile);
	}
    //load train label
	loc = trainLabelFile.rfind(".");
	extension = trainLabelFile.substr(loc+1);
	if (extension == "txt"){
		LoadFeat(trainLabelFile, this->trainLabel,1);
	}
	else if (extension == "xml"){
		LoadFeat(trainLabelFile, this->trainLabel);
	}
	else{
		fprintf(stderr, "%s is not a txt or a xml..\n", trainLabelFile);
	}
    //load test label
	loc = testLabelFile.rfind(".");
	extension = testLabelFile.substr(loc+1);
	if (extension == ".txt"){
		LoadFeat(testLabelFile, this->testLabel,1);
	}
	else if (extension == ".xml"){
		LoadFeat(testLabelFile, this->testLabel);
	}
	else{
		fprintf(stderr, "%s is not a txt or a xml..\n", testLabelFile);
	}
}

SVMmodel::SVMmodel(const string &trainFeatFile, const string &trainLabelFile,
	const string &testFeatFile, const string &testLabelFile,
	const float NegWeight, const float PosWeight)
{
	//load train features
	int loc = trainFeatFile.rfind(".");
	string extension = trainFeatFile.substr(loc + 1);
	if (extension == "txt"){
		LoadFeat(trainFeatFile, this->trainData, DEEP_FEATURE_DIM);
	}
	else if (extension == "xml"){
		LoadFeat(trainFeatFile, this->trainData);
	}
	else{
		fprintf(stderr, "%s is not a txt or a xml..\n", trainFeatFile);
	}
	//load test features
	loc = testFeatFile.rfind(".");
	extension = testFeatFile.substr(loc + 1);
	if (extension == "txt"){
		LoadFeat(testFeatFile, this->testData, DEEP_FEATURE_DIM);
	}
	else if (extension == "xml"){
		LoadFeat(testFeatFile, this->testData);
	}
	else{
		fprintf(stderr, "%s is not a txt or a xml..\n", testFeatFile);
	}
	//load train label
	loc = trainLabelFile.rfind(".");
	extension = trainLabelFile.substr(loc + 1);
	if (extension == "txt"){
		LoadFeat(trainLabelFile, this->trainLabel, 1);
	}
	else if (extension == "xml"){
		LoadFeat(trainLabelFile, this->trainLabel);
	}
	else{
		fprintf(stderr, "%s is not a txt or a xml..\n", trainLabelFile);
	}
	//load test label
	loc = testLabelFile.rfind(".");
	extension = testLabelFile.substr(loc + 1);
	if (extension == "txt"){
		LoadFeat(testLabelFile, this->testLabel, 1);
	}
	else if (extension == ".xml"){
		LoadFeat(testLabelFile, this->testLabel);
	}
	else{
		fprintf(stderr, "%s is not a txt or a xml..\n", testLabelFile);
	}
	this->NegWeight = NegWeight;
	this->PosWeight = PosWeight;
}

SVMmodel::SVMmodel(const int dataSetIdx,const int featDim1,const int featDim2,
	               const string trainImgPathFile){
	//train features
	ostringstream oss;
	oss << dataSetIdx;
	string trainDatafile = "trainData" + oss.str() + ".xml";
	LoadFeat(trainDatafile,this->trainData);

	//train labels
	string trainLabelfile = "trainLabel" + oss.str() + ".xml";
	LoadFeat(trainLabelfile,this->trainLabel);

	//test features
	string testDatafile = "testData" + oss.str() + ".xml";
	LoadFeat(testDatafile,testData);

	//test labels
	string testLabelfile = "testLabel" + oss.str() + ".xml";
	LoadFeat(testLabelfile,this->testLabel);
	cout << "Data loaded:\n";
	cout << this->trainData.rows << "\t" <<this->trainData.cols << "\n";
	cout <<this->testData.rows << "\t" <<this->testData.cols << "\n";
	// wipe out whether exist illegal data;
	WipeIllegalData(this->trainData);
	WipeIllegalData(this->testData);

    //choose the specified dim of features to train and test
	this->testData=this->testData(Range::all(), Range(featDim1, featDim2));
    this->trainData=this->trainData(Range::all(), Range(featDim1, featDim2));
}

SVMmodel::SVMmodel(const int dataSetIdx, const int featDim1, const int featDim2){
	//train features
	ostringstream oss;
	oss << dataSetIdx;
	string trainDatafile = "..\\ExtractFeature\\trainDataAll" + oss.str() + ".xml";
	LoadFeat(trainDatafile, this->trainData);

	//train labels
	string trainLabelfile = "..\\ExtractFeature\\trainLabel" + oss.str() + ".xml";
	LoadFeat(trainLabelfile, this->trainLabel);

	//test features
	string testDatafile = "..\\ExtractFeature\\testDataAll" + oss.str() + ".xml";
	LoadFeat(testDatafile, testData);

	//test labels
	string testLabelfile = "..\\ExtractFeature\\testLabel" + oss.str() + ".xml";
	LoadFeat(testLabelfile, this->testLabel);
	cout << "Data loaded:\n";
	cout << this->trainData.rows << "\t" << this->trainData.cols << "\n";
	cout << this->testData.rows << "\t" << this->testData.cols << "\n";

	// wipe out whether exist illegal data;
	WipeIllegalData(this->trainData);
	WipeIllegalData(this->testData);

	//choose the specified dim of features to train and test
	this->testData = this->testData(Range::all(), Range(featDim1, featDim2));
	this->trainData = this->trainData(Range::all(), Range(featDim1, featDim2));
    //the weight of positive samples in training
    this->PosWeight=1;
    //the weight of negative samples in training
    this->NegWeight=3;
}
//choose required cols of feature from the training data
bool SVMmodel::ChooseSubFeat(Mat &featMat,vector<int> &colMark)
{
    int newCols=accumulate(colMark.begin(),colMark.end(),0);
    Mat newFeatMat=Mat::zeros(featMat.rows,newCols,CV_32FC1);
    int currCol=0;
    for(int i=0;i<colMark.size();i++){
        if(colMark[i]==1){
            Mat colM=newFeatMat.col(currCol);
            featMat.col(i).copyTo(colM);
            currCol++;
		}
	}
    featMat=newFeatMat;
    return true;
}

int SVMmodel::CheckDarkOrExpo(Mat &featData)
{
    //too dark
    if(featData.at<float>(DARK_INDEX)>=DARK_THRESHOLD){
        return 1;
	}
    //over exposure
	else if(featData.at<float>(BRIGHT_INDEX)>=BRIGHT_THRESHOLD){
        return 2;
	}
	else{
        return 0;
	}
}

bool SVMmodel::CheckDarkOrExpo(const string& imgPathFile,Mat &featMat,Mat &labelMat)
{
    Mat noDarkFeat,noDarkLabel;
    vector<string> imgPath;
    LoadImgPath(imgPathFile,imgPath);
//    for( unsigned int i=0;i<imgPath.size();i++){
//        imgPath[i]=imgPath[i];
//	}
    string darkDir="darkImages";
    CheckDir(darkDir);
    int numSample=0;
    for(int i=0;i<featMat.rows;i++){
        string srcImgName=imgPath[i];
        int ifDarkOrExpo=CheckDarkOrExpo(featMat.row(i));
        if(ifDarkOrExpo==1){
            int xx=srcImgName.rfind("\\");
            string imgName=srcImgName.substr(xx);
            imgName=darkDir+imgName;
            CopyFileA(srcImgName.c_str(),imgName.c_str(),FALSE);
            cout<<"dark image:\t"<<srcImgName<<"\n";
		}
		else if(ifDarkOrExpo==2){
            int xx=srcImgName.rfind("\\");
            string imgName=srcImgName.substr(xx);
            imgName=darkDir+imgName;
            CopyFileA(srcImgName.c_str(),imgName.c_str(),FALSE);
            cout<<"bright image:\t"<<srcImgName<<"\n";
		}
		else{
            noDarkFeat.push_back(featMat.row(i));
            noDarkLabel.push_back(labelMat.at<int>(i));
            numSample++;
		}
	}
	noDarkFeat.reshape(1,numSample);
	noDarkLabel.reshape(1,numSample);
    featMat=noDarkFeat;
    labelMat=noDarkLabel;
    if(noDarkFeat.rows==0||noDarkFeat.cols==0){
        return false;
	}
	else{
        return true;
	}
}

//cross-validation to choose best parameters: param.gamma param.C
bool SVMmodel::TrainModel(Mat &trData,Mat & trLabel)
{
	CvSVMParams para;
	para.svm_type = CvSVM::C_SVC;
	para.kernel_type = CvSVM::RBF;
    //para.kernel_type=CvSVM::LINEAR;
	para.term_crit = cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10000, 1e-5);

	Mat myWeights(1, 2, CV_32FC1);
	myWeights.at<float>(0) = this->NegWeight;
	myWeights.at<float>(1) = this->PosWeight;
	// adjust the weights of positive and negative samples;
	// the weight order is ranked by the label's value in ascending order
	CvMat myWeightsXX = myWeights;
	para.class_weights = &myWeightsXX;
	vector<float> precisions(Jnums,0);
	vector<float> recalls(Jnums, 0);
	vector<float> Fscores(Jnums, 0);
	for (int i = 0; i<5; i++)
	{
		cout << i << " split Data" << endl;
		int dataNum = trData.rows;
		int testNum = dataNum / 5;
		int trainNum = dataNum - testNum;
		int  testBegin= i*testNum;
		int testEnd = (i + 1)*testNum;
		cout << testBegin<<"\t"<<testEnd<<endl;
        //train data and test data
		Mat splitTrain,sTrainLabel,splitTest,sTestLabel;
		splitTest = trData(Rect(0, testBegin, trData.cols,testNum));
		splitTrain.push_back(trData(Rect(0, 0, trData.cols, testBegin)));
		splitTrain.push_back(trData(Rect(0, testEnd, trData.cols, trData.rows-testEnd)));
		splitTrain.reshape(1, trainNum);
        //train label and test label
		sTestLabel = trLabel(Range(testBegin,testEnd),Range::all());
		sTrainLabel.push_back(trLabel(Range(0, testBegin),Range::all()));
		sTrainLabel.push_back(trLabel(Range(testEnd,trData.rows),Range::all()));
		sTrainLabel.reshape(1, trainNum);
        vector<float> maxFeats;
        vector<float> minFeats;
        MaxMinInRow(splitTrain,maxFeats,minFeats);
		MatNorm(splitTrain,maxFeats,minFeats);
		MatNorm(splitTest,maxFeats,minFeats);
		CvSVM cateSVMz;
		for (int j = 0; j<Jnums; j++)
		{
			// here is the adjust of parameters;
			para.gamma = 0.25*pow(sqrt(2.0), j / ParaInter);
			para.C = 0.001*pow(10.0, j % ParaInter);
			cateSVMz.train(splitTrain, sTrainLabel, Mat(), Mat(), para);

			Mat examData;
			vector<int> predLabels;
			vector<int> trueLabels;
			for (int k = 0; k < splitTest.rows; k++)
			{
				examData = splitTest.row(k);
				int res = cateSVMz.predict(examData);
				predLabels.push_back(res);
				trueLabels.push_back(sTestLabel.at<int>(k));
			}
            //get precision and recall 
			float precision, recall,Fscore;
			ComputePreRec(predLabels, trueLabels,precision,recall,Fscore);
			Fscores[j]+=Fscore;//select the model to have best Fscore and recall.
			precisions[j] += precision;
			recalls[j] += recall;
			cout << "iteration\t" << j << "\tprecison:\t" << precision << "\trecall:\t" << recall << "\n";
		}
	}
	float bestPrecision = 0,bestFscore=0,bestRecall=0;
	int bestJ = 0;
	for (int i = 0; i < recalls.size(); i++)
	{
		recalls[i] /= 5.0;
		precisions[i] /= 5.0;
		Fscores[i] /= 5.0;
		if (precisions[i] >= 0.75)
		{
			if (Fscores[i]>bestFscore)
			//if (recalls[i]>bestRecall)
			{
				bestPrecision = precisions[i];
				bestFscore = Fscores[i];
				bestRecall = recalls[i];
				bestJ = i;
				cout << "updated,recall: " << recalls[i] << "\tj:\t" << i << "\n";
			}
		}
	}
	this->bestGamma = 0.25*pow(sqrt(2.0), bestJ/ ParaInter);
	this->bestC = 0.001*pow(10.0, bestJ% ParaInter);
	cout << "train result: " << endl;
	cout << bestGamma << "\t" << bestC << "\t" << bestJ <<"\t"<< bestRecall <<  endl;
	return true;
}
// use the best parameters to train SVM model and test it.
bool SVMmodel::TestModel(string &SVMfile,string &imgPathFile,
	ofstream &out_PreRec,const float &Threshold)
{
	CvSVMParams para;
	para.gamma = this->bestGamma;
	para.C = this->bestC;
	para.svm_type = CvSVM::C_SVC;
	para.kernel_type = CvSVM::RBF;
	//para.kernel_type = CvSVM::LINEAR;
	para.term_crit = cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10000, 1e-5);

	Mat myWeights(1, 2, CV_32FC1);
	myWeights.at<float>(0) = this->NegWeight;
	myWeights.at<float>(1) = this->PosWeight;
	// adjust the weights of positive and negative samples;
	// the weight order is ranked by the label's value in ascending order
	CvMat myWeightsXX = myWeights;
	para.class_weights = &myWeightsXX;
	// take the whole training set to train the model; and save the model parameters 
	CvSVM cateSVMfinal;
	cateSVMfinal.train(this->trainData, this->trainLabel, Mat(), Mat(), para);
    //save model.
	cateSVMfinal.save(SVMfile.c_str());
    //save normalization coefficients
	FileStorage fs(SVMfile.c_str(), FileStorage::APPEND);
	fs << "maxDimensions" << this->maxCoeffs;
	fs << "minDimensions" << this->minCoeffs;
	fs.release();

	vector<int> predLabels;
	vector<int> trueLabels;
    vector<string> imgPath;
    LoadImgPath(imgPathFile,imgPath);
    for (unsigned int i=0;i<imgPath.size();i++){
        imgPath[i]="..\\ExtractFeature\\"+imgPath[i];
	}
	for (int i = 0; i < testData.rows; i++)
	{
        int ifDark=CheckDarkOrExpo(testData.row(i));
        if(ifDark==1){
			predLabels.push_back(-1);
			trueLabels.push_back(testLabel.at<int>(i));
            cout<<"Predict as dark image:\t"<<imgPath[i]<<"\n";
		}
        else if(ifDark==2){
			predLabels.push_back(-1);
			trueLabels.push_back(testLabel.at<int>(i));
            cout<<"Predict as over exposure image:\t"<<imgPath[i]<<"\n";
		}
		else{
			int res= cateSVMfinal.predict(testData.row(i));
			predLabels.push_back(res);
			trueLabels.push_back(testLabel.at<int>(i));
		}
	}
	float precision, recall, Fscore;
	ComputePreRec(predLabels, trueLabels, precision, recall, Fscore, imgPathFile, out_PreRec);
	cout << "precision:\t" << precision << "\trecall:\t" << recall<< "\tFscore:\t" << Fscore << endl;
	return true;
}

bool SVMmodel::TestModel(string &SVMfile, ofstream &out_PreRec, const float &Threshold)
{
	CvSVMParams para;
	para.gamma = this->bestGamma;
	para.C = this->bestC;
	para.svm_type = CvSVM::C_SVC;
	para.kernel_type = CvSVM::RBF;
	//para.kernel_type = CvSVM::LINEAR;
	para.term_crit = cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10000, 1e-5);

	Mat myWeights(1, 2, CV_32FC1);
	myWeights.at<float>(0) = this->NegWeight;
	myWeights.at<float>(1) = this->PosWeight;
	// adjust the weights of positive and negative samples;
	// the weight order is ranked by the label's value in ascending order
	CvMat myWeightsXX = myWeights;
	para.class_weights = &myWeightsXX;
	// take the whole training set to train the model; and save the model parameters 
	CvSVM cateSVMfinal;
	cateSVMfinal.train(this->trainData, this->trainLabel, Mat(), Mat(), para);
	//save model.
	cateSVMfinal.save(SVMfile.c_str());
	//save normalization coefficients
	FileStorage fs(SVMfile.c_str(), FileStorage::APPEND);
	fs << "maxDimensions" << this->maxCoeffs;
	fs << "minDimensions" << this->minCoeffs;
	fs.release();

	Mat examDatas;
	vector<int> predLabels;
	vector<int> trueLabels;
	for (int i = 0; i < testData.rows; i++)
	{
		float score = 0;
		float *pRow = testData.ptr<float>(i);
		int res= cateSVMfinal.predict(testData.row(i), true);
    	predLabels.push_back(res);
		trueLabels.push_back(testLabel.at<int>(i));
    }
	//compute precision and recall
	float precision, recall, Fscore;
	ComputePreRec(predLabels, trueLabels, precision, recall, Fscore, out_PreRec);
	cout << "precision:\t" << precision << "\trecall:\t" << recall << "\tFscore:\t" << Fscore << endl;
	return true;
}


bool SVMmodel::TrainAndTest(ofstream &out_PreRec,const float threshold)
{
	ostringstream oss;
    //save normalize coefficients
	vector<float> maxDims, minDims;
	MaxMinInRow(this->trainData, this->maxCoeffs,this->minCoeffs);

	//normalize the training data and test data, details can be seen in the dataProcessNorm.cpp;
	MatNorm(this->trainData,this->maxCoeffs,this->minCoeffs);
	MatNorm(this->testData,this->maxCoeffs,this->minCoeffs);

    // train and validate for best parameters.
	TrainModel(this->trainData, this->trainLabel);

    //test the performace
	oss.str("");
	oss << __TIME__<<__DATE__;
	string SVMfile = "SVM_model." + oss.str() + ".xml";
	WipeIllegalChar(SVMfile);
	cout << SVMfile << endl;
	ofstream outParam("params.txt");
	string testImgPathFile = "testImg"+oss.str()+".txt";
	TestModel(SVMfile,testImgPathFile, out_PreRec,threshold);
	outParam.close();
	system("pause");
	return true;
}

bool SVMmodel::CrossValidAndTest(ofstream &out_PreRec, const float threshold,const int k)
{
	ostringstream oss;
    //filter out too dark or over exposure photos
    oss<<k;
	string trainImgPathFile = "..\\ExtractFeature\\trainImg" + oss.str() + ".txt";
    // choose a subset of the features to do bad photo filtering
    vector<int> colMark(this->trainData.cols,1);
//    for(int i=0;i<colMark.size();i++){
//        if(i==0 || i==1 || i==6 || i==7 || i==12 || i==13 || i==20 || i==21){
//            colMark[i]=1;
//		}
//	}
    ChooseSubFeat(this->trainData,colMark);
    ChooseSubFeat(this->testData,colMark);
    
    CheckDarkOrExpo(trainImgPathFile,this->trainData,this->trainLabel);
	//save normalize coefficients
	MaxMinInRow(this->trainData, this->maxCoeffs, this->minCoeffs);
	MatNorm(this->trainData,this->maxCoeffs,this->minCoeffs);

	//normalize the training data and test data
	MatNorm(this->trainData,this->maxCoeffs,this->minCoeffs);
	MatNorm(this->testData,this->maxCoeffs,this->minCoeffs);

	// train and validate for best parameters.
	TrainModel(this->trainData, this->trainLabel);

	//test the performace
	oss.str("");
	oss << __TIME__ << __DATE__;
	string SVMfile = "SVM_model." + oss.str() + ".xml";
	WipeIllegalChar(SVMfile);
	cout << SVMfile << endl;
    oss.str("");
    oss<<k;
	string testImgPathFile = "..\\ExtractFeature\\testImg" + oss.str() + ".txt";
    //use the best parameters to train and test the model.
	TestModel(SVMfile, testImgPathFile, out_PreRec, threshold);
	//system("pause");
	return true;
}