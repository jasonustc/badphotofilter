
#Documentation

#Description of the project
###ExtractFeature
####Function: Extract features from given dataset and save as training data and testing data in xml format.
####some added functions in the feature extractor:
- **calcHistHSV2:** calculate the HSV histogram based on [1] 
- **CalcDyrAndInc:** calculate the dynamic range and intensity contrast in [2] (maybe included in the next version of our bad photo filter)
- **CalcBlur2:** exactly the same as [3] to compute the blur feature of a photo
###TrainSVM
####Function: train a SVM model to filter out bad photos and test its performance.
#Useful information:
####ExtractFeature:
>- because we split the dataset into 5 part, 4 for training and 1 for testing, so there are 5 training sets and 5 testing sets
> - put all the photos in the folder "data"
> - run the "ExtractPath.py" to extract and save path of photos into "imgPath.txt"
> - all the features will be saved in "trainDataAllx.xml" and "testDataAllx.xml"
>- all the path of photos will be split and saved into "trainImgx.txt" and "testImgx.txt"

####TrainSVM
>- please first copy the same dataset as in ExtractFeature into the folder **"data"**
>- the program will load the data in xmls saved by previous program
>- the filtered dark or overexposure photos will be saved in the **"darkImages"** folder
>- the bad but predicted as good photos will be saved in the **"falseNeg"** folder
>- the good but predicted as bad photos will be saved in the **"falsePos"** folder
>- just the same as **"truePos"** and **"trueNeg"** folder 
>- the SVM model will be saved as **"SVM_model.(time).xml"**
>- the test result is saved in "result.txt", the format is as follows:
 trainingset  | \#TPT | \#TPF | \#FPT | \#FPF | precision | recall | F-score
  -------------  | --- | ---  | --- | ---| ---       | ----  |
 1~5  | xxx | xxx | xxx | xxx | xxx | xxx | xxx

#Reference
    [1] Chengyu Wu, Xiaoying Tai, "Image Retrieval Based on Color and Texture", Fourth International Conference on Fuzzy Systems and Knowledge Discovery (FSKD 2007).
    [2] Tian Xia, Tao Mei, Gang Hua, Yong-Dong Zhang, Xian-Sheng Hua,"Visual quality assessment for web videos", J. Vis. Commun. (2010), doi:10.1016/j.jvcir.2010.06.
    [3] Hanghang Tong,Mingjing Li, Hongjiang Zhang,Changshui Zhang,"Blur Detection for Digital Images Using Wavelet Transform",in Proc. IEEE Int. Conf. Multimedia Expo, Jun. 2004, pp. 17–20.

