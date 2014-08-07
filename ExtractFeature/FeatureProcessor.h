#include "CalAesth.h"
#include "calcBlur.h"
#include "calcBrightness.h"
#include "calcEntropy.h"
#include "Global.h"
#include "myFeatures.h"
#include <opencv2/opencv.hpp>
#include <Windows.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

#define ALL_FEATURE_DIM 46 
#define DEEP_FEATURE_DIM  4096
#define LOCAL_CONTRAST_FEATURE_DIM 12 
#define GLOBAL_FEATURE_DIM 7 
#define EDGE_FEATURE_DIM 14 
#define HSV_FEATURE_DIM  12
#define DARK_THRESHOLD 0.5
#define BRIGHT_THRESHOLD 0.9
#define DARK_INDEX 17 
#define BRIGHT_INDEX 18 

using namespace std;