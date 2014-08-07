#include "global.h"

#ifndef BRIGHTNESS_ACCEPT_RATIO
#define BRIGHTNESS_ACCEPT_RATIO 0.45		// Max intensity value * 0.45
#endif

#ifndef BRIGHTNESS_ACCEPT_FRACTION
#define BRIGHTNESS_ACCEPT_FRACTION 0.30		// How many pixels' intensity in the frame exceed 0.45
#endif

#ifndef DARKNESS_ACCEPT_RATIO
#define DARKNESS_ACCEPT_RATIO 1.55		// Min intensity value * 1.55
#endif

#ifndef DARKNESS_ACCEPT_FRACTION
#define DARKNESS_ACCEPT_FRACTION 0.70		// How many pixels' intensity in the frame less than 1.55
#endif


float CalcBrightness(const IplImage *pSrcImg);
float CalcDarkness(const IplImage *pSrcImg);