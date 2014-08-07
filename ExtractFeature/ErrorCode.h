#ifndef HEADER_ERROR_CODE
#define HEADER_ERROR_CODE

#define FILTER_OK 0  /*The filter works well.*/
#define FILTER_EMPTY_STREAM 1 /*The photo stream is empty*/
#define FILTER_INVALID_STREAM 2 /*Failed to decode the photo stream*/
#define FILTER_EMPTY_PATH 3 /*The image path is empty*/
#define EMPTY_IMAGE_DATA 4 /*Failed to load image data for given image path*/
#define TOO_SMALL_IMAGE_SIZE 5/*The image size is too small to process*/
#define CALC_BLUR_ERROR 6 /*Some error occurs in computing the blur of the image*/
#define EOUT_OF_MEMORY 7 /*The program is out of memory*/
#define E_BLURFAILED 8 /*Failed to compute the blur of the image*/
#define FAIL_TO_LOAD_SVM_PARAM 9 /*Failed to load the SVM model from the given file*/
#define WRONG_SVM_PATH 10 /*The SVM path is wrong.*/
#define WRONG_SVM_PARAM 11 /*The SVM parameters loaded is in the wrong format.*/
#endif