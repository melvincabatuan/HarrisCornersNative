/*
*  ImageProcessing.cpp
* package com.cabatuan.harriscornersnative;
*/
#include <jni.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat * temp = NULL;

extern "C"
jboolean
Java_com_cabatuan_harriscornersnative_CameraPreview_ImageProcessing(
		JNIEnv* env, jobject thiz,
		jint width, jint height,
		jbyteArray NV21FrameData, jintArray outPixels)
{
	jbyte * pNV21FrameData = env->GetByteArrayElements(NV21FrameData, 0);
	jint * poutPixels = env->GetIntArrayElements(outPixels, 0);

	if ( temp == NULL )
    	{
    		temp = new Mat(height, width, CV_8UC1);
    	}

	Mat mGray(height, width, CV_8UC1, (unsigned char *)pNV21FrameData);
	Mat mResult(height, width, CV_8UC4, (unsigned char *)poutPixels);

	Mat HarrisImg = *temp;
    GoodFeaturesToTrackDetector harris_detector( 5000, 0.001, 5, 3, true );
    vector<KeyPoint> keypoints;

    harris_detector.detect( mGray, keypoints );
    drawKeypoints( mGray, keypoints, HarrisImg, Scalar( 0, 255, 0));
	cvtColor(HarrisImg, mResult, CV_BGR2BGRA);

	env->ReleaseByteArrayElements(NV21FrameData, pNV21FrameData, 0);
	env->ReleaseIntArrayElements(outPixels, poutPixels, 0);
	return true;
}


//GoodFeaturesToTrackDetector( int maxCorners, double qualityLevel,
//                                 double minDistance, int blockSize=3,
//                                 bool useHarrisDetector=false, double k=0.04 );