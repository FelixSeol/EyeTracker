#include <jni.h>
#include <opencv2/opencv.hpp>

using namespace cv;
extern "C" JNIEXPORT jstring

JNICALL
Java_com_example_felixxseol_eyetracker_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_felixxseol_eyetracker_MainActivity_ConvertRGBtoGray(JNIEnv *env, jobject instance,
                                                                     jlong matAddrInput,
                                                                     jlong matAddrResult) {

    Mat &matInput = *(Mat *)matAddrInput;

    Mat &matResult = *(Mat *)matAddrResult;


    cvtColor(matInput, matResult, CV_RGBA2GRAY);
}