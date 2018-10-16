#include <jni.h>
#include <opencv2/opencv.hpp>
#include <android/log.h>
#include <constants.h>

#include "findEyeCenter.h"

using namespace cv;
using namespace std;

cv::Point lastPoint;
std::vector<cv::Point> centers;
cv::Point mousePoint;
//
//void findEyes(cv::Mat frame_gray, cv::Rect face, Mat& img_result) {
//    cv::Mat faceROI = frame_gray(face);
//    cv::Mat debugFace = faceROI;
//
//    if (kSmoothFaceImage) {
//        double sigma = kSmoothFaceFactor * face.width;
//        GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
//    }
//    //-- Find eye regions and draw them
//    int eye_region_width = face.width * (kEyePercentWidth/100.0);
//    int eye_region_height = face.width * (kEyePercentHeight/100.0);
//    int eye_region_top = face.height * (kEyePercentTop/100.0);
//    cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
//                           eye_region_top,eye_region_width,eye_region_height);
//    cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
//                            eye_region_top,eye_region_width,eye_region_height);
//
//    //-- Find Eye Centers
//    cv::Point leftPupil = findEyeCenter(faceROI,leftEyeRegion);
//    cv::Point rightPupil = findEyeCenter(faceROI,rightEyeRegion);
//    // get corner regions
//    cv::Rect leftRightCornerRegion(leftEyeRegion);
//    leftRightCornerRegion.width -= leftPupil.x;
//    leftRightCornerRegion.x += leftPupil.x;
//    leftRightCornerRegion.height /= 2;
//    leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
//    cv::Rect leftLeftCornerRegion(leftEyeRegion);
//    leftLeftCornerRegion.width = leftPupil.x;
//    leftLeftCornerRegion.height /= 2;
//    leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
//    cv::Rect rightLeftCornerRegion(rightEyeRegion);
//    rightLeftCornerRegion.width = rightPupil.x;
//    rightLeftCornerRegion.height /= 2;
//    rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
//    cv::Rect rightRightCornerRegion(rightEyeRegion);
//    rightRightCornerRegion.width -= rightPupil.x;
//    rightRightCornerRegion.x += rightPupil.x;
//    rightRightCornerRegion.height /= 2;
//    rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
//    rectangle(debugFace,leftRightCornerRegion,200);
//    rectangle(debugFace,leftLeftCornerRegion,200);
//    rectangle(debugFace,rightLeftCornerRegion,200);
//    rectangle(debugFace,rightRightCornerRegion,200);
//    // change eye centers to face coordinates
//    rightPupil.x += rightEyeRegion.x;
//    rightPupil.y += rightEyeRegion.y;
//    leftPupil.x += leftEyeRegion.x;
//    leftPupil.y += leftEyeRegion.y;
//    // draw eye centers
//    circle(debugFace, rightPupil, 3, 1234);
//    circle(debugFace, leftPupil, 3, 1234);
////
////    cv::Rect roi( cv::Point( 0, 0 ), faceROI.size());
////    cv::Mat destinationROI = img_result( roi );
////    faceROI.copyTo( destinationROI );
//}

float resize(Mat img_src, Mat &img_resize, int resize_width){
    float scale = resize_width / (float)img_src.cols ;

    if (img_src.cols > resize_width) {
        int new_height = cvRound(img_src.rows * scale);
        resize(img_src, img_resize, Size(resize_width, new_height));
    }
    else {
        img_resize = img_src;
    }

    return scale;
}

Point stabilize(std::vector<cv::Point> &points, int windowSize)
{
    float sumX = 0;
    float sumY = 0;
    int count = 0;
    for (int i = std::max(0, (int)(points.size() - windowSize)); i < points.size(); i++)
    {
        sumX += points[i].x;
        sumY += points[i].y;
        ++count;
    }
    if (count > 0)
    {
        sumX /= count;
        sumY /= count;
    }
    return Point(sumX, sumY);
}

cv::Rect getLeftmostEye(std::vector<cv::Rect> &eyes)
{
    int leftmost = 99999999;
    int leftmostIndex = -1;
    for (int i = 0; i < eyes.size(); i++)
    {
        if (eyes[i].tl().x < leftmost)
        {
            leftmost = eyes[i].tl().x;
            leftmostIndex = i;
        }
    }
    return eyes[leftmostIndex];
}

Vec3f getEyeball(Mat &eye, vector<Vec3f> &circles)
{
    vector<int> sums(circles.size(), 0);
    for (int y = 0; y < eye.rows; y++)
    {
        uchar *ptr = eye.ptr<uchar>(y);
        for (int x = 0; x < eye.cols; x++)
        {
            int value = static_cast<int>(*ptr);
            for (int i = 0; i < circles.size(); i++)
            {
                Point center((int)round(circles[i][0]), (int)round(circles[i][1]));
                int radius = (int)round(circles[i][2]);
                if (pow(x - center.x, 2) + pow(y - center.y, 2) < pow(radius, 2))
                {
                    sums[i] += value;
                }
            }
            ++ptr;
        }
    }
    int smallestSum = 999999;
    int smallestSumIndex = -1;
    for (int i = 0; i < circles.size(); i++)
    {
        if (sums[i] < smallestSum)
        {
            smallestSum = sums[i];
            smallestSumIndex = i;
        }
    }
    return circles[smallestSumIndex];
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_example_felixxseol_eyetracker_MainActivity_loadCascade(JNIEnv *env, jobject instance,
                                                                jstring cascadeFileName_) {
    const char *nativeFileNameString = env->GetStringUTFChars(cascadeFileName_, 0);
    string baseDir("/storage/emulated/0/");
    baseDir.append(nativeFileNameString);
    const char *pathDir = baseDir.c_str();

    jlong ret = 0;
    ret = (jlong) new CascadeClassifier(pathDir);

    env->ReleaseStringUTFChars(cascadeFileName_, nativeFileNameString);

    return ret;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_felixxseol_eyetracker_MainActivity_detect(JNIEnv *env, jobject instance,
                                                           jlong cascadeClassifier_face,
                                                           jlong cascadeClassifier_eye,
                                                           jlong matAddrInput,
                                                           jlong matAddrResult) {

    Mat &img_input = *(Mat *) matAddrInput;
    Mat &img_result = *(Mat *) matAddrResult;
    img_result = img_input.clone();

    Mat img_gray;

    cvtColor(img_input, img_gray, COLOR_BGR2GRAY);
    equalizeHist(img_gray, img_gray);
    Mat img_resize;
    float resizeRatio = resize(img_gray, img_resize, 640);

    //-- Detect faces
    std::vector<Rect> faces;
    ((CascadeClassifier *) cascadeClassifier_face)->detectMultiScale( img_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(400, 400) );
    __android_log_print(ANDROID_LOG_DEBUG, (char *) "native-lib :: ", (char *) "face %d found. ", faces.size());
    Rect face = faces[0];
    rectangle(img_result, face, 1234);

//    if(faces.size() == 0) return;
//    findEyes(img_gray, face, img_result);

//    std::vector<Rect> eyes;
//    //-- In each face, detect eyes
//    ((CascadeClassifier *) cascadeClassifier_eye)->detectMultiScale(faceROI, eyes, 1.1, 2,
//                                                                    0 | CASCADE_SCALE_IMAGE,
//                                                                    Size(80, 80));
//    rectangle(img_result, faces[0].tl(), faces[0].br(), Scalar(0, 255, 255), 2);
//    if(eyes.size() != 2) return;
//    for (Rect &eye : eyes) {
//        rectangle(img_result, faces[0].tl() + eye.tl(), faces[0].tl() + eye.br(), Scalar(255, 255, 255), 2);
//        __android_log_print(ANDROID_LOG_DEBUG, (char *) "native-lib :: ", (char *) "eyes %d found. ", eyes.size());
//        findEyeCenter(faceROI, eye);
//    }
//    Rect eyeRect = getLeftmostEye(eyes);
//    Mat eyeROI = img_gray(eyeRect);
//    equalizeHist(eyeROI, eyeROI);
//    std::vector<Vec3f> circles;
//    HoughCircles(eyeROI, circles, CV_HOUGH_GRADIENT, 1, eyeROI.cols / 8, 250, 15, eyeROI.rows / 8, eyeROI.rows / 3);
//    if (circles.size() > 0){
//        Vec3f eyeball = getEyeball(eyeROI, circles);
//        Point center(eyeball[0], eyeball[1]);
//        centers.push_back(center);
//        center = stabilize(centers, 3);
//        if (centers.size() > 1)
//        {
//            Point diff;
//            diff.x = (center.x - lastPoint.x) * 20;
//            diff.y = (center.y - lastPoint.y) * -30;
//        }
//        lastPoint = center;
//        int radius = (int)eyeball[2];
//        circle(img_result, faces[0].tl() + eyeRect.tl() + center, radius, Scalar(0, 0, 255), 2);
//        circle(eyeROI, center, radius, cv::Scalar(255, 255, 255), 2);
//    }
}