#include <jni.h>
#include <opencv2/opencv.hpp>
#include <android/log.h>

using namespace cv;
using namespace std;


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
//    img_result = Mat(img_input.size(), img_input.type(), Scalar::all(2));

    std::vector<Rect> faces;
    Mat img_gray;

    cvtColor(img_input, img_gray, COLOR_BGR2GRAY);
    equalizeHist(img_gray, img_gray);
    Mat img_resize;
    float resizeRatio = resize(img_gray, img_resize, 640);

    //-- Detect faces
    ((CascadeClassifier *) cascadeClassifier_face)->detectMultiScale( img_resize, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    __android_log_print(ANDROID_LOG_DEBUG, (char *) "native-lib :: ", (char *) "face %d found. ", faces.size());
    for( int i = 0; i < faces.size(); i++) {
        double real_facesize_x = faces[0].x / resizeRatio;
        double real_facesize_y = faces[0].y / resizeRatio;
        double real_facesize_width = faces[0].width / resizeRatio;
        double real_facesize_height = faces[0].height / resizeRatio;
        Scalar rect_color;
        Point center(real_facesize_x + real_facesize_width / 2,
                     real_facesize_y + real_facesize_height / 2);

        Rect face_area(real_facesize_x, real_facesize_y, real_facesize_width, real_facesize_height);
        __android_log_print(ANDROID_LOG_DEBUG, (char *) "native-lib :: ", (char *) "Face ROI X : %f, Y : %f", real_facesize_x, real_facesize_y);
        __android_log_print(ANDROID_LOG_DEBUG, (char *) "native-lib :: ", (char *) "Face ROI width : %f, height : %f", real_facesize_width, real_facesize_height);
        if (real_facesize_width > 800){
            rect_color = Scalar(0, 255, 255);
            img_input(face_area).copyTo(img_result(face_area));

            Mat faceROI = img_gray(face_area);
            std::vector<Rect> eyes;
            //-- In each face, detect eyes
            ((CascadeClassifier *) cascadeClassifier_eye)->detectMultiScale(faceROI, eyes, 1.1, 2,
                                                                            0 | CASCADE_SCALE_IMAGE,
                                                                            Size(30, 30));

            for (size_t j = 0; j < eyes.size(); j++) {
                double real_eye_x = real_facesize_x + eyes[j].x;
                double real_eye_y = real_facesize_y + eyes[j].y;
                double real_eye_width = eyes[j].width;
                double real_eye_height = eyes[j].height;
                Point eye_center(real_eye_x + real_eye_width / 2,
                                 real_eye_y + real_eye_height/ 2);
                __android_log_print(ANDROID_LOG_DEBUG, (char *) "native-lib :: ", (char *) "Eye ROI X : %f, Y : %f", real_eye_x, real_eye_y);
                __android_log_print(ANDROID_LOG_DEBUG, (char *) "native-lib :: ", (char *) "Eye ROI width : %f, height : %f", real_eye_width, real_eye_width);
                Rect eye_area(real_eye_x, real_eye_y, eyes[j].width, eyes[j].height);
                if (real_eye_y < 500) {
                    rectangle(img_result, eye_area, Scalar(255, 0, 0), 5, 8, 0);
                }
//              int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
//              circle(img_result, eye_center, radius, Scalar(255, 0, 0), 5, 8, 0);
            }
            rectangle(img_result, face_area, rect_color, 8, 8, 0);
        }
    }
}