#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <clocale>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/videoio.hpp>
using namespace std;
using namespace cv;

double calcMse(const Mat& im1, const Mat& im2);
double getPSNR(const Mat& I1, const Mat& I2);
Mat gray_Conv(Mat& res_Mat);
Mat bgrToHsv(Mat& myBgr);
Mat hsvToBgr(Mat& myHsv);
Mat ChangeBrightness(Mat &img, int value);

int main(int argc, char *argv[]){
	const Mat image_Ref = imread("image_1.jpg");
	Mat image_Without_Ref = imread("image_2.jpg");
	const Mat both_Image;
	if (image_Ref.empty()) {
		cout << "Error: the image has been incorrectly loaded." << endl;	
		return 0;
	}
	if (image_Without_Ref.empty()) {
		cout << "Error: the image has been incorrectly loaded." << endl;
		return 0;
	}
	
	namedWindow("DEFAULT_PICTURES");
	imshow("DEFAULT_PICTURES", image_Ref);
	waitKey(0);
	cvDestroyWindow("DEFAULT_PICTURES");
	
	namedWindow("MY_BGR_TO_GRAY");
	Mat res_1;
	image_Ref.copyTo(res_1);	
	imwrite("C:gray.jpg", res_1);
	long double t1_my_gray = clock();
	Mat res= gray_Conv(res_1);
	long double t2_my_gray = clock();
	t2_my_gray -= t1_my_gray;
	cout << "Clock_of_MY_BGR_TO_GRAY: "<<setprecision(15) << t2_my_gray/CLOCKS_PER_SEC << endl;
	imshow("MY_BGR_TO_GRAY", res);
	waitKey(0);
	cvDestroyWindow("MY_BGR_TO_GRAY");

	namedWindow("BGR_TO_GRAY");
	Mat gray_image;
	Mat gray_image_2;
	long double t1_cv_gray = clock();
	cvtColor(image_Ref, gray_image, COLOR_BGR2GRAY);
	long double t2_cv_gray = clock();
	cvtColor(gray_image, gray_image_2 , COLOR_GRAY2BGR);
	t2_cv_gray -= t1_cv_gray;
	cout << "Clock_of_CV_BGR_TO_GRAY: " << setprecision(15) << t2_cv_gray / CLOCKS_PER_SEC << endl;
	imshow("BGR_TO_GRAY", gray_image_2);
	waitKey(0);
	cvDestroyWindow("BGR_TO_GRAY");

	namedWindow("MY_BGR_TO_HSV");
	Mat myBgrToHsv;
	Mat res_2; image_Ref.copyTo(res_2);
	long double t1_my_bgr2hsv = clock();
	myBgrToHsv = bgrToHsv(res_2);
	long double t2_my_bgr2hsv = clock();
	t2_my_bgr2hsv -= t1_my_bgr2hsv;
	cout << "Clock_of_MY_BGR_TO_HSV: " << setprecision(15) << t2_my_bgr2hsv / CLOCKS_PER_SEC << endl;
	imshow("MY_BGR_TO_HSV", myBgrToHsv);
	waitKey(0);
	cvDestroyWindow("MY_BGR_TO_HSV");

	namedWindow("MY_HSV_TO_BGR");
	Mat myHsvToBgr;
	Mat res_3; myBgrToHsv.copyTo(res_3);
	long double t1_my_hsv2bgr = clock();
	myHsvToBgr = hsvToBgr(res_3);
	long double t2_my_hsv2bgr = clock();
	t2_my_hsv2bgr -= t1_my_hsv2bgr;
	cout << "Clock_of_MY_HSV_TO_BGR: " << setprecision(15) << t2_my_hsv2bgr / CLOCKS_PER_SEC << endl;
	imshow("MY_HSV_TO_BGR", myHsvToBgr);
	waitKey(0);
	cvDestroyWindow("MY_HSV_TO_BGR");

	namedWindow("BGR_TO_HSV");
	Mat BgrToHsv;
	Mat res_4; image_Ref.copyTo(res_4);
	long double t1_bgr2hsv = clock();
	cvtColor(res_4, BgrToHsv, CV_BGR2HSV);
	long double t2_bgr2hsv = clock();
	t2_bgr2hsv -= t1_bgr2hsv;
	cout << "Clock_of_BGR_TO_HSV: " << setprecision(15) << t2_bgr2hsv / CLOCKS_PER_SEC << endl;
	imshow("BGR_TO_HSV", BgrToHsv);
	waitKey(0);
	cvDestroyWindow("BGR_TO_HSV");

	namedWindow("HSV_TO_BGR");
	Mat HsvToBgr;
	Mat res_5; BgrToHsv.copyTo(res_5);
	long double t1_hsv2bgr = clock();
	cvtColor(res_5, HsvToBgr, CV_HSV2BGR);
	long double t2_hsv2bgr = clock();
	t2_hsv2bgr -= t1_hsv2bgr;
	cout << "Clock_of_HSV_TO_BGR: " << setprecision(15) << t2_hsv2bgr / CLOCKS_PER_SEC << endl;
	imshow("HSV_TO_BGR", HsvToBgr);
	waitKey(0);
	cvDestroyWindow("HSV_TO_BGR");

	namedWindow("CHANGE_BRIGHTNESS_BGR");
	Mat chBrthBgr;
	Mat res_6; image_Ref.copyTo(res_6);
	long double t1_chBrth_bgr = clock();
	chBrthBgr = ChangeBrightness(res_6, 2);
	long double t2_chBrth_bgr = clock();
	t2_chBrth_bgr -= t1_chBrth_bgr;
	cout << "Clock_of_CHANGE_BRIGHTNESS_BGR: " << setprecision(15) << t2_chBrth_bgr / CLOCKS_PER_SEC << endl;
	imshow("CHANGE_BRIGHTNESS_BGR", chBrthBgr);
	waitKey(0);
	cvDestroyWindow("CHANGE_BRIGHTNESS_BGR");

	namedWindow("CHANGE_BRIGHTNESS_HSV");
	Mat chBrthHsv;
	Mat res_7; myBgrToHsv.copyTo(res_7);
	long double t1_chBrth_hsv = clock();
	chBrthHsv = ChangeBrightness(res_7, 2);
	long double t2_chBrth_hsv = clock();
	t2_chBrth_hsv -= t1_chBrth_hsv;
	cout << "Clock_of_CHANGE_BRIGHTNESS_HSV: " << setprecision(15) << t2_chBrth_hsv / CLOCKS_PER_SEC << endl;
	imshow("CHANGE_BRIGHTNESS_HSV", chBrthHsv);	
	waitKey(0);
	cvDestroyWindow("CHANGE_BRIGHTNESS_HSV");

	double ans = getPSNR(image_Ref, image_Without_Ref);
	cout << "MY_PICTURES: " << ans << endl;
	double ans_2 = getPSNR(res, gray_image_2);
	cout << "BGR_TO_GRAY: " << ans_2 << endl;
	double ans_3 = getPSNR(chBrthBgr, chBrthHsv);
	cout << "CHANGE_BRIGHTNESS: " << ans_3 << endl;
	cout << "DIFFERENCE_OF_TIME_OF_MY_AND_CV_BGR_TO_GRAY: " << setprecision(15) << (max(t2_my_gray, t2_cv_gray) - min(t2_my_gray, t2_cv_gray)) / CLOCKS_PER_SEC << endl;
	cout << "DIFFERENCE_OF_TIME_OF_MY_AND_CV_BGR_TO_HSV: " << setprecision(15) << (max(t2_my_bgr2hsv, t2_bgr2hsv) - min(t2_my_bgr2hsv, t2_bgr2hsv)) / CLOCKS_PER_SEC << endl;
	cout << "DIFFERENCE_OF_TIME_OF_MY_AND_CV_HSV_TO_BGR: " << setprecision(15) << (max(t2_my_hsv2bgr, t2_hsv2bgr) - min(t2_my_hsv2bgr, t2_hsv2bgr)) / CLOCKS_PER_SEC << endl;
	cout << "DIFFERENCE_OF_TIME_OF_BRTHN_HSV_AND_BGR: " << setprecision(15) << (max(t2_chBrth_hsv, t2_chBrth_bgr) - min(t2_chBrth_hsv, t2_chBrth_bgr)) / CLOCKS_PER_SEC << endl;

	waitKey(0);
	return 0;
}


double calcMse(const Mat& im1, const Mat& im2) {
	double ans = 0;
	int height = im1.rows;
	int width = im2.cols;
	int N = height * width;
	ans += sum((im1 - im2).mul(im1 - im2))[0];
	ans += sum((im1 - im2).mul(im1 - im2))[1];
	ans += sum((im1 - im2).mul(im1 - im2))[2];
	return ans / N;
}
double getPSNR(const Mat& I1, const Mat& I2)
{	
	double M = 255;
	double MSE = calcMse(I1, I2);
	if (MSE == 0) {
		cout << "psnr:   " << "Identical pictures!!!" << endl;
		return 10.0 * log10(M * M);
	}
	else {
		return 10.0 * log10(M * M / MSE);
	}
}

double CalcMod(double a, int b) {
	return a - floor(a / b) * b;
}

Mat gray_Conv(Mat& res_Mat) {
	
	for (int i = 0; i < res_Mat.rows; i++) {
		for (int j = 0; j < res_Mat.cols; j++) {
			res_Mat.at<Vec3b>(i, j)[0] = res_Mat.at<Vec3b>(i, j)[0] * 0.2952 + res_Mat.at<Vec3b>(i, j)[1] * 0.5547 + res_Mat.at<Vec3b>(i, j)[2] * 0.148;
			res_Mat.at<Vec3b>(i, j)[1] = res_Mat.at<Vec3b>(i, j)[0] * 0.2952 + res_Mat.at<Vec3b>(i, j)[1] * 0.5547 + res_Mat.at<Vec3b>(i, j)[2] * 0.148;
			res_Mat.at<Vec3b>(i, j)[2] = res_Mat.at<Vec3b>(i, j)[0] * 0.2952 + res_Mat.at<Vec3b>(i, j)[1] * 0.5547 + res_Mat.at<Vec3b>(i, j)[2] * 0.148;
		}
	}
	return res_Mat;
}

Mat bgrToHsv(Mat& myBgr) {
	myBgr.forEach<Vec3b>([](Vec3b&p, const int*pos) {
		int b = p[0];
		int g = p[1];
		int r = p[2];
		double bb = (double)b / 255.0;
		double gg = (double)g / 255.0;
		double rr = (double)r / 255.0;
		double Cmax = max({ bb,gg,rr });
		double Cmin = min({ bb,gg,rr });
		double d = Cmax - Cmin;
		double H, S, V;
		double qwe = CalcMod((gg - bb) / d, 6);
		if (d == 0) H = 0;
		else if (Cmax == rr) H = 60 * qwe;
		else if (Cmax == gg) H = 60 * (((bb - rr) / d) + 2);
		else if (Cmax == bb) H = 60 * (((bb - rr) / d) + 4);
		if (Cmax == 0) S = 0;
		else S = d / Cmax;
		V = Cmax;
		Vec3d hsv = Vec3d(H, S*255.0, V*255.0);
		p = hsv;
	});
	return myBgr;
}

Mat hsvToBgr(Mat& myHsv) {
	myHsv.forEach<Vec3b>([](Vec3b&p, const int*pos) {
		double H = p[0];
		double S = p[1] / 255.0;
		double V = p[2] / 255.0;
		double C = V * S;
		double X = C * (1 - abs(CalcMod(H / 60, 2) - 1));
		double m = V - C;
		double h = H;

		Vec3d bgr;
		if (h >= 0 && h < 60)		bgr = Vec3d(0, X, C);
		if (h >= 60 && h < 120)		bgr = Vec3d(0, C, X);
		if (h >= 120 && h < 180)	bgr = Vec3d(X, C, 0);
		if (h >= 180 && h < 240)	bgr = Vec3d(C, X, 0);
		if (h >= 240 && h < 300)	bgr = Vec3d(C, 0, X);
		if (h >= 300 && h < 360)	bgr = Vec3d(X, 0, C);

		bgr[0] = (bgr[0] + m) * 255;
		bgr[1] = (bgr[1] + m) * 255;
		bgr[2] = (bgr[2] + m) * 255;
		p = bgr;
	});
	return myHsv;
}

Mat ChangeBrightness(Mat &img, int value) {
	return img * value;
}