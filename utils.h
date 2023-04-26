#ifndef UTILS_H
#define UTILS_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <QImage>
#include <vector>
#include "wavetransform.h"

#define PI 3.1416
#define min(a,b) (a<b?a:b)

using namespace cv;
using namespace std;

extern QImage image1,image2,image3,image4,transform1;  //image3,image4
extern Mat reff,src,image11,image22,image33,image44,full_src,src_recover;  //image33,image44
extern Mat merge11,merge12;  //merge12
extern std::vector<Point2f> imagePoints1, imagePoints2;
extern Mat ref_win, src_win;
extern const string& win1,win2;

std::vector<std::vector<int> > mat2Vec(Mat &SegImgMat); //Mat转为二维
Mat vec2Mat(std::vector<std::vector<int>> &SegImg); //二维转为Mat
double xycover(Mat image11,Mat image22,double mean1,double mean2);//计算协方差
double ComEntropy(Mat img1, Mat img2, double img1_entropy, double img2_entropy);//计算图像的联合熵
double meanstd(Mat image11,int n);//计算图像的平均值和标准差
double gradsAvg(Mat img);// 计算图像的平均梯度
double wrapcom(Mat image11,Mat image22);//计算图像扭曲程度
double Entropy(Mat img);//计算图像的熵
double comEntropy(Mat image11,Mat image22,Mat image33);//计算图像的互信息
double Qulity(Mat image11,Mat image22);//计算两个图像之间的通用图像质量指标
//图片格式转换
Mat QImageToMat(QImage qimage);
QImage MatToQImage(const cv::Mat& mat);
//图像配准函数
void on_mouse1(int event, int x, int y, int flags, void *ustc); //event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号
void on_mouse2(int event, int x, int y, int flags, void *ustc); //event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号
//小波变换
void  laplace_decompose(Mat src,int s,Mat &wave);//小波分解
void wave_recover(Mat full_scale, Mat &original,int level);//小波复原
void ware_operate(Mat &full_scale, int level);//小波操作
//hsi变换
int rgb2hsi(Mat &image,Mat &hsi);
int hsi2rgb(Mat &hsi,Mat &image);
void RGB2HSI(Mat src, Mat &dst);
void HSI2RGB(Mat src, Mat &dst);
void harr_fusion(Mat src1, Mat src2,Mat &dst);
void sar_fusion(Mat src1, Mat src2,Mat &dst);

Mat get2DGaussianKernel(int ksize, double sigma, int ktype = CV_64F);//高斯核

template <typename T> Mat matrix2Mat(T **matrix, int dtype);//数组转换为Mat
template <typename T> T ** matrix2Mat(Mat mat, int dtype);//Mat转换为数组

#endif // UTILS_H
