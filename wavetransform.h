#ifndef WAVETRANSFORM_H
#define WAVETRANSFORM_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

using namespace cv;
using namespace std;
namespace WT {
    class WaveTransform;
}

class WaveTransform
{
public:
//    Mat WDT(const Mat &_src, const string _wname, const int _level);//小波分解
//    Mat IWDT(const Mat &_src, const string _wname, const int _level);//小波重构
//    void wavelet_D(const string _wname, Mat &_lowFilter, Mat &_highFilter);//分解包
//    void wavelet_R(const string _wname, Mat &_lowFilter, Mat &_highFilter);//重构包
//    Mat waveletDecompose(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter);
//    Mat waveletReconstruct(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter);
    Mat WDT(const Mat &_src, const string _wname, const int _level);//小波分解
    Mat IWDT(const Mat &_src, const string _wname, const int _level);//小波重构
    void wavelet_D(const string _wname, Mat &_lowFilter, Mat &_highFilter);//分解包
    void wavelet_R(const string _wname, Mat &_lowFilter, Mat &_highFilter);//重构包
    Mat waveletDecompose(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter);
    Mat waveletReconstruct(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter);
    //Mat sarReconstruct(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter);
};

#endif // WAVETRANSFORM_H
