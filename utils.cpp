#include "utils.h"

//Mat转为二维数组
std::vector<std::vector<int> > mat2Vec(Mat &SegImgMat){
    int row = SegImgMat.rows;
    int col = SegImgMat.cols;
    std::vector<std::vector<int>> SegImg(row, std::vector<int>(col,0));
    //赋值
    for(int i=0;i<row;i++)
        for (int j=0;j<col;j++)
            SegImg[i][j] = (int)(SegImgMat.at<uchar>(i,j));

    return SegImg;
}

//二维数组转为Mat
Mat vec2Mat(std::vector<std::vector<int> > &SegImg){
    int row = (int)SegImg.size();
    int col = (int)SegImg[0].size();
    Mat SegImgMat = Mat(row,col, CV_8U, Scalar::all(0));//初始化和SegImg数组相应的Mat
    //赋值
    for(int i=0;i<row;i++)
        for (int j=0;j<col;j++)
            SegImgMat.at<uchar>(i,j) = SegImg[i][j];
    return SegImgMat;
}

//计算协方差
double xycover(Mat image11,Mat image22,double mean1,double mean2){
    cvtColor(image11,image11,COLOR_RGB2GRAY);
    cvtColor(image22,image22,COLOR_RGB2GRAY);
    image11.convertTo(image11, CV_64FC1);
    image22.convertTo(image22, CV_64FC1);
    double cover=0.0;
    for (int i = 0; i < image11.rows; i++)
    {
        for (int j = 0; j < image11.cols; j++)

        {
            double r1=image11.at<double>(i,j)-mean1;
            double r2=image22.at<double>(i,j)-mean2;
            cover=cover+r1*r2;
        }

    }

    cover=cover/(image11.rows*image22.cols);
    return cover;

}

//计算图像的联合熵
double ComEntropy(Mat img1, Mat img2, double img1_entropy, double img2_entropy)
{
    double temp[256][256] = { 0.0 };

    // 计算联合图像像素的累积值
    for (int m1 = 0, m2 = 0; m1 < img1.rows, m2 < img2.rows; m1++, m2++)
    {    // 有效访问行列的方式
        const uchar* t1 = img1.ptr<uchar>(m1);
        const uchar* t2 = img2.ptr<uchar>(m2);
        for (int n1 = 0, n2 = 0; n1 < img1.cols, n2 < img2.cols; n1++, n2++)
        {
            int i = t1[n1], j = t2[n2];
            temp[i][j] = temp[i][j] + 1;
        }
    }

    // 计算每个联合像素的概率
    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j < 256; j++)

        {
            temp[i][j] = temp[i][j] / (img1.rows*img1.cols);
        }
    }

    double result = 0.0;
    //计算图像联合信息熵
    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j < 256; j++)

        {
            if (temp[i][j] == 0.0)
                result = result;
            else
                result = result - temp[i][j] * (log(temp[i][j]) / log(2.0));
        }
    }

    //得到两幅图像的互信息熵
    result = img1_entropy + img2_entropy-result;

    return result;

}

//计算图像的平均值和标准差
double meanstd(Mat image11,int n){
        Scalar mean;  //均值
        Scalar stddev;  //标准差
        cv::meanStdDev( image11, mean, stddev );  //计算均值和标准差
        double mean_pxl = mean.val[0];
        double stddev_pxl = stddev.val[0];
        return stddev_pxl;
        if(n==0)
                return mean_pxl;
        else
                return stddev_pxl;
}

// 计算图像的平均梯度
double gradsAvg(Mat img){
        cvtColor(img,img,COLOR_RGB2GRAY);
        img.convertTo(img, CV_64FC1);
        double tmp = 0;
        int rows = img.rows - 1;
        int cols = img.cols - 1;
        for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                        double dx = img.at<double>(i, j + 1) - img.at<double>(i, j);
                        double dy = img.at<double>(i + 1, j) - img.at<double>(i, j);
                        double ds = std::sqrt((dx*dx + dy*dy) / 2);
                        tmp += ds;
                }
        }
        double imageAvG = tmp / (rows*cols);
        return imageAvG;

}

//计算图像扭曲程度
double wrapcom(Mat image11,Mat image22){
        cvtColor(image11,image11,COLOR_RGB2GRAY);
        cvtColor(image22,image22,COLOR_RGB2GRAY);
        image11.convertTo(image11, CV_64FC1);
        image22.convertTo(image22, CV_64FC1);
        double sum=0;
        for(int i=0;i<image11.rows;i++){
                for(int j = 0; j <image11.cols; j++){
                        double minus=image11.at<double>(i,j)-image22.at<double>(i,j);
                        sum+=fabs(minus);
                }
        }
        double wrap=sum/(image11.rows*image11.cols);
        return wrap;
}

//计算图像的熵
double Entropy(Mat img){
        double temp[256] = { 0.0 }; // 计算每个像素的累积值
        for (int m = 0; m<img.rows; m++)
        {
                        const uchar* t = img.ptr<uchar>(m);// 有效访问行列的方式
                        for (int n = 0; n<img.cols; n++)
                        {
   int i = t[n];
   temp[i] = temp[i] + 1;
                        }
        }

        for (int i = 0; i<256; i++) // 计算每个像素的概率
        {
                temp[i] = temp[i] / (img.rows*img.cols);
        }

        double result = 0;
        for (int i = 0; i<256; i++)// 计算图像信息熵
        {
                if (temp[i] == 0.0)
                result = result;
        else
                result = result - temp[i] * (log(temp[i]) / log(2.0));
        }
        return result;
}

//计算图像的互信息
double comEntropy(Mat image11,Mat image22,Mat image33){
        double img1_entropy=Entropy(image11);
        double img2_entropy=Entropy(image22);
        double img3_entropy=Entropy(image33);
        double mutual1=ComEntropy(image11, image22, img1_entropy, img2_entropy);
        double mutual2=ComEntropy(image11, image33, img1_entropy, img3_entropy);
        double mutual=mutual1+mutual2;
        return mutual;

}

//计算两个图像之间的通用图像质量指标
double Qulity(Mat image11,Mat image22){
    double mean1=meanstd(image11,0),mean2=meanstd(image22,0);
    double std1=meanstd(image11,1),std2=meanstd(image22,1);
    double cover=xycover(image11,image22,mean1,mean2);
    double ui=(4*cover*mean1*mean2)/((pow(std1,2)*pow(std2,2))*(pow(mean1,2)*pow(mean2,2)));
    return ui;

}


//图片格式转换
Mat QImageToMat(QImage qimage)
{
    Mat mat = Mat(qimage.height(), qimage.width(), CV_8UC4, (uchar*)qimage.bits(), qimage.bytesPerLine());
    Mat mat2 = Mat(mat.rows, mat.cols, CV_8UC3 );
    int from_to[] = { 0,0, 1,1, 2,2 };
    mixChannels( &mat, 1, &mat2, 1, from_to, 3 );
    return mat2;
}
QImage MatToQImage(const cv::Mat& mat)
{
    // 8-bits unsigned, NO. OF CHANNELS = 1
    if(mat.type() == CV_8UC1)
    {
        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
        // Set the color table (used to translate colour indexes to qRgb values)
        image.setColorCount(256);
        for(int i = 0; i < 256; i++)
        {
            image.setColor(i, qRgb(i, i, i));
        }
        // Copy input Mat
        uchar *pSrc = mat.data;
        for(int row = 0; row < mat.rows; row ++)
        {
            uchar *pDest = image.scanLine(row);
            memcpy(pDest, pSrc, mat.cols);
            pSrc += mat.step;
        }
        return image;
    }
    // 8-bits unsigned, NO. OF CHANNELS = 3
    else if(mat.type() == CV_8UC3)
    {
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    else if(mat.type() == CV_8UC4)
    {
//        qDebug() << "CV_8UC4";
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return image.copy();
    }
    else
    {
//        qDebug() << "ERROR: Mat could not be converted to QImage.";
        return QImage();
    }
}

//图像配准函数
void on_mouse1(int event, int x, int y, int flags, void *ustc) //event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号
{
    if (event == EVENT_LBUTTONDOWN)//左键按下，读取初始坐标，并在图像上该点处打点
    {
        Point2f  p = Point2f(x, y);
        circle(ref_win, p, 30, Scalar(0,0,255), -1);
        imshow(win2, ref_win);
        imagePoints1.push_back(p);   //将选中的点存起来
    }
}

void on_mouse2(int event, int x, int y, int flags, void *ustc) //event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号
{
    if (event == EVENT_LBUTTONDOWN)//左键按下，读取初始坐标，并在图像上该点处打点
    {
        Point2f  p = Point2f(x, y);
        circle(src_win, p, 30, Scalar(0, 0, 255), -1);
        imshow(win1, src_win);
        imagePoints2.push_back(p);   //将选中的点存起来
    }
}
//小波分解
void  laplace_decompose(Mat src,int s,Mat &wave)
{
    Mat full_src(src.rows, src.cols, CV_32FC1);
    Mat dst = src.clone();
    dst.convertTo(dst, CV_32FC1);

    for (int m = 0; m < s; m++)
    {
        dst.convertTo(dst, CV_32FC1);
        Mat wave_src(dst.rows, dst.cols, CV_32FC1);
        //列变换，detail=mean-original
        for (int i = 0; i < wave_src.rows; i++)
        {
            for (int j = 0; j < wave_src.cols/2; j++)
            {
                wave_src.at<float>(i, j) = (dst.at<float>(i, 2 * j) + dst.at<float>(i, 2 * j + 1)) / 2;
                wave_src.at<float>(i, j + wave_src.cols/2) = wave_src.at<float>(i, j) - dst.at<float>(i, 2 * j);
            }
        }
        Mat temp = wave_src.clone();
        for (int i = 0; i < wave_src.rows/2; i++)
        {
            for (int j = 0; j < wave_src.cols / 2; j++)
            {
                wave_src.at<float>(i, j) = (temp.at<float>(2 * i, j) + temp.at<float>(2 * i + 1, j)) / 2;
                wave_src.at<float>(i + wave_src.rows / 2, j) = wave_src.at<float>(i, j) - temp.at<float>(2 * i, j);
            }
        }
        dst.release();
        dst = wave_src(Rect(0, 0, wave_src.cols / 2, wave_src.rows /2));
        wave_src.copyTo(full_src(Rect(0, 0, wave_src.cols, wave_src.rows)));
    }
    wave = full_src.clone();
}
//小波复原
void wave_recover(Mat full_scale, Mat &original,int level)
{
    //每一个循环中把一个级数的小波还原
    for (int m = 0; m < level; m++)
    {
        Mat temp = full_scale(Rect(0, 0, full_scale.cols / pow(2, level - m-1), full_scale.rows / pow(2, level - m-1)));

        //先恢复左边
        Mat recover_src(temp.rows, temp.cols, CV_32FC1);
        for (int i = 0; i < recover_src.rows; i++)
        {
            for (int j = 0; j < recover_src.cols/2; j++)
            {
                if (i % 2 == 0)
                    recover_src.at<float>(i, j) = temp.at <float>(i / 2, j) - temp.at<float>(i / 2 + recover_src.rows / 2, j);
                else
                    recover_src.at<float>(i, j) = temp.at <float>(i / 2, j)+ temp.at<float>(i / 2 + recover_src.rows / 2, j);
            }
        }
        Mat temp2 = recover_src.clone();
        //再恢复整个
        for (int i = 0; i < recover_src.rows; i++)
        {
            for (int j = 0; j < recover_src.cols; j++)
            {
                if (j % 2 == 0)
                    recover_src.at<float>(i, j) = temp2.at<float>(i, j / 2) - temp.at<float>(i, j / 2 + temp.cols / 2);
                else
                    recover_src.at<float>(i, j) = temp2.at<float>(i, j / 2) + temp.at<float>(i, j / 2 + temp.cols / 2);
            }
        }
        recover_src.copyTo(temp);
    }
    original = full_scale.clone();
    original.convertTo(original, CV_8UC1);
}

//小波操作
void ware_operate(Mat &full_scale, int level)
{
    //取出最低尺度的那一层，对其进行操作，仅最低尺度那层可以对时域进行操作，其他层只能对频域进行操作
    Mat temp = full_scale(Rect(0, 0, full_scale.cols / 4, full_scale.rows /4));
    temp = temp(Rect(0, 0, temp.cols / 2, temp.rows / 2));
    Mat temp2 = temp.clone();
    //这里对时域操作，降低灰度
    for (int i = 0; i < temp2.rows;i++)
    for (int j = 0; j < temp2.cols; j++)
        temp2.at<float>(i, j) -= 50;
    temp2.copyTo(temp);

    //这里对频域操作，拉伸细节
    //先处理左下角
//    for (int i = temp.rows / 2; i < temp.rows; i++)
//    {
//        for (int j = 0; j < temp.cols / 2; j++)
//        {
//            if (temp.at<float>(i, j)>0)
//                temp.at<float>(i, j) += 5;
//            if (temp.at<float>(i, j) < 0)
//                temp.at<float>(i, j) -= 5;
//        }
//    }
    //再处理右半边
    for (int i = 0; i < temp.rows; i++)
    {
        for (int j = 0; j < temp.cols; j++)
        {
            if (temp.at<float>(i, j)>0)
                temp.at<float>(i, j) += 20;
            if (temp.at<float>(i, j)<0)
                temp.at<float>(i, j) -= 20;
        }
    }
}
//hsi变换
int rgb2hsi(Mat &image,Mat &hsi){
    if(!image.data){
            return -1;
    }
    int nl = image.rows;
    int nc = image.cols;

    //如果图像连续的话将图像数据多维数组转换成一维数组表示
    if(image.isContinuous()){
            nc = nc*nl;

            nl = 1;
    }

    for(int i = 0;i < nl;i++){
            uchar *src = image.ptr<uchar>(i);//image.ptr可以得到图像任意行的首地址
            uchar *dst = hsi.ptr<uchar>(i);
            for(int j = 0;j < nc;j++){
                    float b = src[j*3]/255.0;
                    float g = src[j*3+1]/255.0;
                    float r = src[j*3+2]/255.0;
                    float num = (float)(0.5*((r-g)+(r-b)));
                    float den = (float)sqrt((r-g)*(r-g)+(r-b)*(g-b));
                    float H,S,I;
                    if(den == 0){	//分母不能为0
                            H = 0;
                    }
                    else{
                            double theta = acos(num/den);
                            if(b <= g)
                                    H = theta/(PI*2);
                            else
                                    H = (2*PI - theta)/(2*PI);
                    }
                    float minRGB = min(min(r,g),b);
                    den = r+g+b;
                    if(den == 0)	//分母不能为0
                            S = 0;
                    else
                            S = 1 - 3*minRGB/den;
                    I = den/3.0;
                    //将S分量和H分量都扩充到[0,255]区间以便于显示;
                    //一般H分量在[0,2pi]之间，S在[0,1]之间
                    dst[3*j] = H*255;
                    dst[3*j+1] = S*255;
                    dst[3*j+2] = I*255;
            }
    }
    return 0;
}
int hsi2rgb(Mat &hsi,Mat &image){
    if(!hsi.data){
            return -1;
    }
    int nl = hsi.rows;
    int nc = hsi.cols;
    //如果图像连续的话将图像数据多维数组转换成一维数组表示
    if(hsi.isContinuous()){
            nc = nc*nl;
            nl = 1;
    }


    for(int i = 0;i < nl;i++){
            uchar *src = hsi.ptr<uchar>(i);//hsi.ptr可以得到图像任意行的首地址
            uchar *det = image.ptr<uchar>(i);
            for(int j = 0;j < nc;j++){
                    float h = src[j*3]/255.0;
                    float s = src[j*3+1]/255.0;
                    float i = src[j*3+2]/255.0;
                    h=h*2*PI;
                    float R,G,B;
                    double theta = cos(PI/3-h);

                    if(h<=0&&h<2*PI/3){	//分母不能为0
                        if(theta==0)
                            R=0;
                        else
                            R=i*(1+(s*cos(h))/cos(PI/3-h));
                        B =i*(1-s) ;
                        G=3*i-(R+B);
                     }
                    else if(2*PI/3<=h&&h<4*PI/3){
                       h=h-2*PI/3;
                       if(theta==0)
                           G=0;
                       else
                           G=i*(1+(s*cos(h))/cos(PI/3-h));
                       R=i*(1-s);
                       B=3*i-(R+G);

                    }
                    else if(4*PI/3<=h&&h<2*PI){
                       h=h-4*PI/3;
                       if(theta==0)
                           B=0;
                       else
                            B=i*(1+(s*cos(h))/cos(PI/3-h));
                       G=i*(1-s);
                       R=3*i-(B+G);
                     }

                    //将S分量和H分量都扩充到[0,255]区间以便于显示;
                    //一般H分量在[0,2pi]之间，S在[0,1]之间
//                        namedWindow("4",CV_WINDOW_NORMAL);
//                        imshow("4", hsi);
//                        namedWindow("3",CV_WINDOW_NORMAL);
//                        imshow("3", image);
                    det[3*j] = B*255;
                    det[3*j+1] = G*255;
                    det[3*j+2] = R*255;
            }
    }
    return 0;
}
void RGB2HSI(Mat src, Mat &dst) {
    Mat HSI(src.rows, src.cols, CV_32FC3);
    float r, g, b, H, S, I, num, den, theta, sum, min_RGB;
    for (int i = 0; i<src.rows; i++)
    {
        for (int j = 0; j<src.cols; j++)
        {
            b = src.at<Vec3b>(i, j)[0];
            g = src.at<Vec3b>(i, j)[1];
            r = src.at<Vec3b>(i, j)[2];

            num = 0.5 * ((r - g) + (r - b));
            den = sqrt((r - g)*(r - g) + (r - b)*(g - b));

            if (den == 0) {
                H = 0; // 分母不能为0
            }
            else {
                theta = acos(num / den);
                if (b <= g) {
                    H = theta;
                }
                else {
                    H = (2 * PI - theta);
                }
            }

            min_RGB = min(min(b, g), r); // min(R,G,B)
            sum = b + g + r;
            if (sum == 0)
            {
                S = 0;
            }
            else {
                S = 1 - 3 * min_RGB / sum;
            }

            I = sum / 3.0;

            HSI.at<Vec3f>(i, j)[0] = H;
            HSI.at<Vec3f>(i, j)[1] = S;
            HSI.at<Vec3f>(i, j)[2] = I;
        }
    }
    dst = HSI;
    return;
}
void HSI2RGB(Mat src, Mat &dst) {
    Mat RGB = Mat::zeros(src.size(), CV_32FC3);
    for (int i = 0;i < src.rows;i++) {
        for (int j = 0;j < src.cols;j++) {
            float DH = src.at<Vec3f>(i, j)[0];
            float DS = src.at<Vec3f>(i, j)[1];
            float DI = src.at<Vec3f>(i, j)[2];
            //分扇区显示
            float R, G, B;
            if (DH < (2 * PI / 3) && DH >= 0) {
                B = DI * (1 - DS);
                R = DI * (1 + (DS * cos(DH)) / cos(PI / 3 - DH));
                G = (3 * DI - (R + B));
            }
            else if (DH < (4 * PI / 3) && DH >= (2 * PI / 3)) {
                DH = DH - (2 * PI / 3);
                R = DI * (1 - DS);
                G = DI * (1 + (DS * cos(DH)) / cos(PI / 3 - DH));
                B = (3 * DI - (G + R));
            }
            else {
                DH = DH - (4 * PI / 3);
                G = DI * (1 - DS);
                B = DI * (1 + (DS * cos(DH)) / cos(PI / 3 - DH));
                R = (3 * DI - (G + B));
            }
            RGB.at<Vec3f>(i, j)[0] = B;
            RGB.at<Vec3f>(i, j)[1] = G;
            RGB.at<Vec3f>(i, j)[2] = R;
        }
    }
    dst = RGB;
    return;
}
void harr_fusion(Mat src1, Mat src2,Mat &dst) {
    assert(src1.rows == src2.rows&&src1.cols == src2.cols);
    int row = src1.rows;
    int col = src1.cols;
    Mat src1_gray, src2_gray;
    normalize(src1, src1_gray, 0, 255, CV_MINMAX);
    cvtColor(src2, src2_gray, CV_RGB2GRAY);
    normalize(src2_gray, src2_gray, 0, 255, CV_MINMAX);
    src1_gray.convertTo(src1_gray, CV_32F);
    src2_gray.convertTo(src2_gray, CV_32F);
    WaveTransform m_waveTransform;
    const int level = 2;
    Mat src1_dec = m_waveTransform.WDT(src1_gray, "haar", level);
    Mat src2_dec = m_waveTransform.WDT(src2_gray, "haar", level);
    Mat dec = Mat(row,col, CV_32FC1);
    //融合规则：高频部分采用模值取大的方法，低频部分采用加权平均的方法
    int halfRow = row / (2 * level);
    int halfCol = col / (2 * level);
    for (int i = 0;i < row;i++) {
        for (int j = 0;j < col;j++) {
            if (i > halfRow&&j > halfCol) {
                float p = abs(src1_dec.at<float>(i, j));
                float q = abs(src2_dec.at<float>(i, j));
                if (p > q) {
                    dec.at<float>(i, j) = src1_dec.at<float>(i, j);
                }
                else {
                    dec.at<float>(i, j) = src2_dec.at<float>(i, j);
                }
            }
            else {
                dec.at<float>(i, j) = (src1_dec.at<float>(i, j) + src2_dec.at<float>(i, j)) / 2;
            }
        }
    }
    dst = m_waveTransform.IWDT(dec, "haar", level);
}




void sar_fusion(Mat src1, Mat src2,Mat &dst) {
    assert(src1.rows == src2.rows&&src1.cols == src2.cols);
    int row = src1.rows;
    int col = src1.cols;
    Mat src1_gray, src2_gray;
    normalize(src1, src1_gray, 0, 255, CV_MINMAX);
    cvtColor(src2, src2_gray, CV_RGB2GRAY);
    normalize(src2_gray, src2_gray, 0, 255, CV_MINMAX);
    src1_gray.convertTo(src1_gray, CV_32F);
    src2_gray.convertTo(src2_gray, CV_32F);
    WaveTransform m_waveTransform;
    const int level = 1;
    Mat src1_dec = m_waveTransform.WDT(src1_gray, "haar", level);
    Mat src2_dec = m_waveTransform.WDT(src2_gray, "haar", level);
    Mat dec = Mat(row,col, CV_32FC1);
    //融合规则：高频部分采用模值取大的方法，低频部分采用加权平均的方法
    int halfRow = row / (2 * level);
    int halfCol = col / (2 * level);
    for (int i = 0;i < row;i++) {
        for (int j = 0;j < col;j++) {
            if (i < halfRow&&j < halfCol) {
                float p = abs(src1_dec.at<float>(i, j));
                float q = abs(src2_dec.at<float>(i, j));
                if (p > q) {
                    dec.at<float>(i, j) = src1_dec.at<float>(i, j);
                }
                else {
                    dec.at<float>(i, j) = src2_dec.at<float>(i, j);
                }
            }
            else {

                dec.at<float>(i, j) = (src1_dec.at<float>(i, j) + src2_dec.at<float>(i, j)) / 2;
            }
        }
    }
    dst = m_waveTransform.IWDT(dec, "haar", level);
}


//高斯核
Mat get2DGaussianKernel(int ksize, double sigma, int ktype){
    Mat kernel_1d = getGaussianKernel(ksize, sigma, ktype);

    return kernel_1d*(Mat)(kernel_1d.t());
}

//二维数组转换为Mat
template <typename T> Mat matrix2Mat(T **matrix, int dtype){
    int row = sizeof(matrix)/sizeof(matrix[0]);
    int col = sizeof(matrix[0])/sizeof(matrix[0][0]);

    Mat mat(row, col, dtype);
    for(int i=0;i<row;i++)
        for(int j=0;j<col;j++)
            mat.at<T>(i,j) = matrix[i][j];

    return mat;
}
//Mat转换为数组
template <typename T> T ** matrix2Mat(Mat mat, int dtype){
    Mat dst(mat.size);
    mat.convertTo(dst, dtype, 1,0);
    T *ptrDst[dst.rows];

    for(int i=0;i<dst.rows;i++)
        ptrDst[i] = dst.ptr<T>(i);

    return ptrDst;
}
