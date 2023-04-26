#include "uidemo.h"
#include "ui_uidemo.h"
#include <math.h>
#include "iconhelper.h"
#include <QApplication>
#include <QObject>
#include <QMenu>
#include <QScreen>
#include <QMessageBox>
#include <QFileDialog>
#include <QPixmap>
#include <QImage>
#include <QDateTime>
#include <stdio.h>
#include <QDebug>
#include <QtCore/QCoreApplication>
#include <iostream>
#include <iomanip>
#include <QThread>
#include <QtConcurrent>
#include <QFuture>
#include <QProgressDialog>
#include <QAxWidget>
#include <QAxObject>
#include <QPrinter>
#include "utils.h"
#include "wavetransform.h"
#include <vector>


using namespace cv;
using namespace std;

QImage image1,image2,image3,image4,transform1;  //image3,image4
Mat reff,src,image11,image22,image33,image44,full_src,src_recover;  //image33,image44
Mat merge11,merge12;  //merge12
Mat ref_win, src_win;
vector<Point2f> imagePoints1, imagePoints2;
extern const string& win1="register",win2="base";

UIdemo::UIdemo(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::UIdemo)
{
    ui->setupUi(this);
    this->initForm();
    ui->edgeDetectSaveBtn->setVisible(true);
    ui->registersave->setVisible(true);

    // 读取神经网络模型
//    bldg_net = cv::dnn::readNetFromTensorflow("D:/c++_project/data_project/cache/bldg_model_ep20.pb");
//    water_net = cv::dnn::readNetFromTensorflow("D:/c++_project/data_project/cache/water_model_ep10.pb");
//    vegetation_net = cv::dnn::readNetFromTensorflow("D:/c++_project/data_project/cache/vegetation_model_ep10.pb");
//    road_net = cv::dnn::readNetFromTensorflow("D:/c++_project/data_project/cache/road_model_ep20.pb");
    deeplab_net = torch::jit::load("D:/model_libtorch/traced_deeplab_v3_pytorch_model3.pt");
//    deeplab_net.to(at::kCUDA);


    // 对图像分类模块的四类赋予RGB值
    colors.push_back(cv::Vec3b());
    for (int i = 0; i < 4; ++i)
    {
        cv::Vec3b color;
        if (i == 0)
        {
            color[0] = 159;
            color[1] = 255;
            color[2] = 84;
        }
        else if (i == 1)
        {
            color[0] = 34;
            color[1] = 180;
            color[2] = 238;
        }
        else if (i == 2)
        {
            color[0] = 255;
            color[1] = 191;
            color[2] = 0;
        }
        else if (i == 3)
        {
            color[0] = 38;
            color[1] = 71;
            color[2] = 139;
        }
        colors.push_back(color);
    }
    // 赋予映射，deeplab
    for(int i=0; i<6; i++){
        cv::Vec3b color;
        switch(i){
        case 0:
            color[0] = 255;
            color[1] = 255;
            color[2] = 255;
            break;
        case 1:
            color[0] = 255;
            color[1] = 0;
            color[2] = 0;
            break;
        case 2:
            color[0] = 0;
            color[1] = 255;
            color[2] = 0;
            break;
        case 3:
            color[0] = 255;
            color[1] = 255;
            color[2] = 0;
            break;
        case 4:
            color[0] = 0;
            color[1] = 255;
            color[2] = 255;
            break;
        case 5:
            color[0] = 0;
            color[1] = 0;
            color[2] = 255;
            break;

        }
        colors_deeplab.push_back(color);
    }

}

UIdemo::~UIdemo()
{
    delete ui;
}

bool UIdemo::eventFilter(QObject *watched, QEvent *event)
{
    if (event->type() == QEvent::MouseButtonDblClick)
    {
        if (watched == ui->widgetTitle)
        {
            on_btnMenu_Max_clicked();
            return true;
        }
    }

    return QWidget::eventFilter(watched, event);
}

void UIdemo::initForm()
{
    this->setProperty("form", true);
    this->setProperty("canMove", true);
    this->setWindowFlags(Qt::FramelessWindowHint | Qt::WindowSystemMenuHint | Qt::WindowMinMaxButtonsHint);     //隐藏界面边框,最小最大化按钮等.

    //IconHelper::Instance()->setIcon(ui->labIco, QChar(0xf099), 30);                 //设置主窗口左上角图标
    IconHelper::Instance()->setIcon(ui->btnMenu_Min, QChar(0xf068));
    IconHelper::Instance()->setIcon(ui->btnMenu_Max, QChar(0xf067));
    IconHelper::Instance()->setIcon(ui->btnMenu_Close, QChar(0xf00d));

    //ui->widgetMenu->setVisible(false);
    ui->widgetTitle->installEventFilter(this);
    ui->widgetTitle->setProperty("form", "title");
    ui->widgetTop->setProperty("nav", "top");
    ui->labTitle->setText(u8"多源遥感数据融合平台");
    ui->labTitle->setFont(QFont("Microsoft Yahei", 12));
    this->setWindowTitle(ui->labTitle->text());

    ui->stackedWidget->setStyleSheet("QLabel{font:10pt;}");

    //设置指示器大小
    int addWidth = 20;
    int addHeight = 10;
    int rWidth = 15;
    int ckWidth = 13;
    int scrWidth = 12;
    int borderWidth = 3;

    QStringList qss;
    qss.append(QString("QComboBox::drop-down,QDateEdit::drop-down,QTimeEdit::drop-down,QDateTimeEdit::drop-down{width:%1px;}").arg(addWidth));
    qss.append(QString("QComboBox::down-arrow,QDateEdit[calendarPopup=\"true\"]::down-arrow,QTimeEdit[calendarPopup=\"true\"]::down-arrow,"
                       "QDateTimeEdit[calendarPopup=\"true\"]::down-arrow{width:%1px;height:%1px;right:2px;}").arg(addHeight));
    qss.append(QString("QRadioButton::indicator{width:%1px;height:%1px;}").arg(rWidth));
    qss.append(QString("QCheckBox::indicator,QGroupBox::indicator,QTreeWidget::indicator,QListWidget::indicator{width:%1px;height:%1px;}").arg(ckWidth));
    qss.append(QString("QScrollBar:horizontal{min-height:%1px;border-radius:%2px;}QScrollBar::handle:horizontal{border-radius:%2px;}"
                       "QScrollBar:vertical{min-width:%1px;border-radius:%2px;}QScrollBar::handle:vertical{border-radius:%2px;}").arg(scrWidth).arg(scrWidth / 2));
    qss.append(QString("QWidget#widget_top>QToolButton:pressed,QWidget#widget_top>QToolButton:hover,"
                       "QWidget#widget_top>QToolButton:checked,QWidget#widget_top>QLabel:hover{"
                       "border-width:0px 0px %1px 0px;}").arg(borderWidth));
    qss.append(QString("QWidget#widgetleft>QPushButton:checked,QWidget#widgetleft>QToolButton:checked,"
                       "QWidget#widgetleft>QPushButton:pressed,QWidget#widgetleft>QToolButton:pressed{"
                       "border-width:0px 0px 0px %1px;}").arg(borderWidth));
    this->setStyleSheet(qss.join(""));

    QSize icoSize(32, 32);
    int icoWidth = 10;

    //设置顶部导航按钮
    QList<QToolButton *> tbtns = ui->widgetTop->findChildren<QToolButton *>();
    foreach (QToolButton *btn, tbtns)
    {
        btn->setIconSize(icoSize);
        btn->setMinimumWidth(icoWidth);
        btn->setCheckable(true);
        connect(btn, SIGNAL(clicked()), this , SLOT(buttonClick()));

    }

    //ui->_1->click();
}

void UIdemo::buttonClick()
{
    QToolButton *b = (QToolButton *)sender();
    QString name = b->text();

    QList<QToolButton *> tbtns = ui->widgetTop->findChildren<QToolButton *>();
    foreach (QToolButton *btn, tbtns)
    {
        if (btn == b) {
            btn->setChecked(true);
        } else {
            btn->setChecked(false);
        }
    }


    if (name == "图像配准")
    {
        ui->stackedWidget->setCurrentIndex(0);
    } else if (name == "SAR图像融合")
    {
        ui->stackedWidget->setCurrentIndex(1);
    } else if (name == "红外图像融合")
    {
        ui->stackedWidget->setCurrentIndex(2);
    } else if (name == "地物分类")
    {
        ui->stackedWidget->setCurrentIndex(3);
    } else if (name == "洪涝检测")
    {
        ui->stackedWidget->setCurrentIndex(4);
    } else if (name == "目标检测")
    {
        ui->stackedWidget->setCurrentIndex(5);
    } else if (name == "综合评估")
    {
        ui->stackedWidget->setCurrentIndex(6);
    }
}

void UIdemo::on_btnMenu_Min_clicked()
{
    showMinimized();
}



void UIdemo::on_btnMenu_Max_clicked()
{
    static bool max = false;
    static QRect location = this->geometry();

    if (max)
    {
        this->setGeometry(location);
    }
    else
    {
        location = this->geometry();
        this->setGeometry(qApp->desktop()->availableGeometry());
    }

    this->setProperty("canMove", max);
    max = !max;
}


void UIdemo::on_btnMenu_Close_clicked()
{
    close();
}



// 图像配准-图像选择1
void UIdemo::on_registerImageSelect1_clicked()
{
    QString open_fileName;
    open_fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), ".",
                                                 tr("Image Files (*.png *.jpg *.bmp)"));
    if (open_fileName.isEmpty())
    {
        return;
    }
    else
    {
        QImage img;
        if (!(img.load(open_fileName)))         //加载图像
        {
            QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("打开图像失败").toStdString().c_str(), QString::fromLocal8Bit("打开图像失败!").toStdString().c_str());
            mess.setStyleSheet("background-color: black");
            mess.exec();
            return;
        }
        ui->register1->setPixmap(QPixmap::fromImage(img.scaled(ui->register1->size())));
        register_oriImg_1 = img;
    }
}

// 图像配准-图像选择2
void UIdemo::on_registerImageSelect2_clicked()
{
    QString open_fileName;
    open_fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), ".",
                                                 tr("Image Files (*.png *.jpg *.bmp)"));
    if (open_fileName.isEmpty())
    {
        return;
    }
    else
    {
        QImage img;
        if (!(img.load(open_fileName)))         //加载图像
        {
            QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("打开图像失败").toStdString().c_str(), QString::fromLocal8Bit("打开图像失败!").toStdString().c_str());
            mess.setStyleSheet("background-color: black");
            mess.exec();
            return;
        }
        ui->register2->setPixmap(QPixmap::fromImage(img.scaled(ui->register2->size())));
        register_oriImg_2 = img;
    }
}

// 图像配准-配准操作
void UIdemo::on_imageRegistration_clicked()
{
    if(register_oriImg_1.isNull()||register_oriImg_2.isNull())
    {
        QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("操作失败").toStdString().c_str(), QString::fromLocal8Bit("请正确输入待配准图像").toStdString().c_str());
        mess.setStyleSheet("background-color: black");
        mess.exec();
        return;
    }

    //ui->imageRegistration->setEnabled(false);
    //QApplication::processEvents();

    reff=QImageToMat(register_oriImg_1);
    src=QImageToMat(register_oriImg_2);

    //    //图片高斯滤波处理
    GaussianBlur(reff,reff,Size(3,3),0,0);
    GaussianBlur(src,src,Size(3,3),0,0);

    ref_win = reff.clone();
    src_win = src.clone();


    namedWindow(win1,WINDOW_NORMAL);
    imshow(win1, src_win);
    namedWindow(win2,WINDOW_NORMAL);
    imshow(win2, ref_win);
    //提取特征点
    setMouseCallback(win1, on_mouse2);
    setMouseCallback(win2, on_mouse1);

    char ckey = waitKey();
    if(ckey==0){
        return;
    }

    destroyWindow(win1);
    destroyWindow(win2);

    //求变换矩阵,相当于实现了findhomography功能
    int n=0,m=0,k=0,i=0,j=0,n1=0;
    n = imagePoints1.size();
    n1=imagePoints2.size();
    if(n!=n1){
        QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("操作失败").toStdString().c_str(), QString::fromLocal8Bit("选取的特征点应该相互匹配！").toStdString().c_str());
        imagePoints1.clear();
        imagePoints2.clear();
        mess.setStyleSheet("background-color: black");
        mess.exec();
        return;
    }

    // 定义A矩阵并初始化A为全零矩阵
//    double A[2*n][9];
    double **A = (double**)malloc(sizeof(double*)*2*n);
    for(i=0;i<2*n;i++) A[i] = (double*)malloc(sizeof(double)*9);
    for(i=0;i<2*n;i++)
    {
        for(j=0;j<9;j++)
        {
            A[i][j] = 0;
        }
    }
    //double linex1[n],linex2[n],liney1[n],liney2[n];
    double *linex1 = (double *)malloc(sizeof(double)*n);
    double *liney1 = (double *)malloc(sizeof(double)*n);
    double *linex2 = (double *)malloc(sizeof(double)*n);
    double *liney2 = (double *)malloc(sizeof(double)*n);
    for(i=0;i<n;i++)
    {
        linex1[i] = imagePoints2[i].x;
        liney1[i] = imagePoints2[i].y;
        linex2[i] = imagePoints1[i].x;
        liney2[i] = imagePoints1[i].y;
    }

    j=0;
    for(i=0;i<2*n;i=i+2)
    {
        A[i][0] = imagePoints2[j].x;
        j = j + 1;
    }

    j=0;

    for(i=0;i<2*n;i=i+2)
    {
        A[i][1] = imagePoints2[j].y;
        j = j + 1;
    }

    for(i=0;i<2*n;i=i+2)
    {
        A[i][2] = 1;
    }

    j=0;

    for(i=1;i<=2*n;i=i+2)
    {
        A[i][3] = imagePoints2[j].x;
        j = j + 1;
    }

    j=0;

    for(i=1;i<=2*n;i=i+2)
    {
        A[i][4] = imagePoints2[j].y;
        j = j + 1;
    }

    for(i=1;i<=2*n;i=i+2)
    {
        A[i][5] = 1;
    }

    j=0;

    for(i=0;i<2*n;i=i+2)
    {
        A[i][6] = -linex2[j]*linex1[j];
        j = j + 1;
    }


    j=0;

    for(i=1;i<=2*n;i=i+2)
    {
        A[i][6] = -liney2[j]*linex1[j];
        j = j + 1;
    }

    j=0;

    for(i=0;i<2*n;i=i+2)
    {
        A[i][7] = -linex2[j]*liney1[j];
        j = j + 1;
    }

    j=0;

    for(i=1;i<=2*n;i=i+2)
    {
        A[i][7] = -liney2[j]*liney1[j];
        j = j + 1;
    }

    j=0;

    for(i=0;i<2*n;i=i+2)
    {
        A[i][8] = -linex2[j];
        j = j + 1;
    }

    j=0;

    for(i=1;i<=2*n;i=i+2)
    {
        A[i][8] = -liney2[j];
        j = j + 1;
    }

    //double B[9][2*n];
    double **B = (double**)malloc(sizeof(double*)*9);
    for(i=0;i<9;i++) B[i] = (double *)malloc(sizeof(double)*2*n);

    for(i=0;i<9;i++)
    {
        for(j=0;j<2*n;j++)
        {
            B[i][j] = 0;
        }
    }

    for(i=0;i<2*n;i++)
    {
        for(j=0;j<9;j++)
        {
            B[j][i] = A[i][j];
        }
    }

    double C[9][9];

    for(i=0;i<9;i++)
    {
        for(j=0;j<9;j++)
        {
            C[i][j] = 0;
        }
    }

    for(i=0;i<9;i++)
    {
        for(j=0;j<9;j++)
        {
            for(m=0;m<2*n;m++)
            {
                C[i][j] += B[i][m] * A[m][j];
            }
        }
    }

    Mat myMat = Mat(9, 9, CV_64FC1, C);
    Mat eValuesMat;
    Mat eVectorsMat;

    eigen(myMat, eValuesMat, eVectorsMat);

    double H[3][3];
    k=0;

    for(i=0;i<3;i++)
    {
        for(j=0;j<3;j++)
        {
            H[i][j] = eVectorsMat.at<double>(8,k);
            k = k + 1;
        }
    }

    for(i=0;i<3;i++)
    {
        for(j=0;j<3;j++)
        {
            H[i][j] = H[i][j] / H[2][2];

        }
    }

    Mat homo = Mat(3, 3, CV_64FC1, H);

    Mat imageTransform1;
    warpPerspective(src, imageTransform1, homo, Size(reff.cols, reff.rows));   //变换
    transform1=MatToQImage(imageTransform1);
    ui->register3->setPixmap(QPixmap::fromImage( transform1.scaled(ui->register3->size())));

    ui->label_10->setPixmap(QPixmap::fromImage( transform1.scaled(ui->label_10->size())));
    //            ui->merge2->setPixmap(QPixmap::fromImage(transform1.scaled(ui->merge2->size())));
    //            image2=transform1;
    waitKey(0);
    //ui->imageRegistration->setEnabled(true);
}

// 图像配准-保存
void UIdemo::on_registersave_clicked()
{
    QString filename = QFileDialog::getSaveFileName(this,
                                                    tr("Save Image"),
                                                    "",
                                                    tr("*.bmp;; *.png;; *.jpg;; *.tif;; *.GIF")); //选择路径
    if (filename.isEmpty())
    {
        return;
    }
    else
    {
        if (!(transform1.save(filename))) //保存图像
        {
            QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("操作失败").toStdString().c_str(), QString::fromLocal8Bit("图像为空").toStdString().c_str());
            mess.setStyleSheet("background-color: black");
            mess.exec();
            return;
        }
        QMessageBox mess1(QMessageBox::Information, QString::fromLocal8Bit("保存").toStdString().c_str(), QString::fromLocal8Bit("图片保存成功").toStdString().c_str());
        mess1.setStyleSheet("background-color: black");
        mess1.exec();
    }
}


// 光学图像融合-图像选择1
void UIdemo::on_fusionImageSelect1_clicked()
{
    QString open_fileName;
    open_fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), ".",
                                                 tr("Image Files (*.png *.jpg *.bmp *.tif)"));
    if (open_fileName.isEmpty())
    {
        return;
    }
    else
    {
        //QImage img;
        if (!(image1.load(open_fileName)))         //加载图像
        {
            QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("打开图像失败").toStdString().c_str(), QString::fromLocal8Bit("打开图像失败!").toStdString().c_str());
            mess.setStyleSheet("background-color: black");
            mess.exec();
            return;
        }
        ui->merge1->setPixmap(QPixmap::fromImage(image1.scaled(ui->merge1->size())));
    }
}

// 光学图像融合-图像选择2
void UIdemo::on_fusionImageSelect2_clicked()
{
    QString open_fileName;
    open_fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), ".",
                                                 tr("Image Files (*.png *.jpg *.bmp *.tif)"));
    if (open_fileName.isEmpty())
    {
        return;
    }
    else
    {
        //QImage img;
        if (!(image2.load(open_fileName)))         //加载图像
        {
            QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("打开图像失败").toStdString().c_str(), QString::fromLocal8Bit("打开图像失败!").toStdString().c_str());
            mess.setStyleSheet("background-color: black");
            mess.exec();
            return;
        }
        ui->merge2->setPixmap(QPixmap::fromImage(image2.scaled(ui->merge2->size())));
    }
}

// 光学图像融合-融合操作
QImage UIdemo::imageFusion()
{
//    image11=QImageToMat(image1);
//    image22=QImageToMat(image2);

//    //高斯滤波
//    GaussianBlur(image11,image11,Size(3,3),0,0);
//    GaussianBlur(image22,image22,Size(3,3),0,0);

//    Mat img_hsi,I,hsi,dst;

//    vector <Mat> vecHsi;

//    img_hsi.create(image11.rows,image11.cols,CV_8UC3);//创建一个图像矩阵容器
//    merge11.create(image11.rows,image11.cols,CV_8UC3);
//    //namedWindow("image11",CV_WINDOW_NORMAL);
//    //imshow("image11",image11);

//    rgb2hsi(image11,img_hsi);//hsi变换，将图片从rgb颜色模型转到hsi颜色模型

//    split(img_hsi,vecHsi);//将图像切割成单通道

//    I=vecHsi[2].clone();

//    //调整ahand的大小与ac的大小一致，融合函数addWeighted()要求输入的两个图形尺寸必须相同
//    cv::resize(image22, image22, Size(I.cols, I.rows));

//    dst=I;
//    laplace_decompose(dst,3, full_src);//此处变换出来的图片存在问题，左上角缺了一块
//    ware_operate(full_src, 3);//对小波分解后的SAR图像进行小波操作

//    // Mat src_recover;
//    wave_recover(full_src, src_recover, 3);//进对SAR图像进行小波复原

//    vecHsi[2]=src_recover.clone();

//    merge(vecHsi,hsi);

//    hsi2rgb(hsi,merge11); //进行hsi逆变换

    image11=QImageToMat(image1);
    image22=QImageToMat(image2);

    //高斯滤波
    GaussianBlur(image11,image11,Size(3,3),0,0);
    GaussianBlur(image22,image22,Size(3,3),0,0);

    Mat HSI(image11.size(), CV_32FC3);
    Mat Visible_I(image11.size(), CV_32FC1);
    RGB2HSI(image11, HSI); //hsi变换，将图片从rgb颜色模型转到hsi颜色模型
    for (int i = 0;i < image11.rows;i++) {
        for (int j = 0;j < image11.cols;j++) {
            Visible_I.at<float>(i, j) = HSI.at<Vec3f>(i, j)[2];
        }
    }
    Mat fusion_I;
    sar_fusion(Visible_I, image22, fusion_I);
    Mat fusion_dst = Mat::zeros(image11.size(), CV_32FC3);
    for (int i = 0;i < image11.rows;i++) {
        for (int j = 0;j < image11.cols;j++) {
            fusion_dst.at<Vec3f>(i, j)[2] = fusion_I.at<float>(i, j);
            fusion_dst.at<Vec3f>(i, j)[0] = HSI.at<Vec3f>(i, j)[0];
            fusion_dst.at<Vec3f>(i, j)[1] = HSI.at<Vec3f>(i, j)[1];
        }
    }
    Mat merge12;
    HSI2RGB(fusion_dst, merge11); //进行hsi逆变换
    merge11.convertTo(merge11, CV_8UC3);
    normalize(merge11, merge11, 0, 255, CV_MINMAX);

    Grad_ave = gradsAvg(merge11);//融合图像的平均梯度计算

    wrap = wrapcom(image11,merge11);//融合图像相对于光学图像扭曲程度计算

    entropy = Entropy(merge11); //融合图像平均熵计算

    quality = Qulity(image11,merge11);  //计算光学图像和融合图像的通用图像质量指标
    mutual = comEntropy(merge11,image11,image22);  //融合图像与源图像之间交户信息量计算
    //       namedWindow("111",CV_WINDOW_NORMAL);
    //       imshow("111",merge11);
    QImage merge22= MatToQImage(merge11);
    //融合图像及相关性能分析数据显示
    ui->entropy_label->setNum(entropy);
    ui->entropy_label_3->setNum(entropy);
    ui->Grad_ave->setNum(Grad_ave);
    ui->Grad_ave_3->setNum(Grad_ave);
    ui->mutual->setNum(mutual);
    ui->mutual_3->setNum(mutual);
    ui->wrap->setNum(wrap);
    ui->wrap_3->setNum(wrap);
    ui->quality->setNum(quality);
    ui->quality_3->setNum(quality);

    return merge22;
}

// 光学图像融合-结果展示
void UIdemo::on_imageFusionButton_clicked()
{
    if(image1.isNull()||image2.isNull())
    {
        QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("操作失败").toStdString().c_str(), QString::fromLocal8Bit("请选择要融合的图像").toStdString().c_str());

        mess.setStyleSheet("background-color: black");
        mess.exec();
        return;
    }
    else if(image1.width() != image2.width() || image1.height() != image2.height()){
        QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("操作失败").toStdString().c_str(), QString::fromLocal8Bit("选择融合图像不正确").toStdString().c_str());
        mess.setStyleSheet("background-color: black");
        mess.exec();
        return;
    }
    QFuture<QImage> future = QtConcurrent::run(this, &UIdemo::imageFusion);

    if (future.isRunning())
    {
        QProgressDialog process(this);
        process.setLabelText(tr("processing..."));
        QFont font("ZYSong 18030", 12);
        process.setFont(font);
        process.setWindowTitle(tr("Please Wait"));
        // process.setWindowFlags(Qt::FramelessWindowHint);
        process.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint);

        process.setRange(0, 100000);
        process.setCancelButton(nullptr);
        process.setModal(true);
        // process.setCancelButtonText(tr("cancel"));

        for (int i = 0; i < 100000 && future.isRunning(); i++)
        {
            for (int j = 0; j < 20000; j++)
            {
                process.setValue(i);
            }

            if (i == 99999)
            {
                i = 0;
            }
            QCoreApplication::processEvents();
        }
    }
    fusion_Img = future.result();
    ui->merge3->setPixmap(QPixmap::fromImage(fusion_Img.scaled(ui->merge3->size())));
    ui->label5_1->setPixmap(QPixmap::fromImage(fusion_Img.scaled(ui->label5_1->size())));
}

// 光学图像融合-保存
void UIdemo::on_fusionsave_clicked()
{
    QString filename = QFileDialog::getSaveFileName(this,
                                                    tr("Save Image"),
                                                    "",
                                                    tr("*.bmp;; *.png;; *.jpg;; *.tif;; *.GIF")); //选择路径
    if (filename.isEmpty())
    {
        return;
    }
    else
    {
        if (!(fusion_Img.save(filename))) //保存图像
        {
            QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("操作失败").toStdString().c_str(), QString::fromLocal8Bit("图片为空").toStdString().c_str());
            mess.setStyleSheet("background-color: black");
            mess.exec();
            return;
        }
        // ui.statusBar->showMessage("图片保存成功");
        QMessageBox mess1(QMessageBox::Information, QString::fromLocal8Bit("保存").toStdString().c_str(), QString::fromLocal8Bit("图片保存成功").toStdString().c_str());
        mess1.setStyleSheet("background-color: black");
        mess1.exec();
    }
}

//红外图像融合-图像选择1
void UIdemo::on_fusionImageSelect1_2_clicked()
{
    QString open_fileName;
    open_fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), ".",
                                                 tr("Image Files (*.png *.jpg *.bmp *.tif)"));
    if (open_fileName.isEmpty())
    {
        return;
    }
    else
    {
        //QImage img;
        if (!(image3.load(open_fileName)))         //加载图像
        {
            QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("打开图像失败").toStdString().c_str(), QString::fromLocal8Bit("打开图像失败!").toStdString().c_str());
            mess.setStyleSheet("background-color: black");
            mess.exec();
            return;
        }
        ui->merge1_2->setPixmap(QPixmap::fromImage(image3.scaled(ui->merge1_2->size())));
    }
}


//红外图像融合-图像选择2
void UIdemo::on_fusionImageSelect2_2_clicked()
{
    QString open_fileName;
    open_fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), ".",
                                                 tr("Image Files (*.png *.jpg *.bmp *.tif)"));
    if (open_fileName.isEmpty())
    {
        return;
    }
    else
    {
        //QImage img;
        if (!(image4.load(open_fileName)))         //加载图像
        {
            QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("打开图像失败").toStdString().c_str(), QString::fromLocal8Bit("打开图像失败!").toStdString().c_str());
            mess.setStyleSheet("background-color: black");
            mess.exec();
            return;
        }
        ui->merge2_2->setPixmap(QPixmap::fromImage(image4.scaled(ui->merge2_2->size())));
    }
}

//红外图像融合-融合操作
QImage UIdemo::imageFusion1()
{
    image33=QImageToMat(image3);
    image44=QImageToMat(image4);

    //高斯滤波
    GaussianBlur(image33,image33,Size(3,3),0,0);
    GaussianBlur(image44,image44,Size(3,3),0,0);

    Mat HSI(image33.size(), CV_32FC3);
    Mat Visible_I(image33.size(), CV_32FC1);
    RGB2HSI(image33, HSI); //hsi变换，将图片从rgb颜色模型转到hsi颜色模型
    for (int i = 0;i < image33.rows;i++) {
        for (int j = 0;j < image33.cols;j++) {
            Visible_I.at<float>(i, j) = HSI.at<Vec3f>(i, j)[2];
        }
    }
    Mat fusion_I;
    harr_fusion(Visible_I, image44, fusion_I);
    Mat fusion_dst = Mat::zeros(image33.size(), CV_32FC3);
    for (int i = 0;i < image33.rows;i++) {
        for (int j = 0;j < image33.cols;j++) {
            fusion_dst.at<Vec3f>(i, j)[2] = fusion_I.at<float>(i, j);
            fusion_dst.at<Vec3f>(i, j)[0] = HSI.at<Vec3f>(i, j)[0];
            fusion_dst.at<Vec3f>(i, j)[1] = HSI.at<Vec3f>(i, j)[1];
        }
    }
    Mat merge12;
    HSI2RGB(fusion_dst, merge12); //进行hsi逆变换
    merge12.convertTo(merge12, CV_8UC3);
    normalize(merge12, merge12, 0, 255, CV_MINMAX);

    Grad_ave = gradsAvg(merge12);//融合图像的平均梯度计算

    wrap = wrapcom(image33,merge12);//融合图像相对于光学图像扭曲程度计算

    entropy = Entropy(merge12); //融合图像平均熵计算

    quality = Qulity(image33,merge12);  //计算光学图像和融合图像的通用图像质量指标
    mutual = comEntropy(merge12,image33,image44);  //融合图像与源图像之间交户信息量计算
    //       namedWindow("111",CV_WINDOW_NORMAL);
    //       imshow("111",merge11);
    QImage merge23= MatToQImage(merge12);
    //融合图像及相关性能分析数据显示
    ui->entropy_label_2->setNum(entropy);
    ui->entropy_label_3->setNum(entropy);
    ui->Grad_ave_2->setNum(Grad_ave);
    ui->Grad_ave_3->setNum(Grad_ave);
    ui->mutual_2->setNum(mutual);
    ui->mutual_3->setNum(mutual);
    ui->wrap_2->setNum(wrap);
    ui->wrap_3->setNum(wrap);
    ui->quality_2->setNum(quality);
    ui->quality_3->setNum(quality);

    return merge23;
}


//红外图像融合-结果展示
void UIdemo::on_imageFusionButton_2_clicked()
{
    if(image3.isNull()||image4.isNull())
    {
        QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("操作失败").toStdString().c_str(), QString::fromLocal8Bit("请选择要融合的图像").toStdString().c_str());
        mess.setStyleSheet("background-color: black");
        mess.exec();
        return;
    }
    else if(image1.width() != image2.width() || image1.height() != image2.height()){
        QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("操作失败").toStdString().c_str(), QString::fromLocal8Bit("选择融合图像不正确").toStdString().c_str());
        mess.setStyleSheet("background-color: black");
        mess.exec();
        return;
    }
    QFuture<QImage> future = QtConcurrent::run(this, &UIdemo::imageFusion1);

    if (future.isRunning())
    {
        QProgressDialog process(this);
        process.setLabelText(tr("processing..."));
        QFont font("ZYSong 18030", 12);
        process.setFont(font);
        process.setWindowTitle(tr("Please Wait"));
        // process.setWindowFlags(Qt::FramelessWindowHint);
        process.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint);

        process.setRange(0, 100000);
        process.setCancelButton(nullptr);
        process.setModal(true);
        // process.setCancelButtonText(tr("cancel"));

        for (int i = 0; i < 100000 && future.isRunning(); i++)
        {
            for (int j = 0; j < 20000; j++)
            {
                process.setValue(i);
            }

            if (i == 99999)
            {
                i = 0;
            }
            QCoreApplication::processEvents();
        }
    }
    fusion_Img = future.result();
    ui->merge3_2->setPixmap(QPixmap::fromImage(fusion_Img.scaled(ui->merge3_2->size())));
    ui->label5_4->setPixmap(QPixmap::fromImage(fusion_Img.scaled(ui->label5_4->size())));
    //    ui->label5_1->setPixmap(QPixmap::fromImage(fusion_Img.scaled(ui->label5_1->size())));
}


//红外图像融合-保存
void UIdemo::on_fusionsave_2_clicked()
{
    QString filename = QFileDialog::getSaveFileName(this,
                                                    tr("Save Image"),
                                                    "",
                                                    tr("*.bmp;; *.png;; *.jpg;; *.tif;; *.GIF")); //选择路径
    if (filename.isEmpty())
    {
        return;
    }
    else
    {
        if (!(fusion_Img.save(filename))) //保存图像
        {
            QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("操作失败").toStdString().c_str(), QString::fromLocal8Bit("图片为空").toStdString().c_str());
            mess.setStyleSheet("background-color: black");
            mess.exec();
            return;
        }
        // ui.statusBar->showMessage("图片保存成功");
        QMessageBox mess1(QMessageBox::Information, QString::fromLocal8Bit("保存").toStdString().c_str(), QString::fromLocal8Bit("图片保存成功").toStdString().c_str());
        mess1.setStyleSheet("background-color: black");
        mess1.exec();
    }
}


// 图像分类-图像选择
void UIdemo::on_classifyImageSelect_clicked()
{
    QString open_fileName, basename;
    open_fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), ".",
                                                 tr("Image Files (*.png *.jpg *.bmp)"));
    if (open_fileName.isEmpty())
    {
        return;
    }
    else
    {
        QImage img_read;
        if (!(img_read.load(open_fileName)))
        {
            QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("打开图像失败").toStdString().c_str(), QString::fromLocal8Bit("打开图像失败!").toStdString().c_str());
            mess.setStyleSheet("background-color: black");
            mess.exec();
            return;
        }
        ui->label_12->setPixmap(QPixmap::fromImage(img_read.scaled(ui->label_12->size())));
        cls_oriImg = img_read;
    }
}

// 图像分类-分类操作
QImage UIdemo::classification()
{

    // 图像预处理
    cv::Mat inputs = QImageToMat(cls_oriImg);
    std::cout<<"inputs: ["<<inputs.rows<<", "<<inputs.cols<<", "<<inputs.channels()<<"]"<<endl;
    cv::Mat inputBlob = cv::dnn::blobFromImage(inputs, 1.0, cv::Size(512, 512), false, false);
    std::cout<<"inputBlob: ["<<inputBlob.rows<<", "<<inputBlob.cols<<", "<<inputBlob.channels()<<"]"<<endl;
    cv::Mat inputRGB;
    cv::cvtColor(inputs, inputRGB, cv::COLOR_BGR2RGB);//转为RGB
    std::cout<<"inputRGB: ["<<inputRGB.rows<<", "<<inputRGB.cols<<", "<<inputRGB.channels()<<"]"<<endl;
    //inputRGB.convertTo(inputRGB, CV_32FC3, 1.0/255.0, 0);
    std::cout<<inputRGB.at<cv::Vec3b>(0, 0)<<endl;

    // TODO: 针对deeplabv3+ 做图片预处理，减去均值和裁剪旋转, RGB通道
    //归一化操作
    cv::Mat dst;
    std::vector<double> mean{0.46099097, 0.32533738, 0.32106236};
    std::vector<double> std{0.20980413, 0.1538582, 0.1491854};
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(inputRGB, rgbChannels); //通道分离
    for (int i = 0; i < rgbChannels.size(); i++)
     {
        rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, 1.0/255.0, 0);
        rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, 1.0 / std[i], (0.0 - mean[i]) / std[i]);
    }
    cv::merge(rgbChannels, dst);
    std::cout<<"dst: ["<<dst.rows<<", "<<dst.cols<<", "<<dst.channels()<<"]"<<endl;
    //std::cout<<dst<<endl;

    // Create a vector of inputs.
    torch::Tensor img_tensor = torch::from_blob(dst.data, {1, dst.rows, dst.cols, 3}, torch::kFloat);//.to(torch::kCUDA); //NHWC
    std::cout<<"img_tensor[1]: "<<img_tensor.sizes()<<endl;
    img_tensor = img_tensor.permute({0, 3, 1, 2}); //转为NCHW
    std::cout<<"img_tensor[2]: "<<img_tensor.sizes()<<endl;
    //std::cout<<img_tensor<<endl;
    torch::Tensor output = deeplab_net.forward({img_tensor}).toTensor(); //获取输出, (1, 6, 512, 512)
    std::cout<<"output: "<<output.sizes()<<endl;
    //std::cout<<output<<endl;

    //std::tuple<torch::Tensor, torch::Tensor> max_Classes = torch::max(output, 1); //获取通道最大值
    //获取通道最大值并赋值到output_seg, (1, 1, 512, 512)
    torch::Tensor index = torch::argmax(output.squeeze(0), 0);
    torch::Tensor output_seg = index.unsqueeze(0);
    //std::cout<<index.squeeze(0).select(1, 1)<<endl;

    //转为mat
    output_seg = output_seg.permute({1, 2, 0}).to(torch::kU8);//.to(torch::kCPU); //CHW -> HWC,迁移至CPU
    cv::Mat mask(dst.rows, dst.cols, CV_8UC1, output_seg.data_ptr());
    std::cout<<"mask: ["<<mask.rows<<", "<<mask.cols<<", "<<mask.channels()<<"]"<<endl;
    cv::Mat img_final(dst.rows, dst.cols, CV_8UC3);
    std::cout<<"img_final: ["<<img_final.rows<<", "<<img_final.cols<<", "<<img_final.channels()<<"]"<<endl;

    //转为三通道
    for(int row=0; row < dst.rows; row++){
        //std::cout<<"row["<<row<<"]"<<endl;
        uchar *ptr = mask.ptr<uchar>(row);
        cv::Vec3b *ptrIF = img_final.ptr<cv::Vec3b>(row);
        for(int col=0; col<dst.cols; col++){
            ptrIF[col] = colors_deeplab[(int)ptr[col]];
        }
    }
    std::cout<<"img_final: ["<<img_final.rows<<", "<<img_final.cols<<", "<<img_final.channels()<<"]"<<endl;
    cv::cvtColor(img_final, img_final, CV_RGB2BGR);

    //转为QImage
    QImage resultImg = MatToQImage(img_final);

    return resultImg;

    // 读入四个网络得到输出mask -> (Batch, Channel, Height, Width)
//    vegetation_net.setInput(inputBlob);
//    cv::Mat mask1 = vegetation_net.forward();

//    bldg_net.setInput(inputBlob);
//    cv::Mat mask2 = bldg_net.forward();

//    water_net.setInput(inputBlob);
//    cv::Mat mask3 = water_net.forward();

//    road_net.setInput(inputBlob);
//    cv::Mat mask4 = road_net.forward();


//    const int rows = mask1.size[2];
//    const int cols = mask1.size[3];

//    cv::Mat value1(rows, cols, CV_32FC1, mask1.data);
//    cv::Mat value2(rows, cols, CV_32FC1, mask2.data);
//    cv::Mat value3(rows, cols, CV_32FC1, mask3.data);
//    cv::Mat value4(rows, cols, CV_32FC1, mask4.data);
//    cv::Mat segm(rows, cols, CV_8UC3);

//    for (int row = 0; row < rows; row++)
//    {
//        float *ptrSource1 = value1.ptr<float>(row);   // vegetation
//        float *ptrSource2 = value2.ptr<float>(row);   // bldg
//        float *ptrSource3 = value3.ptr<float>(row);   // water
//        float *ptrSource4 = value4.ptr<float>(row);   // road

//        cv::Vec3b *ptrSegm = segm.ptr<cv::Vec3b>(row);
//        for (int col = 0; col < cols; col++)
//        {
//            // 根据阈值决定所属类别
//            ptrSource1[col] = ptrSource1[col] >= 0.1f? 1.0f : 0;
//            ptrSource2[col] = ptrSource2[col] >= 0.3f? 1.0f : 0;
//            ptrSource3[col] = ptrSource3[col] >= 0.5f? 1.0f : 0;
//            ptrSource4[col] = ptrSource4[col] >= 0.45f? 1.0f : 0;

//            if (ptrSource1[col] == 1.0f)   ptrSegm[col] = colors[1];

//            if (ptrSource3[col] == 1.0f)    ptrSegm[col] = colors[3];

//            if (ptrSource4[col] == 1.0f)    ptrSegm[col] = colors[4];

//            if (ptrSource2[col] == 1.0f)    ptrSegm[col] = colors[2];
//        }
//    }
//    QImage ret = MatToQImage(segm);
//    return ret;


}

// 图像分类-结果展示
void UIdemo::on_pushButton_4_clicked()
{
    if (cls_oriImg.isNull())
    {
        QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("操作失败").toStdString().c_str(), QString::fromLocal8Bit("请选择要分类的图像").toStdString().c_str());
        mess.setStyleSheet("background-color: black");
        mess.exec();
        /*
        QMessageBox::information(this,
                                 tr("操作失败"),
                                 tr("请选择要分类的图像"));
                                 */
        return;
    }
    QFuture<QImage> future = QtConcurrent::run(this, &UIdemo::classification);

    if (future.isRunning())
    {

        QProgressDialog process(this);
        process.setLabelText(tr("processing..."));
        QFont font("ZYSong 18030", 12);
        process.setFont(font);
        process.setWindowTitle(tr("Please Wait"));
        // process.setWindowFlags(Qt::FramelessWindowHint);
        process.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint);

        process.setRange(0, 100000);
        process.setCancelButton(nullptr);
        process.setModal(true);
        // process.setCancelButtonText(tr("cancel"));

        for (int i = 0; i < 100000 && future.isRunning(); i++)
        {
            for (int j = 0; j < 20000; j++)
            {
                process.setValue(i);
            }

            if (i == 99999)
            {
                i = 0;
            }
            QCoreApplication::processEvents();
        }
    }
    clsd_Img = future.result();
    ui->label_14->setPixmap(QPixmap::fromImage(clsd_Img.scaled(ui->label_14->size())));
    ui->label_11->setPixmap(QPixmap::fromImage(clsd_Img.scaled(ui->label_11->size())));
}

// 图像分类-保存
void UIdemo::on_classifyResultSave_clicked()
{
    QString filename = QFileDialog::getSaveFileName(this,
                                                    tr("Save Image"),
                                                    "",
                                                    tr("*.bmp;; *.png;; *.jpg;; *.tif;; *.GIF")); //选择路径
    if (filename.isEmpty())
    {
        return;
    }
    else
    {
        if (!(clsd_Img.save(filename))) //保存图像
        {
            QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("操作失败").toStdString().c_str(), QString::fromLocal8Bit("图片为空").toStdString().c_str());
            mess.setStyleSheet("background-color: black");
            mess.exec();
            return;
        }
        QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("保存").toStdString().c_str(), QString::fromLocal8Bit("图片保存成功").toStdString().c_str());
        mess.setStyleSheet("background-color: black");
        mess.exec();
        return;
    }
}

// 洪涝检测-图像选择
void UIdemo::on_edgeImageSelect_clicked()
{
    QString open_fileName;
    open_fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), ".",
                                                 tr("Image Files (*.png *.jpg *.bmp)"));
    if (open_fileName.isEmpty())
    {
        return;
    }
    else
    {
        QImage oriImg;
        if (!(oriImg.load(open_fileName)))         //加载图像
        {
            QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("打开图像失败").toStdString().c_str(), QString::fromLocal8Bit("打开图像失败!").toStdString().c_str());
            mess.setStyleSheet("background-color: black");
            mess.exec();
            return;
        }
        ui->srcimg->setPixmap(QPixmap::fromImage(oriImg.scaled(ui->srcimg->size())));
        edge_oriImg = oriImg;
        return;
    }
}

// 洪涝检测-检测操作
QImage UIdemo::edgeDetect(const QImage img)
{
    int i,j;
    cv::Mat ImgOrg1 = QImageToMat(img);
    Mat Imgres;
    cv::resize(ImgOrg1, Imgres, Size(),0.2,0.2,INTER_AREA);
    int M,N;
    M = Imgres.rows;
    N = Imgres.cols;
    Mat ImgGry1,ImgGry;
    medianBlur(Imgres,ImgGry1,3);
        Scalar colorTab[] = {    //定义颜色数组
                                 Scalar(0,0,255),
                                 Scalar(255,0,255),
                                 Scalar(0,255,0),
                                 Scalar(255,0,0),
                                 Scalar(0,255,255)
                            };
    cvtColor(ImgOrg1,ImgGry,CV_BGR2GRAY);
        //初始化定义
        int clusterCount = 3;//3分类
        int sampleCount = M * N;//样本点数量
        Mat points(sampleCount, 3, CV_32F, Scalar(10));
        Mat labels;
        Mat centers(clusterCount, 1, points.type());

        //RGB转换到样本数据
        int index = 0;
        for (int row = 0; row < M; row++)
        {
            for (int col = 0; col < N; col++)
            {
                index = row * N + col;
                Vec3b bgr = ImgGry1.at<Vec3b>(row, col);
                //获取每个通道的值
                points.at<float>(index, 0) = static_cast<int>(bgr[0]);
                points.at<float>(index, 1) = static_cast<int>(bgr[1]);
                points.at<float>(index, 2) = static_cast<int>(bgr[2]);
            }
        }
        //运行KMeans
        TermCriteria cirteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
        kmeans(points, clusterCount, labels, cirteria, 3, KMEANS_PP_CENTERS, centers);

        //显示图像分割结果
        Mat result = Mat::zeros(ImgGry1.size(), ImgGry1.type());
        cvtColor(ImgGry1,ImgGry1,CV_BGR2GRAY);
        double E1=0,E2=0,E3=0,D1=0,D2=0,D3=0;
        int count1=0,count2=0,count3=0,counta=0,countb=0,countc=0;
        int label;
        for (int row = 0; row < M; row++)
        {
                for (int col = 0; col < N; col++)
                {
                    index = row * N + col;
                    label = labels.at<int>(index,0) ;
                    result.at<Vec3b>(row, col)[0] = colorTab[label][0];
                    result.at<Vec3b>(row, col)[1] = colorTab[label][1];
                    result.at<Vec3b>(row, col)[2] = colorTab[label][2];
                    if(label == 0)
                    {
                        E1 += (double)ImgGry1.at<uchar>(row,col);
                        count1 += 1;
                    }
                    else if(label == 1)
                    {
                        E2 += (double)ImgGry1.at<uchar>(row,col);
                        count2 += 1;
                    }
                    else if(label == 2)
                    {
                        E3 += (double)ImgGry1.at<uchar>(row,col);
                        count3 += 1;
                    }
                }
        }
        E1 = E1 / count1;
        E2 = E2 / count2;
        E3 = E3 / count3;



        for (int row = 0; row < M; row++)
        {
                for (int col = 0; col < N; col++)
                {
                    index = row * N + col;
                    label = labels.at<int>(index,0) ;
                    if(label == 0)
                    {
                        D1 += ((double)ImgGry1.at<uchar>(row,col) - E1) * ((double)ImgGry1.at<uchar>(row,col) - E1);
                        counta += 1;
                    }
                    else if(label == 1)
                    {
                        D2 += ((double)ImgGry1.at<uchar>(row,col) - E2) * ((double)ImgGry1.at<uchar>(row,col) - E2);
                        countb += 1;
                    }
                    else if(label == 2)
                    {
                        D3 += ((double)ImgGry1.at<uchar>(row,col) - E3) * ((double)ImgGry1.at<uchar>(row,col) - E3);
                        countc += 1;
                    }
                }
        }
        D1 = D1 / counta;
        D2 = D2 / countb;
        D3 = D3 / countc;

           // double E1=55.1148,E2=104.073,E3=184.445,D1=259.835,D2=164.474,D3=858.162;

//        namedWindow("result",WINDOW_NORMAL);
//        imshow("result", result);


    //高斯卷积核生成函数
    Mat gaussianKernel = get2DGaussianKernel(11,2.85);
//        int size=11; //定义卷积核大小
//        double **gaus=new double *[size];
//        int i,j;
//        for(i=0;i<size;i++)
//        {
//            gaus[i]=new double[size];  //动态生成矩阵
//        }
//        double sigma = 2.85;
//        int center=size/2;
//        double sum=0;
//        for(i=0;i<size;i++)
//        {
//            for(j=0;j<size;j++)
//            {
//                gaus[i][j]=(1/(2*PI*sigma*sigma))*exp(-((i-center)*(i-center)+(j-center)*(j-center))/(2*sigma*sigma));
//                sum+=gaus[i][j];
//            }
//        }

    //    for(i=0;i<size;i++)
    //    {
    //        for(j=0;j<size;j++)
    //        {
    //            gaus[i][j]/=sum;
    //            cout<<gaus[i][j]<<"  ";
    //        }
    //        cout<<endl<<endl;
    //    }

    //图像same卷积
    filter2D(ImgGry.clone(),ImgGry,-1,gaussianKernel);
//        Mat Img_1 = ImgGry.clone();
//        copyMakeBorder(ImgGry, Img_1, 5, 5, 5, 5, BORDER_CONSTANT, 0);

//        Mat temp = ImgGry.clone();

//        for (i = 0; i < (Img_1.rows-10); i++) {
//                for (j = 0; j < (Img_1.cols-10); j++) {
//                        double sum = 0.0;
//                        for (int k = 0; k < size; k++) {
//                            for (int l = 0; l < size; l++) {
//                                sum += (double)Img_1.at<uchar>(i + k,j + l) * gaus[k][l];
//                            }
//                        }
//                        ImgGry.at<uchar>(i,j) = sum;
//                }
//            }

    //MRF分割
    cv::resize(ImgGry, ImgGry, Size(),0.2,0.2,INTER_AREA);
    //重新定义大小的函数
    //分割图像索引

    vector<vector<int> > SegImg(M,vector<int>(N,2));  //图像分割标记矩阵

//    for(i=0;i<M;i++)
//    {
//        for(j=0;j<N;j++)
//        {
//            SegImg[i][j] = 2;
//        }
//    }

    int ICM = 4;//ICM算法迭代次数
    double Beta = 1.5;//参数设定-文章上给的
    int T = 8;//外层数据判定阈值
    double A;
    int S;

    double Fx[3];
    //三个高斯分布的均值和标准差
  //  double E1=55.1148,E2=104.073,E3=184.445,D1=259.835,D2=164.474,D3=858.162;
    //    typedef std::normal_distribution<>::param_type Params;
    //    Params p1{E1,D1}, p2{E2,D2}, p3{E3,D3}; //封装为高斯分布的参数形式
    //    std::normal_distribution<> nd;
    auto nd = [](double mu, double sigma, double x)->double{return (1/(sqrt(2*M_PI)*sigma))*exp(-((pow(x-mu,2.))/(2*sigma)));};//lambda函数封装高斯分布
    auto pv = [](vector<vector<int>> A){for_each(A.begin(),A.end(),[](vector<int> col){for_each(col.begin(),col.end(),[](int &n){cout<<" "<<n;}); cout<<endl;});};

    for(i = 1;i < (M-1);i++)
    {
        for(j = 1;j < (N-1);j++)
        {
            double x = (double)ImgGry.at<uchar>(i,j);
            Fx[0] = nd(E1,D1,x);
            Fx[1] = nd(E2,D2,x);
            Fx[2] = nd(E3,D3,x);

            SegImg[i][j] = (int)(std::max_element(Fx, Fx+3) - Fx) + 1;
        }
    }
        Mat SegImgMat = vec2Mat(SegImg);

    //使用forEach对像素点进行并行处理
        //vector<vector<int> > test(M,vector<int>(N,2));
//        typedef cv::Point_<uint8_t> Pixel;
//        SegImgMat.forEach<Pixel>([&](Pixel &pixel, const int *position)->void{
//            int idx=position[0],idy = position[1];
//            double x= (double)ImgGry.at<uchar>(idx,idy);
//            double gd[3] = {nd(E1,D1,x), nd(E2,D2,x),nd(E3,D3,x)};
////            Fx[0] = nd(E1,D1,x);
////            Fx[1] = nd(E2,D2,x);
////            Fx[2] = nd(E3,D3,x);
//            SegImg[idx][idy] = (int)(std::max_element(gd, gd+3) - gd) + 1;
//        });
//        for(i=0;i<M;i++) test[i][0]=test[i][N-1]=2;
//        for(j=0;j<N;j++) test[0][j]=test[M-1][j]=2;


//        SegImg = mat2Vec(SegImgMat);
    //double Mark[2][8]={{0}},Pyx[3]={0},Px[3]={0},Pxy[3]={0};
    vector<vector<int>> Mark(2,vector<int>(8,0));
    double Pyx[3]={0},Px[3]={0},Pxy[3]={0};

    int nn;

    for(nn=1;nn<ICM;nn++)
    {
        int Num1=0,Num2=0,Num3=0;
        double Sum1E1=0,Sum1E2=0,Sum2E1=0,Sum2E2=0,Sum3E1=0,Sum3E2=0;

        for(i=0;i<M;i++)
        {
            for(j=0;j<N;j++)
            {
                int seg = SegImg[i][j];
                double x = (double)ImgGry.at<uchar>(i,j);
                switch(seg){
                case 1:
                    Sum1E1 = Sum1E1 + x*x;
                    Sum1E2 = Sum1E2 + x*x;
                    Num1 = Num1 + 1;
                    break;
                case 2:
                    Sum2E1 = Sum2E1 + x*x;
                    Sum2E2 = Sum2E2 + x*x;
                    Num2 = Num2 + 1;
                    break;
                case 3:
                    Sum3E1 = Sum3E1 + x*x;
                    Sum3E2 = Sum3E2 + x*x;
                    Num3 = Num3 + 1;
                    break;
                default:
                    cout << "SegImg have wrong data." << endl;
                }
                //                                if(SegImg[i][j] == 1)
                //                                {
                //                                    double x = (double)ImgGry.at<uchar>(i,j);
                //                                    Sum1E1 = Sum1E1 + x*x;
                //                                    Sum1E2 = Sum1E2 + x*x;
                //                                    Num1 = Num1 + 1;
                //                                }
                //                                else if(SegImg[i][j] == 2)
                //                                {
                //                                    double x = (double)ImgGry.at<uchar>(i,j);
                //                                    Sum2E1 = Sum2E1 + x*x;
                //                                    Sum2E2 = Sum2E2 + x*x;
                //                                    Num2 = Num2 + 1;
                //                                }
                //                                else if(SegImg[i][j] == 3)
                //                                {
                //                                    double x = (double)ImgGry.at<uchar>(i,j);
                //                                    Sum3E1 = Sum3E1 + x*x;
                //                                    Sum3E2 = Sum3E2 + x*x;
                //                                    Num3 = Num3 + 1;
                //                                }
                //                                else
                //                                {
                //                                    cout << "SegImg have wrong data." << endl;
                //                                }
            }
        }

        E1 = Sum1E1 / Num1;
        D1 = (Sum1E2 / Num1) - E1 * E1;
        E2 = Sum2E1 / Num2;
        D2 = (Sum2E2 / Num2) - E2 * E2;
        E3 = Sum3E1 / Num3;
        D3 = (Sum3E2 / Num3) - E3 * E3;

        for(i=1;i<(M-1);i++)
        {
            for(j=1;j<(N-1);j++)
            {
                //领域模板赋像素值：3*3的领域，除中心外，其余8个像素点
                Mark[0][0] = (double)ImgGry.at<uchar>(i-1,j-1);
                Mark[0][1] = (double)ImgGry.at<uchar>(i-1,j);
                Mark[0][2] = (double)ImgGry.at<uchar>(i-1,j+1);
                Mark[0][3] = (double)ImgGry.at<uchar>(i,j-1);
                Mark[0][4] = (double)ImgGry.at<uchar>(i,j+1);
                Mark[0][5] = (double)ImgGry.at<uchar>(i+1,j-1);
                Mark[0][6] = (double)ImgGry.at<uchar>(i+1,j);
                Mark[0][7] = (double)ImgGry.at<uchar>(i+1,j+1);

                //领域模板赋状态值
                Mark[1][0] = SegImg[i-1][j-1];
                Mark[1][1] = SegImg[i-1][j];
                Mark[1][2] = SegImg[i-1][j+1];
                Mark[1][3] = SegImg[i][j-1];
                Mark[1][4] = SegImg[i][j+1];
                Mark[1][5] = SegImg[i+1][j-1];
                Mark[1][6] = SegImg[i+1][j];
                Mark[1][7] = SegImg[i+1][j+1];

                //进行排序，去除大于阈值的点，冒泡排序
                int Count=8,pos=0,ii,jj;
                double temp_1;
                double M1,M2;
                for(ii=0;ii<(Count-1);ii++)
                {
                    temp_1 = Mark[0][ii];
                    pos = ii;
                    for(jj=(ii+1);jj<Count;jj++)
                    {
                        if(Mark[0][jj]<temp_1)
                        {
                            temp_1 = Mark[0][jj];
                            pos = jj;
                        }
                    }
                    M1 = Mark[0][ii];
                    M2 = Mark[1][ii];
                    Mark[0][ii] = Mark[0][pos];
                    Mark[1][ii] = Mark[1][pos];
                    Mark[0][pos] = M1;
                    Mark[1][pos] = M2;
                }


                double SumUc=0;
                for(ii=0;ii<2;ii++)
                {
                    for(jj=1;jj<3;jj++)
                    {
                        SumUc = SumUc + Mark[ii][jj];
                    }
                }
                double uc;
                uc = SumUc / 4;
                for(ii=0;ii<2;ii++)
                {
                    if((Mark[0][ii]-uc)>T)
                    {
                        Mark[1][ii] = 0;
                    }
                }
                for(ii=6;ii<8;ii++)
                {
                    if((Mark[0][ii]-uc)>T)
                    {
                        Mark[1][ii] = 0;
                    }
                }

                //统计邻域内各状态个数
                int u1=0,u2=0,u3=0;
                for(ii=0;ii<8;ii++)
                {
                    if(Mark[1][ii] == 1)
                    {
                        u1+=1;
                    }
                    else if(Mark[1][ii] == 2)
                    {
                        u2+=1;
                    }
                    else if(Mark[1][ii] == 3)
                    {
                        u3+=1;
                    }
                }

                //计算势函数，只考虑状态，未考虑像素值
                double U1,U2,U3,Z;
                U1 = exp(Beta * (u1 - u2 - u3));
                U2 = exp(Beta * (u2 - u1 - u3));
                U3 = exp(Beta * (u3 - u1 - u2));
                Z = U1 + U2 + U3;

                //计算先验概率
                Px[0] = U1 / Z;
                Px[1] = U2 / Z;
                Px[2] = U3 / Z;

                //条件概率计算
//                Pyx[0] = (1/(sqrt(2*M_PI*D1)))*exp(-((((double)ImgGry.at<uchar>(i,j)-E1)*((double)ImgGry.at<uchar>(i,j)-E1))/(2*D1)));
//                Pyx[1] = (1/(sqrt(2*M_PI*D2)))*exp(-((((double)ImgGry.at<uchar>(i,j)-E2)*((double)ImgGry.at<uchar>(i,j)-E2))/(2*D2)));
//                Pyx[2] = (1/(sqrt(2*M_PI*D3)))*exp(-((((double)ImgGry.at<uchar>(i,j)-E3)*((double)ImgGry.at<uchar>(i,j)-E3))/(2*D3)));
                double x = (double)ImgGry.at<uchar>(i,j);
                Pyx[0] = nd(E1,D1,x);
                Pyx[1] = nd(E2,D2,x);
                Pyx[2] = nd(E3,D3,x);
                //后验概率计算
                for(ii=0;ii<3;ii++)
                {
                    Pxy[ii] = Pyx[ii] * Px[ii];
                }

//                A = Pxy[0];
//                S = 0;
//                for(int k = 1;k < 3;k++)
//                {
//                    if(A<Pxy[k])
//                    {
//                        A = Pxy[k];
//                        S = k;
//                    }
//                }
//                SegImg[i][j] = S + 1;
                SegImg[i][j] = std::max_element(Pxy,Pxy+3)-Pxy+1;
            }
        }
    }

    //int Beach = 85; //滩涂区域像素值

    //识别图像上显示结果
    //    Mat Imgori = Imgres.clone();

    //    for(i=0;i<M;i++)
    //    {
    //        for(j=0;j<N;j++)
    //        {

    ////            //zsy

    //            if(SegImg[i][j] == 3)
    //            {
    //                if((int)Imgori.at<Vec3b>(i,j)[0] >= 190 && (int)Imgori.at<Vec3b>(i,j)[1] >= 190 && (int)Imgori.at<Vec3b>(i,j)[2] >= 190)
    //                {
    //                    continue;
    //                }
    //                else
    //                {
    //                    Imgori.at<Vec3b>(i,j)[0] = 255;
    //                    Imgori.at<Vec3b>(i,j)[1] = 10;
    //                    Imgori.at<Vec3b>(i,j)[2] = 10;
    //                }
    //               //zsy腐蚀操作
    //                int n=1,w=0,k,p;
    //                for(k=-n;k<=n;k++)
    //                {
    //                    for (p=-n;p<=n;p++)
    //                    {
    //                        if((int)Imgori.at<Vec3b>(i+k,j+p)[0] == 255 && (int)Imgori.at<Vec3b>(i+k,j+p)[1] == 10 && (int)Imgori.at<Vec3b>(i+k,j+p)[2] == 10)
    //                        {
    //                            continue;
    //                        }
    //                        else
    //                        {
    //                            w++;
    //                        }
    //                    }
    //                }
    //                if(w<4)
    //                {
    //                    Imgori.at<Vec3b>(i,j)[0] = 255;
    //                    Imgori.at<Vec3b>(i,j)[1] = 255;
    //                    Imgori.at<Vec3b>(i,j)[2] = 255;
    //                }

    //            }
    //        }
    //    }
    //    QImage ret = MatToQImage(Imgori);
    //    return ret;
    //}



    //识别图像上显示结果
    Mat Imgori = Imgres.clone();
    //zsy
    //      vector<vector<int> > out1(M,vector<int>(N));
    //      vector<vector<int> > out2(M,vector<int>(N));
    Mat out1,out2;
    vector<vector<int> > SegImg2(M,vector<int>(N));

    Mat structElement1 = getStructuringElement(MORPH_RECT,Size(3,3),Point(-1,-1));


    Mat structElement2 = getStructuringElement(MORPH_RECT,Size(3,3),Point(-1,-1));
    //

    //Mat SegImgMat = vec2Mat(SegImg);//转换为Mat
    //腐蚀
    erode(SegImgMat,out1,structElement1,Point(-1,-1),1);
    //膨胀
    dilate(out1,out2,structElement2,Point(-1,-1),1);

    SegImg2= mat2Vec(out2);


    //zsy
    for(i=0;i<M;i++)
    {
        for(j=0;j<N;j++)
        {

            if(SegImg2[i][j] == 3)
            {
                if((int)Imgori.at<Vec3b>(i,j)[0] >= 230 || (int)Imgori.at<Vec3b>(i,j)[1] >= 230 || (int)Imgori.at<Vec3b>(i,j)[2] >= 230)
                {
                    continue;
                }
                else
                {
                    Imgori.at<Vec3b>(i,j)[0] = 255;
                    Imgori.at<Vec3b>(i,j)[1] = 10;
                    Imgori.at<Vec3b>(i,j)[2] = 10;
                }
            }
        }
    }
    QImage ret = MatToQImage(Imgori);
    return ret;
}








// 洪涝检测-结果展示
void UIdemo::on_edgeDetect_clicked()
{
    if (edge_oriImg.isNull())
    {
        QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("操作失败").toStdString().c_str(), QString::fromLocal8Bit("请选择要检测的图像").toStdString().c_str());
        mess.setStyleSheet("background-color: black");
        mess.exec();
        return;
    }
    QFuture<QImage> future = QtConcurrent::run(this, &UIdemo::edgeDetect, QImage(edge_oriImg));
    if (future.isRunning())
    {
        QProgressDialog process(this);
        process.setLabelText(tr("processing..."));
        QFont font("ZYSong 18030", 12);
        process.setFont(font);
        process.setWindowTitle(tr("Please Wait"));
        // process.setWindowFlags(Qt::FramelessWindowHint);
        process.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint);

        process.setRange(0, 100000);
        process.setCancelButton(nullptr);
        process.setModal(true);
        // process.setCancelButtonText(tr("cancel"));

        for (int i = 0; i < 100000 && future.isRunning(); i++)
        {
            for (int j = 0; j < 20000; j++)//zsy
            {
                process.setValue(i);
            }

            if (i == 99999)
            {
                i = 0;
            }
            QCoreApplication::processEvents();
        }
    }
    edge_Img = future.result();
    ui->label_9->setPixmap(QPixmap::fromImage(edge_Img.scaled(ui->label_9->size())));
    ui->label5_2->setPixmap(QPixmap::fromImage(edge_Img.scaled(ui->label5_2->size())));
}

// 洪涝检测-保存
void UIdemo::on_edgeDetectSaveBtn_clicked()
{
    QString filename = QFileDialog::getSaveFileName(this,
                                                    tr("Save Image"),
                                                    "",
                                                    tr("*.bmp;; *.png;; *.jpg;; *.tif;; *.GIF")); //选择路径
    if (filename.isEmpty())
    {
        return;
    }
    else
    {
        if (!( edge_Img.save(filename))) //保存图像
        {
            QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("操作失败").toStdString().c_str(), QString::fromLocal8Bit("图片为空").toStdString().c_str());
            mess.setStyleSheet("background-color: black");
            mess.exec();
            return;
        }
        QMessageBox mess1(QMessageBox::Information, QString::fromLocal8Bit("保存").toStdString().c_str(), QString::fromLocal8Bit("图片保存成功").toStdString().c_str());
        mess1.setStyleSheet("background-color: black");
        mess1.exec();
        return;
    }
}

// 综合评估-导出
void UIdemo::on_pushButton_clicked()
{
    QString file_path =QApplication::applicationDirPath()+QString::fromLocal8Bit("/多源数据融合平台综合评估报告.pdf").toStdString().c_str();
    QFile pdfFile(file_path);
    pdfFile.open(QIODevice::WriteOnly);
    QPdfWriter* pWriter = new QPdfWriter(&pdfFile);

    //Init Page
    pWriter->setPageSize(QPagedPaintDevice::A4);
    pWriter->setResolution(300);    //设置dpi 每个平方英寸像素为300
    pWriter->setPageMargins(QMarginsF(30, 30, 30, 30));


    QPainter* pPainter = new QPainter(pWriter);

    //Init Font
    QFont font[5]={QFont("宋体",26,QFont::Bold),QFont("宋体",26,QFont::Bold),QFont("宋体",26,QFont::Normal),QFont("宋体",26,QFont::Normal),QFont("宋体",26,QFont::Normal)};
    font[0].setPixelSize(118);
    font[1].setPixelSize(76);
    font[2].setPixelSize(64);
    font[3].setPixelSize(76);
    font[4].setPixelSize(58);

    //Painter PDF
    qDebug()<<pPainter->viewport();
    int nPDFWidth = pPainter->viewport().width();
    int nPDFHeight = pPainter->viewport().height();

    //在10%的头部居中显示
    int y=0;

    string pixmapname = ".\\image\\logo.png";
    QPixmap pixmap01;
    pixmap01.load(QString::fromStdString(pixmapname));
    int x01 = pixmap01.width();
    int y01 = pixmap01.height();
    //将图像(所有要画的东西)在pdf上按比例尺缩放
    pixmap01 = pixmap01.scaled(x01*0.9, y01*0.9);
    pPainter->drawPixmap(20, y, pixmap01);
    y+=90;


    pPainter->setFont(font[0]);
    QTextOption option(Qt::AlignTop | Qt::AlignHCenter);
    option.setWrapMode(QTextOption::WordWrap);
    pPainter->drawText(QRect(180,y, nPDFWidth, 120),QString::fromLocal8Bit("多源数据融合平台综合评估报告").toStdString().c_str(),option);
    y+=240;
    pPainter->setPen(QPen(QBrush(QColor(0,0,0)),5));
    pPainter->drawLine(0,y,nPDFWidth,y);
    pPainter->drawLine(0,y+18,nPDFWidth,y+18);
    y+=60;

    pPainter->setFont(font[1]);
    pPainter->drawText(QRect(0,y, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("图像配准：").toStdString().c_str());
    pPainter->setFont(font[3]);
    pPainter->drawText(QRect(370,y, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("将光学图像作为基准图，SAR图像作为待配准图，使用手").toStdString().c_str());
    pPainter->setFont(font[3]);
    pPainter->drawText(QRect(0,y+120, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("动配准得到配准图。").toStdString().c_str());
    y+=240;

    //获取界面图片
    pPainter->setFont(font[2]);
    int imageBorder=760;        //设置图片水平边距为150

    QPixmap pixmap11 = QPixmap::grabWidget(ui->register1, ui->register1->rect());
    //     QRect rect1 = painterPixmap1.viewport();
    //     int xb = rect1.width() / pixmap1.width();
    //     int yb = rect1.height() / pixmap1.height();
    //     //将图像(所有要画的东西)在pdf上按比例尺缩放
    //     painterPixmap1.scale(xb*0.4, yb*0.4);

    float x11 = (float)(nPDFWidth-imageBorder*2)/(float)pixmap11.width();
    pixmap11= pixmap11.scaled(nPDFWidth-imageBorder*2, x11*pixmap11.height(),Qt::IgnoreAspectRatio);    //根据大小比例,来放大缩小图片
    pPainter->drawPixmap(20, y, pixmap11);
    QPixmap pixmap12 = QPixmap::grabWidget(ui->register2, ui->register2->rect());

    float x12 = (float)(nPDFWidth-imageBorder*2)/(float)pixmap12.width();
    pixmap12= pixmap12.scaled(nPDFWidth-imageBorder*2, x12*pixmap12.height(),Qt::IgnoreAspectRatio);    //根据大小比例,来放大缩小图片
    pPainter->drawPixmap(760, y, pixmap12);
    QPixmap pixmap13 = QPixmap::grabWidget(ui->register3, ui->register3->rect());

    float x13 = (float)(nPDFWidth-imageBorder*2)/(float)pixmap13.width();
    pixmap13= pixmap13.scaled(nPDFWidth-imageBorder*2, x13*pixmap13.height(),Qt::IgnoreAspectRatio);    //根据大小比例,来放大缩小图片
    pPainter->drawPixmap(1500, y, pixmap13);

    y+=pixmap11.height()+20;

    pPainter->setFont(font[4]);
    pPainter->drawText(QRect(0,y, nPDFWidth, 80), Qt::AlignCenter | Qt::AlignCenter,QString::fromLocal8Bit("基准图                   待配准图                  配准结果").toStdString().c_str());
    y+=110;

    //     pWriter->newPage(); //写下一页

    pPainter->setFont(font[1]);
    pPainter->drawText(QRect(0,y, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("图像融合：").toStdString().c_str());
    pPainter->setFont(font[3]);
    pPainter->drawText(QRect(370,y, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("使用基于小波变换和HIS变换的图像融合方法将光学图像").toStdString().c_str());
    pPainter->setFont(font[3]);
    pPainter->drawText(QRect(0,y+120, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("和配准图融合得到融合图像，对融合性能进行评估。").toStdString().c_str());
    y+=240;

    //获取界面图片
    pPainter->setFont(font[2]);

    QPixmap pixmap21 = QPixmap::grabWidget(ui->merge1, ui->merge1->rect());

    float x21 = (float)(nPDFWidth-imageBorder*2)/(float)pixmap21.width();
    pixmap21= pixmap21.scaled(nPDFWidth-imageBorder*2, x21*pixmap21.height(),Qt::IgnoreAspectRatio);    //根据大小比例,来放大缩小图片
    pPainter->drawPixmap(20, y, pixmap21);
    QPixmap pixmap22 = QPixmap::grabWidget(ui->merge2, ui->merge2->rect());

    float x22 = (float)(nPDFWidth-imageBorder*2)/(float)pixmap22.width();
    pixmap22= pixmap22.scaled(nPDFWidth-imageBorder*2, x22*pixmap22.height(),Qt::IgnoreAspectRatio);    //根据大小比例,来放大缩小图片
    pPainter->drawPixmap(760, y, pixmap22);
    QPixmap pixmap23 = QPixmap::grabWidget(ui->merge3, ui->merge3->rect());

    float x23 = (float)(nPDFWidth-imageBorder*2)/(float)pixmap23.width();
    pixmap23= pixmap23.scaled(nPDFWidth-imageBorder*2, x23*pixmap23.height(),Qt::IgnoreAspectRatio);    //根据大小比例,来放大缩小图片
    pPainter->drawPixmap(1500, y, pixmap23);

    y+=pixmap21.height()+20;

    pPainter->setFont(font[4]);
    pPainter->drawText(QRect(0,y, nPDFWidth, 80), Qt::AlignCenter | Qt::AlignCenter, QString::fromLocal8Bit("光学图像                    配准图                    融合结果").toStdString().c_str());
    y+=110;

    pPainter->setFont(font[2]);
    pPainter->drawText(QRect(100,y, nPDFWidth/2, 70), Qt::AlignVCenter | Qt::AlignLeft,
                       QString::fromLocal8Bit("熵（Entropy） %1").arg(entropy).toStdString().c_str());

    pPainter->setFont(font[2]);
    pPainter->drawText(QRect(nPDFWidth/2+100,y, nPDFWidth/2-100, 70), Qt::AlignVCenter | Qt::AlignLeft,
                       QString::fromLocal8Bit("交互信息量（MI） %1").arg(mutual).toStdString().c_str());

    y+=90;

    pPainter->setFont(font[2]);
    pPainter->drawText(QRect(100,y, nPDFWidth/2, 70), Qt::AlignVCenter | Qt::AlignLeft,
                       QString::fromLocal8Bit("平均梯度（AG） %1").arg(Grad_ave).toStdString().c_str());

    pPainter->setFont(font[2]);
    pPainter->drawText(QRect(nPDFWidth/2+100,y, nPDFWidth/2-100, 70), Qt::AlignVCenter | Qt::AlignLeft,
                       QString::fromLocal8Bit("扭曲程度（Distortion） %1").arg(wrap).toStdString().c_str());

    y+=90;

    pPainter->setFont(font[2]);
    pPainter->drawText(QRect(100,y, nPDFWidth/2, 70), Qt::AlignVCenter | Qt::AlignLeft,
                       QString::fromLocal8Bit("通用图像质量指标（UIQI） %1").arg(quality).toStdString().c_str());

    //     y+=110;

    pWriter->newPage(); //写下一页

    y = 0;

    pPainter->drawPixmap(20, y, pixmap01);
    y+=90;


    pPainter->setFont(font[0]);
    option.setWrapMode(QTextOption::WordWrap);
    pPainter->drawText(QRect(180,y, nPDFWidth, 120),QString::fromLocal8Bit("多源数据融合平台综合评估报告").toStdString().c_str(),option);
    y+=240;
    pPainter->setPen(QPen(QBrush(QColor(0,0,0)),5));
    pPainter->drawLine(0,y,nPDFWidth,y);
    pPainter->drawLine(0,y+18,nPDFWidth,y+18);
    y+=60;

    //获取界面图片
    pPainter->setFont(font[2]);

    QPixmap pixmap21_2 = QPixmap::grabWidget(ui->merge1_2, ui->merge1_2->rect());

    float x21_2 = (float)(nPDFWidth-imageBorder*2)/(float)pixmap21_2.width();
    pixmap21_2= pixmap21_2.scaled(nPDFWidth-imageBorder*2, x21_2*pixmap21_2.height(),Qt::IgnoreAspectRatio);    //根据大小比例,来放大缩小图片
    pPainter->drawPixmap(20, y, pixmap21_2);
    QPixmap pixmap22_2 = QPixmap::grabWidget(ui->merge2_2, ui->merge2_2->rect());

    float x22_2 = (float)(nPDFWidth-imageBorder*2)/(float)pixmap22_2.width();
    pixmap22_2= pixmap22_2.scaled(nPDFWidth-imageBorder*2, x22_2*pixmap22_2.height(),Qt::IgnoreAspectRatio);    //根据大小比例,来放大缩小图片
    pPainter->drawPixmap(760, y, pixmap22_2);
    QPixmap pixmap23_2 = QPixmap::grabWidget(ui->merge3_2, ui->merge3_2->rect());

    float x23_2 = (float)(nPDFWidth-imageBorder*2)/(float)pixmap23_2.width();
    pixmap23_2= pixmap23_2.scaled(nPDFWidth-imageBorder*2, x23_2*pixmap23_2.height(),Qt::IgnoreAspectRatio);    //根据大小比例,来放大缩小图片
    pPainter->drawPixmap(1500, y, pixmap23_2);

    y+=pixmap21_2.height()+20;

    pPainter->setFont(font[4]);
    pPainter->drawText(QRect(0,y, nPDFWidth, 80), Qt::AlignCenter | Qt::AlignCenter, QString::fromLocal8Bit("光学图像                    红外图                    融合结果").toStdString().c_str());
    y+=110;

    pPainter->setFont(font[2]);
    pPainter->drawText(QRect(100,y, nPDFWidth/2, 70), Qt::AlignVCenter | Qt::AlignLeft,
                       QString::fromLocal8Bit("熵（Entropy） %1").arg(entropy).toStdString().c_str());

    pPainter->setFont(font[2]);
    pPainter->drawText(QRect(nPDFWidth/2+100,y, nPDFWidth/2-100, 70), Qt::AlignVCenter | Qt::AlignLeft,
                       QString::fromLocal8Bit("交互信息量（MI） %1").arg(mutual).toStdString().c_str());

    y+=90;

    pPainter->setFont(font[2]);
    pPainter->drawText(QRect(100,y, nPDFWidth/2, 70), Qt::AlignVCenter | Qt::AlignLeft,
                       QString::fromLocal8Bit("平均梯度（AG） %1").arg(Grad_ave).toStdString().c_str());

    pPainter->setFont(font[2]);
    pPainter->drawText(QRect(nPDFWidth/2+100,y, nPDFWidth/2-100, 70), Qt::AlignVCenter | Qt::AlignLeft,
                       QString::fromLocal8Bit("扭曲程度（Distortion） %1").arg(wrap).toStdString().c_str());

    y+=90;

    pPainter->setFont(font[2]);
    pPainter->drawText(QRect(100,y, nPDFWidth/2, 70), Qt::AlignVCenter | Qt::AlignLeft,
                       QString::fromLocal8Bit("通用图像质量指标（UIQI） %1").arg(quality).toStdString().c_str());

    //     y+=110;

    pWriter->newPage(); //写下一页



    //zsy

//    y+=pixmap11.height()+20;

//    pPainter->setFont(font[4]);
//    pPainter->drawText(QRect(0,y, nPDFWidth, 80), Qt::AlignCenter | Qt::AlignCenter,"基准图                   待配准图                  配准结果");
//    y+=110;

//    pPainter->setFont(font[1]);
//    pPainter->drawText(QRect(0,y, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,"红外图像融合：");
//    pPainter->setFont(font[3]);
//    pPainter->drawText(QRect(370,y, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,"使用基于小波变换和HIS变换的图像融合方法将光学图像");
//    pPainter->setFont(font[3]);
//    pPainter->drawText(QRect(0,y+120, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,"和配准图融合得到融合图像，对融合性能进行评估。");
//    y+=240;

//    //获取界面图片
//    pPainter->setFont(font[2]);

//    QPixmap pixmap31 = QPixmap::grabWidget(ui->merge1, ui->merge1->rect());

//    float x31 = (float)(nPDFWidth-imageBorder*2)/(float)pixmap31.width();
//    pixmap31= pixmap31.scaled(nPDFWidth-imageBorder*2, x31*pixmap31.height(),Qt::IgnoreAspectRatio);    //根据大小比例,来放大缩小图片
//    pPainter->drawPixmap(20, y, pixmap21);
//    QPixmap pixmap32 = QPixmap::grabWidget(ui->merge3, ui->merge3->rect());

//    float x32 = (float)(nPDFWidth-imageBorder*2)/(float)pixmap32.width();
//    pixmap22= pixmap32.scaled(nPDFWidth-imageBorder*2, x32*pixmap32.height(),Qt::IgnoreAspectRatio);    //根据大小比例,来放大缩小图片
//    pPainter->drawPixmap(760, y, pixmap22);
//    QPixmap pixmap23 = QPixmap::grabWidget(ui->merge3, ui->merge3->rect());

//    float x23 = (float)(nPDFWidth-imageBorder*2)/(float)pixmap23.width();
//    pixmap23= pixmap23.scaled(nPDFWidth-imageBorder*2, x23*pixmap23.height(),Qt::IgnoreAspectRatio);    //根据大小比例,来放大缩小图片
//    pPainter->drawPixmap(1500, y, pixmap23);

//    y+=pixmap21.height()+20;

//    pPainter->setFont(font[4]);
//    pPainter->drawText(QRect(0,y, nPDFWidth, 80), Qt::AlignCenter | Qt::AlignCenter,"光学图像                    配准图                    融合结果");
//    y+=110;

//    pPainter->setFont(font[2]);
//    pPainter->drawText(QRect(100,y, nPDFWidth/2, 70), Qt::AlignVCenter | Qt::AlignLeft,
//                       QString("熵（Entropy） %1").arg(entropy));

//    pPainter->setFont(font[2]);
//    pPainter->drawText(QRect(nPDFWidth/2+100,y, nPDFWidth/2-100, 70), Qt::AlignVCenter | Qt::AlignLeft,
//                       QString("交互信息量（MI） %1").arg(mutual));

//    y+=90;

//    pPainter->setFont(font[2]);
//    pPainter->drawText(QRect(100,y, nPDFWidth/2, 70), Qt::AlignVCenter | Qt::AlignLeft,
//                       QString("平均梯度（AG） %1").arg(Grad_ave));

//    pPainter->setFont(font[2]);
//    pPainter->drawText(QRect(nPDFWidth/2+100,y, nPDFWidth/2-100, 70), Qt::AlignVCenter | Qt::AlignLeft,
//                       QString("扭曲程度（Distortion） %1").arg(wrap));

//    y+=90;

//    pPainter->setFont(font[2]);
//    pPainter->drawText(QRect(100,y, nPDFWidth/2, 70), Qt::AlignVCenter | Qt::AlignLeft,
//                       QString("通用图像质量指标（UIQI） %1").arg(quality));









    int y1=0;

    pPainter->drawPixmap(20, y1, pixmap01);
    y1+=90;


    pPainter->setFont(font[0]);
    option.setWrapMode(QTextOption::WordWrap);
    pPainter->drawText(QRect(180,y1, nPDFWidth, 120),QString::fromLocal8Bit("多源数据融合平台综合评估报告").toStdString().c_str(),option);
    y1+=240;
    pPainter->setPen(QPen(QBrush(QColor(0,0,0)),5));
    pPainter->drawLine(0,y1,nPDFWidth,y1);
    pPainter->drawLine(0,y1+18,nPDFWidth,y1+18);
    y1+=60;

    pPainter->setFont(font[1]);
    pPainter->drawText(QRect(0,y1, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("图像分类：").toStdString().c_str());
    pPainter->setFont(font[3]);
    pPainter->drawText(QRect(370,y1, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("使用Deeplab v3+神经网络的深度学习对原图像进行分类").toStdString().c_str());
    pPainter->setFont(font[3]);
    pPainter->drawText(QRect(0,y1+120, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("得到分类图像。").toStdString().c_str());
    y1+=260;

    //获取界面图片
    pPainter->setFont(font[2]);

    QPixmap pixmap31 = QPixmap::grabWidget(ui->label_12, ui->label_12->rect());

    float x31 = (float)(nPDFWidth-imageBorder*1.5)/(float)pixmap31.width();
    pixmap31= pixmap31.scaled(nPDFWidth-imageBorder*1.5, x31*pixmap31.height(),Qt::IgnoreAspectRatio);    //根据大小比例,来放大缩小图片
    pPainter->drawPixmap(20, y1, pixmap31);
    QPixmap pixmap32 = QPixmap::grabWidget(ui->label_14, ui->label_14->rect());

    float x32 = (float)(nPDFWidth-imageBorder*1.5)/(float)pixmap32.width();
    pixmap32= pixmap32.scaled(nPDFWidth-imageBorder*1.5, x32*pixmap32.height(),Qt::IgnoreAspectRatio);    //根据大小比例,来放大缩小图片
    pPainter->drawPixmap(1140, y1, pixmap32);

    y1+=pixmap32.height()+20;

    pPainter->setFont(font[4]);
    pPainter->drawText(QRect(0,y1, nPDFWidth, 80), Qt::AlignCenter | Qt::AlignCenter,QString::fromLocal8Bit("原图像                                 分类图").toStdString().c_str());
    y1+=100;

    pPainter->setFont(font[2]);

    QPixmap pixmap33 = QPixmap::grabWidget(ui->label_20, ui->label_20->rect());

    int x33 = pixmap33.width();
    int y33 = pixmap33.height();
    //将图像(所有要画的东西)在pdf上按比例尺缩放
    pixmap33 = pixmap33.scaled(x33*3, y33*3);
    pPainter->drawPixmap(410, y1, pixmap33);

    pPainter->setFont(font[4]);
    pPainter->drawText(QRect(630,y1, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("建筑").toStdString().c_str());

    pPainter->setFont(font[2]);

    QPixmap pixmap34 = QPixmap::grabWidget(ui->label_21, ui->label_21->rect());

    int x34 = pixmap34.width();
    int y34 = pixmap34.height();
    //将图像(所有要画的东西)在pdf上按比例尺缩放
    pixmap34 = pixmap34.scaled(x34*3, y34*3);
    pPainter->drawPixmap(770, y1, pixmap34);

    pPainter->setFont(font[4]);
    pPainter->drawText(QRect(990,y1, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("植被").toStdString().c_str());

    pPainter->setFont(font[2]);

    QPixmap pixmap35 = QPixmap::grabWidget(ui->label_22, ui->label_22->rect());

    int x35 = pixmap35.width();
    int y35 = pixmap35.height();
    //将图像(所有要画的东西)在pdf上按比例尺缩放
    pixmap35 = pixmap35.scaled(x35*3, y35*3);
    pPainter->drawPixmap(1130, y1, pixmap35);

    pPainter->setFont(font[4]);
    pPainter->drawText(QRect(1350,y1, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("道路").toStdString().c_str());

    pPainter->setFont(font[2]);

    QPixmap pixmap36 = QPixmap::grabWidget(ui->label_23, ui->label_23->rect());

    int x36 = pixmap36.width();
    int y36 = pixmap36.height();
    //将图像(所有要画的东西)在pdf上按比例尺缩放
    pixmap36 = pixmap36.scaled(x36*3, y36*3);
    pPainter->drawPixmap(1490, y1, pixmap36);

    pPainter->setFont(font[4]);
    pPainter->drawText(QRect(1710,y1, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("水体").toStdString().c_str());

    pPainter->setFont(font[2]);

    QPixmap pixmap37 = QPixmap::grabWidget(ui->label_33, ui->label_33->rect());

    int x37 = pixmap37.width();
    int y37 = pixmap37.height();
    //将图像(所有要画的东西)在pdf上按比例尺缩放
    pixmap37 = pixmap37.scaled(x37*3, y37*3);
    pPainter->drawPixmap(50, y1, pixmap37);

    pPainter->setFont(font[4]);
    pPainter->drawText(QRect(270,y1, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("汽车").toStdString().c_str());

    pPainter->setFont(font[2]);

    QPixmap pixmap38 = QPixmap::grabWidget(ui->label_35, ui->label_35->rect());

    int x38 = pixmap38.width();
    int y38 = pixmap38.height();
    //将图像(所有要画的东西)在pdf上按比例尺缩放
    pixmap38 = pixmap38.scaled(x38*3, y38*3);
    pPainter->drawPixmap(1850, y1, pixmap38);

    pPainter->setFont(font[4]);
    pPainter->drawText(QRect(2070,y1, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("草坪").toStdString().c_str());


    y1+=pixmap36.height()+80;

    pPainter->setFont(font[1]);
    pPainter->drawText(QRect(0,y1, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("洪涝检测：").toStdString().c_str());
    pPainter->setFont(font[3]);
    pPainter->drawText(QRect(370,y1, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft,QString::fromLocal8Bit("使用基于马尔科夫随机场的图像分割方法对融合图进行分").toStdString().c_str());
    pPainter->setFont(font[3]);
    pPainter->drawText(QRect(0,y1+120, nPDFWidth, 80), Qt::AlignTop | Qt::AlignLeft, QString::fromLocal8Bit("割得到洪涝检测图。").toStdString().c_str());
    y1+=260;

    //获取界面图片
    pPainter->setFont(font[2]);

    QPixmap pixmap41 = QPixmap::grabWidget(ui->srcimg, ui->srcimg->rect());

    float x41 = (float)(nPDFWidth-imageBorder*1.5)/(float)pixmap41.width();
    pixmap41= pixmap41.scaled(nPDFWidth-imageBorder*1.5, x41*pixmap41.height(),Qt::IgnoreAspectRatio);    //根据大小比例,来放大缩小图片
    pPainter->drawPixmap(20, y1, pixmap41);
    QPixmap pixmap42 = QPixmap::grabWidget(ui->label_9, ui->label_9->rect());

    float x42 = (float)(nPDFWidth-imageBorder*1.5)/(float)pixmap42.width();
    pixmap42= pixmap42.scaled(nPDFWidth-imageBorder*1.5, x42*pixmap42.height(),Qt::IgnoreAspectRatio);    //根据大小比例,来放大缩小图片
    pPainter->drawPixmap(1140, y1, pixmap42);

    y1+=pixmap21.height()+40;

    pPainter->setFont(font[4]);
    pPainter->drawText(QRect(0,y1, nPDFWidth, 80), Qt::AlignCenter | Qt::AlignCenter, QString::fromLocal8Bit("融合图                                 检测结果").toStdString().c_str());
    y1+=140;

    //     pWriter->newPage(); //写下一页

    QDateTime current_date_time = QDateTime::currentDateTime();
    QString current_date = current_date_time.toString("yyyy-MM-dd hh:mm:ss ddd");
    pPainter->setFont(font[2]);
    pPainter->drawText(QRect(nPDFWidth/4*2,y1, nPDFWidth/2, 80), Qt::AlignVCenter | Qt::AlignLeft,
                       QString::fromLocal8Bit("报告日期: %1").arg(current_date).toStdString().c_str());

    //     pWriter->newPage(); //写下一页

    //绘制完毕
    delete pPainter;
    delete pWriter;
    pdfFile.close();


    //通过其它PDF阅读器来打开PDF
    QDesktopServices::openUrl(QUrl::fromLocalFile(file_path));
    QMessageBox mess(QMessageBox::Information, QString::fromLocal8Bit("生成PDF").toStdString().c_str(), QString::fromLocal8Bit("保存PDF文件成功！").toStdString().c_str());
    mess.setStyleSheet("background-color: black");
    mess.exec();
    return;
}

void UIdemo::on_btn_3_triggered(QAction *arg1)
{

}

void UIdemo::on_btn_2_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void UIdemo::on_btn_3_clicked()
{
    ui->stackedWidget->setCurrentIndex(1);
}

void UIdemo::on_btn_4_clicked()
{
    ui->stackedWidget->setCurrentIndex(2);
}

void UIdemo::on_btn_5_clicked()
{
    ui->stackedWidget->setCurrentIndex(3);
}

void UIdemo::on_btn_6_clicked()
{
    ui->stackedWidget->setCurrentIndex(4);
}

//void UIdemo::on_btn_7_clicked()
//{
//    ui->stackedWidget->setCurrentIndex(5);
//}

void UIdemo::on_btn_8_clicked()
{
    ui->stackedWidget->setCurrentIndex(6);
}
