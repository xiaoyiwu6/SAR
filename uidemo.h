#ifndef UIDEMO_H
#define UIDEMO_H

#include <QEvent>
#include <QDialog>
#include "opencv2/opencv.hpp"
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS

namespace Ui {
    class UIdemo;
}

class UIdemo : public QDialog
{
    Q_OBJECT
public:
    explicit UIdemo(QWidget *parent = nullptr);
    ~UIdemo();

protected:
    bool eventFilter(QObject *watched, QEvent *event);

private:
    Ui::UIdemo *ui;

    QImage register_oriImg_1;   // 配准原图像1
    QImage register_oriImg_2;   // 配准原图像2

    QImage fusion_oriImg_1;     // 融合原图像1
    QImage fusion_oriImg_2;     // 融合原图像2
    QImage fusion_Img;          // 融合后图像
    double Grad_ave;            // 融合图像的平均梯度
    double wrap;                // 融合图像相对于光学图像扭曲程度
    double entropy;             // 融合图像平均熵
    double quality;             // 光学图像和融合图像的通用图像质量指标
    double mutual;              // 融合图像与源图像之间交户信息量

    cv::dnn::Net bldg_net;      // model
    cv::dnn::Net water_net;
    cv::dnn::Net vegetation_net;
    cv::dnn::Net road_net;
    torch::jit::script::Module deeplab_net;

    QImage cls_oriImg;
    QImage clsd_Img;
    std::vector<cv::Vec3b> colors;
    std::vector<cv::Vec3b> colors_deeplab;


    QImage edge_oriImg;
    QImage edge_Img;

private slots:
    void initForm();
    void buttonClick();
    QImage classification();
    QImage imageFusion();
    QImage imageFusion1();
    QImage edgeDetect(QImage img);

private slots:
    void on_btnMenu_Min_clicked();
    void on_btnMenu_Max_clicked();
    void on_btnMenu_Close_clicked();
    void on_imageRegistration_clicked();
    void on_classifyImageSelect_clicked();
    void on_edgeImageSelect_clicked();
    void on_registerImageSelect2_clicked();
    void on_fusionImageSelect1_clicked();
    void on_fusionImageSelect2_clicked();
    void on_imageFusionButton_clicked();
    void on_registerImageSelect1_clicked();
    void on_registersave_clicked();
    void on_fusionsave_clicked();
    void on_fusionImageSelect1_2_clicked();
    void on_edgeDetect_clicked();
    void on_edgeDetectSaveBtn_clicked();
    void on_pushButton_4_clicked();
    void on_btn_3_triggered(QAction *arg1);
    void on_btn_2_clicked();
    void on_classifyResultSave_clicked();
    void on_pushButton_clicked();
    void on_fusionImageSelect2_2_clicked();
    void on_imageFusionButton_2_clicked();
    void on_fusionsave_2_clicked();
    void on_btn_3_clicked();
    void on_btn_4_clicked();
    void on_btn_5_clicked();
    void on_btn_6_clicked();
    //void on_btn_7_clicked();
    void on_btn_8_clicked();
};

#endif // UIDEMO_H
