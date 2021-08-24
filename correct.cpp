//
// Created by dojing on 2021/8/23.
//
#include <iostream>
#include <opencv2/opencv.hpp>

class ColorCalibrate
{
private:
    float ur, vr;//校正系数
    float ub, vb;
    cv::Mat src;
public:
    ColorCalibrate(const cv::Mat& img);
    void clcCalibateCoefficient();
    cv::Mat getCalibratePlane();
};

ColorCalibrate::ColorCalibrate(const cv::Mat& img)
{
    src = img;
    assert(src.channels()==3);
    ur = 0;
    vr = 0;
    vb = 0;
    ub = 0;
}
void ColorCalibrate::clcCalibateCoefficient()
{
    cv::Mat blur_mat;
    cv::blur(src,blur_mat,cv::Size(5,5));
    double sumSquareB = 0, sumValB = 0;
    double maxSquareB = 0, maxValB = 0;
    double sumSquareR = 0, sumValR = 0;
    double maxSquareR = 0, maxValR = 0;
    double sumValG = 0;
    double maxValG = 0;
    for (int rowCount = 0; rowCount < blur_mat.rows; rowCount++)
    {
        uchar* rowPt = blur_mat.ptr<uchar>(rowCount);
        for (int colCount = 0; colCount < blur_mat.cols*3; colCount += 3)
        {
            maxValB = maxValB < rowPt[colCount] ? rowPt[colCount] : maxValB;
            sumValB += rowPt[colCount];
            sumSquareB += rowPt[colCount] * rowPt[colCount];

            maxValG = maxValG < rowPt[colCount+1] ? rowPt[colCount+1] : maxValG;
            sumValG += rowPt[colCount+1];

            maxValR = maxValR < rowPt[colCount+2] ? rowPt[colCount+2] : maxValR;
            sumValR += rowPt[colCount+2];
            sumSquareR += rowPt[colCount+2] * rowPt[colCount+2];
        }
    }
    maxSquareB = maxValB*maxValB;
    maxSquareR = maxValB*maxValR;
    ub = (sumValG*maxValB-maxValG*sumValB)/(sumSquareB*maxValB-maxSquareB*sumValB);
    vb = (sumSquareB*maxValG-maxSquareB*sumValG)/(sumSquareB*maxValB-maxSquareB*sumValB);
    ur = (sumValG*maxValR-maxValG*sumValR)/(sumSquareR*maxValR-maxSquareR*sumValR);
    vr = (sumSquareR*maxValG-maxSquareR*sumValG)/(sumSquareR*maxValR-maxSquareR*sumValR);
}
cv::Mat ColorCalibrate::getCalibratePlane()
{
    if (ur == 0 || vr == 0 || vb == 0 || vb == 0)
        clcCalibateCoefficient();
    cv::Mat cali(src.size(),src.type());
    for (int rowCount = 0; rowCount < src.rows; rowCount++)
    {
        uchar* rowPt = src.ptr<uchar>(rowCount);
        uchar* caliPt = cali.ptr<uchar>(rowCount);
        for (int colCount = 0; colCount < src.cols*src.channels(); colCount += 3)
        {
            float b = abs(ub*rowPt[colCount] * rowPt[colCount] + vb*rowPt[colCount]);
            float g = rowPt[colCount + 1];
            float r = abs(ur*rowPt[colCount+2] * rowPt[colCount+2] + vr*rowPt[colCount+2]);
            float max = MAX(b,MAX(g,r));
            if(max>255) {
                caliPt[colCount] = b / max * 255;
                caliPt[colCount + 1] = g / max * 255;
                caliPt[colCount + 2] = r / max * 255;
            }else{
                caliPt[colCount] = b;
                caliPt[colCount + 1] = g;
                caliPt[colCount + 2] = r;
            }
        }
    }
    return cali;
}
int main()
{
    std::string pattern_img = "img/*.jpg";//要遍历文件的路径及文件类型
    std::vector<cv::String> image_files;
    cv::glob(pattern_img, image_files);
    if (image_files.size() != 0)
    {
        for (int i = 0; i< image_files.size(); i++)//image_file.size()代表文件中总共的图片个数
        {
            cv::Mat src = cv::imread(image_files[i]);
            ColorCalibrate calibrate(src);
            calibrate.clcCalibateCoefficient();
            cv::Mat cal = calibrate.getCalibratePlane();
            cv::imshow("cal", cal);
            cv::imshow("src", src);
            cv::waitKey();
        }
    }
    return 0;
}