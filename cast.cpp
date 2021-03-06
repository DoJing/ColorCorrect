#include <iostream>
#include <opencv2/opencv.hpp>


void RGB2LAB(cv::Mat rgb, cv::Mat& Lab)
{
    cv::Mat XYZ(rgb.size(), rgb.type());
    cv::Mat_<cv::Vec3b>::iterator begainRGB = rgb.begin<cv::Vec3b>();
    cv::Mat_<cv::Vec3b>::iterator endRGB = rgb.end<cv::Vec3b>();
    cv::Mat_<cv::Vec3b>::iterator begainXYZ = XYZ.begin<cv::Vec3b>();
    int shift = 22;
    for (; begainRGB != endRGB; begainRGB++, begainXYZ++)
    {
        (*begainXYZ)[0] = ((*begainRGB)[0] * 199049 + (*begainRGB)[1] * 394494 + (*begainRGB)[2] * 455033 + 524288) >> (shift-2);
        (*begainXYZ)[1] = ((*begainRGB)[0] * 75675 + (*begainRGB)[1] * 749900 + (*begainRGB)[2] * 223002 + 524288) >> (shift-2);
        (*begainXYZ)[2] = ((*begainRGB)[0] * 915161 + (*begainRGB)[1] * 114795 + (*begainRGB)[2] * 18621 + 524288) >> (shift-2);
    }

    int LabTab[1024];
    for (int i = 0; i < 1024; i++)
    {
        if (i>9)
            LabTab[i] = (int)(pow((float)i / 1020, 1.0F / 3) * (1 << shift) + 0.5);
        else
            LabTab[i] = (int)((29 * 29.0 * i / (6 * 6 * 3 * 1020) + 4.0 / 29) * (1 << shift) + 0.5);
    }
    const int ScaleLC = (int)(16 * 2.55 * (1 << shift) + 0.5);
    const int ScaleLT = (int)(116 * 2.55 + 0.5);
    const int HalfShiftValue = 524288;
    begainXYZ = XYZ.begin<cv::Vec3b>();
    cv::Mat_<cv::Vec3b>::iterator endXYZ = XYZ.end<cv::Vec3b>();
    Lab.create(rgb.size(),rgb.type());
    cv::Mat_<cv::Vec3b>::iterator begainLab = Lab.begin<cv::Vec3b>();
    for (; begainXYZ != endXYZ; begainXYZ++, begainLab++)
    {
        int X = LabTab[(*begainXYZ)[0]];
        int Y = LabTab[(*begainXYZ)[1]];
        int Z = LabTab[(*begainXYZ)[2]];
        int L = ((ScaleLT * Y - ScaleLC + HalfShiftValue) >> shift);
        int A = ((500 * (X - Y) + HalfShiftValue) >> shift) + 128;
        int B = ((200 * (Y - Z) + HalfShiftValue) >> shift) + 128;
        (*begainLab)[0] = L;
        (*begainLab)[1] = A;
        (*begainLab)[2] = B;
    }
}
float colorCheck(const cv::Mat& imgLab)
{
    cv::Mat_<cv::Vec3b>::const_iterator begainIt = imgLab.begin<cv::Vec3b>();
    cv::Mat_<cv::Vec3b>::const_iterator endIt = imgLab.end<cv::Vec3b>();
    float aSum = 0;
    float bSum = 0;
    for (; begainIt != endIt; begainIt++)
    {
        aSum += (float)(*begainIt)[1];
        bSum += (float)(*begainIt)[2];
    }
    int MN = imgLab.cols*imgLab.rows;
    double Da = aSum / MN - 128; // ??????????????????[-128???,127]?????????
    double Db = bSum / MN - 128;

    //????????????
    double D = sqrt(Da*Da+Db*Db);

    begainIt = imgLab.begin<cv::Vec3b>();
    double Ma = 0;
    double Mb = 0;
    for (; begainIt != endIt; begainIt++)
    {
        Ma += abs((*begainIt)[1]-128 - Da);
        Mb += abs((*begainIt)[2]-128 - Db);
    }
    Ma = Ma / MN;
    Mb = Mb / MN;
    //???????????????
    double M = sqrt(Ma*Ma + Mb*Mb);
    //????????????
    float K = (float)(D / M);
    return K;
}
int main()
{
    std::string pattern_img = "img/*.jpg";//???????????????????????????????????????
    std::vector<cv::String> image_files;
    cv::glob(pattern_img, image_files);
    if (image_files.size() != 0)
    {
        for (int i = 0; i< image_files.size(); i++)//image_file.size()????????????????????????????????????
        {
            cv::Mat src = cv::imread(image_files[i]);
            cv::Mat lab;
            RGB2LAB(src,lab);
            float cast = colorCheck(lab);
            std::cout<<cast<<std::endl;
            cv::imshow("src",src);
            cv::imshow("lab",lab);
            cv::waitKey();
        }
    }
    return 0;
}