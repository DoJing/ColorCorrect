//
// Created by dojing on 2021/8/23.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
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

void ColorCal(const cv::Mat& src,cv::Mat& dst)
{
    ColorCalibrate calibrate(src);
    calibrate.clcCalibateCoefficient();
    dst = calibrate.getCalibratePlane();
}
void  GammaTransform(cv::Mat &image, cv::Mat &dist)
{

    cv::Mat imageGamma;
    //灰度归一化
    image.convertTo(imageGamma, CV_64F, 1.0 / 255, 0);

    //伽马变换
    double gamma = 1/1.4;

    pow(imageGamma, gamma, dist);//dist 要与imageGamma有相同的数据类型

    dist.convertTo(dist, CV_8U, 255, 0);
}

/****************************************/
/*   实现自动对比度的函数                  */
/*   目前只有前后中通道调用                */
/*   彩色的已经加入到了函数内部             */
/*****************************************/
void BrightnessAndContrastAuto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent)
{
    CV_Assert(clipHistPercent >= 0);
    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

    int histSize = 256;
    float alpha, beta;
    double minGray = 0, maxGray = 0;

    //to calculate grayscale histogram
    cv::Mat gray;
    if (src.type() == CV_8UC1) gray = src;
    else if (src.type() == CV_8UC3) cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else if (src.type() == CV_8UC4) cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    if (clipHistPercent == 0)
    {
        // keep full available range
        cv::minMaxLoc(gray, &minGray, &maxGray);
    }
    else
    {
        cv::Mat hist; //the grayscale histogram

        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        cv::calcHist(&gray, 1, 0, cv::Mat (), hist, 1, &histSize, &histRange, uniform, accumulate);

        // calculate cumulative distribution from the histogram
        std::vector<float> accumulator(histSize);
        accumulator[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++)
        {
            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
        }

        // locate points that cuts at required value
        float max = accumulator.back();
        clipHistPercent *= max ; //make percent as absolute
        clipHistPercent /= 2.0; // left and right wings
        // locate left cut
        minGray = 0;
        while (accumulator[minGray] < clipHistPercent)
            minGray++;

        // locate right cut
        maxGray = histSize - 1;
        while (accumulator[maxGray] >= (max - clipHistPercent))
            maxGray--;
    }

    // current range
    float inputRange = maxGray - minGray;

    alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
    beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0

    // Apply brightness and contrast normalization
    // convertTo operates with saurate_cast
    src.convertTo(dst, -1, alpha, beta);

    // restore alpha channel from source
    if (dst.type() == CV_8UC4)
    {
        int from_to[] = { 3, 3};
        cv::mixChannels(&src, 4, &dst,1, from_to, 1);
    }
    return;
}

void getFiles(std::string path,std::vector<std::string>& files,std::vector<std::string>& child_dir)
{
    if(path[path.size()-1]!='/')
        path=path.append("/");
    std::string p;
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if ((dir=opendir(path.c_str())) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        //std::cout<<ptr->d_name<<std::endl;
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        else if(ptr->d_type == 8)    ///file
            files.push_back(p.assign(path).append(ptr->d_name));
        else if(ptr->d_type == 10)    ///link file
            continue;
        else if(ptr->d_type == 4)    ///dir
        {
            p.assign(path).append(ptr->d_name);
            child_dir.push_back(p.assign(path).append(ptr->d_name));
            getFiles(p,files,child_dir);
        }
    }
    closedir(dir);

    //sort(files.begin(), files.end());
    //return files;
}
std::vector<std::string> split(std::string str, std::string pattern)
{
    std::string::size_type pos;
    std::vector<std::string> result;
    str += pattern;//扩展字符串以方便操作
    int size = str.size();
    for (int i = 0; i < size; i++)
    {
        pos = str.find(pattern, i);
        if (pos < size)
        {
            std::string s = str.substr(i, pos - i);
            result.push_back(s);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}
int main()
{
    std::string src_dir = "/media/dojing/LINUXWIN/gitHub/Person_reID_baseline_pytorch/mydata_test/pytorch/gallery";
    std::string dst_dir = "/media/dojing/LINUXWIN/gitHub/Person_reID_baseline_pytorch/mydata_test_BC/pytorch/gallery";
    std::vector<cv::String> image_files ,child_dir;
    getFiles(src_dir,image_files,child_dir);
    for(int i = 0;i<child_dir.size();i++){
        std::string dir = child_dir[i].substr(src_dir.size(),child_dir[i].size()-src_dir.size());
        dir = dst_dir+dir;
        mkdir(dir.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
    }
    if (image_files.size() != 0)
    {
        for (int i = 0; i< image_files.size(); i++)//image_file.size()代表文件中总共的图片个数
        {
            cv::Mat src = cv::imread(image_files[i]);
            cv::Mat cal,BC;
            //ColorCal(src,cal);
            BrightnessAndContrastAuto(src,BC,0.05);


            std::string save_name = image_files[i].substr(src_dir.size(),image_files[i].size()-src_dir.size());
            save_name = dst_dir+save_name;
            std::cout<<save_name<<std::endl;

            cv::imwrite(save_name,BC);

//            cv::imshow("cal", cal);
//            cv::imshow("src", src);
//            //cv::imshow("gamma",gamma);
//            cv::imshow("BC",BC);
//            cv::waitKey(0);
        }
    }
    return 0;
}