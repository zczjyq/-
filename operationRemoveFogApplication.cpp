#include<opencv2/opencv.hpp>
#include<iostream>
#include<cmath>
#include<vector>
#include<chrono>

using namespace cv;
using namespace std;

// 参数：block模板大小与omega作用程度
int block = 1; // block * block 的矩形，block 越大，速度越快，但失真越明显
double omega = 0.6; // 除雾程度，[0,1]，值越大，处理后图像颜色越深

// 最小值函数，适用于三通道
double min_3(double g, double b, double r) {
    return std::min({ g, b, r });
}

// 定义去雾函数
Mat defogging(Mat image_in, int block, double omega) {
    vector<Mat> channels(3);
    split(image_in, channels);

    Mat dark_channel = Mat(image_in.rows, image_in.cols, CV_8UC1);
    Mat out = Mat(image_in.rows, image_in.cols, CV_8UC3);

    Mat min_channel = Mat::zeros(image_in.size(), CV_8UC1);
    min(channels[0], channels[1], min_channel);
    min(min_channel, channels[2], dark_channel);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(block, block));
    erode(dark_channel, dark_channel, kernel);

    vector<double> A = { 255.0, 255.0, 255.0 };

    parallel_for_(Range(0, image_in.rows), [&](const Range& range) {
        for (int i = range.start; i < range.end; i++) {
            uchar* dark_channel_ptr = dark_channel.ptr<uchar>(i);
            Vec3b* image_in_ptr = image_in.ptr<Vec3b>(i);
            Vec3b* out_ptr = out.ptr<Vec3b>(i);

            for (int j = 0; j < image_in.cols; j++) {
                double tx = 1.0 - omega * dark_channel_ptr[j] / 255.0;
                if (tx < 0.1) tx = 0.1;

                for (int c = 0; c < 3; c++) {
                    out_ptr[j][c] = saturate_cast<uchar>((image_in_ptr[j][c] - A[c]) / tx + A[c]);
                }
            }
        }
        });

    return out;
}

Mat computeMSCN(const Mat& img, double sigma = 7.0) {
    Mat mu, mu_sq, sigma_img, MSCN;

    // 计算均值
    GaussianBlur(img, mu, Size(7, 7), sigma);
    mu_sq = mu.mul(mu); // 计算 mu^2

    // 计算 img^2
    Mat img_sq = img.mul(img);
    GaussianBlur(img_sq, sigma_img, Size(7, 7), sigma);

    // 确保 sigma_img - mu_sq 非负，并且转换为浮点型
    sigma_img = cv::abs(sigma_img - mu_sq); // 绝对值确保非负
    sigma_img.convertTo(sigma_img, CV_32F); // 确保类型为 CV_32F

    // 调用 sqrt 计算平方根
    cv::sqrt(sigma_img, sigma_img); // sqrt 结果直接保存在 sigma_img 中

    // 计算 MSCN 系数
    MSCN = (img - mu) / (sigma_img + 1.0);

    return MSCN;
}


// 计算 MSCN 的统计特征
vector<double> computeFeatures(const Mat& MSCN) {
    Scalar mean, stddev;
    meanStdDev(MSCN, mean, stddev);

    double mu = mean[0];
    double sigma = stddev[0];

    double skewness = 0.0, kurtosis = 0.0;
    for (int i = 0; i < MSCN.rows; ++i) {
        for (int j = 0; j < MSCN.cols; ++j) {
            double val = MSCN.at<float>(i, j);
            double diff = val - mu;
            skewness += pow(diff, 3);
            kurtosis += pow(diff, 4);
        }
    }

    skewness /= (MSCN.total() * pow(sigma, 3));
    kurtosis = kurtosis / (MSCN.total() * pow(sigma, 4)) - 3.0;

    return { mu, sigma, skewness, kurtosis };
}

// 计算两个特征向量的欧几里得距离
double computeDistance(const vector<double>& features1, const vector<double>& features2) {
    double dist = 0.0;
    for (size_t i = 0; i < features1.size(); ++i) {
        dist += pow(features1[i] - features2[i], 2);
    }
    return sqrt(dist);
}

// 模拟自然图像的统计特征
vector<double> getNaturalSceneModel() {
    return { 0.0, 1.0, 0.0, 3.0 };  // 这是简化后的假设自然场景模型
}

// 计算 NIQE 分数
double calculateNIQE(const Mat& img) {
    Mat gray;

    // 将图像转换为灰度图
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // 转换为浮点数以便计算 MSCN
    gray.convertTo(gray, CV_32F);

    // 计算 MSCN
    Mat MSCN = computeMSCN(gray);

    // 提取图像的统计特征
    vector<double> imgFeatures = computeFeatures(MSCN);

    // 获取自然场景的统计特征
    vector<double> naturalModel = getNaturalSceneModel();

    // 计算欧几里得距离作为 NIQE 分数
    double niqeScore = computeDistance(imgFeatures, naturalModel);

    return niqeScore;
}

int main() {
    string PATH = "D:\\DevelopCode\\pytorch\\imageDehazing\\data\\test.mp4";
    VideoCapture capture;
    capture.open(PATH);

    while (true) {
        Mat ori;
        capture >> ori;

        if (ori.empty()) {
            cout << "视频结束或无法读取帧" << endl;
            break;
        }

        imshow("原图像ori", ori);

        auto start = chrono::high_resolution_clock::now();
        Mat out = defogging(ori, block, omega);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "处理一张图片所花费的时间: " << elapsed.count() << " 秒" << endl;

        imshow("处理后图像out", out);

        // 计算原图像的 NIQE
        double niqeOri = calculateNIQE(ori);
        cout << "原图像的 NIQE 分数: " << niqeOri << endl;

        // 计算处理后图像的 NIQE
        double niqeOut = calculateNIQE(out);
        cout << "处理后图像的 NIQE 分数: " << niqeOut << endl;

        if (waitKey(10) >= 0) break;
    }

    return 0;
}
