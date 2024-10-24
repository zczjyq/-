#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

// 参数：block模板大小与omega作用程度
int block = 1; // block * block 的矩形，block 越大，速度越快，但失真越明显
double omega = 0.65; // 除雾程度，[0,1]，值越大，处理后图像颜色越深

// 最小值函数，适用于三通道
double min_3(double g, double b, double r) {
    return std::min({ g, b, r });
}

// 定义去雾函数
Mat defogging(Mat image_in, int block, double omega) {
    // 分离输入图像的三通道
    vector<Mat> channels(3);
    split(image_in, channels);

    // 创建暗通道和去雾后的输出图像
    Mat dark_channel = Mat(image_in.rows, image_in.cols, CV_8UC1);
    Mat out = Mat(image_in.rows, image_in.cols, CV_8UC3);

    // 使用形态学操作算暗通道（最小值滤波）
    Mat min_channel = Mat::zeros(image_in.size(), CV_8UC1);
    min(channels[0], channels[1], min_channel);
    min(min_channel, channels[2], dark_channel);

    // 使用膨胀操作来模拟block大小的效果
    Mat kernel = getStructuringElement(MORPH_RECT, Size(block, block));
    erode(dark_channel, dark_channel, kernel);

    // 大气光强值A设为固定值
    std::vector<double> A = { 255.0, 255.0, 255.0 };

    // 并行计算每个像素的透射率和去雾后的颜色
    parallel_for_(Range(0, image_in.rows), [&](const Range& range) {
        for (int i = range.start; i < range.end; i++) {
            uchar* dark_channel_ptr = dark_channel.ptr<uchar>(i);
            Vec3b* image_in_ptr = image_in.ptr<Vec3b>(i);
            Vec3b* out_ptr = out.ptr<Vec3b>(i);

            for (int j = 0; j < image_in.cols; j++) {
                // 计算透射率 t(x)
                double tx = 1.0 - omega * dark_channel_ptr[j] / 255.0;
                if (tx < 0.1) tx = 0.1;

                // 根据公式计算去雾后的每个通道值
                for (int c = 0; c < 3; c++) {
                    out_ptr[j][c] = saturate_cast<uchar>((image_in_ptr[j][c] - A[c]) / tx + A[c]);
                }
            }
        }
        });

    return out;
}
int main() {
    //cout << "请输入视频流路径：" << endl;
    string PATH = "D:\\DevelopCode\\pytorch\\imageDehazing\\data\\test.mp4";
    //cin >> PATH;
    VideoCapture capture;
    capture.open(PATH);
    while (true)
    {
        Mat ori;
        capture >> ori;
        imshow("原图像ori", ori);
        waitKey(10);
        //算法实现
        // 开始计时
        auto start = chrono::high_resolution_clock::now();

        // 算法实现
        Mat out = defogging(ori, block, omega);

        // 结束计时
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        // 输出处理一张图片所花费的时间
        cout << "处理一张图片所花费的时间: " << elapsed.count() << " 秒" << endl;
        imshow("处理后图像out", out);
        //算法结束
        waitKey(30);
    }
    return 0;
}
