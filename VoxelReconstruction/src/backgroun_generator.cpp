#include <opencv2\opencv.hpp>
#include <iostream>
#include "background_generator.h"

using namespace cv;

Mat3b get__mean(const std::vector<Mat3b>& images)
{
    if (images.empty()) return Mat3b();

    // Create a 0 initialized image to use as accumulator
    Mat m(images[0].rows, images[0].cols, CV_64FC3);
    m.setTo(Scalar(0, 0, 0, 0));

    // Use a temp image to hold the conversion of each input image to CV_64FC3
    // This will be allocated just the first time, since all your images have
    // the same size.
    Mat temp;
    for (int i = 0; i < images.size(); ++i)
    {
        // Convert the input images to CV_64FC3 ...
        images[i].convertTo(temp, CV_64FC3);

        // ... so you can accumulate
        m += temp;
    }

    // Convert back to CV_8UC3 type, applying the division to get the actual mean
    m.convertTo(m, CV_8U, 1. / images.size());
    return m;
}

void create_bg(std::string bg_moviepath, std::string outpath) {
    VideoCapture cap(bg_moviepath);
    std::vector<Mat3b> images;
    Mat frame;
    int frameStep = 20;
    for (int i = 0; i < cap.get(cv::CAP_PROP_FRAME_COUNT); i++) {
        cap.read(frame);
        if (i % frameStep == 0) {
            continue;
        }
        images.push_back(frame);
    }
    Mat3b background = get__mean(images);
    imwrite(outpath, background);
}

void create_backgrounds() {
    std::string datapath = "data/cam";
    for (int i = 0; i < 4; i++) {
        std::string videopath = datapath + std::to_string(i + 1) + "/background.avi";
        std::string outpath = datapath + std::to_string(i + 1) + "/background.png";
        std::cout << "videopath: " << videopath << std::endl;
        std::cout << "outpath: " << outpath << std::endl;
        create_bg(videopath, outpath);
    }
}

