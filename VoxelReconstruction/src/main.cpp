#include <cstdlib>
#include <string>
#include <stdio.h>
#include <iostream>
#include <numeric>
#include <random>

#include "utilities/General.h"
#include "VoxelReconstruction.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#define DATA_PATH "data" + std::string(PATH_SEP)

using namespace nl_uu_science_gmt;
using namespace cv;

cv::Mat src, erosion_dst, dilation_dst, threshold_dst;

int dilation_elem = 0;
int dilation_size = 0;

std::vector<int> img_size = { 644, 486 };

int const max_elem = 2;
int const max_kernel_size = 21;

void Dilation(int, void*);

cv::Mat Erosion(int erosion_elem, int erosion_size, cv::Mat& src)
{
    int erosion_type = 0;

    if (erosion_elem == 0) { erosion_type = cv::MORPH_RECT; }
    else if (erosion_elem == 1) { erosion_type = cv::MORPH_CROSS; }
    else if (erosion_elem == 2) { erosion_type = cv::MORPH_ELLIPSE; }

    cv::Mat element = cv::getStructuringElement(erosion_type,
        cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
        cv::Point(erosion_size, erosion_size));

    cv::erode(src, erosion_dst, element);
    imshow("Erosion Demo", erosion_dst);
    return erosion_dst;
}

cv::Mat Dilation(int dilation_elem, int dilation_size, cv::Mat& src)
{
    int dilation_type = 0;
    if (dilation_elem == 0) { dilation_type = cv::MORPH_RECT; }
    else if (dilation_elem == 1) { dilation_type = cv::MORPH_CROSS; }
    else if (dilation_elem == 2) { dilation_type = cv::MORPH_ELLIPSE; }
    cv::Mat element = cv::getStructuringElement(dilation_type,
        cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        cv::Point(dilation_size, dilation_size));
    cv::dilate(src, dilation_dst, element);
    imshow("Dilation Demo", dilation_dst);
    return dilation_dst;
}

void performPostProcessing() {
    using namespace std::literals;
    
    src = cv::imread(DATA_PATH + "cam1/0.png"s, cv::IMREAD_COLOR);
    cv::Mat   hsv_img, mask, gray_img, initial_thresh;
    cv::Mat   second_thresh, add_res, and_thresh, xor_thresh;
    cv::Mat   result_thresh, rr_thresh, final_thresh;
    // Load source Image
    imshow("Original Image", src);
    cvtColor(src, hsv_img, cv::COLOR_BGR2HSV);
    imshow("HSV Image", hsv_img);

    //imwrite("HSV Image.jpg", hsv_img);

    inRange(hsv_img, cv::Scalar(15, 45, 45), cv::Scalar(65, 255, 255), mask);
    imshow("Mask Image", mask);

    cvtColor(src, gray_img, cv::COLOR_BGR2GRAY);
    adaptiveThreshold(gray_img, initial_thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 257, 2);
    imshow("AdaptiveThresh Image", initial_thresh);

    add(mask, initial_thresh, add_res);
    erode(add_res, add_res, cv::Mat(), cv::Point(-1, -1), 1);
    dilate(add_res, add_res, cv::Mat(), cv::Point(-1, -1), 5);
    imshow("Bitwise Res", add_res);

    threshold(gray_img, second_thresh, 170, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    imshow("TreshImge", second_thresh);

    bitwise_and(add_res, second_thresh, and_thresh);
    imshow("andthresh", and_thresh);

    bitwise_xor(add_res, second_thresh, xor_thresh);
    imshow("xorthresh", xor_thresh);

    bitwise_or(and_thresh, xor_thresh, result_thresh);
    imshow("Result image", result_thresh);

    bitwise_and(add_res, result_thresh, final_thresh);
    imshow("Final Thresh", final_thresh);
    erode(final_thresh, final_thresh, cv::Mat(), cv::Point(-1, -1), 5);

    bitwise_and(src, src, rr_thresh, final_thresh);
    imshow("Segmented Image", rr_thresh);
    imwrite("Segmented Image.jpg", rr_thresh);

    cv::waitKey(0);
}

cv::Mat3b getMean(const std::vector<cv::Mat3b>& images){
    if (images.empty()) return cv::Mat3b();

    // Create a 0 initialized image to use as accumulator
    cv::Mat m(images[0].rows, images[0].cols, CV_64FC3);
    m.setTo(cv::Scalar(0, 0, 0, 0));

    // Use a temp image to hold the conversion of each input image to CV_64FC3
    // This will be allocated just the first time, since all images have
    // the same size.
    cv::Mat temp;
    for (int i = 0; i < images.size(); ++i)
    {
        // Convert the input images to CV_64FC3 ...
        images[i].convertTo(temp, CV_64FC3);

        // ... so we can accumulate
        m += temp;
    }

    // Convert back to CV_8UC3 type, applying the division to get the actual mean
    m.convertTo(m, CV_8U, 1. / images.size());
    return m;
}

float get_cam_segm_fitness(
    std::vector<int> hsv_params, std::vector<Camera*> m_cam_views, int cam_idx, Scene3DRenderer scene3d,
    double total_pix, std::vector<Mat> manual_masks
) {
    // Get camera and corresponding manual mask
    Camera* cam = m_cam_views[cam_idx];
    Mat manual_mask = manual_masks[cam_idx];

    // Process foreground using new HSV values
    scene3d.setHThreshold(hsv_params[0]);
    scene3d.setSThreshold(hsv_params[1]);
    scene3d.setVThreshold(hsv_params[2]);
    scene3d.processForeground(cam);
    Mat auto_mask = cam->getForegroundImage();

    // Get mask fitness by comparing to manually-created mask
    Mat xor_thresh;
    imshow("Manual", manual_mask);
    imshow("Auto", auto_mask);
    std::cout << "Size of auto:" << auto_mask.cols << std::endl;
    waitKey(0);
    bitwise_xor(auto_mask, manual_mask, xor_thresh);
    imshow("Xor thresh", xor_thresh);
    waitKey(0);
    double white_pix = static_cast<double>(cv::countNonZero(xor_thresh));
    double fitness = white_pix / total_pix;

    return fitness;
}

std::vector<int> get_hsv_params(std::vector<Camera*> m_cam_views, Scene3DRenderer scene3d) {
    std::vector<Mat> manual_masks;
    for (int c = 0; c < m_cam_views.size(); c++) {
        // Advance cam frame to make sure it contains an active frame
        Camera* cam = m_cam_views[c];
        cam->getVideoFrame(5);

        // Read- and threshold manual mask
        Mat tmp = imread(
            DATA_PATH + "cam" + std::to_string(c + 1) + std::string(PATH_SEP) + "manual_mask.png"
        );
        Mat manual_mask;
        threshold(tmp, manual_mask, 250, 255, THRESH_BINARY);
        manual_masks.push_back(manual_mask);
    }

    std::vector<int> optima = { 123, 123, 123 };
    int dt_since_update = 0;
    int search_threshold = 10;
    double total_pix = static_cast<double>(img_size[0] * img_size[1]);
    float best_fitness = 0.0f;
    float greediness = 2.0f;
    float stddev = 10000.0f;
    std::vector<int> hsv_sample;
    std::default_random_engine gen;
    int hsv_element;
    int j = 0;
    while (true) {
        std::cout << "Starting segmentation parameter tuning..." << std::endl;

        // Sample HSV space
        hsv_sample.clear();
        for (int i = 0; i < 3; i++) {
            std::normal_distribution<double> distr(optima[i], stddev);
            hsv_element = distr(gen);
            hsv_sample.push_back(hsv_element);
        }

        // Test segmentation fitness values for all cameras
        std::vector<float> fitnesses;
        for (int c = 0; c < m_cam_views.size(); c++) {
            float fitness = get_cam_segm_fitness(hsv_sample, m_cam_views, c, scene3d, total_pix, manual_masks);
            fitnesses.push_back(fitness);
        }
        float avg_fitness = std::accumulate(fitnesses.begin(), fitnesses.end(), 0.0f) / 4.0f;

        // Update if sample fitness is better than previous best value
        if (avg_fitness > best_fitness) {
            best_fitness = avg_fitness;
            optima = hsv_sample;
            dt_since_update = 0;
        }
        else {
            dt_since_update++;
        }

        // Decrease standard deviation to favor samples close to current optima
        stddev /= greediness;

        if (j % 10 == 0) {
            std::cout << "  iteration: " << j << ". stddev: " << stddev << ".\n";
        }

        // Stop searching if no updates have occurred for a set number of iterations
        if (dt_since_update > search_threshold) {
            std::cout << "Finished segmentation parameter tuning." << std::endl;
            break;
        }

        j++;
    }

    return optima;
}

int main(int argc, char** argv){
	using namespace std::literals;

	if (argc > 1) {
		if (argv[1] == "-b"s || argv[1] == "--background"s) {
			std::cout << "Starting mean background image calculation..." << std::endl;

            cv::Mat frame, gray;
            // vector to store the pixel coordinates of detected checker board corners 
            std::vector<cv::Point2f> corner_pts;
            bool success;
            std::vector<cv::Mat3b> images;

            // Loop over each camera to create the background image file
            for (size_t camera = 1; camera < 5; camera++)
            {
                images.clear();

                // Looping over all the frames in the movie file
                cv::VideoCapture cap(DATA_PATH + "cam"s + std::to_string(camera) + "/"s + General::BackgroundVideoFile);
                int i = 0;
                int frameStep = 20;
                std::cout << "Looping over every " << frameStep << "th frame of " << cap.get(cv::CAP_PROP_FRAME_COUNT) << " total." << std::endl;
                int usedImages = 0;

                for (int i = 0; i < cap.get(cv::CAP_PROP_FRAME_COUNT); i++) {
                    cap.read(frame);
                    //cv::imshow("Bla", frame);
                    //cv::waitKey(20);
                    if (i % frameStep != 0) {
                        continue;
                    }

                    images.push_back(frame);
                }

                // Compute the mean
                cv::Mat3b meanImage = getMean(images);

                // Show result
                cv::imshow("Mean image", meanImage);
                cv::waitKey();
                cv::imwrite(DATA_PATH + "cam"s + std::to_string(camera) + "/"s + General::BackgroundImageFile, meanImage);
            }
		}
	}
    else {
        VoxelReconstruction::showKeys();
        VoxelReconstruction vr(DATA_PATH, 4);

        std::vector<Camera*> m_cam_views = vr.get_cam_views();
        Scene3DRenderer scene3d = vr.run(argc, argv, false, false);
        std::vector<int> hsv_params = get_hsv_params(m_cam_views, scene3d);
        //performPostProcessing();
    }

	return EXIT_SUCCESS;
}

