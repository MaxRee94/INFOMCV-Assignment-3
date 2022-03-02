#include <cstdlib>
#include <string>
#include <stdio.h>
#include <iostream>
#include <numeric>
#include <random>
#include <list>

#include "utilities/General.h"
#include "VoxelReconstruction.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#define DATA_PATH "data" + std::string(PATH_SEP)

using namespace nl_uu_science_gmt;
using namespace std;
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

//void performPostProcessing() {
//    using namespace std::literals;
//    
//    src = cv::imread(DATA_PATH + "cam1/0.png"s, cv::IMREAD_COLOR);
//    cv::Mat   hsv_img, mask, gray_img, initial_thresh;
//    cv::Mat   second_thresh, add_res, and_thresh, xor_thresh;
//    cv::Mat   result_thresh, rr_thresh, final_thresh;
//    // Load source Image
//    imshow("Original Image", src);
//    cvtColor(src, hsv_img, cv::COLOR_BGR2HSV);
//    imshow("HSV Image", hsv_img);
//
//    //imwrite("HSV Image.jpg", hsv_img);
//
//    inRange(hsv_img, cv::Scalar(15, 45, 45), cv::Scalar(65, 255, 255), mask);
//    imshow("Mask Image", mask);
//
//    cvtColor(src, gray_img, cv::COLOR_BGR2GRAY);
//    adaptiveThreshold(gray_img, initial_thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 257, 2);
//    imshow("AdaptiveThresh Image", initial_thresh);
//
//    add(mask, initial_thresh, add_res);
//    erode(add_res, add_res, cv::Mat(), cv::Point(-1, -1), 1);
//    dilate(add_res, add_res, cv::Mat(), cv::Point(-1, -1), 5);
//    imshow("Bitwise Res", add_res);
//
//    threshold(gray_img, second_thresh, 170, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
//    imshow("TreshImge", second_thresh);
//
//    bitwise_and(add_res, second_thresh, and_thresh);
//    imshow("andthresh", and_thresh);
//
//    bitwise_xor(add_res, second_thresh, xor_thresh);
//    imshow("xorthresh", xor_thresh);
//
//    bitwise_or(and_thresh, xor_thresh, result_thresh);
//    imshow("Result image", result_thresh);
//
//    bitwise_and(add_res, result_thresh, final_thresh);
//    imshow("Final Thresh", final_thresh);
//    erode(final_thresh, final_thresh, cv::Mat(), cv::Point(-1, -1), 5);
//
//    bitwise_and(src, src, rr_thresh, final_thresh);
//    imshow("Segmented Image", rr_thresh);
//    imwrite("Segmented Image.jpg", rr_thresh);
//
//    cv::waitKey(0);
//}


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

std::vector<Mat> get_manual_masks(std::vector<Camera*> m_cam_views) {
    std::vector<Mat> manual_masks;
    for (int c = 0; c < m_cam_views.size(); c++) {
        // Advance cam frame to make sure it contains an active frame
        Camera* cam = m_cam_views[c];
        cam->getVideoFrame(10);

        // Read- and threshold manual mask
        Mat tmp = imread(
            DATA_PATH + "cam" + std::to_string(c + 1) + std::string(PATH_SEP) + "manual_mask.png"
        );
        std::vector<Mat> channels;
        split(tmp, channels);
        Mat manual_mask;
        threshold(channels[0], manual_mask, 150, 255, THRESH_BINARY);
        manual_masks.push_back(manual_mask);
    }
    return manual_masks;
}

double get_cam_segm_fitness(
    std::vector<int> hsv_params, std::vector<Camera*> m_cam_views, int cam_idx, Scene3DRenderer* scene3d,
    double total_pix, std::vector<Mat> manual_masks, std::vector<int> post_params, bool show = false
) {
    // Get camera and corresponding manual mask
    Camera* cam = m_cam_views[cam_idx];
    Mat manual_mask = manual_masks[cam_idx];

    // Get automatically generated mask
    scene3d->processForeground(cam);
    Mat result = cam->getForegroundImage();

    // Get mask fitness by comparing to manually-created mask
    Mat xor_thresh;
    bitwise_xor(result, manual_mask, xor_thresh);
    if (show) {
        imshow("Manual", manual_mask);
        imshow("Post-processed", result);
        imshow("Xor thresh", xor_thresh);
    }
    double white_pix = static_cast<double>(cv::countNonZero(xor_thresh));
    double fitness = 1.0 - white_pix / total_pix;

    return fitness;
}

std::vector<vector<int>> get_bg_segm_params(std::vector<Camera*> m_cam_views, Scene3DRenderer* scene3d) {
    // Hyperparameters
    int iteration_threshold = 100;
    float convergence_velocity = 1.5f;
    float stdev_multiplier = 2.0f;
    float stdev_multiplier_minimum = 0.0001f;
    vector<int> post_iteration_range = { -5, 5 };

    // Initial values
    std::vector<int> hsv_optima = { 128, 128, 128 }; // Start with mean values (255/2)
    std::vector<int> post_optima = { 0, 0 }; // Start with mean values (neither dilation nor erosion)
    int dt_since_update = 0;
    double total_pix = static_cast<double>(img_size[0] * img_size[1]);
    double best_fitness = 0.0;
    std::vector<int> hsv_sample;
    std::vector<int> post_sample;
    std::default_random_engine gen;
    int hsv_element;
    int post_element;
    double fitness;
    double avg_fitness;
    int j = 0;
    std::vector<float> fitnesses;
    std::vector<Mat> manual_masks = get_manual_masks(m_cam_views);

    // Search loop
    std::cout << "Starting segmentation parameter tuning..." << std::endl;
    while (true) {
        // Sample HSV space
        hsv_sample.clear();
        for (int i = 0; i < 3; i++) {
            std::normal_distribution<float> distr(hsv_optima[i], stdev_multiplier * 255.0f);
            hsv_element = distr(gen);

            // Clamp values to interval [0, 255]
            if (hsv_element < 0) {
                hsv_element = 0;
            }
            if (hsv_element > 255) {
                hsv_element = 255;
            }
            hsv_sample.push_back(hsv_element);
        }

        // Sample space of possible combinations of post-processing steps
        post_sample.clear();
        for (int i = 0; i < post_optima.size(); i++) {
            std::normal_distribution<float> distr(post_optima[i], stdev_multiplier * post_iteration_range[1]);
            post_element = distr(gen);

            // Clamp values to post_iteration_range
            if (post_element < post_iteration_range[0]) {
                post_element = post_iteration_range[0];
            }
            if (post_element > post_iteration_range[1]) {
                post_element = post_iteration_range[1];
            }
            post_sample.push_back(post_element);
        }

        // Set HSV values and post proc params on Scene3dRenderer
        scene3d->setHThreshold(hsv_sample[0]);
        scene3d->setSThreshold(hsv_sample[1]);
        scene3d->setVThreshold(hsv_sample[2]);
        scene3d->setPostProcParams(post_sample);

        // Test segmentation fitness values for all cameras
        fitnesses.clear();
        for (int c = 0; c < m_cam_views.size(); c++) {
            fitness = get_cam_segm_fitness(
                hsv_sample, m_cam_views, c, scene3d, total_pix, manual_masks, post_sample
            );
            fitnesses.push_back(fitness);
        }
        avg_fitness = std::accumulate(fitnesses.begin(), fitnesses.end(), 0.0) / 4.0;

        // Update if sample fitness is better than previous best value
        if (avg_fitness > best_fitness) {
            best_fitness = avg_fitness;
            hsv_optima = hsv_sample;
            post_optima = post_sample;
            dt_since_update = 0;
        }
        else {
            dt_since_update++;
        }

        if (j % 10 == 0) {
            std::cout << "  iteration: " << j << ". stdev_multiplier: " << stdev_multiplier << ".\n";
        }

        // Decrease standard deviation to favor samples close to current optima
        stdev_multiplier = max(stdev_multiplier / convergence_velocity, stdev_multiplier_minimum);

        // Stop searching if no updates have occurred for a set number of iterations
        if (dt_since_update > iteration_threshold) {
            std::cout << "Finished segmentation parameter tuning." << std::endl;
            break;
        }

        j++;
    }

    for (int c = 0; c < m_cam_views.size(); c++) {
        double fitness = get_cam_segm_fitness(
            hsv_optima, m_cam_views, c, scene3d, total_pix, manual_masks, post_optima, true
        );
    }
    std::cout << "Best fitness: " << best_fitness << " after " << j << " iterations." << std::endl;
    std::cout << "Found hsv optima: H " << hsv_optima[0] << " S " << hsv_optima[1] << " V " << hsv_optima[2] << std::endl;
    std::cout << "Found post-processing optima: " << post_optima[0] << ", " << post_optima[1];
    //waitKey();

    return { hsv_optima, post_optima };
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
        else if (argv[1] == "-m"s || argv[1] == "--manual"s) {
            VoxelReconstruction::showKeys();
            VoxelReconstruction vr(DATA_PATH, 4);
            vr.run(argc, argv);
        }
	}
    else {
        VoxelReconstruction vr(DATA_PATH, 4);

        // Tune background segmentation parameters
        std::vector<Camera*> m_cam_views = vr.get_cam_views();
        Scene3DRenderer scene3d = vr.run(argc, argv, false, false, false);
        std::vector<vector<int>> bg_segm_params = get_bg_segm_params(m_cam_views, &scene3d);

        // Run without manual slider interface, using auto-generated foregrounds
        vr.run(argc, argv, true, false, true);
    }

	return EXIT_SUCCESS;
}

