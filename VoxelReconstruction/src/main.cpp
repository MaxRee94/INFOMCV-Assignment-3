#include <cstdlib>
#include <string>
#include <stdio.h>
#include <iostream>
#include <numeric>
#include <random>
#include <list>

#include "utilities/General.h"
#include "VoxelReconstruction.h"
//#include "controllers/Reconstructor.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#define DATA_PATH "data" + string(PATH_SEP)

using namespace nl_uu_science_gmt;
using namespace std;
using namespace cv;

cv::Mat src, erosion_dst, dilation_dst, threshold_dst;

int dilation_elem = 0;
int dilation_size = 0;

double inside_mask_weight = 8.0;

vector<int> img_size = { 644, 486 };

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
//    using namespace literals;
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


cv::Mat3b getMean(const vector<cv::Mat3b>& images){
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

        if (temp.cols == 0) {
            break;
        }

        // ... so we can accumulate
        m += temp;
    }

    // Convert back to CV_8UC3 type, applying the division to get the actual mean
    m.convertTo(m, CV_8U, 1. / images.size());
return m;
}

// Load manual masks of frames 301 and 1201 (indices 300 and 1200)
vector<vector<Mat>> get_manual_masks(vector<Camera*> m_cam_views) {
    vector<vector<Mat>> manual_masks;
    vector<int> frame_indices = { 300, 1200 };
    for (int c = 0; c < m_cam_views.size(); c++) {
        vector<Mat> cam_masks;
        Camera* cam = m_cam_views[c];
        for (int j = 0; j < 2; j++) {
            // Advance cam frame to make sure it contains an active frame
            Mat frame = cam->getVideoFrame(frame_indices.at(j));

            // Read- and threshold manual mask
            Mat tmp = imread(
                DATA_PATH + "cam" + to_string(c + 1) + string(PATH_SEP) + "manual_background_" + to_string(j + 1) + ".png"
            );
            //imshow("c: " + to_string(c) + ", j: " + to_string(j), tmp);

            vector<Mat> channels;
            split(tmp, channels);
            Mat manual_mask;
            threshold(channels[0], manual_mask, 150, 255, THRESH_BINARY);
            cam_masks.push_back(manual_mask);
        }
        manual_masks.push_back(cam_masks);
    }
    //waitKey(0);
    return manual_masks;
}

void update_cam_foreground(
    vector<int> hsv_params, vector<Camera*> m_cam_views, int cam_idx, Scene3DRenderer* scene3d,
    double total_pix, vector<vector<Mat>> manual_masks, vector<int> post_params, int frame_indx
) {
    // Get camera and corresponding manual mask
    Camera* cam = m_cam_views[cam_idx];
    cam->getVideoFrame(frame_indx);
    //Mat manual_mask = manual_masks[cam_idx][frame_indx];

    // Get automatically generated mask
    scene3d->processForeground(cam);
    Mat result = cam->getForegroundImage();
}

vector<vector<int>> get_updated_voxels(Reconstructor* reconstructor) {
    // Get updated voxel model
    reconstructor->update();
    vector<vector<int>> auto_visible_voxels = reconstructor->getVisibleVoxelCoords();

    return auto_visible_voxels;
}

double assess_foregrounds(vector<vector<int>>* manual_visible_voxels,
    Reconstructor* reconstructor, vector<vector<int>>* all_voxels, double noise_penalty)
{
    vector<vector<int>> auto_visible_voxels = get_updated_voxels(reconstructor);
    int overlapping = 0;
    double fitness;

    if (auto_visible_voxels.size() < manual_visible_voxels->size() * 3) {
        for (int i = 0; i < manual_visible_voxels->size(); i++) {
            for (int j = 0; j < auto_visible_voxels.size(); j++) {
                if (manual_visible_voxels->at(i) == auto_visible_voxels[j]) {
                    overlapping++;
                }
            }
        }
        //fitness = (double)overlapping / (double)manual_visible_voxels->size();
        fitness = (double)overlapping / (double)manual_visible_voxels->size() - ((double)(
            auto_visible_voxels.size() - overlapping) / (double)manual_visible_voxels->size()) * noise_penalty;
    }
    else {
        fitness = 0.0;
    }

    /*if (auto_visible_voxels.size() > 7000) {
        fitness *= 0.7;
    }*/

    cout << "fitness: " << fitness << endl;
    if (fitness > -0.5) {
        cout << "---num of autovoxels: " << auto_visible_voxels.size() << endl;
        cout << "---num of vox: " << reconstructor->get_number_of_voxels() << endl;
        cout << "---overlapping:" << overlapping << endl;
    }

    return fitness;
}

vector<vector<vector<int>>> get_manual_voxelmodels(
    Reconstructor* reconstructor, vector<vector<Mat>> manual_masks, vector<Camera*> m_cam_views
) {
    vector<vector<vector<int>>> all_voxel_models;

    // Set manual masks as foreground images for all cameras
    for (int j = 0; j < 2; j++) {
        for (int c = 0; c < 4; c++) {
            Camera* cam = m_cam_views[c];
            cam->setForegroundImage(manual_masks[c][j]);
            if (cam->getForegroundImage().cols == 0) {
                cout << "found wrong image: " << c << ", " << j << endl;
            }
        }
        cout << "Getting voxel model for frame " << to_string(j) << "..." << endl;
        vector<vector<int>> manual_visible_voxels = get_updated_voxels(reconstructor);
        all_voxel_models.push_back(manual_visible_voxels);
        
        //cout << "num of voxels in manual model: " << manual_visible_voxels.size() << endl;
    }

    return all_voxel_models;
}

vector<vector<int>> get_bg_segm_params(vector<Camera*> m_cam_views, Scene3DRenderer* scene3d, vector<float>* contour_optima) {
    // Hyperparameters
    int iteration_threshold = 35;
    float convergence_velocity = 1.15f;
    float stdev_multiplier = 2.0f;
    float stdev_multiplier_minimum = 0.005f;
    vector<int> post_iteration_range = { -5, 5 };
    vector<float> contour_scale_range = { 0.0, 200.0 };
    double noise_penalty = 0.8;

    // Initial values
    vector<int> hsv_optima = { 5, 28, 47 }; 
    vector<int> eros_dil_optima = { 1, 0 }; 
    int dt_since_update = 0;
    double total_pix = static_cast<double>(img_size[0] * img_size[1]);
    double best_fitness = 0.0;
    vector<int> hsv_sample;
    vector<int> eros_dil_sample;
    vector<float> contour_sample;
    default_random_engine gen;
    int hsv_element;
    int eros_dil_element;
    float contour_element;
    double fitness;
    int j = 0;
    vector<int> frame_indices = { 300, 1200 };
    vector<vector<Mat>> manual_masks = get_manual_masks(m_cam_views);
    Reconstructor reconstructor(m_cam_views, true);
    vector<vector<int>> all_voxels = reconstructor.getAllVoxelCoords();
    vector<vector<vector<int>>> manual_visible_voxels = get_manual_voxelmodels(&reconstructor, manual_masks, m_cam_views);
    //return { {0,0,0}, {0,0} };

    // Search loop
    cout << "Starting segmentation parameter tuning..." << endl;
    while (true) {
        // Sample HSV space
        hsv_sample.clear();
        for (int i = 0; i < 3; i++) {
            normal_distribution<float> distr(hsv_optima[i], stdev_multiplier * 255.0f);
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

        // Sample space of possible combinations of erosion/dilation steps
        eros_dil_sample.clear();
        for (int i = 0; i < eros_dil_optima.size(); i++) {
            normal_distribution<float> distr(eros_dil_optima[i], stdev_multiplier * post_iteration_range[1]);
            eros_dil_element = distr(gen);

            // Clamp values to post_iteration_range
            if (eros_dil_element < post_iteration_range[0]) {
                eros_dil_element = post_iteration_range[0];
            }
            if (eros_dil_element > post_iteration_range[1]) {
                eros_dil_element = post_iteration_range[1];
            }
            eros_dil_sample.push_back(eros_dil_element);
        }

        // Sample space of possible contour application settings
        contour_sample.clear();
        for (int i = 0; i < contour_optima->size(); i++) {
            normal_distribution<float> distr(contour_optima->at(i), stdev_multiplier / 10.0 * contour_scale_range[1]);
            contour_element = distr(gen);

            // Clamp values to post_iteration_range
            if (contour_element < contour_scale_range[0]) {
                contour_element = contour_scale_range[0];
            }
            if (contour_element > contour_scale_range[1]) {
                contour_element = contour_scale_range[1];
            }
            contour_sample.push_back(contour_element);
        }

        // Set HSV values and post proc params on Scene3dRenderer
        scene3d->setHThreshold(hsv_sample[0]);
        scene3d->setSThreshold(hsv_sample[1]);
        scene3d->setVThreshold(hsv_sample[2]);
        scene3d->setErDilParams(eros_dil_sample);
        scene3d->setContourParams(contour_sample);

        // Test segmentation fitness values for all cameras and frames
        fitness = 0.0;
        for (int j = 0; j < 2; j++) {
            for (int c = 0; c < m_cam_views.size(); c++) {
                update_cam_foreground(
                    hsv_sample, m_cam_views, c, scene3d, total_pix, manual_masks, eros_dil_sample, frame_indices.at(j)
                );
            }
            fitness += assess_foregrounds(
                &manual_visible_voxels.at(j), &reconstructor, &all_voxels, noise_penalty
            );
        }
        fitness /= 2.0; // take average fitness of two sampled frames

        // Update if sample fitness is better than previous best value
        if (fitness > best_fitness) {
            best_fitness = fitness;
            hsv_optima = hsv_sample;
            eros_dil_optima = eros_dil_sample;
            (*contour_optima)[0] = contour_sample[0];
            (*contour_optima)[1] = contour_sample[1];
            (*contour_optima)[2] = contour_sample[2];
            (*contour_optima)[3] = contour_sample[3];
            dt_since_update = 0;
        }
        else {
            dt_since_update++;
        }

        if (j % 3 == 0) {
            cout << "  iteration: " << j << ". stdev_multiplier: " << stdev_multiplier << " Current best fitness: " << best_fitness << ".\n";
            cout << "Current hsv optima: H " << hsv_optima[0] << " S " << hsv_optima[1] << " V " << hsv_optima[2] << endl;
            cout << "Current erosion/dilation optima: " << eros_dil_optima[0] << ", " << eros_dil_optima[1] << endl;
            cout << "Current contours optima: " << contour_optima->at(0) << ", " << contour_optima->at(1) << ", " << contour_optima->at(2) << ", " << contour_optima->at(3) << endl;
        }

        // Decrease standard deviation to favor samples close to current optima
        stdev_multiplier = max(stdev_multiplier / convergence_velocity, stdev_multiplier_minimum);

        // Stop searching if no updates have occurred for a set number of iterations
        if (dt_since_update > iteration_threshold) {
            cout << "Finished segmentation parameter tuning." << endl;
            break;
        }

        j++;
    }

    cout << "Final fitness: " << best_fitness << " after " << j << " iterations." << endl;
    cout << "Found hsv optima: H " << hsv_optima[0] << " S " << hsv_optima[1] << " V " << hsv_optima[2] << endl;
    cout << "Found erosion/dilation optima: " << eros_dil_optima[0] << ", " << eros_dil_optima[1];
    cout << "Found contours optima: " << contour_optima->at(0) << ", " << contour_optima->at(1) << ", " << contour_optima->at(2) << ", " << contour_optima->at(3) << endl;
    
    return { hsv_optima, eros_dil_optima };
}

string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

int main(int argc, char** argv){
	using namespace literals;

	if (argc > 1) {
		if (argv[1] == "-b"s || argv[1] == "--background"s) {
			cout << "Starting mean background image calculation..." << endl;

            cv::Mat frame, gray;
            // vector to store the pixel coordinates of detected checker board corners 
            vector<cv::Point2f> corner_pts;
            bool success;
            vector<cv::Mat3b> images;

            // Loop over each camera to create the background image file
            for (size_t camera = 1; camera < 5; camera++)
            {
                images.clear();

                // Looping over all the frames in the movie file
                cv::VideoCapture cap(DATA_PATH + "cam"s + to_string(camera) + "/"s + General::BackgroundVideoFile);
                int i = 0;
                int frameStep = 20;
                cout << "Looping over every " << frameStep << "th frame of " << cap.get(cv::CAP_PROP_FRAME_COUNT) << " total." << endl;
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
                cv::imwrite(DATA_PATH + "cam"s + to_string(camera) + "/"s + General::BackgroundImageFile, meanImage);
            }
		}
        else if (argv[1] == "-m"s || argv[1] == "--manual"s) {
            VoxelReconstruction::showKeys();
            VoxelReconstruction vr(DATA_PATH, 4);
            vr.run(argc, argv, {0, 0, 0}, {0, 0}, { 30.0f, 30.0f, 80.0f, 50.0f });
        }
        else if (argv[1] == "-skip"s || argv[1] == "--skip_tuning"s) {
            VoxelReconstruction::showKeys();
            VoxelReconstruction vr(DATA_PATH, 4);
            vr.run(argc, argv, { 5, 15, 56 }, { 1, 0 }, { 80.0f, 30.0f, 120.0f, 80.0f }, true, false, true);
        }
        else if (argv[1] == "-c"s || argv[1] == "--clustering"s) {
            VoxelReconstruction::showKeys();
            VoxelReconstruction vr(DATA_PATH, 4);
            vr.run(argc, argv, { 5, 28, 47 }, { -1, 3 }, true, false, true);
            //Mat hsv_image, hsv_image_bg;
            //cv::Mat bg = cv::imread(DATA_PATH + "cam2/" + "background.png");

            //Mat src = imread(DATA_PATH + "cam2/" + "clustering.png", 1);
            //
            //#pragma region BgSegm
            //cvtColor(src, hsv_image, cv::COLOR_BGR2HSV);  // from BGR to HSV color space
            //cvtColor(bg, hsv_image_bg, cv::COLOR_BGR2HSV);  // from BGR to HSV color space

            //vector<Mat> channels, bgchannels;
            //split(hsv_image, channels);  // Split the HSV-channels for further analysis
            //split(hsv_image_bg, bgchannels);

            //// Background subtraction H
            //Mat tmp, foreground, background;
            //absdiff(channels[0], bgchannels[0], tmp);
            //threshold(tmp, foreground, 5, 255, cv::THRESH_BINARY);

            //// Background subtraction S
            //absdiff(channels[1], bgchannels[1], tmp);
            //threshold(tmp, background, 28, 255, cv::THRESH_BINARY);
            //bitwise_and(foreground, background, foreground);

            //// Background subtraction V
            //absdiff(channels[2], bgchannels[2], tmp);
            //threshold(tmp, background, 47, 255, cv::THRESH_BINARY);
            //bitwise_or(foreground, background, foreground);
            //#pragma endregion
            //
            //Mat samples(foreground.rows * foreground.cols, 1, CV_32F);

            //for (int y = 0; y < foreground.rows; y++)
            //    for (int x = 0; x < foreground.cols; x++)
            //        for (int z = 0; z < 1; z++)
            //            samples.at<float>(y + x * foreground.rows, z) = foreground.at<Vec3b>(y, x)[z];


            //int clusterCount = 4;
            //Mat labels;
            //int attempts = 5;
            //Mat centers;
            //kmeans(samples, clusterCount, labels, TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);


            //Mat new_image(src.size(), src.type());
            //for (int y = 0; y < src.rows; y++)
            //    for (int x = 0; x < src.cols; x++)
            //    {
            //        int cluster_idx = labels.at<int>(y + x * src.rows, 0);
            //        new_image.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
            //        new_image.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
            //        new_image.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
            //    }
            //imshow("clustered image", new_image);
            //waitKey(0);

            //Scalar colorTab[] =
            //{
            //    Scalar(0, 0, 255),
            //    Scalar(0,255,0),
            //    Scalar(255,100,100),
            //    Scalar(255,0,255),
            //    Scalar(0,255,255)
            //};
        }
	}
    else {
        VoxelReconstruction vr(DATA_PATH, 4);

        // Tune background segmentation parameters
        vector<Camera*> m_cam_views = vr.get_cam_views();
        Scene3DRenderer scene3d = vr.run(argc, argv, { 0, 0, 0 }, { 0, 0 }, { 0.0f, 0.0f, 0.0f, 0.0f }, false, false, false);
        vector<float> contour_optima = { 50.0f, 30.0f, 80.0f, 80.0f };
        vector<vector<int>> bg_segm_params = get_bg_segm_params(m_cam_views, &scene3d, &contour_optima);

        vr.run(argc, argv, bg_segm_params[0], bg_segm_params[1], contour_optima, true, false, true);
    }

	return EXIT_SUCCESS;
}

