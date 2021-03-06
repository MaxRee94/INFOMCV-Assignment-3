/*
 * Scene3DRenderer.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Scene3DRenderer.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <stddef.h>
#include <string>
#include <algorithm>
#include <iostream>

#include "../utilities/General.h"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Scene properties class (mostly called by Glut)
 */
Scene3DRenderer::Scene3DRenderer(
		Reconstructor &r, const vector<Camera*> &cs, vector<int> optimal_hsv_values, bool manual_hsv) :
				m_reconstructor(r),
				m_cameras(cs),
				m_num(4),
				m_sphere_radius(1850)
{
	m_width = 640;
	m_height = 480;
	m_quit = false;
	m_paused = false;
	m_rotate = false;
	m_camera_view = true;
	m_show_volume = true;
	m_show_grd_flr = true;
	m_show_cam = true;
	m_show_org = true;
	m_show_arcball = false;
	m_show_info = true;
	m_fullscreen = false;

	// Read the checkerboard properties (XML)
	FileStorage fs;
	fs.open(m_cameras.front()->getDataPath() + ".." + string(PATH_SEP) + General::CBConfigFile, FileStorage::READ);
	if (fs.isOpened())
	{
		fs["CheckerBoardWidth"] >> m_board_size.width;
		fs["CheckerBoardHeight"] >> m_board_size.height;
		fs["CheckerBoardSquareSize"] >> m_square_side_len;
	}
	fs.release();

	m_current_camera = 0;
	m_previous_camera = 0;

	m_number_of_frames = m_cameras.front()->getFramesAmount();
	m_current_frame = 0;
	m_previous_frame = -1;

	const int H = optimal_hsv_values[0];
	const int S = optimal_hsv_values[1];
	const int V = optimal_hsv_values[2];
	m_h_threshold = H;
	m_ph_threshold = H;
	m_s_threshold = S;
	m_ps_threshold = S;
	m_v_threshold = V;
	m_pv_threshold = V;

	rValue = 128;
	prValue = rValue;
	gValue = 128;
	pgValue = gValue;
	bValue = 128;
	pbValue = bValue;

	createTrackbar("R", VIDEO_WINDOW, &rValue, 255);
	createTrackbar("G", VIDEO_WINDOW, &gValue, 255);
	createTrackbar("B", VIDEO_WINDOW, &bValue, 255);

	
	createTrackbar("Frame", VIDEO_WINDOW, &m_current_frame, m_number_of_frames - 2);
	createTrackbar("H", VIDEO_WINDOW, &m_h_threshold, 255);
	createTrackbar("S", VIDEO_WINDOW, &m_s_threshold, 255);
	createTrackbar("V", VIDEO_WINDOW, &m_v_threshold, 255);

	createFloorGrid();
	setTopView();
}

/**
 * Deconstructor
 * Free the memory of the floor_grid pointer vector
 */
Scene3DRenderer::~Scene3DRenderer()
{
	for (size_t f = 0; f < m_floor_grid.size(); ++f)
		for (size_t g = 0; g < m_floor_grid[f].size(); ++g)
			delete m_floor_grid[f][g];
}

/**
 * Process the current frame on each camera
 */
bool Scene3DRenderer::processFrame()
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_current_frame == m_previous_frame + 1)
		{
			m_cameras[c]->advanceVideoFrame();
		}
		else if (m_current_frame != m_previous_frame)
		{
			m_cameras[c]->getVideoFrame(m_current_frame);
		}
		assert(m_cameras[c] != NULL);
		processForeground(m_cameras[c]);
	}
	return true;
}

Mat Scene3DRenderer::applyContourFiltering(Mat input, Mat camFrame, float toplevel_size_thresh, float embedded_size_thresh, string pass_name) {
	findContours(input, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);
	vector<int> large_contours;
	vector<float> areas;
	vector<int> removed_contours;
	vector<int> embedded_contours;
	cv::Scalar green = cv::Scalar(50, 255, 50);
	cv::Scalar blue = cv::Scalar(255, 50, 50);
	for (int i = 0; i < contours.size(); i++) {
		float area = contourArea(contours[i]);
		areas.push_back(area);
		int col_offset = max(0, ((int)area - 10000) / 50);
		col_offset = min(255, col_offset);
		//cout << "contour area of contour " << i << " is: " << area << endl;
		//cout << "colour offset: " << col_offset << endl;
		cv::Scalar color = cv::Scalar(red[0] + col_offset, red[1] + col_offset, red[2]);
		drawContours(camFrame, contours, i, color, 1, LINE_8, hierarchy, 10);
		if (area > toplevel_size_thresh) {
			large_contours.push_back(i);
		}
		if (hierarchy[i][3] != -1) {
			embedded_contours.push_back(i);
			continue;
		}
		// Remove top-level contours smaller than size threshold
		if (area < toplevel_size_thresh) {
			drawContours(camFrame, contours, i, green, FILLED, 8, hierarchy);
			drawContours(input, contours, i, black, FILLED, 8, hierarchy);
			removed_contours.push_back(i);
		}
	}
	for (int i = 0; i < embedded_contours.size(); i++) {
		int c = embedded_contours[i];
		int parent = hierarchy[c][3];
		bool parent_was_removed = find(removed_contours.begin(), removed_contours.end(), parent) != removed_contours.end();
		if (areas[c] < embedded_size_thresh && !parent_was_removed) {
			drawContours(camFrame, contours, c, blue, FILLED, 8, hierarchy);
			drawContours(input, contours, c, white, FILLED, 8, hierarchy);
		}
	}
	//cv::imshow(pass_name, camFrame);
	return input;
}

void Scene3DRenderer::initPostProcessed(Mat input, Camera* camera) {
	Mat camFrame = camera->getFrame();
	input = applyContourFiltering(input, camFrame, contour_params[0], contour_params[1], "contours pass 1");

	// Execute dilation/erosion sequence according to given parameters.
	// Positive numbers correspond to dilation, negative to erosion.
	// Zero values mean that neither dilation nor erosion is applied.
	Point anchor = Point(-1, -1);
	Mat kernel = Mat();
	
	//cv::imshow("Before post proc", input);
	for (int i = 0; i < eros_dilat_params.size(); i++) {
		if (eros_dilat_params[i] < 0) {
			erode(input, input, kernel, anchor, -eros_dilat_params[i]);
		}
		else if (eros_dilat_params[i] > 0) {
			dilate(input, input, kernel, anchor, eros_dilat_params[i]);
	    }
	}

	int num_white_pix = countNonZero(input);
	if (num_white_pix == 0) {
		camera->setForegroundImage(input);
		return;
	}
	if (num_white_pix == (input.cols * input.rows)) {
		bitwise_not(input, input);
		camera->setForegroundImage(input);
		return;
	}

	// Find contours
	//cv::imshow("After eros/dil, before contours", input);
	//cout << "contour params number: " << contour_params.size() << endl;
	input = applyContourFiltering(input, camFrame, contour_params[2], contour_params[3], "contours pass 2");
	
	threshold(input, input, 20, 255, CV_THRESH_BINARY);
	//cv::imshow("AFter", input);
	
	//waitKey(0);

	camera->setForegroundImage(input);
}


/**
 * Separate the background from the foreground
 * ie.: Create an 8 bit image where only the foreground of the scene is white (255)
 */
void Scene3DRenderer::processForeground(
	Camera* camera)
{
	assert(!camera->getFrame().empty());
	Mat hsv_image;
	camera->m_colored_frame = camera->getFrame();
	cvtColor(camera->getFrame(), hsv_image, CV_BGR2HSV);  // from BGR to HSV color space

	//imshow("hsv img", hsv_image);
	//std::cout << "Using hsv thresholds: " << m_h_threshold << ", " << m_s_threshold << ", " << m_v_threshold << std::endl;

	vector<Mat> channels;
	split(hsv_image, channels);  // Split the HSV-channels for further analysis

	// Background subtraction H
	Mat tmp, foreground, background;
	absdiff(channels[0], camera->getBgHsvChannels().at(0), tmp);
	threshold(tmp, foreground, m_h_threshold, 255, CV_THRESH_BINARY);

	// Background subtraction S
	absdiff(channels[1], camera->getBgHsvChannels().at(1), tmp);
	threshold(tmp, background, m_s_threshold, 255, CV_THRESH_BINARY);
	bitwise_and(foreground, background, foreground);

	// Background subtraction V
	absdiff(channels[2], camera->getBgHsvChannels().at(2), tmp);
	threshold(tmp, background, m_v_threshold, 255, CV_THRESH_BINARY);
	bitwise_or(foreground, background, foreground);

	// Post processing
	initPostProcessed(foreground, camera);
}

/**
 * Set currently visible camera to the given camera id
 */
void Scene3DRenderer::setCamera(
		int camera)
{
	m_camera_view = true;

	if (m_current_camera != camera)
	{
		m_previous_camera = m_current_camera;
		m_current_camera = camera;
		m_arcball_eye.x = m_cameras[camera]->getCameraPlane()[0].x;
		m_arcball_eye.y = m_cameras[camera]->getCameraPlane()[0].y;
		m_arcball_eye.z = m_cameras[camera]->getCameraPlane()[0].z;
		m_arcball_up.x = 0.0f;
		m_arcball_up.y = 0.0f;
		m_arcball_up.z = 1.0f;
	}
}

/**
 * Set the 3D scene to bird's eye view
 */
void Scene3DRenderer::setTopView()
{
	m_camera_view = false;
	if (m_current_camera != -1)
		m_previous_camera = m_current_camera;
	m_current_camera = -1;

	m_arcball_eye = vec(0.0f, 0.0f, 10000.0f);
	m_arcball_centre = vec(0.0f, 0.0f, 0.0f);
	m_arcball_up = vec(0.0f, 1.0f, 0.0f);
}

/**
 * Create a LUT for the floor grid
 */
void Scene3DRenderer::createFloorGrid()
{
	const int size = m_reconstructor.getSize() / m_num;
	const int z_offset = 3;

	// edge 1
	vector<Point3i*> edge1;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge1.push_back(new Point3i(-size * m_num, y, z_offset));

	// edge 2
	vector<Point3i*> edge2;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge2.push_back(new Point3i(x, size * m_num, z_offset));

	// edge 3
	vector<Point3i*> edge3;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge3.push_back(new Point3i(size * m_num, y, z_offset));

	// edge 4
	vector<Point3i*> edge4;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge4.push_back(new Point3i(x, -size * m_num, z_offset));

	m_floor_grid.push_back(edge1);
	m_floor_grid.push_back(edge2);
	m_floor_grid.push_back(edge3);
	m_floor_grid.push_back(edge4);
}

} /* namespace nl_uu_science_gmt */
