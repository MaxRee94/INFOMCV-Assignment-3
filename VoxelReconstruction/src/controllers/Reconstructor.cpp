/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Reconstructor.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <queue>
#include <iostream>
#include<conio.h>
#include<math.h>


#include "../utilities/General.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Voxel reconstruction class
 */
Reconstructor::Reconstructor(
		const vector<Camera*> &cs, bool init_voxels) :
				m_cameras(cs),
				m_height(2048),
				m_step(32)
{
	vector<Point2f> temp = vector<Point2f>(NULL);
	for (size_t i = 0; i < 4; i++)
	{
		center_coordinates.push_back(temp);
	}

	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_plane_size.area() > 0)
			assert(m_plane_size.width == m_cameras[c]->getSize().width && m_plane_size.height == m_cameras[c]->getSize().height);
		else
			m_plane_size = m_cameras[c]->getSize();
	}

	const size_t edge = 2 * m_height;
	m_voxels_amount = (edge / m_step) * (edge / m_step) * (m_height / m_step);
	std::cout << "Amnt of voxels: " << m_voxels_amount << std::endl;

	if (init_voxels) {
		initialize();
	}
}

/**
 * Deconstructor
 * Free the memory of the pointer vectors
 */
Reconstructor::~Reconstructor()
{
	for (size_t c = 0; c < m_corners.size(); ++c)
		delete m_corners.at(c);
	for (size_t v = 0; v < m_voxels.size(); ++v)
		delete m_voxels.at(v);
}

/**
 * Create some Look Up Tables
 * 	- LUT for the scene's box corners
 * 	- LUT with a map of the entire voxelspace: point-on-cam to voxels
 * 	- LUT with a map of the entire voxelspace: voxel to cam points-on-cam
 */
void Reconstructor::initialize()
{
	// Cube dimensions from [(-m_height, m_height), (-m_height, m_height), (0, m_height)]
	const int xL = -m_height;
	const int xR = m_height;
	const int yL = -m_height + 100;
	const int yR = m_height + 100;
	const int zL = 0;
	const int zR = m_height;
	const int plane_y = (yR - yL) / m_step;
	const int plane_x = (xR - xL) / m_step;
	const int plane = plane_y * plane_x;

	// Save the 8 volume corners
	// bottom
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zL));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zL));

	// top
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zR));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zR));

	// Acquire some memory for efficiency
	cout << "Initializing " << m_voxels_amount << " voxels ";
	m_voxels.resize(m_voxels_amount);

	int z;
	int pdone = 0;
#pragma omp parallel for schedule(guided) private(z) shared(pdone)
	for (z = zL; z < zR; z += m_step)
	{
		const int zp = (z - zL) / m_step;
		int done = cvRound((zp * plane / (double) m_voxels_amount) * 100.0);

#pragma omp critical
		if (done > pdone)
		{
			pdone = done;
			cout << done << "%..." << flush;
		}

		int y, x;
		for (y = yL; y < yR; y += m_step)
		{
			const int yp = (y - yL) / m_step;

			for (x = xL; x < xR; x += m_step)
			{
				const int xp = (x - xL) / m_step;

				// Create all voxels
				Voxel* voxel = new Voxel;
				voxel->x = x;
				voxel->y = y;
				voxel->z = z;
				voxel->camera_projection = vector<Point>(m_cameras.size());
				voxel->valid_camera_projection = vector<int>(m_cameras.size(), 0);

				const int p = zp * plane + yp * plane_x + xp;  // The voxel's index

				for (size_t c = 0; c < m_cameras.size(); ++c)
				{
					Point point = m_cameras[c]->projectOnView(Point3f((float) x, (float) y, (float) z));

					// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
					voxel->camera_projection[(int) c] = point;

					// If it's within the camera's FoV, flag the projection
					if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height)
						voxel->valid_camera_projection[(int) c] = 1;
				}

				//Writing voxel 'p' is not critical as it's unique (thread safe)
				m_voxels[p] = voxel;
			}
		}
	}

	cout << "done!" << endl;
}

double distanceCalculate(double x1, double y1, double x2, double y2)
{
	double x = x1 - x2; //calculating number to square in next step
	double y = y1 - y2;
	double dist;

	dist = pow(x, 2) + pow(y, 2);       //calculating Euclidean distance
	dist = sqrt(dist);

	return dist;
}

void Reconstructor::get_floodfill_subset(std::vector<Reconstructor::Voxel*>* cluster, vector<int>* included_indices, std::vector<Reconstructor::Voxel*>* subset) {
	queue<int> subset_queue;
	vector<int> neighbors;
	vector<int> x_neighbor_range;
	vector<int> y_neighbor_range;
	vector<int> z_neighbor_range;
	Reconstructor::Voxel* vox;
	Reconstructor::Voxel* neighbor;
	int x, y, z, i;
	bool x_neighbor, y_neighbor, z_neighbor;
	subset_queue.push(included_indices->at(0));
	while (!subset_queue.empty()) {
		//if (subset_queue.size() % 200 == 0) {
		//	cout << "queue size: " << subset_queue.size() << endl;
		//	//waitKey(0);
		//}
		i = subset_queue.front();
		subset_queue.pop();
		if (i > cluster->size()) {
			cout << "i: " << i << " and cluster size: " << cluster->size() << endl;
		}
		vox = cluster->at(i);
		subset->push_back(vox);
		x_neighbor_range = { vox->x - m_step, vox->x + m_step };
		y_neighbor_range = { vox->y - m_step, vox->y + m_step };
		z_neighbor_range = { vox->z - m_step, vox->z + m_step };

		// Get neighbors
		int j_min = max(i - 12000, 0);
		int max_j = min(i + 12000, (int)cluster->size());
		int j;
		neighbors.clear();
		for (j = j_min; j < max_j; j++) {
			neighbor = cluster->at(j);
			x = neighbor->x;
			y = neighbor->y;
			z = neighbor->z;
			x_neighbor = x_neighbor_range[0] <= x && x <= x_neighbor_range[1];
			y_neighbor = y_neighbor_range[0] <= y && y <= y_neighbor_range[1];
			z_neighbor = z_neighbor_range[0] <= z && z <= z_neighbor_range[1];
			if (x_neighbor && y_neighbor && z_neighbor) {
				neighbors.push_back(j);
			}
		}
		
		// Add neighbors to queue and included indices
		for (int j = 0; j < neighbors.size(); j++) {
			if (find(included_indices->begin(), included_indices->end(), j) != included_indices->end()) continue;
			subset_queue.push(j);
			included_indices->push_back(j);
		}
	}
}

bool Reconstructor::is_person(std::vector<Reconstructor::Voxel*>* subset) {
	bool is_person = false;
	float height_thresh = m_height_thresh * m_height;
	for (int i = 0; i < subset->size(); i++) {
		if (subset->at(i)->z < height_thresh) {
			is_person = true;
		}
	}
	return is_person;
}

/**
 * Count the amount of camera's each voxel in the space appears on,
 * if that amount equals the amount of cameras, add that voxel to the
 * visible_voxels vector
 */
void Reconstructor::update()
{
	std::vector<Voxel*> visible_voxels;

	int v;
#pragma omp parallel for schedule(guided) private(v) shared(visible_voxels)
	for (v = 0; v < (int) m_voxels_amount; ++v)
	{
		int camera_counter = 0;
		Voxel* voxel = m_voxels[v];
		/*Voxel* prev_voxel = NULL;

		if (!prev_visible_voxels.empty()) {
			prev_voxel = prev_visible_voxels[v];
		}*/

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			if (voxel->valid_camera_projection[c])
			{
				const Point point = voxel->camera_projection[c];

				//If there's a white pixel on the foreground image at the projection point, add the camera
				if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;
			}
		}

		// If the voxel is present on all cameras
		if (camera_counter == m_cameras.size())
		{
			std::vector<double> cam_distances;
			for (size_t c = 0; c < m_cameras.size(); ++c) {
				cam_distances.clear();
				cam_distances.push_back(distanceCalculate(m_cameras[c]->getCameraLocation().x, m_cameras[c]->getCameraLocation().y, voxel->x, voxel->y));
				voxel->closest_camera_index = std::min_element(cam_distances.begin(), cam_distances.end()) - cam_distances.begin();
			}

#pragma omp critical //push_back is critical
			visible_voxels.push_back(voxel);
		}
	}

	
	m_visible_voxels.clear();
	m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end());

	// Clustering
	Scalar colorTab[] =
	{
		Scalar(0, 0, 255),
		Scalar(0,255,0),
		Scalar(255,100,100),
		Scalar(255,0,255),
		Scalar(0,255,255)
	};

	vector<Point2f> m_groundCoordinates(m_visible_voxels.size());

	for (int i = 0; i < (int)m_visible_voxels.size(); i++) {
		m_groundCoordinates[i] = Point2f(m_visible_voxels[i]->x, m_visible_voxels[i]->y);
	}

	int clusterCount = 4;
	Mat labels;
	int attempts = 5;
	vector<Point2f> centers;
	kmeans(m_groundCoordinates, clusterCount, labels, TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

	// Make separate cluster vectors
	vector<vector<Voxel*>> clusters = { {}, {}, {}, {}};
	int clusterIdx;
	for (int i = 0; i < (int)m_visible_voxels.size(); i++) {
		clusterIdx = labels.at<int>(i);
		clusters[clusterIdx].push_back(m_visible_voxels[i]);
	}

	// Filter out floating voxel clouds
	std::vector<Voxel*> filtered_visible_voxels;
	int sampleIndex;
	vector<Voxel*> subset;
	Voxel* vox_sample;
	int remaining_voxels = 0;
	float height_thresh = m_height_thresh * m_height;
	for (int i = 0; i < 4; i++) {
		if (!voxel_post_processing) {
			break;
		}

		vector<Voxel*> cluster = clusters[i];
		while (true) {
			subset.clear();
			
			// Select voxel from bottom portion of voxel space in order to ensure it is part of the 'person' subset
			while (true) {
				sampleIndex = rand() % cluster.size();
				vox_sample = cluster[sampleIndex];
				if (vox_sample->z < height_thresh) {
					break;
				}
			}

			// Get 'person'-subset of the cluster using flood fill algorithm
			vector<int> included_indices = {sampleIndex};
			get_floodfill_subset(&cluster, &included_indices, &subset);

			//bool person = is_person(&subset);
			if (subset.size() > 500) {
				// Add all voxels from the subset to the visible voxels vector.
				cout << "Setting visible voxels for cluster " << i << "..." << endl;
				for (int j = 0; j < subset.size(); j++) {
					filtered_visible_voxels.push_back(subset[j]);
					remaining_voxels++;
				}

				cout << "Person subset size: " << subset.size() << endl;

				// All remaining voxels in the cluster can be ignored
				break;
			}
		}
		cout << "Cluster remaining voxels: " << remaining_voxels << endl;
		cout << "Total cluster voxels: " << cluster.size() << endl;
		remaining_voxels = 0;
	}

	if (voxel_post_processing) {
		// Replace visible voxels with filtered vector
		m_visible_voxels.clear();
		m_visible_voxels.insert(m_visible_voxels.end(), filtered_visible_voxels.begin(), filtered_visible_voxels.end());

		//return;

		// Re-run k-means after filtering
		for (int i = 0; i < (int)m_visible_voxels.size(); i++) {
			m_groundCoordinates[i] = Point2f(m_visible_voxels[i]->x, m_visible_voxels[i]->y);
		}

		centers.clear();
		kmeans(m_groundCoordinates, clusterCount, labels, TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

	}
	
	// Get frame(s) of camera(s) for voxel projection
	vector<Mat> frames;
	for (int c = 0; c < 4; c++) {
		frames.push_back(m_cameras[c]->getFrame());
	}
	//Mat frame = m_cameras[1]->getFrame();
	Point proj;
	Vec3b p_color;

	// Count cluster sizes
	vector<int> clusterSizes = {0,0,0,0};
	for (int i = 0; i < (int)m_visible_voxels.size(); i++) {
		int clusterIdx = labels.at<int>(i);
		//m_visible_voxels[i]->color = colorTab[clusterIdx];
		clusterSizes[clusterIdx]++;
	}
	//return;

	// Initialize people point Mats
	vector<vector<Mat>> people_Points(4, {Mat(), Mat(), Mat(), Mat()});
	for (int clust = 0; clust < 4; clust++) {
		for (int cam = 0; cam < 4; cam++) {
			people_Points[clust][cam] = Mat(clusterSizes[clust], 3, CV_64FC1);
		}
	}

	// Get color matrices of visible voxels of each cluster
	vector<int> clusterPointIndices = { 0,0,0,0 };
	for (int i = 0; i < (int)m_visible_voxels.size(); i++) {
		int clusterIdx = labels.at<int>(i);

		for (int c = 0; c < 4; c++) {
			p_color = frames[c].at<Vec3b>(m_visible_voxels[i]->camera_projection[c]);

			people_Points[clusterIdx][c].at<double>(clusterPointIndices[clusterIdx], 0) = (double)p_color[0];
			people_Points[clusterIdx][c].at<double>(clusterPointIndices[clusterIdx], 1) = (double)p_color[1];
			people_Points[clusterIdx][c].at<double>(clusterPointIndices[clusterIdx], 2) = (double)p_color[2];
		}

		clusterPointIndices[clusterIdx]++;
	}

	if (color_models.empty()) {
		color_models.clear();
		
		for (size_t clusterIdx = 0; clusterIdx < 4; clusterIdx++)
		{
			Ptr<EM> em_model = EM::create();
			//Set K
			em_model->setClustersNumber(4);
			//Set covariance matrix type
			em_model->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
			//Convergence condition
			em_model->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1));

			//train
			Mat training_labels;
			em_model->trainEM(people_Points[clusterIdx][1], noArray(), training_labels, noArray());
			
			color_models.push_back(em_model);
		}
	}

	// Online classification
	Mat sample(1, 3, CV_64FC1);
	std::map<int, int> clusterClassifications;
	vector<float> min_diffs;
	vector<float> max_diffs;
	vector<float> cam_bests;
	vector<float> cam_labels;
	for (int clusterIndx = 0; clusterIndx < 4; clusterIndx++) {
		vector<int> clusterLabels;
		vector<vector<double>> avg_model_likelihoods = { 4, { 0.0, 0.0, 0.0, 0.0 } };
		vector<int> labelCounts;

		min_diffs.clear();
		max_diffs.clear();
		cam_bests.clear();
		cam_labels.clear();
		
		for (int row = 0; row < people_Points[clusterIndx][0].rows; row++) {
			
			for (int cam = 0; cam < 4; cam++) {
				sample.at<double>(0) = people_Points[clusterIndx][cam].at<double>(row, 0);
				sample.at<double>(1) = people_Points[clusterIndx][cam].at<double>(row, 1);
				sample.at<double>(2) = people_Points[clusterIndx][cam].at<double>(row, 2);

				for (int modelIndx = 0; modelIndx < 4; modelIndx++) {
					Vec2d predict = color_models[modelIndx]->predict(sample, noArray());
					Vec2d predict2 = color_models[modelIndx]->predict2(sample, noArray());

					double likelihood = predict2[0];
					avg_model_likelihoods[modelIndx][cam] += likelihood;
				}
			}
		}
		
		vector<float> local_diffs = vector<float>(3);
		float clusterSize = (float)people_Points[clusterIndx][0].rows;
		for (int cam = 0; cam < 4; cam++) {

			// Average likelihoods and collect cam best + cam label
			float cam_best = -1e15;
			float cam_label;
			for (int mod = 0; mod < 4; mod++) {
				avg_model_likelihoods[mod][cam] /= clusterSize;
				if (avg_model_likelihoods[mod][cam] > cam_best) {
					cam_best = avg_model_likelihoods[mod][cam];
					cam_label = mod;
				}
			}

			// Collect local_diffs
			for (int mod = 0; mod < 4; mod++) {
				if (avg_model_likelihoods[mod][cam] != cam_best) {
					local_diffs[mod] = cam_best - avg_model_likelihoods[mod][cam];
				}
			}
			float min_diff = *min_element(local_diffs.begin(), local_diffs.end());
			float max_diff = *max_element(local_diffs.begin(), local_diffs.end());

			min_diffs.push_back(min_diff);
			max_diffs.push_back(max_diff);
			cam_bests.push_back(cam_best);
			cam_labels.push_back(cam_label);
		}

		// Weight likelihoods according to minimum difference
		vector<float> weighted_likelihoods;
		float largest_mindiff = *max_element(min_diffs.begin(), min_diffs.end());
		for (int c = 0; c < 4; c++) {
			float min_diff = min_diffs[c];
			float cam_best = cam_bests[c];
			float cam_weight = min_diff / largest_mindiff;
			float weighted_likelihood = cam_best * cam_weight;
			weighted_likelihoods.push_back(weighted_likelihood);
		}
		int cam_with_highest_weighted_likelihood = max_element(weighted_likelihoods.begin(), weighted_likelihoods.end()) - weighted_likelihoods.begin();
		float final_label = cam_labels[cam_with_highest_weighted_likelihood];

		clusterClassifications[clusterIndx] = final_label;

		// Add vertices into main center vector
		center_coordinates[final_label].push_back(centers[clusterIndx]);

		if (m_cameras[0]->getVideoFrameIndex() == 50) {
			for (size_t i = 0; i < 4; i++)
			{
				for (size_t j = 0; j < center_coordinates[i].size(); j++)
				{
					cout << center_coordinates[i][j] << endl;
				}
			}
		}
	}
	//waitKey(0);

	// Assign colors to each voxel based on GMM predictions
	for (int i = 0; i < (int)m_visible_voxels.size(); i++) {
		int clusterIdx = labels.at<int>(i);
		int label = clusterClassifications[clusterIdx];
		m_visible_voxels[i]->color = colorTab[label];
	}
	
}

} /* namespace nl_uu_science_gmt */
