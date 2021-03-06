/*
 * Reconstructor.h
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#ifndef RECONSTRUCTOR_H_
#define RECONSTRUCTOR_H_

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <stddef.h>
#include <vector>

#include "Camera.h"

namespace nl_uu_science_gmt
{

class Reconstructor
{
public:
	/*
	 * Voxel structure
	 * Represents a 3D pixel in the half space
	 */

	bool voxel_post_processing;

	struct Voxel
	{
		int x, y, z;                               // Coordinates
		cv::Scalar color;                          // Color
		std::vector<cv::Point> camera_projection;  // Projection location for camera[c]'s FoV (2D)
		std::vector<int> valid_camera_projection;  // Flag if camera projection is in camera[c]'s FoV
		int closest_camera_index;
	};

private:
	const std::vector<Camera*> &m_cameras;  // vector of pointers to cameras
	const int m_height;                     // Cube half-space height from floor to ceiling
	const int m_step;                       // Step size (space between voxels)

	std::vector<cv::Point3f*> m_corners;    // Cube half-space corner locations


	size_t m_voxels_amount;                 // Voxel count
	cv::Size m_plane_size;                  // Camera FoV plane WxH

	float m_height_thresh = 0.05;			// Height threshold (fraction of m_height) for determining whether a voxel cloud is a person

	std::vector<Voxel*> m_voxels;           // Pointer vector to all voxels in the half-space
	std::vector<Voxel*> m_visible_voxels;   // Pointer vector to all visible voxels
	std::vector<Voxel*> prev_visible_voxels; // Pointer vector to all voxels of previous frame
	std::vector<cv::Ptr<cv::ml::EM>> color_models;				// vector of color models
	std::vector<std::vector<cv::Point2f>> center_coordinates;	// Coordinates of centers per cluster in every frame

	void initialize();

public:
	Reconstructor(
			const std::vector<Camera*> &, bool = true);
	virtual ~Reconstructor();

	void update();

	const std::vector<Voxel*>& getVisibleVoxels() const
	{
		return m_visible_voxels;
	}

	const std::vector<std::vector<int>> getVisibleVoxelCoords() const
	{
		std::vector<std::vector<int>> visible_voxel_coords;
		for (int i = 0; i < m_visible_voxels.size(); i++) {
			Voxel* m_visible_voxel = m_visible_voxels[i];
			std::vector<int> coords = { m_visible_voxel->x, m_visible_voxel->y, m_visible_voxel->z };
			visible_voxel_coords.push_back(coords);
		}
		return visible_voxel_coords;
	}

	const std::vector<std::vector<int>> getAllVoxelCoords() const
	{
		std::vector<std::vector<int>> voxel_coords;
		for (int i = 0; i < m_voxels.size(); i++) {
			Voxel* m_voxel = m_voxels[i];
			std::vector<int> coords = { m_voxel->x, m_voxel->y, m_voxel->z };
			voxel_coords.push_back(coords);
		}
		return voxel_coords;
	}

	void get_floodfill_subset(std::vector<Voxel*>*, std::vector<int>* included_indices, std::vector<Voxel*>*);

	bool is_person(std::vector<Voxel*>* subset);

	const std::vector<Voxel*>& getVoxels() const
	{
		return m_voxels;
	}

	int get_number_of_voxels() {
		return m_voxels_amount;
	}

	void setVisibleVoxels(
			const std::vector<Voxel*>& visibleVoxels)
	{
		m_visible_voxels = visibleVoxels;
	}

	void setVoxels(
			const std::vector<Voxel*>& voxels)
	{
		m_voxels = voxels;
	}

	const std::vector<cv::Point3f*>& getCorners() const
	{
		return m_corners;
	}

	int getSize() const
	{
		return m_height;
	}

	const cv::Size& getPlaneSize() const
	{
		return m_plane_size;
	}
};

} /* namespace nl_uu_science_gmt */

#endif /* RECONSTRUCTOR_H_ */
