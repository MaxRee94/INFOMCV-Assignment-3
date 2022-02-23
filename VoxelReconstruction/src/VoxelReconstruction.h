/*
 * VoxelReconstruction.h
 *
 *  Created on: Nov 13, 2013
 *      Author: coert
 */

#ifndef VOXELRECONSTRUCTION_H_
#define VOXELRECONSTRUCTION_H_

#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include "controllers/Camera.h"

using namespace cv;
namespace nl_uu_science_gmt
{

class VoxelReconstruction
{
	const std::string m_data_path;
	const int m_cam_views_amount;

	std::vector<Camera*> m_cam_views;

public:
	VoxelReconstruction(const std::string &, const int);
	virtual ~VoxelReconstruction();

	std::vector<Camera*> get_cam_views();
	static void showKeys();

	void run(int, char**);
};

} /* namespace nl_uu_science_gmt */

#endif /* VOXELRECONSTRUCTION_H_ */
