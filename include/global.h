//
// Created by lab on 2021/8/25.
//

#ifndef LIVOX_CAMERA_CALIB_GLOBAL_H
#define LIVOX_CAMERA_CALIB_GLOBAL_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <mutex>
#include <thread>
#include <binders.h>
#include <boost/thread.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/imgproc/types_c.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PCLPointCloud2.h> //PCL is migrating to PointCloud2

#include <pcl/common/common_headers.h>

//will use filter objects "passthrough" and "voxel_grid" in this example
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h> //PCL的PCD格式文件的输入输出头文件
#include <pcl/point_types.h> //PCL对各种格式的点的支持头文件
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
//#include <pcl/filters/statistical_outlier_oval.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "getfile.h"
#include "tic_toc.h"
#include "PinholeCam.h"
#include <omp.h>

typedef pcl::PointXYZI PoinT;
typedef pcl::PointXYZRGB DisplayType;

using namespace std;
using namespace pcl;


struct param
{
//    board info=================================================
    Eigen::Vector2i board_size;
    double gride_side_length;
    double gride_diagonal_length;
    int need_undistort;
    int board_order;

//  board segmentation params ===========================
    int noise_reflactivity_thd;
    double angle_thd;
    double board_edge_thd;

//    path info=================================================
    string data_root_path;
    string cheesboard_path;
    string img_path;
    string raw_pc_path;
    string error_path_pnp;
    string error_path_opt;
    string framesets_id_path;

    string sim_img_corner_path;
    string sim_pc_corner_path;

    Eigen::Vector3d sim_gt_cl_t;
    Eigen::Vector3d sim_gt_cl_rpy;
    Eigen::Matrix3d sim_gt_R_cl;

//    for black block extract==================================
    double reflactivity_dec_step;
    double block_us_radius;
    double black_pt_cluster_dst_thd;

//    grid fitting optimization parameters==================================
    double uniform_sampling_radius;
    double neighbour_radius_rate;
    double cost_radius_rate;
    int max_iter_time;

//    for final calibration parameters=====================================
    int use_pnp_init;

//    others=====================================================
    string corner_detect_method;
    double debug_param;
    int apply_noise_test;
    int DEBUG_SHOW;
    int display_remove_oclusion;
    double noise_x;  //noise for debug
    double noise_y;
    double noise_yaml;

};

extern param CfgParam;
extern PinholeCam Cam;
extern Eigen::Matrix4f T_base;

void
readParameters(std::string config_file);

#endif //LIVOX_CAMERA_CALIB_GLOBAL_H
