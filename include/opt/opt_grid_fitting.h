//
// Created by lab on 2021/12/9.
//

#ifndef CHECKERBOARD_LC_CALIB_OPT_GRID_FITTING_H
#define CHECKERBOARD_LC_CALIB_OPT_GRID_FITTING_H

#include "global.h"
#include "utils.h"
#include "board_fitting_cost.h"
#include "opt_plane_param.hpp"
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/keypoints/uniform_sampling.h>

void
opt_grid_fitting(pcl::PointCloud<PoinT>::Ptr &cloud_src,
                 Eigen::Matrix4d &T_bl,
                 vector<Eigen::Vector3d> &est_pc_corners,
                 visualization::PCLVisualizer::Ptr viewer = nullptr)
{
//   most time, you don't have to do this
    if (CfgParam.uniform_sampling_radius != 0)
    {
        pcl::UniformSampling<PoinT> US;
        US.setInputCloud(cloud_src);
        US.setRadiusSearch(CfgParam.uniform_sampling_radius);
        US.filter(*cloud_src);
    }

    vector<Eigen::Vector3d> row_points;
    vector<Eigen::Vector3d> col_points;
    generate_std_corners(row_points, col_points);

    pcl::PointCloud<DisplayType>::Ptr clouds2show(new pcl::PointCloud<DisplayType>);

//  If you want to add some disturbance to see the optimization effect
    Eigen::Matrix4f T_noise_w1_w0 = Eigen::Matrix4f::Identity();
    if (CfgParam.apply_noise_test)
    {
        T_noise_w1_w0 =
            SE2_to_SE3<float>(CfgParam.noise_x,
                              CfgParam.noise_y,
                              CfgParam.noise_yaml);  //current to ith world coordinate
        pcl::transformPointCloud(*cloud_src, *cloud_src, T_noise_w1_w0);
        T_bl = T_noise_w1_w0.cast<double>() * T_bl;
    }

    if (viewer != nullptr)
        add_pointClouds_show(viewer, cloud_src);
    pcl::PointCloud<PoinT>::Ptr pc_iter(new pcl::PointCloud<PoinT>);
    pcl::copyPointCloud(*cloud_src, *pc_iter);

    Eigen::Matrix4f T_base_laser;
    T_base_laser.setIdentity();
    bool shouldQuit = false;
    int iter = 0;
    //   Start creating optimization
    while (!shouldQuit)
    {
        double T_se2_wl[3] = {0.0, 0.0, 0.0}; //x, y, yaw Initial value
        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
        ceres::Problem problem;
        problem.AddParameterBlock(T_se2_wl, 3);

        pcl::KdTreeFLANN<PoinT>::Ptr kdtree(new pcl::KdTreeFLANN<PoinT>);
        kdtree->setInputCloud(pc_iter);

        for (int pt_idx = 0; pt_idx < col_points.size(); pt_idx++)
        {
            ceres::CostFunction *cost_function = BOARD_FITTING_COST::Create(
                col_points[pt_idx], false, pc_iter, kdtree);
            problem.AddResidualBlock(cost_function,
                                     loss_function,
                                     T_se2_wl);
        }

        for (int pt_idx = 0; pt_idx < row_points.size(); pt_idx++)
        {
            ceres::CostFunction *cost_function = BOARD_FITTING_COST::Create(
                row_points[pt_idx], true, pc_iter, kdtree);
            problem.AddResidualBlock(cost_function,
                                     loss_function,
                                     T_se2_wl);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 1000;
        options.minimizer_progress_to_stdout = false;
        options.check_gradients = false;
        options.function_tolerance = 1e-15;
        ceres::Solver::Summary sum;
        ceres::Solve(options, &problem, &sum);

        Eigen::Matrix4f
            T_w1_w0 = SE2_to_SE3<float>(T_se2_wl[0], T_se2_wl[1], T_se2_wl[2]);  //current to ith world coordinate
        pcl::transformPointCloud(*pc_iter, *pc_iter, T_w1_w0);

        if (viewer != nullptr)
            add_pointClouds_show(viewer, pc_iter);

        cout << "cost:" << sum.initial_cost << " --> " << sum.final_cost << endl;
        double decrease_rate = (sum.initial_cost - sum.final_cost) / sum.initial_cost;
        iter++;

        if ((abs(decrease_rate) < 1e-6) || iter > CfgParam.max_iter_time)
        {
            if (decrease_rate > 0)
                T_base_laser = T_w1_w0 * T_base_laser;
            shouldQuit = true;
        }
        else
        {
            T_base_laser = T_w1_w0 * T_base_laser;
        }
    }
    cout << "Grid fitting optimization finished" << endl;

    Eigen::Matrix4d final_T_base_laser = T_base_laser.cast<double>() * T_bl;
    Eigen::Matrix4d final_T_laser_base = final_T_base_laser.inverse();
    calc_transfered_standard_corners(est_pc_corners, final_T_laser_base);
}


#endif //CHECKERBOARD_LC_CALIB_OPT_GRID_FITTING_H
