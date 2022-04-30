//
// Created by lab on 2021/12/13.
//

#ifndef CHECKERBOARD_LC_CALIB_OPT_REPRJ_CALIB_H
#define CHECKERBOARD_LC_CALIB_OPT_REPRJ_CALIB_H
#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PCLPointCloud2.h> //PCL is migrating to PointCloud2
#include "global.h"
#include "pose_reprj_cost.hpp"

void
pose_reprj_opt(
    vector<vector<Eigen::Vector2d>> &img_norm_points,
    vector<vector<Eigen::Vector3d>> &pc_corner_points,
    double *Tcl,
    bool debug_show = false)
{

    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.2);
    ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    problem.AddParameterBlock(Tcl, 4, q_parameterization);
    problem.AddParameterBlock(Tcl + 4, 3);

    if (debug_show)
        cout << " ================== start solve pose ================== " << endl;

    for (int frame_idx = 0; frame_idx < pc_corner_points.size(); ++frame_idx)
    {
        ceres::CostFunction *cost_function;
        cost_function = POSE_REPRJ_COST::Create(pc_corner_points[frame_idx],
                                                img_norm_points[frame_idx]
        );
        problem.AddResidualBlock(cost_function,
                                 loss_function,
                                 Tcl, Tcl + 4);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 1000;
//CfgParam.debug_param
    if (debug_show)
        options.minimizer_progress_to_stdout = true;
    options.check_gradients = false;
    options.function_tolerance = 1e-15;

    ceres::Solver::Summary sum;
    ceres::Solve(options, &problem, &sum);
    if (debug_show)
        cout << sum.FullReport() << endl;

//    -------------------------------------------
}


#endif //CHECKERBOARD_LC_CALIB_OPT_REPRJ_CALIB_H
