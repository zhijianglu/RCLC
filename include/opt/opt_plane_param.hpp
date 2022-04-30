//
// Created by lab on 2021/6/19.
//
#pragma once

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


using namespace std;

// 代价函数的计算模型

struct PLANE_FITTING_COST
{

    PLANE_FITTING_COST(double x, double y, double z) : _x(x), _y(y), _z(z) {}

    template<typename T>
    bool
    operator()(
        const T *const abcd,
        T *residual) const
    {
//        Distance from point to plane
        residual[0] = T(abs(abcd[0] * _x + abcd[1] * _y + abcd[2] * _z + abcd[3])
                            / sqrt(
                                abcd[0] * abcd[0] + abcd[1] * abcd[1] + abcd[2] * abcd[2]
                            ));
        return true;
    }
    const double _x, _y, _z;
};

int
pc_plane_fitting(pcl::PointCloud<PoinT>::Ptr &Points, Eigen::Vector4d &param, double reflactive_thd = 100)
{
//    cout<<" ================== start solve plane parameters ================== "<<endl;
    double abcd[4] = {0.1, 0.1, 1, 1}; // Estimation of abcd parameters

    // Constructing least squares problem
    ceres::Problem problem;

    for (int i = 0; i < Points->size(); i++)
    {
        if (Points->points[i].intensity < reflactive_thd)
        {
            // The points with too low reflectivity are unreliable,
            // including the black points on the calibration plate.
            // The reflectivity is too low and the depth is unreliable
            continue;
        }
        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PLANE_FITTING_COST, 1, 4>(
                new PLANE_FITTING_COST(Points->points[i].x,
                                       Points->points[i].y,
                                       Points->points[i].z)
            ),
            loss_function,
            abcd
        );
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    param = Eigen::Vector4d(abcd[0], abcd[1], abcd[2], abcd[3]);

    return 0;
}

