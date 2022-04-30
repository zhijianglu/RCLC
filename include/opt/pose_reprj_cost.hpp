//
// Created by lab on 2021/6/19.
//
#ifndef POSE_REPRJ_COST_FUNCTIONS_H
#define POSE_REPRJ_COST_FUNCTIONS_H


#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

#include "../global.h"
#include "types.h"

using namespace std;

struct POSE_REPRJ_COST
{
    POSE_REPRJ_COST(const vector<Eigen::Vector3d> pc_pts,
                    const vector<Eigen::Vector2d> img_pts) :
        _pc_pts(pc_pts),
        _img_pts(img_pts) {}

    template<typename T>
    bool
    operator()(const T *q, const T *t, T *residual) const
    {
        Eigen::Quaternion<T> Tcl_q{q[3], q[0], q[1], q[2]};
        Eigen::Matrix<T, 3, 1> Tcl_t{t[0], t[1], t[2]};
        Eigen::Matrix<T, 3, 3> Tcl_R = Tcl_q.matrix();

        residual[0] = T(0);
        T depth_max = T(0);
        for (int i = 0; i < _img_pts.size(); ++i)
        {
            Eigen::Matrix<T, 3, 1> pc_l = Tcl_R * _pc_pts[i].template cast<T>() + Tcl_t;
            T depth_norm = pc_l.norm();
            if (depth_norm > depth_max)
            {
                depth_max = depth_norm;
            }

            Eigen::Matrix<T, 2, 1>
                err_vec(T(_img_pts[i].x()) - pc_l.x() / pc_l.z(), T(_img_pts[i].y()) - pc_l.y() / pc_l.z());

            T error = (err_vec.norm() / depth_norm);
            residual[0] += error;
        }
        residual[0] *= depth_max;
        return true;
    }

    static ceres::CostFunction *
    Create(const vector<Eigen::Vector3d> pc_pts,
           const vector<Eigen::Vector2d> img_pts
    )
    {
        return (new ceres::AutoDiffCostFunction<
            POSE_REPRJ_COST, 1, 4, 3>(
            new POSE_REPRJ_COST(pc_pts,
                                img_pts)));
    }

    const vector<Eigen::Vector3d> _pc_pts;
    const vector<Eigen::Vector2d> _img_pts;
};

#endif