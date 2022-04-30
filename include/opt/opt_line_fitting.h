//
// Created by lab on 2021/6/19.
//
#pragma once

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <ceres/ceres.h>
#include <chrono>
#include "global.h"


using namespace std;

// 代价函数的计算模型

struct LINE_FITTING_COST
{

    LINE_FITTING_COST(Eigen::Vector3d &_centroid_point, vector<Eigen::Vector3d>& _row_points) :
    centroid_point(_centroid_point), row_points(_row_points){

    }
    // 残差的计算
    template<typename T>
    bool
    operator()(
        const T *const abcd,     // 模型参数，有4维
        T *residual) const     // 残差
    {

        residual[0] = T(0);
//        点到平面的距离
        Eigen::Matrix<T,3,1> board_x_axis;
        board_x_axis.x() = abcd[0];
        board_x_axis.y() = abcd[1];
        board_x_axis.z() = abcd[2];
        board_x_axis.normalize();
        for (int i = 0; i < row_points.size(); ++i)
        {
            residual[0] += T((row_points[i] - centroid_point).template cast<T>().cross(board_x_axis).norm());
        }
        return true;
    }
    Eigen::Vector3d centroid_point;
    vector<Eigen::Vector3d> row_points;
};

void
line_fitting(vector<Eigen::Vector3d> row_lists, Eigen::Matrix<double,6,1> &param)
{
    vector<pair<vector<Eigen::Vector3d>,Eigen::Vector3d>> row_line_points; // points centroid

//    cout<<" ================== start solve plane parameters ================== "<<endl;
    double abc[3] = {param[0], param[1],param[2]}; // abc参数的估计值给定初始值
    int origin_id = CfgParam.board_size.y()/2;
    // 构建最小二乘问题
    ceres::Problem problem;
//    param.bottomRows(3).setZero();
    int cnt = 0;
    for (int row = 0; row < CfgParam.board_size.y(); ++row)
    {
        vector<Eigen::Vector3d> points;
        Eigen::Vector3d centroid_point(0,0,0);
        for (int col = 0; col < CfgParam.board_size.x() / 2 ; ++col)
        {
            points.push_back(row_lists[cnt]);
            centroid_point += row_lists[cnt];
//            param.bottomRows(3) +=row_lists[cnt];
            cnt++;
        }
        centroid_point /= double(CfgParam.board_size.x() / 2);

        if (row == origin_id)
            param.bottomRows(3) = centroid_point;

        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);

        problem.AddResidualBlock(     // 向问题中添加误差项
            // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
            new ceres::AutoDiffCostFunction<LINE_FITTING_COST, 1, 3>(
                new LINE_FITTING_COST(centroid_point, points)
            ),
            loss_function,            // 核函数，这里不使用，为空
            abc                 // 待估计参数
        );
    }
    //param.bottomRows(3) /= double(row_lists.size());

    // 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;                // 优化信息
    ceres::Solve(options, &problem, &summary);  // 开始优化

    param.topRows(3) = Eigen::Vector3d (abc[0], abc[1], abc[2]);
    param.topRows(3).normalize();

    return;
}

