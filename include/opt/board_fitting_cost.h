//
// Created by lab on 2021/12/10.
//

#ifndef CHECKERBOARD_LC_CALIB_BOARD_FITTING_COST_H
#define CHECKERBOARD_LC_CALIB_BOARD_FITTING_COST_H

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>

#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PCLPointCloud2.h> //PCL is migrating to PointCloud2
#include "global.h"
using ceres::CostFunction;
using ceres::Problem;
using ceres::SizedCostFunction;
using ceres::Solve;
using ceres::Solver;

class BOARD_FITTING_COST:
    public SizedCostFunction<1 /* number of residuals */,
                                                   3 /* size of first parameter */>
{
public:
    virtual ~BOARD_FITTING_COST() {}

    BOARD_FITTING_COST(Eigen::Vector3d _target_point,
                       bool _is_row,
                       pcl::PointCloud<PoinT>::Ptr _cloud_src,
                       pcl::KdTreeFLANN<PoinT>::Ptr _kdtree
    ) :
        target_point(_target_point), is_row(_is_row), kdtree(_kdtree)
    {
        double factor = sqrt(2.0);
//        double factor = 1.2;
        neighbour_radius = CfgParam.neighbour_radius_rate * CfgParam.gride_side_length;
        double cost_radius = CfgParam.cost_radius_rate * CfgParam.gride_side_length;
        double cost_gradient_radius = cost_radius * factor;

        if (neighbour_radius > cost_gradient_radius)
            max_centroid_displacement = neighbour_radius - cost_gradient_radius;
        else
        {
            cost_gradient_radius = 0.9 * neighbour_radius;
            cost_radius = cost_gradient_radius / factor;
            cout << "warnning! neighbour_radius should not be larger than  neighbour_radius / sqrt(2) \n"
                 << "reset cost_gradient_radius to  0.9 * neighbour_radius  = " << cost_gradient_radius << "\n"
                 << "reset cost_radius to cost_gradient_radius / sqrt(2)  = " << cost_radius
                 << endl;
        }
        cost_radius2 = cost_radius * cost_radius;
        cost_gradient_radius2 = cost_gradient_radius * cost_gradient_radius;
//        update neighbour points
        update_neighbour_points(_cloud_src);
    }

    static ceres::CostFunction *
    Create(Eigen::Vector3d _target_point,
           bool _is_row,
           pcl::PointCloud<PoinT>::Ptr _cloud_src,
           pcl::KdTreeFLANN<PoinT>::Ptr _kdtree
    )
    {
        return new BOARD_FITTING_COST(_target_point, _is_row, _cloud_src, _kdtree);
    }

    virtual bool
    Evaluate(double const *const *parameters,
             double *residuals,
             double **jacobians) const
    {

        double theta = (parameters[0][2] * double(M_PI)) / (180.0);

        Eigen::Quaternion<double> Q1((double) cos(theta / 2.0), (0.00), (0.0), (sin(theta / 2.0)));

        double cost_intensity_down = 0.0;
        double cost_intensity_up = 0.0;
        double cost_intensity_left = 0.0;
        double cost_intensity_right = 0.0;
        double cost_cnt_down = 0.0;
        double cost_cnt_up = 0.0;
        double cost_cnt_right = 0.0;
        double cost_cnt_left = 0.0;

        double gradient_intensity_down = 0.0;
        double gradient_intensity_up = 0.0;
        double gradient_intensity_left = 0.0;
        double gradient_intensity_right = 0.0;
        double gradient_cnt_down = 0.0;
        double gradient_cnt_up = 0.0;
        double gradient_cnt_right = 0.0;
        double gradient_cnt_left = 0.0;

        double interval_intensity_down = 0.0;
        double interval_intensity_up = 0.0;
        double interval_intensity_left = 0.0;
        double interval_intensity_right = 0.0;
        double interval_cnt_down = 0.0;
        double interval_cnt_up = 0.0;
        double interval_cnt_right = 0.0;
        double interval_cnt_left = 0.0;

        int inlier_cnt = 0;

        Eigen::Matrix<double, 3, 3> R_se2_wl = Q1.matrix().template cast<double>();
        Eigen::Matrix<double, 3, 1>
            t_se2_wl = Eigen::Matrix<double, 3, 1>(parameters[0][0], parameters[0][1], double(0.0));

        Eigen::Matrix<double, 3, 1> center_l = R_se2_wl.transpose() * (target_point.template cast<double>() - t_se2_wl);
        double centroid_displacement = (center_l - target_point.template cast<double>()).norm();
        double punishment = double(CfgParam.debug_param);

        double ref_x = double(target_point.x());
        double ref_y = double(target_point.y());

        for (int i = 0; i < neighbour_points->size(); ++i)
        {
            PoinT &neighbout_pt = neighbour_points->points[i];

            double neig_x = double(neighbout_pt.x);
            double neig_y = double(neighbout_pt.y);
            double neig_intensity = double(neighbout_pt.intensity);

            Eigen::Matrix<double, 3, 1>
                transfered_point = R_se2_wl * (Eigen::Matrix<double, 3, 1>(neig_x, neig_y, double(0.0)) + t_se2_wl);

            double neig_x_transfered = transfered_point.x();
            double neig_y_transfered = transfered_point.y();

            double dx = max(double(10e-15), abs(ref_x - neig_x_transfered));
            double dy = max(double(10e-15), abs(ref_y - neig_y_transfered));
            double distance_2 = dx * dx + dy * dy;

            if (distance_2 > cost_gradient_radius2)
            {
                continue;  //Calculate the cost within the current search scope
            }

            if (neig_y_transfered > ref_y)
            {
                gradient_intensity_down += neig_intensity;
                gradient_cnt_down += 1;
                if (distance_2 < cost_radius2)
                {
                    cost_intensity_down += neig_intensity;
                    cost_cnt_down += 1;
                }
                else
                {
                    interval_intensity_down += neig_intensity;
                    interval_cnt_down += 1;
                }
            }
            else
            {
                gradient_intensity_up += neig_intensity;
                gradient_cnt_up += 1;
                if (distance_2 < cost_radius2)
                {
                    cost_intensity_up += neig_intensity;
                    cost_cnt_up += 1;
                }
                else
                {
                    interval_intensity_up += neig_intensity;
                    interval_cnt_up += 1;
                }
            }

            if (neig_x_transfered > ref_x)
            {
                gradient_intensity_right += neig_intensity;
                gradient_cnt_right += 1;
                if (distance_2 < (cost_radius2))
                {
                    cost_intensity_right += neig_intensity;
                    cost_cnt_right += 1;
                }
                else
                {
                    interval_intensity_right += neig_intensity;
                    interval_cnt_right += 1;
                }
            }
            else
            {
                gradient_intensity_left += neig_intensity;
                gradient_cnt_left += 1;

                if (distance_2 < cost_radius2)
                {
                    cost_intensity_left += neig_intensity;
                    cost_cnt_left += 1;
                }
                else
                {
                    interval_intensity_left += neig_intensity;
                    interval_cnt_left += 1;
                }
            }

        }
        double C_d = cost_intensity_down / cost_cnt_down;
        double C_u = cost_intensity_up / cost_cnt_up;
        double C_l = cost_intensity_left / cost_cnt_left;
        double C_r = cost_intensity_right / cost_cnt_right;

        double dC_d = interval_intensity_down / interval_cnt_down;
        double dC_u = interval_intensity_up / interval_cnt_up;
        double dC_l = interval_intensity_left / interval_cnt_left;
        double dC_r = interval_intensity_right / interval_cnt_right;

        double cost_ud = abs(C_d - C_u);
        double cost_lr = abs(C_r - C_l);

        double gradient_x = abs(C_r - dC_r) - abs(C_l - dC_l);
        double gradient_y = abs(C_d - dC_d) - abs(C_u - dC_u);

        if (is_row)
        {
//            residuals[0] = 1.0-tanh((cost_ud - cost_lr)/100.0);
            residuals[0] = 1.0-tanh((cost_ud )/100.0);
        }
        else
        {
//            residuals[0] = 1.0-tanh((cost_lr - cost_ud)/100.0);
            residuals[0] = 1.0-tanh((cost_lr )/100.0);
        }


        if (centroid_displacement > double(max_centroid_displacement))
        {
//            residuals[0] *= double(5.0);
//            cout << "warning! out side of search region!!" << endl;
        }

//        calculate Jacobin
        double dC_dx = gradient_x;
        double dC_dy = gradient_y;

        double curr_x = parameters[0][0];
        double curr_y = parameters[0][1];

        double dp_dtheta1 = -curr_x * cos(theta) - curr_y * sin(theta);
        double dp_dtheta2 = curr_x * sin(theta) - curr_y * cos(theta);
        double dC_dtheta = gradient_x * dp_dtheta1 + gradient_y * dp_dtheta2;

        if (jacobians != NULL && jacobians[0] != NULL) {
            jacobians[0][0] = dC_dx;
            jacobians[0][1] = dC_dy;
            jacobians[0][2] = dC_dtheta;
        }
        return true;
    }

    void
    update_neighbour_points(pcl::PointCloud<PoinT>::Ptr _cloud_src)
    {
        PoinT searchPoint;
        vector<int> pointIdxNKNSearch;
        std::vector<float> pointNKNSquaredDistance;
        searchPoint.x = target_point.x();
        searchPoint.y = target_point.y();
        searchPoint.z = target_point.z();
        kdtree->radiusSearch(searchPoint, neighbour_radius, pointIdxNKNSearch, pointNKNSquaredDistance);
        neighbour_points.reset(new PointCloud<PoinT>);
        boost::shared_ptr<std::vector<int>> index_ptr = boost::make_shared<std::vector<int>>(pointIdxNKNSearch);
        pcl::ExtractIndices<PoinT> eifilter(true);
        eifilter.setInputCloud(_cloud_src);
        eifilter.setIndices(index_ptr);
        eifilter.filter(*neighbour_points);
    }

    template<typename T>
    bool
    operator()(const T *xy_yaw_wl, T *residual) const
    {

        T ang = (xy_yaw_wl[2] * T(M_PI)) / T(180.0);

        Eigen::Quaternion<T> Q1((T) cos(ang / 2.0), T(0), T(0), T(sin(ang / 2.0)));

        T intensity_down = T(0);
        T intensity_up = T(0);

        T intensity_left = T(0);
        T intensity_right = T(0);

        T cnt_down = T(0);
        T cnt_up = T(0);
        T cnt_right = T(0);
        T cnt_left = T(0);

        int inlier_cnt = 0;

        Eigen::Matrix<T, 3, 3> R_se2_wl = Q1.matrix().template cast<T>();
        Eigen::Matrix<T, 3, 1> t_se2_wl = Eigen::Matrix<T, 3, 1>(xy_yaw_wl[0], xy_yaw_wl[1], T(0.0));

        Eigen::Matrix<T, 3, 1> center_l = R_se2_wl.transpose() * (target_point.template cast<T>() - t_se2_wl);
        T centroid_displacement = (center_l - target_point.template cast<T>()).norm();
        T punishment = T(CfgParam.debug_param);

        T ref_x = T(target_point.x());
        T ref_y = T(target_point.y());

        for (int i = 0; i < neighbour_points->size(); ++i)
        {
            PoinT &neighbout_pt = neighbour_points->points[i];

            T neig_x = T(neighbout_pt.x);
            T neig_y = T(neighbout_pt.y);
            T neig_intensity = T(neighbout_pt.intensity);

            Eigen::Matrix<T, 3, 1>
                transfered_point = R_se2_wl * (Eigen::Matrix<T, 3, 1>(neig_x, neig_y, T(0.0)) + t_se2_wl);

            T neig_x_transfered = transfered_point.x();
            T neig_y_transfered = transfered_point.y();

            T dx = max(T(10e-15), abs(ref_x - neig_x_transfered));
            T dy = max(T(10e-15), abs(ref_y - neig_y_transfered));
            T distance_2 = dx * dx + dy * dy;

            if (distance_2 > T(cost_radius2))
            {
                continue;  //在当前搜索范围内计算代价
            }
            inlier_cnt++;

//           Calculate the first residual
            T cost_ud = neig_intensity * abs(dx);
            if (neig_y_transfered > ref_y)
            {
                intensity_down += cost_ud;
                cnt_down += T(1);
            }
            else
            {
                intensity_up += cost_ud;
                cnt_up += T(1);
            }

//          Calculate the second residual
            T cost_lr = neig_intensity * abs(dy);
            if (neig_x_transfered > T(ref_x))
            {
                intensity_right += cost_lr;
                cnt_right += T(1);
            }
            else
            {
                intensity_left += cost_lr;
                cnt_left += T(1);
            }
        }

        T cost_ud = T((abs(intensity_down / cnt_down - intensity_up / cnt_up)));
        T cost_lr = T((abs(intensity_right / cnt_right - intensity_left / cnt_left)));

        if (is_row)
        {
            residual[0] = T(100.0) / (cost_ud - cost_lr);
        }
        else
        {
            residual[0] = T(100.0) / (cost_lr - cost_ud);
        }
        if (inlier_cnt == 0)
        {
            residual[0] = T(10e20);
        }
        else if (centroid_displacement > T(max_centroid_displacement))  //说明跑到外面去了
        {
            residual[0] *= T(5.0);
            cout << "warning! out side of search region!!" << endl;
        }
        return true;
    }

    Eigen::Vector3d target_point;
    bool is_row;
    pcl::KdTreeFLANN<PoinT>::Ptr kdtree;
    double neighbour_radius;
    double cost_radius2;
    double cost_gradient_radius2;
    pcl::PointCloud<PoinT>::Ptr neighbour_points;
    double max_centroid_displacement;
};

#endif //CHECKERBOARD_LC_CALIB_BOARD_FITTING_COST_H
