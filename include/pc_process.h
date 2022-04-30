//
// Created by lab on 2021/12/8.
//

#ifndef CHECKERBOARD_LC_CALIB_PC_PROCESS_H
#define CHECKERBOARD_LC_CALIB_PC_PROCESS_H

#include "global.h"
#include "utils.h"
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
#include "opt_line_fitting.h"
#include "opt_grid_fitting.h"

void
cut_pc(pcl::PointCloud<PoinT>::Ptr &cloudIn,
       pcl::PointCloud<PoinT>::Ptr &cloudOut,
       double range_start,
       double range_stop,
       string target)
{
    pcl::PassThrough<PoinT> pass_intensity;
    pass_intensity.setInputCloud(cloudIn);//这个参数得是指针，类对象不行
    pass_intensity.setFilterFieldName(target);//设置想在哪个坐标轴上操作
    pass_intensity.setFilterLimits(range_start, range_stop);
    pass_intensity.setFilterLimitsNegative(false);//保留（true就是删除，false就是保留而删除此区间外的）
    pass_intensity.filter(*cloudOut);//输出到结果指针
}

void
pc_plane_prj(pcl::PointCloud<PoinT>::Ptr &Points, Eigen::Vector4d &plane_param_l)
{
    int tol_num = Points->size();
#pragma omp parallel
#pragma omp for
    for (int pc_idx = 0; pc_idx < tol_num; ++pc_idx)
    {
        PoinT &pt = Points->points[pc_idx];
        Eigen::Vector2d point_direction = Eigen::Vector2d(pt.x / pt.z, pt.y / pt.z);
        Points->points[pc_idx].z = -plane_param_l.w()
            / (plane_param_l.x() * double(point_direction[0]) + plane_param_l.y() * double(point_direction[1])
                + plane_param_l.z());
        pt.x = pt.z * point_direction[0];
        pt.y = pt.z * point_direction[1];
    }
}

void
EulerCluster(PointCloud<PoinT>::Ptr &target_clouds, PointCloud<PoinT>::Ptr &output)
{
    vector<PointIndices> ece_inlier;
    search::KdTree<PoinT>::Ptr tree(new search::KdTree<PoinT>);
    EuclideanClusterExtraction<PoinT> ece;
    tree.reset();
    ece_inlier.clear();
    ece.setInputCloud(target_clouds);
    ece.setClusterTolerance(0.03);  //这里将标定板上所有的黑色块都要能聚成一类,边缘部分的反射率低的要剔除,所以这里设置成0.04,只要能和边缘部分区分就行
    ece.setMinClusterSize(500);
    ece.setMaxClusterSize(2500000);
    ece.setSearchMethod(tree);
    ece.extract(ece_inlier);
    boost::shared_ptr<std::vector<int>>
        index_ptr = boost::make_shared<std::vector<int>>(ece_inlier[0].indices);  //把最多的挑出来就行了
    pcl::ExtractIndices<PoinT> eifilter(true);
    eifilter.setInputCloud(target_clouds);
    eifilter.setIndices(index_ptr);
    eifilter.filter(*output);
}

void cut_roi(PointCloud<PoinT>::Ptr& raw_board, PoinT min_p,PoinT max_p){

    pcl::PassThrough<PoinT> pass;
    pass.setInputCloud(raw_board);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(min_p.x , max_p.x );
    pass.setFilterLimitsNegative(false);//保留（true就是删除，false就是保留而删除此区间外的）
    pass.filter(*raw_board);

    pass.setInputCloud(raw_board);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(min_p.y , max_p.y );
    pass.setFilterLimitsNegative(false);
    pass.filter(*raw_board);

    pass.setInputCloud(raw_board);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(min_p.z , max_p.z );
    pass.setFilterLimitsNegative(false);//保留（true就是删除，false就是保留而删除此区间外的）
    pass.filter(*raw_board);//输出到结果指针
}


void
eular_cluster_iter(const pcl::PointCloud<PoinT>::Ptr &cloudInRaw,
                   double reflactive_thd,
                   vector<Eigen::Vector3d> &block_centroid,
                   visualization::PCLVisualizer::Ptr viewer = nullptr)
{
//  First, uniform distribution
    pcl::PointCloud<PoinT>::Ptr blocks_cloud(new pcl::PointCloud<PoinT>);
    pcl::UniformSampling<PoinT> US;
    US.setInputCloud(cloudInRaw);
    US.setRadiusSearch(CfgParam.block_us_radius);
    US.filter(*blocks_cloud);
    vector<PointCloud<PoinT>::Ptr> block_clouds_inlier;

    vector<int> block_centroid_inlier_num;
    int block_num = CfgParam.board_size.x() * CfgParam.board_size.y() / 2;
    bool shoule_quit = false;
    double curr_reflactive_thd = reflactive_thd / CfgParam.reflactivity_dec_step;
    int iterative_time = 0;
    while (!shoule_quit)
    {
        iterative_time++;
        curr_reflactive_thd *= CfgParam.reflactivity_dec_step;
        pcl::PassThrough<PoinT> pass_intensity;
        pass_intensity.setInputCloud(blocks_cloud);
        pass_intensity.setFilterFieldName("intensity");
        pass_intensity.setFilterLimits(3, curr_reflactive_thd);
        pass_intensity.setFilterLimitsNegative(false);
        pass_intensity.filter(*blocks_cloud);

        vector<PointIndices> ece_inlier;
        search::KdTree<PoinT>::Ptr tree(new search::KdTree<PoinT>);
        EuclideanClusterExtraction<PoinT> ece;
        ece.setInputCloud(blocks_cloud);
        ece.setClusterTolerance(CfgParam.black_pt_cluster_dst_thd);
        ece.setMinClusterSize(3);
        ece.setMaxClusterSize(25000);
        ece.setSearchMethod(tree);
        ece.extract(ece_inlier);

        ExtractIndices<PoinT> ext;
        ext.setInputCloud(blocks_cloud);

        if (ece_inlier.size() < block_num)
        {
            continue;
        }

        for (int i_seg = 0; i_seg < ece_inlier.size(); i_seg++)
        {
            PointCloud<PoinT>::Ptr ext_cloud(new PointCloud<PoinT>);
            boost::shared_ptr<std::vector<int>>
                index_ptr = boost::make_shared<std::vector<int>>(ece_inlier[i_seg].indices);
            pcl::ExtractIndices<PoinT> eifilter(true);
            eifilter.setInputCloud(blocks_cloud);
            eifilter.setIndices(index_ptr);
            eifilter.filter(*ext_cloud);

            Eigen::Vector4f centroid_inlier_f;
            Eigen::Vector3d centroid_curr;
            pcl::compute3DCentroid(*ext_cloud, centroid_inlier_f);
            centroid_curr = centroid_inlier_f.topRows(3).cast<double>();

            if (i_seg < block_num)
            {
                if (i_seg > 0 && (double(ext_cloud->size()) / double(block_clouds_inlier[i_seg - 1]->size())) < 0.65)
                {
                    break;
                }
                block_clouds_inlier.push_back(ext_cloud);
                block_centroid.push_back(centroid_curr);
                block_centroid_inlier_num.push_back(ext_cloud->size());
            }
            else
            {
                int nearest_block_idx = -1;
                double nearest_dst = 10e3;
                for (int i = 0; i < block_centroid.size(); ++i)
                {
                    float dst = (block_centroid[i] - centroid_curr).norm();
                    if (dst < nearest_dst)
                    {
                        nearest_dst = dst;
                        nearest_block_idx = i;
                    }
                }

                if (nearest_dst < 0.5 * CfgParam.gride_diagonal_length)
                {
                    Eigen::Vector3d &old = block_centroid[nearest_block_idx];
                    block_centroid[nearest_block_idx] =
                        (old * block_centroid_inlier_num[nearest_block_idx] + centroid_curr * ext_cloud->size())
                            / double(block_centroid_inlier_num[nearest_block_idx] + ext_cloud->size());
                    block_centroid_inlier_num[nearest_block_idx] += ext_cloud->size();
                    *block_clouds_inlier[nearest_block_idx] = *block_clouds_inlier[nearest_block_idx] + *ext_cloud;
                }
            }
        }

        if (block_centroid.size() < block_num)
        {
            block_clouds_inlier.clear();
            block_centroid.clear();
            block_centroid_inlier_num.clear();
            continue;
        }
        shoule_quit = true;
    }

    if (viewer != NULL)
    {
        viewer->removeAllPointClouds();
        PointCloud<DisplayType>::Ptr block_centroid_clouds(new PointCloud<DisplayType>);
        assert(block_clouds_inlier.size() == block_num);
        for (int i = 0; i < block_clouds_inlier.size(); ++i)
        {
            Eigen::Vector3d &centroid_curr = block_centroid[i];
            DisplayType block_pts;

            int rgb[3];
            rand_rgb(rgb);

            block_pts.x = centroid_curr.x();
            block_pts.y = centroid_curr.y();
            block_pts.z = centroid_curr.z();
            block_pts.r = rgb[0];
            block_pts.g = rgb[1];
            block_pts.b = rgb[2];
            block_centroid_clouds->push_back(block_pts);
            visualization::PointCloudColorHandlerCustom<PoinT> rgb2(block_clouds_inlier[i], rgb[0], rgb[1], rgb[2]);
            viewer->addPointCloud(block_clouds_inlier[i], rgb2, "block_clouds" + to_string(i), 0);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                     1,
                                                     "block_clouds" + to_string(i));
        }
        viewer->addPointCloud<DisplayType>(block_centroid_clouds, "block_clouds_centroid", 0);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                 10,
                                                 "block_clouds_centroid");
        viewer->spin();
    }
}

int
calc_laser_coor(vector<Eigen::Vector3d> &v_grid_centroid,
                Eigen::Vector4d &board_plane_param,
                Eigen::Matrix4d &Tbl,
                visualization::PCLVisualizer::Ptr viewer = nullptr,
                pcl::PointCloud<PoinT>::Ptr BoardCloudIn = nullptr)
{
    Tbl.setIdentity();
    Eigen::Vector3d ref0;
    int ref_pt_idx;
    get_ref_pt(v_grid_centroid, ref0, ref_pt_idx);
    //    计算所有点与参考点的距离
    vector<pair<double, int>> ref_distance_lists;  // distance, index
    for (int i = 0; i < v_grid_centroid.size(); ++i)
    {
        if (i != ref_pt_idx)
        {
            ref_distance_lists.push_back(make_pair((v_grid_centroid[i] - v_grid_centroid[ref_pt_idx]).norm(), i));
        }
    }
    std::sort(ref_distance_lists.begin(), ref_distance_lists.end());

    //x轴是第一排点的方向，方向为自左往右
    Eigen::Vector3d board_x_axis;
    Eigen::Vector3d &ref1 = v_grid_centroid[ref_distance_lists[1].second];
    Eigen::Vector3d &ref2 = v_grid_centroid[ref_distance_lists[2].second];
    if (ref1.x() < ref2.x() && ref1.y() < ref2.y())//只要标定板roll角度别超过45度，都能满足，所以采集数据的时候尽量满足这一点
        board_x_axis = (ref0 - ref1).normalized();
    else
        board_x_axis = (ref0 - ref2).normalized();

//    准备svd分解求直线拟合：
    int n_pt = CfgParam.board_size.x() / 2;
    Eigen::MatrixXd A_j;
    A_j.resize(n_pt, 2);
    Eigen::Vector3d center_point;
    vector<Eigen::Vector3d> v_first_row_points;
    for (int i = 0; i < v_grid_centroid.size(); ++i)
    {
        Eigen::Vector3d &pt = v_grid_centroid[i];
        double dist = (ref0 - pt).cross(board_x_axis).norm();
        if (dist < CfgParam.gride_side_length / 2.0)
        {
            center_point += pt;
            v_first_row_points.push_back(pt);
        }
    }

    if (v_first_row_points.size() != n_pt)
    {
        cout << "point num of first row:" << v_first_row_points.size() << " != " << n_pt << endl;
        cout
            << "Detect block points filed, you can check the eular cluster iter module for detail.  Abandon this frame!"
            << endl;
        return -1;
    }
    else
    {
        Eigen::Matrix<double, 6, 1> Spatial_line_param;
        calc_row_line_fitting(v_first_row_points, board_plane_param, Spatial_line_param);
        board_x_axis = Spatial_line_param.topRows(3);
        board_x_axis.normalize();
    }

    Eigen::Vector3d board_z_axis = board_plane_param.topRows(3);
    board_z_axis.normalize();
    Eigen::Vector3d board_y_axis = board_x_axis.cross(board_z_axis);
    int origin_col_id = CfgParam.board_size.x() / 2;
    int origin_row_id = CfgParam.board_size.y() / 2;
    Eigen::Vector3d board_coor_origin_tmp(
        -CfgParam.gride_side_length * (origin_col_id - 0.5),
        CfgParam.gride_side_length * (origin_row_id - 0.5),
        0
    );

    if (board_x_axis.x() < 0)
        board_x_axis *= -1;
    if (board_y_axis.y() < 0)
        board_y_axis *= -1;
    if (board_z_axis.z() < 0)
        board_z_axis *= -1;

    Eigen::Matrix3d R_lb;
    R_lb.col(0) = board_x_axis;
    R_lb.col(1) = board_y_axis;
    R_lb.col(2) = board_z_axis;

    Tbl.block(0, 0, 3, 3) = R_lb.transpose();
    Eigen::Vector3d board_coor_origin = R_lb * board_coor_origin_tmp + ref0;
    Tbl.block(0, 3, 3, 1) = -Tbl.block(0, 0, 3, 3) * board_coor_origin;

    if (viewer != nullptr)
    {
        PoinT p1;
        PoinT px;
        PoinT py;
        PoinT pz;
        p1.x = board_coor_origin.x();
        p1.y = board_coor_origin.y();
        p1.z = board_coor_origin.z();
        Eigen::Vector3d x_line = board_coor_origin + 0.4 * board_x_axis;
        px.x = x_line.x();
        px.y = x_line.y();
        px.z = x_line.z();
        viewer->addLine<PoinT>(p1, px, 0, 255, 255, "x_axis");
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 6, "x_axis");

        board_y_axis.normalize();
        Eigen::Vector3d y_line = board_coor_origin + 0.3 * board_y_axis;
        py.x = y_line.x();
        py.y = y_line.y();
        py.z = y_line.z();
        viewer->addLine<PoinT>(p1, py, 0, 255, 255, "y_axis");
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 6, "y_axis");

        Eigen::Vector3d z_line = board_coor_origin + 0.1 * board_z_axis;
        pz.x = z_line.x();
        pz.y = z_line.y();
        pz.z = z_line.z();
        viewer->addLine<PoinT>(p1, pz, 0, 255, 255, "z_axis");
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 6, "z_axis");
        viewer->spin();
        viewer->removeAllShapes();
    }
}

int
calc_laser_coor_v1(vector<Eigen::Vector3d> &v_grid_centroid,
                   Eigen::Vector4d &board_plane_param,
                   Eigen::Matrix4d &Tbl,
                   visualization::PCLVisualizer::Ptr viewer = nullptr)
{
    Tbl.setIdentity();
    Eigen::Vector3d ref0;
    int ref_pt_idx;
    get_ref_pt(v_grid_centroid, ref0, ref_pt_idx);
    // Calculate the distance between all points and the reference point
    vector<pair<double, int>> ref_distance_lists;  // distance, index
    for (int i = 0; i < v_grid_centroid.size(); ++i)
    {
        if (i != ref_pt_idx)
        {
            ref_distance_lists.push_back(make_pair((v_grid_centroid[i] - v_grid_centroid[ref_pt_idx]).norm(), i));
        }
    }
    std::sort(ref_distance_lists.begin(), ref_distance_lists.end());

    //The x-axis is the direction of the first row of points
    Eigen::Vector3d board_x_axis_tmp;
    Eigen::Vector3d &ref1 = v_grid_centroid[ref_distance_lists[1].second];
    Eigen::Vector3d &ref2 = v_grid_centroid[ref_distance_lists[2].second];
    if (ref1.x() < ref2.x() && ref1.y() < ref2.y())//只要标定板roll角度别超过45度，都能满足，所以采集数据的时候尽量满足这一点
        board_x_axis_tmp = (ref0 - ref1).normalized();
    else
        board_x_axis_tmp = (ref0 - ref2).normalized();

//  Extract each point of each line, which is unordered
    vector<pair<double, int>> row_lists;  // distance, index
    for (int i = 0; i < v_grid_centroid.size(); ++i)
    {
        Eigen::Vector3d &pt = v_grid_centroid[i];
        double dist = (ref0 - pt).cross(board_x_axis_tmp).norm();
        row_lists.push_back(make_pair(dist, i));
    }
    std::sort(row_lists.begin(), row_lists.end());
    //The distance from each point to the first row is arranged from small to large, which distinguishes the points of different rows
    vector<Eigen::Vector3d> row_points;
    for (int i = 0; i < row_lists.size(); ++i)
    {
        row_points.push_back(v_grid_centroid[row_lists[i].second]);
    }

    Eigen::Matrix<double, 6, 1> x_axis_param;
    x_axis_param.topRows(3) = board_x_axis_tmp;
    line_fitting(row_points, x_axis_param);
    Eigen::Vector3d board_x_axis = x_axis_param.topRows(3);
    Eigen::Vector3d board_z_axis = board_plane_param.topRows(3);
    board_z_axis.normalize();
    Eigen::Vector3d board_y_axis = board_x_axis.cross(board_z_axis);

    if (board_x_axis.x() < 0)
        board_x_axis *= -1;
    if (board_y_axis.y() < 0)
        board_y_axis *= -1;
    if (board_z_axis.z() < 0)
        board_z_axis *= -1;

    Eigen::Matrix3d R_lb;
    R_lb.col(0) = board_x_axis;
    R_lb.col(1) = board_y_axis;
    R_lb.col(2) = board_z_axis;

    Tbl.block(0, 0, 3, 3) = R_lb.transpose();

    Eigen::Vector3d board_coor_origin_tmp;


    if (CfgParam.board_size.y() % 2 != 0)  //In the case of odd rows
    {
        board_coor_origin_tmp = Eigen::Vector3d(
            -CfgParam.gride_side_length * (0.5),
            -CfgParam.gride_side_length * (0.5),
            0
        );
    }
    else
    {  // Even rows
        board_coor_origin_tmp = Eigen::Vector3d(
            CfgParam.gride_side_length * (0.5),
            -CfgParam.gride_side_length * (0.5),
            0
        );
    }

    Eigen::Vector3d board_coor_origin = R_lb * board_coor_origin_tmp + x_axis_param.bottomRows(3);
    Tbl.block(0, 3, 3, 1) = -Tbl.block(0, 0, 3, 3) * board_coor_origin;
    Tbl.block(0, 3, 3, 1) = -Tbl.block(0, 0, 3, 3) * board_coor_origin;

    if (viewer != nullptr)
    {
        PoinT p1;
        PoinT px;
        PoinT py;
        PoinT pz;
        p1.x = board_coor_origin.x();
        p1.y = board_coor_origin.y();
        p1.z = board_coor_origin.z();
        Eigen::Vector3d x_line = board_coor_origin + 0.4 * board_x_axis_tmp;
        px.x = x_line.x();
        px.y = x_line.y();
        px.z = x_line.z();
        viewer->addLine<PoinT>(p1, px, 0, 255, 255, "x_axis");
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 6, "x_axis");

        board_y_axis.normalize();
        Eigen::Vector3d y_line = board_coor_origin + 0.3 * board_y_axis;
        py.x = y_line.x();
        py.y = y_line.y();
        py.z = y_line.z();
        viewer->addLine<PoinT>(p1, py, 0, 255, 255, "y_axis");
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 6, "y_axis");

        Eigen::Vector3d z_line = board_coor_origin + 0.1 * board_z_axis;
        pz.x = z_line.x();
        pz.y = z_line.y();
        pz.z = z_line.z();
        viewer->addLine<PoinT>(p1, pz, 0, 255, 255, "z_axis");
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 6, "z_axis");
        viewer->spin();
        viewer->removeAllShapes();
    }
}

void
eular_cluster(pcl::PointCloud<PoinT>::Ptr &cloudInRaw,
              pcl::PointCloud<PoinT>::Ptr &cloudOut,
              visualization::PCLVisualizer::Ptr viewer = nullptr)
{

    pcl::PointCloud<PoinT>::Ptr cloudIn(new pcl::PointCloud<PoinT>);
    pcl::UniformSampling<PoinT> US;
    US.setInputCloud(cloudInRaw);
    US.setRadiusSearch(CfgParam.block_us_radius);
    US.filter(*cloudIn);

    vector<PointIndices> ece_inlier;
    search::KdTree<PoinT>::Ptr tree(new search::KdTree<PoinT>);
    EuclideanClusterExtraction<PoinT> ece;
    ece.setInputCloud(cloudIn);
    ece.setClusterTolerance(0.0085);
    ece.setMinClusterSize(3);
    ece.setMaxClusterSize(25000);
    ece.setSearchMethod(tree);
    ece.extract(ece_inlier);

    ExtractIndices<PoinT> ext;
    ext.setInputCloud(cloudIn);

    vector<PointCloud<PoinT>::Ptr> blocks;
    for (int i_seg = 0; i_seg < ece_inlier.size(); i_seg++)
    {
        PointCloud<PoinT>::Ptr ext_cloud(new PointCloud<PoinT>);
        boost::shared_ptr<std::vector<int>>
            index_ptr = boost::make_shared<std::vector<int>>(ece_inlier[i_seg].indices);
        pcl::ExtractIndices<PoinT> eifilter(true);
        eifilter.setInputCloud(cloudIn);
        eifilter.setIndices(index_ptr);
        eifilter.filter(*ext_cloud);
        blocks.push_back(ext_cloud);
    }

//    if(viewer!=NULL){
//        viewer->removeAllPointClouds();
//        for (int i = 0; i < blocks.size(); ++i)
//        {
//            Eigen::Vector4f centroid_inlier_f;
//            Eigen::Vector3d centroid_curr;
//            pcl::compute3DCentroid(*blocks[i], centroid_inlier_f);
//            centroid_curr = centroid_inlier_f.topRows(3).cast<double>();
//
//            int *rgb1 = rand_rgb();
//            visualization::PointCloudColorHandlerCustom<PoinT>rgb2(blocks[i], rgb1[0], rgb1[1], rgb1[2]);
//            delete[]rgb1;
//            viewer->addPointCloud(blocks[i], rgb2, to_string(i));
//        }
//        viewer->spin();
//    }

    if (viewer != NULL)
    {
        PointCloud<DisplayType>::Ptr block_centroid(new PointCloud<DisplayType>);
        for (int i = 0; i < blocks.size(); ++i)
        {
            Eigen::Vector4f centroid_inlier_f;
            Eigen::Vector3d centroid_curr;
            pcl::compute3DCentroid(*blocks[i], centroid_inlier_f);
            centroid_curr = centroid_inlier_f.topRows(3).cast<double>();
            DisplayType block_pts;
            int rgb[3];
            rand_rgb(rgb);
            block_pts.x = centroid_curr.x();
            block_pts.y = centroid_curr.y();
            block_pts.z = centroid_curr.z();
            block_pts.r = rgb[0];
            block_pts.g = rgb[1];
            block_pts.b = rgb[2];
            block_centroid->push_back(block_pts);
        }
        viewer->removeAllPointClouds();
        viewer->addPointCloud<DisplayType>(block_centroid, "centroid");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "centroid");
        viewer->spin();
    }

}

void
est_board_corner(pcl::PointCloud<PoinT>::Ptr &chess_board_processe,
                 vector<Eigen::Vector3d> &board_corner,
                 visualization::PCLVisualizer::Ptr viewer = nullptr)
{

    if (CfgParam.board_order == 1)
    {
        Eigen::Matrix4f T_grid_shift;
        T_grid_shift.setIdentity();
        T_grid_shift(0, 0) = -1.0f;
        pcl::transformPointCloud(*chess_board_processe, *chess_board_processe, T_grid_shift);
    }
    Eigen::Vector4d board_plane_param;
    pc_plane_fitting(chess_board_processe, board_plane_param);
    pc_plane_prj(chess_board_processe, board_plane_param);

    Eigen::VectorXd board_intensity;
    board_intensity.resize(chess_board_processe->size());

#pragma omp parallel
#pragma omp for
    for (int i = 0; i < chess_board_processe->size(); ++i)
        board_intensity[i] = chess_board_processe->points[i].intensity;

    vector<double> mu{15, 100};
    vector<double> sigma{15, 15};
    GMMFit::fit_1d(board_intensity, 2, mu, sigma, false);

    double reflactive_thd = mu[0] + 2.0 * sigma[0];

    vector<Eigen::Vector3d> block_centroid;
    eular_cluster_iter(chess_board_processe, reflactive_thd, block_centroid);
    Eigen::Matrix4d T_bl;
    calc_laser_coor_v1(block_centroid, board_plane_param, T_bl);
    pcl::transformPointCloud(*chess_board_processe, *chess_board_processe, T_bl.cast<float>());
    if (viewer != nullptr)
        display_standard_gride_lines(viewer);

    opt_grid_fitting(chess_board_processe, T_bl, board_corner);

//    you can debug show to intermediate results
    if (viewer != nullptr)
    {
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();
    }

    if (CfgParam.board_order == 1)
    {
        int n_pt_per_row = CfgParam.board_size.x() - 1;
        vector<Eigen::Vector3d> board_corner_tmp(board_corner.size());
        for (int row_idx = 0; row_idx < CfgParam.board_size.y() - 1; ++row_idx)
        {
            for (int col_idx = 0; col_idx < CfgParam.board_size.x() - 1; ++col_idx)
            {
                auto &pt = board_corner[row_idx * n_pt_per_row + col_idx];
                board_corner_tmp[row_idx * n_pt_per_row + n_pt_per_row - col_idx - 1] =
                    Eigen::Vector3d(-pt.x(), pt.y(), pt.z());
            }
        }
        board_corner = board_corner_tmp;
    }
}

#endif //CHECKERBOARD_LC_CALIB_PC_PROCESS_H
