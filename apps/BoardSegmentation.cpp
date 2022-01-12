#include <iostream>
#include "global.h"
#include "utils.h"
#include "GMMFit.h"
#include "visualize.h"
#include "pc_process.h"
#include "opt_plane_param.hpp"
#include "opt_grid_fitting.h"
#include "opt_reprj_calib.h"
#include "global.h"


void
cut_edges(const PointCloud<PoinT>::Ptr &final_board_black_part,
          const Eigen::Vector4d &plane_param,
          PointCloud<PoinT>::Ptr &raw_cloud,
          PointCloud<PoinT>::Ptr &final_output
)
{
//    step1: 计算右上右下,左上左下 四个顶点
    PoinT min_p, max_p;
    pcl::getMinMax3D(*final_board_black_part, min_p, max_p);
    Eigen::Vector3d c(
        (max_p.x + min_p.x) / 2.0,
        (max_p.y + min_p.y) / 2.0,
        (max_p.z + min_p.z) / 2.0
    );
//    Eigen::Vector4f c;
//    pcl::compute3DCentroid(*final_board_black_part, c);
    vector<pair<double, int>> corner_pt{4, make_pair(0.0, 0)};
    vector<Eigen::Vector3d> corner_pt_e;
    for (int idx = 0; idx < final_board_black_part->size(); ++idx)
    {
        PoinT &p = final_board_black_part->points[idx];
        double dx = p.x - c.x();
        double dy = p.y - c.y();
        double dz = p.z - c.z();

        double dst_2 = dx * dx + dy * dy + dz * dz;
        int axis = 0;

        if (p.x > c.x())
        {
            if (p.y > c.y())
                axis = 0;//右下
            else
                axis = 1; //右上
        }
        else
        {
            if (p.y > c.y())
                axis = 2;//左下
            else
                axis = 3;//左上
        }

        if (dst_2 > corner_pt[axis].first)
        {
            corner_pt[axis].first = dst_2;
            corner_pt[axis].second = idx;
        }
    }

    Eigen::Vector3d centroid = c.topRows(3).cast<double>();

    if(CfgParam.board_order==0){
        for (int i = 0; i < 2; ++i)
        {
            Eigen::Vector3d edge_pt(final_board_black_part->points[corner_pt[i].second].x,
                                    final_board_black_part->points[corner_pt[i].second].y,
                                    final_board_black_part->points[corner_pt[i].second].z);
            Eigen::Vector3d board_edge_thd = (edge_pt - centroid).normalized() * CfgParam.board_edge_thd;
            corner_pt_e.emplace_back(edge_pt + board_edge_thd);  //右下，右上
        }
        corner_pt_e.push_back(centroid + 1.0 * (centroid - corner_pt_e[1])); //左下
        corner_pt_e.push_back(centroid + 1.0 * (centroid - corner_pt_e[0])); //左上
    }else{
        corner_pt_e.resize(4);
        for (int i = 2; i < 4; ++i)
        {
            Eigen::Vector3d edge_pt(final_board_black_part->points[corner_pt[i].second].x,
                                    final_board_black_part->points[corner_pt[i].second].y,
                                    final_board_black_part->points[corner_pt[i].second].z);
            Eigen::Vector3d board_edge_thd = (edge_pt - centroid).normalized() * CfgParam.board_edge_thd;
            corner_pt_e[i] = (edge_pt + board_edge_thd);  //右下，右上，左下
        }
        corner_pt_e[0]=(centroid + 1.0 * (centroid - corner_pt_e[3]));
        corner_pt_e[1]=(centroid + 1.0 * (centroid - corner_pt_e[2])); //左上
    }


//    开始切割
    double norm_sq = sqrt(
        plane_param[0] * plane_param[0] + plane_param[1] * plane_param[1] + plane_param[2] * plane_param[2]
    );

//    计算出四个平面来

    Eigen::Vector4d norm_down;
    Eigen::Vector4d norm_up;
    Eigen::Vector4d norm_right;
    Eigen::Vector4d norm_left;
    Eigen::Vector3d plane_norm;

    plane_norm = plane_param.topRows(3);

    norm_down.topRows(3) = ((corner_pt_e[0] - corner_pt_e[2]).cross(corner_pt_e[2])).normalized();
    norm_down[3] = -corner_pt_e[0].dot(norm_down.topRows(3));

    norm_up.topRows(3) = ((corner_pt_e[1] - corner_pt_e[3]).cross(corner_pt_e[3])).normalized();
    norm_up[3] = -corner_pt_e[1].dot(norm_up.topRows(3));

    norm_right.topRows(3) = ((corner_pt_e[1] - corner_pt_e[0]).cross(corner_pt_e[0])).normalized();
    norm_right[3] = -corner_pt_e[1].dot(norm_right.topRows(3));

    norm_left.topRows(3) = ((corner_pt_e[3] - corner_pt_e[2]).cross(corner_pt_e[2])).normalized();
    norm_left[3] = -corner_pt_e[3].dot(norm_left.topRows(3));

    vector<int> field;
    Eigen::Vector3d centroid_e(c.x(), c.y(), c.z());

    field.emplace_back(((centroid_e.dot(norm_up.topRows(3)) + norm_up[3]) > 0) ? 1 : -1);
    field.emplace_back(((centroid_e.dot(norm_down.topRows(3)) + norm_down[3]) > 0) ? 1 : -1);
    field.emplace_back(((centroid_e.dot(norm_right.topRows(3)) + norm_right[3]) > 0) ? 1 : -1);
    field.emplace_back(((centroid_e.dot(norm_left.topRows(3)) + norm_left[3]) > 0) ? 1 : -1);

    for (int i = 0; i < raw_cloud->size(); ++i)
    {
        Eigen::Vector3d p(raw_cloud->points[i].x, raw_cloud->points[i].y, raw_cloud->points[i].z);

        if (
            (abs(p.dot(plane_norm) + plane_param[3]) / norm_sq) > 0.1
            )
        {
            continue;
        }

        if (
            (p.dot(norm_up.topRows(3)) + norm_up[3]) * field[0] < 0
            )
            continue;

        if (
            (p.dot(norm_down.topRows(3)) + norm_down[3]) * field[1] < 0
            )
            continue;

        if (
            (p.dot(norm_right.topRows(3)) + norm_right[3]) * field[2] < 0
            )
            continue;

        if (
            (p.dot(norm_left.topRows(3)) + norm_left[3]) * field[3] < 0
            )
            continue;

        final_output->push_back(raw_cloud->points[i]);
    }

}


int
main(int argc, char *argv[])
{
//    读取参数
    readParameters("../cfg/config_real.yaml");

    visualization::PCLVisualizer::Ptr viewer;
    if (CfgParam.DEBUG_SHOW)
    {
        viewer.reset(new visualization::PCLVisualizer("3d view"));
        viewer->setBackgroundColor(0, 0, 0);
        viewer->addCoordinateSystem(0.3);
    }

    vector<string> pc_path;
    load_file_path(CfgParam.raw_pc_path, pc_path);

//    string pc_path_intput =
//        "/media/lab/915c94cc-246e-42f5-856c-faf39f774d2c/CalibData/real_world_data_outdoor/pc_xyz";
//    vector<int> idx2process{22 ,26 ,48};
//    for (auto pc_idx: idx2process){

#pragma omp parallel
#pragma omp for
    for (int pc_idx = 0; pc_idx < pc_path.size(); ++pc_idx)
    {

        std::string fileData = pc_path[pc_idx];
//        std::string fileData = pc_path_intput+"/"+ to_string(pc_idx)+".pcd";

        PointCloud<PoinT>::Ptr raw_cloud(new PointCloud<PoinT>);
        pcl::io::loadPCDFile<PoinT>(fileData, *raw_cloud);

        cout << " ********************************* " << endl;
        cout << pc_idx << " current processing pointcloud: " << pc_path[pc_idx] << endl;

        //第一次体素化下采样******************************************************
        TicToc timer;
        //距离太远的无效的点删除掉
        pcl::PassThrough<PoinT> pass;
        pass.setInputCloud(raw_cloud);//这个参数得是指针，类对象不行
        pass.setFilterFieldName("z");//设置想在哪个坐标轴上操作
        pass.setFilterLimits(2, 10);
        pass.setFilterLimitsNegative(false);//保留（true就是删除，false就是保留而删除此区间外的）
        pass.filter(*raw_cloud);//输出到结果指针

        VoxelGrid<PoinT> vox;
        PointCloud<PoinT>::Ptr vox_cloud(new PointCloud<PoinT>);
        vox.setInputCloud(raw_cloud);
        vox.setLeafSize(0.005, 0.005, 0.005);
        vox.filter(*vox_cloud);

        //平面分割(RANSAC)********************************************************
        SACSegmentation<PoinT> sac;
        PointIndices::Ptr inliner(new PointIndices);
        ModelCoefficients::Ptr coefficients(new ModelCoefficients);
        PointCloud<PoinT>::Ptr sac_cloud(new PointCloud<PoinT>);
        sac.setInputCloud(vox_cloud);
        sac.setMethodType(SAC_RANSAC);
        sac.setModelType(SACMODEL_PLANE);
        sac.setMaxIterations(800);
        sac.setDistanceThreshold(0.03);

        //提取平面(展示并输出)******************************************************
        PointCloud<PoinT>::Ptr ext_cloud(new PointCloud<PoinT>);
        PointCloud<PoinT>::Ptr ext_cloud_rest(new PointCloud<PoinT>);
        PointCloud<PoinT>::Ptr target_plane(new PointCloud<PoinT>);

        int pt_size = vox_cloud->size();
        double closest_plane_dist = 1000;
        double closest_plane_z = 1000;
        ExtractIndices<PoinT> ext;
        srand((unsigned) time(NULL));//刷新时间的种子节点需要放在循环体外面
        Eigen::Vector4d target_plane_param;
        while (vox_cloud->size() > pt_size * 0.01)//当提取的点数小于总数的3/10时，跳出循环
        {
            ext.setInputCloud(vox_cloud);
            sac.segment(*inliner, *coefficients);

            Eigen::Vector4d plane_param
                (coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);

            //按照索引提取点云*************
            ext.setIndices(inliner);
            ext.setNegative(false);
            ext.filter(*ext_cloud);
            ext.setNegative(true);
            ext.filter(*ext_cloud_rest);
            *vox_cloud = *ext_cloud_rest;

            if ( abs(plane_param.topRows(3).normalized().z()) <= CfgParam.angle_thd )
                continue;

            Eigen::Vector4f centroid;  //质心
            pcl::compute3DCentroid(*ext_cloud, centroid); // 计算质心
            double plane_distance = centroid.topRows(3).norm();
            if ((closest_plane_dist - plane_distance) > 0.3
            )
            {
                closest_plane_z = centroid.z();
                closest_plane_dist = plane_distance;
                *target_plane = *ext_cloud;
                target_plane_param = plane_param;
            }
        }
        cout << " RANSAC plane detection cost:" << timer.toc_tic() << endl;


        if (CfgParam.DEBUG_SHOW)
        {
            add_pointClouds_show(viewer, target_plane, true, "step1: pre detected target plane");
        }

//       平面聚类提取到了标定板之后，可能还含有部分支架的点云，边缘出血点的噪声点云做干扰，现在使用欧式聚类提取最大
//        去除边界点
        Eigen::VectorXd board_intensity;
        board_intensity.resize(target_plane->size());

#pragma omp parallel
#pragma omp for
        for (int i = 0; i < target_plane->size(); ++i)
            board_intensity[i] = target_plane->points[i].intensity;

        vector<double> mu{15, 100};
        vector<double> sigma{15, 15};
        GMMFit::fit_1d(board_intensity, 2, mu, sigma, false);
        double reflactive_thd = mu[0] + 2.50 * sigma[0];

        pcl::PassThrough<PoinT> pass_intensity;
        pass_intensity.setInputCloud(target_plane);//这个参数得是指针，类对象不行
        pass_intensity.setFilterFieldName("intensity");//设置想在哪个坐标轴上操作
        pass_intensity.setFilterLimits(CfgParam.noise_reflactivity_thd,reflactive_thd);
        pass_intensity.setFilterLimitsNegative(false);//保留（true就是删除，false就是保留而删除此区间外的）
        pass_intensity.filter(*target_plane);//输出到结果指针

//        再一次聚类将最多的点聚出来,边缘点剔除掉
        PointCloud<PoinT>::Ptr final_board_black_part(new PointCloud<PoinT>);
        PointCloud<PoinT>::Ptr raw_board(new PointCloud<PoinT>);
        PointCloud<PoinT>::Ptr final_board(new PointCloud<PoinT>);

        vector<PointIndices> ece_inlier;
        search::KdTree<PoinT>::Ptr tree(new search::KdTree<PoinT>);
        EuclideanClusterExtraction<PoinT> ece;
        tree.reset();
        ece_inlier.clear();
        ece.setInputCloud(target_plane);
        ece.setClusterTolerance(0.025);  //这里将标定板上所有的黑色块都要能聚成一类,边缘部分的反射率低的要剔除,所以这里设置成0.04,只要能和边缘部分区分就行
        ece.setMinClusterSize(1000);
        ece.setMaxClusterSize(2500000);
        ece.setSearchMethod(tree);
        ece.extract(ece_inlier);
        boost::shared_ptr<std::vector<int>>
            index_ptr = boost::make_shared<std::vector<int>>(ece_inlier[0].indices);  //把最多的挑出来就行了
        pcl::ExtractIndices<PoinT> eifilter(true);
        eifilter.setInputCloud(target_plane);
        eifilter.setIndices(index_ptr);
        eifilter.filter(*final_board_black_part);

        if (CfgParam.DEBUG_SHOW)
        {
            add_pointClouds_show(viewer, final_board_black_part, true, " step2: cut block part ");
        }

//        截断
        PoinT min_p, max_p;
//        final_board_black_part->is_dense = false;
//        std::vector<int> mapping;
//        pcl::removeNaNFromPointCloud(*final_board_black_part, *final_board_black_part, mapping);
        pcl::getMinMax3D(*final_board_black_part, min_p, max_p);
        cout << min_p << " " << max_p << endl;

        double exceed_thd = CfgParam.board_edge_thd;
        pass.setInputCloud(raw_cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(min_p.x - exceed_thd, max_p.x + exceed_thd);
        pass.setFilterLimitsNegative(false);//保留（true就是删除，false就是保留而删除此区间外的）
        pass.filter(*raw_board);

        pass.setInputCloud(raw_board);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(min_p.y - exceed_thd, max_p.y + exceed_thd);
        pass.setFilterLimitsNegative(false);
        pass.filter(*raw_board);

        pass.setInputCloud(raw_board);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(min_p.z - exceed_thd, max_p.z + exceed_thd);
        pass.setFilterLimitsNegative(false);//保留（true就是删除，false就是保留而删除此区间外的）
        pass.filter(*raw_board);//输出到结果指针
        if (CfgParam.DEBUG_SHOW)
        {
            add_pointClouds_show(viewer, raw_board, true, " step3: cut edges!");
        }

        cut_edges(final_board_black_part,
                  target_plane_param,
                  raw_board,
                  final_board
        );

        if (CfgParam.DEBUG_SHOW)
        {
            add_pointClouds_show(viewer, final_board, true, " step4: final!");
        }

//        pcl::io::savePCDFileBinary(CfgParam.cheesboard_path + "/" + get_name(pc_path[pc_idx]) + ".pcd",
//                                   *final_board_black_part);
        pcl::io::savePCDFileBinary(CfgParam.cheesboard_path + "/" + get_name(pc_path[pc_idx]) + ".pcd", *final_board);

    }

    if (CfgParam.DEBUG_SHOW)
    {
        viewer->spin();
    }
//    viewer->removeAllPointClouds();  // 移除当前所有点云

    return 0;
}

