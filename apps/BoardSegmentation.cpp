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


int
main(int argc, char *argv[])
{
//    read parameter
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

// if use multi thread
#pragma omp parallel
#pragma omp for
    for (int pc_idx = 0; pc_idx < pc_path.size(); ++pc_idx)
    {
        std::string fileData = pc_path[pc_idx];

        PointCloud<PoinT>::Ptr raw_cloud_read(new PointCloud<PoinT>);
        pcl::io::loadPCDFile<PoinT>(fileData, *raw_cloud_read);

        PointCloud<PoinT>::Ptr raw_cloud(new PointCloud<PoinT>);

        cout << " ********************************* " << endl;
        cout << pc_idx << " current processing pointcloud: " << pc_path[pc_idx] << endl;

        pcl::PassThrough<PoinT> pass;
        pass.setInputCloud(raw_cloud_read);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(3, 6);  //choose property range according to your data
        pass.setFilterLimitsNegative(false);
        pass.filter(*raw_cloud);

        if (CfgParam.DEBUG_SHOW)
        {
            add_pointClouds_show(viewer, raw_cloud, true, "raw");
        }
        VoxelGrid<PoinT> vox;
        PointCloud<PoinT>::Ptr vox_cloud(new PointCloud<PoinT>);
        pcl::UniformSampling<pcl::PointXYZI> US0;
        US0.setInputCloud(raw_cloud);
        US0.setRadiusSearch(0.008f);
        US0.filter(*vox_cloud);

        // RANSAC clustering
        SACSegmentation<PoinT> sac;
        PointIndices::Ptr inliner(new PointIndices);
        ModelCoefficients::Ptr coefficients(new ModelCoefficients);
        PointCloud<PoinT>::Ptr sac_cloud(new PointCloud<PoinT>);
        sac.setInputCloud(vox_cloud);
        sac.setMethodType(SAC_RANSAC);
        sac.setModelType(SACMODEL_PLANE);
        sac.setMaxIterations(800);
        sac.setDistanceThreshold(0.08);

        //extract candidate plane and apply condition to select chessboard point cloud
        PointCloud<PoinT>::Ptr ext_cloud(new PointCloud<PoinT>);
        PointCloud<PoinT>::Ptr ext_cloud_rest(new PointCloud<PoinT>);
        PointCloud<PoinT>::Ptr target_plane(new PointCloud<PoinT>);
        PointCloud<PoinT>::Ptr target_plane_black(new PointCloud<PoinT>);
        PointCloud<PoinT>::Ptr target_plane_white(new PointCloud<PoinT>);
        PointCloud<PoinT>::Ptr final_plane(new PointCloud<PoinT>);

        int pt_size = vox_cloud->size();
        double closest_plane_dist = 1000;
        ExtractIndices<PoinT> ext;
        srand((unsigned) time(NULL));
        Eigen::Vector4d target_plane_param;
        while (vox_cloud->size() > pt_size * 0.05)
        {
            ext.setInputCloud(vox_cloud);
            sac.segment(*inliner, *coefficients);

            Eigen::Vector4d plane_param
                (coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);

            ext.setIndices(inliner);
            ext.setNegative(false);
            ext.filter(*ext_cloud);
            ext.setNegative(true);
            ext.filter(*ext_cloud_rest);
            *vox_cloud = *ext_cloud_rest;

            if (ext_cloud->size() < pt_size * 0.02 || abs(plane_param.topRows(3).normalized().x()) <= CfgParam.angle_thd)
                continue;

            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*ext_cloud, centroid);
            double plane_distance = centroid.topRows(3).norm();
            if ((closest_plane_dist - plane_distance) > 0.3)
            {
                closest_plane_dist = plane_distance;
                *target_plane = *ext_cloud;
                target_plane_param = plane_param;
            }
        }

        EulerCluster(target_plane, target_plane);
        PoinT min_p, max_p;
        pcl::getMinMax3D(*target_plane, min_p, max_p);
        cut_roi(raw_cloud, min_p, max_p);

        if (CfgParam.DEBUG_SHOW)
        {
            add_pointClouds_show(viewer, target_plane, true, "step1: pre detected target plane");
        }

        Eigen::VectorXd board_intensity;
        board_intensity.resize(target_plane->size());

//        GMM based intensity clustering
#pragma omp parallel
#pragma omp for
        for (int i = 0; i < target_plane->size(); ++i)
            board_intensity[i] = target_plane->points[i].intensity;
        vector<double> mu{15, 100};
        vector<double> sigma{15, 15};
        GMMFit::fit_1d(board_intensity, 2, mu, sigma, false);
        double reflactive_thd = mu[0] + 1.7 * sigma[0];

//        extract black block points
        pcl::PassThrough<PoinT> pass_intensity;
        pass_intensity.setInputCloud(raw_cloud);
        pass_intensity.setFilterFieldName("intensity");
        pass_intensity.setFilterLimits(CfgParam.noise_reflactivity_thd, reflactive_thd);
        pass_intensity.setFilterLimitsNegative(false);
        pass_intensity.filter(*target_plane_black);
        EulerCluster(target_plane_black, target_plane_black);
        if (CfgParam.DEBUG_SHOW)
        {
            add_pointClouds_show(viewer, target_plane_black, true, "target black points");
        }

//        extract white block points
        pass_intensity.setInputCloud(raw_cloud);
        pass_intensity.setFilterFieldName("intensity");
        pass_intensity.setFilterLimits(reflactive_thd, 150);
        pass_intensity.setFilterLimitsNegative(false);
        pass_intensity.filter(*target_plane_white);
        EulerCluster(target_plane_white, target_plane_white);
        if (CfgParam.DEBUG_SHOW)
        {
            add_pointClouds_show(viewer, target_plane_white, true, "target black points");
        }

        *final_plane = *target_plane_white + *target_plane_black;
        if (CfgParam.DEBUG_SHOW)
        {
            add_pointClouds_show(viewer, final_plane, true, "step1: pre detected target plane");
        }

//        save chessboard point clouds
        pcl::io::savePCDFileBinary(CfgParam.cheesboard_path + "/" + get_name(pc_path[pc_idx]) + ".pcd",
                                   *final_plane);
    }

    if (CfgParam.DEBUG_SHOW)
    {
        viewer->spin();
    }

    return 0;
}

