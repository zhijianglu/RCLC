#include <iostream>
#include "global.h"
#include "utils.h"
#include "GMMFit.h"
#include "visualize.h"
#include "pc_process.h"
#include "opt_plane_param.hpp"
#include "opt_grid_fitting.h"

int
main()
{
    readParameters("../cfg/config_real.yaml");
    vector<string> v_curr_pc_path;
    load_file_path(CfgParam.cheesboard_path, v_curr_pc_path);

    visualization::PCLVisualizer::Ptr viewer;
    if (CfgParam.DEBUG_SHOW)
    {
        viewer.reset(new visualization::PCLVisualizer("3d view"));
        viewer->setBackgroundColor(0, 0, 0);
        viewer->addCoordinateSystem(0.3);
    }

    cout<<"You can add some pose disturbances to the parameter file to view the optimization effect"<<endl;

    for (int frame_idx = 0; frame_idx < v_curr_pc_path.size(); ++frame_idx)
    {
        pcl::PointCloud<PoinT>::Ptr chess_board_processe(new pcl::PointCloud<PoinT>);
        pcl::PointCloud<PoinT>::Ptr chess_board_raw(new pcl::PointCloud<PoinT>);
        pcl::io::loadPCDFile<PoinT>(v_curr_pc_path[frame_idx], *chess_board_raw);
        pcl::transformPointCloud(*chess_board_raw,*chess_board_raw,T_base);

        pcl::copyPointCloud(*chess_board_raw, *chess_board_processe);
//        load ground truth data
        vector<Eigen::Vector3d>v_est_pc_corner;
        cout << "board: " << frame_idx << ":" << v_curr_pc_path[frame_idx]<<endl;
//    Calculate the white block point cloud fitting plane
        Eigen::Vector4d board_plane_param;
        pc_plane_fitting(chess_board_processe, board_plane_param);
        pc_plane_prj(chess_board_processe, board_plane_param);
        if (CfgParam.DEBUG_SHOW) add_pointClouds_show(viewer,chess_board_processe);

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
        eular_cluster_iter(chess_board_processe, reflactive_thd, block_centroid, viewer);
        Eigen::Matrix4d T_bl;

        calc_laser_coor_v1(block_centroid, board_plane_param,T_bl);
        pcl::transformPointCloud(*chess_board_processe, *chess_board_processe, T_bl.cast<float>());
        if(viewer!= nullptr) display_standard_gride_lines(viewer);

        opt_grid_fitting(chess_board_processe, T_bl, v_est_pc_corner, viewer);
        if(viewer!= nullptr)
        {
            viewer->removeAllPointClouds();
            viewer->removeAllShapes();
        }
        cout<<"finished "<<v_curr_pc_path[frame_idx]<<endl;
    }
    return 0;
}
