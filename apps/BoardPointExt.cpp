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
    readParameters("../cfg/config_sim.yaml");
    vector<string> v_curr_pc_path;
    load_file_path(CfgParam.cheesboard_path, v_curr_pc_path);

    vector<string> v_pc_corner_path;
    load_file_path(CfgParam.sim_pc_corner_path, v_pc_corner_path);


    visualization::PCLVisualizer::Ptr viewer;
    if (CfgParam.DEBUG_SHOW)
    {
        viewer.reset(new visualization::PCLVisualizer("3d view"));
        viewer->setBackgroundColor(0, 0, 0);
        viewer->addCoordinateSystem(0.3);
    }
    vector<double> v_error(v_curr_pc_path.size(),0);
//#pragma omp parallel
//#pragma omp for
//    v_curr_pc_path = {"/media/lab/915c94cc-246e-42f5-856c-faf39f774d2c/CalibData/real_world_data/board/57.pcd","/media/lab/915c94cc-246e-42f5-856c-faf39f774d2c/CalibData/real_world_data/board_/0.pcd"};
    for (int frame_idx = 0; frame_idx < v_curr_pc_path.size(); ++frame_idx)
    {
//    load data
        pcl::PointCloud<PoinT>::Ptr chess_board_processe(new pcl::PointCloud<PoinT>);
        pcl::PointCloud<PoinT>::Ptr chess_board_raw(new pcl::PointCloud<PoinT>);
        pcl::io::loadPCDFile<PoinT>(v_curr_pc_path[frame_idx], *chess_board_raw);
        pcl::copyPointCloud(*chess_board_raw, *chess_board_processe);
//        load ground truth data
        vector<Eigen::Vector3d>v_gt_pc_corner;
        vector<Eigen::Vector3d>v_est_pc_corner;
        read_sim_pc_corner(v_pc_corner_path[frame_idx], v_gt_pc_corner);

//        if (CfgParam.DEBUG_SHOW) add_pointClouds_show(viewer,chess_board_processe);
//    计算白色区块点云拟合平面
        Eigen::Vector4d board_plane_param;
        pc_plane_fitting(chess_board_processe, board_plane_param);
        pc_plane_prj(chess_board_processe, board_plane_param);
//        if (CfgParam.DEBUG_SHOW) add_pointClouds_show(viewer,chess_board_processe);

        Eigen::VectorXd board_intensity;
        board_intensity.resize(chess_board_processe->size());

#pragma omp parallel
#pragma omp for
        for (int i = 0; i < chess_board_processe->size(); ++i)
            board_intensity[i] = chess_board_processe->points[i].intensity;

        vector<double> mu{15, 100};
        vector<double> sigma{15, 15};
        GMMFit::fit_1d(board_intensity, 2, mu, sigma, false);
        cout << "board: " << frame_idx << ":" << v_curr_pc_path[frame_idx]
             << "  mu:" << mu[0] << " " << mu[1]
             << "  sigma:" << sigma[0] << " " << sigma[1] << endl;

        double reflactive_thd = mu[0] + 2.0 * sigma[0];
        vector<Eigen::Vector3d> block_centroid;
        eular_cluster_iter(chess_board_processe, reflactive_thd, block_centroid);
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

        for (int i = 0; i < v_gt_pc_corner.size(); ++i)
        {
            v_error[frame_idx] += (v_gt_pc_corner[i] - v_est_pc_corner[i]).norm();
        }

        if(viewer!= nullptr)
        {
            display_points(viewer, v_est_pc_corner, {0, 255, 255});
            display_points(viewer, v_gt_pc_corner, {0, 0, 255});
            add_pointClouds_show(viewer, chess_board_raw, false);
        }
    }
    int nP_per_frame = (CfgParam.board_size.x()-1)*(CfgParam.board_size.y()-1);
    double error_sum = 0;
    cout<<"frame corner error:";
    for (int i = 0; i < v_error.size(); ++i)
    {
        error_sum += v_error[i];
        cout<<"["<<i<<": "<<v_error[i]/double(nP_per_frame) <<"]" ;
    }
    cout << "\nMean error of estimated 3D corner:" << error_sum / double(v_error.size() * nP_per_frame) << endl;

    return 0;
}
