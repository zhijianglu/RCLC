//
// Created by lab on 2021/8/25.
//

#include "global.h"

param CfgParam;
PinholeCam Cam;
Eigen::Matrix4f T_base;
void
readParameters(std::string config_file)
{
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    cout << "start loading parameters from:"<<config_file << endl;

//    load cam info:
    int cam_width;
    int cam_height;
    double fx;
    double fy;
    double cx;
    double cy;
    double k1;
    double k2;
    double k3;
    double r1;
    double r2;
    cam_width = fsSettings["cam_width"];
    cam_height = fsSettings["cam_height"];
    fx = fsSettings["fx"];
    fy = fsSettings["fy"];
    cx = fsSettings["cx"];
    cy = fsSettings["cy"];
    k1 = fsSettings["k1"];
    k2 = fsSettings["k2"];
    k3 = fsSettings["k3"];
    r1 = fsSettings["r1"];
    r2 = fsSettings["r2"];
    Cam = PinholeCam(cam_width, cam_height, fx, fy, cx, cy, k1, k2, k3, r1, r2);
    CfgParam.need_undistort = fsSettings["need_undistort"];

//    load board info:
    CfgParam.board_size.x() = fsSettings["board_width"];
    CfgParam.board_size.y() = fsSettings["board_height"];
    CfgParam.gride_side_length = fsSettings["gride_side_length"];
    CfgParam.board_order = fsSettings["board_order"];
    CfgParam.gride_diagonal_length = CfgParam.gride_side_length * sqrt(2.0f);

//    load  segmentation params info:
    CfgParam.noise_reflactivity_thd = fsSettings["noise_reflactivity_thd"];
    CfgParam.angle_thd = fsSettings["angle_thd"];
    CfgParam.board_edge_thd = fsSettings["board_edge_thd"];

//    load data path info
    fsSettings["data_root_path"] >> CfgParam.data_root_path;
    fsSettings["cheesboard_path"] >> CfgParam.cheesboard_path;
    CfgParam.cheesboard_path = CfgParam.data_root_path + CfgParam.cheesboard_path;

    fsSettings["raw_pc_path"] >> CfgParam.raw_pc_path;
    CfgParam.raw_pc_path = CfgParam.data_root_path + CfgParam.raw_pc_path;

    fsSettings["img_path"] >> CfgParam.img_path;
    CfgParam.img_path=CfgParam.data_root_path+CfgParam.img_path;

    fsSettings["framesets_id_path"] >> CfgParam.framesets_id_path;
    CfgParam.framesets_id_path=CfgParam.data_root_path+CfgParam.framesets_id_path;


    fsSettings["error_path_pnp"] >> CfgParam.error_path_pnp;
    CfgParam.error_path_pnp = CfgParam.data_root_path + CfgParam.error_path_pnp;

    fsSettings["error_path_opt"] >> CfgParam.error_path_opt;
    CfgParam.error_path_opt = CfgParam.data_root_path + CfgParam.error_path_opt;

    fsSettings["sim_img_corner_path"] >> CfgParam.sim_img_corner_path;
    fsSettings["sim_pc_corner_path"] >> CfgParam.sim_pc_corner_path;
    CfgParam.sim_img_corner_path = CfgParam.data_root_path + CfgParam.sim_img_corner_path;
    CfgParam.sim_pc_corner_path = CfgParam.data_root_path + CfgParam.sim_pc_corner_path;

//    load sim data info
    CfgParam.sim_gt_cl_t.x() = fsSettings["sim_gt_cl_tx"];
    CfgParam.sim_gt_cl_t.y() = fsSettings["sim_gt_cl_ty"];
    CfgParam.sim_gt_cl_t.z() = fsSettings["sim_gt_cl_tz"];
    CfgParam.sim_gt_cl_rpy.x() = fsSettings["sim_gt_cl_r"];
    CfgParam.sim_gt_cl_rpy.y() = fsSettings["sim_gt_cl_p"];
    CfgParam.sim_gt_cl_rpy.z() = fsSettings["sim_gt_cl_y"];
    Eigen::AngleAxisd  rollAngle(CfgParam.sim_gt_cl_rpy.z(), Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd   yawAngle(CfgParam.sim_gt_cl_rpy.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(CfgParam.sim_gt_cl_rpy.x(), Eigen::Vector3d::UnitX());
    CfgParam.sim_gt_R_cl = rollAngle * yawAngle * pitchAngle;

//    关于标定板点聚类提取预处理
    CfgParam.reflactivity_dec_step = fsSettings["reflactivity_dec_step"];
    CfgParam.block_us_radius = fsSettings["block_us_radius"];
    CfgParam.black_pt_cluster_dst_thd = fsSettings["black_pt_cluster_dst_thd"];

//    grid fitting optimization parameters
    CfgParam.uniform_sampling_radius = fsSettings["uniform_sampling_radius"];
    CfgParam.neighbour_radius_rate = fsSettings["neighbour_radius_rate"];
    CfgParam.cost_radius_rate = fsSettings["cost_radius_rate"];
    CfgParam.max_iter_time = fsSettings["max_iter_time"];

//    for final calibration parameters
    CfgParam.use_pnp_init= fsSettings["use_pnp_init"];

//    load other params
    CfgParam.apply_noise_test = fsSettings["apply_noise_test"];
    CfgParam.DEBUG_SHOW = fsSettings["DEBUG_SHOW"];
    CfgParam.display_remove_oclusion = fsSettings["display_remove_oclusion"];
    CfgParam.debug_param = fsSettings["debug_param"];
    CfgParam.noise_x = fsSettings["noise_x"];
    CfgParam.noise_y = fsSettings["noise_y"];
    CfgParam.noise_yaml = fsSettings["noise_yaml"];
    fsSettings["corner_detect_method"] >> CfgParam.corner_detect_method;

//  T_base for convert the lidar coordinate axis to the coordinate axis sequence of the camera (Z forward, y downward)
//  May not necessary but for better understanding of the 3D points.
    T_base.setIdentity();
    T_base.block(0,0,3,3)<<
                         0, -1, 0,
                         0, 0, -1,
                         1, 0,  0;

    cout << "parameters loaded!" << endl;

}

