#include <iostream>
#include "global.h"
#include "utils.h"
#include "GMMFit.h"
#include "visualize.h"
#include "pc_process.h"
#include "opt_plane_param.hpp"
#include "opt_grid_fitting.h"
#include "opt_reprj_calib.h"


void
calib_opt(int n_frame2opt,
          int it_time,
          vector<vector<Eigen::Vector2d>> &v_est_img_corner_pt,
          vector<vector<Eigen::Vector3d>> &v_est_pc_corner_pt,
          ofstream &error_log_pnp,
          ofstream &error_log_opt,
          ofstream &framesets_id_log
)
{
    int tol_frame_num = v_est_pc_corner_pt.size();
    //    生成当前随机选取帧的列表
    int *rand_ids = new int[tol_frame_num];
    get_random_lists(tol_frame_num, rand_ids, 12 * n_frame2opt + it_time);

    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point3f> objectPoints;

    std::vector<std::vector<Eigen::Vector2d>> e_imagePoints(n_frame2opt);
    std::vector<std::vector<Eigen::Vector3d>> e_objectPoints(n_frame2opt);

    int corner_num = (CfgParam.board_size.x() - 1) * (CfgParam.board_size.y() - 1);  //28
    for (int frame_idx = 0; frame_idx < n_frame2opt; ++frame_idx)
    {
        framesets_id_log << rand_ids[frame_idx] << " ";
        for (int pt_idx = 0; pt_idx < corner_num; ++pt_idx)
        {
            Eigen::Vector2d pt_2d = v_est_img_corner_pt[rand_ids[frame_idx]][pt_idx];
            Eigen::Vector3d pt_3d = v_est_pc_corner_pt[rand_ids[frame_idx]][pt_idx];
            imagePoints.emplace_back(pt_2d.x(), pt_2d.y());
            objectPoints.emplace_back(pt_3d.x(), pt_3d.y(), pt_3d.z());
            e_imagePoints[frame_idx].push_back(Cam.imgPt2norm(pt_2d));
            e_objectPoints[frame_idx].push_back(pt_3d);
        }
    }

    framesets_id_log << endl;
    cv::Mat cv_r, R_cl, cv_t; //r是旋转向量，根据openCV封装，输出是 R_cl 世界坐标系到相机坐标系，且t是基于相机坐标系下的
    Eigen::Matrix4d Twc_curr;
    cv::solvePnPRansac(objectPoints,
                       imagePoints,
                       Cam.cameraMatrix,
                       Cam.distCoeffs,
                       cv_r,
                       cv_t
//                       ,
//                    false,
//                    cv::SOLVEPNP_UPNP
    );

// r为旋转向量形式，用 Rodrigues 公式转换为矩阵
    cv::Rodrigues(cv_r, R_cl);
    //将变换矩阵转到eigen格式
    Eigen::Matrix3d e_R_cl;
    Eigen::Vector3d e_t_cl;
    cv::cv2eigen(R_cl, e_R_cl);
    cv::cv2eigen(cv_t, e_t_cl);
    evaluate(e_R_cl, e_t_cl, imagePoints, objectPoints, error_log_pnp);

//todo: 开始进行线特征的位姿优化!!!
    double Tcl[7] = {0, 0, 0, 1, 0, 0, 0};
    if (CfgParam.use_pnp_init)
    {
        Eigen::Map<Eigen::Quaterniond> q(Tcl);
        Eigen::Map<Eigen::Vector3d> trans(Tcl + 4);
        q = Eigen::Quaterniond(e_R_cl);
        trans = e_t_cl;
    }
    pose_reprj_opt(e_imagePoints, e_objectPoints, Tcl);
    e_R_cl = Eigen::Quaterniond(Tcl[3], Tcl[0], Tcl[1], Tcl[2]).matrix();
    e_t_cl = Eigen::Vector3d(Tcl[4], Tcl[5], Tcl[6]);
    evaluate(e_R_cl, e_t_cl, imagePoints, objectPoints, error_log_opt);
}

int
main(int argc, char *argv[])
{
    int start_num = atoi(argv[1]); //包括在内
    int end_num = atoi(argv[2]);   //包括在内
    int num2iterate = 20;

    readParameters("../cfg/config_sim.yaml");
    vector<string> v_pc_path;
    vector<string> v_img_path;
    vector<string> v_pc_corner_path;
    vector<string> v_img_corner_path;

    load_file_path(CfgParam.cheesboard_path, v_pc_path);
    load_file_path(CfgParam.img_path, v_img_path);
    load_file_path(CfgParam.sim_pc_corner_path, v_pc_corner_path);
    load_file_path(CfgParam.sim_img_corner_path, v_img_corner_path);
    assert(v_pc_path.size() == v_img_path.size());

    vector<vector<Eigen::Vector3d>> v_est_pc_corner(v_pc_path.size());
    vector<vector<Eigen::Vector2d>> v_est_img_corner(v_pc_path.size());
    vector<vector<Eigen::Vector3d>> v_gt_pc_corner(v_pc_path.size());
    vector<vector<Eigen::Vector2d>> v_gt_img_corner(v_pc_path.size());

    vector<double> v_error(v_pc_path.size(), 0);

//    estimate all frame corners in parallel
#pragma omp parallel
#pragma omp for
    for (int frame_idx = 0; frame_idx < v_pc_path.size(); ++frame_idx)
    {
//      load data
        pcl::PointCloud<PoinT>::Ptr chess_board(new pcl::PointCloud<PoinT>);
        pcl::io::loadPCDFile<PoinT>(v_pc_path[frame_idx], *chess_board);
        cv::Mat img = cv::imread(v_img_path[frame_idx]);
//      load ground truth data
        read_sim_pc_corner(v_pc_corner_path[frame_idx], v_gt_pc_corner[frame_idx]);
        read_sim_img_corner(v_img_corner_path[frame_idx], v_gt_img_corner[frame_idx]);

//      estimate corners
        est_img_corner(img, v_est_img_corner[frame_idx]);
        est_board_corner(chess_board, v_est_pc_corner[frame_idx]);
    }

    for (int num_2opt = start_num; num_2opt < end_num + 1; ++num_2opt)
    {
        string log_file_name;
        if (num_2opt < 10)
            log_file_name = "/0" + to_string(num_2opt) + ".txt";
        else
            log_file_name = "/" + to_string(num_2opt) + ".txt";

        ofstream error_log_pnp(CfgParam.error_path_pnp + log_file_name);
        ofstream error_log_opt(CfgParam.error_path_opt + log_file_name);
        ofstream framesets_id_log(CfgParam.framesets_id_path + log_file_name);

        error_log_pnp << "e_tx e_ty e_tz e_r e_p e_y norm_t norm_rpy e_prj" << endl;
        error_log_opt << "e_tx e_ty e_tz e_r e_p e_y norm_t norm_rpy e_prj" << endl;
        framesets_id_log << "frame set id" << endl;
        cout << "Number of set: " << num_2opt << endl;

        for (int it = 0; it < num2iterate; ++it)
        {
            cout << "    iter time:" << it << endl;
            calib_opt(num_2opt, it, v_est_img_corner,
                      v_est_pc_corner, error_log_pnp, error_log_opt, framesets_id_log);
        }
        error_log_pnp.close();
        error_log_opt.close();
        framesets_id_log.close();
    }

//    start evaluate the results
//    logs
    ofstream corner_error_norm_log_path (CfgParam.error_path_pnp+"/../corner3D_error_norm.txt");
    ofstream corner_error_x_log_path    (CfgParam.error_path_pnp+"/../corner3D_error_x.txt");
    ofstream corner_error_y_log_path    (CfgParam.error_path_pnp+"/../corner3D_error_y.txt");
    ofstream corner_error_z_log_path    (CfgParam.error_path_pnp+"/../corner3D_error_z.txt");
    int nP_per_row = CfgParam.board_size.x()-1;
    int nP_per_frame = (CfgParam.board_size.x()-1)*(CfgParam.board_size.y()-1);
    for (int frame_idx = 0; frame_idx < v_pc_path.size(); ++frame_idx)
    {
        ofstream est_pc_corner_file(CfgParam.error_path_pnp  + "/../pc_corner_est/" + to_string(frame_idx) + ".txt");
        ofstream est_img_corner_file(CfgParam.img_path  + "/../corner_positions_est/" + to_string(frame_idx) + ".txt");

        est_pc_corner_file
            << "# This file is the estimated pc corner. The order of the corner is from right down to left up. \n"
            << "# Coresponding pc path:\n"
            << "* "<<v_pc_corner_path[frame_idx]
            << endl;

        est_img_corner_file
            << "# This file is the estimated pc corner. The order of the corner is from right down to left up. \n"
            << "# Coresponding img path:\n"
            << "* "<<v_img_corner_path[frame_idx]
            << endl;

        for (int pt_idx = 0; pt_idx < v_gt_pc_corner[frame_idx].size(); ++pt_idx)
        {
            Vector3d error_vec = v_gt_pc_corner[frame_idx][pt_idx] - v_est_pc_corner[frame_idx][pt_idx];
            v_error[frame_idx] += (error_vec).norm();
            corner_error_norm_log_path << error_vec.norm() << " ";
            corner_error_x_log_path    << error_vec.x() << " ";
            corner_error_y_log_path    << error_vec.y() << " ";
            corner_error_z_log_path    << error_vec.z() << " ";
            est_pc_corner_file.precision(18);
            est_img_corner_file.precision(18);
            est_pc_corner_file << v_est_pc_corner[frame_idx][pt_idx].x() << " " << v_est_pc_corner[frame_idx][pt_idx].y() << " " << v_est_pc_corner[frame_idx][pt_idx].z() << " ";
            est_img_corner_file << v_est_img_corner[frame_idx][pt_idx].x() << " " << v_est_img_corner[frame_idx][pt_idx].y() << " " ;

            if ((pt_idx + 1) % (int) nP_per_row == 0)
            {
                est_pc_corner_file << endl;
                est_img_corner_file << endl;
            }
        }
        est_img_corner_file.close();
        est_pc_corner_file.close();
        corner_error_norm_log_path <<"\n";
        corner_error_x_log_path    <<"\n";
        corner_error_y_log_path    <<"\n";
        corner_error_z_log_path    <<"\n";
    }

    corner_error_norm_log_path.close();
    corner_error_x_log_path.close();
    corner_error_y_log_path.close();
    corner_error_z_log_path.close();

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
