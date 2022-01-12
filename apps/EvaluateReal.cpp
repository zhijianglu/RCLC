#include <iostream>
#include "global.h"
#include "utils.h"
#include "GMMFit.h"
#include "visualize.h"
#include "pc_process.h"
#include "opt_plane_param.hpp"
#include "opt_grid_fitting.h"
#include "opt_reprj_calib.h"

vector<Eigen::Vector3d> all_pc_corner;

vector<Eigen::Vector2d> all_img_corner;

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

    Eigen::Quaterniond Q_cl(e_R_cl);
    error_log_pnp << calc_norm_reprj_error(e_R_cl, e_t_cl, all_img_corner, all_pc_corner) << " "
                  << Q_cl.w() << " " << Q_cl.x() << " " << Q_cl.y() << " " << Q_cl.z() << " " << e_t_cl.x() << " "
                  << e_t_cl.y() << " " << e_t_cl.z() << endl;

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

    Q_cl = Eigen::Quaterniond(e_R_cl);
    error_log_opt << calc_norm_reprj_error(e_R_cl, e_t_cl, all_img_corner, all_pc_corner) << " "
                  << Q_cl.w() << " " << Q_cl.x() << " " << Q_cl.y() << " " << Q_cl.z() << " " <<
                  e_t_cl.x() << " " << e_t_cl.y() << " " << e_t_cl.z() << endl;


}

int
main(int argc, char *argv[])
{
    int start_num = atoi(argv[1]); //包括在内
    int end_num = atoi(argv[2]);   //包括在内
    int num2iterate = 30;

    readParameters("../cfg/config_real.yaml");
    string cam_name = CfgParam.error_path_pnp.substr(CfgParam.error_path_pnp.find_last_of("/") - 1, 1);

    vector<string> v_img_path;
    vector<string> v_img_corner_path;
    load_file_path(CfgParam.img_path, v_img_path);
    load_file_path(CfgParam.img_path + "/../matlab_corner_positions_est/"+cam_name, v_img_corner_path);

    vector<vector<Eigen::Vector3d>> v_est_pc_corner(v_img_corner_path.size());
    vector<vector<Eigen::Vector2d>> v_est_img_corner(v_img_corner_path.size());

//    estimate all frame corners in parallel
//#pragma omp parallel
//#pragma omp for
    for (int frame_idx = 0; frame_idx < v_img_corner_path.size(); ++frame_idx)
    {
//      load data
        string cheesboard_pc_path = CfgParam.cheesboard_path+"/"+ get_name(v_img_corner_path[frame_idx])+".pcd";
        cout << "frame: " << cheesboard_pc_path << " detecting" << endl;

        pcl::PointCloud<PoinT>::Ptr chess_board(new pcl::PointCloud<PoinT>);
        pcl::io::loadPCDFile<PoinT>(cheesboard_pc_path, *chess_board);
        cv::Mat img = cv::imread(v_img_path[frame_idx]);

        if (CfgParam.corner_detect_method == "matlab")
            read_img_corner_matlab(v_img_corner_path[frame_idx], v_est_img_corner[frame_idx]);
        else
            est_img_corner(img, v_est_img_corner[frame_idx]);
        est_board_corner(chess_board, v_est_pc_corner[frame_idx]);
    }


    for (int frame_idx = 0; frame_idx < v_img_corner_path.size(); ++frame_idx)
    {
        for (int pt_id = 0; pt_id < v_est_pc_corner[frame_idx].size(); ++pt_id)
        {
            all_pc_corner.push_back(v_est_pc_corner[frame_idx][pt_id]);
            all_img_corner.push_back(v_est_img_corner[frame_idx][pt_id]);
            cout << v_est_pc_corner[frame_idx][pt_id] << endl;
        }
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
    int nP_per_row = CfgParam.board_size.x() - 1;
    bool record_img_corner = true;
    if (CfgParam.corner_detect_method == "matlab")
        record_img_corner = false;

    for (int frame_idx = 0; frame_idx < v_img_corner_path.size(); ++frame_idx)
    {

        ofstream est_pc_corner_file
            (CfgParam.error_path_pnp + "/../pc_corner_est/" + get_name(v_img_corner_path[frame_idx]) + ".txt");
        ofstream est_img_corner_file;

        est_pc_corner_file
            << "# This file is the estimated pc corner. The order of the corner is from right down to left up. \n"
            << "# Coresponding pc path:\n"
            << endl;

        if (record_img_corner)
            est_img_corner_file
                << "# This file is the estimated pc corner. The order of the corner is from right down to left up. \n"
                << "# Coresponding img path:\n"
                << "* " << v_img_corner_path[frame_idx]
                << endl;

        for (int pt_idx = 0; pt_idx < v_est_pc_corner[frame_idx].size(); ++pt_idx)
        {
            est_pc_corner_file.precision(18);
            est_pc_corner_file << v_est_pc_corner[frame_idx][pt_idx].x() << " "
                               << v_est_pc_corner[frame_idx][pt_idx].y() << " "
                               << v_est_pc_corner[frame_idx][pt_idx].z() << " ";

            if (record_img_corner)
            {
                est_img_corner_file.precision(18);
                est_img_corner_file << v_est_img_corner[frame_idx][pt_idx].x() << " "
                                    << v_est_img_corner[frame_idx][pt_idx].y() << " ";
            }


            if ((pt_idx + 1) % (int) nP_per_row == 0)
            {
                est_pc_corner_file << endl;
                if (record_img_corner)
                    est_img_corner_file << endl;
            }
        }
        if (record_img_corner)
            est_img_corner_file.close();
        est_pc_corner_file.close();
    }

    return 0;
}
