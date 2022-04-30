#include <iostream>
#include "global.h"
#include "utils.h"
#include "GMMFit.h"
#include "visualize.h"
#include "pc_process.h"
#include "opt_plane_param.hpp"
#include "opt_grid_fitting.h"
#include "opt_reprj_calib.h"

using namespace cv;
void
calib_opt(
              vector<vector<Eigen::Vector2d>> &v_est_img_corner_pt,
              vector<vector<Eigen::Vector3d>> &v_est_pc_corner_pt,
              Eigen::Matrix3d &e_R_cl,
              Eigen::Vector3d &e_t_cl
)
{
    int tol_frame_num = v_est_pc_corner_pt.size();
    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point3f> objectPoints;

    std::vector<std::vector<Eigen::Vector2d>> e_imagePoints(tol_frame_num);
    std::vector<std::vector<Eigen::Vector3d>> e_objectPoints(tol_frame_num);

    int corner_num = (CfgParam.board_size.x() - 1) * (CfgParam.board_size.y() - 1);
    for (int frame_idx = 0; frame_idx < tol_frame_num; ++frame_idx)
    {
        for (int pt_idx = 0; pt_idx < corner_num; ++pt_idx)
        {
            Eigen::Vector2d pt_2d = v_est_img_corner_pt[frame_idx][pt_idx];
            Eigen::Vector3d pt_3d = v_est_pc_corner_pt[frame_idx][pt_idx];
            imagePoints.emplace_back(pt_2d.x(), pt_2d.y());
            objectPoints.emplace_back(pt_3d.x(), pt_3d.y(), pt_3d.z());
            e_imagePoints[frame_idx].push_back(Cam.imgPt2norm(pt_2d));
            e_objectPoints[frame_idx].push_back(pt_3d);
        }
    }

    cv::Mat cv_r, R_cl, cv_t;
    Eigen::Matrix4d Twc_curr;
    cv::solvePnPRansac(objectPoints,
                       imagePoints,
                       Cam.cameraMatrix,
                       Cam.distCoeffs,
                       cv_r,
                       cv_t,
                    false,
                    cv::SOLVEPNP_UPNP
    );

//  cv_r is the form of rotation vector, which is converted into matrix by Rodrigues formula
    cv::Rodrigues(cv_r, R_cl);
    cv::cv2eigen(R_cl, e_R_cl);
    cv::cv2eigen(cv_t, e_t_cl);

//  Start the pose optimization of all features
    double Tcl[7] = {0, 0, 0, 1, 0, 0, 0};
    Eigen::Map<Eigen::Quaterniond> q(Tcl);
    Eigen::Map<Eigen::Vector3d> trans(Tcl + 4);
    q = Eigen::Quaterniond(e_R_cl);
    trans = e_t_cl;
    pose_reprj_opt(e_imagePoints, e_objectPoints, Tcl);
    e_R_cl = Eigen::Quaterniond(Tcl[3], Tcl[0], Tcl[1], Tcl[2]).matrix();
    e_t_cl = Eigen::Vector3d(Tcl[4], Tcl[5], Tcl[6]);
}

int
main(int argc, char *argv[])
{
//    read parameters
    readParameters("../cfg/config_real.yaml");
    vector<string> v_pc_path;
    vector<string> v_img_path;
    load_file_path(CfgParam.cheesboard_path, v_pc_path);
    load_file_path(CfgParam.img_path, v_img_path);
    assert(v_pc_path.size() == v_img_path.size());

//   Generates a list of currently randomly selected frames
    int tol_frame_num = v_img_path.size();
    int n_frame2opt = min(5, int(v_img_path.size()));
    int *rand_ids = new int[tol_frame_num];
    get_random_lists(tol_frame_num, rand_ids);
    cout << "Randomly select " << n_frame2opt << " frames to optimize" << endl;
    vector<vector<Eigen::Vector3d>> v_est_pc_corner(n_frame2opt);
    vector<vector<Eigen::Vector2d>> v_est_img_corner(n_frame2opt);

//    estimate all frame corners in parallel
#pragma omp parallel
#pragma omp for
    for (int frame_idx = 0; frame_idx < n_frame2opt; ++frame_idx)
    {
//      load data
        int selected_id = rand_ids[frame_idx];
        cout << "frame: " << v_pc_path[selected_id] << " detecting corners" << endl;
        pcl::PointCloud<PoinT>::Ptr chess_board(new pcl::PointCloud<PoinT>);
        pcl::io::loadPCDFile<PoinT>(v_pc_path[selected_id], *chess_board);
        pcl::transformPointCloud(*chess_board,*chess_board,T_base);
        cv::Mat img = cv::imread(v_img_path[selected_id]);
        est_img_corner(img, v_est_img_corner[frame_idx]);
        est_board_corner(chess_board, v_est_pc_corner[frame_idx]);
    }

    Eigen::Matrix3d e_R_cl;
    Eigen::Vector3d e_t_cl;
    calib_opt(v_est_img_corner,v_est_pc_corner, e_R_cl, e_t_cl);
    cout << e_R_cl << endl;
    cout << e_t_cl << endl;
    Eigen::Matrix4f T_tmp;
    Eigen::Matrix4f T_cl;
    T_tmp.setIdentity();
    T_tmp.block(0, 0, 3, 3) = e_R_cl.cast<float>();
    T_tmp.block(0, 3, 3, 1) = e_t_cl.cast<float>();
    T_cl = T_tmp * T_base;

    cout << "Final Extrinsic Parameters:\n" << T_cl << endl;


//    todo: colorize the point cloud for quality check
    visualization::PCLVisualizer::Ptr viewer;
    viewer.reset(new visualization::PCLVisualizer("3d view"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(0.3);

    Mat img_show;
    pcl::PointCloud<PoinT>::Ptr pc_raw(new pcl::PointCloud<PoinT>);
    pcl::PointCloud<DisplayType>::Ptr pc_show(new pcl::PointCloud<DisplayType>);
    img_show = imread(v_img_path[0]);
    Cam.undistort(img_show,img_show);
    vector<string> v_raw_pc_path;
    load_file_path(CfgParam.raw_pc_path, v_raw_pc_path);
    pcl::io::loadPCDFile<PoinT>(v_raw_pc_path[0], *pc_raw);
    transformPointCloud(*pc_raw,*pc_raw,T_cl);
    pcl::copyPointCloud(*pc_raw,*pc_show);

#pragma omp parallel
#pragma omp for
    for (int pt_id = 0; pt_id < pc_show->size(); ++pt_id)
    {
        DisplayType &pt = pc_show->points[pt_id];
        Eigen::Vector2d pix = Cam.Pt2img(Eigen::Vector3d(pt.x,pt.y,pt.z));
        if(!Cam.is_in_FoV(pix))
            continue;
        pt.r = img_show.at<Vec3b>(pix.y(),pix.x())[2];
        pt.g = img_show.at<Vec3b>(pix.y(),pix.x())[1];
        pt.b = img_show.at<Vec3b>(pix.y(),pix.x())[0];
    }
    add_pointClouds_show(viewer, pc_show, true);

    return 0;
}
