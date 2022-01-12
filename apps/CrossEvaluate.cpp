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
read_poses(string file_path, vector<Matrix4d> &v_Tcl)
{

    std::string data_line_pose;
    std::string header;
    Eigen::Vector2d corner;
    ifstream corner_ifs(file_path);

    bool first_line = true;
    while (!corner_ifs.eof() && std::getline(corner_ifs, data_line_pose) )
    {
        std::istringstream poseData(data_line_pose);
        poseData >> header;
        if (header == "#" || header == "*"|| first_line)
        {
            first_line = false;
            continue;
        }
        Eigen::Quaterniond Q_cl;
        Eigen::Matrix3d e_R_cl;
        Eigen::Vector3d e_t_cl;
        double pixel_err = atof(header.c_str());
        poseData >> Q_cl.w() >> Q_cl.x() >> Q_cl.y() >> Q_cl.z() >> e_t_cl.x() >> e_t_cl.y() >> e_t_cl.z();
        Matrix4d Tcl;
        Tcl.setIdentity();
        Tcl.block(0, 0, 3, 3) = Q_cl.toRotationMatrix();
        Tcl.block(0, 3, 3, 1) = e_t_cl;
        v_Tcl.push_back(Tcl);
    }
}

void
Load_pose(string L_data_path,
          string R_data_path,
          vector<vector<Matrix4d>> &iter_pose_L,
          vector<vector<Matrix4d>> &iter_pose_R)
{
    vector<string> v_L_data;
    vector<string> v_R_data;
    load_file_path(L_data_path, v_L_data);
    load_file_path(R_data_path, v_R_data);
    for (int i = 0; i < v_L_data.size(); ++i)
    {
        vector<Matrix4d> pose_L;
        vector<Matrix4d> pose_R;
        read_poses(v_L_data[i], pose_L);
        read_poses(v_R_data[i], pose_R);
        iter_pose_L.push_back(pose_L);
        iter_pose_R.push_back(pose_R);
    }
}

int
main(int argc, char *argv[])
{
    readParameters("../cfg/config_real.yaml");
    Eigen::Vector3d t_lr_gt(-0.119994768246006, -0.000174184734047787, 0.000301631548705875);
    Eigen::Matrix3d R_lr_gt;
    R_lr_gt << 0.999999338992956, -5.13812077544178e-05, 0.00114863990135836,
        4.56035660014599e-05, 0.999987351218247, 0.00502944567806551,
        -0.00114888379145621, -0.00502938997149094, 0.999986692562730;

    string data_name = "real_world_data_"+string(argv[1]);
    string algorithm_name = string(argv[2]);

    string data_root_path = "/media/lab/915c94cc-246e-42f5-856c-faf39f774d2c/CalibData/"+data_name;
    string   L_intermediate_results_path = data_root_path + "/intermediate_results/L/"+algorithm_name+"";
    string   R_intermediate_results_path = data_root_path + "/intermediate_results/R/"+algorithm_name+"";
    ofstream cross_error_r_file (data_root_path+"/intermediate_results/cross_error_r_"+algorithm_name+".txt");
    ofstream cross_error_t_file (data_root_path+"/intermediate_results/cross_error_t_"+algorithm_name+".txt");

    vector<vector<Matrix4d>> iter_pose_L;
    vector<vector<Matrix4d>> iter_pose_R;
    Load_pose(L_intermediate_results_path, R_intermediate_results_path, iter_pose_L, iter_pose_R);
    for (int frame_num = 0; frame_num < iter_pose_L.size(); ++frame_num)
    {
        for (int iter_id = 0; iter_id < iter_pose_L[frame_num].size(); ++iter_id)
        {
            Eigen::Matrix4d T_cL_l = iter_pose_L[frame_num][iter_id];
            Eigen::Matrix4d T_cR_l = iter_pose_R[frame_num][iter_id];

            Eigen::Matrix4d T_cL_cR = T_cR_l*T_cL_l.inverse();
            cout<<"\n"<<T_cL_cR<<endl;
            double error_r = FrobeniusNorm(Eigen::Matrix3d::Identity() - R_lr_gt.transpose() * T_cL_cR.block(0, 0, 3, 3));
            double error_t = (T_cL_cR.block(0,3,3,1) - t_lr_gt).norm();
            cout << frame_num<<":"<<iter_id<<" error_t:" << error_t << "  error_r:" << error_r << endl;
            cross_error_r_file<<error_r<<" ";
            cross_error_t_file<<error_t<<" ";
        }
        cross_error_r_file<<endl;
        cross_error_t_file<<endl;
    }
    cross_error_r_file.close();
    cross_error_t_file.close();
    return 0;
}

