//
// Created by lab on 2021/12/8.
//

#ifndef CHECKERBOARD_LC_CALIB_UTILS_H
#define CHECKERBOARD_LC_CALIB_UTILS_H
#include "global.h"
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>


void
load_file_path(string file_path, std::vector<string> &vstr_file_path)
{

    if (getdir(file_path, vstr_file_path) >= 0)
    {
//        printf("found %d files in folder %s!\n",
//               (int) vstr_file_path.size(),
//               file_path.c_str());
    }
    else if (getFile(file_path.c_str(), vstr_file_path) >= 0)
    {
//        printf("found %d files in file %s!\n",
//               (int) vstr_file_path.size(),
//               file_path.c_str());
    }
    else
    {
        printf("could not load file list! wrong path / file?\n");
    }

    sort(vstr_file_path.begin(), vstr_file_path.end(), [](string x, string y)
         {
             string s_time = x
                 .substr(x.find_last_of("/") + 1, x.find_last_of(".") - x.find_last_of("/") - 1);
             double a_stamp = atof(s_time.c_str());

             s_time = y
                 .substr(y.find_last_of("/") + 1, y.find_last_of(".") - y.find_last_of("/") - 1);
             double b_stamp = atof(s_time.c_str());
             return a_stamp < b_stamp;
         }
    );
}

void
calc_transfered_standard_corners(vector<Eigen::Vector3d> &v_est_pc_corner_pt, Eigen::Matrix4d &T_laser_base)
{
    double gride_side_length = CfgParam.gride_side_length;
    double first_corner_x = double(int((CfgParam.board_size.x() - 1) / 2)) * gride_side_length;
    double first_corner_y = double(int((CfgParam.board_size.y() - 1) / 2)) * gride_side_length;

    for (int row = 0; row < CfgParam.board_size.y() - 1; ++row)
    {
        for (int col = 0; col < CfgParam.board_size.x() - 1; ++col)
        {
            Eigen::Vector4d std_pt(first_corner_x - double(col) * CfgParam.gride_side_length,
                                   first_corner_y - double(row) * CfgParam.gride_side_length,
                                   0,
                                   1
            );
            v_est_pc_corner_pt.push_back((T_laser_base * std_pt).topRows(3));
        }
    }
}

void
read_sim_pc_corner(string file_path, vector<Eigen::Vector3d> &v_gt_pc_corner_pt)
{

    std::string data_line_pose;
    std::string header;
    Eigen::Vector3d corner;
    ifstream corner_ifs(file_path);

    while (std::getline(corner_ifs, data_line_pose) && !corner_ifs.eof())
    {
        std::istringstream poseData(data_line_pose);
        poseData >> header;
        if (header == "#" || header == "*")
            continue;

        corner.x() = atof(header.c_str());
        poseData >> corner.y() >> corner.z();
        v_gt_pc_corner_pt.push_back(corner);
        for (int j = 0; j < CfgParam.board_size.x() - 2; ++j)
        {
            poseData >> corner.x() >> corner.y() >> corner.z();
            v_gt_pc_corner_pt.push_back(corner);
        }
    }
}

void
read_img_corner_matlab(string file_path, vector<Eigen::Vector2d> &v_img_corner_pt)
{
    std::string data_line_pose;
    std::string header;
    Eigen::Vector2d corner;
    ifstream corner_ifs(file_path);

    while (std::getline(corner_ifs, data_line_pose) && !corner_ifs.eof())
    {
        std::istringstream poseData(data_line_pose);
        poseData >> header;
        if (header == "#" || header == "*")
            continue;

        corner.x() = atof(header.c_str());
        poseData >> corner.y();
        v_img_corner_pt.push_back(corner);
        for (int j = 0; j < CfgParam.board_size.x() - 2; ++j)
        {
            poseData >> corner.x() >> corner.y();
            v_img_corner_pt.push_back(corner);
        }
    }
}

void
read_sim_img_corner(string file_path, vector<Eigen::Vector2d> &v_gt_img_corner_pt)
{
    std::string data_line_pose;
    std::string header;
    Eigen::Vector2d corner;
    ifstream corner_ifs(file_path);

    while (std::getline(corner_ifs, data_line_pose) && !corner_ifs.eof())
    {
        std::istringstream poseData(data_line_pose);
        poseData >> header;
        if (header == "#" || header == "*")
            continue;

        corner.x() = atof(header.c_str());
        poseData >> corner.y();
        v_gt_img_corner_pt.push_back(corner);
        for (int j = 0; j < CfgParam.board_size.x() - 2; ++j)
        {
            poseData >> corner.x() >> corner.y();
            v_gt_img_corner_pt.push_back(corner);
        }
    }
}

void
rand_rgb(int *rgb)
{//随机产生颜色
    rgb[0] = rand() % 255;
    rgb[1] = rand() % 255;
    rgb[2] = rand() % 255;
}

/*
 * 计算提取的标定板黑色块中心点中，取出想要的参考点（右上角点），注意，这里一定要是标定板右上角是黑色块，这是预先设定好的合理摆放规则
 *  先计算点云中心，然后以该点为中心，按照xy轴平行的方向，将这些点分成四个块，位于右上角的块里面，距离中心最近的那个点就是参考点
 */
int
get_ref_pt(vector<Eigen::Vector3d> &v_grid_centroid, Eigen::Vector3d &ref_pt, int &ref_idx)
{
    Eigen::Vector3d c(0, 0, 0);
    for (int i = 0; i < v_grid_centroid.size(); ++i)
    {
        c += Eigen::Vector3d(v_grid_centroid[i].x(), v_grid_centroid[i].y(), v_grid_centroid[i].z());
    }
    c /= v_grid_centroid.size();

    vector<pair<double, int>> corner_pt{4, make_pair(0.0, 0)};
    vector<Eigen::Vector3d> corner_pt_e;

    for (int idx = 0; idx < v_grid_centroid.size(); ++idx)
    {
        Eigen::Vector3d &p = v_grid_centroid[idx];

        double dst_2 = (p - c).norm();
        int axis = 0;

        if (p.x() > c.x())
        {
            if (p.y() > c.y())
                axis = 0;//右下
            else
                axis = 1; //右上
        }
        else
        {
            if (p.y() > c.y())
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
    ref_pt = v_grid_centroid[corner_pt[1].second];
    ref_idx = corner_pt[1].second;
}

void
calc_line_param(vector<Eigen::Vector2d> &points, Eigen::Vector3d &params)
{
//    这里使用svd分解进行DLT求解，计算拟合空间直线
    Eigen::Vector2d col_center_point;
    col_center_point.setZero();
    int n_pt = points.size();

    for (int i = 0; i < n_pt; ++i)
        col_center_point += points[i];

    col_center_point /= n_pt;

    Eigen::MatrixXd A_j;
    A_j.resize(n_pt, 2);

    for (int i = 0; i < n_pt; ++i)
    {
        A_j.row(i) = (points[i] - col_center_point).transpose();
    }

    Eigen::Vector3d line_abc;           // 结果保存到这个变量
    line_abc.setZero();
    Eigen::MatrixXd U, V, W;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A_j, Eigen::ComputeThinU | Eigen::ComputeThinV);
    U = svd.matrixU();
    V = svd.matrixV();
    W = svd.singularValues();

    params[0] = V.col(1)[0];
    params[1] = V.col(1)[1];
//    params[2] = -V.col(1).topRows(2).dot(col_center_point);
    params[2] = -(params[0] * col_center_point[0] + params[1] * col_center_point[1]);
}

void
calc_row_line_fitting(
    vector<Eigen::Vector3d> &pc_cross_rows,
    Eigen::Vector4d &v_plane_param_l,
    Eigen::Matrix<double, 6, 1> &Spatial_line_param)
{
    vector<Eigen::Vector2d> points;
    Eigen::Vector3d centroid(0, 0, 0);
    for (int pt_id = 0; pt_id < pc_cross_rows.size(); ++pt_id)
    {
        Eigen::Vector3d &p3d = pc_cross_rows[pt_id];
        points.push_back(p3d.topRows(2));  //x, y, 参与计算
        centroid += p3d;
    }
    centroid /= (double) pc_cross_rows.size();
    Eigen::Vector3d prj_line2d_direction;
    calc_line_param(points, prj_line2d_direction);
    Eigen::Vector3d line2d_direction(-prj_line2d_direction[1] / prj_line2d_direction[0], 1.0, 0);  //投影直线的方向向量
    line2d_direction.normalize();
    Eigen::Vector3d p1_inline2d = Eigen::Vector3d(0, -prj_line2d_direction[2] / prj_line2d_direction[1], 0);

    Eigen::Vector3d c = p1_inline2d - centroid;
    Eigen::Vector4d plane_x_param;
    plane_x_param.topRows(3) = line2d_direction.cross(c);
    plane_x_param[3] = -plane_x_param.topRows(3).dot(centroid);

    Eigen::Vector3d board_norm = v_plane_param_l.topRows(3);
    Eigen::Vector3d prj_norm = plane_x_param.topRows(3);
    Eigen::Vector3d line_direction = board_norm.cross(prj_norm);
    line_direction.normalize();
    Spatial_line_param.topRows(3) = line_direction;
    Spatial_line_param.bottomRows(3) = centroid;
}

template<typename T>
Eigen::Matrix<T, 4, 4>
SE2_to_SE3(double x, double y, double yaw_ang)
{
    Eigen::Matrix<T, 4, 4> T_wc;
    T_wc.setIdentity();
    T_wc.block(0, 3, 3, 1) << x, y, 0;
    double ang_opt = (yaw_ang * double(M_PI)) / double(180.0);
    Eigen::Quaterniond Q_opt(cos(ang_opt / 2.0), 0.0, 0.0, sin(ang_opt / 2.0));
    T_wc.block(0, 0, 3, 3) = Q_opt.matrix().cast<T>();
    return T_wc;
}

void
generate_std_corners(vector<Eigen::Vector3d> &row_points, vector<Eigen::Vector3d> &col_points)
{
    int centroid_x_id = CfgParam.board_size.x() / 2;
    int centroid_y_id = CfgParam.board_size.y() / 2;
    double gride_side_length = CfgParam.gride_side_length;
    double row_shift_x = centroid_x_id * gride_side_length;
    double row_shift_y = double(centroid_y_id - 1) * gride_side_length;

    for (int r = 0; r < CfgParam.board_size.y() - 1; ++r)
    {
        for (int c = 0; c < CfgParam.board_size.x(); ++c)
        {
            row_points.emplace_back(gride_side_length / 2.0 + c * gride_side_length - row_shift_x,
                                    r * gride_side_length - row_shift_y,
                                    0);
        }
    }

    double col_shift_y = double(centroid_y_id) * gride_side_length;
    double col_shift_x = double(centroid_x_id - 1) * gride_side_length;
    for (int c = 0; c < CfgParam.board_size.x() - 1; ++c)
    {
        for (int r = 0; r < CfgParam.board_size.y(); ++r)
        {
            col_points.emplace_back(c * gride_side_length - col_shift_x,
                                    gride_side_length / 2.0 + r * gride_side_length - col_shift_y,
                                    0.0);
        }
    }
}

void
est_img_corner(cv::Mat &curr_img, vector<Eigen::Vector2d> &img_corner_points)
{
    if (CfgParam.need_undistort)
        Cam.undistort(curr_img, curr_img);

    bool use_sb = true;
//        cout << "==========" << endl;


    vector<cv::Point2f> image_points_buf;  /* 缓存每幅图像上检测到的角点 */

    cv::Size cv_board_size(CfgParam.board_size.x() - 1, CfgParam.board_size.y() - 1);

    if (use_sb)
    {

        if (0 == findChessboardCornersSB(curr_img, cv_board_size, image_points_buf,  cv::CALIB_CB_ACCURACY))
        {
            cout << "corner detect failed, retry" << endl;

            if (0 == findChessboardCornersSB(curr_img, cv_board_size, image_points_buf,  cv::CALIB_CB_ACCURACY))
            {
                exit(1);
            }
            else
            {
                cout << "corner detected in second try" << endl;
            }
        }
    }
    else
    {
        if (0 == findChessboardCorners(curr_img, cv_board_size, image_points_buf))
            exit(1);

        cv::Mat view_gray;
        cvtColor(curr_img, view_gray, CV_RGB2GRAY);
        find4QuadCornerSubpix(view_gray, image_points_buf, cv::Size(11, 11)); //对粗提取的角点进行精确化  Size是角点搜索窗口的尺寸
    }

//      这里统一转换成右下角到左上角的顺序,本工程同一采用右下角到左上角的角点顺序:容器内第一个位置的点x和y的值都应该是比最后一个点要大
    bool rd2lu = false;
    int num = image_points_buf.size();
    if (image_points_buf[0].x > image_points_buf.back().x &&
        image_points_buf[0].y > image_points_buf.back().y
        )
    {
        rd2lu = true;
    }

    for (int j = 0; j < image_points_buf.size(); ++j)
    {
        cv::Point2f p;
        if (rd2lu)
            p = image_points_buf[j];
        else
            p = image_points_buf[num - 1 - j];
        img_corner_points.emplace_back(p.x, p.y);
    }
}

int
get_random_lists(int sampleNum, int *rand_ids, int seed = -1)
{
//    srand(time(0));
    if (seed == -1)
        srand(time(0));
    else
        srand(seed);
//    int * rand_ids = new int[sampleNum];//打乱下标
    //原始下标，图像
    int ids[sampleNum];
    for (int i = 0; i < sampleNum; i++)
        ids[i] = i;

    //打乱下标
    int j = 0;
    for (int i = sampleNum; i > 0; i--)
    {
        int tmpID = rand() % i;
        rand_ids[j++] = ids[tmpID];
        ids[tmpID] = ids[i - 1];
    }

}

bool
isRotationMatirx(Eigen::Matrix3d R)
{
    double err = 1e-5;

    Eigen::Matrix3d shouldIdenity;
    shouldIdenity = R * R.transpose();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    if ((shouldIdenity - I).norm() > err)
    {
        cout<<"(shouldIdenity - I).norm():"<<(shouldIdenity - I).norm()<<endl;
        return false;
    }
    else
        return true;
}

Eigen::Vector3d
rotationMatrixToEulerAngles(const Eigen::Matrix3d &R)
{
    assert(isRotationMatirx(R));
    double sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if (!singular)
    {
        x = atan2(R(2, 1), R(2, 2));
        y = atan2(-R(2, 0), sy);
        z = atan2(R(1, 0), R(0, 0));
    }
    else
    {
        x = atan2(-R(1, 2), R(1, 1));
        y = atan2(-R(2, 0), sy);
        z = 0;
    }
    return {x, y, z};
}
Eigen::Matrix3d
euler2RotationMatrix(Eigen::Vector3d pyr)
{
    Eigen::AngleAxisd rollAngle(pyr.z(), Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd yawAngle(pyr.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(pyr.x(), Eigen::Vector3d::UnitX());
    Eigen::Matrix3d R;
    R = rollAngle * yawAngle * pitchAngle;
    return R;
}
double
FrobeniusNorm(Eigen::Matrix3d e_R)
{
    double sum = 0;
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            sum += (e_R(r, c) * e_R(r, c));
        }
    }
    return sqrt(sum);
}

double
calc_norm_reprj_error(const Eigen::Matrix3d &e_R_cl,
                      const Eigen::Vector3d &e_t_cl,
                      const std::vector<Eigen::Vector2d> &imagePoints,
                      const std::vector<Eigen::Vector3d> &objectPoints)
{
    double norm_reprj_error = 0;
    double depth_max = 0;
    std::vector<double> v_weighted_error;
    for (int i_pt = 0; i_pt < objectPoints.size(); ++i_pt)
    {
        Eigen::Vector3d P_l = objectPoints[i_pt];
        Eigen::Vector3d P_c = e_R_cl * P_l + e_t_cl;
        Eigen::Vector3d nP_c = P_c / P_c.z();
        double pt_norm = nP_c.norm();
        if (depth_max < pt_norm)
            depth_max = pt_norm;

        Eigen::Vector2d np_c_prj = Cam.normPt2img(nP_c);
        Eigen::Vector2d np_c_tag = imagePoints[i_pt];
        norm_reprj_error += (pt_norm * (np_c_prj - np_c_tag).norm());
    }
    return norm_reprj_error / (depth_max*double(objectPoints.size()));
}

double
calc_reprj_error(const Eigen::Matrix3d &e_R_cl,
                 const Eigen::Vector3d &e_t_cl,
                 const std::vector<cv::Point2f> &imagePoints,
                 const std::vector<cv::Point3f> &objectPoints)
{
    double e_prj_pnp = 0;
    for (int i_pt = 0; i_pt < objectPoints.size(); ++i_pt)
    {
        Eigen::Vector3d P_l(objectPoints[i_pt].x, objectPoints[i_pt].y, objectPoints[i_pt].z);
        Eigen::Vector3d P_c = e_R_cl * P_l + e_t_cl;
        Eigen::Vector3d nP_c = P_c / P_c.z();
//        Eigen::Vector2d np_c_prj(nP_c.x() * CfgParam.fx + CfgParam.cx, nP_c.y() * CfgParam.fy + CfgParam.cy);
        Eigen::Vector2d np_c_prj = Cam.normPt2img(nP_c);
        Eigen::Vector2d np_c_tag(imagePoints[i_pt].x, imagePoints[i_pt].y);
        e_prj_pnp += (np_c_prj - np_c_tag).norm();
    }
    return e_prj_pnp / double(objectPoints.size());
}

void
evaluate(const Eigen::Matrix3d &e_R_cl,
         const Eigen::Vector3d &e_t_cl,
         const std::vector<cv::Point2f> &imagePoints,
         const std::vector<cv::Point3f> &objectPoints,
         ofstream &error_log)
{


    Eigen::Vector3d est_eulerAngle = rotationMatrixToEulerAngles(e_R_cl);
    Eigen::Vector3d error_eulerAngle = est_eulerAngle - CfgParam.sim_gt_cl_rpy;
    Eigen::Vector3d error_t_cl = e_t_cl - CfgParam.sim_gt_cl_t;
    double e_prj_pnp = calc_reprj_error(e_R_cl, e_t_cl, imagePoints, objectPoints);
    error_log << error_t_cl.x() << " " << error_t_cl.y() << " " << error_t_cl.z() << " " << error_eulerAngle.x()
              << " " << error_eulerAngle.y() << " " << error_eulerAngle.z() << " " << error_t_cl.norm() << " "
              << FrobeniusNorm(Eigen::Matrix3d::Identity() - CfgParam.sim_gt_R_cl.transpose() * e_R_cl) << " "
              << e_prj_pnp << endl;
}

static std::string get_name(std::string & s){

    int start_pos = s.find_last_of("/");
    int stop_pos = s.find_last_of(".");
    std::string s_out = s.substr(start_pos + 1, stop_pos - start_pos - 1);
    return s_out;
}

#endif //CHECKERBOARD_LC_CALIB_UTILS_H
