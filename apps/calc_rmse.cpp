//
// Created by lab on 2022/1/8.
//


#include <string>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
using namespace std;
using namespace Eigen;

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


Eigen::Vector3d
rotationMatrixToEulerAngles(const Eigen::Matrix3d R)
{
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


int main(int argc, char **argv) {
    Eigen::Vector3d t_lr_gt(-0.119994768246006, -0.000174184734047787, 0.000301631548705875);
    Eigen::Matrix3d R_lr_gt;
    R_lr_gt << 0.999999338992956, 4.56035660014599e-05, -0.00114888379145621,
        -5.13812077544178e-05, 0.999987351218247, -0.00502938997149094,
        0.00114863990135836, 0.00502944567806551 , 0.999986692562730;
    Eigen::Vector3d eulerAngle_gt = rotationMatrixToEulerAngles(R_lr_gt);
    cout<<"gt trans"<<1000.0*t_lr_gt.transpose()<<endl;
    cout<<"gt eular"<<1000.0*eulerAngle_gt.transpose()<<endl;

//    //indoor 第六组误差:
//    vector<double> err
//        {0.00290322, 0.00672774, 0.00917004, 0.00397149, 0.00503796, 0.00216223, 0.010214, 0.00587072, 0.00331752,
//         0.00353078, 0.00644781, 0.00535275, 0.00568292, 0.00705918, 0.00455075, 0.0072905, 0.00322361, 0.00574616,
//         0.00600903, 0.00438096, 0.00864532, 0.00814755, 0.00484872, 0.00285108, 0.0118701, 0.00390083, 0.0071121,
//         0.00431152, 0.00494256, 0.00552353};

    //outdoor 第六组误差:
    vector<double> err
        {0.00681016, 0.00754979, 0.00543653, 0.00620938, 0.00664616, 0.00542783, 0.00776292, 0.00686705, 0.00473081,
         0.00515263, 0.00697178, 0.00625069, 0.0064709, 0.00647485, 0.00598959, 0.00676013, 0.00605955, 0.00527193,
         0.0058098, 0.00494896, 0.00551903, 0.00576172, 0.0066745, 0.00528792, 0.00622681, 0.00480505, 0.00581097,
         0.0056832, 0.00565476, 0.0056538};

    vector<pair<double, int>> err_lists;  // distance, index
    for (int i = 0; i < err.size(); ++i)
    {
        err_lists.push_back(make_pair(err[i], i));
    }
    std::sort(err_lists.begin(), err_lists.end());

    cout<<err_lists[int(err.size()/2)].first<<" "<<err_lists[int(err.size()/2)].second<<endl;
//    中位数所在次数求解出来了 13
//indoor R  0.999776 -0.0177691 -0.00314187 0.0110607 -0.0426148 -0.087164 -0.0198392
//indoor L 0.999731 -0.0202243 -0.00235128 0.0111032 0.0748039 -0.0858426 -0.02457

//qw qx qy qz x y z
//outdoor R  0.999795 -0.0168687 -0.000874041 0.0111297 -0.0496541 -0.0806369 0.0058639
//outdoor L 0.999747 -0.0195042 -0.00029811 0.011238 0.068088 -0.0805071 0.000370015

//------------------------------------------------------------

// indoor
//    Eigen::Quaterniond Q_cl_R(0.999776, -0.0177691 ,-0.00314187, 0.0110607  );
//    Eigen::Vector3d e_t_cl_R( -0.0426148 ,-0.087164 ,-0.0198392);
//    Eigen::Quaterniond Q_cl_L(0.999731 ,-0.0202243, -0.00235128 ,0.0111032 );
//    Eigen::Vector3d  e_t_cl_L( 0.0748039, -0.0858426, -0.02457);

    //outdoor
    Eigen::Quaterniond Q_cl_R(0.999795, -0.0168687, -0.000874041, 0.0111297 );
    Eigen::Vector3d e_t_cl_R( -0.0496541, -0.0806369, 0.0058639);
    Eigen::Quaterniond Q_cl_L(0.999747, -0.0195042, -0.00029811 ,0.011238 );
    Eigen::Vector3d  e_t_cl_L( 0.068088, -0.0805071 ,0.000370015);

    Matrix4d Tcl_R;
    cout<<"R t:"<<1000.0*rotationMatrixToEulerAngles(Q_cl_R.matrix()).transpose()<<endl;
    cout<<"R eular:"<<1000.0*e_t_cl_R.transpose()<<endl;
    Tcl_R.setIdentity();
    Tcl_R.block(0, 0, 3, 3) = Q_cl_R.toRotationMatrix();
    Tcl_R.block(0, 3, 3, 1) = e_t_cl_R;
    Matrix4d Tcl_L;
    cout<<"L t:"<<1000.0*e_t_cl_L.transpose()<<endl;
    cout<<"L eular:"<<1000.0*rotationMatrixToEulerAngles(Q_cl_L.matrix()).transpose()<<endl;
    Tcl_L.setIdentity();
    Tcl_L.block(0, 0, 3, 3) = Q_cl_L.toRotationMatrix();
    Tcl_L.block(0, 3, 3, 1) = e_t_cl_L;

    Eigen::Matrix4d T_cL_cR = Tcl_R*Tcl_L.inverse();
    cout<<"T cl cr eular:"<<1000.0*rotationMatrixToEulerAngles(T_cL_cR.block(0,0,3,3)).transpose()<<endl;
    cout<<"eulerAngle_gt eular:"<<1000.0*eulerAngle_gt.transpose()<<endl;
    cout<<"difference t:"<<1000.0*(T_cL_cR.block(0,3,3,1)-t_lr_gt).transpose()<<endl;
    cout<<"difference r:"<<1000.0*(rotationMatrixToEulerAngles(T_cL_cR.block(0,0,3,3))-eulerAngle_gt).transpose()<<endl;
}
//gt trans -119.995 -0.174185  0.301632
//gt eular   5.02947   -1.14864 -0.0513812
//
//indoor
//R t:-35.6079 -5.88929  22.2303
//R eular:-42.6148  -87.164 -19.8392
//L t: 74.8039 -85.8426   -24.57
//L eular:-40.5014  -4.2522  22.2976
//T cl cr eular:   4.92869   -1.52763 -0.0462151
//eulerAngle_gt eular:   5.02947   -1.14864 -0.0513812
//difference t:2.54194 -1.2659 4.73766
//difference r: -0.100777  -0.378992 0.00516618


//outdoor
//R t:-33.7564 -1.37224  22.2862
//R eular:-49.6541 -80.6369   5.8639
//L t:  68.088 -80.5071 0.370015
//L eular: -39.0151 -0.157693   22.4838
//T cl cr eular: 5.28472 -1.09601 -0.19652
//eulerAngle_gt eular:   5.02947   -1.14864 -0.0513812
//difference t:  2.26847 0.0585949   5.54309
//difference r: 0.255252 0.0526271 -0.145139

/*
\begin{table}[h]
\centering
\begin{tabular}{cccccccc}
\toprule
    scene  &         item   & $t_x$ & $t_y$ & $t_z$ & $yaw$ & $pitch$ & $roll$ \\ \hline
\multicolumn{1}{l}{\multirow{4}{*}{indoor}}  & $\mathrm{T}_{\mathrm{LC}_r}$ & -35.608 & -5.890 & 22.230 & -42.615  &-87.164    & -19.840 \\ %\cline{2-2}
\multicolumn{1}{l}{}                         & $\mathrm{T}_{\mathrm{LC}_l}$ & 74.804  & -85.843 &  -24.57 & -40.5014  &-4.2522    &  22.298  \\ %\cline{2-2}
\multicolumn{1}{l}{}                         & Difference                   & 2.542  & -1.266  & 4.738 & -0.101 & -0.379 &  0.005  \\ %\cline{2-2}
\multicolumn{1}{l}{}                         & RMSD                         & 2.931& 2.45& 4.772& 0.607 & 0.474371 & 0.276031  \\ \hline
\multicolumn{1}{l}{\multirow{4}{*}{outdoor}} & $\mathrm{T}_{\mathrm{LC}_r}$ & -33.756 & -1.372  & 22.286  & -49.654 & -80.637  &  5.864  \\ %\cline{2-2}
\multicolumn{1}{l}{}                         & $\mathrm{T}_{\mathrm{LC}_l}$ & 68.088   & -80.507  & 0.370 & -39.015 & -0.158 &   22.484  \\ %\cline{2-2}
\multicolumn{1}{l}{}                         & Difference                   & 2.268  & 0.059 & 5.543  & 0.255 & 0.0526 & -0.145 \\ %\cline{2-2}
\multicolumn{1}{l}{}                         & RMSD                         & 3.061 &1.612 &4.992 & 0.381 & 0.506 & 0.199  \\ \bottomrule
\end{tabular}
\end{table}
 */