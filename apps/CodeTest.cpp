//
// Created by lab on 2021/12/14.
//
#include "global.h"
#include "utils.h"

#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/keypoints/uniform_sampling.h>
int main(){
    readParameters("../cfg/config_sim.yaml");

    string pc_path = "/new_space/Calibrate/other_methods/ILCC_ws/data3/pcd";
    string out_pc_path = "/new_space/Calibrate/other_methods/ILCC_ws/ILCC/pcd";
    vector<string> target_pc_files;
    load_file_path(pc_path, target_pc_files);
    for (int frame_idx = 0; frame_idx < target_pc_files.size(); ++frame_idx)
    {
        pcl::PointCloud<PoinT>::Ptr chess_board(new pcl::PointCloud<PoinT>);
        stringstream file_name;
        file_name << out_pc_path <<  "/" << setw(4) << setfill('0') << frame_idx << ".txt";
        pcl::io::loadPCDFile<PoinT>(target_pc_files[frame_idx], *chess_board);
        pcl::UniformSampling<PoinT> US;
        US.setInputCloud(chess_board);
        US.setRadiusSearch(0.1);
        US.filter(*chess_board);

    }

}