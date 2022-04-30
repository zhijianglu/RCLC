//
// Created by lab on 2021/12/8.
//

#ifndef CHECKERBOARD_VIS_H
#define CHECKERBOARD_VIS_H
#include "global.h"
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>

void display_standard_gride_lines(visualization::PCLVisualizer::Ptr &viewer){
    if(viewer == nullptr)
        return;
    viewer->removeAllShapes();
    double init_x = -(CfgParam.board_size.x()/2.0)*CfgParam.gride_side_length;
    double init_y = -(int(CfgParam.board_size.y()/2.0)-1)*CfgParam.gride_side_length;
    Eigen::Vector3d p0(init_x, init_y, 0.0);
    for (int line_id = 0; line_id < CfgParam.board_size.y() - 1; ++line_id)
    {
        DisplayType p1;
        p1.x = p0.x();
        p1.y = p0.y() + CfgParam.gride_side_length * (double) line_id;
        p1.z = 0;
        p1.r = 150;
        p1.g = 150;
        p1.b = 150;

        DisplayType p2;
        p2.x = p0.x() + CfgParam.gride_side_length * CfgParam.board_size.x();
        p2.y = p0.y() + CfgParam.gride_side_length * (double) line_id;
        p2.z = 0;
        p2.r = 150;
        p2.g = 150;
        p2.b = 150;

        viewer->addLine<DisplayType>(p1, p2, 0, 255, 255, "row_2d" + to_string(line_id));

        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                                            6,
                                            "row_2d" + to_string(line_id));
    }

    p0.x() += CfgParam.gride_side_length;
    p0.y() -= CfgParam.gride_side_length;
    for (int line_id = 0; line_id < CfgParam.board_size.x() - 1; ++line_id)
    {

        DisplayType p1;
        p1.x = p0.x() + CfgParam.gride_side_length * (double) line_id;
        p1.y = p0.y();
        p1.z = 0;
        p1.r = 150;
        p1.g = 150;
        p1.b = 150;

        DisplayType p2;
        p2.x = p0.x() + CfgParam.gride_side_length * (double) line_id;
        p2.y = p0.y() + CfgParam.gride_side_length * CfgParam.board_size.y();;
        p2.z = 0;
        p2.r = 150;
        p2.g = 150;
        p2.b = 150;

        viewer->addLine<DisplayType>(p1, p2, 0, 255, 255, "col_2d" + to_string(line_id));

        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                                            6,
                                            "col_2d" + to_string(line_id));
    }
}

void display_points(visualization::PCLVisualizer::Ptr viewer, vector<Eigen::Vector3d> points,vector<int>colors, int display_size=15, bool remove_before = false){

    if(viewer==nullptr)
        return;
    if(remove_before)
        viewer->removeAllPointClouds();

    pcl::PointCloud<DisplayType>::Ptr cloud2show(new pcl::PointCloud<DisplayType>);
    cloud2show->resize(points.size());
    for (int i = 0; i < points.size(); ++i)
    {
        DisplayType pt;
        cloud2show->points[i].x = points[i].x();
        cloud2show->points[i].y = points[i].y();
        cloud2show->points[i].z = points[i].z();
        cloud2show->points[i].r = colors[0];
        cloud2show->points[i].g = colors[1];
        cloud2show->points[i].b = colors[2];
    }
    static int display_idx = 0;
    viewer->addPointCloud<DisplayType>(cloud2show, "points"+to_string(display_idx));
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             display_size,
                                             "points"+to_string(display_idx));
    display_idx++;
}

void add_pointClouds_show(visualization::PCLVisualizer::Ptr viewer, pcl::PointCloud<PoinT>::Ptr cloud2show, bool remove_before= true, string cloud_id= "cloud", int show_size = 1){
    if (remove_before)
        viewer->removeAllPointClouds();
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI>
        fildColor(cloud2show, "intensity"); // 按照z字段进行渲染

    viewer->addPointCloud<PoinT>(cloud2show, fildColor, cloud_id);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             show_size,
                                             cloud_id);
    viewer->spin();
}

void add_pointClouds_show(visualization::PCLVisualizer::Ptr viewer, pcl::PointCloud<DisplayType>::Ptr cloud2show, bool remove_before= true, string cloud_id= "cloud", int show_size = 1){
    if (remove_before)
        viewer->removeAllPointClouds();

    viewer->addPointCloud<DisplayType>(cloud2show, cloud_id);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             show_size,
                                             cloud_id);
    viewer->spin();
}


//void eular_cluster()

#endif //CHECKERBOARD_VIS_H
