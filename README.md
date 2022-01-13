# Livox-LiDAR-Camera Calibrator

### Laser Reflectance Feature Assisted Accurate Extrinsic Calibration for Non-repetitive LiDAR and Camera Systems

In this repository, we aim to build a automatic calibration tool for Livox-LiDAR-Camera system.

Since the paper related to this work are under peer-review, only the calibration results and part of the code are shown here this repository. The core optimization code has not yet been released, and will be fully open sourced in the near future.

If you have any questions or suggestions, or want to obtain the data supporting our paper, you can contact me by email: bit20lzc@163.com 

## Requirements
- PCL (>1.7)
- Eigen3(3.3.4)
- OpenCV (>3.0)
- ceres

## Status

- grid fitting process:

  <img src="resources/grid_fitting.gif" style="zoom:100%;" />

- Reproject results point cloud to image:

  <img src="resources/reprj-img.png" style="zoom:50%;" />

- Reproject results image pixel map to point clouds:

  <img src="resources/reprj-pc.png" style="zoom:100%;" />

  

## simulation process

The complete code of simulation tool has been uploaded to   [Livox_Cam_Simulator](https://github.com/zhijianglu/Livox_Cam_Simulator.git). Some result as shown in the following figures.

- The scan model of Livox LiDAR:

  <img src="resources/total.gif" style="zoom:50%;" />

- The zed camera combined with Livox LiDAR:

  <img src="resources/LC-model.png" style="zoom:50%;" />

- The Gazebo scene:

   <img src="resources/gazebo_scene.png" style="zoom:50%;" />

- The rviz visulation :

   <img src="resources/rviz_pc.png" style="zoom:50%;" />

Point clouds with reflectance intensity which mapped according to the color of the materials:

 <img src="resources/with_intensity.png" style="zoom:50%;" />



## Configuration(example by avia)

- laser_min_range: 0.1  // min detection range

- laser_max_range: 200.0  // max detection range

- horizontal_fov: 70.4   //°

- vertical_fov: 77.2    //°

- ros_topic: scan // topic in ros

- samples: 24000  // number of points in each scan loop

- downsample: 1 // we can increment this para to decrease the consumption

- publish_pointcloud_type: 0 // 0 for sensor_msgs::PointCloud, 1 for sensor_msgs::Pointcloud2(PointXYZ), 2 for sensor_msgs::PointCloud2(LivoxPointXyzrtl) 3 for livox_ros_driver::CustomMsg.

- LiDAR-Camera pose: (set  LiDAR coordinate system as the world system)

  ```
      <arg name="zed2_x" default="0.04"/>
      <arg name="zed2_y" default="0.08"/>
      <arg name="zed2_z" default="-0.061"/>
  
      <arg name="zed2_roll" default="0.015"/>
      <arg name="zed2_pitch" default="0.032"/>
      <arg name="zed2_yaw" default="0.061"/>
  ```


