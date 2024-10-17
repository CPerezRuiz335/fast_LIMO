#pragma once 

#include <Eigen/Dense>
#include <Eigen/Geometry>


namespace fast_limo {
  
  struct Extrinsics{
    Eigen::Affine3d imu2baselink_T;
    Eigen::Affine3d lidar2baselink_T;

    Extrinsics() {
      imu2baselink_T.setIdentity();
      lidar2baselink_T.setIdentity();
    }
  };

  struct IMUmeas{
    double stamp;
    double dt; // defined as the difference between the current and the previous measurement
    Eigen::Vector3d ang_vel;
    Eigen::Vector3d lin_accel;
    Eigen::Quaterniond q;
  };

}