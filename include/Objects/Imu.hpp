#pragma once  

#include <Eigen/Dense>

namespace fast_limo {

  struct Imu {
    double stamp;
    Eigen::Vector3d ang_vel;
    Eigen::Vector3d lin_accel;
    Eigen::Quaterniond q;
  };


}