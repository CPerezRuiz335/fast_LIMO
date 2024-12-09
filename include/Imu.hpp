#pragma once

#include <Eigen/Dense>

namespace limoncello {
  
  struct Imu {
    double stamp;
    Eigen::Vector3d ang_vel;
    Eigen::Vector3d lin_accel;
    Eigen::Quaterniond q;
  };

}