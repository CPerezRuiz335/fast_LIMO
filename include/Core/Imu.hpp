#pragma once

#include <Eigen/Dense>

#include "Utils/Config.hpp"

struct Imu {
  double stamp;
  Eigen::Vector3d ang_vel;
  Eigen::Vector3d lin_accel;
  Eigen::Quaterniond q;

  Imu() : stamp(0.),
          ang_vel(Eigen::Vector3d::Zero()),
          lin_accel(Eigen::Vector3d::Zero()),
          q(Eigen::Quaterniond::Identity()) {}
};

Imu imu2baselink(const Imu& imu, const double& dt) {
  
  Config& cfg = Config::getInstance();

  static Eigen::Matrix3d R = cfg.sensors.extrinsics.imu2baselink_T.linear();
  static Eigen::Vector3d t = cfg.sensors.extrinsics.imu2baselink_T.translation();

  Eigen::Vector3d ang_vel_cg = R * imu.ang_vel;
  static Eigen::Vector3d ang_vel_cg_prev = ang_vel_cg;

  Eigen::Vector3d lin_accel_cg = R * imu.lin_accel;
  lin_accel_cg = lin_accel_cg
                  + ((ang_vel_cg - ang_vel_cg_prev) / dt).cross(-t)
                  + ang_vel_cg.cross(ang_vel_cg.cross(-t));

  ang_vel_cg_prev = ang_vel_cg;

  Imu out = imu;

  out.ang_vel   = ang_vel_cg;
  out.lin_accel = lin_accel_cg;

  Eigen::Quaterniond q(R);
  q.normalize();
  out.q = q * imu.q;

  return out;
}