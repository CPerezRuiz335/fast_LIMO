#pragma once

#include <boost/circular_buffer.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "use-ikfom.hpp"
#include "Config.hpp"
#include "Imu.hpp"


struct State {

  struct IMUbias {
    Eigen::Vector3d gyro;
    Eigen::Vector3d accel;
  } b;           

  Eigen::Vector3d p;      // position in global/world frame
  Eigen::Quaterniond q;   // orientation in global/world frame
  Eigen::Vector3d v;      // linear velocity
  Eigen::Vector3d g;      // gravity vector
  
  Eigen::Vector3d w;      // angular velocity (IMU input)
  Eigen::Vector3d a;      // linear acceleration (IMU input)

  // Extrinsics
  Eigen::Affine3f I2L;

  double stamp;

  State() : stamp(0.0) { 
    Config& cfg = Config::getInstance();

    q = Eigen::Quaterniond(0., 0., 0., 1.);
    p.setZero();
    v.setZero();
    w.setZero();
    a.setZero();
    g = Eigen::Vector3d(0., 0., cfg.sensors.extrinsics.gravity);

    // Extrinsics 
    I2L = cfg.sensors.extrinsics.lidar2baselink_T.cast<float>();

    b.gyro.setZero();
    b.accel.setZero();
  }


  State(const state_ikfom& s, const Imu& imu = Imu()) {
    
    a = imu.lin_accel;
    w = imu.ang_vel;
    stamp = imu.stamp;

    // Odom
    q = s.rot;
    p = s.pos;
    v = s.vel;

    // Gravity
    g = s.grav.get_vect();

    // IMU bias
    b.gyro = s.bg;
    b.accel = s.ba;

    // Offset LiDAR-IMU
    I2L.linear() = s.offset_R_L_I.toRotationMatrix().cast<float>();
    I2L.translation() = s.offset_T_L_I.cast<float>();
  }


  void update(const double& t) {
    // R ⊞ (w - bw - nw)*dt
    // v ⊞ (R*(a - ba - na) + g)*dt
    // p ⊞ (v*dt + 1/2*(R*(a - ba - na) + g)*dt*dt)

    double dt = t - stamp;
    assert(0 <= dt and dt < 1); // TODO
    // Exp orientation
    Eigen::Vector3d w_corrected = w - b.gyro;
    double w_norm = w_corrected.norm();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d K;

    if (w_norm > 1.e-7) {
      Eigen::Vector3d r = w_corrected / w_norm;
      K << 0.0, -r[2],  r[1],
          r[2],   0.0, -r[0],
         -r[1],  r[0],   0.0;

      double r_ang = w_norm * dt;
      R += std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
    }

    // Acceleration
    Eigen::Vector3d a0 = q._transformVector(a - b.accel);
    a0 += g;

    // Orientation
    Eigen::Quaterniond q_update(R);
    q *= q_update;

    // Position
    p += v*dt + 0.5*a0*dt*dt;

    // Velocity
    v += a0*dt;
  }

  Eigen::Affine3f affine3f() const {
    Eigen::Affine3d transform = Eigen::Affine3d::Identity();

    transform.rotate(q);
    transform.translation() = p;

    return transform.cast<float>();
  }

};

typedef boost::circular_buffer<State> States;
