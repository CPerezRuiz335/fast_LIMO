#pragma once

#include <boost/circular_buffer.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "use-ikfom.hpp"
#include "Config.hpp"


namespace limoncello {

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
    g = Eigen::Vector3d(0., 0., cfg.gravity);

    // Extrinsics 
    Config& cfg = Config::getInstance();
    float roll  = cfg.extrinsics.roll;
    float pitch = cfg.extrinsics.pitch;
    float yaw   = cfg.extrinsics.yaw;

    Eigen::Matrix3f R = Eigen::AngleAxisf(roll  * M_PI/180., Eigen::Vector3f::UnitX()) *
                        Eigen::AngleAxisf(pitch * M_PI/180., Eigen::Vector3f::UnitY()) *
                        Eigen::AngleAxisf(yaw * M_PI/180., Eigen::Vector3f::UnitZ());

    I2L.rotation() = R;
    I2L.translation() = cfg.extrinsics.t;

    b.gyro.setZero();
    b.accel.setZero();
  }


  State(const state_ikfom& s, const Imu& imu = Imu()) {
    
    a = imu.a;
    w = imu.w;
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
    I2L.rotation() = s.offset_R_L_I.cast<float>();
    I2L.translation() = s.offset_T_L_I.cast<float>();
  }


  void update(const double& t) {
    // R ⊞ (w - bw - nw)*dt
    // v ⊞ (R*(a - ba - na) + g)*dt
    // p ⊞ (v*dt + 1/2*(R*(a - ba - na) + g)*dt*dt)

    double dt = t - stamp;
    assert(dt >= 0); // TODO

    // Exp orientation
    Eigen::Vector3f w_corrected = w - b.gyro;
    float w_norm = w_corrected.norm();
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();

    if (w_norm > 1.e-7) {
      Eigen::Vector3f r = w_corrected / w_norm;
      Eigen::Matrix3f K << 0.0, -r[2],  r[1],
                         r[2],   0.0, -r[0],
                        -r[1],  r[0],   0.0;

      float r_ang = w_norm * dt;
      R += std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
    }

    // Acceleration
    Eigen::Vector3f a0 = q._transformVector(a - b.accel);
    a0 += g;

    // Orientation
    Eigen::Quaternionf q_update(R);
    q *= q_update;

    // Position
    p += v*dt + 0.5*a0*dt*dt;

    // Velocity
    v += a0*dt;
  }

  Eigen::Affine3f affine3f() const {
    Affine3d transform = Affine3d::Identity();

    transform.rotate(q);
    transform.translate(p);

    return transform.cast<float>();
  }

};

typedef boost::circular_buffer<State> States;

}