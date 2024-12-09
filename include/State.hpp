#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "IKFoM_toolkit/esekfom/esekfom.hpp"

#include "PCL.hpp"
#include "Config.hpp"


typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2; 
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

MTK_BUILD_MANIFOLD(state_ikfom,
  ((vect3, pos))
  ((SO3, rot))
  ((SO3, offset_R_L_I))
  ((vect3, offset_T_L_I))
  ((vect3, vel))
  ((vect3, bg))
  ((vect3, ba))
  ((S2, grav))
);

MTK_BUILD_MANIFOLD(input_ikfom,
  ((vect3, acc))
  ((vect3, gyro))
);

MTK_BUILD_MANIFOLD(process_noise_ikfom,
  ((vect3, ng))
  ((vect3, na))
  ((vect3, nbg))
  ((vect3, nba))
);


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
    q.setIndentity();
    p.setZero();
    v.setZero();
    w.setZero();
    a.setZero();
    g.setZero();

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


  State(const state_ikfom& s, const Imu& imu) {
    
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
    I2L.translation() = s.offset_R_L_I.cast<float>();
  }


  void update(const double& t) {
    // R ⊞ (w - bw - nw)*dt
    // v ⊞ (R*(a - ba - na) + g)*dt
    // p ⊞ (v*dt + 1/2*(R*(a - ba - na) + g)*dt*dt)

    double dt = t - stamp;
    if (dt < 0)
      dt = 1. / Config::getInstance().imu.hz;

    // Exp orientation
    Eigen::Vector3f w_corrected = w - b.gyro;
    float w_norm = w_corrected.norm();
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();

    if (w_norm > 1.e-7){
      Eigen::Vector3f r = w_corrected / w_norm;
      igen::Matrix3f K << 0.0, -r[2],  r[1],
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
};

}