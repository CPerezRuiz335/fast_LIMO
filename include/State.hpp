#ifndef STATE_H
#define STATE_H  

#include <ros/ros.h>
#include <Eigen/Dense>

#include "Utils/use-ikfom.hpp"
#include "Imu.hpp"

struct State {

 public:

  Eigen::Vector3f p;      // position in global/world frame
  Eigen::Quaternionf q;   // orientation in global/world frame
  Eigen::Vector3f v;      // linear velocity
  Eigen::Vector3f g;      // gravity vector
  
  Eigen::Vector3f w;      // angular velocity (IMU input)
  Eigen::Vector3f a;      // linear acceleration (IMU input)

  // Extrinsics
  Eigen::Quaternionf qI2L; // quaternion IMU 2 LiDAR
  Eigen::Vector3f pI2L;    // position IMU 2 LiDAR

  double stamp;

  struct IMUbias {
    Eigen::Vector3f gyro;
    Eigen::Vector3f accel;
  } b;                    // IMU bias in base_link/body frame 

 private:

  static esekfom::esekf<state_ikfom, 12, input_ikfom> iKFoM_;

 public:

  State() : stamp(0.0) { 
    q.setIdentity();
    p.setZero();
    v.setZero();
    w.setZero();
    a.setZero();
    g.setZero();
    pI2L.setZero();
    qI2L.setIdentity();

    b.gyro.setZero();
    b.accel.setZero();
  }

  static State from_iKFoM() {
    const state_ikfom s = iKFoM_.get_x();
    State out;

    out.q = s.rot.cast<float>();
    out.p = s.pos.cast<float>();
    out.v = s.vel.cast<float>();

    out.g = s.grav.get_vect().cast<float>();

    out.b.gyro = s.bg.cast<float>();
    out.b.accel = s.ba.cast<float>();

    out.qI2L = s.offset_R_L_I.cast<float>();
    out.pI2L = s.offset_T_L_I.cast<float>();

    return out;
  }

  static void predict(const IMU& imu) {

  }

  // void update(double t) {
	// 	ROS_ASSERT(t >= stamp);

  //   double dt = t - stamp;
  //   float theta, norm;
  //   Eigen::Vector3f w, u;
    
  //   w = this->w - b.gyro;
  //   norm = w.norm();
  //   u = w / norm;
  //   theta = norm * dt;

  //   Eigen::Quaternionf q_update;
    
  //   if (norm > 1.e-7) {
  //     Eigen::Vector3f v = u * std::sin(theta);

  //     q_update.w() = std::cos(theta);
  //     q_update.x() = v.x();
  //     q_update.y() = v.y();
  //     q_update.z() = v.z();
  //   } else {
  //     q_update.setIdentity();
  //   }

  //   Eigen::Vector3f acc = this->q * (a - b.accel) + this->g;

  //   this->p += this->v*dt + 0.5*acc*dt*dt;
  //   this->q *= q_update;
  //   this->v += acc*dt;
  // }
  
  Eigen::Matrix4f affine() const {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block(0, 0, 3, 3) = q.toRotationMatrix();
    T.block(0, 3, 3, 1) = p;

    return T;
  }

  Eigen::Matrix4f affine_extr() const {
    // Transform matrix
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block(0, 0, 3, 3) = qI2L.toRotationMatrix();
    T.block(0, 3, 3, 1) = pI2L;

    return T;
  }
};

typedef std::vector<State> States;

#endif // STATE