/*
 Copyright (c) 2024 Oriol Martínez @fetty31

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef __FASTLIMO_STATE_HPP__
#define __FASTLIMO_STATE_HPP__

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <vector> 
#include "IKFoM/use-ikfom.hpp"
// #include "IKFoM/common_lib.hpp"
// #include "IKFoM/esekfom.hpp"

#include <sophus/so3.hpp>
#include <cassert>

namespace fast_limo {

struct State{

  struct IMUbias;

  Eigen::Vector3d p;      // position in global/world frame
  Eigen::Quaterniond q;   // orientation in global/world frame
  Eigen::Vector3d v;      // linear velocity
  Eigen::Vector3d g;      // gravity vector
  
  Eigen::Vector3d w;      // angular velocity (IMU input)
  Eigen::Vector3d a;      // linear acceleration (IMU input)

  // Offsets
  Eigen::Affine3d IL_T;

  double time;

  struct IMUbias {
    Eigen::Vector3d gyro;
    Eigen::Vector3d accel;
  } b;                    // IMU bias in base_link/body frame 

  State() : time(0.0) { 
    q.setIdentity();
    p.setZero();
    v.setZero();
    w.setZero();
    a.setZero();
    g.setZero();
    IL_T.setIdentity();

    b.gyro.setZero();  
    b.accel.setZero();
  }


  State(state_ikfom& s) {
    // Odom
    q = s.rot.unit_quaternion();
    p = s.pos;
    v = s.vel;

    // Gravity
    g = s.grav;

    // IMU bias
    b.gyro = s.bg;
    b.accel = s.ba;

    // Offset LiDAR-IMU
    IL_T.setIdentity();
    IL_T.rotate(s.offset_R_L_I.unit_quaternion());
    IL_T.translate(s.offset_T_L_I);
  }

  State(state_ikfom& s, double t) : State(s) { 
    time = t;
  }
  
  State(state_ikfom& s,
        double t,
        Eigen::Vector3d a,
        Eigen::Vector3d w) : State(s, t) {
    a = a;
    w = w;
  }

  void update(double t){

      // R ⊞ (w - bw - nw)*dt
      // v ⊞ (R*(a - ba - na) + g)*dt
      // p ⊞ (v*dt + 1/2*(R*(a - ba - na) + g)*dt*dt)

      // Time between IMU samples
      double dt = t - time;

      Eigen::Vector3d a0 = q._transformVector(a - b.accel);

      p += v*dt + 0.5*(a0 + g) * dt*dt;
      q *= Sophus::SO3d::exp( (w - b.gyro)*dt ).unit_quaternion(); 
      v += (a0 + g) * dt;
  }

  Eigen::Affine3d get_RT(){
    // Transformation matrix
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.rotate(q);
    T.translate(p);

    return T;
  }

  Eigen::Affine3d get_RT_inv(){
    Eigen::Affine3d T = get_RT().inverse();

    return T;
  }

  Eigen::Affine3d get_extr_RT(){
    // Transform matrix
    return IL_T;
  }

  Eigen::Affine3d get_extr_RT_inv(){
    return IL_T.inverse();
  }
};

typedef std::vector<fast_limo::State> States;

} // fast_limo

#endif