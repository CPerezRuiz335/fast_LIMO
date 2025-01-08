#pragma once

#include <execution>
#include <numeric>
#include <algorithm>

#include <Eigen/Dense>

#include "Octree.hpp"

#include "use-ikfom.hpp"
#include "Config.hpp"
#include "State.hpp"
#include "Profiler.hpp"
#include "PCL.hpp"
#include "Map.hpp"


Eigen::Matrix<double, 24, 1> get_f(state_ikfom& s, const input_ikfom& in) {
  Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
  vect3 omega = in.gyro - s.bg;
  vect3 a_inertial = s.rot * (in.acc - s.ba);
  
  for (int i = 0; i < 3; i++ ) {
    res(i)      = s.vel[i];
    res(i + 3)  = omega[i]; 
    res(i + 12) = a_inertial[i] + s.grav[i]; 
  }

  return res;
}


Eigen::Matrix<double, 24, 24> df_dx(state_ikfom& s, const input_ikfom& in) {
  Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();
  cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
  
  vect3 acc = in.acc - s.ba;
  cov.template block<3, 3>(12, 3)  = -s.rot.toRotationMatrix()*MTK::hat(acc);
  cov.template block<3, 3>(12, 18) = -s.rot.toRotationMatrix();
  // Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
  // Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
  // s.S2_Mx(grav_matrix, vec, 21);
  cov.template block<3, 3>(12, 21) =  Eigen::Matrix3d::Identity();
  cov.template block<3, 3>(3, 15)  = -Eigen::Matrix3d::Identity();

  return cov;
}


Eigen::Matrix<double, 24, 12> df_dw(state_ikfom& s, const input_ikfom& in) {
  Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
  cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix();
  cov.template block<3, 3>(3, 0)  = -Eigen::Matrix3d::Identity();
  cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();
  cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();
  
  return cov;
}

void init_IKFoM(esekfom::esekf<state_ikfom, 12, input_ikfom>& instance) {
  Config& cfg = Config::getInstance(); 

	// Initialize IKFoM
	instance.init_dyn_runtime_share(get_f,
		                              df_dx,
		                              df_dw,
		                              cfg.ikfom.max_iters,
		                              cfg.ikfom.tolerance);
}


void setIKFoM_state(esekfom::esekf<state_ikfom, 12, input_ikfom>& ikfom,
                    const State& state) {
  
  Config& cfg = Config::getInstance(); 

  state_ikfom init_state = ikfom.get_x();
  init_state.rot = state.q;
  init_state.pos = state.p;
  init_state.grav = -state.g;
  init_state.bg = state.b.gyro;
  init_state.ba = state.b.accel;

  init_state.offset_R_L_I = SO3(cfg.sensors.extrinsics.lidar2baselink_T.linear());
  init_state.offset_T_L_I = cfg.sensors.extrinsics.lidar2baselink_T.translation();
  ikfom.change_x(init_state); // set initial state

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = ikfom.get_P();
  init_P.setIdentity();
  init_P *= 1e-3f; 
  
  ikfom.change_P(init_P);
}


void predict(esekfom::esekf<state_ikfom, 12, input_ikfom>& ikfom,
             const Imu& imu,
             double& dt) {

PROFC_NODE("predict")

  Config& cfg = Config::getInstance();

	input_ikfom in;
	in.acc = imu.lin_accel;
	in.gyro = imu.ang_vel;

	Eigen::Matrix<double, 12, 12> Q = Eigen::Matrix<double, 12, 12>::Identity();
	Q.block<3, 3>(0, 0) = cfg.ikfom.covariance.gyro * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(3, 3) = cfg.ikfom.covariance.accel * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(6, 6) = cfg.ikfom.covariance.bias_gyro * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(9, 9) = cfg.ikfom.covariance.bias_accel * Eigen::Matrix3d::Identity();

  ikfom.predict(dt, Q, in);
}


void update(esekfom::esekf<state_ikfom, 12, input_ikfom>& ikfom,
            PointCloudT::Ptr& cloud,
            thuni::Octree& map) {

PROFC_NODE("update")

  Config& cfg = Config::getInstance();

  Matches first_matches;

  int query_iters = cfg.ikfom.query_iters;

  auto h_model = [&](state_ikfom& updated_state,
							       esekfom::dyn_share_datastruct<double>& ekfom_data) {

    if (map.size() == 0) {
      ekfom_data.h_x = Eigen::MatrixXd::Zero(0, 12);
      ekfom_data.h.resize(0);	
      return;
    }

    int N = cloud->size();

    std::vector<bool> chosen(N, false);
    Matches matches(N);

    State S(updated_state);

    if (query_iters-- > 0) {
      std::vector<int> indices(N);
      std::iota(indices.begin(), indices.end(), 0);
      
      std::for_each(
        std::execution::par_unseq,
        indices.begin(),
        indices.end(),
        [&](int i) {
          PointT pt = cloud->points[i];
          Eigen::Vector3f p(pt.x, pt.y, pt.z);
          Eigen::Vector3f g = S.affine3f() * S.I2L * p; // global coords 

          std::vector<pcl::PointXYZ> neighbors;
          std::vector<float> pointSearchSqDis;
          map.knnNeighbors(pcl::PointXYZ(g(0), g(1), g(2)),
                           cfg.ikfom.plane.points,
                           neighbors,
                           pointSearchSqDis);
          
          if (neighbors.size() < cfg.ikfom.plane.points 
              or pointSearchSqDis.back() > cfg.ikfom.plane.max_sqrt_dist)
                return;
          
          Eigen::Vector4f p_abcd = Eigen::Vector4f::Zero();
          if (not estimate_plane(p_abcd, neighbors, cfg.ikfom.plane.plane_threshold))
            return;
          
          chosen[i] = true;
          matches[i] = Match(p, g, p_abcd);
        }
      );

      for (int i = 0; i < N; i++) {
        if (chosen[i])
          first_matches.push_back(matches[i]);        
      }

    } else {
      for (auto& match : first_matches) {
        match.global = S.affine3f() * S.I2L * match.local; 
      }
      
    }

    ekfom_data.h_x = Eigen::MatrixXd::Zero(first_matches.size(), 12);
    ekfom_data.h.resize(first_matches.size());	

    std::vector<int> indices(first_matches.size());
    std::iota(indices.begin(), indices.end(), 0);

    // For each match, calculate its derivative and distance
    std::for_each (
      std::execution::par_unseq,
      indices.begin(),
      indices.end(),
      [&](int i) {
        Match match = first_matches[i];
        Eigen::Vector3f p_imu   = S.affine3f().inverse() * match.global;
        Eigen::Vector3f p_lidar = S.I2L.inverse() * p_imu;

        // Rotation matrices
        Eigen::Matrix3f R_inv = S.q.conjugate().toRotationMatrix().cast<float>();
        Eigen::Matrix3f I_R_L_inv = S.I2L.linear().transpose().cast<float>();

        // Set correct dimensions
        Eigen::Vector3f n = match.n.head(3);

        // Calculate measurement Jacobian H (:= dh/dx)
        Eigen::Vector3f C = R_inv * n;
        Eigen::Vector3f B = p_lidar.cross(I_R_L_inv * C);
        Eigen::Vector3f A = p_imu.cross(C);
        
        ekfom_data.h_x.block<1, 6>(i,0) << n(0), n(1), n(2), A(0), A(1), A(2);

        if (cfg.ikfom.estimate_extrinsics)
          ekfom_data.h_x.block<1, 6>(i,6) << B(0), B(1), B(2), C(0), C(1), C(2);

        ekfom_data.h(i) = -match.dist2plane();
      }
    );
  };

  ikfom.update_iterated_dyn_runtime_share(cfg.ikfom.lidar_noise, h_model);

}

