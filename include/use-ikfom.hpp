#pragma once

#include <Eigen/Dense>

#include "IKFoM_toolkit/esekfom/esekfom.hpp"

#include "octree2/Octree.h"

#include "Config.hpp"
#include "State.hpp"


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


namespace limoncello {

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


Eigen::Matrix<double, 24, 23> df_dx(state_ikfom& s, const input_ikfom& in) {
  Eigen::Matrix<double, 24, 23> cov = Eigen::Matrix<double, 24, 23>::Zero();
  cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
  
  vect3 acc = in.acc - s.ba;
  cov.template block<3, 3>(12, 3)  = -s.rot.toRotationMatrix()*MTK::hat(acc);
  cov.template block<3, 3>(12, 18) = -s.rot.toRotationMatrix();
  Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
  Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
  s.S2_Mx(grav_matrix, vec, 21);
  cov.template block<3, 2>(12, 21) =  grav_matrix;
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
  init_state.grav = S2(-state.g);
  init_state.bg = state.b.gyro;
  init_state.ba = state.b.accel;

  init_state.offset_R_L_I = SO3(cfg.extrinsics.lidar2imu_T.rotation().cast<double>());
  init_state.offset_T_L_I = cfg.extrinsics.lidar2imu_T.translation().cast<double>();
  ikfom.change_x(init_state); // set initial state

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = ikfom.get_P();
  init_P.setIdentity();
  init_P *= 1e-3f; 
  
  ikfom.change_P(init_P);
}


void predict(esekfom::esekf<state_ikfom, 12, input_ikfom>& ikfom,
             const Imu& imu,
             const double& dt) {

  Config& cfg = Config::getInstance();

	input_ikfom in;
	in.acc = imu.lin_accel;
	in.gyro = imu.ang_vel;

	Eigen::Matrix<double, 12, 12> Q = Eigen::Matrix<double, 12, 12>::Identity();
	Q.block<3, 3>(0, 0) = config.ikfom.cov_gyro * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(3, 3) = config.ikfom.cov_acc * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(6, 6) = config.ikfom.cov_bias_gyro * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(9, 9) = config.ikfom.cov_bias_acc * Eigen::Matrix3d::Identity();

  ikfom.predict(dt, Q, in);
}


void update(esekfom::esekf<state_ikfom, 12, input_ikfom>& ikfom,
            PointCloudT::Ptr& cloud,
            thuni::Octree& map) {

  Config& config = Config::getInstance();

  Matches first_matches;

  auto h_model = [&](state_ikfom& updated_state,
							       esekfom::dyn_share_datastruct<double>& ekfom_data) {
    
    if (map.size() == 0) {
      ekfom_data.h_x = Eigen::MatrixXd::Zero(0, 12);
      ekfom_data.h.resize(0);	
      return;
    }

    int N = cloud->size();

    std::vector<bool> chosen(N, false);
    std::vector<Match> matches(N);

    State S(updated_state);

    if (clean_matches.empty()) {
      std::vector<int> indices(N);
      std::iota(indices.begin(), indices.end(), 0);
      
      std::for_each(
        std::execution::par_unseq,
        indices.begin(),
        indices.end(),
        [&](int i) {
          PointT pt = cloud->points[i];
          Eigen::Vector3f p(pt.x, pt.y, pt.z);
          Eigen::Vector3f g = (S.affine3f() * S.I2L) * p; // global coords 

          std::vector<pcl::PointXYZ> neighbors;
          std::vector<float> pointSearchSqDis;
          map.knnNeighbors(pcl::PointXYZ(g(0), g(1), g(2)),
                           config.ikfom.mapping.num_match_points,
                           neighbors,
                           pointSearchSqDis);
          
          if (near_points.size() < config.ikfom.mapping.num_match_points 
              or pointSearchSqDis.back() > config.ikfom.mapping.further_point_dist)
                return;
          
          Eigen::Vector4f p_abcd = Eigen::Vector4f::Zero();
          if (not estimate_plane(p_abcd, near_points, config.ikfom.mapping.plane_threshold))
            return;
          
          chosen[i] = true;
          matches[i] = Match(bl4_point, g, p_abcd);
        }
      );

      for (int i = 0; i < N; i++) {
        if (chosen[i])
          clean_matches.push_back(matches[i]);        
      }

    } else {
      for (auto& match : clean_matches) {
        match.global = (S.affine3f() * S.I2L) * match.local; 
      }
      
    }

    ekfom_data.h_x = Eigen::MatrixXd::Zero(clean_matches.size(), 12);
    ekfom_data.h.resize(clean_matches.size());	

    std::vector<int> indices(clean_matches.size());
    std::iota(indices.begin(), indices.end(), 0);

    // For each match, calculate its derivative and distance
    std::for_each (
      std::execution::par,
      indices.begin(),
      indices.end(),
      [&](int i) {
        Match match = clean_matches[i];
        Eigen::Vector3f p_imu   = S.affine3f().inverse() * match.global;
        Eigen::Vector3f p_lidar = S.I2L.inverse() * p_imu;

        // Rotation matrices
        Eigen::Matrix3f R_inv = S.rot.conjugate().toRotationMatrix().cast<float>();
        Eigen::Matrix3f I_R_L_inv = S.I2L.rotation().transpose().cast<float>();

        // Set correct dimensions
        Eigen::Vector3f n = match.n.head(3);

        // Calculate measurement Jacobian H (:= dh/dx)
        Eigen::Vector3f C = R_inv * n;
        Eigen::Vector3f B = p_lidar.cross(I_R_L_inv * C);
        Eigen::Vector3f A = p_imu.cross(C);
        
        ekfom_data.h_x.block<1, 6>(i,0) << n(0), n(1), n(2), A(0), A(1), A(2);

        if (config.ikfom.estimate_extrinsics)
          ekfom_data.h_x.block<1, 6>(i,6) << B(0), B(1), B(2), C(0), C(1), C(2);

        ekfom_data.h(i) = -match.dist2plane();
      }
    )
  }

  ikfom.update_iterated_dyn_runtime_share(cfg.lidar.noise, h_model);

}

} // namespace limoncello