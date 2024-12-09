#pragma once

#include <Eigen/Dense>

#include "IKFoM_toolkit/esekfom/esekfom.hpp"

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
  
  for (int i = 0; i < 3; i++ ){
    res(i)      = s.vel[i];
    res(i + 3)  =  omega[i]; 
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


void setIKFoM_state(esekfom::esekf<state_ikfom, 12, input_ikfom>& instance,
                    const State& state) {
  
  Config& cfg = Config::getInstance(); 

  state_ikfom init_state = instance.get_x();
  init_state.rot = state.q.cast<double> ();
  init_state.pos = state.p.cast<double> ();
  init_state.grav = S2(state.g);
  init_state.bg = this->state.b.gyro.cast<double>();
  init_state.ba = this->state.b.accel.cast<double>();

  // set up offsets (LiDAR -> BaseLink transform == LiDAR pose w.r.t. BaseLink)
  init_state.offset_R_L_I = /*MTK::*/SO3(this->extr.lidar2baselink.R.cast<double>());
  init_state.offset_T_L_I = this->extr.lidar2baselink.t.cast<double>();
  this->_iKFoM.change_x(init_state); // set initial state

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = this->_iKFoM.get_P();
  init_P.setIdentity();
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = 0.00001; 
  
  this->_iKFoM.change_P(init_P);

}

} // namespace limoncello