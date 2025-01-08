#include <iostream>
#include <vector>
#include <iomanip>
#include <Eigen/Dense>

#include <manif/manif.h>
#include <manif/SE3.h>
#include "manif/SO3.h"
#include "manif/Bundle.h"
#include "manif/Rn.h"


using namespace Eigen;


using BundleT = manif::Bundle<double,
    manif::SE3,
    manif::R3,
    manif::R3,
    manif::R3,
    manif::R3,
    manif::SO3,
    manif::R3
>;

int main(int argc, char** argv) {
  Eigen::Quaterniond q(0., 0., 0., 1);

  std::cout << std::setprecision(0);

  manif::SO3 initial_rot_extrinsics(q);
  manif::SE3d initial_pos;

  initial_pos.setIdentity();
  // initial_rot_extrinsics.setIdentity();

  Eigen::Vector3d zero_vec = Eigen::Vector3d::Zero();
  manif::R3d zero = manif::R3d(zero_vec);


  BundleT X = BundleT(initial_pos,
            manif::R3d(zero_vec), 
            zero,
            manif::R3d(zero_vec), 

            zero, 
            manif::SO3d(q),
            zero);


  BundleT Y = X;


  BundleT::Tangent t = X - Y;
  std::cout << t.coeffs() << std::endl;
  Eigen::Matrix<double, 6, 1> tan;
  tan.setOnes();
  t.element<0>().coeffs() = tan;
  std::cout << t*0.1 << std::endl;
  
  BundleT::Tangent a = BundleT::Tangent::Zero();

  
  Eigen::Matrix<double, 24, 24> J_x, J_u, df_dx, P, G_f;
  P.setIdentity();
  df_dx.setIdentity();

  Eigen::Matrix<double, 24,  1> u;
  u.setOnes();
  
  Eigen::Matrix<double, 24, 12> df_dw, G_w;
  df_dw.setZero();
  df_dw.template block<3, 3>(12, 3) = Eigen::Matrix3d::Identity();
  df_dw.template block<3, 3>(3, 0)  = -Eigen::Matrix3d::Identity();
  df_dw.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();
  df_dw.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();

  Eigen::Matrix<double, 12, 12> Q;
  Q.setIdentity();

  X.plus(BundleT::Tangent(u), J_x, J_u);

  G_f = J_x + J_u*df_dx;
  G_w = J_u*df_dw;  // 24x24 * 24x12 = 24x12

  P = G_f * P * G_f.transpose() + G_w * Q * G_w.transpose();

  std::cout << "J_u:\n" << J_u << std::endl;
  std::cout << "J_x:\n" << J_x << std::endl;
  std::cout << "DoF SE3: " << manif::SE3d::DoF << std::endl;
  std::cout << "Dim SE3: " << manif::SE3d::Dim << std::endl;
  std::cout << "Budnle size: " << X.BundleSize << std::endl;
  
  manif::SO3d::Jacobian J_vout_m, J_vout_v;


  initial_rot_extrinsics.act(Eigen::Vector3d(1, 1, 1), J_vout_m);
  std::cout << initial_rot_extrinsics.rotation() * zero_vec << std::endl;
  std::cout << J_vout_m << std::endl;
  std::cout << J_vout_v << std::endl;

  X.element<4>() = manif::R3d(Eigen::Vector3d(1, 3, 5));
  std::cout << "Changes r3: " << X.element<4>().coeffs() + Eigen::Vector3d(0,0,0) << std::endl;

  manif::skew(zero_vec);



  return 0;
}
