/*
 *  Copyright (c) 2019--2023, The University of Hong Kong
 *  All rights reserved.
 *
 *  Author: Dongjiao HE <hdj65822@connect.hku.hk>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ESEKFOM_EKF_HPP
#define ESEKFOM_EKF_HPP

#include <iostream>
#include <vector>
#include <cstdlib>

#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "../mtk/types/vect.hpp"
#include "../mtk/types/SOn.hpp"
#include "../mtk/types/S2.hpp"
#include "../mtk/startIdx.hpp"
#include "../mtk/build_manifold.hpp"
#include "util.hpp"


#include <manif/manif.h>
#include <manif/SO3.h>
#include <manif/Bundle.h>
#include <manif/Rn.h>

//#define USE_sparse


namespace esekfom {

  using Matrix24d = Eigen::Matrix<double, 24, 24> ;
  using Matrix24x12d = Eigen::Matrix<double, 24, 12> ;
  using Matrix12d = Eigen::Matrix<double, 12, 12> ;

  using BundleT = manif::Bundle<double,
      manif::R3,  // position
      manif::SO3, // rotation
      manif::SO3, // imu2lidar rotation
      manif::R3,  // imu2lidar translation
      manif::R3,  // velocity
      manif::R3,  // angular bias
      manif::R3,  // acceleartion bias
      manif::R3   // gravity
  >;

  using Tangent = typename BundleT::Tangent; 


using namespace Eigen;

//used for iterated error state EKF update
//for the aim to calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise covariance (R) at the same time, by only one function.
//applied for measurement as a manifold.
template<typename S, typename M, int measurement_noise_dof = M::DOF>
struct share_datastruct
{
	bool valid;
	bool converge;
	M z;
	Eigen::Matrix<typename S::scalar, M::DOF, measurement_noise_dof> h_v;
	Eigen::Matrix<typename S::scalar, M::DOF, S::DOF> h_x;
	Eigen::Matrix<typename S::scalar, measurement_noise_dof, measurement_noise_dof> R;
};

//used for iterated error state EKF update
//for the aim to calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise covariance (R) at the same time, by only one function.
//applied for measurement as an Eigen matrix whose dimension is changing
template<typename T>
struct dyn_share_datastruct
{
	bool valid;
	bool converge;
	Eigen::Matrix<T, Eigen::Dynamic, 1> z;
	Eigen::Matrix<T, Eigen::Dynamic, 1> h;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_v;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_x;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> R;
};

//used for iterated error state EKF update
//for the aim to calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise covariance (R) at the same time, by only one function.
//applied for measurement as a dynamic manifold whose dimension or type is changing
template<typename T>
struct dyn_runtime_share_datastruct
{
	bool valid;
	bool converge;
	//Z z;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_v;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_x;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> R;
};

template<typename state, int process_noise_dof, typename input = state, typename measurement=state, int measurement_noise_dof=0>
class esekf{

	typedef esekf self;
	enum{
		n = state::DOF, m = state::DIM, l = measurement::DOF
	};

public:
	
	typedef typename state::scalar scalar_type;
	typedef Matrix<scalar_type, n, n> cov;
	typedef Matrix<scalar_type, m, n> cov_;
	typedef SparseMatrix<scalar_type> spMt;
	typedef Matrix<scalar_type, n, 1> vectorized_state;
	typedef Matrix<scalar_type, m, 1> flatted_state;
	typedef flatted_state processModel(state &, const input &);
	typedef Eigen::Matrix<scalar_type, m, n> processMatrix1(state &, const input &);
	typedef Eigen::Matrix<scalar_type, m, process_noise_dof> processMatrix2(state &, const input &);
	typedef Eigen::Matrix<scalar_type, process_noise_dof, process_noise_dof> processnoisecovariance;
	typedef measurement measurementModel(state &, bool &);
	typedef measurement measurementModel_share(state &, share_datastruct<state, measurement, measurement_noise_dof> &);
	typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, 1> measurementModel_dyn(state &, bool &);
	typedef void measurementModel_dyn_share(state &,  dyn_share_datastruct<scalar_type> &);
	typedef Eigen::Matrix<scalar_type ,l, n> measurementMatrix1(state &, bool&);
	typedef Eigen::Matrix<scalar_type , Eigen::Dynamic, n> measurementMatrix1_dyn(state &, bool&);
	typedef Eigen::Matrix<scalar_type ,l, measurement_noise_dof> measurementMatrix2(state &, bool&);
	typedef Eigen::Matrix<scalar_type ,Eigen::Dynamic, Eigen::Dynamic> measurementMatrix2_dyn(state &, bool&);
	typedef Eigen::Matrix<scalar_type, measurement_noise_dof, measurement_noise_dof> measurementnoisecovariance;
	typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> measurementnoisecovariance_dyn;

	esekf(const state &x = state(), const cov  &P = cov::Identity()): x_(x), P_(P) { };

	//receive system-specific models and their differentions
	//for measurement as a dynamic manifold whose dimension  or type is changing.
	//calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) 
	//and the noise covariance (R) at the same time, by only one function (h_dyn_share_in).
	//for any scenarios where it is needed
	void init_dyn_runtime_share(processModel f_in,
								processMatrix1 f_x_in,
								processMatrix2 f_w_in,
								int maximum_iteration,
								double tolerance)
	{
		f = f_in;
		f_x = f_x_in;
		f_w = f_w_in;

		maximum_iter = maximum_iteration;
		for(int i=0; i<n; i++)
		{
			limit[i] = tolerance;
		}

		x_.build_S2_state();
		x_.build_SO3_state();
		x_.build_vect_state();
	}

	// iterated error state EKF propogation
	void predict(double &dt, processnoisecovariance &Q, const input &i_in){
		flatted_state f_ = f(x_, i_in);
		cov_ f_x_ = f_x(x_, i_in);
		Matrix<scalar_type, m, process_noise_dof> f_w_ = f_w(x_, i_in);
		

// Manif
		Matrix24d Gx, Gf; // Adjoint_X(u)^{-1}, J_r(u)  Sola-18, [https://arxiv.org/abs/1812.01537]
    X = X.plus(Tangent(f_ * dt), Gx, Gf);
// Manif

		Matrix24d    Fx = Gx + Gf * f_x_ * dt; // He-2021, [https://arxiv.org/abs/2102.03804] Eq. (26)
    Matrix24x12d Fw = Gf * f_w_ * dt;      // He-2021, [https://arxiv.org/abs/2102.03804] Eq. (27)

		P_ = Fx * P_ * Fx.transpose() + Fw * Q * Fw.transpose(); 

		x_.pos = X.element<0>().coeffs();
		x_.rot = X.element<1>().quat();
		x_.offset_R_L_I = X.element<2>().quat();
		x_.offset_T_L_I = X.element<3>().coeffs();
		x_.vel = X.element<4>().coeffs();
		x_.bg = X.element<5>().coeffs();
		x_.ba = X.element<6>().coeffs();
		x_.grav = X.element<7>().coeffs();

	}
	

	// Modified version used in Fast-LIO2
	//iterated error state EKF update modified for one specific system.
	template<typename measurementModel_dyn_runtime_share>
	void update_iterated_dyn_runtime_share(double R, 
																			   measurementModel_dyn_runtime_share h) {

		dyn_share_datastruct<scalar_type> dyn_share;
		dyn_share.valid = true;
		dyn_share.converge = true;
		state x_propagated = x_;
		int t = 0;

		BundleT X_ = X;
		cov P__ = P_;

		Matrix<scalar_type, n, 1> K_h;
		Matrix<scalar_type, n, n> K_x;

		Tangent dx_new = Tangent::Zero();
		for(int i=-1; i<maximum_iter; i++)
		{
			dyn_share.valid = true;
			h(x_, dyn_share);


			Eigen::Matrix<scalar_type, Eigen::Dynamic, 12> h_x_ = dyn_share.h_x;

			Matrix24d J, J_;
      dx_new = X_.minus(X, J, J_); // Xu-2021, [https://arxiv.org/abs/2107.06829] Eq. (11)
			vectorized_state dx__;
			x_.boxminus(dx__, x_propagated);
			// dx_new.coeffs() = dx__;

      P__ = J.inverse() * P_ * J.inverse().transpose();

			Eigen::Matrix<scalar_type, 12, 12> HTH;


			cov P_temp = (P__/R).inverse();
			HTH = h_x_.transpose() * h_x_;
			P_temp. template block<12, 12>(0, 0) += HTH;

			cov P_inv = P_temp.inverse();
			K_h = P_inv. template block<n, 12>(0, 0) * h_x_.transpose() * dyn_share.h;
			K_x.setZero();
			K_x. template block<n, 12>(0, 0) = P_inv. template block<n, 12>(0, 0) * HTH;
			Tangent dx_ = K_h + (K_x - Matrix<scalar_type, n, n>::Identity()) * J.inverse() * dx_new; 

			X_ = X_.plus(dx_);
			x_.boxplus(dx_.coeffs());

			dyn_share.converge = true;
			for(int i = 0; i < n ; i++)
			{
				if(std::fabs(dx_[i]) > limit[i])
				{
					dyn_share.converge = false;
					break;
				}
			}
			if(dyn_share.converge) t++;

			if(!t && i == maximum_iter - 2)
			{
				dyn_share.converge = true;
			}

			if(t > 1 || i == maximum_iter - 1)
			{
				X = X_;
				P_ = (Matrix24d::Identity() - K_x) * P__;
				x_.pos = X.element<0>().coeffs();
				x_.rot = X.element<1>().quat();
				x_.offset_R_L_I = X.element<2>().quat();
				x_.offset_T_L_I = X.element<3>().coeffs();
				x_.vel = X.element<4>().coeffs();
				x_.bg = X.element<5>().coeffs();
				x_.ba = X.element<6>().coeffs();
				x_.grav = X.element<7>().coeffs();
				return;
			}
		}
	}

	void change_x(state &input_state)
	{
		x_ = input_state;

		if((!x_.vect_state.size())&&(!x_.SO3_state.size())&&(!x_.S2_state.size()))
		{
			x_.build_S2_state();
			x_.build_SO3_state();
			x_.build_vect_state();
		}

		X = BundleT(manif::R3d(x_.pos),
                manif::SO3d(x_.rot.normalized()),
                manif::SO3d(x_.offset_R_L_I.normalized()),
                manif::R3d(x_.offset_T_L_I),
                manif::R3d(x_.vel),
                manif::R3d(x_.bg),
                manif::R3d(x_.ba),
                manif::R3d(x_.grav)); // NOTE

	}

	void change_P(cov &input_cov)
	{
		P_ = input_cov;
	}

	const state& get_x() const {
		return x_;
	}

	const cov& get_P() const {
		return P_;
	}

private:
	state x_;
	measurement m_;
	cov P_;
	spMt l_;
	spMt f_x_1;
	spMt f_x_2;
	cov F_x1 = cov::Identity();
	cov F_x2 = cov::Identity();
	cov L_ = cov::Identity();

	processModel *f;
	processMatrix1 *f_x;
	processMatrix2 *f_w;

	measurementModel *h;
	measurementMatrix1 *h_x;
	measurementMatrix2 *h_v;

	measurementModel_dyn *h_dyn;
	measurementMatrix1_dyn *h_x_dyn;
	measurementMatrix2_dyn *h_v_dyn;

	measurementModel_share *h_share;
	measurementModel_dyn_share *h_dyn_share;

	int maximum_iter = 0;
	scalar_type limit[n];
	
	template <typename T>
    T check_safe_update( T _temp_vec )
    {
        T temp_vec = _temp_vec;
        if ( std::isnan( temp_vec(0, 0) ) )
        {
            temp_vec.setZero();
            return temp_vec;
        }
        double angular_dis = temp_vec.block( 0, 0, 3, 1 ).norm() * 57.3;
        double pos_dis = temp_vec.block( 3, 0, 3, 1 ).norm();
        if ( angular_dis >= 20 || pos_dis > 1 )
        {
            printf( "Angular dis = %.2f, pos dis = %.2f\r\n", angular_dis, pos_dis );
            temp_vec.setZero();
        }
        return temp_vec;
    }

	BundleT X;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace esekfom

#endif //  ESEKFOM_EKF_HPP
