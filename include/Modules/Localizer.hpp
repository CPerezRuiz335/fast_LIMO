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

#ifndef __FASTLIMO_LOCALIZER_HPP__
#define __FASTLIMO_LOCALIZER_HPP__

#ifndef HAS_CUPID
#include <cpuid.h>
#endif

#define FAST_LIMO_v "1.0.0"

// System
#include <ctime>
#include <iomanip>
#include <future>
#include <ios>
#include <sys/times.h>
#include <sys/vtimes.h>

#include <iostream>
#include <sstream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <string>

#include <climits>
#include <cmath>

#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <algorithm>
#include <execution>

#include <utility>

#include <boost/format.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/adjacent_filtered.hpp>
#include <boost/range/adaptor/filtered.hpp>

// #include "fast_limo/Common.hpp"
#include "Objects/State.hpp"
#include "Utils/Config.hpp"
#include "Utils/PCL.hpp"

#include "Utils/Algorithms.hpp"
#include "Objects/Imu.hpp"

#include "octree2/Octree.h"


namespace fast_limo {

	struct Match {
		Eigen::Vector4f local;
		Eigen::Vector4f global;
		Eigen::Vector4f n; // normal vector

		Match() = default;
		Match(Eigen::Vector4f& local_,
			    Eigen::Vector4f& global_,
					Eigen::Vector4f& n_) : local(local_), 
																 global(global_),
																 n(n_) {};

		pcl::PointXYZINormal toPointXYZINormal() {
			pcl::PointXYZINormal p;
			p.x = global.x();
			p.y = global.y();
			p.z = global.z();
			p.intensity = 0.;
			p.normal_x = n(0);
			p.normal_y = n(1);
			p.normal_z = n(2);

			return p;
		}

		float dist2plane() {
			return n(0)*global(0) + n(1)*global(1) + n(2)*global(2) + n(3); 
		}
	};

	enum class SensorType {
		OUSTER,
		VELODYNE,
		HESAI, 
		LIVOX,
		UNKNOWN 
	};

	class Localizer {

		private:
			thuni::Octree map;

			PointCloudT::Ptr pc2match_; // pointcloud to match in Xt2 (last_state) frame
			esekfom::esekf<state_ikfom, 12, input_ikfom> iKFoM_;
			std::mutex mtx_ikfom_;

			State state_, last_state_;
			IMUmeas last_imu_;

			// PCL Filters
			pcl::CropBox<PointType> crop_filter_;
			pcl::VoxelGrid<PointType> voxel_filter_;

			// Point Clouds
			PointCloudT::ConstPtr original_scan_; // in base_link/body frame
			PointCloudT::ConstPtr deskew_scan_; // in global/world frame
			PointCloudT::Ptr final_raw_scan_;     // in global/world frame
			PointCloudT::Ptr final_scan_;         // in global/world frame
			MatchPointCloud::Ptr matches_; // in global/world

			// Time related var.
			double scan_stamp_;
			double prev_scan_stamp_;

			double imu_stamp_;
			double prev_imu_stamp_;
			double first_imu_stamp_;

			// Gravity
			double gravity_;

			// Flags
			bool imu_calibrated_;

			// IMU buffer
			boost::circular_buffer<IMUmeas> imu_buffer_;

			// Propagated states buffer
			boost::circular_buffer<State> propagated_buffer_;
			std::mutex mtx_prop_; // mutex for avoiding multiple thread access to the buffer
			std::condition_variable cv_prop_stamp_;

			// Debugging
			unsigned char calibrating_ = 0;

			// Threads
			std::thread debug_thread_;

				// Buffers
			boost::circular_buffer<double> cpu_times_;
			boost::circular_buffer<double> imu_rates_;
			boost::circular_buffer<double> lidar_rates_;
			boost::circular_buffer<double> cpu_percents_;

				// CPU specs
			std::string cpu_type_;
			clock_t lastCPU_, lastSysCPU_, lastUserCPU_;
			int numProcessors_;

				// Other
			chrono::duration<double> elapsed_time_;  // pointcloud callback elapsed time
			int deskew_size_;                        // steps taken to deskew (FoV discretization)
			int propagated_size_;                    // number of integrated states

			// Save matches
			std::vector<Match> clean_matches;

		// FUNCTIONS

		public:
			Localizer();
			void init();

			// Callbacks 
			void updateIMU(IMUmeas& raw_imu);
			void updatePointCloud(PointCloudT::Ptr& raw_pc, double time_stamp);

			// Get output
			PointCloudT::Ptr get_pointcloud();
			PointCloudT::Ptr get_finalraw_pointcloud();
			PointCloudT::Ptr get_pc2match_pointcloud();
			MatchPointCloud::Ptr get_matches_pointcloud();
			PointCloudT::ConstPtr get_orig_pointcloud();
			PointCloudT::ConstPtr get_deskewed_pointcloud();


			// Matches& get_matches();/

			State getWorldState();  // get state in body/base_link frame
			State getBodyState();   // get state in LiDAR frame

			std::vector<double> getPoseCovariance(); // get eKF covariances
			std::vector<double> getTwistCovariance();// get eKF covariances
			
			// Status info
			bool is_calibrated();

			// Backpropagation
			void propagateImu(const IMUmeas& imu);

		private:
			void init_iKFoM();
			void init_iKFoM_state();

			PointCloudT::Ptr deskewPointCloud(PointCloudT::Ptr& pc, double& start_time);

			States integrateImu(double start_time, double end_time);

			bool propagatedFromTimeRange(double start_time, double end_time,
										boost::circular_buffer<State>::reverse_iterator& begin_prop_it,
										boost::circular_buffer<State>::reverse_iterator& end_prop_it);

			void getCPUinfo();
			void debugVerbose();

			void h_share_model(state_ikfom &updated_state,
												 esekfom::dyn_share_datastruct<double> &ekfom_data);

			Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in, const double& dt);
			Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in, const double& dt);
			Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in, const double& dt);

			static Eigen::Matrix<double, 24, 1> get_f_wrapper(state_ikfom& s,
																												const input_ikfom& in,
																												const double& dt) {
					return Localizer::getInstance().get_f(s, in, dt);
			}

			static Eigen::Matrix<double, 24, 23> df_dx_wrapper(state_ikfom& s,
																												const input_ikfom& in,
																												const double& dt) {
					return Localizer::getInstance().df_dx(s, in, dt);
			}

			static Eigen::Matrix<double, 24, 12> df_dw_wrapper(state_ikfom& s,
																												const input_ikfom& in,
																												const double& dt) {
					return Localizer::getInstance().df_dw(s, in, dt);
			}

			static void h_share_model_wrapper(state_ikfom& updated_state,
																				esekfom::dyn_share_datastruct<double>& ekfom_data) {
					Localizer::getInstance().h_share_model(updated_state, ekfom_data);
			}


			// SINGLETON 
			public:
				static Localizer& getInstance(){
					static Localizer* loc = new Localizer();
					return *loc;
				}

			private:
				// Disable copy/move functionality
				Localizer(const Localizer&) = delete;
				Localizer& operator=(const Localizer&) = delete;
				Localizer(Localizer&&) = delete;
				Localizer& operator=(Localizer&&) = delete;

	};

}

#endif