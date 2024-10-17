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

// System
#include <ctime>
#include <iomanip>
#include <algorithm>
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

// Boost
#include <boost/format.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/adjacent_filtered.hpp>
#include <boost/range/adaptor/filtered.hpp>


#include "fast_limo/Objects/State.hpp"
#include "fast_limo/Utils/Config.hpp"
#include "fast_limo/Utils/PCL.hpp"
#include "fast_limo/Objects/Meas.hpp"
#include "fast_limo/Utils/Algorithms.hpp"

#include "ikd-Tree/ikd_Tree/ikd_Tree.h"

// #include "IKFoM/use-ikfom.hpp"
// #include "IKFoM/use-ikfom.hpp"
// #include "IKFoM/common_lib.hpp"
#include "IKFoM/esekfom.hpp"

using namespace fast_limo;

namespace fast_limo {

enum class SensorType { OUSTER, VELODYNE, HESAI, LIVOX, UNKNOWN };

class Localizer {

	// VARIABLES

	public:
    // pointcloud to match in Xt2 (last_state) frame
		pcl::PointCloud<PointType>::ConstPtr pc2match_;
		KD_TREE<MapPoint>::Ptr map_;

		// Iterated Kalman Filter on Manifolds (FASTLIOv2)
		esekfom::esekf iKFoM_;
		std::mutex mtx_ikfom;

		State state, last_state;
		Extrinsics extr;
		SensorType sensor;
		IMUmeas last_imu;

		// Config struct
		Config config;


		// PCL Filters
		pcl::CropBox<PointType> crop_filter;
		pcl::VoxelGrid<PointType> voxel_filter;

		// Time related var.
		double scan_stamp;
		double prev_scan_stamp;
		double scan_dt;

		double imu_stamp;
		double prev_imu_stamp;
		double imu_dt;
		double first_imu_stamp;
		double last_propagate_time_;
		double imu_calib_time_;

		// Gravity
		double gravity_;

		// Flags
		bool imu_calibrated_;

		// OpenMP max threads
		int num_threads_;

		// IMU buffer
		boost::circular_buffer<IMUmeas> imu_buffer;

		// Propagated states buffer
		boost::circular_buffer<State> propagated_buffer;
		std::mutex mtx_prop; // mutex for avoiding multiple thread access to the buffer
		std::condition_variable cv_prop_stamp;


		// IMU axis matrix 
		Eigen::Matrix3d imu_accel_sm_;
		/*(if your IMU doesn't comply with axis system ISO-8855, 
		this matrix is meant to map its current orientation with respect to the standard axis system)
			Y-pitch
			^   
			|  
			| 
			|
	  Z-yaw o-----------> X-roll
		*/

		// Debugging
		unsigned char calibrating = 0;

			// Threads
		std::thread debug_thread;

			// Buffers
		boost::circular_buffer<double> cpu_times;
		boost::circular_buffer<double> imu_rates;
		boost::circular_buffer<double> lidar_rates;
		boost::circular_buffer<double> cpu_percents;

			// CPU specs
		std::string cpu_type;
		clock_t lastCPU, lastSysCPU, lastUserCPU;
		int numProcessors;

			// Other
		chrono::duration<double> elapsed_time;  // pointcloud callback elapsed time
		int deskew_size;                        // steps taken to deskew (FoV discretization)
		int propagated_size;                    // number of integrated states

	// FUNCTIONS

	public:
  // Point Clouds
		PointCloudT::ConstPtr original_scan; // in base_link/body frame
		PointCloudT::ConstPtr deskewed_scan; // in global/world frame
		PointCloudT::Ptr final_raw_scan;     // in global/world frame
		PointCloudT::Ptr final_scan;         // in global/world frame

		Localizer();
		void init(Config& cfg);

		// Callbacks 
		void updateIMU(IMUmeas& raw_imu);
		void updatePointCloud(pcl::PointCloud<PointType>::Ptr& raw_pc, double time_stamp);

		std::vector<double> getPoseCovariance();
		std::vector<double> getTwistCovariance();
		State getWorldState();  // get state in body/base_link frame
		// State getBodyState();   // get state in LiDAR frame

		double get_propagate_time();

		// Status info
		bool is_calibrated();

		// Config
		void set_sensor_type(uint8_t type);

		// Backpropagation
		void propagateImu(const IMUmeas& imu);

	private:
		void init_iKFoM();
		void init_iKFoM_state();

		IMUmeas imu2baselink(IMUmeas& imu);

		PointCloudT::Ptr deskewPointCloud(PointCloudT::Ptr& pc, double& start_time);

		// States integrateImu(double start_time, double end_time);
                  
		bool isInRange(PointType& p);

		void getCPUinfo();
		void debugVerbose();

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