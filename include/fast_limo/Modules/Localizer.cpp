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

#include "fast_limo/Modules/Localizer.hpp"

// class fast_limo::Localizer
	// public

Localizer::Localizer() : scan_stamp(0.0), 
                          prev_scan_stamp(0.0),
                          scan_dt(0.1),
                          deskew_size(0),
                          propagated_size(0),
                          numProcessors(0),
                          imu_stamp(0.0),
                          prev_imu_stamp(0.0),
                          imu_dt(0.005),
                          first_imu_stamp(0.0),
                          last_propagate_time_(-1.0),
                          imu_calib_time_(3.0),
                          gravity_(9.81),
                          imu_calibrated_(false) { 

    original_scan  = PointCloudT::ConstPtr (new PointCloudT);
    deskewed_scan  = PointCloudT::ConstPtr (new PointCloudT);
    pc2match_      = PointCloudT::ConstPtr (new PointCloudT);
    final_raw_scan = PointCloudT::Ptr (new PointCloudT);
    final_scan     = PointCloudT::Ptr (new PointCloudT);

    map_ = KD_TREE<MapPoint>::Ptr (new KD_TREE<MapPoint>(0.3, 0.6, 0.2));

    // Init cfg values
    config.ikfom.mapping.NUM_MATCH_POINTS = 5;
    config.ikfom.mapping.MAX_NUM_PC2MATCH = 1.e+4;
    config.ikfom.mapping.MAX_DIST_PLANE   = 2.0;
    config.ikfom.mapping.PLANE_THRESHOLD  = 5.e-2;
    config.ikfom.mapping.local_mapping    = false;

    config.ikfom.mapping.ikdtree.cube_size = 300.0;
    config.ikfom.mapping.ikdtree.rm_range  = 200.0;

  //   map->Set_delete_criterion_param(config.ikfom.mapping.ikdtree.delete_param);
  //   map->Set_balance_criterion_param(config.ikfom.mapping.ikdtree.balance_param);
  //   map->set_downsample_param(config.ikfom.mapping.ikdtree.voxel_size);

}


void Localizer::init(Config& cfg){

  using namespace Eigen;

  // Save config
  config = cfg;

  // Set num of threads
  num_threads_ = omp_get_max_threads();
  if(num_threads_ > config.num_threads) num_threads_ = config.num_threads;

  // Initialize Iterated Kalman Filter on Manifolds
  init_iKFoM();

  // Set buffer capacity
  imu_buffer.set_capacity(2000);
  propagated_buffer.set_capacity(2000);

  // PCL filters setup
  crop_filter.setNegative(true);
  crop_filter.setMin(Vector4f(config.filters.cropBoxMin[0], 
                                     config.filters.cropBoxMin[1],
                                     config.filters.cropBoxMin[2], 1.0));

  crop_filter.setMax(Vector4f(config.filters.cropBoxMax[0],
                                     config.filters.cropBoxMax[1],
                                     config.filters.cropBoxMax[2], 1.0));

  voxel_filter.setLeafSize(config.filters.leafSize[0],
                           config.filters.leafSize[1],
                           config.filters.leafSize[2]);

  // LiDAR sensor type
  set_sensor_type(config.sensor_type); 

  // IMU intrinsics
  imu_accel_sm_ = Map<Matrix3d>(config.intrinsics.imu_sm.data(), 3, 3);
  state.b.accel = Map<Vector3d>(config.intrinsics.accel_bias.data(), 3);
  state.b.gyro  = Map<Vector3d>(config.intrinsics.gyro_bias.data(), 3);

  // Extrinsics
  extr.imu2baselink_T.translation() = Map<Vector3d>(config.extrinsics.imu2baselink_t.data(), 3);
  extr.imu2baselink_T.rotate(Map<Matrix3d>(config.extrinsics.imu2baselink_R.data(), 3, 3));

  extr.lidar2baselink_T.translation() = Map<Vector3d>(config.extrinsics.lidar2baselink_t.data(), 3);
  extr.lidar2baselink_T.rotate(Map<Matrix3d>(config.extrinsics.lidar2baselink_R.data(), 3, 3));

  // Avoid unnecessary warnings from PCL
  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  // Initial calibration
  if( not (config.gravity_align || config.calibrate_accel || config.calibrate_gyro) ){ // no need for automatic calibration
    imu_calibrated_ = true;
    init_iKFoM_state();
  }

  // Calibration time
  imu_calib_time_ = config.imu_calib_time;

  // CPU info
  getCPUinfo();

  if(config.verbose){
    // set up buffer capacities
    imu_rates.set_capacity(1000);
    lidar_rates.set_capacity(1000);
    cpu_times.set_capacity(1000);
    cpu_percents.set_capacity(1000);
  }
}


bool Localizer::is_calibrated(){
  return imu_calibrated_;
}

void Localizer::set_sensor_type(uint8_t type){
  if(type < 5)
    sensor = static_cast<fast_limo::SensorType>(type);
  else 
    sensor = fast_limo::SensorType::UNKNOWN;
}

State Localizer::getWorldState(){

  if(not is_calibrated())
    return State();

  state_ikfom s = iKFoM_.get_x();
  State out = State(s);

  out.w    = last_imu.ang_vel;                      // set last IMU meas
  out.a    = last_imu.lin_accel;                    // set last IMU meas
  out.time = imu_stamp;                             // set current time stamp 

  // out.v = out.q.toRotationMatrix().transpose() * out.v;   // local velocity vector

  return out;
}

std::vector<double> Localizer::getPoseCovariance() {
    if(not this->is_calibrated())
        return std::vector<double>(36, 0);

    Eigen::Matrix<double, 24, 24> P = iKFoM_.get_P();
    Eigen::Matrix<double, 6, 6> P_pose;
    P_pose.block<3, 3>(0, 0) = P.block<3, 3>(3, 3);
    P_pose.block<3, 3>(0, 3) = P.block<3, 3>(3, 0);
    P_pose.block<3, 3>(3, 0) = P.block<3, 3>(0, 3);
    P_pose.block<3, 3>(3, 3) = P.block<3, 3>(0, 0);

    std::vector<double> cov(P_pose.size());
    Eigen::Map<Eigen::MatrixXd>(cov.data(), P_pose.rows(), P_pose.cols()) = P_pose;

    return cov;
}

std::vector<double> Localizer::getTwistCovariance(){
    if(not is_calibrated())
        return std::vector<double>(36, 0);

    Eigen::Matrix<double, 24, 24> P = iKFoM_.get_P();
    Eigen::Matrix<double, 6, 6> P_odom = Eigen::Matrix<double, 6, 6>::Zero();
    P_odom.block<3, 3>(0, 0) = P.block<3, 3>(6, 6);
    P_odom.block<3, 3>(3, 3) = config.ikfom.cov_gyro * Eigen::Matrix<double, 3, 3>::Identity();

    std::vector<double> cov(P_odom.size());
    Eigen::Map<Eigen::MatrixXd>(cov.data(), P_odom.rows(), P_odom.cols()) = P_odom;

    return cov;
}


double Localizer::get_propagate_time(){
  return last_propagate_time_;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////           Principal callbacks/threads        /////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Localizer::updatePointCloud(PointCloudT::Ptr& raw_pc, double time_stamp){

  auto start_time = chrono::system_clock::now();

  if(raw_pc->points.size() < 1){
    std::cout << "FAST_LIMO::Raw PointCloud is empty!\n";
    return;
  }

  if(!imu_calibrated_)
    return;

  if(imu_buffer.empty()){
    std::cout << "FAST_LIMO::IMU buffer is empty!\n";
    return;
  }

  // Remove NaNs
    std::vector<int> idx;
    raw_pc->is_dense = false;
    pcl::removeNaNFromPointCloud(*raw_pc, *raw_pc, idx);

  // Crop Box Filter (1 m^2)
    if(config.filters.crop_active){
      crop_filter.setInputCloud(raw_pc);
      crop_filter.filter(*raw_pc);
    }

  // Distance & Time Rate filters
    static float min_dist = static_cast<float>(config.filters.min_dist);
    static int rate_value = config.filters.rate_value;
    std::function<bool(boost::range::index_value<PointType&, long>)> filter_f;
    
    if(config.filters.dist_active && config.filters.rate_active){
      filter_f = [this](boost::range::index_value<PointType&, long> p)
        { return (Eigen::Vector3d(p.value().x, p.value().y, p.value().z).norm() > min_dist)
              && (p.index()%rate_value == 0) && isInRange(p.value()); };
    }
    else if(config.filters.dist_active){
      filter_f = [this](boost::range::index_value<PointType&, long> p)
        { return (Eigen::Vector3d(p.value().x, p.value().y, p.value().z).norm() > min_dist) &&
              isInRange(p.value()); };
    }
    else if(config.filters.rate_active){
      filter_f = [this](boost::range::index_value<PointType&, long> p)
        { return (p.index()%rate_value == 0) && isInRange(p.value()); };
    }else{
      filter_f = [this](boost::range::index_value<PointType&, long> p)
        { return isInRange(p.value()); };
    }
    auto filtered_pc = raw_pc->points 
          | boost::adaptors::indexed()
          | boost::adaptors::filtered(filter_f);

    pcl::PointCloud<PointType>::Ptr input_pc (boost::make_shared<pcl::PointCloud<PointType>>());
    for (auto it = filtered_pc.begin(); it != filtered_pc.end(); it++) {
      input_pc->points.push_back(it->value());
    }

  if(config.debug) // debug only
    original_scan = boost::make_shared<pcl::PointCloud<PointType>>(*input_pc); // LiDAR frame

  // Motion compensation
    pcl::PointCloud<PointType>::Ptr deskewed_Xt2_pc_ (boost::make_shared<pcl::PointCloud<PointType>>());
    deskewed_Xt2_pc_ = deskewPointCloud(input_pc, time_stamp);

  // Voxel Grid Filter
    if (config.filters.voxel_active) { 
      pcl::PointCloud<PointType>::Ptr current_scan_
        (boost::make_shared<pcl::PointCloud<PointType>>(*deskewed_Xt2_pc_));
      voxel_filter.setInputCloud(current_scan_);
      voxel_filter.filter(*current_scan_);
      pc2match_ = current_scan_;
    } else {
      pc2match_ = deskewed_Xt2_pc_;
    }

  if (pc2match_->points.size() > 1) {
    // iKFoM observation stage
    mtx_ikfom.lock();

    pcl::PointCloud<MapPoint>::Ptr pc2matchMap(new pcl::PointCloud<MapPoint>);
    for (const auto& p_ : pc2match_->points) {
      MapPoint p;
      p.x = p_.x;
      p.y = p_.y;
      p.z = p_.z;
      pc2matchMap->points.push_back(p);
    }

    iKFoM_.update_iterated_dyn_share_modified(
      0.001 /*LiDAR noise*/,
      pc2matchMap,
      map_,
      config.ikfom.MAX_NUM_ITERS,
      config.ikfom.estimate_extrinsics
    );

    // Get output state from iKFoM
    state_ikfom s = iKFoM_.get_x();
    fast_limo::State corrected_state = fast_limo::State(s);

    // Update current state estimate
    corrected_state.b.gyro  = state.b.gyro;
    corrected_state.b.accel = state.b.accel;
    state      = corrected_state;
    state.w    = last_imu.ang_vel;
    state.a    = last_imu.lin_accel;


    // Get estimated offset
    extr.lidar2baselink_T = Eigen::Affine3d::Identity();

    // Transform deskewed pc 
      // Get deskewed scan to add to map
    Eigen::Affine3d T;  
    T.setIdentity();
    State tmp = propagated_buffer.back();
    T.translate(tmp.p);
    T.rotate(tmp.q);

    pcl::PointCloud<PointType>::Ptr mapped_scan (boost::make_shared<pcl::PointCloud<PointType>>());
    pcl::transformPointCloud (*pc2match_, *mapped_scan, T);
    pcl::transformPointCloud (*pc2match_, *final_scan, T); // mapped_scan = final_scan (for now)

    if(config.debug) // save final scan without voxel grid
      pcl::transformPointCloud (*deskewed_Xt2_pc_, *final_raw_scan, corrected_state.get_RT());


    MapPoints tmp_f;
    if (map_->size() > 0) {
      for (const auto& p : mapped_scan->points) {
        MapPoint p_;
        p_.x = p.x;
        p_.y = p.y;
        p_.z = p.z;
        tmp_f.push_back(p_);
      }

      map_->Add_Points(tmp_f, true);
    } else {
      for (const auto& p : pc2match_->points) {
        MapPoint p_;
        p_.x = p.x;
        p_.y = p.y;
        p_.z = p.z;
        tmp_f.push_back(p_);
      }

      map_->Build(tmp_f);
    }

    mtx_ikfom.unlock();
  } else {
    std::cout << "-------------- FAST_LIMO::NULL ITERATION --------------\n";
  }

  if(config.verbose){
    // fill stats
    if(prev_scan_stamp > 0.0) lidar_rates.push_front( 1. / (scan_stamp - prev_scan_stamp) );
    if(calibrating > 0) cpu_times.push_front(elapsed_time.count());
    else cpu_times.push_front(0.0);
    
    if(calibrating < UCHAR_MAX) calibrating++;

    // debug thread
    debug_thread = std::thread( &Localizer::debugVerbose, this );
    debug_thread.detach();
  }

  prev_scan_stamp = scan_stamp;
}


void Localizer::updateIMU(IMUmeas& raw_imu){

  imu_stamp = raw_imu.stamp;
  IMUmeas imu = imu2baselink(raw_imu);

  if(first_imu_stamp <= 0.0)
    first_imu_stamp = imu.stamp;

  if(config.verbose) 
      imu_rates.push_front( 1./imu.dt );

  // IMU calibration procedure - do only while the robot is in stand still!
  if (not imu_calibrated_) {

    static int num_samples = 0;
    static Eigen::Vector3d gyro_avg (0., 0., 0.);
    static Eigen::Vector3d accel_avg (0., 0., 0.);
    static bool print = true;

    if ((imu.stamp - first_imu_stamp) < imu_calib_time_) {

      num_samples++;

      gyro_avg[0] += imu.ang_vel[0];
      gyro_avg[1] += imu.ang_vel[1];
      gyro_avg[2] += imu.ang_vel[2];

      accel_avg[0] += imu.lin_accel[0];
      accel_avg[1] += imu.lin_accel[1];
      accel_avg[2] += imu.lin_accel[2];

      if (print) {
        std::cout << std::endl << " Calibrating IMU for " << imu_calib_time_ << " seconds... \n";
        std::cout.flush();
        print = false;
      }

    } else {

      gyro_avg /= num_samples;
      accel_avg /= num_samples;

      Eigen::Vector3d grav_vec (0., 0., gravity_);

      state.q = imu.q;

      if (config.gravity_align) {

        std::cout << " Accel mean: " << "[ " << accel_avg[0] << ", " << accel_avg[1] << ", " << accel_avg[2] << " ]\n";

        // Estimate gravity vector - Only approximate if biases have not been pre-calibrated
        grav_vec = (accel_avg - state.b.accel).normalized() * abs(gravity_);
        Eigen::Quaterniond grav_q = Eigen::Quaterniond::FromTwoVectors(grav_vec, Eigen::Vector3d(0., 0., gravity_));
        
        std::cout << " Gravity mean: " << "[ " << grav_vec[0] << ", " << grav_vec[1] << ", " << grav_vec[2] << " ]\n";

        // set gravity aligned orientation
        state.q = grav_q;

      }

      if (config.calibrate_accel) {

        // subtract gravity from avg accel to get bias
        state.b.accel = accel_avg - grav_vec;

        std::cout << " Accel biases [xyz]: " << to_string_with_precision(state.b.accel[0], 8) << ", "
                          << to_string_with_precision(state.b.accel[1], 8) << ", "
                          << to_string_with_precision(state.b.accel[2], 8) << std::endl;
      }

      if (config.calibrate_gyro) {

        state.b.gyro = gyro_avg;

        std::cout << " Gyro biases  [xyz]: " << to_string_with_precision(state.b.gyro[0], 8) << ", "
                          << to_string_with_precision(state.b.gyro[1], 8) << ", "
                          << to_string_with_precision(state.b.gyro[2], 8) << std::endl;
      }

      state.q.normalize();

      // Set initial KF state
      init_iKFoM_state();

      // Set calib flag
      imu_calibrated_ = true;

      // Initial attitude
      auto euler = state.q.toRotationMatrix().eulerAngles(2, 1, 0);
      double yaw = euler[0] * (180.0/M_PI);
      double pitch = euler[1] * (180.0/M_PI);
      double roll = euler[2] * (180.0/M_PI);

      // use alternate representation if the yaw is smaller
      if (abs(remainder(yaw + 180.0, 360.0)) < abs(yaw)) {
        yaw   = remainder(yaw + 180.0,   360.0);
        pitch = remainder(180.0 - pitch, 360.0);
        roll  = remainder(roll + 180.0,  360.0);
      }
      std::cout << " Estimated initial attitude:" << std::endl;
      std::cout << "   Roll  [deg]: " << to_string_with_precision(roll, 4) << std::endl;
      std::cout << "   Pitch [deg]: " << to_string_with_precision(pitch, 4) << std::endl;
      std::cout << "   Yaw   [deg]: " << to_string_with_precision(yaw, 4) << std::endl;
      std::cout << std::endl;

    }

  } else {
    // Apply the calibrated bias to the new IMU measurements
    Eigen::Vector3d lin_accel_corrected = (imu_accel_sm_ * imu.lin_accel) - state.b.accel;
    Eigen::Vector3d ang_vel_corrected = imu.ang_vel - state.b.gyro;

    imu.lin_accel = lin_accel_corrected;
    imu.ang_vel   = ang_vel_corrected;

    last_imu = imu;

    // Store calibrated IMU measurements into imu buffer for manual integration later.
    imu_buffer.push_front(imu);

    // iKFoM propagate state
    propagateImu(imu);
    cv_prop_stamp.notify_one(); // Notify PointCloud thread that propagated IMU data exists for this time

  }

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////          KF propagation model        /////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Localizer::propagateImu(const IMUmeas& imu){
  input_ikfom in;
  in.acc = imu.lin_accel.cast<double>();
  in.gyro = imu.ang_vel.cast<double>();

  Eigen::Matrix<double, 12, 12> Q = Eigen::Matrix<double, 12, 12>::Identity();
  Q.block<3, 3>(0, 0) = config.ikfom.cov_gyro * Eigen::Matrix<double, 3, 3>::Identity();
  Q.block<3, 3>(3, 3) = config.ikfom.cov_acc * Eigen::Matrix<double, 3, 3>::Identity();
  Q.block<3, 3>(6, 6) = config.ikfom.cov_bias_gyro * Eigen::Matrix<double, 3, 3>::Identity();
  Q.block<3, 3>(9, 9) = config.ikfom.cov_bias_acc * Eigen::Matrix<double, 3, 3>::Identity();

  // Propagate IMU measurement
  double dt = imu.dt;
  mtx_ikfom.lock();
  iKFoM_.predict(dt, Q, in);
  mtx_ikfom.unlock();

  // Save propagated state for motion compensation
  mtx_prop.lock();
  state_ikfom s = iKFoM_.get_x();
  propagated_buffer.push_back( fast_limo::State(s, 
                            imu.stamp, imu.lin_accel, imu.ang_vel)
                    );
  mtx_prop.unlock();

  last_propagate_time_ = imu.stamp;
}


// private

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////          Aux. functions        ///////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Localizer::init_iKFoM() {
  // Initialize IKFoM
  return;
}

void Localizer::init_iKFoM_state() {
  state_ikfom init_state = iKFoM_.get_x();
  init_state.rot = Sophus::SO3d(state.q.cast<double> ());
  init_state.pos = state.p.cast<double> ();
  init_state.grav = Eigen::Vector3d(0., 0., -gravity_);
  init_state.bg = state.b.gyro.cast<double>();
  init_state.ba = state.b.accel.cast<double>();

  // set up offsets (LiDAR -> BaseLink transform == LiDAR pose w.r.t. BaseLink)
  init_state.offset_R_L_I = Sophus::SO3d(extr.lidar2baselink_T.rotation());
  init_state.offset_T_L_I = extr.lidar2baselink_T.translation();
  iKFoM_.change_x(init_state); // set initial state

  Eigen::Matrix<double, 24, 24> init_P = iKFoM_.get_P();
  init_P.setIdentity();
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = 0.00001; 
  
  iKFoM_.change_P(init_P);
}

IMUmeas Localizer::imu2baselink(IMUmeas& imu){

  IMUmeas imu_baselink;

  double dt = imu.stamp - prev_imu_stamp;
  
  if ( (dt == 0.) || (dt > 0.1) ) { dt = 1.0/400.0; }

  // // Transform angular velocity (will be the same on a rigid body, so just rotate to baselink frame)
  // Eigen::Vector3d ang_vel_cg = extr.imu2baselink_T.rotation() * imu.ang_vel;

  // static Eigen::Vector3d ang_vel_cg_prev = ang_vel_cg;

  // // Transform linear acceleration (need to account for component due to translational difference)
  // Eigen::Vector3d lin_accel_cg = extr.imu2baselink_T.rotation() * imu.lin_accel;

  // lin_accel_cg = lin_accel_cg
  //         + ((ang_vel_cg - ang_vel_cg_prev) / dt).cross(-extr.imu2baselink_T.translation())
  //         + ang_vel_cg.cross(ang_vel_cg.cross(-extr.imu2baselink_T.translation()));

  // ang_vel_cg_prev = ang_vel_cg;

  imu_baselink.ang_vel   = imu.ang_vel;
  imu_baselink.lin_accel = imu.lin_accel;
  imu_baselink.dt        = dt;
  imu_baselink.stamp     = imu.stamp;

  // Eigen::Quaterniond q(extr.imu2baselink_T.rotation());
  // q.normalize();
  imu_baselink.q = imu.q;

  prev_imu_stamp = imu.stamp;

  return imu_baselink;

}

PointCloudT::Ptr Localizer::deskewPointCloud(PointCloudT::Ptr& pc, double& start_time){

  if(pc->points.size() < 1) 
    return boost::make_shared<PointCloudT>();

  // individual point timestamps should be relative to this time
  double sweep_ref_time = start_time;
  bool end_of_sweep = config.end_of_sweep;

  // sort points by timestamp
  std::function<bool(const PointType&, const PointType&)> point_time_cmp;
  std::function<double(PointType&)> extract_point_time;

  if (sensor == fast_limo::SensorType::OUSTER) {

    point_time_cmp = [&end_of_sweep](const PointType& p1, const PointType& p2)
    {   if (end_of_sweep) return p1.t > p2.t; 
      else return p1.t < p2.t; };
    extract_point_time = [&sweep_ref_time, &end_of_sweep](PointType& pt)
    {   if (end_of_sweep) return sweep_ref_time - pt.t * 1e-9f; 
      else return sweep_ref_time + pt.t * 1e-9f; };

  } else if (sensor == fast_limo::SensorType::VELODYNE) {
    
    point_time_cmp = [&end_of_sweep](const PointType& p1, const PointType& p2)
    {   if (end_of_sweep) return p1.time > p2.time; 
      else return p1.time < p2.time; };
    extract_point_time = [&sweep_ref_time, &end_of_sweep](PointType& pt)
    {   if (end_of_sweep) return sweep_ref_time - pt.time; 
      else return sweep_ref_time + pt.time; };

  } else if (sensor == fast_limo::SensorType::HESAI) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
    { return p1.timestamp < p2.timestamp; };
    extract_point_time = [](PointType& pt)
    { return pt.timestamp; };

  } else if (sensor == fast_limo::SensorType::LIVOX) {
    
    point_time_cmp = [](const PointType& p1, const PointType& p2)
    { return p1.timestamp < p2.timestamp; };
    extract_point_time = [](PointType& pt)
    { return pt.timestamp * 1e-9f; };
  } else {
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "FAST_LIMO::FATAL ERROR: LiDAR sensor type unknown or not specified!\n";
    std::cout << "-------------------------------------------------------------------\n";
    return boost::make_shared<pcl::PointCloud<PointType>>();
  }

  // copy points into deskewed_scan_ in order of timestamp
  pcl::PointCloud<PointType>::Ptr deskewed_scan_ (boost::make_shared<pcl::PointCloud<PointType>>());
  deskewed_scan_->points.resize(pc->points.size());
  
  std::partial_sort_copy(pc->points.begin(), pc->points.end(),
              deskewed_scan_->points.begin(), deskewed_scan_->points.end(), point_time_cmp);

  if(deskewed_scan_->points.size() < 1){
    std::cout << "FAST_LIMO::ERROR: failed to sort input pointcloud!\n";
    return boost::make_shared<pcl::PointCloud<PointType>>();
  }

  // compute offset between sweep reference time and IMU data
  double offset = 0.0;
  if (config.time_offset) {
    offset = imu_stamp - extract_point_time(deskewed_scan_->points[deskewed_scan_->points.size()-1]) - 1.e-4; // automatic sync (not precise!)
    if(offset > 0.0) offset = 0.0; // don't jump into future
  }

  // Set scan_stamp for next iteration
  double first_stamp = extract_point_time(deskewed_scan_->points.front()) + offset;
  scan_stamp = extract_point_time(deskewed_scan_->points.back()) + offset;

  // deskewed pointcloud w.r.t last known state prediction
  pcl::PointCloud<PointType>::Ptr deskewed_Xt2_scan_ (boost::make_shared<pcl::PointCloud<PointType>>());
  deskewed_Xt2_scan_->points.resize(deskewed_scan_->points.size());

  state_ikfom s = iKFoM_.get_x();
  last_state = fast_limo::State(s); // baselink/body frame

  auto it = std::lower_bound(propagated_buffer.begin(), propagated_buffer.end(), first_stamp, 
        [](const State& item, double target) { return item.time < target; } );

  int first = std::distance(propagated_buffer.begin(), it);
  first = first > 0 ? first-1 : first;

  std::cout << std::setprecision(40);

  for (int k = 0; k < deskewed_scan_->points.size(); k++) {
    
    double p_time = extract_point_time(deskewed_scan_->points[k]) + offset;
    while (it != propagated_buffer.end() and p_time >= (it+1)->time) {
      it++;
    }

    State X0 = *it;
    // X0.update(p_time);

    Eigen::Affine3d T = X0.get_RT() * X0.IL_T;

    auto pt = deskewed_scan_->points[k];
    Eigen::Vector3d pt_h(pt.x, pt.y, pt.z); 

    Eigen::Vector3d pt_transformed = T * pt_h;

    // Apply the transformation for the Xt2 frame
    Eigen::Vector3d pt2_h = last_state.IL_T.inverse() * last_state.get_RT_inv() * pt_transformed;

    // Update pt2 with the transformed coordinates
    pt.x = static_cast<float>(pt2_h.x());
    pt.y = static_cast<float>(pt2_h.y());
    pt.z = static_cast<float>(pt2_h.z());
    pt.intensity = pt.intensity;

    // Save pt2 to the deskewed Xt2 scan
    deskewed_Xt2_scan_->points[k] = pt;
  }

  // debug info
  deskew_size = deskewed_Xt2_scan_->points.size(); 
  propagated_size = 20;

  if(config.debug && deskew_size > 0) // debug only
    deskewed_scan = deskewed_scan_;

  // return deskewed_Xt2_scan_; 
  return deskewed_scan_;
}


bool Localizer::isInRange(PointType& p){
  if(not config.filters.fov_active) return true;
  return fabs(atan2(p.y, p.x)) < config.filters.fov_angle;
}


// States Localizer::integrateImu(double start_time, double end_time){

//   if (propagated_buffer.empty() || propagated_buffer.back().time < end_time) {
//     // Wait for the latest IMU data
//     std::cout << "PROPAGATE WAITING...\n";
//     std::cout << "     - buffer time: " << propagated_buffer.front().time << std::endl;
//     std::cout << "     - end scan time: " << end_time << std::endl;
//     std::unique_lock<decltype(mtx_prop)> lock(mtx_prop);
//     cv_prop_stamp.wait(lock, [this, &end_time]{ return propagated_buffer.back().time >= end_time; });
//   }
// }






// ##################### DEBUG ##########################
void Localizer::getCPUinfo(){ // CPU Specs
  char CPUBrandString[0x40];
  memset(CPUBrandString, 0, sizeof(CPUBrandString));

  cpu_type = "";

  #ifdef HAS_CPUID
  unsigned int CPUInfo[4] = {0,0,0,0};
  __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
  unsigned int nExIds = CPUInfo[0];
  for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
    __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    if (i == 0x80000002)
    memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000003)
    memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000004)
    memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
  }
  cpu_type = CPUBrandString;
  boost::trim(cpu_type);
  #endif

  FILE* file;
  struct tms timeSample;
  char line[128];

  lastCPU = times(&timeSample);
  lastSysCPU = timeSample.tms_stime;
  lastUserCPU = timeSample.tms_utime;

  file = fopen("/proc/cpuinfo", "r");
  numProcessors = 0;
  while(fgets(line, 128, file) != nullptr) {
    if (strncmp(line, "processor", 9) == 0) numProcessors++;
  }
  fclose(file);
}

void Localizer::debugVerbose(){

  // Average computation time
  double avg_comp_time =
    std::accumulate(cpu_times.begin(), cpu_times.end(), 0.0) / cpu_times.size();

  // Average sensor rates
  double avg_imu_rate =
    std::accumulate(imu_rates.begin(), imu_rates.end(), 0.0) / imu_rates.size();
  double avg_lidar_rate =
    std::accumulate(lidar_rates.begin(), lidar_rates.end(), 0.0) / lidar_rates.size();

  // RAM Usage
  double vm_usage = 0.0;
  double resident_set = 0.0;
  std::ifstream stat_stream("/proc/self/stat", std::ios_base::in); //get info from proc directory
  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string utime, stime, cutime, cstime, priority, nice;
  std::string num_threads, itrealvalue, starttime;
  unsigned long vsize;
  long rss;
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
        >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
        >> utime >> stime >> cutime >> cstime >> priority >> nice
        >> num_threads >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest
  stat_stream.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // for x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;

  // CPU Usage
  struct tms timeSample;
  clock_t now;
  double cpu_percent;
  now = times(&timeSample);
  if (now <= lastCPU || timeSample.tms_stime < lastSysCPU ||
    timeSample.tms_utime < lastUserCPU) {
    cpu_percent = -1.0;
  } else {
    cpu_percent = (timeSample.tms_stime - lastSysCPU) + (timeSample.tms_utime - lastUserCPU);
    cpu_percent /= (now - lastCPU);
    cpu_percent /= numProcessors;
    cpu_percent *= 100.;
  }
  lastCPU = now;
  lastSysCPU = timeSample.tms_stime;
  lastUserCPU = timeSample.tms_utime;
  cpu_percents.push_front(cpu_percent);
  double avg_cpu_usage =
    std::accumulate(cpu_percents.begin(), cpu_percents.end(), 0.0) / cpu_percents.size();

  // ------------------------------------- PRINT OUT -------------------------------------

  printf("\033[2J\033[1;1H");
  std::cout << std::endl
        << "+-------------------------------------------------------------------+" << std::endl;
  std::cout   << "|                        Fast LIMO  v 1.0.0                         |"
        << std::endl;
  std::cout   << "+-------------------------------------------------------------------+" << std::endl;

  std::time_t curr_time = scan_stamp;
  std::string asc_time = std::asctime(std::localtime(&curr_time)); asc_time.pop_back();
  std::cout << "| " << std::left << asc_time;
  std::cout << std::right << std::setfill(' ') << std::setw(42)
    << "Elapsed Time: " + to_string_with_precision(imu_stamp - first_imu_stamp, 2) + " seconds "
    << "|" << std::endl;

  if ( !cpu_type.empty() ) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << cpu_type + " x " + std::to_string(numProcessors)
    << "|" << std::endl;
  }

  if (sensor == fast_limo::SensorType::OUSTER) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Sensor Rates: Ouster @ " + to_string_with_precision(avg_lidar_rate, 2)
                  + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
    << "|" << std::endl;
  } else if (sensor == fast_limo::SensorType::VELODYNE) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Sensor Rates: Velodyne @ " + to_string_with_precision(avg_lidar_rate, 2)
                    + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
    << "|" << std::endl;
  } else if (sensor == fast_limo::SensorType::HESAI) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Sensor Rates: Hesai @ " + to_string_with_precision(avg_lidar_rate, 2)
                  + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
    << "|" << std::endl;
  } else if (sensor == fast_limo::SensorType::LIVOX) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Sensor Rates: Livox @ " + to_string_with_precision(avg_lidar_rate, 2)
                  + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
    << "|" << std::endl;
  } else {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Sensor Rates: Unknown LiDAR @ " + to_string_with_precision(avg_lidar_rate, 2)
                      + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
    << "|" << std::endl;
  }

  std::cout << "|===================================================================|" << std::endl;

  State final_state = getWorldState();

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Position     {W}  [xyz] :: " + to_string_with_precision(final_state.p(0), 4) + " "
                  + to_string_with_precision(final_state.p(1), 4) + " "
                  + to_string_with_precision(final_state.p(2), 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Orientation  {W} [wxyz] :: " + to_string_with_precision(final_state.q.w(), 4) + " "
                  + to_string_with_precision(final_state.q.x(), 4) + " "
                  + to_string_with_precision(final_state.q.y(), 4) + " "
                  + to_string_with_precision(final_state.q.z(), 4)
    << "|" << std::endl;

  auto euler = final_state.q.toRotationMatrix().eulerAngles(2, 1, 0);
  double yaw = euler[0] * (180.0/M_PI);
  double pitch = euler[1] * (180.0/M_PI);
  double roll = euler[2] * (180.0/M_PI);

  // use alternate representation if the yaw is smaller
  if (abs(remainder(yaw + 180.0, 360.0)) < abs(yaw)) {
    yaw   = remainder(yaw + 180.0,   360.0);
    pitch = remainder(180.0 - pitch, 360.0);
    roll  = remainder(roll + 180.0,  360.0);
  }
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "             {W} [ypr] :: " + to_string_with_precision(yaw, 4) + " "
                  + to_string_with_precision(pitch, 4) + " "
                  + to_string_with_precision(roll, 4)
    << "|" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Lin Velocity {B}  [xyz] :: " + to_string_with_precision(final_state.v(0), 4) + " "
                  + to_string_with_precision(final_state.v(1), 4) + " "
                  + to_string_with_precision(final_state.v(2), 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Ang Velocity {B}  [xyz] :: " + to_string_with_precision(final_state.w(0), 4) + " "
                  + to_string_with_precision(final_state.w(1), 4) + " "
                  + to_string_with_precision(final_state.w(2), 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Accel Bias        [xyz] :: " + to_string_with_precision(final_state.b.accel(0), 8) + " "
                  + to_string_with_precision(final_state.b.accel(1), 8) + " "
                  + to_string_with_precision(final_state.b.accel(2), 8)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Gyro Bias         [xyz] :: " + to_string_with_precision(final_state.b.gyro(0), 8) + " "
                  + to_string_with_precision(final_state.b.gyro(1), 8) + " "
                  + to_string_with_precision(final_state.b.gyro(2), 8)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Gravity Est.      [xyz] :: " + to_string_with_precision(final_state.g(0), 8) + " "
                  + to_string_with_precision(final_state.g(1), 8) + " "
                  + to_string_with_precision(final_state.g(2), 8)
    << "|" << std::endl;

  std::cout << "|                                                                   |" << std::endl;


  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "LiDAR -> BaseLink     [t] :: " + to_string_with_precision(final_state.IL_T.translation()(0), 4) + " "
                  + to_string_with_precision(final_state.IL_T.translation()(1), 4) + " "
                  + to_string_with_precision(final_state.IL_T.translation()(2), 4)
    << "|" << std::endl;
  
  euler = final_state.IL_T.rotation().eulerAngles(2, 1, 0);
  yaw = euler[0] * (180.0/M_PI);
  pitch = euler[1] * (180.0/M_PI);
  roll = euler[2] * (180.0/M_PI);

  // use alternate representation if the yaw is smaller
  if (abs(remainder(yaw + 180.0, 360.0)) < abs(yaw)) {
    yaw   = remainder(yaw + 180.0,   360.0);
    pitch = remainder(180.0 - pitch, 360.0);
    roll  = remainder(roll + 180.0,  360.0);
  }
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "                      [ypr] :: " + to_string_with_precision(yaw, 4) + " "
                  + to_string_with_precision(pitch, 4) + " "
                  + to_string_with_precision(roll, 4)
    << "|" << std::endl;

  std::cout << "|                                                                   |" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Deskewed points: " + std::to_string(deskew_size) << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Integrated states: " + std::to_string(propagated_size) << "|" << std::endl;
  std::cout << "|                                                                   |" << std::endl;

  std::cout << std::right << std::setprecision(2) << std::fixed;
  std::cout << "| Computation Time :: "
    << std::setfill(' ') << std::setw(6) << cpu_times.front()*1000. << " ms    // Avg: "
    << std::setw(6) << avg_comp_time*1000. << " / Max: "
    << std::setw(6) << *std::max_element(cpu_times.begin(), cpu_times.end())*1000.
    << "     |" << std::endl;
  std::cout << "| Cores Utilized   :: "
    << std::setfill(' ') << std::setw(6) << (cpu_percent/100.) * numProcessors << " cores // Avg: "
    << std::setw(6) << (avg_cpu_usage/100.) * numProcessors << " / Max: "
    << std::setw(6) << (*std::max_element(cpu_percents.begin(), cpu_percents.end()) / 100.)
            * numProcessors
    << "     |" << std::endl;
  std::cout << "| CPU Load         :: "
    << std::setfill(' ') << std::setw(6) << cpu_percent << " %     // Avg: "
    << std::setw(6) << avg_cpu_usage << " / Max: "
    << std::setw(6) << *std::max_element(cpu_percents.begin(), cpu_percents.end())
    << "     |" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "RAM Allocation   :: " + to_string_with_precision(resident_set/1000., 2) + " MB"
    << "|" << std::endl;

  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}