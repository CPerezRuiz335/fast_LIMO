#pragma once

#include <ros/ros.h>

#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>


struct Config {

	bool verbose;
	bool debug;

  struct Topics {
  	struct {
  		std::string lidar;
  		std::string imu;
  	} input;

  	struct {
  		std::string state;
  		std::string frame;
  	} output;
  	
  	std::string frame_id;
  } topics;


  struct Sensors {
  	struct { 
  		int type; 
  		bool end_of_sweep;
  	} lidar;

  	struct {
  		int hz;
  	} imu;

  	struct {
  		bool gravity_align;
  		bool accel;
  		bool gyro;
  		float time;
  	} calibration;

  	bool time_offset;
    float TAI_offset;

  	struct {
  		Eigen::Affine3d imu2baselink_T;
  		Eigen::Affine3d lidar2baselink_T;
  		float gravity;
  	} extrinsics;

  	struct {
  		Eigen::Vector3d accel_bias;
  		Eigen::Vector3d gyro_bias;
  		Eigen::Matrix3d sm;
  	} intrinsics;

  } sensors;

  struct Filters {
    struct {
    	bool active;
    	Eigen::Vector4d leaf_size;
    } voxel_grid;

    struct {
    	bool active;
    	float value;
    } min_distance;

    struct {
    	bool active;
    	float value;
    } fov;

    struct {
			bool active;
			int value;
    } rate_sampling;

  } filters;

  struct IKFoM {
  	int query_iters;
  	int max_iters;
  	float tolerance;
  	bool estimate_extrinsics;
  	float lidar_noise;

  	struct {
  		float gyro;
  		float accel;
  		float bias_gyro;
  		float bias_accel;
  	} covariance;

  	struct {
  		int points;
  		float max_sqrt_dist;
  		float plane_threshold;
  	} plane;
  } ikfom;

  struct iOctree {
    bool order;
    float min_extent;
    int bucket_size;
    bool downsample;
  } ioctree;


  // Function to fill configuration using ROS NodeHandle
  void fill(ros::NodeHandle& nh) {
    
  	nh.getParam("verbose", verbose);
  	nh.getParam("debug",   debug);

    // TOPICS
    nh.getParam("topics/input/lidar",  topics.input.lidar);
    nh.getParam("topics/input/imu",    topics.input.imu);
    nh.getParam("topics/output/state", topics.output.state);
    nh.getParam("topics/output/frame", topics.output.frame);
    nh.getParam("topics/frame_id",     topics.frame_id);


  	// SENSORS
    nh.getParam("sensors/lidar/type",          sensors.lidar.type);
    nh.getParam("sensors/lidar/end_of_sweep",  sensors.lidar.end_of_sweep);
    nh.getParam("sensors/imu/hz",              sensors.imu.hz);

    nh.getParam("sensors/calibration/gravity_align", sensors.calibration.gravity_align);
    nh.getParam("sensors/calibration/accel",         sensors.calibration.accel);
    nh.getParam("sensors/calibration/gyro",          sensors.calibration.gyro);
    nh.getParam("sensors/calibration/time",          sensors.calibration.time);

    nh.getParam("sensors/time_offset", sensors.time_offset);
    nh.getParam("sensors/TAI_offset", sensors.TAI_offset);


    std::vector<double> tmp;
    nh.getParam("sensors/extrinsics/imu2baselink/t", tmp);

    sensors.extrinsics.imu2baselink_T.setIdentity();
    sensors.extrinsics.imu2baselink_T.translate(Eigen::Vector3d(tmp[0], tmp[1], tmp[2]));

    nh.getParam("sensors/extrinsics/imu2baselink/R", tmp);
    Eigen::Matrix3d R_imu = (
    	  Eigen::AngleAxisd(tmp[0] * M_PI/180., Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(tmp[1] * M_PI/180., Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(tmp[2] * M_PI/180., Eigen::Vector3d::UnitZ())
      ).toRotationMatrix();

    sensors.extrinsics.imu2baselink_T.rotate(R_imu);

    nh.getParam("sensors/extrinsics/lidar2baselink/t", tmp);

    sensors.extrinsics.lidar2baselink_T.setIdentity();
    sensors.extrinsics.lidar2baselink_T.translate(Eigen::Vector3d(tmp[0], tmp[1], tmp[2]));

    nh.getParam("sensors/extrinsics/lidar2baselink/R", tmp);
    Eigen::Matrix3d R_lidar = (
    	  Eigen::AngleAxisd(tmp[0] * M_PI/180., Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(tmp[1] * M_PI/180., Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(tmp[2] * M_PI/180., Eigen::Vector3d::UnitZ())
      ).toRotationMatrix();

    sensors.extrinsics.lidar2baselink_T.rotate(R_lidar);

    nh.getParam("sensors/extrinsics/gravity", sensors.extrinsics.gravity);

    nh.getParam("sensors/intrinsics/accel_bias", tmp);
    sensors.intrinsics.accel_bias = Eigen::Vector3d(tmp[0], tmp[1], tmp[2]);

    nh.getParam("sensors/intrinsics/gyro_bias", tmp);
    sensors.intrinsics.gyro_bias = Eigen::Vector3d(tmp[0], tmp[1], tmp[2]);

    nh.getParam("sensors/intrinsics/sm", tmp);
    sensors.intrinsics.sm << tmp[0], tmp[1], tmp[2],
    												 tmp[3], tmp[4], tmp[5],
    												 tmp[6], tmp[7], tmp[8];


	  // FILTERS
    nh.getParam("filters/voxel_grid/active", filters.voxel_grid.active);
    nh.getParam("filters/voxel_grid/leaf_size", tmp);
    filters.voxel_grid.leaf_size = Eigen::Vector4d(tmp[0], tmp[1], tmp[2], 1.);

    nh.getParam("filters/min_distance/active", filters.min_distance.active);
    nh.getParam("filters/min_distance/value", filters.min_distance.value);

    nh.getParam("filters/fov/active", filters.fov.active);
    nh.getParam("filters/fov/value", filters.fov.value);

    nh.getParam("filters/rate_sampling/active", filters.rate_sampling.active);
    nh.getParam("filters/rate_sampling/value", filters.rate_sampling.value);


    // IKFoM
    nh.getParam("IKFoM/query_iters",         ikfom.query_iters);
    nh.getParam("IKFoM/max_iters",           ikfom.max_iters);
    nh.getParam("IKFoM/tolerance",           ikfom.tolerance);
    nh.getParam("IKFoM/estimate_extrinsics", ikfom.estimate_extrinsics);
    nh.getParam("IKFoM/lidar_noise",         ikfom.lidar_noise);

    
    nh.getParam("IKFoM/covariance/gyro",       ikfom.covariance.gyro);
    nh.getParam("IKFoM/covariance/accel",      ikfom.covariance.accel);
    nh.getParam("IKFoM/covariance/bias_gyro",  ikfom.covariance.bias_gyro);
    nh.getParam("IKFoM/covariance/bias_accel", ikfom.covariance.bias_accel);

    nh.getParam("IKFoM/plane/points",          ikfom.plane.points);
    nh.getParam("IKFoM/plane/max_sqrt_dist",   ikfom.plane.max_sqrt_dist);
    nh.getParam("IKFoM/plane/plane_threshold", ikfom.plane.plane_threshold);


    // iOctree
    nh.getParam("iOctree/order", ioctree.order);
    nh.getParam("iOctree/min_extent", ioctree.min_extent);
    nh.getParam("iOctree/bucket_size", ioctree.bucket_size);
    nh.getParam("iOctree/downsample", ioctree.downsample);
  }

  static Config& getInstance() {
    static Config* config = new Config();
    return *config;
  }

 private:
  // Singleton pattern
  Config() = default;

  // Delete copy/move so extra instances can't be created/moved.
  Config(const Config&) = delete;
  Config& operator=(const Config&) = delete;
  Config(Config&&) = delete;
  Config& operator=(Config&&) = delete;
};

