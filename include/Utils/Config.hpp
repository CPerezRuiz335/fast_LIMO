#pragma once

#include <ros/ros.h>

#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include <iomanip> 
#include <type_traits>

// Helper to check if a type is printable with std::cout
template <typename T, typename = void>
struct is_printable : std::false_type {};

template <typename T>
struct is_printable<T, std::void_t<decltype(std::cout << std::declval<T>())>> : std::true_type {};

// Helper to check if a type is iterable
template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>> : std::true_type {};

// Generic print function for printable objects
template <typename T>
typename std::enable_if<is_printable<T>::value && !is_iterable<T>::value>::type
print(const T& value, int precision = 3) {
    std::cout << std::fixed << std::setprecision(precision) << value << std::endl;
    std::cout.unsetf(std::ios::fixed);
    std::cout.precision(6); // Restore default
}

// Print function for iterable containers
template <typename Container>
typename std::enable_if<is_iterable<Container>::value>::type
print(const Container& container, int precision = 3) {
    std::cout << "[";
    bool first = true;
    for (const auto& element : container) {
        if (!first) std::cout << ", ";
        first = false;
        print(element, precision); // Recursively print elements
    }
    std::cout << "]" << std::endl;
}

// Specialization for Eigen matrices/vectors
template <typename Derived>
void print(const Eigen::MatrixBase<Derived>& matrix, int precision = 3) {
    std::cout << std::fixed << std::setprecision(precision) << matrix << std::endl;
    std::cout.unsetf(std::ios::fixed);
    std::cout.precision(6); // Restore default
}

// Fallback for unsupported types
template <typename T>
typename std::enable_if<!is_printable<T>::value && !is_iterable<T>::value>::type
print(const T&, int = 3) {
    std::cerr << "Error: Type is not printable!" << std::endl;
}

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

      struct {
        double roll, pitch, yaw;
        Eigen::Vector3d t;
      } lidar2baselink;

      struct {
        double roll, pitch, yaw;
        Eigen::Vector3d t;
      } imu2baselink;

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

