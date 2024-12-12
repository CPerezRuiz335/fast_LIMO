#pragma once

#include <vector>

#define PCL_NO_PRECOMPILE
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include "Config.hpp"



namespace velodyne_ros {
  struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    float time;
    std::uint8_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}

POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity)
  (float, time, time)
  (std::uint16_t, ring, ring)
)

namespace ouster_ros {
  struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;             // equivalent to signal
    std::uint32_t t;
    std::uint16_t reflectivity;
    std::uint16_t ring;           // equivalent to channel
    std::uint16_t ambient;       // equivalent to near_ir
    std::uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}

POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity)
  (std::uint32_t, t, t)
  (std::uint16_t, reflectivity, reflectivity)
  (std::uint16_t, ring, ring)
  (std::uint16_t, ambient, ambient)
  (std::uint32_t, range, range)
)

struct EIGEN_ALIGN16 PointT {
  PCL_ADD_POINT4D;
  float intensity;
  float range;
  double stamp; // Legacy, change to stamp (TODO)
  std::uint16_t ring;
  std::uint16_t reflectivity;
  std::uint16_t ambient;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(PointT,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity)
  (float, range, range)
  (double, stamp, stamp) // Legacy, change to stamp (TODO)
  (std::uint16_t, ring, ring)
  (std::uint16_t, reflectivity, reflectivity)
  (std::uint16_t, ambient, ambient)
)

namespace limoncello {

typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<ouster_ros::Point> OusterPointCloud;
typedef pcl::PointCloud<velodyne_ros::Point> VelodynePointCloud;


static void point2PointT(const ouster_ros::Point& o, PointT& p) {
  p.x = o.x;
  p.y = o.y;
  p.z = o.z;
  p.intensity = o.intensity;

  p.range = static_cast<float>(o.range) / 1000.0f; // in milimeters
  p.stamp = static_cast<double>(o.t) * 1e-9f;

  p.ring = o.ring;
  p.reflectivity = o.reflectivity;
  p.ambient = o.ambient;
}


static void point2PointT(const velodyne_ros::Point& v, PointT& p) {
  p.x = v.x;
  p.y = v.y;
  p.z = v.z;
  p.intensity = v.intensity;
  p.stamp = v.time;
  p.ring = v.ring;
  p.range = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}


template <typename CloudT>
static void adapter(const sensor_msgs::PointCloud2::ConstPtr& msg,
                    PointCloudT::Ptr& frame) {
  
  Config& cfg = Config::getInstance();
  
  typename CloudT::Ptr tmp(new CloudT);
  pcl::fromROSMsg(*msg, *tmp);
  
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*tmp, *tmp, indices);

  frame->reserve(tmp->size());
  
  double start_time = msg->header.stamp.toSec();

  for (const auto& p : tmp->points) {
    PointT my_p;
    point2PointT(p, my_p);
    
    if (cfg.lidar.end_of_sweep)
      my_p.stamp = start_time - my_p.stamp;
    else
      my_p.stamp = start_time + my_p.stamp;

    frame->points.push_back(my_p);
  }
}


void fromROSmsg2PointT(const sensor_msgs::PointCloud2::ConstPtr& msg,
                       PointCloudT::Ptr& frame) {

  Config& cfg = Config::getInstance();

  switch (cfg.sensors.lidar.type) {
    case 0:
      adapter<VelodynePointCloud>(msg, frame); break;
    case 1:
      adapter<OusterPointCloud>(msg, frame); break;

    // default: //TODO  
      // break;
  }
}

} // namespace limoncello