#pragma once

#include <Eigen/Dense>

#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>

namespace fast_limo {

// Ensure 16-byte alignment for this structure
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;  // Adds the x, y, z, and w (padding) members

    float intensity;
    union {
        std::uint32_t t;  // Ouster time since the beginning of scan in nanoseconds
        float time;       // Velodyne time since the beginning of scan in seconds
        double timestamp; // Hesai absolute timestamp in seconds, Livox timestamp in nanoseconds
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // Ensures alignment for Eigen
};

}  // namespace fast_limo

// Register the custom point type with PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(fast_limo::Point,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (std::uint32_t, t, t)  // Comment or uncomment based on sensor type
                                  (float, time, time)
                                  (double, timestamp, timestamp))

namespace fast_limo {
// Typedefs
typedef fast_limo::Point PointType;  // Corrected to refer to fast_limo::Point
typedef pcl::PointXYZ MapPoint;
typedef std::vector<MapPoint, Eigen::aligned_allocator<MapPoint>> MapPoints;
typedef std::vector<PointType, Eigen::aligned_allocator<PointType>> LocPoints;  // Corrected to refer to fast_limo::Point
typedef pcl::PointCloud<PointType> PointCloudT;

}