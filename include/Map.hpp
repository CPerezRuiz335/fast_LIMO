#pragma once

#include <vector>

#include <Eigen/Dense>

#include "PCL.hpp"


namespace limoncello {

struct alignas(16) Vec3 {
  float x;
  float y;
  float z;
};


inline bool estimate_plane(Eigen::Vector4f& pabcd,
                    const std::vector<pcl::PointXYZ>& pts,
                    const double& thresh) {

  int NUM_MATCH_POINTS = pts.size();
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(NUM_MATCH_POINTS, 3);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> b(NUM_MATCH_POINTS, 1);
  
  A.setZero();
  b.setOnes();
  b *= -1.0f;

  for (int j = 0; j < NUM_MATCH_POINTS; j++) {
    A(j,0) = pts[j].x;
    A(j,1) = pts[j].y;
    A(j,2) = pts[j].z;
  }

  Eigen::Matrix<float, 3, 1> normvec = A.colPivHouseholderQr().solve(b);
  Eigen::Vector4f pca_result;

  float n = normvec.norm();
  pca_result(0) = normvec(0) / n;
  pca_result(1) = normvec(1) / n;
  pca_result(2) = normvec(2) / n;
  pca_result(3) = 1.0 / n;
  
  pabcd = pca_result;

  for (int j = 0; j < pts.size(); j++) {
    double dist2point = std::fabs(pabcd(0) * pts[j].x 
                      + pabcd(1) * pts[j].y
                      + pabcd(2) * pts[j].z
                      + pabcd(3));
    
    if (dist2point > thresh)
      return false;
  }

  return true;
}


struct Match {
  Eigen::Vector3f local;
  Eigen::Vector3f global;
  Eigen::Vector4f n; // normal vector

  Match() = default;
  Match(Eigen::Vector3f& local_,
        Eigen::Vector3f& global_,
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

}