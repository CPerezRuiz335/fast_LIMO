#pragma once

#include <Eigen/Dense>

#include <ros/ros.h>

#include <tf2/convert.h>
#include <tf2_ros/transform_broadcaster.h>

// #include <geometry_msgs/QuaternionStamped.h>
// #include <geometry_msgs/PointStamped.h>
// #include <geometry_msgs/PoseStamped.h>
// #include <geometry_msgs/TransformStamped.h>

#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>

#include "Imu.hpp"
#include "State.hpp"
#include "PCL.hpp"
#include "Config.hpp"


namespace limoncello {

Imu fromROS(const sensor_msgs::Imu::ConstPtr& in) {
  Imu out;
  out.stamp = in->header.stamp.toSec();

  out.ang_vel(0) = in->angular_velocity.x;
  out.ang_vel(1) = in->angular_velocity.y;
  out.ang_vel(2) = in->angular_velocity.z;

  out.lin_accel(0) = in->linear_acceleration.x;
  out.lin_accel(1) = in->linear_acceleration.y;
  out.lin_accel(2) = in->linear_acceleration.z;

  tf2::fromMsg(in->orientation, out.q);

  return out;
}

void publish(const PointCloudT::Ptr& cloud, 
             const ros::NodeHandle& nh,
             const std::string& topic,
             const std::string& frame_id) {
  
  sensor_msgs::PointCloud2 out;
  pcl::toROSMsg(*cloud, out);
  oc_ros.header.stamp = ros::Time::now();

}

void publish(State& state, 
             ros::NodeHandle& nh,
             const std::string& topic,
             const std::string& frame_id,
             const esekfom::esekf<state_ikfom, 12, input_ikfom>::cov& P) {

  Config& cfg = Config::getInstance();


  nav_msgs::Odometry out;

  // Pose/Attitude
  out.pose.pose.position    = tf2::toMsg(state.p);
  out.pose.pose.orientation = tf2::toMsg(state.q);

  // Twist
  out.twist.twist.linear.x = state.v(0);
  out.twist.twist.linear.y = state.v(1);
  out.twist.twist.linear.z = state.v(2);

  out.twist.twist.angular.x = state.w(0);
  out.twist.twist.angular.y = state.w(1);
  out.twist.twist.angular.z = state.w(2);


  // Covariances
  Eigen::Matrix<double, 6, 6> P_pose = Eigen::Matrix<double, 6, 6>::Zero();
  P_pose.block<3, 3>(0, 0) = P.block<3, 3>(3, 3);
  P_pose.block<3, 3>(0, 3) = P.block<3, 3>(3, 0);
  P_pose.block<3, 3>(3, 0) = P.block<3, 3>(0, 3);
  P_pose.block<3, 3>(3, 3) = P.block<3, 3>(0, 0);

  std::vector<double> cov_pose(P_pose.size());
  Eigen::Map<Eigen::MatrixXd>(cov_pose.data(), P_pose.rows(), P_pose.cols()) = P_pose;

  Eigen::Matrix<double, 6, 6> P_twist = Eigen::Matrix<double, 6, 6>::Zero();

  P_twist.block<3, 3>(0, 0) = P.block<3, 3>(6, 6);
  P_twist.block<3, 3>(3, 3) = cfg.ikfom.cov_gyro * Eigen::Matrix3d::Identity();

  std::vector<double> cov_twist(P_twist.size());
  Eigen::Map<Eigen::MatrixXd>(cov_twist.data(), P_twist.rows(), P_twist.cols()) = P_twist;

  for (int i=0; i < cov_pose.size(); i++) {
    out.pose.covariance[i]  = cov_pose[i];
    out.twist.covariance[i] = cov_twist[i];
  }

  out.header.frame_id = frame_id;
  out.header.stamp = ros::Time::now();

  static ros::Publisher pub = nh.advertise<nav_msgs::Odometry>(topic, 1000);

  pub.publish(out);  
}

}
