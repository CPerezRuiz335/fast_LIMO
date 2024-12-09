#pragma once

#include <ros/ros.h>

#include <tf2/convert.h>
#include <tf2_ros/transform_broadcaster.h>

#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>

#include "Imu.hpp"
#include "State.hpp"
#include "PCL.hpp"


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
  
  // TODO

}

void publish(const State& state, 
              const ros::NodeHandle& nh,
              const std::string& topic,
              const std::string& frame_id) {
  
  // TODO
  
}

}
