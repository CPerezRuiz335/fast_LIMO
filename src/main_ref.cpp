#include <mutex>
#include <condition_variable>

#include <Eigen/Dense>

#include <ros/ros.h>

#include <tf2/convert.h>

#include <geometry_msgs/Vector3.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>

#include "State.hpp"
#include "use-ikfom.hpp"
#include "Imu.hpp"
#include "ROSutils.hpp"



using namespace limoncello;

class Manager {
  State state_;

  Publisher publisher_;

  States state_buffer_;
  
  Imu prev_imu_;
  double first_imu_stamp_;
  double prev_scan_stamp_;

  bool imu_calibrated_;

  std::mutex mtx_state_;
  std::mutex mtx_buffer_;

  std::condition_variable cv_prop_stamp_;

  ros::NodeHandle nh_;

  esekfom::esekf<state_ikfom, 12, input_ikfom> IKFoM_;
  
public:
  Manager() : first_imu_stamp_(0.0), prev_scan_stamp_(0.0) {
    init_IKFoM(IKFoM_);
    imu_calibrated_ = not Config::getInstance().calibrate_imu; 
  };
  
  ~Manager() = default;

  void imu_callback(const sensor_msgs::Imu::ConstPtr& msg) {

#ifdef PROFILE
SWRI_PROFILE("imu_callback");
#endif

    Config& cfg = Config::getInstance();

    Imu imu = fromROS(msg);

    if (first_imu_stamp_ < 0.)
      first_imu_stamp_ = imu.stamp;

    double dt = imu.stamp - prev_imu_.stamp;
    dt = (dt < 0 or dt > 0.1) ? 1./cfg.imu.hz : dt;

    if (not imu_calibrated_) {
      static int N(0);
      static Eigen::Vector3d gyro_avg(0., 0., 0.);
      static Eigen::Vector3d accel_avg(0., 0., 0.);
      static Eigen::Vector3d grav_vec(0., 0., cfg.gravity);

      if ((imu.stamp - first_imu_stamp_) < cfg.imu.calib_time) {
        gyro_avg  += imu.ang_vel;
        accel_avg += imu.lin_accel; 
        N++;

      } else {
        gyro_avg /= N;
        accel_avg /= N;

        if (cfg.calibration.gravity_align) {
          grav_vec = accel_avg.normalized() * abs(cfg.gravity);
          Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(
                                        grav_vec, 
                                        Eigen::Vector3d(0., 0., cfg.gravity));
          state_.q = q;
          state_.g = grav_vec;
        }
        
        if (cfg.calibration.accel_bias)
          state_.b.accel = accel_avg - grav_vec;

        if (cfg.calibration.gyro_bias)
          state_.b.gyro = gyro_avg;

        setIKFoM_state(IKFoM_, state_);
      }

    } else {
      imu = imu2baselink(imu, dt);
      imu = correct_imu(imu, state_.b.gyro, state_.b.accel, cfg.imu.intrinsics.sm);

      mtx_state_.lock();
        predict(IKFoM_, imu, dt);
      mtx_state_.unlock();

      prev_imu_ = imu;

      state_ = State(IKFoM_.get_x(), imu);

      mtx_buffer_.lock();
        state_buffer_.push_front(state_);
      mtx_buffer_.unlock();

      cv_prop_stamp_.notify_one();

      publish(state_, nh_, cfg.topics.out.state, cfg.topics.frame_id, IKFoM_.get_P());
    }

  }

  void lidar_callback(const sensor_msgs::PointCloud2::ConstPtr& msg) {

#ifdef PROFILE
SWRI_PROFILE("lidar_callback");
#endif

    PointCloudT::Ptr raw(boost::make_shared<PointCloudT>());
    pcl::fromROSMsg(*msg, *raw);

    if (raw->points.empty()) {
      ROS_ERROR("[LIMONCELLO] Raw PointCloud is empty!");
      return;
    }

    if (not imu_calibrated_)
      return;
    
    if (state_buffer_.empty()) {
      ROS_ERROR("[LIMONCELLO] No IMUs received");
      return;
    }

    PointCloudT::Ptr deskewed(boost::make_shared<PointCloudT>());
{
#ifdef PROFILE
SWRI_PROFILE("deskew");
#endif  

    double offset = 0.0;
    if (cfg.deskew.time_offset) { // automatic sync (not precise!)
      offset = state_.stamp - raw->points.back().stamp - 1.e-4; 
      if (offset > 0.0) offset = 0.0; // don't jump into future
    }

    double end_stamp = raw->points.back().stamp + offset;
    if (state_buffer_.empty() || state_buffer_.front().stamp < end_stamp) {
      ROS_INFO_STREAM(
        "PROPAGATE WAITING... \n"
        "     - buffer time: " << state_buffer_.front().stamp << "\n"
        "     - end scan time: " << scan_stamp);

      std::unique_lock<decltype(mtx_buffer_)> lock(mtx_buffer_);
      cv_prop_stamp_.wait(lock, [this, &end_stamp] { 
          return state_buffer_.front().stamp >= end_stamp;
      });

    } 

    deskewed = deskew(raw, state_, state_buffer_, offset);
}

    PointCloudT::Ptr processed(boost::make_shared<PointCloudT>()); 
{
#ifdef PROFILE
SWRI_PROFILE("preprocess");
#endif
    processed = process(raw);
}
   

    PointCloudT::Ptr downsampled(boost::make_shared<PointCloudT>());
{
#ifdef PROFILE
SWRI_PROFILE("downsample");
#endif 
    downsampled  = downsample(processed);
}

    if (downsampled->points.empty()) {
      ROS_ERROR("[LIMONCELLO] Processed & downsampled cloud is empty!");
      return;
    }

{
#ifdef PROFILE
SWRI_PROFILE("update");
#endif 
    mtx_state_.lock();
      update(IKFoM_, downsampled);
    mtx_state_.unlock();
}

    PointCloudT::Ptr global(boost::make_shared<PointCloudT>());
    pcl::transformPointCloud(*deskewed, *global,
                              state_.get_RT() * state_.get_extr_RT());


    // Publish
    publish(global, nh_, cfg.topics.out.global_frame, cfg.topics.frame_id);
  }

};


int main(int argc, char** argv) {

  ros::init(argc, argv, "limoncello");
  ros::NodeHandle nh("~");

  // Setup config parameters TODO
  Config& config = Config::getInstance();
  config.fill(nh);

  // Initialize manager (reads from config)
  Manager manager = Manager();

  // Subscribers
  ros::Subscriber lidar_sub = nh.subscribe(config.topics.in.lidar,
                                           1000,
                                           &Manager::lidar_callback,
                                           &manager,
                                           ros::TransportHints().tcpNoDelay());

  ros::Subscriber imu_sub = nh.subscribe(config.topics.in.imu,
                                         1000,
                                         &Manager::imu_callback,
                                         &manager,
                                         ros::TransportHints().tcpNoDelay());

  ros::AsyncSpinner spinner(0);
  spinner.start();
  
  ros::waitForShutdown();

  return 0;

}

