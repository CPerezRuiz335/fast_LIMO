#include <mutex>
#include <condition_variable>

#include <Eigen/Dense>

#include <ros/ros.h>

#include <tf2/convert.h>

#include <geometry_msgs/Vector3.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>


class Manager {
  limoncello::State state_;

  limoncello::Publisher publisher_;

  limoncello::States state_buffer_;
  
  double prev_imu_stamp_;
  double first_imu_stamp_;
  double prev_scan_stamp_;

  bool imu_calibrated_;

  std::mutex mtx_state_;
  std::mutex mtx_buffer_;

  std::condition_variable cv_prop_stamp_;
  
  public:
    Manager();
    ~Manager() = default;

    void imu_callback(const sensor_msgs::Imu::ConstPtr& msg) {

#ifdef DEBUG
  SWRI_PROFILE("imu_callback");
#endif

      Config& config = Config::getInstance();

      Imu imu = limoncello::fromROS(msg);

      if (first_imu_stamp_ < 0.)
        first_imu_stamp_ = imu.stamp;

      if (not imu_calibrated_) {
        static int N(0);
		    static Eigen::Vector3f gyro_avg(0., 0., 0.);
		    static Eigen::Vector3f accel_avg(0., 0., 0.);

        if ((imu.stamp - first_imu_stamp_) < config.imu_calib_time) {
          gyro_avg += imu.ang_vel;
          accel_avg += imu.lin_accel; 
          N++;
        } else {
          gyro_avg /= N;
			    accel_avg /= N;

          state_.initilize(gyro_avg, accel_avg);
        }

      } else {
        limoncello::correct_imu(imu, 
                                state_.b.gyro, 
                                state_.b.accel,
                                config.intrinsics.imu_sm);

        mtx_state_.lock();
          state_.predict(imu);
        mtx_state_.unlock();

        mtx_buffer_.lock();
          state_buffer_.push_front(state_);
        mtx_buffer_.unlock();
        
        cv_prop_stamp_.notify_one();
      }

    }

    void lidar_callback(const sensor_msgs::PointCloud2::ConstPtr& msg) {

#ifdef DEBUG
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


      PointCloudT::Ptr preprocessed(boost::make_shared<PointCloudT>()); 
      {
#ifdef DEBUG
  SWRI_PROFILE("preprocess");
#endif
        preprocessed = limoncello::preprocess(raw);
      }

      PointCloudT::Ptr deskewed(boost::make_shared<PointCloudT>());
      {
#ifdef DEBUG
  SWRI_PROFILE("deskew");
#endif   
        double offset = 0.0;
        if (config.time_offset) { // automatic sync (not precise!)
          offset = state_.stamp - preprocessed->points.back().stamp - 1.e-4; 
          if (offset > 0.0) offset = 0.0; // don't jump into future
        }

        double end_stamp = preprocessed->points.back().stamp + offset;
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

        deskewed = limoncello::deskew(preprocessed, state_, state_buffer_, offset);
      }

      PointCloudT::Ptr downsampled(boost::make_shared<PointCloudT>());
      {
#ifdef DEBUG
  SWRI_PROFILE("downsample");
#endif 
        downsampled  = limoncello::downsample(deskewed);
      }

      if (downsampled->points.empty()) {
        ROS_ERROR("[LIMONCELLO] Processed & downsampled cloud is empty!");
        return;
      }

      {
#ifdef DEBUG
  SWRI_PROFILE("update");
#endif 
        mtx_state_.lock();
          state_.update(downsampled);
        mtx_state_.unlock();
      }

      PointCloudT::Ptr global(boost::make_shared<PointCloudT>());
      pcl::transformPointCloud(*deskewed, *global,
		                           state_.get_RT() * state_.get_extr_RT());

      }
    }
};


int main(int argc, char** argv) {

  ros::init(argc, argv, "limovelo");
  ros::NodeHandle nh("~");

  // Setup config parameters TODO
  limoncello::Config& config = limoncello::Config::getInstance();
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

