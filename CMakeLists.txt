cmake_minimum_required(VERSION 3.0.2)
project(fast_limo VERSION 0.0.1)

# Set C++ standard and compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -pthread -fexceptions -O3 -lm -lblas -llapack")

# Find Boost and TBB
find_package(Boost REQUIRED COMPONENTS system)
find_package(TBB REQUIRED)

# Include Boost directories
include_directories(${Boost_INCLUDE_DIRS})

# Find Eigen3 and PCL
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)

# Find ROS packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_ros
  sensor_msgs
  nav_msgs
)

# Include directories
include_directories(
  include
  external/ioctree
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Catkin package configuration
catkin_package(
  CATKIN_DEPENDS
    roscpp
    pcl_ros
    sensor_msgs
    nav_msgs
  DEPENDS PCL
  INCLUDE_DIRS include
)

# IKFOM
add_library(IKFOM INTERFACE)

target_include_directories(IKFOM INTERFACE
  external/IKFoM/
  external/IKFoM/IKFoM_toolkit 
)

# Link libraries
set(LINK_LIBRARIES
  IKFOM
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  ${TBB_IMPORTED_TARGETS}
  ${OpenMP_LIBS}
)

# Create executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Link the executable with libraries
target_link_libraries(${PROJECT_NAME}
  ${LINK_LIBRARIES}
)
