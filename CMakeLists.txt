cmake_minimum_required(VERSION 3.0.2)
project(fast_limo VERSION 0.0.1)

# Set C++ standard and compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -std=c++17 -pthread -fexceptions -O3 -lm -lblas -llapack")

# Find OpenMP
find_package(OpenMP QUIET)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
  message(WARNING "OpenMP not found.")
endif()

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
  geometry_msgs
  visualization_msgs
  nav_msgs
  tf2
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
    geometry_msgs
    tf2
    visualization_msgs
    nav_msgs
  DEPENDS PCL
  INCLUDE_DIRS include
)

# IKFOM
add_library(IKFOM INTERFACE)

target_link_libraries(IKFOM INTERFACE ${PCL_LIBRARIES} ${OpenMP_LIBS})
target_include_directories(IKFOM INTERFACE
  external/IKFoM/
  external/IKFoM/IKFoM_toolkit 
)

# Collect source files
set(SOURCE_FILES
  src/main.cpp
  include/Modules/Localizer.cpp
)

include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.6.0
)

FetchContent_MakeAvailable(yaml-cpp)

# Link libraries
set(LINK_LIBRARIES
  PUBLIC yaml-cpp::yaml-cpp
  IKFOM
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  ${TBB_IMPORTED_TARGETS}
  ${OpenMP_LIBS}
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Link the executable with libraries
target_link_libraries(${PROJECT_NAME}
  ${LINK_LIBRARIES}
)
