cmake_minimum_required(VERSION 3.0.2)
project(manif-test VERSION 0.0.1)

# Set C++ standard and compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

# Find Boost and TBB
find_package(Boost REQUIRED COMPONENTS system)

# Include Boost directories
include_directories(${Boost_INCLUDE_DIRS})

# Find Eigen3 and PCL
find_package(Eigen3 REQUIRED)

# Manif
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/manif)

# Link libraries
set(LINK_LIBRARIES
  PRIVATE manif
  ${Boost_LIBRARIES}
)

# Create executable
add_executable(${PROJECT_NAME} main.cpp)

# Link the executable with libraries
target_link_libraries(${PROJECT_NAME}
  ${LINK_LIBRARIES}
)
