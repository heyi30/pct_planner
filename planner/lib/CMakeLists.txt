cmake_minimum_required(VERSION 3.13.0)
project(planning_core CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED true)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -march=native") # have to enable because of gtsam



find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)
include_directories(
  src
  ${EIGEN3_INCLUDE_DIR}

)

add_subdirectory(src/sparse_a_star)
