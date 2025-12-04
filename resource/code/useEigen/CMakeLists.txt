cmake_minimum_required(VERSION 3.16)

project(useEigen)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)

include_directories("/usr/include/eigen3")

add_executable(useEigen eigenMatrix.cpp)
add_executable(useGeometry useGeometry.cpp)
add_executable(TEST 3dVector.cpp)
