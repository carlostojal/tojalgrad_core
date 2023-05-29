cmake_minimum_required(VERSION 3.25)
project(tojalgrad_core)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)

include_directories(include ${Eigen3_INCLUDE_DIRS})

add_library(tojalgrad_core SHARED src/nn/Neuron.cpp)

target_link_libraries(tojalgrad_core ${Eigen3_LIBRARIES})
