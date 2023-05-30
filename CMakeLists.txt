cmake_minimum_required(VERSION 3.22)
project(tojalgrad_core)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)

add_subdirectory(lib/googletest)

include_directories(include ${Eigen3_INCLUDE_DIRS})

add_library(tojalgrad_core SHARED src/nn/Neuron.cpp src/nn/Activation.cpp src/nn/layers/Linear.cpp src/nn/layers/Layer.cpp src/nn/Model.cpp)
target_link_libraries(tojalgrad_core ${Eigen3_LIBRARIES})


add_executable(test test/NeuronActivationTests.cpp test/LayerTests.cpp test/ModelTests.cpp)
target_link_libraries(test tojalgrad_core gtest gtest_main)
