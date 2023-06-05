cmake_minimum_required(VERSION 3.22)
project(tojalgrad_core)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)

find_package(OpenCL QUIET)

if(OpenCL_FOUND)
    set(USE_OPENCL 1)
    include_directories(${OpenCL_INCLUDE_DIRS})
    add_definitions(-DUSE_OPENCL)
    message(STATUS "OpenCL found, using GPU acceleration.")
else()
    message(WARNING "OpenCL not found, using CPU only!")
endif()

add_subdirectory(lib/googletest)

include_directories(include ${Eigen3_INCLUDE_DIRS})

add_library(tojalgrad_core SHARED src/nn/Neuron.cpp src/nn/Activation.cpp src/nn/layers/Dense.cpp src/nn/layers/Layer.cpp src/nn/models/Model.cpp src/nn/models/BobNet.cpp src/nn/Loss.cpp include/tojalgrad/nn/Loss.h)
target_link_libraries(tojalgrad_core ${Eigen3_LIBRARIES})


add_executable(test test/NeuronActivationTests.cpp test/LayerTests.cpp test/ModelTests.cpp test/LossTests.cpp test/NeuronTrainingTests.cpp)
target_link_libraries(test tojalgrad_core gtest gtest_main)
