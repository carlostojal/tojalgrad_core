cmake_minimum_required(VERSION 3.25)
project(tojalgrad_core)

set(CMAKE_CXX_STANDARD 20)

add_library(tojalgrad_core SHARED library.cpp)
