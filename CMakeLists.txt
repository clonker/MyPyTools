cmake_minimum_required(VERSION 3.3)
project(PyTools)

ADD_SUBDIRECTORY(libs/pybind11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

pybind11_add_module(super_resolution_tools src/super-resolution/main.cpp)
pybind11_add_module(finite_differences src/finite-differences/finite-differences.cpp)