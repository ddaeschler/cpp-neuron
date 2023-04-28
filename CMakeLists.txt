cmake_minimum_required(VERSION 3.25)
project(cpp_neuron)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_PREFIX_PATH "./eigen-3.4.0")
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_executable(cpp_neuron main.cpp layer.cpp layer.h activation.cpp activation.h initialization.cpp initialization.h)

target_include_directories(cpp_neuron PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(cpp_neuron Eigen3::Eigen)