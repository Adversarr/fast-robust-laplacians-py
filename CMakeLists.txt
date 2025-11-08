cmake_minimum_required(VERSION 3.10...4.0)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set the project name
project(fast-robust-laplacian-py)

# Recurse in to pybind
set(PYBIND11_NEWPYTHON ON)
add_subdirectory(deps/pybind11)

# Enable OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  set(USE_OPENMP ON)
endif()

# set location of eigen for geometry-central
set(GC_EIGEN_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen" CACHE PATH "my path")

# geometry-central
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(deps/geometry-central)

pybind11_add_module(fast_robust_laplacian_bindings 
  src/cpp/point_cloud_utilities.cpp
  src/cpp/core.cpp
)

include_directories(fast_robust_laplacian_bindings ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)
include_directories(fast_robust_laplacian_bindings ${CMAKE_CURRENT_SOURCE_DIR}/deps/jc_voronoi/include)

if(USE_OPENMP)
  target_link_libraries(fast_robust_laplacian_bindings PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(fast_robust_laplacian_bindings PRIVATE geometry-central)

install(TARGETS fast_robust_laplacian_bindings LIBRARY DESTINATION .)