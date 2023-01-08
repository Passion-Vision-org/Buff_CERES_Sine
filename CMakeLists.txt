
 
cmake_minimum_required(VERSION 3.1)
 
#project(matplotlibcpp_test)
project(ceres_sin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Ceres REQUIRED)#这个需要改一下
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(glog REQUIRED)
find_package(Python3 COMPONENTS Development NumPy)
#find_package(matplotlib_cpp REQUIRED)

include_directories(
  ${CERES_INCLUDE_DIRS}
)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/home/xws/matplotlib-cpp/")
 
add_executable(ceres_sin ceres_sin.cpp)
#add_executable(ceres_sin debug.h my_pre.h my_pre.cpp filter_param.yaml)#这个需要改一下啦
#add_executable(matplotlibcpp_test matplotlibcpp_test.cpp)


target_link_libraries(ceres_sin PRIVATE
      ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${Eigen3_LIBRARIES}
    Python3::Python Python3::NumPy
)
#target_link_libraries(matplotlibcpp_test Python3::Python Python3::NumPy)


 
#find_package(Python3 COMPONENTS Development NumPy)
 
target_include_directories(ceres_sin PRIVATE ${Python2_INCLUDE_DIRS} ${Python2_NumPy_INCLUDE_DIRS})

#include_directories("/home/bobo/third_storage/matplotlib-cpp/")

#target_link_libraries(test_every Python3::Python Python3::NumPy)
