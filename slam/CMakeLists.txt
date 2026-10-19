cmake_minimum_required(VERSION 3.16)
project(smoke_rescue_rgbd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4.4 REQUIRED)
find_package(Eigen3 3.1 REQUIRED)
find_package(Pangolin REQUIRED)

set(ORB_SLAM3_ROOT "/opt/ORB_SLAM3" CACHE PATH "ORB-SLAM3 source tree")

add_executable(smoke_rescue_rgbd rgbd_runner.cc)
target_include_directories(smoke_rescue_rgbd PRIVATE
    ${ORB_SLAM3_ROOT}
    ${ORB_SLAM3_ROOT}/include
    ${ORB_SLAM3_ROOT}/include/CameraModels
    ${ORB_SLAM3_ROOT}/Thirdparty/DBoW2
    ${ORB_SLAM3_ROOT}/Thirdparty/g2o
    ${ORB_SLAM3_ROOT}/Thirdparty/Sophus
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)
target_link_libraries(smoke_rescue_rgbd PRIVATE
    ${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so
    ${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib/libg2o.so
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    Eigen3::Eigen
    boost_serialization
    crypto
    pthread
)

install(TARGETS smoke_rescue_rgbd RUNTIME DESTINATION bin)
