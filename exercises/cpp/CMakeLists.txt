cmake_minimum_required(VERSION 3.20)
project(SpotterKinect CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Both come from vcpkg.json, installed automatically at configure time (see CMakePresets.json).
# libfreenect2 is our own port in ports/libfreenect2 (D14).
find_package(freenect2 REQUIRED)
find_package(OpenCV REQUIRED)

get_filename_component(repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

foreach(example 01_Kinect_Depth_Peek 02_Kinect_Detect_Distance 03_Kinect_Depth_Blobs 04_Kinect_Track_Smooth 05_Kinect_Kalman 05b_Kinect_Two_Thresholds 05c_Kinect_Background_Detect 06_Kinect_Radar 07_Kinect_Capture_Average 07b_Kinect_Far_Range)
    add_executable(${example} ${example}/main.cpp)
    target_include_directories(${example} PRIVATE ${freenect2_INCLUDE_DIRS})
    target_link_libraries(${example} PRIVATE ${freenect2_LIBRARIES} ${OpenCV_LIBS})
    target_compile_definitions(${example} PRIVATE SPOTTER_REPO_ROOT="${repo_root}")
endforeach()
