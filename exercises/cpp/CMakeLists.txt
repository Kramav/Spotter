cmake_minimum_required(VERSION 3.20)
project(SpotterKinect CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Both come from vcpkg.json, installed automatically at configure time (see CMakePresets.json).
# libfreenect2 is our own port in ports/libfreenect2 (D14).
find_package(freenect2 REQUIRED)
find_package(OpenCV REQUIRED)

get_filename_component(repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

# The original examples, kept as they were in legacy/ (D34). New ones link the spotter library.
foreach(example 01_Kinect_Depth_Peek 02_Kinect_Detect_Distance 03_Kinect_Depth_Blobs 04_Kinect_Track_Smooth 05_Kinect_Kalman 05b_Kinect_Two_Thresholds 05c_Kinect_Background_Detect 06_Kinect_Radar 06b_Kinect_Radar_Walls 06c_Kinect_Wall_Filter 07_Kinect_Capture_Average 07b_Kinect_Far_Range 07c_Kinect_Depth_Background 07d_Kinect_Remove_Person 08_Kinect_Point_Cloud 08b_Kinect_Object_Mesh 09_Kinect_Find_Floor 09b_Kinect_Room_Mesh 09c_Kinect_Room_Model 10_Kinect_Two_Views 11_Kinect_ICP 11b_Kinect_Feature_Pose 13_Kinect_Room_Scan 13b_Kinect_Room_Scan_HQ 13c_Kinect_Room_Scan_Assist 13d_Kinect_Room_Scan_Far 13e_Kinect_Room_Scan_Autosave 13z_Kinect_Room_Rebuild)
    add_executable(${example} legacy/${example}/main.cpp)
    target_include_directories(${example} PRIVATE ${freenect2_INCLUDE_DIRS})
    target_link_libraries(${example} PRIVATE ${freenect2_LIBRARIES} ${OpenCV_LIBS})
    target_compile_definitions(${example} PRIVATE SPOTTER_REPO_ROOT="${repo_root}")
endforeach()

# The presentation demos (legacy/presentation/README.md): fullscreen remixes of the examples above, for an exhibition.
foreach(demo Demo1_You_In_3D Demo2_Invisibility_Cloak Demo3_Booth_Radar Demo4_Room_Twin Demo5_How_Tall Demo6_Room_Filled)
    add_executable(${demo} legacy/presentation/${demo}/main.cpp)
    target_include_directories(${demo} PRIVATE ${freenect2_INCLUDE_DIRS})
    target_link_libraries(${demo} PRIVATE ${freenect2_LIBRARIES} ${OpenCV_LIBS})
    target_compile_definitions(${demo} PRIVATE SPOTTER_REPO_ROOT="${repo_root}")
endforeach()

# The shared library (D34): new and re-implemented examples link it.
file(GLOB spotter_sources CONFIGURE_DEPENDS spotter/src/*.cpp)
add_library(spotter STATIC ${spotter_sources})
target_include_directories(spotter PUBLIC spotter/include PRIVATE ${freenect2_INCLUDE_DIRS})
target_link_libraries(spotter PUBLIC ${freenect2_LIBRARIES} ${OpenCV_LIBS})

# One self-check per module, on synthetic data (no Kinect). Run them all: ctest --test-dir build
enable_testing()
file(GLOB spotter_checks CONFIGURE_DEPENDS spotter/checks/check_*.cpp)
foreach(check_source ${spotter_checks})
    get_filename_component(check ${check_source} NAME_WE)
    add_executable(${check} ${check_source})
    target_include_directories(${check} PRIVATE ${freenect2_INCLUDE_DIRS})   # check_frame compares with libfreenect2
    target_link_libraries(${check} PRIVATE spotter)
    target_compile_definitions(${check} PRIVATE SPOTTER_REPO_ROOT="${repo_root}")
    add_test(NAME ${check} COMMAND ${check})
endforeach()

# The examples re-written on the library (examples/README.md).
add_subdirectory(examples)
