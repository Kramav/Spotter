cmake_minimum_required(VERSION 3.20)
project(SpotterKinect CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Both come from vcpkg.json, installed automatically at configure time (see CMakePresets.json).
# libfreenect2 is our own port in ports/libfreenect2 (D14).
find_package(freenect2 REQUIRED)
find_package(OpenCV REQUIRED)

get_filename_component(repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

# Warnings on our own code (the review's initiative 7): the library, the console guard and the checks. Headers
# included with <...> (OpenCV, libfreenect2, the standard library) stay quiet on MSVC.
if(MSVC)
    # _CRT_SECURE_NO_WARNINGS: MSVC calls getenv, localtime, strerror and _wfopen unsafe, but their _s
    # replacements don't exist on Linux (D7).
    set(spotter_warnings /W4 /external:anglebrackets /external:W0 /D_CRT_SECURE_NO_WARNINGS)
else()
    set(spotter_warnings -Wall -Wextra)
endif()

# The original examples and presentation demos, kept as they were in legacy/ (D34): an archive, built only
# when asked (D47): cmake --preset default -DSPOTTER_BUILD_LEGACY=ON. Git tag legacy-archive marks them too.
option(SPOTTER_BUILD_LEGACY "Also build the legacy/ archive (34 programs, into build/legacy)" OFF)
if(SPOTTER_BUILD_LEGACY)
    foreach(example 01_Kinect_Depth_Peek 02_Kinect_Detect_Distance 03_Kinect_Depth_Blobs 04_Kinect_Track_Smooth 05_Kinect_Kalman 05b_Kinect_Two_Thresholds 05c_Kinect_Background_Detect 06_Kinect_Radar 06b_Kinect_Radar_Walls 06c_Kinect_Wall_Filter 07_Kinect_Capture_Average 07b_Kinect_Far_Range 07c_Kinect_Depth_Background 07d_Kinect_Remove_Person 08_Kinect_Point_Cloud 08b_Kinect_Object_Mesh 09_Kinect_Find_Floor 09b_Kinect_Room_Mesh 09c_Kinect_Room_Model 10_Kinect_Two_Views 11_Kinect_ICP 11b_Kinect_Feature_Pose 13_Kinect_Room_Scan 13b_Kinect_Room_Scan_HQ 13c_Kinect_Room_Scan_Assist 13d_Kinect_Room_Scan_Far 13e_Kinect_Room_Scan_Autosave 13z_Kinect_Room_Rebuild)
        add_executable(${example} legacy/${example}/main.cpp)
        target_include_directories(${example} PRIVATE ${freenect2_INCLUDE_DIRS})
        target_link_libraries(${example} PRIVATE ${freenect2_LIBRARIES} ${OpenCV_LIBS})
        target_compile_definitions(${example} PRIVATE SPOTTER_REPO_ROOT="${repo_root}")
        set_target_properties(${example} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/legacy)
    endforeach()

    # The presentation demos (legacy/presentation/README.md): fullscreen remixes of the examples above, for an exhibition.
    foreach(demo Demo1_You_In_3D Demo2_Invisibility_Cloak Demo3_Booth_Radar Demo4_Room_Twin Demo5_How_Tall Demo6_Room_Filled)
        add_executable(${demo} legacy/presentation/${demo}/main.cpp)
        target_include_directories(${demo} PRIVATE ${freenect2_INCLUDE_DIRS})
        target_link_libraries(${demo} PRIVATE ${freenect2_LIBRARIES} ${OpenCV_LIBS})
        target_compile_definitions(${demo} PRIVATE SPOTTER_REPO_ROOT="${repo_root}")
        set_target_properties(${demo} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/legacy)
    endforeach()
endif()

# The shared library (D34): new and re-implemented examples link it.
file(GLOB spotter_sources CONFIGURE_DEPENDS spotter/src/*.cpp)
add_library(spotter STATIC ${spotter_sources})
target_include_directories(spotter PUBLIC spotter/include PRIVATE ${freenect2_INCLUDE_DIRS})
target_link_libraries(spotter PUBLIC ${freenect2_LIBRARIES} ${OpenCV_LIBS})
if(WIN32)
    target_link_libraries(spotter PUBLIC ole32 shell32 uuid)   # file_dialog: the Windows Save As / Open dialogs
endif()
# room_folder refuses a room folder inside build/ (a clean rebuild deletes it) and knows the repo's room_scans/ (D48).
target_compile_definitions(spotter PRIVATE SPOTTER_BUILD_DIR="${CMAKE_BINARY_DIR}" SPOTTER_REPO_ROOT="${repo_root}")
target_compile_options(spotter PRIVATE ${spotter_warnings})

# The console guard (D49): linked into every program in examples/, so its console never closes by itself on an
# error or crash (and, double-clicked, waits for Enter at the end). OBJECT, so it always links; the checks don't use it.
add_library(spotter_console_guard OBJECT spotter/console_guard/console_guard.cpp)
target_link_libraries(spotter_console_guard PUBLIC ${OpenCV_LIBS})
target_compile_options(spotter_console_guard PRIVATE ${spotter_warnings})

# Self-checks on synthetic data (no Kinect): 10 for library modules (not every module has its own: stats, detection,
# walls, features_orb, room_scan and room_folder are exercised through check_room; source_freenect2, radar, view and
# file_dialog have none), plus three that include an example's own code (13e's session, 13x's fill, 13w's shell).
# The room checks share spotter/checks/test_room.hpp. Run them all: ctest --test-dir build
enable_testing()
file(GLOB spotter_checks CONFIGURE_DEPENDS spotter/checks/check_*.cpp)
foreach(check_source ${spotter_checks})
    get_filename_component(check ${check_source} NAME_WE)
    add_executable(${check} ${check_source})
    target_include_directories(${check} PRIVATE ${freenect2_INCLUDE_DIRS})   # check_frame compares with libfreenect2
    target_link_libraries(${check} PRIVATE spotter)
    target_compile_options(${check} PRIVATE ${spotter_warnings})
    target_compile_definitions(${check} PRIVATE SPOTTER_REPO_ROOT="${repo_root}" SPOTTER_BUILD_DIR="${CMAKE_BINARY_DIR}")
    add_test(NAME ${check} COMMAND ${check})
endforeach()

# The examples re-written on the library (examples/README.md).
add_subdirectory(examples)
