cmake_minimum_required(VERSION 3.20)

# vcpkg.json dependencies live here, shared with the libfreenect2 build (see README).
set(VCPKG_INSTALLED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed CACHE PATH "")

project(SpotterKinect CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/install)
find_package(freenect2 REQUIRED)
find_package(OpenCV REQUIRED)

foreach(example 01_Kinect_Depth_Peek)
    add_executable(${example} ${example}/main.cpp)
    target_include_directories(${example} PRIVATE ${freenect2_INCLUDE_DIRS})
    target_link_libraries(${example} PRIVATE ${freenect2_LIBRARIES} ${OpenCV_LIBS})

    # Windows looks for DLLs next to the .exe. vcpkg copies OpenCV's; libfreenect2 and its
    # dependencies are copied here. (Linux finds them through the rpath CMake sets.)
    # Exit code C0000135 = a DLL is missing from this list: `dumpbin /dependents` names it.
    if(WIN32)
        add_custom_command(TARGET ${example} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/install/bin/freenect2.dll
            ${VCPKG_INSTALLED_DIR}/x64-windows/bin/libusb-1.0.dll
            ${VCPKG_INSTALLED_DIR}/x64-windows/bin/turbojpeg.dll
            ${VCPKG_INSTALLED_DIR}/x64-windows/bin/spng.dll   # turbojpeg needs it
            $<TARGET_FILE_DIR:${example}>)
    endif()
endforeach()
