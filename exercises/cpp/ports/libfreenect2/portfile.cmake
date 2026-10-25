# libfreenect2 has no official vcpkg port, so this is ours (D9, D14).
# Pinned to the last upstream commit (2020-03-01), checked by hash, with our OpenMP patch (D11).

vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)   # one shared library, even on static Linux triplets
set(VCPKG_BUILD_TYPE release)              # the examples only build Release

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO OpenKinect/libfreenect2
    REF fd64c5d9b214df6f6a55b4419357e51083f15d93
    SHA512 34f3f407dbc47a73b4fec92965943ba403916898dcb38ccbc02205f027531fe07b60a13842ef4225d27f6a2e11ba7d5925064faeaea29b34062b6daaadc52839
    PATCHES openmp.patch
)

# libfreenect2's own libusb finder doesn't know vcpkg's layout, so name the library directly.
if(VCPKG_TARGET_IS_WINDOWS)
    set(LIBUSB_LIBRARIES "${CURRENT_INSTALLED_DIR}/lib/libusb-1.0.lib")
else()
    set(LIBUSB_LIBRARIES "${CURRENT_INSTALLED_DIR}/lib/libusb-1.0.a;udev;pthread")   # static libusb needs udev
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5   # its CMakeLists predates CMake 3.5
        -DBUILD_EXAMPLES=OFF
        -DBUILD_OPENNI2_DRIVER=OFF
        -DENABLE_CXX11=ON
        -DENABLE_OPENCL=OFF                  # CPU decoding only: the Raspberry Pi has no GPU path (D11)
        -DENABLE_CUDA=OFF
        -DENABLE_OPENGL=OFF
        -DENABLE_VAAPI=OFF
        -DENABLE_TEGRAJPEG=OFF
        -DCMAKE_DISABLE_FIND_PACKAGE_PkgConfig=ON   # else its finder asks pkg-config and ignores the two lines below
        "-DLibUSB_INCLUDE_DIRS=${CURRENT_INSTALLED_DIR}/include/libusb-1.0"
        "-DLibUSB_LIBRARIES=${LIBUSB_LIBRARIES}"
)
vcpkg_cmake_install()

# Its installed freenect2Config.cmake has absolute build paths baked in. Replace it with ours.
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib/cmake" "${CURRENT_PACKAGES_DIR}/lib/pkgconfig")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/freenect2Config.cmake" DESTINATION "${CURRENT_PACKAGES_DIR}/share/freenect2")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/APACHE20" "${SOURCE_PATH}/GPL2")
