# find_package(freenect2) for our vcpkg port. Same variables as upstream's config, but found
# relative to this file, so it works wherever vcpkg_installed ends up.
get_filename_component(_freenect2_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
find_library(freenect2_LIBRARY freenect2 PATHS "${_freenect2_root}/lib" NO_DEFAULT_PATH REQUIRED)
set(freenect2_LIBRARIES ${freenect2_LIBRARY})
set(freenect2_INCLUDE_DIRS "${_freenect2_root}/include")
set(freenect2_FOUND TRUE)
