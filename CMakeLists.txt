cmake_minimum_required(VERSION 3.25)
project(sGLOH_opencv)

# Set C++ standard and compiler options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Set vcpkg toolchain file
set(CMAKE_TOOLCHAIN_FILE "${HOME}/tools/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# Find OpenCV package and include directories
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(sGLOH_opencv src/main.cpp src/sGLOH2.cpp src/sGLOH2.hpp src/tests.hpp)
target_link_libraries(sGLOH_opencv ${OpenCV_LIBS})
