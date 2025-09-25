cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(LibTorchDemo)

# compile options
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 14)

# package
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/libtorch)

add_executable(digit digit.cpp)
# libtorch
target_link_libraries(digit ${TORCH_LIBRARIES})
target_link_libraries(digit ${OpenCV_LIBS})