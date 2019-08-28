project(TRAIN_PYTORCH_TEST)
cmake_minimum_required(VERSION 3.2)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_definitions(-std=c++11)

# find torch 
set(${TORCH_INCLUDE_DIRS} "")
set(${TORCH_LIBRARIES} "")
if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{TORCH12}/libtorch-release")
else()
    list(APPEND CMAKE_PREFIX_PATH "$ENV{TORCH12}/libtorch-debug")
endif()
find_package(Torch REQUIRED)
message(STATUS "TORCH FOUND!")
message(STATUS "Torch inc: " ${TORCH_INCLUDE_DIRS})
message(STATUS "Torch lib: " ${TORCH_LIBRARIES})



file(GLOB SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

add_executable(train_pytorch ${SRC})
target_link_libraries(train_pytorch "${TORCH_LIBRARIES}")
set_property(TARGET train_pytorch PROPERTY CXX_STANDARD 11)


message(STATUS "Notice that, current build type -> " ${CMAKE_BUILD_TYPE})