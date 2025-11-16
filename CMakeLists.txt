cmake_minimum_required(VERSION 3.18)
project(cuda-gridadvisor LANGUAGES CXX CUDA)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 3090 = sm_86, RTX 2080 Ti = sm_75, A100 = sm_80
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(device_query
    src/utils/device_query.cu
)
target_include_directories(device_query PUBLIC include)

add_executable(test_device_query
    src/test_device_query.cu
)
target_link_libraries(test_device_query device_query)

install(TARGETS test_device_query DESTINATION bin)

# Test profiler executable
add_executable(test_profiler
    src/test_profiler.cu
)
target_link_libraries(test_profiler device_query)

# Benchmark kernels library
add_library(benchmarks
    benchmarks/memory_bound.cu
    benchmarks/compute_bound.cu
    benchmarks/balanced.cu
)
target_include_directories(benchmarks PUBLIC include)

# Test benchmarks executable
add_executable(test_benchmarks
    src/test_benchmarks.cu
)
target_link_libraries(test_benchmarks device_query benchmarks)

# Test feature extraction
add_executable(test_features
    src/test_features.cu
)
target_link_libraries(test_features device_query benchmarks)

# Main GridAdvisor tool
add_executable(gridadvisor
    src/gridadvisor.cu
)
target_link_libraries(gridadvisor device_query benchmarks)
