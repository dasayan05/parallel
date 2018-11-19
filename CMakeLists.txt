cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(parallel LANGUAGES CXX CUDA)

# find all CUDA sources
file(GLOB_RECURSE CU_SOURCES "src/*.cu" "src/*.cpp")

# library of all kernels
add_library(${PROJECT_NAME} STATIC ${CU_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC "include")

# binary executables
file(GLOB CPP_BINS "bin/*.cpp")
foreach(bin_file ${CPP_BINS})
	get_filename_component(filename ${bin_file} NAME_WE)
	add_executable(${filename} ${bin_file})
	target_link_libraries(${filename} PRIVATE ${PROJECT_NAME})
endforeach(bin_file)