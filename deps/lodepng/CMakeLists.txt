cmake_minimum_required (VERSION 2.6)
project (LodePNG)

set (LodePNG_INCLUDE_DIR "${PROJECT_SOURCE_DIR}" CACHE INTERNAL "")
set (LodePNG_SRC_DIR     "${PROJECT_SOURCE_DIR}" CACHE INTERNAL "")

file(GLOB_RECURSE SOURCES "${LodePNG_SRC_DIR}/*.cpp" "${LodePNG_SRC_DIR}/*.c")

include_directories("${LodePNG_INCLUDE_DIR}")

add_library(LodePNG STATIC ${SOURCES})
