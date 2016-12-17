cmake_minimum_required (VERSION 2.6)
project (TinyOBJLoader)

set (TinyOBJLoader_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" CACHE INTERNAL "")
set (TinyOBJLoader_SRC_DIR      "${PROJECT_SOURCE_DIR}" CACHE INTERNAL "")

file(GLOB_RECURSE SOURCES "${TinyOBJLoader_SRC_DIR}/*.cc")

include_directories("${TinyOBJLoader_INCLUDE_DIRS}")

add_library(TinyOBJLoader STATIC ${SOURCES})
