cmake_minimum_required (VERSION 3.0.2)
project(renderer LANGUAGES CXX)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif(NOT CMAKE_BUILD_TYPE)


#### Check Dependencies ####
# https://cmake.org/cmake/help/v3.0/module/FindOpenGL.html
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)


#### Directory definition ####

set (PROJECT_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
set (PROJECT_SRC_DIR     "${PROJECT_SOURCE_DIR}/src")
set (PROJECT_DEPS_DIR    "${PROJECT_SOURCE_DIR}/deps")
# PROJECT_SOURCE_DIR points to the project root folder
# PROJECT_BINARY_DIR points to the project build folder


#### Adding dependencies ####

# Adding external libs
add_subdirectory(${PROJECT_DEPS_DIR})


#### Source Files ####

# Gets the list of files inside the src directory
# !! Call cmake every time that you add a new file to update the SOURCES var
file(GLOB_RECURSE SOURCES "${PROJECT_SRC_DIR}/*.cpp" "${PROJECT_SRC_DIR}/*.c")
file(GLOB_RECURSE SOURCES_CUDA "${PROJECT_SRC_DIR}/*.cu")


#### Header Files ####

# -- CUDA --
include_directories(${CUDA_INCLUDE_DIRS})

# -- Renderer Includes --
include_directories("${PROJECT_INCLUDE_DIRS}")
cuda_include_directories("${PROJECT_INCLUDE_DIRS}")

## External includes

# -- JSON 11 --
include_directories(${JSON11_INCLUDE_DIRS})
cuda_include_directories("${JSON11_INCLUDE_DIRS}")

# -- Tiny OBJ Loader --
include_directories(${TinyOBJLoader_INCLUDE_DIRS})
cuda_include_directories("${TinyOBJLoader_INCLUDE_DIRS}")

# -- FreeGLUT 3 --
include_directories(${FreeGLUT3_INCLUDE_DIRS})
cuda_include_directories("${FreeGLUT3_INCLUDE_DIRS}")

# -- GLEW 2 --
include_directories(${GLEW_INCLUDE_DIRS})
cuda_include_directories("${GLEW_INCLUDE_DIRS}")

#### Compilation ####

# Compilation Flags
set(COMPILE_FLAGS "")
set(LINKER_FLAGS  "")

if (${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    set(COMPILE_FLAGS "${COMPILE_FLAGS} -W -Wall -pedantic -std=c++11 -Wshadow -O2 -g")
    set(LINKER_FLAGS  "${LINKER_FLAGS} -lm")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    set(COMPILE_FLAGS "${COMPILE_FLAGS} -W -Wall -pedantic -std=c++11 -Wshadow -O2 -g")
    set(LINKER_FLAGS  "${LINKER_FLAGS} -lm")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
  # using Intel C++
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(COMPILE_FLAGS "${COMPILE_FLAGS} /Wall")
    set(LINKER_FLAGS  "${LINKER_FLAGS}")
endif()

# Cmake compilation flags redefinition
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${LINKER_FLAGS}")


# findCUDA Options
set(CUDA_VERBOSE_BUILD ON)
set(CUDA_SEPARABLE_COMPILATION ON) # Allow multiple CUDA files compilation

list (APPEND CUDA_NVCC_FLAGS "-std=c++11")
# list (APPEND CUDA_NVCC_FLAGS "-std=c++11;-g;-G;-pg;-lineinfo")

# Generating the commands to compile .cu into .ptx
CUDA_WRAP_SRCS(renderer PTX CUDA_PTX_FILES ${SOURCES_CUDA})

cuda_add_executable(renderer ${SOURCES} ${SOURCES_CUDA} ${CUDA_PTX_FILES})

target_link_libraries(renderer
    JSON11
    TinyOBJLoader
    freeglut_static
    GLEW
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_curand_LIBRARY}
)
