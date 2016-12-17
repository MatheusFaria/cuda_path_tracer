# CUDA Path Tracer

Path Tracer done in CUDA to pratice CUDA programming and apply computer graphics concepts

## Scene File Format

You can find examples at samples folder.

## Dependencies

* CUDA SDK 7.5
* OpenGL 3+

## Build (Linux and MacOS)

To build you need to install CMake 3.0.

```
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
```

## Build (Windows)

To build you need to install CMake 3.0.

```
    $ mkdir build
    $ cd build
    $ cmake .. -G "Visual Studio 12 2013 Win64"
```

## Reference Links

(GLUT Tutorial)[http://www.lighthouse3d.com/tutorials/glut-tutorial/]
(CUDA and OpenGL Interop - NVIDIA)[https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st]
(CUDA and OpenGL Interop)[https://rauwendaal.net/2011/12/02/writing-to-3d-opengl-textures-in-cuda-4-1-with-3d-surface-writes/]
(findCUDA - CMake)[https://cmake.org/cmake/help/v3.0/module/FindCUDA.html]
