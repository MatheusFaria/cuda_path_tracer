# -- JSON 11 --
add_subdirectory(json11)

# -- Tiny OBJ Loader --
add_subdirectory(tiny_obj_loader)

# -- FreeGLUT 3 --
set(FREEGLUT_BUILD_SHARED_LIBS OFF)
add_subdirectory(freeglut3)
set(FreeGLUT3_INCLUDE_DIRS "deps/freeglut3/include" PARENT_SCOPE)

# -- GLEW 2 --
add_subdirectory("glew2")
