#ifndef __CAMERA_HPP__
#define __CAMERA_HPP__

#include "basic_math_3d.hpp"
#include "cuda_definitions.h"
#include "ray.hpp"

class Camera {

public:
    HOST_GPU Camera();
    HOST_GPU Camera(unsigned int width, unsigned int height,
                    float fov,
                    Vector3 eye, Vector3 center, Vector3 up);

    HOST_GPU Ray generateRay(float x, float y) const;

    HOST_GPU unsigned int screen_size() const
    {
        return width * height;
    }

    unsigned int width, height;

private:
    Vector3 origin;      // Where the camera is located
    Vector3 left_corner; // Lower left corner of the image plane
    Vector3 horizontal;  // Horizontal image plane size
    Vector3 vertical;    // Vertical image plane size
};

#endif
