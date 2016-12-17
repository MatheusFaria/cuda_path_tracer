#ifndef __RENDERER_INTERSECTION_HPP__
#define __RENDERER_INTERSECTION_HPP__

#include <cfloat>

#include "basic_math_3d.hpp"
#include "cuda_definitions.h"
#include "material.hpp"

#define INTERSECTION_OFFSET 0.001f

struct Intersection {
    HOST_GPU Intersection()
        : t_min(INTERSECTION_OFFSET), t_max(FLT_MAX)
    {};

    Vector3 point;
    Vector3 normal;
    float t_min, t_max;
    unsigned int material_id;
};

#endif
