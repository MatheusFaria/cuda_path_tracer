#ifndef __BOUNDING_BOX_HPP__
#define __BOUNDING_BOX_HPP__

#include "basic_math_3d.hpp"
#include "cuda_definitions.h"
#include "ray.hpp"

class AABB {

public:
    HOST_GPU AABB();
    HOST_GPU AABB(const Vector3 &p1, const Vector3 &p2);
    HOST_GPU AABB(const Vector3 &p1);

    HOST_GPU AABB join(const AABB &box);
    HOST_GPU AABB join(const Vector3 &p);

    HOST_GPU Vector3 centroid() const;

    HOST_GPU bool intersect(const Ray &ray, float *tmin, float *tmax) const;

    Vector3 lower_bound;
    Vector3 upper_bound;
};

#endif
