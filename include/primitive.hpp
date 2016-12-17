#ifndef __RENDERER_PRIMITIVE_HPP__
#define __RENDERER_PRIMITIVE_HPP__

#include "aabb.hpp"
#include "basic_math_3d.hpp"
#include "cuda_definitions.h"
#include "intersection.hpp"
#include "material.hpp"
#include "ray.hpp"

class Triangle {

public:
    HOST_GPU Triangle(const Vector3& _p1=Vector3(),
                      const Vector3& _p2=Vector3(),
                      const Vector3& _p3=Vector3(),
                      unsigned int _material_id=0);

    HOST_GPU bool intersect(const Ray& ray, Intersection& intersection) const;

    HOST_GPU AABB aabb() const;

    Vector3 p1, p2, p3;
    unsigned int material_id;
};

class Sphere {

public:
    HOST_GPU Sphere();
    HOST_GPU Sphere(const Vector3& _position, float _radius);

    HOST_GPU bool intersect(const Ray& ray, Intersection& intersection) const;

private:
    Vector3 position;
    float radius;
};

#endif
