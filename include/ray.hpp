#ifndef __RENDERER_RAY_HPP__
#define __RENDERER_RAY_HPP__

#include "basic_math_3d.hpp"
#include "cuda_definitions.h"

class Ray {

public:
    HOST_GPU Ray(const Vector3 &_o=Vector3(),
                 const Vector3 &_d=Vector3())
                 : o(_o), d(_d) {}
    HOST_GPU Ray(const Ray &r) : Ray(r.o, r.d) {}

    HOST_GPU inline Ray& operator=(const Ray &r)
    {
        o = r.o; d = r.d;
        return *this;
    }

    HOST_GPU inline Vector3 origin()    const { return o; }
    HOST_GPU inline Vector3 direction() const { return d; }

    HOST_GPU inline Vector3 operator()(float t) const
    {
        return o + d * t; // Parametric ray form
    }

    inline Ray operator*(const Matrix4x4 & M) const
    {
        return Ray(multiplyPoint(M, o), M * d);
    }


    // Fields
    Vector3 o; // origin
    Vector3 d; // direction
};

inline Ray operator*(const Matrix4x4 & M, const Ray & r)
{
    return r * M;
}

struct RayNode {
    HOST_GPU RayNode(Ray _ray, Vector3 _color, unsigned int _node_index)
        : ray(_ray), color(_color), node_index(_node_index) {};

    HOST_GPU RayNode(unsigned int _node_index=0)
        : RayNode(Ray(), Vector3(), _node_index) {};

    HOST_GPU RayNode(const RayNode& raynode)
        : RayNode(raynode.ray, raynode.color, raynode.node_index) {};

    HOST_GPU inline RayNode& operator=(const RayNode& r)
    {
        ray = r.ray; color = r.color; node_index = r.node_index;
        return *this;
    }

    Ray ray;
    Vector3 color; // TODO: Remove color
    unsigned int node_index;
    char pad[24];

    HOST_GPU inline bool operator<(const RayNode& r) const
    {
        return node_index < r.node_index;
    }
};

#endif
