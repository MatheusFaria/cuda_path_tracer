#include "aabb.hpp"

#include <cfloat>

HOST_GPU AABB::AABB()
    : lower_bound(Vector3(FLT_MAX)),
      upper_bound(Vector3(-FLT_MAX)) {}

HOST_GPU AABB::AABB(const Vector3 &p1, const Vector3 &p2)
{
    for (int i = 0; i < 3; ++i)
    {
        upper_bound[i] = (p1[i] > p2[i] ? p1[i] : p2[i]);
        lower_bound[i] = (p1[i] < p2[i] ? p1[i] : p2[i]);
    }
}

HOST_GPU AABB::AABB(const Vector3 &p1)
    : upper_bound(p1), lower_bound(p1) {}

HOST_GPU AABB
AABB::join(const AABB &box)
{
    join(box.lower_bound);
    join(box.upper_bound);

    return *this;
}

HOST_GPU AABB
AABB::join(const Vector3 &p)
{
    lower_bound.x = (lower_bound.x < p.x ? lower_bound.x : p.x);
    lower_bound.y = (lower_bound.y < p.y ? lower_bound.y : p.y);
    lower_bound.z = (lower_bound.z < p.z ? lower_bound.z : p.z);

    upper_bound.x = (upper_bound.x > p.x ? upper_bound.x : p.x);
    upper_bound.y = (upper_bound.y > p.y ? upper_bound.y : p.y);
    upper_bound.z = (upper_bound.z > p.z ? upper_bound.z : p.z);

    return *this;
}

HOST_GPU Vector3
AABB::centroid() const
{
    return (upper_bound + lower_bound)/2.0f;
}

HOST_GPU bool
AABB::intersect(const Ray &ray, float *tmin, float *tmax) const
{
    float t0 = 0, t1 = FLT_MAX;
    bool hitted = true;

    for (int i = 0; i < 3; ++i)
    {
        float inv_ray_d = 1.f/ray.d[i];
        float tNear = (lower_bound[i] - ray.o[i]) * inv_ray_d;
        float tFar  = (upper_bound[i] - ray.o[i]) * inv_ray_d;

        // if(tNear > tFar)
        // {
        //     auto tmp = tNear;
        //     tNear = tFar;
        //     tFar = tmp;
        // }

        bool swp = tNear > tFar;
        auto tmp = tNear;
        tNear = !!swp * tFar + !swp * tNear;
        tFar  = !!swp * tmp  + !swp * tFar;

        t0 = (t0 > tNear ? t0 : tNear);
        t1 = (t1 < tFar  ? t1 : tFar);

        hitted &= !(t0 > t1);
        // if (t0 > t1) return false;
    }

    *tmin = (!!hitted * t0) + (!hitted * (*tmin));
    *tmax = (!!hitted * t1) + (!hitted * (*tmax));

    return hitted;
}
