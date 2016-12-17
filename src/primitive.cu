#include "primitive.hpp"


HOST_GPU Triangle::Triangle(const Vector3& _p1, const Vector3& _p2,
                            const Vector3& _p3, unsigned int _material_id)
    : p1(_p1), p2(_p2), p3(_p3), material_id(_material_id) {}

HOST_GPU bool
Triangle::intersect(const Ray& ray, Intersection& intersection) const
{
    Vector3 e1 = p2 - p1;
    Vector3 e2 = p3 - p1;
    Vector3 s1 = ray.d.cross(e2);

    float div = s1.dot(e1);

    if(div == 0.f) return false;

    float invDiv = 1.f/div;

    Vector3 s = ray.o - p1;
    float b1 = s.dot(s1) * invDiv;

    if(b1 < 0.f || b1 > 1.f) return false;

    Vector3 s2 = s.cross(e1);
    float b2 = ray.d.dot(s2) * invDiv;
    if(b2 < 0.f || b1 + b2 > 1.f) return false;

    float t = e2.dot(s2) * invDiv;

    if(t < intersection.t_min || t > intersection.t_max) return false;

    intersection.point = ray(t);
    intersection.normal = e1.cross(e2).normalize();
    intersection.t_max = t;
    intersection.material_id = material_id;

    // Barycentric Coordinates Vector3(b1, b2, 1 - b1 - b2)

    return true;
}

HOST_GPU AABB
Triangle::aabb() const
{
    return AABB(p1, p2).join(p3);
}

// Sphere

HOST_GPU Sphere::Sphere() : Sphere(Vector3(), 0.f) {}

HOST_GPU Sphere::Sphere(const Vector3& _position, float _radius)
    : position(_position), radius(_radius) {}

HOST_GPU bool
Sphere::intersect(const Ray& ray, Intersection& intersection) const
{
    // Sphere Intersection
    Vector3 origin_to_center = ray.o - position;

    float A = ray.d.dot(ray.d);
    float B = 2.f * ray.d.dot(origin_to_center);
    float C = origin_to_center.dot(origin_to_center) - radius * radius;

    float discriminant = B*B - 4*A*C;

    if(discriminant < 0) return false;

    float t;
    if(discriminant == 0)
        t = -B/(2.f * A);
    else
    {
        float t0, t1;

        float sqrt_discriminant = sqrt(discriminant);

        float q;
        if(B < 0) q = -.5f * (B - sqrt_discriminant);
        else      q = -.5f * (B + sqrt_discriminant);

        t0 = q / A;
        t1 = C / q;

        t = t0 < t1 ? t0 : t1;
        if(t < intersection.t_min || t > intersection.t_max)
            t = t0 > t1 ? t0 : t1;
    }

    if(t < intersection.t_min || t > intersection.t_max)
        return false;

    intersection.point = ray(t);
    intersection.normal = (intersection.point - position).normalize();
    intersection.t_max = t;

    return true;
}
