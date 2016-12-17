#include "camera.hpp"

HOST_GPU Camera::Camera()
    : Camera(0, 0, 0.f, Vector3(), Vector3(), Vector3()) {}

HOST_GPU Camera::Camera(unsigned int _width, unsigned int _height,
               float fov,
               Vector3 eye, Vector3 center, Vector3 up)
    : width(_width), height(_height)
{
    // Perspective
    auto aspect = float(width)/float(height);
    auto half_height = tan(radians(fov)/2);
    auto half_width = aspect * half_height;

    // Look At vector basis
    auto w = (eye - center).normalize();
    auto u = (up.cross(w)).normalize();
    auto v = w.cross(u);

    origin = eye;
    left_corner = origin + -half_width * u + -half_height * v - w;
    horizontal  = 2.0f * half_width * u;
    vertical    = 2.0f * half_height * v;
}

HOST_GPU Ray
Camera::generateRay(float x, float y) const
{
    return Ray(origin, left_corner + x * horizontal + y * vertical - origin);
}
