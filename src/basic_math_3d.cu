#include "basic_math_3d.hpp"


HOST_GPU Vector3::Vector3() : Vector3(0, 0, 0) {}
HOST_GPU Vector3::Vector3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
HOST_GPU Vector3::Vector3(float v) : Vector3(v, v, v) {}
HOST_GPU Vector3::Vector3(const Vector3 & v) : Vector3(v.x, v.y, v.z) {}

HOST_GPU Vector3 &
Vector3::operator=(const Vector3 & v)
{
    x = v.x; y = v.y; z = v.z;
    return *this;
}

HOST_GPU Vector3
Vector3::operator+(const Vector3 & v) const
{
    return Vector3(x + v.x, y + v.y, z + v.z);
}
HOST_GPU Vector3 &
Vector3::operator+=(const Vector3 & v)
{
    x += v.x; y += v.y; z += v.z;
    return *this;
}

HOST_GPU Vector3
Vector3::operator-(const Vector3 & v) const
{
    return Vector3(x - v.x, y - v.y, z - v.z);
}
HOST_GPU Vector3 &
Vector3::operator-=(const Vector3 & v)
{
    x -= v.x; y -= v.y; z -= v.z;
    return *this;
}

HOST_GPU Vector3
Vector3::operator*(float scale_factor) const
{
    return Vector3(x * scale_factor, y * scale_factor, z * scale_factor);
    // Reverse operation defined after this class
}
HOST_GPU Vector3 &
Vector3::operator*=(float scale_factor)
{
    x *= scale_factor; y *= scale_factor; z *= scale_factor;
    return *this;
    // No reverse operation available
}

HOST_GPU Vector3
Vector3::operator*(const Vector3 & v) const
{
    return Vector3(x*v.x, y*v.y, z*v.z);
}
HOST_GPU Vector3 &
Vector3::operator*=(const Vector3 & v)
{
    x *= v.x; y *= v.y; z *= v.z;
    return *this;
}

HOST_GPU Vector3
Vector3::operator/(float divisor) const
{
    float scale_factor = 1.f / divisor;
    return (*this) * scale_factor;
    // No reverse operation available
}
HOST_GPU Vector3 &
Vector3::operator/=(float divisor)
{
    float scale_factor = 1.f / divisor;
    (*this) *= scale_factor;
    return *this;
    // No reverse operation available
}

HOST_GPU Vector3
Vector3::operator-() const
{
    return Vector3(-x, -y, -z);
}

HOST_GPU bool
Vector3::operator==(const Vector3 & v) const
{
    return v.x == x && v.y == y && v.z == z;
}
HOST_GPU bool
Vector3::operator!=(const Vector3 & v) const
{
    return !((*this) == v);
}

HOST_GPU bool
Vector3::operator<(const Vector3 & v) const
{
    if (x == v.x)
    {
        if (y == v.y)
            return z < v.z;
        return y < v.y;
    }
    return x < v.x;
}
HOST_GPU bool
Vector3::operator>(const Vector3 & v) const
{
    return v < (*this);
}

HOST_GPU Vector3
Vector3::cross(const Vector3 & v) const
{
    return Vector3((y * v.z) - (z * v.y),
                   (z * v.x) - (x * v.z),
                   (x * v.y) - (y * v.x));
}

HOST_GPU float
Vector3::dot(const Vector3 & v) const
{
    return x * v.x + y * v.y + z * v.z;
}

HOST_GPU float
Vector3::lengthSquare() const
{
    return x * x + y * y + z * z;
}

HOST_GPU float
Vector3::length() const
{
    return sqrt(lengthSquare());
}

HOST_GPU Vector3
Vector3::normalize() const
{
    auto len = length();
    if (len == 0) return INVALID_VECTOR3;

    return (*this) / length();
}

HOST_GPU Vector3
Vector3::refract(const Vector3 & normal, float n1, float n2) const
{
    // Reference: http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf

    // normal vector needs to be unit length
    // *this  vector needs to be unit length

    float n = n1 / n2;
    float cosI = -normal.dot(*this);

    float sinT2 = n * n * (1.0f - cosI * cosI);
    if(sinT2 > 1.0f) return INVALID_VECTOR3; // Total Internal Reflection

    float cosT = sqrt(1.0f - sinT2);
    return (*this) * n + normal * (n * cosI - cosT);
}

HOST_GPU Vector3
Vector3::reflect(const Vector3 & normal) const
{
    // normal vector needs to be unit length
    // *this  vector needs to be unit length

    return *this + normal * (-2.0f * normal.dot(*this));
}

HOST_GPU float
Vector3::reflectance(const Vector3 & normal, float n1, float n2) const
{
    // Reference: http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf

    // normal vector needs to be unit length
    // *this  vector needs to be unit length

    float r0 = (n1 - n2) / (n1 + n2);
    r0 *= r0;

    float cosI = -(normal.dot(*this));

    if (n1 > n2)
    {
        const float n = n1 / n2;
        const float sinT2 = n * n * (1.0f - cosI * cosI);
        if (sinT2 > 1.0f) return 1.0f; // Total Internal Reflection
        cosI = sqrt(1.0f - sinT2);
    }

    const float c = 1.0f - cosI;
    return r0 + (1.0f - r0) * c * c * c * c * c;
}

HOST_GPU float
Vector3::transmitance(const Vector3 & normal, float n1, float n2) const
{
    return 1.0f - reflectance(normal, n1, n2);
}

HOST_GPU float
Vector3::operator[](int index) const
{
    return (&x)[index];
}
HOST_GPU float &
Vector3::operator[](int index)
{
    return (&x)[index];
}

HOST_GPU Vector3
operator*(float scale_factor, const Vector3 & v)
{
    return v * scale_factor;
}

// Functions that change the vector instance instead of creating a new object

HOST_GPU void
cross(Vector3 & v1, const Vector3 & v2)
{
    auto x = v1.x, y = v1.y, z = v1.z;
    v1.x = (y * v2.z) - (z * v2.y);
    v1.y = (z * v2.x) - (x * v2.z);
    v1.z = (x * v2.y) - (y * v2.x);
}

HOST_GPU bool
normalize(Vector3 & v1)
{
    auto len = v1.length();
    if (len == 0) return false;

    len = 1.0f/len;
    v1.x *= len;
    v1.y *= len;
    v1.z *= len;

    return true;
}

HOST_GPU void
clamp(Vector3 & v1, float a, float b)
{
    v1.x = b < v1.x ? b : v1.x;
    v1.y = b < v1.y ? b : v1.y;
    v1.z = b < v1.z ? b : v1.z;

    v1.x = a > v1.x ? a : v1.x;
    v1.y = a > v1.y ? a : v1.y;
    v1.z = a > v1.z ? a : v1.z;
}

HOST_GPU bool
refract(Vector3 & incident, const Vector3 & normal, float n1, float n2)
{
    // normal vector needs to be unit length
    // incident vector needs to be unit length

    float n = n1 / n2;
    float cosI = -normal.dot(incident);

    float sinT2 = n * n * (1.0f - cosI * cosI);
    if(sinT2 > 1.0f) return false; // Total Internal Reflection

    float cosT = sqrt(1.0f - sinT2);
    incident *= n;
    incident += normal * (n * cosI - cosT);

    return true;
}

HOST_GPU void
reflect(Vector3 & incident, const Vector3 & normal)
{
    // normal vector needs to be unit length
    // *this  vector needs to be unit length

    incident -= normal * (2.0f * normal.dot(incident));
}
