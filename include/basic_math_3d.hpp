#ifndef __BASIC_MATH_3D_MATH_HPP__
#define __BASIC_MATH_3D_MATH_HPP__

#include <algorithm>
#include <cfloat>
#include <istream>
#include <ostream>

#ifndef __CUDACC__
#include <cmath>
#endif

#include "cuda_definitions.h"

#ifndef M_PI
#define M_PI 3.14159265359
#endif

#ifndef radians
#define radians(x) ((x)*M_PI/180.0)
#endif

// ============================ Vector3 Class =================================

/**
 * Vector3 defines a 3D vector with arithmetics and specific vector
 *    operations
 */

// Constant Vector3

// This vector is returned when a function acheives an invalid state
#ifndef INVALID_VECTOR3
#define INVALID_VECTOR3 Vector3(-FLT_MAX)
#endif
/*
 *  It is defined as a macro because it is needed inside the Vector3 class.
 *    And since this is a header only library, it can't be initialized as a
 *    class static const field, neither predefined before the class definition
 */

class Vector3 {

public:
    HOST_GPU Vector3();
    HOST_GPU Vector3(float _x, float _y, float _z);
    HOST_GPU Vector3(float v);
    HOST_GPU Vector3(const Vector3 & v);

    HOST_GPU Vector3 & operator=(const Vector3 & v);

    // Basic Arithmetics Operators

    HOST_GPU Vector3 operator+(const Vector3 & v) const;
    HOST_GPU Vector3 & operator+=(const Vector3 & v);

    HOST_GPU Vector3 operator-(const Vector3 & v) const;
    HOST_GPU Vector3 & operator-=(const Vector3 & v);

    HOST_GPU Vector3 operator*(float scale_factor) const;
    HOST_GPU Vector3 & operator*=(float scale_factor);

    // Component-wise multiplication
    HOST_GPU Vector3 operator*(const Vector3 & v) const;
    HOST_GPU Vector3 & operator*=(const Vector3 & v);

    HOST_GPU Vector3 operator/(float divisor) const;
    HOST_GPU Vector3 & operator/=(float divisor);

    HOST_GPU Vector3 operator-() const;

    // Logic Operators
    HOST_GPU bool operator==(const Vector3 & v) const;
    HOST_GPU bool operator!=(const Vector3 & v) const;

    HOST_GPU bool operator<(const Vector3 & v) const;
    HOST_GPU bool operator>(const Vector3 & v) const;

    // Specific Operations
    HOST_GPU Vector3 cross(const Vector3 & v) const;
    HOST_GPU float dot(const Vector3 & v) const;

    HOST_GPU float lengthSquare() const;
    HOST_GPU float length() const;
    HOST_GPU Vector3 normalize() const;

    HOST_GPU Vector3 reflect(const Vector3 & normal) const;
    HOST_GPU Vector3 refract(const Vector3 & normal, float n1, float n2) const;
    HOST_GPU float reflectance(const Vector3 & normal, float n1, float n2) const;
    HOST_GPU float transmitance(const Vector3 & normal, float n1, float n2) const;

    // Element access operators
    HOST_GPU float operator[](int index) const;
    HOST_GPU float & operator[](int index);

    // Fields
    float x, y, z;

private:

    // Outputs a Vector3
    friend std::ostream & operator<<(std::ostream & os, const Vector3 & v)
    {
        os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
        return os;
    }

    // Input a Vector3
    friend std::istream & operator>>(std::istream & is, Vector3 & v)
    {
        is >> v.x >> v.y >> v.z;
        return is;
    }
};

// Reverse Operations

HOST_GPU Vector3 operator*(float scale_factor, const Vector3 & v);

// Functions that change the vector instance instead of creating a new object

HOST_GPU void cross(Vector3 & v1, const Vector3 & v2);
HOST_GPU bool normalize(Vector3 & v1);
HOST_GPU void clamp(Vector3 & v1, float a, float b);
HOST_GPU bool refract(Vector3 & incident, const Vector3 & normal, float n1,
                    float n2);
HOST_GPU void reflect(Vector3 & incident, const Vector3 & normal);



// ============================= Matrix Class =================================

// Constant Matrix4x4

// This matrix is returned when a function acheives an invalid state
#ifndef INVALID_MATRIX4X4
#define INVALID_MATRIX4X4 Matrix4x4(-FLT_MAX)
/*
 *  It is defined as a macro because it is needed inside the Matrix4x4 class.
 *    And since this is a header only library, it can't be initialized as a
 *    class static const field, neither predefined before the class definition
 */
#endif

class Matrix4x4 {

public:
    Matrix4x4(float m00, float m01, float m02, float m03,
              float m10, float m11, float m12, float m13,
              float m20, float m21, float m22, float m23,
              float m30, float m31, float m32, float m33)
    {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
        m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
    }

    Matrix4x4(): Matrix4x4(1, 0, 0, 0,
                           0, 1, 0, 0,
                           0, 0, 1, 0,
                           0, 0, 0, 1) {}

    Matrix4x4(const float value) : Matrix4x4(value, value, value, value,
                                             value, value, value, value,
                                             value, value, value, value,
                                             value, value, value, value) {}

    Matrix4x4(const Matrix4x4 & M)
    {
        for(int i = 0; i < 16; ++i)
            (*this)[i] = M[i];
    }

    inline Matrix4x4 & operator=(const Matrix4x4 & M)
    {
        for(int i = 0; i < 16; ++i)
            (*this)[i] = M[i];

        return (*this);
    }


    // Basic Arithmetics Operators
    Matrix4x4 operator+(const Matrix4x4 & B) const
    {
        Matrix4x4 C;
        for(int i = 0; i < 16; ++i)
            C[i] = (*this)[i] + B[i];

        return C;
    }
    Matrix4x4 & operator+=(const Matrix4x4 & B)
    {
        for(int i = 0; i < 16; ++i)
            (*this)[i] += B[i];

        return *this;
    }

    Matrix4x4 operator-(const Matrix4x4 & B) const
    {
        Matrix4x4 C;
        for(int i = 0; i < 16; ++i)
            C[i] = (*this)[i] - B[i];

        return C;
    }
    Matrix4x4 & operator-=(const Matrix4x4 & B)
    {
        for(int i = 0; i < 16; ++i)
            (*this)[i] -= B[i];

        return *this;
    }

    Matrix4x4 operator-() const
    {
        Matrix4x4 C;
        for(int i = 0; i < 16; ++i)
            C[i] = -(*this)[i];

        return C;
    }

    Matrix4x4 operator*(const Matrix4x4 & B) const
    {
        Matrix4x4 C(0);
        for(int i = 0; i < 4; ++i)
            for(int j = 0; j < 4; ++j)
                for(int k = 0; k < 4; ++k)
                    C[i * 4 + j] += m[i][k] * B[k * 4 + j];

        return C;
    }
    Matrix4x4 & operator*=(const Matrix4x4 & B)
    {
        Matrix4x4 C(0);
        for(int i = 0; i < 4; ++i)
            for(int j = 0; j < 4; ++j)
                for(int k = 0; k < 4; ++k)
                    C[i * 4 + j] += m[i][k] * B[k * 4 + j];

        (*this) = C;
        return *this;
    }

    inline Matrix4x4 operator*(const float scale_factor) const
    {
        Matrix4x4 C;

        for(int i = 0; i < 16; ++i)
            C[i] = (*this)[i] * scale_factor;

        return C;
        // inverse operation defined after this class
    }
    inline Matrix4x4 & operator*=(const float scale_factor)
    {
        for(int i = 0; i < 16; ++i)
            (*this)[i] *= scale_factor;

        return (*this);
    }

    inline Matrix4x4 operator/(const float scale_factor) const
    {
        return (*this) * (1.0f/scale_factor);
    }
    inline Matrix4x4 & operator/=(const float scale_factor)
    {
        (*this) *= 1.0f/scale_factor;
        return (*this);
    }


    // Logic Operators

    inline bool operator==(const Matrix4x4 & B) const
    {
        for(int i = 0; i < 16; ++i)
            if(B[i] != (*this)[i]) return false;

        return true;
    }
    inline bool operator!=(const Matrix4x4 & B) const
    {
        return !(*this == B);
    }


    // Specific Operations

    inline Matrix4x4 transpose() const
    {
        return Matrix4x4(m[0][0], m[1][0], m[2][0], m[3][0],
                         m[0][1], m[1][1], m[2][1], m[3][1],
                         m[0][2], m[1][2], m[2][2], m[3][2],
                         m[0][3], m[1][3], m[2][3], m[3][3]);
    }

    inline float determinant() const
    {
        auto a = m[0][0] * det3x3(m[1][1], m[1][2], m[1][3],
                                  m[2][1], m[2][2], m[2][3],
                                  m[3][1], m[3][2], m[3][3]);

        auto b = m[0][1] * det3x3(m[1][0], m[1][2], m[1][3],
                                  m[2][0], m[2][2], m[2][3],
                                  m[3][0], m[3][2], m[3][3]);

        auto c = m[0][2] * det3x3(m[1][0], m[1][1], m[1][3],
                                  m[2][0], m[2][1], m[2][3],
                                  m[3][0], m[3][1], m[3][3]);

        auto d = m[0][3] * det3x3(m[1][0], m[1][1], m[1][2],
                                  m[2][0], m[2][1], m[2][2],
                                  m[3][0], m[3][1], m[3][2]);

        return a - b + c - d;
    }

    inline Matrix4x4 inverse() const
    {
        // Computing the inverse matrix with the determinant and the adjugate
        //    A^-1 = 1/det(A) * adj(A)
        //    adj(A) = transpose(C)
        //         C is the cofactor matrix

        Matrix4x4 M; // Adjugate Matrix ( adj(*this) )

        // Lines indices. L[3] is not used.
        int L[4] = {1, 2, 3, -111};
        int negative = 1;

        for (int i = 0; i < 4; ++i)
        {
            // Rows indices. R[3] is not used.
            int R[4] = {1, 2, 3, -111};

            for (int j = 0; j < 4; ++j)
            {
                M.m[j][i] = negative * det3x3(
                    m[L[0]][R[0]], m[L[0]][R[1]], m[L[0]][R[2]],
                    m[L[1]][R[0]], m[L[1]][R[1]], m[L[1]][R[2]],
                    m[L[2]][R[0]], m[L[2]][R[1]], m[L[2]][R[2]]
                );

                negative *= -1;
                R[j]--;
            }

            negative *= -1;
            L[i]--;
        }

        auto det = m[0][0] * M.m[0][0] +
                   m[0][1] * M.m[1][0] +
                   m[0][2] * M.m[2][0] +
                   m[0][3] * M.m[3][0];

        if (det == 0) return INVALID_MATRIX4X4;

        return M / det;
    }

    // Element access operators

    inline float operator[](int index) const
    {
        return m[index / 4][index % 4];
    }
    inline float & operator[](int index)
    {
        return m[index / 4][index % 4];
    }

    // Fields
    float m[4][4];

private:
    friend std::ostream & operator<<(std::ostream & os,
                                    const Matrix4x4 & matrix)
    {
        for (int i = 0; i < 4; ++i)
        {
            if(i) os << "\n";

            os << "[";
            for (int j = 0; j < 4; ++j)
            {
                if(j) os << ", ";
                os << matrix.m[i*4 + j];
            }
            os << "]";
        }

        return os;
    }

    // Input a Matrix4x4
    friend std::istream & operator>>(std::istream & is, Matrix4x4 & M)
    {
        for(int i = 0; i < 16; ++i)
            is >> M[i];
        return is;
    }

    inline static float det3x3(float m00, float m01, float m02,
                               float m10, float m11, float m12,
                               float m20, float m21, float m22)
    {
        return ((m00 * m11 * m22) + (m01 * m12 * m20) + (m02 * m10 * m21)
              - (m20 * m11 * m02) - (m21 * m12 * m00) - (m22 * m10 * m01));
    }
};


// Reverse operations

inline Matrix4x4 operator*(const float scale_factor, const Matrix4x4 & M)
{
    return M * scale_factor;
}


// Extra operations

inline Vector3 operator*(const Matrix4x4 & M, const Vector3 & v)
{
    // In this case the Vector3 is like [x, y, z, 0], and the last component
    // is disconsidered

    // If you want this component to be considered, use the multiplyPoint

    return Vector3(
        M.m[0][0] * v.x + M.m[0][1] * v.y + M.m[0][2] * v.z,
        M.m[1][0] * v.x + M.m[1][1] * v.y + M.m[1][2] * v.z,
        M.m[2][0] * v.x + M.m[2][1] * v.y + M.m[2][2] * v.z
    );
}


// Functions that change the matrix instance instead of creating a new object

inline void transpose(Matrix4x4 & M)
{
    for(int i = 0; i < 4; ++i)
        for(int j = i; j < 4; ++j)
            std::swap(M.m[i][j], M.m[j][i]);
}

inline bool inverse(Matrix4x4 & M)
{
    M = M.inverse();
    if(M == INVALID_MATRIX4X4) return false;
    return true;
}


// Extra methods

inline Vector3 multiplyPoint(const Matrix4x4 & M, const Vector3 & v)
{
    // This functions interpreters the Vector3 as a 4-component vector,
    // where the forth component is 1. [x, y, z, 1]
    // The homogeneaus divide is applied to the resulting vector

    Vector3 Mv(
        M.m[0][0] * v.x + M.m[0][1] * v.y + M.m[0][2] * v.z + M.m[0][3],
        M.m[1][0] * v.x + M.m[1][1] * v.y + M.m[1][2] * v.z + M.m[1][3],
        M.m[2][0] * v.x + M.m[2][1] * v.y + M.m[2][2] * v.z + M.m[2][3]
    );

    auto w = M.m[3][0] * v.x +
             M.m[3][1] * v.y +
             M.m[3][2] * v.z +
             M.m[3][3];

    return Mv / w; // homogeneaus divide
}



// ============================ Transform Methods =============================

inline Matrix4x4 translate(float x, float y, float z)
{
    return Matrix4x4(1, 0, 0, x,
                     0, 1, 0, y,
                     0, 0, 1, z,
                     0, 0, 0, 1);
}
inline Matrix4x4 translate(const Vector3 & delta)
{
    return translate(delta.x, delta.y, delta.z);
}

inline Matrix4x4 scale(float x, float y, float z)
{
    return Matrix4x4(x, 0, 0, 0,
                     0, y, 0, 0,
                     0, 0, z, 0,
                     0, 0, 0, 1);
}
inline Matrix4x4 scale(const Vector3 & delta)
{
    return scale(delta.x, delta.y, delta.z);
}

inline Matrix4x4 rotate(float angle, const Vector3 & axis)
{
    // axis should be unit length

    angle = radians(angle);
    auto cos_a = cos(angle), sin_a = sin(angle);
    auto cos_a_1 = 1 - cos_a;

    return Matrix4x4(
        cos_a + axis.x * axis.x * cos_a_1,
        axis.x * axis.y * cos_a_1 - axis.z * sin_a,
        axis.x * axis.z * cos_a_1 + axis.y * sin_a,
        0,

        axis.y * axis.x * cos_a_1 + axis.z * sin_a,
        cos_a + axis.y * axis.y * cos_a_1,
        axis.y * axis.z * cos_a_1 - axis.x * sin_a,
        0,

        axis.z * axis.x * cos_a_1 - axis.y * sin_a,
        axis.z * axis.y * cos_a_1 + axis.x * sin_a,
        cos_a + axis.z * axis.z * cos_a_1,
        0,

        0, 0, 0, 1
    );
}
inline Matrix4x4 rotateX(float angle)
{
    angle = radians(angle);
    auto cos_a = cos(angle), sin_a = sin(angle);
    return Matrix4x4(1,     0,      0, 0,
                     0, cos_a, -sin_a, 0,
                     0, sin_a,  cos_a, 0,
                     0,     0,      0, 1);
}
inline Matrix4x4 rotateY(float angle)
{
    angle = radians(angle);
    auto cos_a = cos(angle), sin_a = sin(angle);
    return Matrix4x4( cos_a, 0, sin_a, 0,
                          0, 1,     0, 0,
                     -sin_a, 0, cos_a, 0,
                          0, 0,     0, 1);
}
inline Matrix4x4 rotateZ(float angle)
{
    angle = radians(angle);
    auto cos_a = cos(angle), sin_a = sin(angle);
    return Matrix4x4(cos_a, -sin_a, 0, 0,
                     sin_a,  cos_a, 0, 0,
                         0,      0, 1, 0,
                         0,      0, 0, 1);
}

inline Matrix4x4 lookAt(const Vector3 & eye, const Vector3 & center,
                        const Vector3 & up_vector)
{
    Vector3 view = eye - center;
    if (!normalize(view)) return INVALID_MATRIX4X4;

    Vector3 strafe = up_vector.cross(view);
    if (!normalize(strafe)) return INVALID_MATRIX4X4;

    Vector3 up = view.cross(strafe);

    return Matrix4x4(strafe.x, strafe.y, strafe.z, -(eye.dot(strafe)),
                         up.x,     up.y,     up.z, -(eye.dot(    up)),
                       view.x,   view.y,   view.z, -(eye.dot(  view)),
                            0,        0,        0,                 1);
}

inline Matrix4x4 perspectiveProjection(float fov, float aspect,
                                       float _near, float _far)
{
    auto far_near = 1.0f / (_far - _near);
    auto a = -(_far + _near)* far_near;
    auto b = -(2 * _far * _near) * far_near;
    auto S = 1.f / tanf(radians(fov) * 0.5f);

    return Matrix4x4(S / aspect, 0,  0, 0,
                              0, S,  0, 0,
                              0, 0,  a, b,
                              0, 0, -1, 0);
}


// ============================= Util Functions ===============================

/**
    Remaps the value v ranged between [old0, old1] to [new0, new1]

    @param v value to be remaped
    @param old0 start from the source range
    @param old1 end of the source range
    @param new0 start from the destiny range
    @param new1 end of the destiny range
    @return a remaped value between the destiny range
*/
template <typename T>
inline T remap(const T & v, const T & old0, const T & old1,
                            const T & new0, const T & new1)
{
    auto d_old0 = static_cast<double>(old0);
    auto d_old1 = static_cast<double>(old1);

    double percentage = 0.0;
    if (std::abs(d_old1 - d_old0) > DBL_EPSILON)
        percentage = (static_cast<double>(v) - d_old0) / (d_old1 - d_old0);

    return static_cast<T>(percentage * (new1 - new0) + new0);
}

template <typename T>
inline T clamp(const T & v, const T & a, const T & b)
{
    return std::max(a, std::min(v, b));
}

#endif
