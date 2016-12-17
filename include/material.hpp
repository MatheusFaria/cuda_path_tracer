#ifndef __RENDERER_MATERIAL_HPP__
#define __RENDERER_MATERIAL_HPP__

#include <curand.h>
#include <curand_kernel.h>

#include "basic_math_3d.hpp"
#include "cuda_definitions.h"
#include "ray.hpp"


enum MaterialTypes
{
    LAMBERTIAN,
    METAL,
    DIELETRIC,
    DIFFUSE_LIGHT,
};

extern GPU_ONLY Vector3 random_in_unit_sphere(curandState* rand_state);
extern GPU_ONLY Vector3 random_cosine_direction(curandState* rand_state);

class Material {
public:
    HOST_GPU Material(MaterialTypes _type=LAMBERTIAN,
                      const Vector3& _albedo=Vector3(),
                      float _modifier=0)
        : type(_type), albedo(_albedo), modifier(_modifier) {}

    GPU_ONLY bool scatter(const Ray& ray, const Vector3& point,
                          const Vector3& normal, Vector3& attenuation,
                          Ray& out_ray, curandState* rand_state) const;

    HOST_GPU Vector3 emit(const Ray& ray) const;

    HOST_GPU bool operator==(const Material& material) const;
    HOST_GPU bool operator<(const Material& material) const;


    MaterialTypes type;
    Vector3 albedo;

    // Can be interpreted as fuzzyness for metals and as the
    // refractive_index for Dieletrics.
    float modifier;

    char pad[12];
};

#endif
