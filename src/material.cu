#include "material.hpp"

GPU_ONLY Vector3
random_in_unit_sphere(curandState* rand_state)
{
    Vector3 p;
    do {
        p = 2 * Vector3(curand_uniform(rand_state),
                        curand_uniform(rand_state),
                        curand_uniform(rand_state)) - Vector3(1);
    } while(p.lengthSquare() >= 1.0f);

    return p;
}

GPU_ONLY Vector3
random_cosine_direction(curandState* rand_state)
{
    // Generates a random direction in a hemisphere

    // x = cos(phi) * sin(theta)
    // y = sin(phi) * sin(theta)
    // z = cos(theta)

    // Using sin^2 + cos^2 = 1  => sin = sqrt(1 - cos^2) & cos = sqrt(1 - sin^2)

    // Phi is a pencentage of 360
    float phi = 2 * M_PI * curand_uniform(rand_state);

    // Theta will vary from 0 to 90 (since we are working with a hemisphere)
    // Its sin will be in [0, 1]
    float sin_theta = curand_uniform(rand_state);

    return Vector3(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        sqrt(1 - sin_theta * sin_theta)
    );
}

GPU_ONLY bool
Material::scatter(const Ray& ray, const Vector3& point, const Vector3& normal,
                  Vector3& attenuation, Ray& out_ray,
                  curandState* rand_state) const
{
    switch(type) {
    case LAMBERTIAN:
        // Orthonormal basis oriented by the normal
        Vector3 w, u, v;
        w = normal;
        // Pick a possible orthogonal vector based on the dominant axis
        // if (w.x > 0.9) [0, 1, 0] // if x is dominant pick the y axis
        // else [1, 0, 0]
        v = w.cross(Vector3(!(fabs(w.x) > 0.9), !!(fabs(w.x) > 0.9), 0));
        u = w.cross(v);

        auto rand_dir = random_cosine_direction(rand_state);

        Vector3 d = u * rand_dir.x + v * rand_dir.y + w * rand_dir.z;

        out_ray = Ray(point, d.normalize());
        attenuation = albedo;
        return true;

    case METAL:
        Vector3 reflected = ray.d.normalize().reflect(normal);
        out_ray = Ray(point, reflected + modifier *
                                         random_in_unit_sphere(rand_state));
        attenuation = albedo;
        return out_ray.d.dot(normal) > 0;

    case DIELETRIC:
        attenuation = Vector3(1);
        auto ray_d = ray.d.normalize();

        float n1 = 1, n2 = modifier;
        auto norm = normal;

        if (ray_d.dot(norm) > 0)
        {
            norm = -norm;
            n1 = modifier;
            n2 = 1;
        }

        auto reflect_probability = ray_d.reflectance(norm, n1, n2);

        if (curand_uniform(rand_state) <= reflect_probability)
            out_ray = Ray(point, ray_d.reflect(norm));
        else
            out_ray = Ray(point, ray_d.refract(norm, n1, n2));

        return true;

    case DIFFUSE_LIGHT:
        attenuation = albedo;
        out_ray = ray;
        return true;
    default:
        break;
    };

    return false;
}

HOST_GPU Vector3
Material::emit(const Ray &) const
{
    if (type == DIFFUSE_LIGHT) return albedo;
    return Vector3(0);
}

HOST_GPU bool
Material::operator==(const Material& material) const
{
    return    type == material.type
           && albedo == material.albedo
           && modifier == material.modifier;
}

HOST_GPU bool
Material::operator<(const Material& material) const
{
    if (type == material.type)
    {
        if (albedo == material.albedo)
            return modifier < material.modifier;
        return albedo < material.albedo;
    }
    return type < material.type;
}
