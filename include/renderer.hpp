#ifndef __RENDERER_HPP__
#define __RENDERER_HPP__

#include <string>

#include "gl_includes.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

#include "bvh.hpp"
#include "material.hpp"
#include "primitive.hpp"
#include "ray.hpp"
#include "scene.hpp"

namespace renderer
{
    struct Options
    {
        Options(std::string title="", unsigned int samples=0,
                unsigned int bounces=0)
            : project_title(title), n_samples(samples), n_bounces(bounces) {}

        std::string project_title;
        unsigned int n_samples;
        unsigned int n_bounces;
    };

    struct KernelData
    {
        BVH bvh;

        Material    * materials;
        cudaArray   * graphics_array;
        RayNode     * rays;
        curandState * rand_state;
        unsigned int * active_rays_count;

        unsigned int n_materials;
    };

    void setup();
    void tearDown();

    void renderLoop();

    void cudaErrorCheck(cudaError_t err);
    void cudaCheckKernelSuccess();
    void glErrorCheck();

    extern Options options;
    extern Scene scene;
    extern KernelData kernel_data;
};

#endif
