#include "render_kernel.h"

#include <cstdlib>

#include "basic_math_3d.hpp"
#include "bvh.hpp"
#include "camera.hpp"
#include "cuda_definitions.h"
#include "log_macros.h"
#include "material.hpp"
#include "ray.hpp"
#include "renderer.hpp"
#include "scene.hpp"

using namespace renderer;

// Globals
surface<void, cudaSurfaceType2D> screen_surface;
unsigned int sample_count;

__global__ void visualizeDeviceBlocks(dim3 screen_size)
{
    // Picks a random color for each kernel block on the screen

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= screen_size.x || j >= screen_size.y)
        return; // Out of texture bounds

    auto threadId = blockIdx.x * blockDim.y + blockIdx.y;

    curandState randState;
    curand_init(threadId, 0, 0, &randState);

    surf2Dwrite(make_float4(curand_uniform(&randState),
                            curand_uniform(&randState),
                            curand_uniform(&randState), 1),
                screen_surface, i * sizeof(float4), j);
}

__global__ void cameraRayKernel(
    curandState * rand_state, RayNode * rays, Camera cam)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= cam.width || j >= cam.height)
        return; // Out of texture bounds

    auto index = i * cam.width + j;
    curandState randState = rand_state[index];

    const float x = (i + curand_uniform(&randState)) / float(cam.width);
    const float y = (j + curand_uniform(&randState)) / float(cam.height);

    rays[index].ray = cam.generateRay(x, y);
    rays[index].color = Vector3(1);
    rand_state[index] = randState;
}

__global__ void renderBounceKernel(
    curandState * rand_state, RayNode * rays, Camera cam, BVH bvh,
    Material * materials, const unsigned int n_samples,
    const unsigned int n_bounces, const unsigned int bounce)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= cam.width || j >= cam.height)
        return; // Out of texture bounds

    auto index = i * cam.width + j;
    curandState randState = rand_state[index];

    Ray ray = rays[index].ray;
    Vector3 sample_color = rays[index].color;
    Vector3 emission;

    Intersection intersection;

    if (bvh.intersect(ray, intersection))
    {
        Vector3 attenuation;
        Ray scattered_ray;

        auto material = materials[intersection.material_id];
        auto scatters = material.scatter(
            ray, intersection.point, intersection.normal, attenuation,
            scattered_ray, &randState
        );
        emission = material.emit(ray);

        sample_color *= attenuation;
        ray = scattered_ray;
    }

    rays[index].ray = ray;
    rays[index].color = sample_color;

    if(n_bounces == bounce)
    {
        sample_color *= emission;

        float4 current_color;
        surf2Dread(&current_color, screen_surface, i * sizeof(float4), j);

        sample_color += Vector3(current_color.x, current_color.y,
                                current_color.z) * (n_samples - 1);
        sample_color /= n_samples;

        clamp(sample_color, 0.0f, 1.0f);

        surf2Dwrite(make_float4(sample_color.x,
                                sample_color.y,
                                sample_color.z, 1),
                    screen_surface, i * sizeof(float4), j);
    }

    rand_state[index] = randState;
}

void
renderKernelCall()
{
    sample_count++;

    dim3 blockDim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3  gridDim((scene.camera.width  - 1)/blockDim.x + 1,
                  (scene.camera.height - 1)/blockDim.y + 1, 1);

    cudaErrorCheck(cudaBindSurfaceToArray(
        screen_surface,
        kernel_data.graphics_array)
    );

    float total_time = 0;

    CUDA_MEASURE_TIME_WRAPPER_BEGIN;
    cameraRayKernel<<<gridDim, blockDim>>>(
        kernel_data.rand_state, kernel_data.rays, scene.camera
    );
    CUDA_MEASURE_TIME_WRAPPER_END(total_time);
    CUDA_SYNC_AND_CHECK_KERNEL;

    for(int bounce = 1; bounce <= options.n_bounces; ++bounce)
    {
        CUDA_MEASURE_TIME_WRAPPER_BEGIN;
        renderBounceKernel<<<gridDim, blockDim>>>(
            kernel_data.rand_state, kernel_data.rays, scene.camera,
            kernel_data.bvh, kernel_data.materials, sample_count,
            options.n_bounces, bounce
        );
        CUDA_MEASURE_TIME_WRAPPER_END(total_time);
        CUDA_SYNC_AND_CHECK_KERNEL;
    }

    INFO("Total time: " << total_time);
}
