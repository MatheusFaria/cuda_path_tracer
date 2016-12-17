#include "initialization_kernels.h"

#include "cuda_definitions.h"
#include "log_macros.h"

__global__ void initRandomStatesKernel(
    curandState * rand_state, const Camera cam)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= cam.width || j >= cam.height)
        return; // Out of texture bounds

    auto threadId = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y
                                     * (blockIdx.x + gridDim.x * blockIdx.y));

    auto index = i * cam.width + j;
    curand_init(threadId, 0, 0, &rand_state[index]);
}

void initRandomStates(
    curandState * rand_state, const Camera& cam
)
{
    INFO("Generating random states...");

    dim3 blockDim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3  gridDim((cam.width  - 1)/blockDim.x + 1,
                  (cam.height - 1)/blockDim.y + 1, 1);

    initRandomStatesKernel<<<gridDim, blockDim>>>(rand_state, cam);
    CUDA_SYNC_AND_CHECK_KERNEL;

    INFO("Random states generated!");
}


__global__ void initRaysKernel(
    RayNode * rays, const Camera cam
)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= cam.width || j >= cam.height)
        return; // Out of texture bounds

    auto index = i * cam.width + j;
    rays[index] = RayNode();
}

void initRays(
    RayNode * rays, const Camera& cam
)
{
    INFO("Initializing rays array...");

    dim3 blockDim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3  gridDim((cam.width  - 1)/blockDim.x + 1,
                  (cam.height - 1)/blockDim.y + 1, 1);

    initRaysKernel<<<gridDim, blockDim>>>(rays, cam);
    CUDA_SYNC_AND_CHECK_KERNEL;

    INFO("Rays initialized!");
}
