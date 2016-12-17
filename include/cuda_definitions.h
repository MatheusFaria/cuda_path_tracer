#ifndef __CUDAS_DEFS_H__
#define __CUDAS_DEFS_H__

#ifndef CUDA_BLOCK_SIZE
#define CUDA_BLOCK_SIZE 16
#endif

#ifdef __CUDACC__
#define HOST_GPU  __host__ __device__
#define HOST_ONLY __host__
#define GPU_ONLY  __device__
#else
#define HOST_GPU
#define HOST_ONLY
#define GPU_ONLY
#endif

// Both CUDA_MEASURE_TIME_WRAPPER need to be used together
// The time is measured in milliseconds

#define CUDA_MEASURE_TIME_WRAPPER_BEGIN                              \
    do {                                                             \
         cudaEvent_t startCMTW, stopCMTW;                            \
                                                                     \
         cudaEventCreate(&startCMTW);                                \
         cudaEventCreate(&stopCMTW);                                 \
                                                                     \
         cudaEventRecord(startCMTW)

#define CUDA_MEASURE_TIME_WRAPPER_END(total_time)                    \
         cudaEventRecord(stopCMTW);                                  \
                                                                     \
         cudaEventSynchronize(stopCMTW);                             \
                                                                     \
         float time_spentCMTW = 0;                                   \
         cudaEventElapsedTime(&time_spentCMTW, startCMTW, stopCMTW); \
         printf(":> %f\n", time_spentCMTW);                          \
         total_time += time_spentCMTW;                               \
    } while (0)

#endif

#define CUDA_SYNC_AND_CHECK_KERNEL                                   \
    do {                                                             \
        renderer::cudaCheckKernelSuccess();                          \
        renderer::cudaErrorCheck(cudaDeviceSynchronize());           \
    } while (0)
