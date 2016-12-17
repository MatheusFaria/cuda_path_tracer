#ifndef __INIT_KERNEL_H__
#define __INIT_KERNEL_H__

#include "camera.hpp"
#include "ray.hpp"
#include "renderer.hpp"

extern void initRandomStates(
    curandState * rand_state, const Camera& cam
);

extern void initRays(
    RayNode * rays, const Camera& cam
);

#endif
