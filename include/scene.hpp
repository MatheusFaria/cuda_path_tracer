#ifndef __SCENE_HPP__
#define __SCENE_HPP__

#include <vector>

#include "bvh.hpp"
#include "camera.hpp"
#include "material.hpp"
#include "cuda_definitions.h"

class Scene {
public:
    HOST_ONLY Scene(
        const Camera& _camera=Camera(),
        const BVH& _bvh=BVH(),
        const std::vector<Material>& _materials=std::vector<Material>()
    ) : camera(_camera), bvh(_bvh), materials(_materials){}

    Camera camera;
    BVH bvh;
    std::vector<Material> materials;
};

#endif
