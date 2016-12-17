#ifndef __BVH_HPP__
#define __BVH_HPP__

#include <vector>

#include "log_macros.h"
#include "aabb.hpp"
#include "cuda_definitions.h"
#include "intersection.hpp"
#include "primitive.hpp"
#include "ray.hpp"

class Node {
public:
    HOST_GPU Node(Triangle _object=Triangle()) : object(_object)
        { child[0] = child[1] = 0; }

    HOST_GPU inline bool is_leaf() const { return child[0] == child[1]; }

    HOST_GPU inline Node& operator=(const Node& node)
    {
        child[0] = node.child[0];
        child[1] = node.child[1];
        box = node.box;
        object = node.object;
        return *this;
    }

    unsigned int child[2];
    Triangle object;
    AABB box;
    char pad[24];
};

class BVH {
public:
    HOST_GPU BVH() : nodes(nullptr), size(0) {}

    HOST_GPU BVH(Node * _nodes, unsigned int _size)
        : nodes(_nodes), size(_size), n_nodes((size + 1)/2) {};

    HOST_GPU BVH(unsigned int _n_nodes)
        : size(_n_nodes * 2  - 1), n_nodes(_n_nodes)
    {
        if(size > 0) nodes = new Node[size];
        else nodes = nullptr;
    };

    HOST_ONLY BVH toGPU() const;

    HOST_ONLY void generate(const Triangle* objects);
    GPU_ONLY bool intersect(const Ray& ray, Intersection &intersection) const;

    Node* nodes;
    unsigned int size;
    unsigned int n_nodes;

private:
    inline unsigned int offset() const { return n_nodes - 1; }
};


class PrimitiveList {
public:
    HOST_GPU PrimitiveList(Triangle * _triangles=nullptr, unsigned int _size=0)
        : nodes(_triangles), size(_size) {};

    HOST_GPU PrimitiveList(std::vector<Triangle> triangles_vec)
        : size(triangles_vec.size())
    {
        nodes = new Triangle[size];
        memcpy(nodes, &(triangles_vec[0]), sizeof(Triangle) * size);
    };

    HOST_GPU bool intersect(const Ray& ray, Intersection &intersection) const
    {
        bool ret = false;
        for(int i = 0; i < size; ++i)
            ret |= nodes[i].intersect(ray, intersection);
        return ret;
    }

    HOST_ONLY PrimitiveList toGPU() const;

    Triangle * nodes;
    int size;
};
#endif
