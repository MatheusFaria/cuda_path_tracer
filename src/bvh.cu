#include "bvh.hpp"

#include <algorithm>
#include <queue>
#include <utility>

#include "log_macros.h"
#include "renderer.hpp"

#if defined(_WIN32) && !defined(__clz)
#define __clz __lzcnt
#endif

struct TriangleMorton {
    Triangle triangle;
    unsigned int mortonCode;

    bool operator<(const struct TriangleMorton& t) const
    {
        return mortonCode < t.mortonCode;
    }
};


// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
HOST_ONLY unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
HOST_ONLY unsigned int morton3D(float x, float y, float z)
{
    ASSERT(x >=0 && x <= 1, "morton3D: x [ " << x << "] out of range [0, 1]");
    ASSERT(y >=0 && y <= 1, "morton3D: y [ " << y << "] out of range [0, 1]");
    ASSERT(z >=0 && z <= 1, "morton3D: z [ " << z << "] out of range [0, 1]");

    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);

    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);

    return xx * 4 + yy * 2 + zz;
}

HOST_ONLY unsigned int findSplit(const TriangleMorton* morton_codes,
                                 unsigned int first, unsigned int last)
{
    // morton_codes needs to be sorted

    auto first_code = morton_codes[first].mortonCode;
    auto  last_code = morton_codes[last].mortonCode;

    // If the codes in this range are equal, return the range mid point
    if (first_code == last_code)
        return (first + last) >> 1; // (first + last)/2

    auto prefix = __clz(first_code ^ last_code);
    auto split = first;
    auto step = last - first;

    do {
        step = (step + 1) >> 1;
        auto new_split = split + step;

        if (new_split < last)
        {
            auto split_code = morton_codes[new_split].mortonCode;
            auto split_prefix = __clz(first_code ^ split_code);

            if(split_prefix > prefix)
                split = new_split;
        }

    } while(step > 1);

    return split;
}

HOST_GPU AABB bvhGetAABB(Node* nodes, unsigned int index = 0)
{
    if(nodes[index].is_leaf())
        nodes[index].box = nodes[index].object.aabb();
    else
        nodes[index].box = bvhGetAABB(nodes, nodes[index].child[0]).join(
                                bvhGetAABB(nodes, nodes[index].child[1]));

    return nodes[index].box;
}

HOST_ONLY void
BVH::generate(const Triangle* objects)
{
    INFO("Generating the BVH...");
    TriangleMorton* morton_triangles = new TriangleMorton[n_nodes];
    for(int i = 0; i < n_nodes; ++i)
    {
        morton_triangles[i].triangle = objects[i];

        auto center = objects[i].aabb().centroid();
        morton_triangles[i].mortonCode = morton3D(center.x, center.y, center.z);
    }

    INFO("Sorting objects by their morton code...");
    std::sort(morton_triangles, morton_triangles + n_nodes);

    using ii = std::pair<unsigned int, unsigned int>;
    using iii = std::pair<unsigned int, ii>;

    std::queue<iii> to_process;
    to_process.push(iii(0, ii(0, n_nodes - 1)));

    while(!to_process.empty())
    {
        auto index = to_process.front().first;
        auto first = to_process.front().second.first;
        auto  last = to_process.front().second.second;

        ASSERT(index < size, "Index out of range. Not enough space alocated on"
                             " BVH. Index: " << index << " Size: " << size
                             << ". [first, last] == [" << first << ", "
                             << last << "]");

        to_process.pop();

        auto split = findSplit(morton_triangles, first, last);

        ASSERT(split >= first && split <= last,
               "Split out of range: split [first, last] => "
               << split << " [" << first << ", " << last << "]");

        nodes[index] = Node();

        if (first == split)
        {
            nodes[index].child[0] = first + offset();
            nodes[first + offset()] = Node(morton_triangles[first].triangle);
        }
        else
        {
            nodes[index].child[0] = split;
            to_process.push(iii(split, ii(first, split)));
        }

        if (last == split + 1)
        {
            nodes[index].child[1] = last + offset();
            nodes[last + offset()] = Node(morton_triangles[last].triangle);
        }
        else
        {
            nodes[index].child[1] = split + 1;
            to_process.push(iii(split + 1, ii(split + 1, last)));
        }
    }

    delete[] morton_triangles;

    bvhGetAABB(nodes);
}

GPU_ONLY bool
BVH::intersect(const Ray& ray, Intersection &intersection) const
{
    unsigned int node_stack[64];
    auto node_stack_top = node_stack;

    *node_stack_top++ = 0; // pushing the root node to the stack

    bool hit = false;

    while(node_stack_top != node_stack)
    {
        auto node_index = *--node_stack_top; // pop
        auto node = nodes[node_index];

        auto is_leaf = node.is_leaf();

        hit |= is_leaf && node.object.intersect(ray, intersection);

        auto t_min = intersection.t_min, t_max = intersection.t_max;
        for(int i = 0; i < 2; ++i)
        {
            auto child = nodes[node.child[i]];
            auto child_hit = !is_leaf &&
                             child.box.intersect(ray, &t_min, &t_max);

            *node_stack_top = node.child[i] * !!child_hit;
            node_stack_top += !!child_hit;

            // if(child.box.intersect(ray, &t_min, &t_max))
            //     *node_stack_top++ = node.child[i];
        }
    }

    return hit;
}

HOST_ONLY BVH
BVH::toGPU() const
{
    BVH bvh;

    bvh.size = size;
    bvh.n_nodes = n_nodes;

    auto node_size = sizeof(Node);
    INFO("Allocating BVH data (" << size * node_size << ")...");
    INFO("Node Size: " << node_size);

    renderer::cudaErrorCheck(cudaMalloc((void**)&(bvh.nodes), size * node_size));
    renderer::cudaErrorCheck(cudaMemcpy(bvh.nodes, nodes, size * node_size,
                                        cudaMemcpyHostToDevice));
    return bvh;
}



HOST_ONLY PrimitiveList
PrimitiveList::toGPU() const
{
    PrimitiveList lst;

    lst.size = size;

    auto node_size = sizeof(Triangle);
    INFO("Allocating List data (" << size * node_size << ")...");
    INFO("Node Size: " << node_size);

    renderer::cudaErrorCheck(cudaMalloc((void**)&(lst.nodes),
                                        size * node_size));
    renderer::cudaErrorCheck(cudaMemcpy(lst.nodes, nodes, size * node_size,
                                        cudaMemcpyHostToDevice));

    return lst;
}
