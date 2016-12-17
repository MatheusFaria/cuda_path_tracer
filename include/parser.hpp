#ifndef __PARSER_HPP__
#define __PARSER_HPP__

#include <string>

#include "renderer.hpp"
#include "scene.hpp"

extern bool parseScene(const std::string filepath, Scene& scene,
                       renderer::Options& options);

#endif
