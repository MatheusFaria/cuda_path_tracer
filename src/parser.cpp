#include "parser.hpp"

#include <fstream>
#include <map>
#include <vector>

#include "json11.hpp"
#include <tiny_obj_loader.h>

#include "aabb.hpp"
#include "basic_math_3d.hpp"
#include "bvh.hpp"
#include "camera.hpp"
#include "log_macros.h"
#include "material.hpp"
#include "primitive.hpp"

using namespace json11;

bool hasAttribute(Json json, const std::string& attr,
                  const std::string& parent_attr, Json::Type type,
                  unsigned int lenght=0)
{
    if(json[attr].is_null())
    {
        WARN("Missing " << attr << " in " << parent_attr);
        return false;
    }

    if(json[attr].type() != type)
    {
        WARN(parent_attr << "." << attr << " has wrong type.");
        return false;
    }

    if(lenght && type == Json::ARRAY
       && json[attr].array_items().size() != lenght)
    {
        WARN(parent_attr << "." << attr << " array has "
             << json[attr].array_items().size() << " elements. It should have "
             << lenght);
        return false;
    }

    return true;
}

Vector3 parseVector3(Json json)
{
    auto json_array = json.array_items();
    return Vector3(json_array[0].number_value(),
                   json_array[1].number_value(),
                   json_array[2].number_value());
}

bool
loadMesh(std::string mesh_filename, unsigned int material_id,
         const Matrix4x4& T, std::vector<Triangle>& triangles_vec,
         AABB& bbox)
{
    INFO("Loading mesh " << mesh_filename << " ...");

    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = tinyobj::LoadObj(shapes, materials,
                                       mesh_filename.c_str());

    if(!err.empty())
    {
        WARN("Mesh parse error: " << err);
        return false;
    }

    for(auto shape: shapes)
    {
        auto   indices = shape.mesh.indices;
        auto positions = shape.mesh.positions;

        DEBUG("Load Mesh: " << indices.size() / 3 << " triangles found");
        for (size_t i = 0; i < indices.size() / 3; i++)
        {
            auto idx1 = 3 * indices[3 * i + 0];
            auto idx2 = 3 * indices[3 * i + 1];
            auto idx3 = 3 * indices[3 * i + 2];
            Vector3 v1(positions[idx1 + 0],
                       positions[idx1 + 1],
                       positions[idx1 + 2]);
            Vector3 v2(positions[idx2 + 0],
                       positions[idx2 + 1],
                       positions[idx2 + 2]);
            Vector3 v3(positions[idx3 + 0],
                       positions[idx3 + 1],
                       positions[idx3 + 2]);

            triangles_vec.push_back(Triangle(multiplyPoint(T, v1),
                                             multiplyPoint(T, v2),
                                             multiplyPoint(T, v3),
                                             material_id));
            bbox.join(triangles_vec.back().aabb());
        }
    }

    INFO("Mesh " << mesh_filename << " loaded");
    return true;
}

bool
parseScene(const std::string filepath, Scene& scene, renderer::Options& options)
{
    INFO("Parsing file " << filepath << " ...");

    if (filepath.empty())
    {
        WARN("Parser error: " << filepath << " is empty!");
        return false;
    }

    std::ifstream file(filepath);

    // Checks if the file exists
    if (!file.good())
    {
        WARN("The file " << filepath << " does not exist");
        return false;
    }

    std::string lines = "";
    std::string line;
    while(std::getline(file, line))
        lines += line;

    std::string err;
    auto json = Json::parse(lines, err);
    if (!err.empty())
    {
        WARN("JSON parse error: " << err);
        return false;
    }

    // Project parse
    if (   !hasAttribute(json, "project", "root", Json::OBJECT)
        || !hasAttribute(json["project"], "name", "project", Json::STRING)
        || !hasAttribute(json["project"], "n_samples", "project", Json::NUMBER)
        || !hasAttribute(json["project"], "n_bounces", "project", Json::NUMBER)
       )
    {
        WARN("JSON Syntax Error: no project found or incomplete");
        return false;
    }

    options.project_title = json["project"]["name"].string_value();
    options.n_samples = json["project"]["n_samples"].int_value();
    options.n_bounces = json["project"]["n_bounces"].int_value();
    DEBUG(json["project"].dump());

    // Camera parse
    if (   !hasAttribute(json, "camera", "root", Json::OBJECT)
        || !hasAttribute(json["camera"], "film", "camera", Json::OBJECT)
        || !hasAttribute(json["camera"]["film"], "width",  "camera.film", Json::NUMBER)
        || !hasAttribute(json["camera"]["film"], "height", "camera.film", Json::NUMBER)
        || !hasAttribute(json["camera"], "fov", "camera", Json::NUMBER)
        || !hasAttribute(json["camera"], "position", "camera", Json::ARRAY, 3)
        || !hasAttribute(json["camera"], "lookat", "camera", Json::ARRAY, 3)
        || !hasAttribute(json["camera"], "up", "camera", Json::ARRAY, 3)
       )
    {
        WARN("JSON Syntax Error: no camera found or incomplete");
        return false;
    }

    Camera camera(
        json["camera"]["film"]["width"].int_value(),
        json["camera"]["film"]["height"].int_value(),
        json["camera"]["fov"].number_value(),
        parseVector3(json["camera"]["position"]),
        parseVector3(json["camera"]["lookat"]),
        parseVector3(json["camera"]["up"])
    );

    // Objects parse
    if (!hasAttribute(json, "objects", "root", Json::ARRAY))
    {
        WARN("JSON Syntax Error: no objects found or incomplete");
        return false;
    }

    std::map<Material, unsigned int> material_id;
    std::vector<Material> materials;
    std::vector<Triangle> triangles_vec;
    AABB bbox;
    bbox.join(parseVector3(json["camera"]["position"]));

    for(auto obj: json["objects"].array_items())
    {
        if (   !hasAttribute(obj, "filename", "objects[i]", Json::STRING)
            || !hasAttribute(obj, "transforms", "objects[i]", Json::ARRAY)
            || !hasAttribute(obj, "material", "objects[i]", Json::OBJECT)
            || !hasAttribute(obj["material"], "type", "objects[i].material",
                             Json::STRING)
           )
        {
            WARN("JSON Syntax Error: bad object");
            return false;
        }

        std::string obj_filename = obj["filename"].string_value();

        // Transforms Parse
        Matrix4x4 T;
        for (auto transform: obj["transforms"].array_items())
        {
            if (transform.type() != Json::ARRAY)
            {
                WARN("JSON Syntax Error: transform is not an array. "
                     << transform.dump());
                return false;
            }

            auto transform_items = transform.array_items();

            if (transform_items[0].type() != Json::STRING)
            {
                WARN("JSON Syntax Error: first transform field should be a "
                     << "string with its type. " << transform.dump());
                return false;
            }

            auto type = transform_items[0].string_value();

            if (type == "transalation")
            {
                if (   transform_items.size() != 2
                    || transform_items[1].type() != Json::ARRAY
                    || transform_items[1].array_items().size() != 3)
                {
                    WARN("JSON Syntax Error: the translation additional field "
                         << "should be an array. " << transform.dump());
                    return false;
                }

                T = translate(parseVector3(transform_items[1])) * T;
            }
            else if (type == "scale")
            {
                if (   transform_items.size() != 2
                    || transform_items[1].type() != Json::ARRAY
                    || transform_items[1].array_items().size() != 3)
                {
                    WARN("JSON Syntax Error: the translation additional field "
                         << "should be an array. " << transform.dump());
                    return false;
                }

                T = scale(parseVector3(transform_items[1])) * T;
            }
            else if (type == "rotation")
            {
                if (   transform_items.size() != 3
                    || transform_items[1].type() != Json::NUMBER
                    || transform_items[2].type() != Json::STRING)
                {
                    WARN("JSON Syntax Error: the translation additional field "
                         << "should be an array. " << transform.dump());
                    return false;
                }

                float angle = transform_items[1].number_value();

                for(auto axis: transform_items[2].string_value())
                {
                         if (axis == 'x') T = rotateX(angle) * T;
                    else if (axis == 'y') T = rotateY(angle) * T;
                    else if (axis == 'z') T = rotateZ(angle) * T;
                    else
                        WARN("JSON Syntax Error: Unkown rotation axis " << axis);
                }
            }
            else
                WARN("JSON Syntax Error: Unkown transform type " << type);
        }

        // Material Parse
        std::string material_type = obj["material"]["type"].string_value();
        Material material;

        if (material_type == "lambertian")
        {
            if (!hasAttribute(obj["material"], "albedo", "objects[i].material",
                              Json::ARRAY, 3))
            {
                WARN("JSON Syntax Error: material missing albedo");
                return false;
            }

            material = Material(LAMBERTIAN,
                                parseVector3(obj["material"]["albedo"]));
        }
        else if (material_type == "metal")
        {
            if (!hasAttribute(obj["material"], "albedo", "objects[i].material",
                              Json::ARRAY, 3)
                || !hasAttribute(obj["material"], "fuzziness",
                                 "objects[i].material", Json::NUMBER))
            {
                WARN("JSON Syntax Error: metal material wrong syntax");
                return false;
            }

            material = Material(METAL,
                                parseVector3(obj["material"]["albedo"]),
                                obj["material"]["fuzziness"].number_value());
        }
        else if (material_type == "dieletric")
        {
            if (!hasAttribute(obj["material"], "ior",
                              "objects[i].material", Json::NUMBER))
            {
                WARN("JSON Syntax Error: dieletric material wrong syntax");
                return false;
            }

            material = Material(DIELETRIC, Vector3(),
                        obj["material"]["ior"].number_value());
        }
        else if (material_type == "light")
        {
            if (!hasAttribute(obj["material"], "color", "objects[i].material",
                              Json::ARRAY, 3))
            {
                WARN("JSON Syntax Error: light missing color");
                return false;
            }

            material = Material(DIFFUSE_LIGHT,
                                parseVector3(obj["material"]["color"]));
        }
        else
            WARN("JSON Syntax Error: Unkown material " << material_type);

        if (material_id.find(material) == material_id.end())
        {
            DEBUG("Parse: New Material type(" << material.type << ") "
                  << " albedo " << material.albedo
                  << " modifier(" << material.modifier << ")");
            material_id[material] = materials.size();
            materials.push_back(material);
        }

        bool loaded = loadMesh(obj_filename, material_id[material], T,
                               triangles_vec, bbox);

        if (!loaded) return false;
    }

    DEBUG(triangles_vec.size() << " triangles found.");
    INFO("Scene Initial Bounding Box: " << bbox.lower_bound << " "
         << bbox.upper_bound);
    INFO("Transforming scene to unit cube...");

    auto T = translate(-bbox.lower_bound);
    DEBUG("Translate scene by: " << -bbox.lower_bound);

    auto scale_vec = bbox.upper_bound - bbox.lower_bound;
    auto scale_factor = 1.0f/std::max(std::max(scale_vec.x, scale_vec.y),
                                      scale_vec.z);
    auto S = scale(scale_factor);
    DEBUG("Scale scene by: " << scale_factor);

    auto M = S * T;

    for(auto& triangle: triangles_vec)
    {
        triangle.p1 = multiplyPoint(M, triangle.p1);
        triangle.p2 = multiplyPoint(M, triangle.p2);
        triangle.p3 = multiplyPoint(M, triangle.p3);
    }

    camera = Camera(
        json["camera"]["film"]["width"].int_value(),
        json["camera"]["film"]["height"].int_value(),
        json["camera"]["fov"].number_value(),
        multiplyPoint(M, parseVector3(json["camera"]["position"])),
        multiplyPoint(M, parseVector3(json["camera"]["lookat"])),
        parseVector3(json["camera"]["up"])
    );


    scene.camera = camera;
    scene.materials = materials;
    scene.bvh = BVH(triangles_vec.size());
    scene.bvh.generate(&(triangles_vec[0]));

    // Parse Info
    INFO("---- Parse Info ----");
    INFO("\tNumber of triangles: " << triangles_vec.size());
    INFO("\tNumber of materials: " << scene.materials.size());
    return true;
}


