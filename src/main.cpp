#include "gl_includes.h"

#ifdef _WIN32
// This deactivates the min and max functions defined on the minwindef.h
#define NOMINMAX
#endif

#define FREEGLUT_STATIC
#include "GL/freeglut.h"

#ifdef _WIN32
// Both are defined on minwindef.h included by freeglut
// They are being undefined here to allow the use of both var names
#undef far
#undef near
#endif

#include "basic_math_3d.hpp"
#include "renderer.hpp"
#include "log_macros.h"
#include "parser.hpp"

using namespace std;

// Function Declarations

void drawLoop(void);
void glutResize(int, int);

// Main

int main(int argc, char **argv)
{
    std::string scene_path;
    scene_path = "../samples/complex_scene.json";
    scene_path = "../samples/reflective_buddha.json";
    scene_path = "../samples/caustics.json";
    scene_path = "../samples/cornell_box.json";

    bool parse_ok = parseScene(scene_path, renderer::scene, renderer::options);

    if (!parse_ok) ERROR("Parse incomplete!");

    // GLUT Setup
    INFO("GLUT Setup...");
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(renderer::scene.camera.width,
                       renderer::scene.camera.height);
    glutCreateWindow(renderer::options.project_title.c_str());
    glutReshapeFunc(glutResize);
    glutDisplayFunc(drawLoop);
    glutIdleFunc(drawLoop);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                  GLUT_ACTION_GLUTMAINLOOP_RETURNS);


    // GLEW Setup
    INFO("GLEW Setup...");
    glewExperimental = true;
    auto glewReturn = glewInit();
    if (glewReturn != GLEW_OK)
    {
        ERROR("Failed to initialize GLEW: " << glewGetErrorString(glewReturn));
        return -1;
    }

    // Renderer Setup
    renderer::setup();

    INFO("Init render loop...");
    glutMainLoop();
    INFO("Finish render loop...");

    // Render Tear Down
    renderer::tearDown();

    return 0;
}


// Function Definitions

void drawLoop(void)
{
    renderer::renderLoop();
    glutSwapBuffers();
}

void glutResize(int, int)
{
    glutReshapeWindow(renderer::scene.camera.width,
                      renderer::scene.camera.height);
}
