#include "renderer.hpp"

#include <cstring>

#include "initialization_kernels.h"
#include "log_macros.h"
#include "render_kernel.h"

// Variables
GLuint texture_id;       // Main texture ID
GLuint vertex_buffer;    // VBO
GLuint shaders_program;  // Compiled shaders

renderer::Options renderer::options;
Scene renderer::scene;
renderer::KernelData renderer::kernel_data;

cudaGraphicsResource * graphics_resource; // GL and CUDA shared resource

// Helper functions declarations
GLuint loadShaders(const std::string vertex_code,
                   const std::string fragment_code);
void logGLInfo(bool with_extensions=false);

void logCudaDeviceInfo(int device);
void logCudaDevicesInfo();


// Renderer Module Functions

void
renderer::setup()
{
    INFO("Renderer Setup...");

    INFO("OpenGL Setup...");
    // There is no need for GL depth test in a ray tracer
    glDisable(GL_DEPTH_TEST);

    glClearColor(0.0, 0.0, 0.0, 1.0);            // Black background
    glViewport(0, 0, scene.camera.width, scene.camera.height); // GL Screen size

    glErrorCheck();

    // Creating the VAO
    INFO("Creating GL Vertices...");
    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    // Vertices
    static const GLfloat vertices[] = {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
    };

    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glErrorCheck();

    // Texture Creation
    INFO("Creating Texture...");
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, scene.camera.width,
                 scene.camera.height, 0, GL_RGBA, GL_FLOAT, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    glErrorCheck();

    // Shaders Load
    INFO("Shaders Compiling...");
    shaders_program = loadShaders(
        // Vertex Shader Code
        "#version 330 core\n"
        "layout(location = 0) in vec3 position;"
        "out vec2 UV;"
        "void main() {"
        "    gl_Position = vec4(position, 1.0);"
        "    UV = vec2(position) * 0.5 + 0.5;"
        "}",

        // Fragment Shader Code
        "#version 330 core\n"
        "in vec2 UV;"
        "out vec3 color;"
        "uniform sampler2D textureSampler;"
        "void main() {"
        "    color = texture(textureSampler, UV).rgb;"
        "}"
    );

    glErrorCheck();

    logGLInfo();


    // CUDA setup
    INFO("CUDA Setup...");

    int device_count;
    renderer::cudaErrorCheck(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
        ERROR("CUDA Error: No cuda device found");
    else
    {
        INFO("CUDA: using device 0");
        cudaErrorCheck(cudaSetDevice(0));
        logCudaDeviceInfo(0);
    }

    INFO("CUDA GL Interop...");
    cudaErrorCheck(cudaGraphicsGLRegisterImage(
        &graphics_resource, texture_id, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsSurfaceLoadStore)
    );

    INFO("Zero filling kernel data...");
    memset(&kernel_data, 0, sizeof(KernelData));

    INFO("Preparing BVH...");
    kernel_data.bvh = scene.bvh.toGPU();

    INFO("Allocating material data (" <<
         scene.materials.size() * sizeof(Material) << ")...");
    INFO("Material Size: " << sizeof(Material));

    cudaErrorCheck(cudaMalloc((void**)&kernel_data.materials,
                               scene.materials.size() * sizeof(Material)));
    cudaErrorCheck(cudaMemcpy(
        kernel_data.materials, &(scene.materials[0]),
        scene.materials.size() * sizeof(Material), cudaMemcpyHostToDevice
    ));
    kernel_data.n_materials = scene.materials.size();


    auto ray_array_size = scene.camera.width * scene.camera.height
                          * sizeof(RayNode);
    INFO("Allocating rays data (" << ray_array_size << ")...");
    INFO("RayNode Struct Size: " << sizeof(RayNode));

    cudaErrorCheck(cudaMalloc((void**)&kernel_data.rays, ray_array_size));

    initRays(kernel_data.rays, scene.camera);

    auto rand_array_size = scene.camera.width * scene.camera.height
                           * sizeof(curandState);
    INFO("Allocating rand state array (" << rand_array_size << ")...");
    INFO("RandState Size: " << sizeof(curandState));

    cudaErrorCheck(cudaMalloc((void**)&kernel_data.rand_state, rand_array_size));

    initRandomStates(kernel_data.rand_state, scene.camera);
}

void
renderer::tearDown()
{
    INFO("Rendered Tear Down...");

    INFO("OpenGL Tear Down...");
    glDeleteTextures(1, &texture_id);
    glDeleteBuffers(1, &vertex_buffer);
    glDeleteProgram(shaders_program);

    INFO("CUDA Tear Down...");
    cudaErrorCheck(cudaGraphicsUnregisterResource(graphics_resource));

    INFO("Kernel Data Tear Down...");
    cudaErrorCheck(cudaFree(kernel_data.materials));
    cudaErrorCheck(cudaFree(kernel_data.rand_state));
    cudaErrorCheck(cudaFree(kernel_data.rays));

    cudaErrorCheck(cudaFree(kernel_data.bvh.nodes));
}

void
renderer::renderLoop()
{
    // CUDA Loop
    cudaErrorCheck(cudaGraphicsMapResources(1, &graphics_resource, 0));

    cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(
        &kernel_data.graphics_array, graphics_resource,
        0, 0)
    );

    renderKernelCall();

    cudaErrorCheck(cudaGraphicsUnmapResources(1, &graphics_resource, 0));


    // OpenGL Loop
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaders_program);

    glBindTexture(GL_TEXTURE_2D, texture_id);

    // Vertex buffer at location=0
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glDrawArrays(GL_TRIANGLES, 0, 3 * 2);

    glErrorCheck();

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
}

void
renderer::cudaErrorCheck(cudaError_t err)
{
    if (err != cudaSuccess)
        ERROR("CUDA Error (" << err << "): " << cudaGetErrorString(err));
}

void
renderer::cudaCheckKernelSuccess()
{
    auto cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
        ERROR("CUDA kernel failed (" << cudaStatus << "): "
              << cudaGetErrorString(cudaStatus));
}

void
renderer::glErrorCheck()
{
    GLenum err = GL_NO_ERROR;
    while((err = glGetError()) != GL_NO_ERROR)
        ERROR(gluErrorString(err));
}



// Helper Functions

GLuint loadShaders(const std::string vertex_code,
                   const std::string fragment_code)
{
    auto compileShader = [](GLuint & shader_id,
                            const std::string shader_code) -> bool
    {
        char const * source_ptr = shader_code.c_str();
        glShaderSource(shader_id, 1, &source_ptr, NULL);
        glCompileShader(shader_id);

        GLint result = GL_FALSE;
        int log_length;

        glGetShaderiv(shader_id, GL_COMPILE_STATUS, &result);
        glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &log_length);

        if (log_length > 0)
        {
            GLchar * error_msg = new GLchar[log_length + 1];
            glGetShaderInfoLog(shader_id, log_length, NULL, error_msg);
            ERROR("Shaders: " << error_msg);
        }

        return result != GL_FALSE;
    };

    GLuint vertex_shader_id = glCreateShader(GL_VERTEX_SHADER);
    auto vertex_shader_status = compileShader(vertex_shader_id,
                                              vertex_code);

    GLuint frag_shader_id = glCreateShader(GL_FRAGMENT_SHADER);
    auto frag_shader_status = compileShader(frag_shader_id,
                                            fragment_code);

    if (!vertex_shader_status || !frag_shader_status)
        ERROR("Shaders: Could not compile shaders!");

    GLuint program_id = glCreateProgram();
    glAttachShader(program_id, vertex_shader_id);
    glAttachShader(program_id, frag_shader_id);
    glLinkProgram(program_id);

    GLint result = GL_FALSE;
    int log_length;

    glGetProgramiv(program_id, GL_LINK_STATUS, &result);
    glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &log_length);

    if (log_length > 0)
    {
        GLchar * error_msg = new GLchar[log_length + 1];
        glGetProgramInfoLog(program_id, log_length, NULL, error_msg);
        ERROR("Shaders: " << error_msg);
    }

    glDetachShader(program_id, vertex_shader_id);
    glDetachShader(program_id, frag_shader_id);

    glDeleteShader(vertex_shader_id);
    glDeleteShader(frag_shader_id);

    return program_id;
}

void logGLInfo(bool with_extensions)
{
    // OpenGL must be already initializated

    INFO("-- OpenGL Infomation --")
    INFO("OpenGL Vendor:   " << glGetString(GL_VENDOR));
    INFO("OpenGL Renderer: " << glGetString(GL_RENDERER));
    INFO("OpenGL Version:  " << glGetString(GL_VERSION));
    if (with_extensions)
        INFO("OpenGL Extensions: " << glGetString(GL_EXTENSIONS));
}

void logCudaDeviceInfo(int device)
{
    INFO("Device " << device << ":");
    cudaDeviceProp dev_prop;
    renderer::cudaErrorCheck(cudaGetDeviceProperties(&dev_prop, device));

    INFO("\tName: " << dev_prop.name);

    INFO("\n\t---Memory---");
    INFO("\tTotal Global Memory: " << dev_prop.totalGlobalMem);
    INFO("\tTotal Constant Memory: " << dev_prop.totalConstMem);
    INFO("\tShared Memory Per Block: " << dev_prop.sharedMemPerBlock);
    INFO("\tRegister Per Block: " << dev_prop.regsPerBlock);

    INFO("\n\t---Threads, Blocks, Dimensions---");
    INFO("\tWarp Size: " << dev_prop.warpSize);
    INFO("\tMax Threads Per Block: " << dev_prop.maxThreadsPerBlock);
    INFO("\tMax Dimension Size: " << dev_prop.maxThreadsDim[0] << " "
                                  << dev_prop.maxThreadsDim[1] << " "
                                  << dev_prop.maxThreadsDim[2]);
    INFO("\tMax Grid Size: " << dev_prop.maxGridSize[0] << " "
                             << dev_prop.maxGridSize[1] << " "
                             << dev_prop.maxGridSize[2]);
}

void logCudaDevicesInfo()
{
    int device_count;
    renderer::cudaErrorCheck(cudaGetDeviceCount(&device_count));

    // Get devices stats
    for (int i = 0; i < device_count; ++i)
        logCudaDeviceInfo(i);
}
