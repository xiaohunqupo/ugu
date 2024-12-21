#include <limits>
#include <mutex>
#include <random>
#include <thread>

#include "glad/gl.h"
#include "ugu/camera.h"
#include "ugu/renderable_mesh.h"
#include "ugu/renderer/gl/renderer.h"
#include "ugu/timer.h"
#include "ugu/util/image_util.h"
#include "ugu/util/string_util.h"

#include <GLFW/glfw3.h>  // Will drag system OpenGL headers

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to
// maximize ease of testing and compatibility with old VS compilers. To link
// with VS2010-era libraries, VS2015+ requires linking with
// legacy_stdio_definitions.lib, which we do using this pragma. Your own project
// should not be affected, as you are likely to link with a newer binary of GLFW
// that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && \
    !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

int g_width = 1280;
int g_height = 720;

ugu::RendererGlPtr g_renderer = nullptr;
std::vector<ugu::RenderableMeshPtr> g_meshes;
std::unordered_map<ugu::RenderableMeshPtr, Eigen::Affine3f> g_model_matrices;

namespace {

void Draw(GLFWwindow *window) {
  glClear(GL_COLOR_BUFFER_BIT);

  glViewport(0, 0, g_width, g_height);

  for (const auto &mesh : g_meshes) {
    g_renderer->SetMesh(mesh, g_model_matrices.at(mesh), false);
  }
  g_renderer->SetViewport(0, 0, g_width, g_height);
  g_renderer->Draw();

  glClear(GL_DEPTH_BUFFER_BIT);

  glfwSwapBuffers(window);
}

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void window_size_callback(GLFWwindow *window, int width, int height) {
  (void)window;

  if (width < 1 && height < 1) {
    return;
  }

  g_width = width;
  g_height = height;
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  (void)width, (void)height;

  Draw(window);
}

void SetupWindow(GLFWwindow *window) {
  if (window == NULL) return;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // Enable vsync

  // glfwSetCursorPosCallback(window, cursor_pos_callback);

  // glfwSetKeyCallback(window, key_callback);

  // glfwSetMouseButtonCallback(window, mouse_button_callback);

  // glfwSetScrollCallback(window, mouse_wheel_callback);

  // glfwSetDropCallback(window, drop_callback);

  // glfwSetCursorEnterCallback(window, cursor_enter_callback);

  glfwSetWindowSizeCallback(window, window_size_callback);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
}

Eigen::Vector3f default_clear_color = {0.45f, 0.55f, 0.60f};
Eigen::Vector3f default_wire_color = {0.1f, 0.1f, 0.1f};

}  // namespace

int main(int, char **) {
  // Setup window
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) return 1;

  const char *glsl_version = "#version 330";

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Create window with graphics context
  GLFWwindow *window =
      glfwCreateWindow(g_width, g_height, "UGU Mesh Viewer", NULL, NULL);

  SetupWindow(window);

  const int version = gladLoadGL(glfwGetProcAddress);
  if (version == 0) {
    fprintf(stderr, "Failed to load OpenGL 3.x/4.x libraries!\n");
    return 1;
  }

  glEnable(GL_DEPTH_TEST);

  g_meshes.resize(1);
  g_meshes[0] = ugu::RenderableMesh::Create();
  g_meshes[0]->LoadObj("../data/bunny/bunny.obj");
  g_model_matrices[g_meshes[0]] = Eigen::Affine3f::Identity();

  ugu::PinholeCameraPtr camera =
      std::make_shared<ugu::PinholeCamera>(g_width, g_height, 45.f);
  Eigen::Affine3d c2w = Eigen::Affine3d::Identity();
  auto stats = g_meshes[0]->stats();
  c2w.translation() = Eigen::Vector3d(
      stats.center.x(), stats.center.y(),
      stats.center.z() + 2 * (stats.bb_max.z() - stats.bb_min.z()));
  camera->set_c2w(c2w);
  g_renderer = std::make_shared<ugu::RendererGl>();
  g_renderer->SetSize(static_cast<uint32_t>(g_width),
                      static_cast<uint32_t>(g_height));
  g_renderer->SetCamera(camera);
  g_renderer->Init();

  g_renderer->SetBackgroundColor(default_clear_color);
  g_renderer->SetWireColor(default_wire_color);

  g_renderer->SetShowWire(false);
  g_renderer->SetFlatNormal(true);

  g_renderer->ClearGlState();
  for (const auto &mesh : g_meshes) {
    g_renderer->SetMesh(mesh, g_model_matrices.at(mesh), false);
  }

  g_renderer->Init();

  //  Main loop
  const double mean = 0.0;
  const double stddev = (stats.bb_max - stats.bb_min).maxCoeff() * 0.01;
  std::default_random_engine generator;
  std::normal_distribution<double> dist(mean, stddev);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

#if 0
    for (const auto &mesh : g_meshes) {
      // auto vertices = mesh->vertices();
      auto renderable_vertices_org = mesh->renderable_vertices;

      for (auto &v : mesh->renderable_vertices) {
        v.pos +=
            Eigen::Vector3f(dist(generator), dist(generator), dist(generator));
      }
      // for (size_t i = 0; i < vertices.size(); i++) {
      //   mesh->renderable_vertices[i].pos = vertices[i];
      // }

      mesh->UpdateMesh();

      // mesh->set_vertices(vertices);
      mesh->renderable_vertices = renderable_vertices_org;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
#endif

    glfwMakeContextCurrent(window);
    Draw(window);

    glViewport(0, 0, g_width, g_height);
  }

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
