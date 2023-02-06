#version 330
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

uniform vec2 WIN_SCALE;

in vec3 vFragPos[];
in vec3 vViewPos[];
in vec2 vTexCoords[];
in vec3 vNormal[];
in vec3 vWldNormal[];
in vec3 vVertexColor[];
in vec3 vVertexId[];

out vec3 fragPos;
out vec3 viewPos;
out vec2 texCoords;
out vec3 normal;
out vec3 wldNormal;
out vec3 vertexColor;
out vec3 vertexId;
out vec2 bary;
out vec3 dist;

void main() {
  vec3[3] dists =
      vec3[](vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0));

  vec4 p0_3d = gl_in[0].gl_Position;
  vec4 p1_3d = gl_in[1].gl_Position;
  vec4 p2_3d = gl_in[2].gl_Position;

  vec2 p0 = p0_3d.xy / p0_3d.w;
  vec2 p1 = p1_3d.xy / p1_3d.w;
  vec2 p2 = p2_3d.xy / p2_3d.w;

  vec2 v10 = WIN_SCALE * (p1 - p0);
  vec2 v20 = WIN_SCALE * (p2 - p0);
  float area0 = abs(v10.x * v20.y - v10.y * v20.x);
  float h0 = area0 / length(v10 - v20);
  dists[0] = vec3(h0, 0.0, 0.0);

  vec2 v01 = WIN_SCALE * (p0 - p1);
  vec2 v21 = WIN_SCALE * (p2 - p1);
  float area1 = abs(v01.x * v21.y - v01.y * v21.x);
  float h1 = area1 / length(v01 - v21);
  dists[1] = vec3(0.0, h1, 0.0);

  vec2 v02 = WIN_SCALE * (p0 - p2);
  vec2 v12 = WIN_SCALE * (p1 - p2);
  float area2 = abs(v02.x * v12.y - v02.y * v12.x);
  float h2 = area2 / length(v02 - v12);
  dists[2] = vec3(0.0, 0.0, h2);

  for (int i = 0; i < gl_in.length(); ++i) {
    gl_Position = gl_in[i].gl_Position;
    gl_PrimitiveID = gl_PrimitiveIDIn;
    // fid = gl_PrimitiveIDIn;
    fragPos = vFragPos[i];
    viewPos = vViewPos[i];
    texCoords = vTexCoords[i];
    normal = vNormal[i];
    wldNormal = vWldNormal[i];
    vertexColor = vVertexColor[i];
    vertexId = vVertexId[i];
    if (i == 0) {
      bary = vec2(0.0, 0.0);
    } else if (i == 1) {
      bary = vec2(1.0, 0.0);
    } else {
      bary = vec2(0.0, 1.0);
    }
    dist = dists[i];
    EmitVertex();
  }
  EndPrimitive();
}
