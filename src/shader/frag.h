/*
 * Automatically generated by script/glsl2header.py
 */
/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#pragma once

#include <string>

namespace ugu {

static inline std::string frag_deferred_code =
    R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
uniform sampler2D gId;
uniform sampler2D gFace;

uniform bool showWire;
uniform vec3 wireCol;
uniform float nearZ;
uniform float farZ;
uniform vec3 bkgCol;

const int N_POSITIONS = 32;
uniform vec3 selectedPositions[N_POSITIONS];
uniform float selectedPosDepthTh;
uniform vec2 viewportOffset;

struct Light {
  vec3 Position;
  vec3 Color;

  float Linear;
  float Quadratic;
};
const int NR_LIGHTS = 32;
uniform Light lights[NR_LIGHTS];
uniform vec3 viewPos;

void main() {
  // retrieve data from gbuffer
  vec3 FragPos = texture(gPosition, TexCoords).rgb;
  vec3 Normal = texture(gNormal, TexCoords).rgb;
  vec3 Diffuse = texture(gAlbedoSpec, TexCoords).rgb;
  float Specular = texture(gAlbedoSpec, TexCoords).a;
  vec4 Face = texture(gFace, TexCoords);
  vec4 Id = texture(gId, TexCoords);

  // then calculate lighting as usual
  vec3 lighting = Diffuse * 0.1;  // hard-coded ambient component
  vec3 viewDir = normalize(viewPos - FragPos);
  for (int i = 0; i < NR_LIGHTS; ++i) {
    // diffuse
    vec3 lightDir = normalize(lights[i].Position - FragPos);
    vec3 diffuse = max(dot(Normal, lightDir), 0.0) * Diffuse * lights[i].Color;
    // specular
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(Normal, halfwayDir), 0.0), 16.0);
    vec3 specular = lights[i].Color * spec * Specular;
    // attenuation
    float distance = length(lights[i].Position - FragPos);
    float attenuation = 1.0 / (1.0 + lights[i].Linear * distance +
                               lights[i].Quadratic * distance * distance);
    diffuse *= attenuation;
    specular *= attenuation;
    lighting += diffuse + specular;
  }
  // FragColor = vec4((Id.y * 3) * 0.2, 0.5, 0.6, 1.0);
  vec4 wireCol4 = vec4(wireCol, 1.0);
  float wire = mix(0.0, Specular, showWire);
  float depth = Id.z;
  // FragColor = vec4(Specular, Specular, Specular, 1.0);
  FragColor = vec4(Diffuse, 1.0) * (1.0 - wire) + wire * wireCol4;
  bool is_frg = nearZ < depth && depth < farZ;
  FragColor =
      mix(vec4(bkgCol, 1.0), FragColor, vec4(is_frg));

  vec4 selectPosColor = vec4(1.0, 0.0, 0.0, 1.0);
  const float SELECT_COLOR_RADIUS = 10;
  for (int i = 0; i < N_POSITIONS; ++i) {
    // Ignore defualt [0, 0]
    if (selectedPositions[i].x < 1.0 && selectedPositions[i].y < 1.0) {
      continue;
    }
    // Handle occulsion by depth check
    if (is_frg && selectedPositions[i].z - depth > selectedPosDepthTh) {
      continue;
    }
    vec2 posInBuf = gl_FragCoord.xy - viewportOffset;
    float dist = distance(posInBuf, selectedPositions[i].xy);
    if (dist <= SELECT_COLOR_RADIUS) {
      FragColor = selectPosColor;
    }
  }
})";
static inline std::string frag_gbuf_code =
    R"(
#version 330
layout(location = 0) out vec3 gPosition;
layout(location = 1) out vec3 gNormal;
layout(location = 2) out vec4 gAlbedoSpec;
layout(location = 3) out vec4 gId;
layout(location = 4) out vec4 gFace;

in vec3 fragPos;
in vec3 viewPos;
in vec2 texCoords;
in vec3 normal;
in vec3 wldNormal;
in vec3 vertexColor;
in vec3 vertexId;
in vec2 bary;
in vec3 dist;

uniform sampler2D texture_diffuse1;
uniform sampler2D texture_specular1;

void main() {
  // store the fragment position vector in the first gbuffer texture
  gPosition = fragPos;
  // also store the per-fragment normals into the gbuffer
  gNormal = normalize(normal);
  // and the diffuse per-fragment color
  gAlbedoSpec.rgb = texture(texture_diffuse1, texCoords).rgb;
  // store specular intensity in gAlbedoSpec's alpha component
  gAlbedoSpec.a = texture(texture_specular1, texCoords).r;

  gId.x = float(gl_PrimitiveID + 1);  // vertedId.x;
  gId.y = vertexId.y;

  gId.z = -viewPos.z; // Linear depth

  gFace.xy = bary;
  gFace.zw = texCoords;

  vec3 dist_vec = dist;
  float d = min(dist_vec[0], min(dist_vec[1], dist_vec[2]));
  float I = exp2(-2.0 * d * d);
  //  Use specular for wire intensity
  gAlbedoSpec.a = clamp(I, 0.0, 1.0);
})";
}  // namespace ugu
