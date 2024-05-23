/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/texturing/visibility_tester.h"

namespace ugu {

enum TextureMappingType { kSimpleProjection = 0 };

enum TexturingOutputUvType {
  kGenerateSimpleTile = 0,
  kUseOriginalMeshUv = 1,
  kGenerateSimpleTriangles = 2,
  kGenerateSimpleCharts = 3,
  kConcatHorizontally = 4,
  kConcatVertically = 5,
};

struct TextureMappingOption {
  ViewSelectionCriteria criteria = ViewSelectionCriteria::kMaxArea;
  TextureMappingType type = TextureMappingType::kSimpleProjection;
  TexturingOutputUvType uv_type = TexturingOutputUvType::kGenerateSimpleTile;
  std::string texture_base_name = "ugutex";
  int tex_w = 1024;
  int tex_h = 1024;
  int padding_kernel = 3;
};

bool TextureMapping(const std::vector<std::shared_ptr<Keyframe>>& keyframes,
                    const VisibilityInfo& info, Mesh* mesh,
                    const TextureMappingOption& option);

}  // namespace ugu
