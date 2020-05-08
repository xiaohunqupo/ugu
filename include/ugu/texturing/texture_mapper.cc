/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "texture_mapper.h"

namespace {

bool MakeTiledImage(
    const std::vector<std::shared_ptr<ugu::Keyframe>>& keyframes,
    ugu::Image3b* tiled, int x_tile_num, int y_tile_num) {
  // TODO:: different image size

  int org_w = keyframes[0]->color.cols;
  int org_h = keyframes[0]->color.rows;
  int tex_w = x_tile_num * keyframes[0]->color.cols;
  int tex_h = y_tile_num * keyframes[0]->color.rows;

  *tiled = ugu::Image3b::zeros(tex_h, tex_w);

  std::vector<std::array<int, 2>> kf_tile_pos_list(keyframes.size());
  for (int i = 0; i < static_cast<int>(keyframes.size()); i++) {
    auto& pos = kf_tile_pos_list[i];
    pos[0] = i % x_tile_num;
    pos[1] = i / x_tile_num;
    int tex_pos_x = pos[0] * org_w;
    int tex_pos_y = pos[1] * org_h;

    unsigned char* base_adr = tiled->data + (tex_pos_y * tex_w + tex_pos_x) * 3;

    // copy per line
    for (int j = 0; j < keyframes[0]->color.rows; j++) {
      std::memcpy(base_adr + tex_w * 3 * (j),
                  keyframes[i]->color.data + org_w * 3 * j, org_w * 3);
    }
  }

  return true;
}

bool GenerateSimpleTileTextureAndUv(
    const std::vector<std::shared_ptr<ugu::Keyframe>>& keyframes,
    const ugu::VisibilityInfo& info, ugu::Mesh* mesh,
    const ugu::TextureMappingOption& option,
    const std::unordered_map<int, std::vector<ugu::FaceInfoPerKeyframe>>&
        bestkfid2faceid,
    const std::vector<ugu::FaceInfoPerKeyframe>& faceid2bestkf) {
  // Make tiled image and get tile xy
  ugu::Image3b texture;

  // Make tiled image and get tile xy
  int x_tile_num =
      keyframes.size() < 5 ? static_cast<int>(keyframes.size()) : 5;
  int y_tile_num = static_cast<int>(keyframes.size() / x_tile_num) + 1;

  MakeTiledImage(keyframes, &texture, x_tile_num, y_tile_num);

  // Convert projected_tri to UV by tile xy
  std::vector<Eigen::Vector2f> uv;
  std::vector<Eigen::Vector3i> uv_indices;
  std::unordered_map<int, int> id2index;

  for (int i = 0; i < static_cast<int>(keyframes.size()); i++) {
    id2index.emplace(keyframes[i]->id, i);
  }

  int org_w = keyframes[0]->color.cols;
  int org_h = keyframes[0]->color.rows;
  int tex_w = x_tile_num * keyframes[0]->color.cols;
  int tex_h = y_tile_num * keyframes[0]->color.rows;

  for (int i = 0; i < static_cast<int>(info.face_info_list.size()); i++) {
    // Get corresponding kf_id, index and projected_tri
    const auto& bestkf = faceid2bestkf[i];
    std::array<Eigen::Vector2f, 3> texture_tri, uv_tri;
    if (bestkf.kf_id < 0) {
      // face is not visible, set invalid value
      std::fill(uv_tri.begin(), uv_tri.end(), Eigen::Vector2f::Zero());

    } else {
      // Calc image position on tiled image
      int tile_pos_x = id2index[bestkf.kf_id] % x_tile_num;
      int tile_pos_y = id2index[bestkf.kf_id] / x_tile_num;
      int tex_pos_x = tile_pos_x * org_w;
      int tex_pos_y = tile_pos_y * org_h;
      for (int j = 0; j < 3; j++) {
        texture_tri[j].x() = bestkf.projected_tri[j].x() + tex_pos_x;
        texture_tri[j].y() = bestkf.projected_tri[j].y() + tex_pos_y;

        // Convert to 0-1 UV
        uv_tri[j].x() = (texture_tri[j].x() + 0.5f) / tex_w;
        uv_tri[j].y() = 1.0f - ((texture_tri[j].y() + 0.5f) / tex_h);
      }
    }
    uv.push_back(uv_tri[0]);
    uv.push_back(uv_tri[1]);
    uv.push_back(uv_tri[2]);

    int uv_size = static_cast<int>(uv.size());
    uv_indices.push_back(
        Eigen::Vector3i(uv_size - 3, uv_size - 2, uv_size - 1));
  }

  mesh->set_uv(uv);
  mesh->set_uv_indices(uv_indices);

  std::vector<ugu::ObjMaterial> materials(1);
  materials[0].name = option.texture_base_name;
  materials[0].diffuse_tex = texture;
  mesh->set_materials(materials);

  std::vector<int> material_ids(mesh->vertex_indices().size(), 0);
  mesh->set_material_ids(material_ids);

  return true;
}

template <typename T>
inline float EdgeFunction(const T& a, const T& b, const T& c) {
  return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]);
}

inline ugu::Vec3b BilinearInterpolation(float x, float y,
                                        const ugu::Image3b& image) {
  std::array<int, 2> pos_min = {{0, 0}};
  std::array<int, 2> pos_max = {{0, 0}};
  pos_min[0] = static_cast<int>(std::floor(x));
  pos_min[1] = static_cast<int>(std::floor(y));
  pos_max[0] = pos_min[0] + 1;
  pos_max[1] = pos_min[1] + 1;

  // really need these?
  if (pos_min[0] < 0.0f) {
    pos_min[0] = 0.0f;
  }
  if (pos_min[1] < 0.0f) {
    pos_min[1] = 0.0f;
  }
  if (image.cols <= pos_max[0]) {
    pos_max[0] = image.cols - 1;
  }
  if (image.rows <= pos_max[1]) {
    pos_max[1] = image.rows - 1;
  }

  float local_u = x - pos_min[0];
  float local_v = y - pos_min[1];

  // bilinear interpolation
  ugu::Vec3b color;
  for (int i = 0; i < 3; i++) {
    color[i] =
        (1.0f - local_u) * (1.0f - local_v) *
            image.at<ugu::Vec3b>(pos_min[1], pos_min[0])[i] +
        local_u * (1.0f - local_v) *
            image.at<ugu::Vec3b>(pos_max[1], pos_min[0])[i] +
        (1.0f - local_u) * local_v *
            image.at<ugu::Vec3b>(pos_min[1], pos_max[0])[i] +
        local_u * local_v * image.at<ugu::Vec3b>(pos_max[1], pos_max[0])[i];
  }

  return color;
}

bool PaddingSimple(ugu::Image3b* texture, ugu::Image1b* mask, int kernel) {
  ugu::Image3b org_texture;
  texture->copyTo(org_texture);
  ugu::Image1b org_mask;
  mask->copyTo(org_mask);

  int hk = kernel / 2;

  for (int j = hk; j < texture->rows - hk; j++) {
    for (int i = hk; i < texture->cols - hk; i++) {
      // Skip valid
      if (org_mask.at<unsigned char>(j, i) == 255) {
        continue;
      }

      std::vector<ugu::Vec3b> valid_pixels;
      for (int jj = -hk; jj <= hk; jj++) {
        int y = j + jj;
        for (int ii = -hk; ii <= hk; ii++) {
          int x = i + ii;
          if (org_mask.at<unsigned char>(y, x) == 255) {
            valid_pixels.push_back(org_texture.at<ugu::Vec3b>(y, x));
          }
        }
      }

      if (valid_pixels.empty()) {
        continue;
      }

      // Don't use Vec3b to avoid overflow
      ugu::Vec3f average_color{0, 0, 0};
      for (auto& p : valid_pixels) {
        average_color[0] += p[0];
        average_color[1] += p[1];
        average_color[2] += p[2];
      }

      average_color[0] /= valid_pixels.size();
      average_color[1] /= valid_pixels.size();
      average_color[2] /= valid_pixels.size();

      mask->at<unsigned char>(j, i) = 255;
      texture->at<ugu::Vec3b>(j, i)[0] = average_color[0];
      texture->at<ugu::Vec3b>(j, i)[1] = average_color[1];
      texture->at<ugu::Vec3b>(j, i)[2] = average_color[2];
    }
  }

  return true;
}

bool RasterizeTriangle(const std::array<Eigen::Vector2f, 3>& src_tri,
                       const ugu::Image3b& src,
                       const std::array<Eigen::Vector2f, 3>& target_tri,
                       ugu::Image3b* target, ugu::Image1b* mask) {
  // Area could be negative
  float area = EdgeFunction(target_tri[0], target_tri[1], target_tri[2]);
  if (std::abs(area) < std::numeric_limits<float>::min()) {
    area = area > 0 ? std::numeric_limits<float>::min()
                    : -std::numeric_limits<float>::min();
  }
  float inv_area = 1.0f / area;

  // Loop for bounding box of the target triangle
  int xmin =
      std::min({target_tri[0].x(), target_tri[1].x(), target_tri[2].x()}) - 1;
  xmin = std::max(0, std::min(xmin, target->cols - 1));
  int xmax =
      std::max({target_tri[0].x(), target_tri[1].x(), target_tri[2].x()}) + 1;
  xmax = std::max(0, std::min(xmax, target->cols - 1));

  int ymin =
      std::min({target_tri[0].y(), target_tri[1].y(), target_tri[2].y()}) - 1;
  ymin = std::max(0, std::min(ymin, target->rows - 1));
  int ymax =
      std::max({target_tri[0].y(), target_tri[1].y(), target_tri[2].y()}) + 1;
  ymax = std::max(0, std::min(ymax, target->rows - 1));

  for (int y = ymin; y <= ymax; y++) {
    for (int x = xmin; x <= xmax; x++) {
      Eigen::Vector2f pixel_sample(static_cast<float>(x),
                                   static_cast<float>(y));
      float w0 = EdgeFunction(target_tri[1], target_tri[2], pixel_sample);
      float w1 = EdgeFunction(target_tri[2], target_tri[0], pixel_sample);
      float w2 = EdgeFunction(target_tri[0], target_tri[1], pixel_sample);
      // Barycentric in the target triangle
      w0 *= inv_area;
      w1 *= inv_area;
      w2 *= inv_area;

      // Barycentric coordinate should be positive inside of the triangle
      // Skip outside of the target triangle
      if (w0 < 0 || w1 < 0 || w2 < 0) {
        continue;
      }

      // Barycentric to src image patch
      Eigen::Vector2f src_pos =
          w0 * src_tri[0] + w1 * src_tri[1] + w2 * src_tri[2];
      target->at<ugu::Vec3b>(y, x) =
          BilinearInterpolation(src_pos.x(), src_pos.y(), src);

      if (mask != nullptr) {
        mask->at<unsigned char>(y, x) = 255;
      }
    }
  }

  return true;
}

bool GenerateTextureOnOriginalUv(
    const std::vector<std::shared_ptr<ugu::Keyframe>>& keyframes,
    const ugu::VisibilityInfo& info, ugu::Mesh* mesh,
    const ugu::TextureMappingOption& option,
    const std::unordered_map<int, std::vector<ugu::FaceInfoPerKeyframe>>&
        bestkfid2faceid,
    const std::vector<ugu::FaceInfoPerKeyframe>& faceid2bestkf) {
  ugu::Image3b texture = ugu::Image3b::zeros(option.tex_h, option.tex_w);
  ugu::Image1b mask = ugu::Image1b::zeros(option.tex_h, option.tex_w);

  // Rasterization to original UV
  // Loop per face
  for (int i = 0; i < static_cast<int>(info.face_info_list.size()); i++) {
    // Get corresponding kf_id, index and projected_tri
    const auto& bestkf = faceid2bestkf[i];
    if (bestkf.kf_id < 0) {
      continue;
    }
    const auto& color = keyframes[bestkf.kf_id]->color;

    // Get triangle on target image
    std::array<Eigen::Vector2f, 3> target_tri_uv;
    std::array<Eigen::Vector2f, 3> target_tri;

    const std::array<Eigen::Vector2f, 3>& src_tri = bestkf.projected_tri;

    for (int j = 0; j < 3; j++) {
      target_tri_uv[j] = mesh->uv()[mesh->uv_indices()[i][j]];
      // TODO: Bilinear interpolation for float image coordinate
      target_tri[j].x() = target_tri_uv[j].x() * option.tex_w - 0.5f;
      target_tri[j].y() = (1.0f - target_tri_uv[j].y()) * option.tex_h - 0.5f;
    }

    RasterizeTriangle(src_tri, color, target_tri, &texture, &mask);
  }

  // Add padding for atlas boundaries to avoid invalid color bleeding at
  // rendering
  PaddingSimple(&texture, &mask, option.padding_kernel);

  std::vector<ugu::ObjMaterial> materials(1);
  materials[0].name = option.texture_base_name;
  materials[0].diffuse_tex = texture;
  mesh->set_materials(materials);

  std::vector<int> material_ids(mesh->vertex_indices().size(), 0);
  mesh->set_material_ids(material_ids);

  return true;
}

bool GenerateSimpleTrianglesTextureAndUv(
    const std::vector<std::shared_ptr<ugu::Keyframe>>& keyframes,
    const ugu::VisibilityInfo& info, ugu::Mesh* mesh,
    const ugu::TextureMappingOption& option,
    const std::unordered_map<int, std::vector<ugu::FaceInfoPerKeyframe>>&
        bestkfid2faceid,
    const std::vector<ugu::FaceInfoPerKeyframe>& faceid2bestkf) {
  // Padding must be at least 2
  // to pad right/left and up/down
  const int padding_tri = 2;

  int rect_num = static_cast<int>((mesh->vertex_indices().size() + 1) / 2);
  int pix_per_rect = option.tex_h * option.tex_w / rect_num;
  if (pix_per_rect < 6) {
    return false;
  }
  int max_rect_edge_len = 100;
  int sq_len =
      std::min(static_cast<int>(std::sqrt(pix_per_rect)), max_rect_edge_len);
  /*
   * example. rect_w = 4
   * * is padding on diagonal (fixed)
   * + is upper triangle, - is lower triangle
   * ++++**
   * +++**-
   * ++**--
   * +**---
   * **----
   *
   */

  int max_rect_num = (option.tex_w / (sq_len + 2 + padding_tri)) *
                     (option.tex_h / (sq_len + 1 + padding_tri));
  while (max_rect_num < rect_num) {
    sq_len--;
    if (sq_len < 3) {
      return false;
    }
    max_rect_num = (option.tex_w / (sq_len + 2 + padding_tri)) *
                   (option.tex_h / (sq_len + 1 + padding_tri));
  }

  int rect_w = sq_len + 2;
  int rect_h = sq_len + 1;

  ugu::Image3b texture = ugu::Image3b::zeros(option.tex_h, option.tex_w);
  ugu::Image1b mask = ugu::Image1b::zeros(option.tex_h, option.tex_w);

  // Loop per face
  int rect_w_num = option.tex_w / (rect_w + padding_tri);
  // int rect_h_num = option.tex_h / (rect_h + padding_tri);
  std::vector<Eigen::Vector2f> uv;
  std::vector<Eigen::Vector3i> uv_indices;
  for (int i = 0; i < static_cast<int>(info.face_info_list.size()); i++) {
    // Get corresponding kf_id, index and projected_tri
    const auto& bestkf = faceid2bestkf[i];
    if (bestkf.kf_id < 0) {
      uv.push_back(Eigen::Vector2f::Zero());
      uv.push_back(Eigen::Vector2f::Zero());
      uv.push_back(Eigen::Vector2f::Zero());
      int uv_size = static_cast<int>(uv.size());
      uv_indices.push_back(
          Eigen::Vector3i(uv_size - 3, uv_size - 2, uv_size - 1));
      continue;
    }
    const auto& color = keyframes[bestkf.kf_id]->color;

    const std::array<Eigen::Vector2f, 3>& src_tri = bestkf.projected_tri;

    int rect_id = i / 2;
    int rect_x = rect_id % rect_w_num;
    int rect_y = rect_id / rect_w_num;

    std::array<Eigen::Vector2f, 3> target_tri, target_tri_uv;
    bool lower = i % 2 == 0;
    if (lower) {
      int rect_x_min = (rect_w + padding_tri) * rect_x + 2;
      int rect_x_max = rect_x_min + sq_len - 1;
      int rect_y_min = (rect_h + padding_tri) * rect_y;
      int rect_y_max = rect_y_min + sq_len - 1;

      target_tri[0] = Eigen::Vector2f{rect_x_min, rect_y_min};
      target_tri[1] = Eigen::Vector2f{rect_x_max, rect_y_min};
      target_tri[2] = Eigen::Vector2f{rect_x_max, rect_y_max};
    } else {
      int rect_x_min = (rect_w + padding_tri) * rect_x;
      int rect_x_max = rect_x_min + sq_len - 1;
      int rect_y_min = (rect_h + padding_tri) * rect_y + 1;
      int rect_y_max = rect_y_min + sq_len - 1;

      target_tri[0] = Eigen::Vector2f{rect_x_min, rect_y_min};
      target_tri[1] = Eigen::Vector2f{rect_x_min, rect_y_max};
      target_tri[2] = Eigen::Vector2f{rect_x_max, rect_y_max};
    }

    RasterizeTriangle(src_tri, color, target_tri, &texture, &mask);

    for (int j = 0; j < 3; j++) {
      target_tri_uv[j].x() = (target_tri[j].x() + 0.5f) / option.tex_w;
      target_tri_uv[j].y() = 1.0f - ((target_tri[j].y() + 0.5f) / option.tex_h);
    }

    uv.push_back(target_tri_uv[0]);
    uv.push_back(target_tri_uv[1]);
    uv.push_back(target_tri_uv[2]);
    int uv_size = static_cast<int>(uv.size());
    uv_indices.push_back(
        Eigen::Vector3i(uv_size - 3, uv_size - 2, uv_size - 1));
  }

  // Add padding for atlas boundaries to avoid invalid color bleeding at
  // rendering
  PaddingSimple(&texture, &mask, 3);

  mesh->set_uv(uv);
  mesh->set_uv_indices(uv_indices);

  std::vector<ugu::ObjMaterial> materials(1);
  materials[0].name = option.texture_base_name;
  materials[0].diffuse_tex = texture;
  mesh->set_materials(materials);

  std::vector<int> material_ids(mesh->vertex_indices().size(), 0);
  mesh->set_material_ids(material_ids);

  return true;
}

bool SimpleTextureMapping(
    const std::vector<std::shared_ptr<ugu::Keyframe>>& keyframes,
    const ugu::VisibilityInfo& info, ugu::Mesh* mesh,
    const ugu::TextureMappingOption& option) {
  // Init
  std::unordered_map<int, std::vector<ugu::FaceInfoPerKeyframe>>
      bestkfid2faceid;
  for (int i = 0; i < static_cast<int>(keyframes.size()); i++) {
    bestkfid2faceid.emplace(keyframes[i]->id,
                            std::vector<ugu::FaceInfoPerKeyframe>());
  }
  std::vector<ugu::FaceInfoPerKeyframe> faceid2bestkf(
      info.face_info_list.size());

  std::function<ugu::FaceInfoPerKeyframe(const ugu::FaceInfo&)> get_best_kfid;
  if (option.criteria == ugu::ViewSelectionCriteria::kMinViewingAngle) {
    get_best_kfid = [](const ugu::FaceInfo& info) -> ugu::FaceInfoPerKeyframe {
      return info.visible_keyframes[info.min_viewing_angle_index];
    };
  } else if (option.criteria == ugu::ViewSelectionCriteria::kMinDistance) {
    get_best_kfid = [](const ugu::FaceInfo& info) -> ugu::FaceInfoPerKeyframe {
      return info.visible_keyframes[info.min_distance_index];
    };
  } else if (option.criteria == ugu::ViewSelectionCriteria::kMaxArea) {
    get_best_kfid = [](const ugu::FaceInfo& info) -> ugu::FaceInfoPerKeyframe {
      return info.visible_keyframes[info.max_area_index];
    };
  } else {
    ugu::LOGE("ViewSelectionCriteria %d is not implemented\n", option.criteria);
    return false;
  }

  // Select the best kf_id for each face in iterms of criteria
  for (int i = 0; i < static_cast<int>(info.face_info_list.size()); i++) {
    const auto& face_info = info.face_info_list[i];

    if (face_info.visible_keyframes.empty()) {
      faceid2bestkf[i].kf_id = -1;
      faceid2bestkf[i].face_id = i;
      continue;
    }

    // Get the best kf id and projected_tri
    ugu::FaceInfoPerKeyframe bestkf = get_best_kfid(face_info);
    bestkfid2faceid[bestkf.kf_id].push_back(bestkf);
    faceid2bestkf[i] = bestkf;
  }

  bool ret_tex_gen = false;
  if (option.uv_type == ugu::OutputUvType::kGenerateSimpleTile) {
    ret_tex_gen = GenerateSimpleTileTextureAndUv(
        keyframes, info, mesh, option, bestkfid2faceid, faceid2bestkf);
  } else if (option.uv_type == ugu::OutputUvType::kGenerateSimpleTriangles) {
    ret_tex_gen = GenerateSimpleTrianglesTextureAndUv(
        keyframes, info, mesh, option, bestkfid2faceid, faceid2bestkf);

  } else if (option.uv_type == ugu::OutputUvType::kUseOriginalMeshUv) {
    if (mesh->uv().empty() ||
        mesh->uv_indices().size() != mesh->vertex_indices().size()) {
      ugu::LOGE(
          "OutputUvType kUseOriginalMeshUv is specified but UV on mesh is "
          "invalid\n");
      return false;
    }
    ret_tex_gen = GenerateTextureOnOriginalUv(keyframes, info, mesh, option,
                                              bestkfid2faceid, faceid2bestkf);

  } else {
    ugu::LOGE("OutputUvType %d is not implemented\n", option.uv_type);
    return false;
  }

  return ret_tex_gen;
}

}  // namespace

namespace ugu {

bool TextureMapping(const std::vector<std::shared_ptr<Keyframe>>& keyframes,
                    const VisibilityInfo& info, Mesh* mesh,
                    const TextureMappingOption& option) {
  if (option.type == TextureMappingType::kSimpleProjection) {
    return SimpleTextureMapping(keyframes, info, mesh, option);
  }
  LOGE("TextureMappingType %d is not implemented", option.type);
  return false;
}

}  // namespace ugu
