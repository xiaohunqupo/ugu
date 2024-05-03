#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "ugu/registration/nonrigid.h"

namespace nb = nanobind;

struct NonrigidIcpParams {
  bool check_self_itersection = false;
  bool dst_check_geometry_border = false;
  bool src_check_geometry_border = false;
  double start_dist_th = 0.075;
  double end_dist_th = 0.05;
  double start_deg_th = 60.0;
  double end_deg_th = 45.0;
  double max_alpha = 2.0;
  double min_alpha = 0.1;
  double gamma = 10.0;
  int step = 20;
  int step_ignore_lmks = 10;
};

template <typename T>
std::vector<Eigen::Vector<T, 3>> Convert3dvec(
    nb::ndarray<T, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> arg) {
  auto v = arg.view<T, nb::ndim<2>>();
  std::vector<Eigen::Vector<T, 3>> converted(v.shape(0));

  assert(v.shape(1) == 3);

  for (size_t i = 0; i < v.shape(0); ++i) {
    for (size_t j = 0; j < v.shape(1); ++j) {
      converted[i][j] = v(i, j);
    }
  }

  return converted;
}

template <typename T>
void Convert3dvec(
    std::vector<Eigen::Vector<T, 3>> vec,
    nb::ndarray<T, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> arg) {
  auto v = arg.view<T, nb::ndim<2>>();

  assert(v.shape(1) == 3);

  for (size_t i = 0; i < v.shape(0); ++i) {
    for (size_t j = 0; j < v.shape(1); ++j) {
      v(i, j) = vec[i][j];
    }
  }
}

nb::ndarray<float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> NonrigidIcp(
    nb::ndarray<float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>
        src_verts,
    nb::ndarray<int, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>
        src_indices,
    nb::ndarray<int, nb::shape<-1>, nb::c_contig, nb::device::cpu> src_lmk_fids,
    nb::ndarray<float, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu>
        src_lmk_barys,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig, nb::device::cpu>
        lmk_weights,
    nb::ndarray<int, nb::shape<-1>, nb::c_contig, nb::device::cpu>
        src_ignore_face_ids,
    nb::ndarray<float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>
        dst_verts,
    nb::ndarray<int, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>
        dst_indices,
    nb::ndarray<float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>
        dst_lmks,
    NonrigidIcpParams params) {
  ugu::NonRigidIcp nicp;
  nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> result;

  ugu::MeshPtr src_mesh = ugu::Mesh::Create();
  src_mesh->set_vertices(Convert3dvec(src_verts));
  src_mesh->set_vertex_indices(Convert3dvec(src_indices));
  src_mesh->CalcNormal();

  ugu::MeshPtr dst_mesh = ugu::Mesh::Create();
  dst_mesh->set_vertices(Convert3dvec(dst_verts));
  dst_mesh->set_vertex_indices(Convert3dvec(dst_indices));
  dst_mesh->CalcNormal();

  nicp.SetSrc(*src_mesh);
  std::vector<double> betas(lmk_weights.shape(0));
  for (size_t i = 0; i < lmk_weights.shape(0); i++) {
    betas[i] = lmk_weights(i);
  }

  std::vector<ugu::PointOnFace> src_landmarks;
  if (src_lmk_fids.shape(0) == src_lmk_barys.shape(0)) {
    for (size_t i = 0; i < src_lmk_fids.shape(0); i++) {
      ugu::PointOnFace pof;
      pof.fid = src_lmk_fids(i);
      pof.u = src_lmk_barys(i, 0);
      pof.v = src_lmk_barys(i, 1);
      src_landmarks.push_back(pof);
    }
  }

  nicp.SetSrcLandmarks(src_landmarks, betas);
  nicp.SetDst(*dst_mesh);
  std::vector<Eigen::Vector3f> dst_landmark_positions = Convert3dvec(dst_lmks);
  nicp.SetDstLandmarkPositions(dst_landmark_positions);

  float start_dist = static_cast<float>(params.end_dist_th);
  float end_dist = static_cast<float>(params.end_dist_th);
  nicp.SetCorrespDistTh(start_dist);
  std::set<uint32_t> ignore_face_ids;
  for (size_t i = 0; i < src_ignore_face_ids.shape(0); i++) {
    ignore_face_ids.insert(static_cast<uint32_t>(src_ignore_face_ids(i)));
  }

  nicp.SetIgnoreFaceIds(ignore_face_ids);

  float start_deg = static_cast<float>(params.start_deg_th);
  float end_deg = static_cast<float>(params.end_deg_th);
  nicp.Init(params.check_self_itersection, ugu::radians(start_deg),
            params.dst_check_geometry_border, params.src_check_geometry_border);

  double max_alpha = params.max_alpha;
  double min_alpha = params.min_alpha;
  double gamma = params.gamma;
  int step = params.step;
  ugu::MeshPtr deformed;
  for (int i = 1; i <= step; ++i) {
    double ratio = static_cast<double>(i) / static_cast<double>(step);

    double alpha = max_alpha - (max_alpha - min_alpha) * ratio;

    // Ignore landmark at later stages
    if (i > params.step_ignore_lmks) {
      std::fill(betas.begin(), betas.end(), 0.0);
      nicp.SetSrcLandmarks(src_landmarks, betas);
    }

    ugu::LOGI("Iteration %d with alpha %f\n", i, alpha);

    nicp.SetCorrespNormalTh(static_cast<float>(
        ugu::radians(start_deg - ratio * (start_deg - end_deg))));

    nicp.SetCorrespDistTh(
        static_cast<float>(start_dist - ratio * (start_dist - end_dist)));

    nicp.Registrate(alpha, gamma);

    // if (i % 1 == 0) {
    //   deformed = nicp.GetDeformedSrc();
    //   deformed->WriteObj(out_dir, "2_nonrigid_" + ugu::zfill(i, 3));
    // }
  }
  deformed = nicp.GetDeformedSrc();
  Convert3dvec(deformed->vertices(), src_verts);

  return src_verts;
}

NB_MODULE(ugu_py, m) {
  nb::class_<NonrigidIcpParams>(m, "NonrigidIcpParams")
      .def(nb::init<const bool &, const bool &, const bool &, const double &,
                    const double &, const double &, const double &,
                    const double &, const double &, const double &, const int &,
                    const int &>())
      .def_rw("check_self_itersection",
              &NonrigidIcpParams::check_self_itersection)
      .def_rw("dst_check_geometry_border",
              &NonrigidIcpParams::dst_check_geometry_border)
      .def_rw("src_check_geometry_border",
              &NonrigidIcpParams::src_check_geometry_border)
      .def_rw("start_dist_th", &NonrigidIcpParams::start_dist_th)
      .def_rw("end_dist_th", &NonrigidIcpParams::end_dist_th)
      .def_rw("start_deg_th", &NonrigidIcpParams::start_deg_th)
      .def_rw("end_deg_th", &NonrigidIcpParams::end_deg_th)
      .def_rw("max_alpha", &NonrigidIcpParams::max_alpha)
      .def_rw("min_alpha", &NonrigidIcpParams::min_alpha)
      .def_rw("gamma", &NonrigidIcpParams::gamma)
      .def_rw("step", &NonrigidIcpParams::step)
      .def_rw("step_ignore_lmks", &NonrigidIcpParams::step_ignore_lmks);
  m.def("NonrigidIcp", &NonrigidIcp);
}
