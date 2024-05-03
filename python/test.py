import sys
import os
import os
this_abs_path = os.path.abspath(os.path.join(os.path.abspath(__file__), ".."))
sys.path.insert(0, os.path.abspath(os.path.join(this_abs_path, "../lib/Release")))
# print(os.listdir(sys.path[0]))
import ugu_py
import numpy as np
import json
import obj_io


def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.

    Parameters
    ----------
    src : (M, N) array_like
        Source coordinates.
    dst : (M, N) array_like
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.

    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.

    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`

    """
    src = np.asarray(src)
    dst = np.asarray(dst)

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T


def load_lmk(path, verts, indices):
    src_fids = []
    src_barys = []
    with open(path) as fp:
        j = json.load(fp)
        for p in j:
            src_fids.append(int(p[0]))
            src_barys.append([float(p[1]), float(p[2])])
    src_fids = np.array(src_fids, dtype=np.int32)
    src_barys = np.array(src_barys, dtype=np.float32)
    src_lmks = []
    for fid, uv in zip(src_fids, src_barys):
        src_lmks.append(verts[indices[fid][0]] * uv[0] + verts[indices[fid][1]] * uv[1] + verts[indices[fid][2]] * (1.0 - uv[0] - uv[1]))
    src_lmks = np.array(src_lmks, dtype=np.float32)
    return src_fids, src_barys, src_lmks


src = obj_io.loadObj("../data/face/ict-facekit_tri.obj")
dst = obj_io.loadObj("../data/face/max-planck.obj")

src_fids, src_barys, src_lmks = load_lmk(os.path.join(this_abs_path, "../data/face/ict-facekit_lmk.json"), src.verts, src.indices)
dst_fids, dst_barys, dst_lmks = load_lmk(os.path.join(this_abs_path, "../data/face/max-planck_lmk.json"), dst.verts, dst.indices)

T = umeyama(src_lmks, dst_lmks, True)
R = T[:3, :3]
t = T[:3, 3]

src_verts = (R @ src.verts.T).T + t
src.verts = src_verts
obj_io.saveObj("umeyama.obj", src)

lmk_w = np.ones(len(src_fids), dtype=np.float32)
lmk_w[9] = 10.0

src_movable_face_ids = []
with open(os.path.join(this_abs_path, "../data/face/ict-facekit_movable_faces.txt"), "r") as fp:
    for line in fp:
        src_movable_face_ids.append(int(line.rstrip()))
src_ignore_face_ids = set(range(len(src.indices)))
for fid in src_movable_face_ids:
    src_ignore_face_ids.remove(fid)
src_ignore_face_ids = np.array(list(src_ignore_face_ids), dtype=np.int32)
params = ugu_py.NonrigidIcpParams(True, False, False, 0.075, 0.05, 60.0, 45.0, 2.0, 0.1, 10.0, 20, 10)
deformed_verts = ugu_py.NonrigidIcp(src_verts, src.indices, src_fids, src_barys, lmk_w, src_ignore_face_ids, dst.verts, dst.indices, dst_lmks, params)

src.verts = deformed_verts
obj_io.saveObj("deformed.obj", src)
