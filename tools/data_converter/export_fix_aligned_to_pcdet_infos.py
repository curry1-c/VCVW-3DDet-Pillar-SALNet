#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export VCVW trainval_fix_aligned (pcd + txt labels with quaternion) to an OpenPCDet CustomDataset-style folder:
  OUT_ROOT/
    infos_train.pkl
    infos_val.pkl
    id_map.txt
    ImageSets/train.txt
    ImageSets/val.txt
    training/velodyne/000000.bin ...
    points -> training/velodyne   (symlink)
    velodyne -> training/velodyne (symlink)

IMPORTANT:
- This script writes gt_boxes_lidar in the SAME coordinate frame as points AFTER your get_lidar() axis transform:
    x' = z,  y' = -x,  z' = -y
- So during training set DISABLE_GT_AXISFIX=1 (do NOT axis-fix gt again in dataset),
  but you can keep the points transform in get_lidar().

Label format per line (space-separated):
  Class tx ty tz sx sy sz qw qx qy qz
where (tx,ty,tz) is box center in original (camera-like) coords, (sx,sy,sz) are box sizes along x,y,z,
and quaternion (qw,qx,qy,qz) gives box rotation in the same original coords.

Env vars:
  VCVW_ROOT        default: /mnt/f/Data/VCVW3D_vehicle_dataset/trainval_fix_aligned
  OUT_ROOT         default: data/vcvw_fix_aligned_pcdet
  MAX_PER_SPLIT    default: 0  (0 means all)
  PERM             default: 0,1,2
  SIGN             default: 1,1,1
  CENTER_SCALE     default: 0.33
  SIZE_SCALE       default: 1.0
"""

import os
import math
import pickle
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except Exception as e:
    raise RuntimeError("open3d is required to read .pcd files. Install it in your pcdet env.") from e


# ---------------- user knobs ----------------
VCVW_ROOT = Path(os.environ.get("VCVW_ROOT", "/mnt/f/Data/VCVW3D_vehicle_dataset/trainval_fix_aligned"))
OUT_ROOT = Path(os.environ.get("OUT_ROOT", "data/vcvw_fix_aligned_pcdet"))
MAX_PER_SPLIT = int(os.environ.get("MAX_PER_SPLIT", "0"))

PERM = tuple(int(x) for x in os.environ.get("PERM", "0,1,2").split(","))
SIGN = np.array([float(x) for x in os.environ.get("SIGN", "1,1,1").split(",")], dtype=np.float32)

CENTER_SCALE = float(os.environ.get("CENTER_SCALE", "0.33"))
SIZE_SCALE = float(os.environ.get("SIZE_SCALE", "1.0"))

# points axis transform you already use in get_lidar():
# x' = z, y' = -x, z' = -y
A_AXISFIX = np.array([[0, 0, 1],
                      [-1, 0, 0],
                      [0, -1, 0]], dtype=np.float64)


def _quat_to_rot(qw, qx, qy, qz):
    """Quaternion -> rotation matrix. Quaternion can be unnormalized."""
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n < 1e-8:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)]
    ], dtype=np.float64)
    return R


def _perm_sign_matrix(perm, sign):
    """Build orthonormal matrix M so that v' = M v equals v'[i] = sign[i] * v[perm[i]]."""
    P = np.zeros((3, 3), dtype=np.float64)
    for new_i, old_i in enumerate(perm):
        P[new_i, old_i] = 1.0
    S = np.diag(sign.astype(np.float64))
    return S @ P


def _yaw_from_rot(R):
    """Yaw around +Z axis from rotation matrix (standard lidar yaw)."""
    return float(math.atan2(R[1, 0], R[0, 0]))


def load_labels(lbl_path: Path):
    """
    Return:
      names: (N,) object names (str)
      boxes: (N,7) [x,y,z,dx,dy,dz,yaw] in AXISFIX frame (same as points after your get_lidar transform)
    """
    names = []
    boxes = []

    M = _perm_sign_matrix(PERM, SIGN)            # raw -> perm/sign frame
    T = A_AXISFIX @ M                             # raw -> axisfix frame

    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ss = line.split()
            if len(ss) < 11:
                # tolerate bad lines
                continue

            cls = ss[0]
            tx, ty, tz, sx, sy, sz, qw, qx, qy, qz = map(float, ss[1:11])

            # center and size in raw label coords
            c_raw = np.array([tx, ty, tz], dtype=np.float64)
            s_raw = np.array([sx, sy, sz], dtype=np.float64)

            # apply PERM/SIGN + scales
            c1 = (M @ c_raw) * CENTER_SCALE     # (3,)
            s1 = (np.array([s_raw[PERM[0]], s_raw[PERM[1]], s_raw[PERM[2]]], dtype=np.float64) * SIZE_SCALE)

            # axisfix center: c2 = A * c1
            c2 = (A_AXISFIX @ c1.reshape(3, 1)).reshape(3)

            # axisfix size: (dx,dy,dz) -> (dz,dx,dy) after the above A mapping
            # because x' <- z, y' <- x, z' <- y (sign doesn't matter for lengths)
            s2 = np.array([s1[2], s1[0], s1[1]], dtype=np.float64)

            # rotation: raw -> perm/sign -> axisfix
            R_raw = _quat_to_rot(qw, qx, qy, qz)
            R1 = M @ R_raw @ M.T
            R2 = A_AXISFIX @ R1 @ A_AXISFIX.T
            yaw = _yaw_from_rot(R2)

            names.append(cls)
            boxes.append([c2[0], c2[1], c2[2], s2[0], s2[1], s2[2], yaw])

    if len(boxes) == 0:
        return np.array(names), np.zeros((0, 7), dtype=np.float32)
    return np.array(names), np.array(boxes, dtype=np.float32)


def read_split_ids(root: Path):
    """
    Return train_ids, val_ids as list[str] (UIDs).
    Priority:
      1) root/labels_train + root/labels_val (txt files)
      2) root/ImageSets/train.txt + val.txt
      3) root/train.txt + val.txt
      4) fallback: split all label filenames 80/20
    """
    # 1) labels_train/labels_val
    lt = root / "labels_train"
    lv = root / "labels_val"
    if lt.exists() and lv.exists():
        train_ids = sorted([p.stem for p in lt.glob("*.txt")])
        val_ids = sorted([p.stem for p in lv.glob("*.txt")])
        if train_ids and val_ids:
            return train_ids, val_ids

    # 2) ImageSets
    imgsets = root / "ImageSets"
    cand = [
        (imgsets / "train.txt", imgsets / "val.txt"),
        (root / "train.txt", root / "val.txt"),
    ]
    for a, b in cand:
        if a.exists() and b.exists():
            train_ids = [x.strip() for x in a.read_text().splitlines() if x.strip()]
            val_ids = [x.strip() for x in b.read_text().splitlines() if x.strip()]
            # strip extensions if present
            train_ids = [Path(x).stem for x in train_ids]
            val_ids = [Path(x).stem for x in val_ids]
            return train_ids, val_ids

    # 4) fallback: split all labels
    lbl_dir = root / "labels"
    all_ids = sorted([p.stem for p in lbl_dir.glob("*.txt")])
    if not all_ids:
        raise RuntimeError(f"No labels found in {lbl_dir}")
    n = int(len(all_ids) * 0.8)
    return all_ids[:n], all_ids[n:]


def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    pcd_dir = VCVW_ROOT / "pcd"
    lbl_dir = VCVW_ROOT / "labels"
    if not pcd_dir.exists():
        raise RuntimeError(f"PCD dir not found: {pcd_dir}")
    if not lbl_dir.exists():
        raise RuntimeError(f"Label dir not found: {lbl_dir}")

    train_ids, val_ids = read_split_ids(VCVW_ROOT)

    if MAX_PER_SPLIT > 0:
        train_ids = train_ids[:MAX_PER_SPLIT]
        val_ids = val_ids[:MAX_PER_SPLIT]

    out_velo = OUT_ROOT / "training" / "velodyne"
    out_imgsets = OUT_ROOT / "ImageSets"
    mkdir(out_velo)
    mkdir(out_imgsets)

    # symlinks for CustomDataset convenience
    try:
        for ln in ["points", "velodyne"]:
            link = OUT_ROOT / ln
            if link.exists() or link.is_symlink():
                continue
            link.symlink_to(Path("training/velodyne"))
    except Exception:
        pass

    id_map_path = OUT_ROOT / "id_map.txt"
    infos_train = []
    infos_val = []

    def dump_one(uid: str, idx_int: int, split: str):
        idx = f"{idx_int:06d}"
        pcd_path = pcd_dir / f"{uid}.pcd"
        lbl_path = lbl_dir / f"{uid}.txt"

        if not pcd_path.exists():
            raise FileNotFoundError(f"Missing PCD: {pcd_path}")
        if not lbl_path.exists():
            raise FileNotFoundError(f"Missing label: {lbl_path}")

        pcd = o3d.io.read_point_cloud(str(pcd_path))
        pts = np.asarray(pcd.points).astype(np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise RuntimeError(f"Bad PCD points: {pcd_path}, shape={pts.shape}")

        # write .bin (x,y,z,intensity=0) in RAW coords
        pts4 = np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)], axis=1)
        out_bin = out_velo / f"{idx}.bin"
        pts4.tofile(str(out_bin))

        names, gt_boxes = load_labels(lbl_path)

        info = {
            "point_cloud": {"lidar_idx": idx},
            "annos": {
                "name": names,
                "gt_boxes_lidar": gt_boxes
            }
        }
        return idx, info

    # write mapping + infos
    with open(id_map_path, "w", encoding="utf-8") as f_map:
        base = 0
        print(f"[train] total={len(train_ids)}")
        for i, uid in enumerate(train_ids):
            if i % 5 == 0:
                print(f"[train] {i}/{len(train_ids)} uid={uid}")
            idx, info = dump_one(uid, base + i, "train")
            f_map.write(f"{idx} {uid}\n")
            infos_train.append(info)

        base = len(train_ids)
        print(f"[val] total={len(val_ids)} (start_idx={base})")
        for i, uid in enumerate(val_ids):
            if i % 5 == 0:
                print(f"[val] {i}/{len(val_ids)} uid={uid}")
            idx, info = dump_one(uid, base + i, "val")
            f_map.write(f"{idx} {uid}\n")
            infos_val.append(info)

    # write ImageSets
    (out_imgsets / "train.txt").write_text("\n".join([f"{i:06d}" for i in range(0, len(train_ids))]) + "\n")
    (out_imgsets / "val.txt").write_text("\n".join([f"{i:06d}" for i in range(len(train_ids), len(train_ids)+len(val_ids))]) + "\n")

    # write infos
    with open(OUT_ROOT / "infos_train.pkl", "wb") as f:
        pickle.dump(infos_train, f, protocol=4)
    with open(OUT_ROOT / "infos_val.pkl", "wb") as f:
        pickle.dump(infos_val, f, protocol=4)

    print("[OK] train ->", OUT_ROOT / "infos_train.pkl", "samples=", len(infos_train))
    print("[OK] val   ->", OUT_ROOT / "infos_val.pkl", "samples=", len(infos_val))
    print("[DONE]", OUT_ROOT)


if __name__ == "__main__":
    main()
