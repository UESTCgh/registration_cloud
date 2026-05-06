"""
两两配准演示：加载数据，运行 FPFH+ICP，评估点云厚度，保存截图到 result/

用法:
    python3 scripts/pairwise_demo.py
"""

import sys
import os
import shutil
import copy
import argparse
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import register, estimate_density, compute_thickness, save_screenshot

DATA_DIR = "data"
OUT_DIR = "result"

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/default.yaml")
args = parser.parse_args()

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)


def thickness_report(src_aligned, dst, label, voxel_size):
    """计算并打印配准后的厚度指标"""
    merged = src_aligned + dst
    t = compute_thickness(merged, k=max(20, int(1.0 / max(voxel_size, 0.001))))

    print(f"    —— {label} 配准后点云厚度 ——")
    print(f"      RMS: {t['rms']:.6f}  均值: {t['mean']:.6f}")
    print(f"      中位数: {t['median']:.6f}  标准差: {t['std']:.6f}  最大值: {t['max']:.6f}")
    return t


# ============================================================
# 第一步：密度估算
# ============================================================

print("=" * 70)
print("第一步：计算所有点云的密度")
print("=" * 70)

pcd_files = [
    "Dino.pcd", "Elephant.pcd", "Chair.pcd",
    "1.pcd", "2.pcd",
    "cloud_bin_0.pcd", "cloud_bin_1.pcd",
    "online_source.pcd",
    "lab2_sfm.ply", "lab2_kinect.ply"
]

for f in pcd_files:
    fp = os.path.join(DATA_DIR, f)
    if os.path.exists(fp):
        p = o3d.io.read_point_cloud(fp)
        d = estimate_density(p)
        print(f"  {f:25s} -> 点数: {len(p.points):8d}  密度: {d:.6f}")

# ============================================================
# 第二步：两两配准 + 厚度评估
# ============================================================

print("\n" + "=" * 70)
print("第二步：两两配准 + 点云厚度评估")
print("=" * 70)

scenarios = [
    ("cloud_bin_0.pcd", "cloud_bin_1.pcd", 0.05, 0.02, 64, "cloud_bin"),
    ("1.pcd", "2.pcd", 0.005, 0.002, 64, "1_and_2"),
    ("lab2_sfm.ply", "lab2_kinect.ply", 0.03, 0.03, 64, "lab2"),
]

for src_f, dst_f, vs, th, max_iter, label in scenarios:
    print(f"\n--- {label}: {src_f} + {dst_f} ---")
    print(f"    voxel_size={vs}, threshold={th}")

    src = o3d.io.read_point_cloud(os.path.join(DATA_DIR, src_f))
    dst = o3d.io.read_point_cloud(os.path.join(DATA_DIR, dst_f))

    if label == "lab2":
        src = src.voxel_down_sample(0.008)

    # 配准前厚度（两片叠在一起未对齐）
    t_before = compute_thickness(src + dst, k=max(20, int(1.0 / max(vs, 0.001))))
    print(f"    配准前厚度 — RMS: {t_before['rms']:.6f}  均值: {t_before['mean']:.6f}")

    src.paint_uniform_color([1, 0.706, 0])
    dst.paint_uniform_color([0, 0.651, 0.929])
    save_screenshot([src, dst], os.path.join(OUT_DIR, f"{label}_before.png"),
                    f"{label} 配准前")

    result = register(src, dst, vs, th, max_iterations=max_iter)
    src_aligned = copy.deepcopy(src).transform(result.transformation)

    print(f"    FGR+ICP fitness={result.fitness:.4f}  RMSE(inlier)={result.inlier_rmse:.6f}")

    # 配准后厚度
    thickness_report(src_aligned, dst, label, vs)

    print(f"    变换矩阵:\n{result.transformation}")

    save_screenshot([src_aligned, dst], os.path.join(OUT_DIR, f"{label}_after.png"),
                    f"{label} 配准后")

# ============================================================
# 第三步：同源点云人为变换后配准 + 厚度验证
# ============================================================

print("\n" + "=" * 70)
print("第三步：同源点云人为变换后配准 + 厚度验证")
print("=" * 70)

peizhun_cases = [
    ("Dino.pcd", 0.05, 0.02, 64, 0.01),
    ("Elephant.pcd", 0.05, 0.02, 64, 0.01),
    ("Chair.pcd", 16, 16, 300, 1.0),
]

for pcd_f, vs, th, max_iter, verify_offset in peizhun_cases:
    print(f"\n--- {pcd_f} (voxel={vs}, threshold={th}) ---")

    src = o3d.io.read_point_cloud(os.path.join(DATA_DIR, pcd_f))
    T = np.eye(4)
    T[:3, :3] = src.get_rotation_matrix_from_xyz(
        (np.pi / 3, np.pi / 5, -np.pi / 6))
    T[0, 3] = 1.5
    T[1, 3] = 3
    dst = copy.deepcopy(src).transform(T)

    src.paint_uniform_color([0, 1, 0])
    dst.paint_uniform_color([0, 0, 1])

    save_screenshot([src, dst],
                    os.path.join(OUT_DIR, f"{pcd_f.split('.')[0].lower()}_before.png"),
                    f"{pcd_f} 配准前")

    result = register(src, dst, vs, th, max_iterations=max_iter)
    src_aligned = copy.deepcopy(src).transform(result.transformation)

    print(f"    FGR+ICP fitness={result.fitness:.4f}  RMSE(inlier)={result.inlier_rmse:.6f}")
    thickness_report(src_aligned, dst, pcd_f, vs)

    # 平移验证
    T_v = np.eye(4)
    T_v[1, 3] = verify_offset
    src_verify = copy.deepcopy(src_aligned).transform(T_v)
    save_screenshot([dst, src_verify],
                    os.path.join(OUT_DIR, f"{pcd_f.split('.')[0].lower()}_after.png"),
                    f"{pcd_f} 配准后(平移验证)")

print(f"\n完成！结果保存在 {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
