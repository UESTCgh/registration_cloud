"""实际多视角点云增量拼接（不做数据生成，直接加载已有扫描）

用法:
    python3 scripts/real_multi_view.py --voxel 0.1 --threshold 0.05
"""

import sys
import os
import shutil
import argparse
import json
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import MultiViewStitcher, estimate_density, compute_thickness, save_screenshot

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="data/multi")
parser.add_argument("--voxel", type=float, default=None)
parser.add_argument("--threshold", type=float, default=None)
parser.add_argument("--output_dir", default="result/multi")
args = parser.parse_args()

INPUT = args.input_dir
OUT = args.output_dir

# 加载所有视角
view_files = sorted([f for f in os.listdir(INPUT) if f.endswith(('.pcd', '.ply'))])
if not view_files:
    print(f"ERROR: {INPUT}/ 下没有 .pcd 或 .ply 文件")
    sys.exit(1)

views = [o3d.io.read_point_cloud(os.path.join(INPUT, vf)) for vf in view_files]
n_views = len(views)
print(f"加载 {n_views} 个视角")

# 密度分析
print("\n--- 点云密度 ---")
for i, (vf, v) in enumerate(zip(view_files, views)):
    d = estimate_density(v)
    print(f"  [{i}] {vf}: {len(v.points)} pts, density={d:.6f}")

avg_density = np.mean([estimate_density(v) for v in views])

# 自动推荐参数
voxel_size = args.voxel if args.voxel is not None else avg_density * 10
threshold  = args.threshold if args.threshold is not None else avg_density * 5
print(f"\n平均密度: {avg_density:.6f}")
print(f"推荐 voxel_size={voxel_size:.4f} (密度×10)")
print(f"推荐 threshold={threshold:.4f} (密度×5)")

if args.voxel is None:
    resp = input("使用推荐参数? [Y/n]: ").strip().lower()
    if resp and resp != 'y':
        voxel_size = float(input("voxel_size: "))
        threshold  = float(input("threshold: "))

# 清空输出
if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)

# 增量拼接
stitcher = MultiViewStitcher(voxel_size, threshold, min_fitness=0.25,
                              downsample_every=3, max_model_points=5000)
stitcher.init_with_view(views[0])

print(f"\n--- 增量配准 (voxel={voxel_size:.4f}, threshold={threshold:.4f}) ---")
for i in range(1, n_views):
    n_model = len(stitcher.model_down.points)
    n_view = len(views[i].points)
    print(f"  [{i}] -> 模型 ({n_model} pts) ...", end=" ", flush=True)
    result, ref = stitcher.add_view(views[i])
    if ref == -1:
        tag = " [模型]"
    else:
        tag = f" [兜底 ref={ref}]"
    flag = " !" if result.fitness < 0.2 else ""
    print(f"fitness={result.fitness:.4f} RMSE={result.inlier_rmse:.6f}{tag}{flag}")

    if i % 3 == 0 or i == n_views - 1:
        print(f"      模型: {len(stitcher.model.points)} pts -> 降采样 {len(stitcher.model_down.points)} pts")

# 保存
model_final = stitcher.stitched()
o3d.io.write_point_cloud(os.path.join(OUT, "stitched.pcd"), model_final)
for i in range(1, n_views):
    partial = stitcher.intermediate(i)
    o3d.io.write_point_cloud(os.path.join(OUT, f"step_{i:03d}.pcd"), partial)

# 厚度评估
t = compute_thickness(model_final)
print(f"\n最终模型厚度 — RMS: {t['rms']:.6f}  均值: {t['mean']:.6f}  中位数: {t['median']:.6f}")

# 截图
colors = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,0.5,0],[0.5,0,1]]
for i, v in enumerate(views):
    v.paint_uniform_color(colors[i % len(colors)])
save_screenshot(views, os.path.join(OUT, "all_views.png"), "all views")
model_final.paint_uniform_color([0, 0.651, 0.929])
save_screenshot([model_final], os.path.join(OUT, "stitched.png"), "stitched")

print(f"\nDone. Result: {OUT}/")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")
