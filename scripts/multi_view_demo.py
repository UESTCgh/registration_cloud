"""
多视角增量拼接演示：
  Phase 1 — 生成模拟多视角数据集 → data/multi/
  Phase 2 — 增量式拼接 → result/multi/

用法:
    python3 scripts/multi_view_demo.py
    python3 scripts/multi_view_demo.py --pcd Chair.pcd --views 6
    python3 scripts/multi_view_demo.py --gen_only
    python3 scripts/multi_view_demo.py --noise 0.002
    python3 scripts/multi_view_demo.py --angle_range 75 --angle_min 15
"""

import sys
import os
import shutil
import copy
import argparse
import json
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import MultiViewStitcher, estimate_density, save_screenshot

# ============================================================
# 参数
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--pcd", default="Dino.pcd")
parser.add_argument("--views", type=int, default=8)
parser.add_argument("--voxel", type=float, default=None)
parser.add_argument("--threshold", type=float, default=None)
parser.add_argument("--angle_range", type=float, default=60,
                    help="相邻视角最大旋转角 (度)")
parser.add_argument("--trans_range", type=float, default=0.3,
                    help="相邻视角最大平移量")
parser.add_argument("--angle_min", type=float, default=0,
                    help="相邻视角最小旋转角 (度)")
parser.add_argument("--noise", type=float, default=0.001)
parser.add_argument("--hidden", action="store_true",
                    help="启用隐藏点去除（每视角只保留可见表面，模拟真实扫描）")
parser.add_argument("--gen_only", action="store_true")
args = parser.parse_args()

DATA_DIR = "data"
MULTI_DIR = "data/multi"
OUT_DIR = "result/multi"

for d in [MULTI_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

# 预设参数
PRESETS = {
    "Dino.pcd":      (0.05, 0.02),
    "Elephant.pcd":  (0.05, 0.02),
    "Chair.pcd":     (16,   16),
    "1.pcd":         (0.005, 0.002),
    "cloud_bin_0.pcd": (0.05, 0.02),
}
def_vs, def_th = PRESETS.get(args.pcd, (0.05, 0.02))
voxel_size = args.voxel if args.voxel is not None else def_vs
threshold  = args.threshold if args.threshold is not None else def_th

DATA_PATH = os.path.join(DATA_DIR, args.pcd)
SEED = 42

# ============================================================
# Phase 1: 生成多视角数据集（带隐藏点去除）
# ============================================================

need_gen = (not os.path.exists(os.path.join(MULTI_DIR, "view_000.pcd")) or
            args.gen_only)

if not need_gen:
    with open(os.path.join(MULTI_DIR, "gt_poses.json")) as f:
        prev = json.load(f)
    need_gen = (prev.get("source") != args.pcd or
                prev.get("views") != args.views or
                prev.get("noise") != args.noise)

if need_gen:
    for f in os.listdir(MULTI_DIR):
        os.remove(os.path.join(MULTI_DIR, f))

    print("=" * 70)
    print(f"Phase 1: 从 {args.pcd} 生成 {args.views} 个多视角数据")
    print(f"  旋转范围 [{args.angle_min} deg, {args.angle_range} deg]")
    print(f"  平移范围 +/-{args.trans_range}")
    print(f"  高斯噪声 sigma={args.noise}")
    print(f"  隐藏点去除: {'开启' if args.hidden else '关闭'}")
    print("=" * 70)

    original = o3d.io.read_point_cloud(DATA_PATH)
    print(f"原始点云: {len(original.points)} 点, 密度={estimate_density(original):.6f}")
    np.random.seed(SEED)

    # 固定相机位置（世界坐标系），隐藏点去除只保留从该位置可见的表面
    bbox = original.get_axis_aligned_bounding_box()
    cam_pos = bbox.get_center() + np.array([0, 0, bbox.get_max_extent() * 3])

    gt_poses = []
    for i in range(args.views):
        T = np.eye(4)
        if i > 0:
            max_rad = np.radians(args.angle_range)
            min_rad = np.radians(args.angle_min)
            def _sample_angle():
                a = np.random.uniform(min_rad, max_rad)
                return a * np.random.choice([-1, 1])
            rx, ry, rz = _sample_angle(), _sample_angle(), _sample_angle()
            R_delta = original.get_rotation_matrix_from_xyz((rx, ry, rz))
            t_delta = np.random.uniform(-args.trans_range, args.trans_range, 3)
            T[:3, :3] = gt_poses[i-1][:3, :3] @ R_delta
            T[:3, 3]  = gt_poses[i-1][:3, 3] + t_delta

        pcd_view = copy.deepcopy(original).transform(T)
        if args.noise > 0:
            pts = np.asarray(pcd_view.points)
            pts += np.random.randn(*pts.shape).astype(np.float64) * args.noise
            pcd_view.points = o3d.utility.Vector3dVector(pts)

        # 隐藏点去除：物体转过去的面不可见，只保留朝向相机的表面
        if args.hidden:
            _, pt_map = pcd_view.hidden_point_removal(cam_pos, bbox.get_max_extent() * 10)
            pcd_view = pcd_view.select_by_index(pt_map)

        gt_poses.append(T)
        o3d.io.write_point_cloud(
            os.path.join(MULTI_DIR, f"view_{i:03d}.pcd"), pcd_view)
        print(f"  保存 view_{i:03d}.pcd ({len(pcd_view.points)} 点)")

    with open(os.path.join(MULTI_DIR, "gt_poses.json"), "w") as f:
        json.dump({"source": args.pcd, "views": args.views,
                   "noise": args.noise, "poses": [T.tolist() for T in gt_poses]},
                  f, indent=2)

    if args.gen_only:
        print("\n仅生成模式，结束。")
        sys.exit(0)
else:
    print("数据集已存在，跳过生成。")
    with open(os.path.join(MULTI_DIR, "gt_poses.json")) as f:
        gt_poses = [np.array(p) for p in json.load(f)["poses"]]

# ============================================================
# Phase 2: 增量式拼接
# ============================================================

print("\n" + "=" * 70)
print("Phase 2: 增量式多视角拼接")
print(f"  参数: voxel_size={voxel_size}, threshold={threshold}")
print("=" * 70)

view_files = sorted([f for f in os.listdir(MULTI_DIR) if f.endswith(".pcd")])
views = [o3d.io.read_point_cloud(os.path.join(MULTI_DIR, vf))
         for vf in view_files]
n_views = len(views)
print(f"加载 {n_views} 个视角")

# 自动调整配准分辨率：确保降采样后每视角仍有足够点 (>300) 供 FPFH 匹配
avg_pts = np.mean([len(v.points) for v in views])
if avg_pts < 1000 and args.hidden:
    # 部分视图场景：缩小 voxel_size 保持特征丰富度
    vs_ratio = np.sqrt(300.0 / max(avg_pts, 1))
    reg_voxel = max(voxel_size * vs_ratio * 0.3, voxel_size * 0.15)
    reg_threshold = threshold * (reg_voxel / voxel_size)
    print(f"  部分视图检测 (avg {avg_pts:.0f} pts/view)")
    print(f"  自适应配准参数: voxel_size={reg_voxel:.4f}, threshold={reg_threshold:.4f}")
else:
    reg_voxel = voxel_size
    reg_threshold = threshold

with open(os.path.join(MULTI_DIR, "gt_poses.json")) as f:
    gt_poses = [np.array(p) for p in json.load(f)["poses"]]

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

stitcher = MultiViewStitcher(reg_voxel, reg_threshold,
                              min_fitness=0.5, downsample_every=3)
stitcher.init_with_view(views[0])

print("\n--- 增量配准过程 ---")
for i in range(1, n_views):
    print(f"  注册视角 {i} -> 模型 ({len(stitcher.model_down.points)} 点) ...",
          end=" ", flush=True)
    result = stitcher.add_view(views[i])
    tag = "" if not stitcher.fallbacks[-1] else " [回退]"
    print(f"fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}{tag}")

    if i % stitcher.downsample_every == 0 or i == n_views - 1:
        print(f"  L- 模型: {len(stitcher.model.points)} 点 -> "
              f"降采样: {len(stitcher.model_down.points)} 点")

# 保存结果
model_final = stitcher.stitched()
o3d.io.write_point_cloud(os.path.join(OUT_DIR, "stitched.pcd"), model_final)

for i in range(1, n_views):
    partial = stitcher.intermediate(i)
    o3d.io.write_point_cloud(
        os.path.join(OUT_DIR, f"incremental_step_{i:03d}.pcd"), partial)

# 可视化
colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0],
          [1,0,1], [0,1,1], [1,0.5,0], [0.5,0,1]]
for i, v in enumerate(views):
    v.paint_uniform_color(colors[i % len(colors)])
save_screenshot(views, os.path.join(OUT_DIR, "all_views.png"), "所有视角")

original = o3d.io.read_point_cloud(DATA_PATH)
original.paint_uniform_color([0, 1, 0])
model_final.paint_uniform_color([0, 0.651, 0.929])
save_screenshot([original, model_final],
                os.path.join(OUT_DIR, "compare.png"),
                "green=original  blue=stitched")

# 误差评估
print("\n--- 误差评估 ---")
print(f"{'view':<6} {'trans_err(m)':>12} {'rot_err(deg)':>12}")
print("-" * 32)
for i in range(n_views):
    diff = gt_poses[i] @ stitcher.poses[i]
    t_err = np.linalg.norm(diff[:3, 3])
    r_err = np.arccos(np.clip((np.trace(diff[:3, :3]) - 1) / 2, -1, 1))
    print(f"  {i:<4}  {t_err:12.4f}  {np.degrees(r_err):12.2f}")

print(f"\ndone.")
print(f"  dataset: {MULTI_DIR}/  ({n_views} views)")
print(f"  result:  {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"    {f}")
