"""
多视角点云拼接：
  Phase 1 — 从单片点云生成模拟多视角数据集，保存到 data_multi/
  Phase 2 — 加载数据集，成对配准构建位姿链，拼接点云，保存到 result_multi/

用法:
    python3 multi_view.py                           # 默认 Dino, 8视角
    python3 multi_view.py --pcd Chair.pcd --views 6 --voxel 16 --threshold 16
    python3 multi_view.py --noise 0.002             # 加大噪声
    python3 multi_view.py --gen_only                # 仅生成数据集
"""

import open3d as o3d
import numpy as np
import copy
import os
import sys
import argparse
import json

# ============================================================
# 参数
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--pcd", default="Dino.pcd")
parser.add_argument("--views", type=int, default=8)
parser.add_argument("--voxel", type=float, default=None)
parser.add_argument("--threshold", type=float, default=None)
parser.add_argument("--angle_range", type=float, default=60)
parser.add_argument("--trans_range", type=float, default=0.3)
parser.add_argument("--noise", type=float, default=0.001)
parser.add_argument("--gen_only", action="store_true")
args = parser.parse_args()

DATA_DIR = "data"
MULTI_DIR = "data_multi"
OUT_DIR = "result_multi"

for d in [MULTI_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

# 默认参数表
DEFAULT_PARAMS = {
    "Dino.pcd":      (0.05, 0.02),
    "Elephant.pcd":  (0.05, 0.02),
    "Chair.pcd":     (16,   16),
    "1.pcd":         (0.005, 0.002),
    "cloud_bin_0.pcd": (0.05, 0.02),
}
def_voxel, def_th = DEFAULT_PARAMS.get(args.pcd, (0.05, 0.02))
voxel_size  = args.voxel if args.voxel is not None else def_voxel
threshold   = args.threshold if args.threshold is not None else def_th

DATA_PATH = os.path.join(DATA_DIR, args.pcd)

# ============================================================
# 工具函数
# ============================================================

def midu(pcd):
    point = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    dd = np.zeros(point.shape[0])
    for i in range(point.shape[0]):
        [_, _, dis] = kdtree.search_knn_vector_3d(point[i], 2)
        dd[i] = dis[1]
    return np.mean(np.sqrt(dd))

def preprocess(pcd, vs):
    pcd_down = pcd.voxel_down_sample(vs)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=vs * 2.0, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=vs * 5.0, max_nn=100))
    return pcd_down, fpfh

def pairwise_register(src, dst, vs, th):
    """FPFH 粗配准 + 点到点 ICP 精配准"""
    dt = 1.5 * vs
    src_d, src_f = preprocess(src, vs)
    dst_d, dst_f = preprocess(dst, vs)

    fgr = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        src_d, dst_d, src_f, dst_f,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=dt,
            iteration_number=64, maximum_tuple_count=1000))

    icp = o3d.pipelines.registration.registration_icp(
        src, dst, th, fgr.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    return icp

def save_screenshot(geometries, filename, name="View"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=1024, height=768, visible=False)
    for g in geometries:
        vis.add_geometry(g)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename, do_render=True)
    vis.destroy_window()

# ============================================================
# Phase 1: 生成多视角数据集
# ============================================================

if not os.path.exists(os.path.join(MULTI_DIR, "view_000.pcd")):
    do_generate = True
elif not os.path.exists(os.path.join(MULTI_DIR, "gt_poses.json")):
    do_generate = True
else:
    # 检查是否匹配当前参数
    with open(os.path.join(MULTI_DIR, "gt_poses.json")) as f:
        prev = json.load(f)
    do_generate = (prev.get("source") != args.pcd or
                   prev.get("views") != args.views or
                   prev.get("noise") != args.noise)

if args.gen_only:
    do_generate = True

if do_generate:
    # 清空旧数据
    for f in os.listdir(MULTI_DIR):
        os.remove(os.path.join(MULTI_DIR, f))

    print("=" * 70)
    print(f"Phase 1: 从 {args.pcd} 生成 {args.views} 个多视角数据")
    print(f"  旋转范围 ±{args.angle_range}°, 平移范围 ±{args.trans_range}")
    print(f"  高斯噪声 σ={args.noise}")
    print("=" * 70)

    original = o3d.io.read_point_cloud(DATA_PATH)
    print(f"原始点云: {len(original.points)} 点, 密度={midu(original):.6f}")
    np.random.seed(42)

    gt_poses = []  # 从 original 到各视角的变换 (4x4)

    for i in range(args.views):
        T = np.eye(4)
        if i == 0:
            pass  # identity
        else:
            rx = np.random.uniform(-1, 1) * np.radians(args.angle_range) * 0.5
            ry = np.random.uniform(-1, 1) * np.radians(args.angle_range) * 0.5
            rz = np.random.uniform(-1, 1) * np.radians(args.angle_range)
            R_prev = gt_poses[i-1][:3, :3]
            t_prev = gt_poses[i-1][:3, 3]
            R_delta = original.get_rotation_matrix_from_xyz((rx, ry, rz))
            t_delta = np.random.uniform(-args.trans_range, args.trans_range, 3)
            T[:3, :3] = R_prev @ R_delta
            T[:3, 3]  = t_prev + t_delta

        pcd_view = copy.deepcopy(original).transform(T)

        if args.noise > 0:
            pts = np.asarray(pcd_view.points)
            pts += np.random.randn(*pts.shape).astype(np.float64) * args.noise
            pcd_view.points = o3d.utility.Vector3dVector(pts)

        gt_poses.append(T)
        filename = f"view_{i:03d}.pcd"
        o3d.io.write_point_cloud(os.path.join(MULTI_DIR, filename), pcd_view)
        print(f"  保存 {filename}")

    gt_data = {"source": args.pcd, "views": args.views,
               "noise": args.noise,
               "angle_range": args.angle_range,
               "trans_range": args.trans_range,
               "poses": [T.tolist() for T in gt_poses]}
    with open(os.path.join(MULTI_DIR, "gt_poses.json"), "w") as f:
        json.dump(gt_data, f, indent=2)
    print(f"  Ground truth 位姿 → gt_poses.json")

    if args.gen_only:
        print("\n仅生成模式，结束。")
        sys.exit(0)
else:
    print("数据集已存在，跳过生成。")

# ============================================================
# Phase 2: 增量式多视角配准
# ============================================================

print("\n" + "=" * 70)
print("Phase 2: 增量式多视角配准")
print(f"  参数: voxel_size={voxel_size}, threshold={threshold}")
print("  策略: 每个新视角配准到已累积模型，逐步融合")
print("=" * 70)

view_files = sorted([f for f in os.listdir(MULTI_DIR) if f.endswith(".pcd")])
views = [o3d.io.read_point_cloud(os.path.join(MULTI_DIR, vf))
         for vf in view_files]
n_views = len(views)
print(f"加载 {n_views} 个视角")

gt_path = os.path.join(MULTI_DIR, "gt_poses.json")
gt_poses = None
if os.path.exists(gt_path):
    with open(gt_path) as f:
        gt_poses = [np.array(p) for p in json.load(f)["poses"]]

# ============================================================
# 增量配准核心
# ============================================================

print("\n--- 增量配准过程 ---")

# 以 view_0 为初始模型
model = copy.deepcopy(views[0])
model_down = model.voxel_down_sample(voxel_size)
estimated_poses = [np.eye(4)]  # view_i → 参考系 的变换
prev_pose = np.eye(4)          # 上一个视角的位姿（用于回退）

MIN_FITNESS = 0.3  # 低于此阈值视为配准失败，启用回退

for i in range(1, n_views):
    new_view = views[i]

    # 策略1: 将新视角配准到当前累积模型
    print(f"  注册视角 {i} → 模型 (模型 {len(model_down.points)} 点) ...", end=" ", flush=True)
    result = pairwise_register(new_view, model_down, voxel_size, threshold)

    if result.fitness < MIN_FITNESS:
        # 策略2 (回退): 配准到上一帧，再通过链式累积得到全局位姿
        print(f"fitness={result.fitness:.4f} 过低!", end=" ")
        print(f"回退到链式 {i}→{i-1} ...", end=" ", flush=True)
        result_chain = pairwise_register(new_view, views[i-1], voxel_size, threshold)

        # T_chain: new_view → view_{i-1}
        # accumulated: view_{i-1} → 参考系 = estimated_poses[i-1]
        T_ref = np.dot(estimated_poses[i-1], result_chain.transformation)
        print(f"fitness={result_chain.fitness:.4f}, RMSE={result_chain.inlier_rmse:.6f}")
    else:
        T_ref = result.transformation
        print(f"fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}")

    estimated_poses.append(T_ref)

    # 将新视角变换到参考系并融入模型
    new_aligned = copy.deepcopy(new_view).transform(T_ref)
    model += new_aligned

    # 周期性降采样控制模型规模
    if i % 3 == 0 or i == n_views - 1:
        model_down = model.voxel_down_sample(voxel_size)
        print(f"  └─ 融合后模型: {len(model.points)} 点 → 降采样: {len(model_down.points)} 点")

# ============================================================
# 拼接结果
# ============================================================

print("\n--- 最终拼接 ---")

model_final = model.voxel_down_sample(max(voxel_size * 0.5, 0.002))
print(f"  最终模型: {len(model_final.points)} 点")

o3d.io.write_point_cloud(os.path.join(OUT_DIR, "stitched.pcd"), model_final)

# 同时重建并保存每个增量步骤的中间模型
for i in range(1, n_views):
    partial = o3d.geometry.PointCloud()
    for j in range(i + 1):
        partial += copy.deepcopy(views[j]).transform(estimated_poses[j])
    partial_down = partial.voxel_down_sample(max(voxel_size * 0.5, 0.002))
    o3d.io.write_point_cloud(
        os.path.join(OUT_DIR, f"incremental_step_{i:03d}.pcd"), partial_down)

# ============================================================
# 可视化
# ============================================================

# 各视角（不同颜色）
colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0],
          [1,0,1], [0,1,1], [1,0.5,0], [0.5,0,1],
          [0,1,0.5], [0.5,0.5,0], [1,0.5,0.5], [0.5,1,0.5]]
for i, v in enumerate(views):
    v.paint_uniform_color(colors[i % len(colors)])
save_screenshot(views, os.path.join(OUT_DIR, "all_views.png"),
                "所有视角（不同颜色）")

# 原始点云（绿色）vs 拼接结果（蓝色）
original = o3d.io.read_point_cloud(DATA_PATH)
original.paint_uniform_color([0, 1, 0])
model_final.paint_uniform_color([0, 0.651, 0.929])
save_screenshot([original, model_final],
                os.path.join(OUT_DIR, "compare.png"),
                "绿色=原始  蓝色=增量拼接")

# 最终结果单独
model_final.paint_uniform_color([0, 0.651, 0.929])
save_screenshot([model_final],
                os.path.join(OUT_DIR, "stitched.png"),
                "增量拼接最终结果")

# ============================================================
# 误差评估
# ============================================================

if gt_poses is not None:
    print("\n--- 误差评估 ---")
    print(f"{'视角':<6} {'平移误差':>10} {'旋转误差':>10}")
    print("-" * 28)
    for i in range(n_views):
        # estimated_poses[i]: view_i → 参考系 (即 view_0 = original)
        # gt_poses[i]: original → view_i
        # 期望: estimated_poses[i] ≈ inv(gt_poses[i]), 即 gt @ estimated ≈ I
        diff = gt_poses[i] @ estimated_poses[i]
        t_err = np.linalg.norm(diff[:3, 3])
        r_err = np.arccos(np.clip((np.trace(diff[:3, :3]) - 1) / 2, -1, 1))
        print(f"  {i:<4}  {t_err:8.4f}   {np.degrees(r_err):8.2f}°")

# ============================================================
# 总结
# ============================================================

print(f"\n完成！")
print(f"  数据集: {MULTI_DIR}/  ({n_views} 个 .pcd + gt_poses.json)")
print(f"  结果:   {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"    {f}")
print(f"\n  拼接点云: {OUT_DIR}/stitched.pcd")
