"""
多视角点云拼接：
  Phase 1 — 从单片点云生成模拟多视角数据集，保存到 data_multi/
  Phase 2 — 加载数据集，成对配准 + 位姿图优化，结果保存到 result_multi/

用法:
    python3 multi_view.py                           # 默认 Dino, 8视角
    python3 multi_view.py --pcd Chair.pcd --views 6 --voxel 16 --threshold 16
    python3 multi_view.py --noise 0.002             # 加大噪声
    python3 multi_view.py --gen_only                # 仅生成数据集
    python3 multi_view.py --reg_only                # 仅配准（需先生成）
"""

import open3d as o3d
import numpy as np
import copy
import os
import sys
import shutil
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
parser.add_argument("--reg_only", action="store_true")
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
    """FPFH 粗配准 + ICP 精配准 → (icp_result, is_valid)"""
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

    valid = icp.fitness > 0.2 and icp.inlier_rmse < th * 3
    return icp, valid

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

if not args.reg_only:
    print("=" * 70)
    print(f"Phase 1: 从 {args.pcd} 生成 {args.views} 个多视角数据")
    print(f"  旋转范围 ±{args.angle_range}°, 平移范围 ±{args.trans_range}")
    print(f"  高斯噪声 σ={args.noise}")
    print("=" * 70)

    original = o3d.io.read_point_cloud(DATA_PATH)
    print(f"原始点云: {len(original.points)} 点, 密度={midu(original):.6f}")
    np.random.seed(42)

    gt_transforms = []
    gt_poses = []  # 从原始点云到各视角的变换（即各视角的"相机外参"）

    for i in range(args.views):
        T = np.eye(4)
        if i == 0:
            pass  # identity
        else:
            # 在前一个位姿基础上叠加小旋转 + 小平移
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

        # 添加高斯噪声模拟传感器噪声
        if args.noise > 0:
            pts = np.asarray(pcd_view.points)
            pts += np.random.randn(*pts.shape).astype(np.float64) * args.noise
            pcd_view.points = o3d.utility.Vector3dVector(pts)

        gt_poses.append(T)

        filename = f"view_{i:03d}.pcd"
        o3d.io.write_point_cloud(os.path.join(MULTI_DIR, filename), pcd_view)
        print(f"  保存 {filename}")

    # 保存 ground truth 位姿
    gt_data = {"source": args.pcd, "views": args.views,
               "noise": args.noise,
               "angle_range": args.angle_range,
               "trans_range": args.trans_range,
               "poses": [T.tolist() for T in gt_poses]}
    with open(os.path.join(MULTI_DIR, "gt_poses.json"), "w") as f:
        json.dump(gt_data, f, indent=2)
    print(f"  Ground truth 位姿已保存到 gt_poses.json")

    if args.gen_only:
        print("\n仅生成模式，结束。")
        sys.exit(0)

# ============================================================
# Phase 2: 加载数据集并配准
# ============================================================

print("\n" + "=" * 70)
print("Phase 2: 加载多视角数据集并进行配准")
print("=" * 70)

# 加载所有视角
views = []
view_files = sorted([f for f in os.listdir(MULTI_DIR) if f.endswith(".pcd")])
for vf in view_files:
    views.append(o3d.io.read_point_cloud(os.path.join(MULTI_DIR, vf)))

n_views = len(views)
print(f"加载了 {n_views} 个视角")

# 加载 ground truth（如果存在）
gt_path = os.path.join(MULTI_DIR, "gt_poses.json")
gt_poses = None
if os.path.exists(gt_path):
    with open(gt_path) as f:
        gt_poses = [np.array(p) for p in json.load(f)["poses"]]

# ============================================================
# Step 2a: 所有成对配准（相邻 + 回环）
# ============================================================

print("\n--- 成对配准 ---")

# 定义要配准的边
pairs_to_register = []
# 相邻边
for i in range(n_views - 1):
    pairs_to_register.append((i, i+1, "sequential"))
# 回环边
pairs_to_register.append((n_views - 1, 0, "loop"))

registration_results = {}
for src_i, dst_i, ptype in pairs_to_register:
    print(f"  {ptype:10s} {src_i} → {dst_i} ...", end=" ", flush=True)
    result, valid = pairwise_register(views[src_i], views[dst_i],
                                      voxel_size, threshold)
    tag = "OK" if valid else "LOW"
    print(f"fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f} [{tag}]")
    registration_results[(src_i, dst_i)] = (result, valid)

# ============================================================
# Step 2b: 构建位姿图
# ============================================================

print("\n--- 构建位姿图 ---")

pose_graph = o3d.pipelines.registration.PoseGraph()
pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

# 累积变换：从连续配准链计算各节点位姿
for i in range(1, n_views):
    prev_pose = pose_graph.nodes[i-1].pose
    T = registration_results[(i-1, i)][0].transformation
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(np.dot(prev_pose, T)))

# 添加边
for (src_i, dst_i), (result, valid) in registration_results.items():
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        views[src_i], views[dst_i], voxel_size * 1.5, result.transformation)
    uncertain = not valid
    pose_graph.edges.append(
        o3d.pipelines.registration.PoseGraphEdge(
            src_i, dst_i, result.transformation, info, uncertain=uncertain))
    print(f"  添加边 {src_i}→{dst_i} (uncertain={uncertain})")

# 保存优化前位姿
poses_before = [np.copy(n.pose) for n in pose_graph.nodes]

# ============================================================
# Step 2c: 位姿图优化
# ============================================================

print("\n--- 位姿图优化 ---")

print("优化前平移:")
for i in range(min(5, n_views)):
    print(f"  节点{i}: {poses_before[i][:3,3]}")

option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=voxel_size * 1.5,
    edge_prune_threshold=0.25,
    reference_node=0)
o3d.pipelines.registration.global_optimization(
    pose_graph,
    o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
    option)

poses_after = [np.copy(n.pose) for n in pose_graph.nodes]

print("优化后平移:")
for i in range(min(5, n_views)):
    print(f"  节点{i}: {poses_after[i][:3,3]}")

# ============================================================
# Step 2d: 拼接与可视化
# ============================================================

print("\n--- 拼接点云 ---")

# 优化前的拼接
stitched_before = o3d.geometry.PointCloud()
for i, v in enumerate(views):
    stitched_before += copy.deepcopy(v).transform(np.linalg.inv(poses_before[i]))

# 优化后的拼接
stitched_after = o3d.geometry.PointCloud()
for i, v in enumerate(views):
    stitched_after += copy.deepcopy(v).transform(np.linalg.inv(poses_after[i]))

# 降采样
before_down = stitched_before.voxel_down_sample(
    max(voxel_size * 0.5, 0.002))
after_down = stitched_after.voxel_down_sample(
    max(voxel_size * 0.5, 0.002))

print(f"  拼接点数: 优化前={len(before_down.points)}, 优化后={len(after_down.points)}")

# 保存拼接结果
o3d.io.write_point_cloud(os.path.join(OUT_DIR, "stitched_before.pcd"), before_down)
o3d.io.write_point_cloud(os.path.join(OUT_DIR, "stitched_after.pcd"), after_down)

# 截图
colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0],
          [1,0,1], [0,1,1], [1,0.5,0], [0.5,0,1],
          [0,1,0.5], [0.5,0.5,0], [1,0.5,0.5], [0.5,1,0.5]]
for i, v in enumerate(views):
    v.paint_uniform_color(colors[i % len(colors)])
save_screenshot(views, os.path.join(OUT_DIR, "all_views.png"), "所有视角")

# 优化前（橙色）vs 优化后（蓝色）
before_down.paint_uniform_color([1, 0.706, 0])
after_down.paint_uniform_color([0, 0.651, 0.929])
save_screenshot([before_down, after_down],
                os.path.join(OUT_DIR, "before_vs_after.png"),
                "橙色=优化前 蓝色=优化后")

# ============================================================
# Step 2e: 误差评估
# ============================================================

if gt_poses is not None:
    print("\n--- 误差评估 ---")
    print(f"{'视角':<6} {'优化前平移':>10} {'优化后平移':>10} {'优化前旋转':>10} {'优化后旋转':>10}")
    for i in range(n_views):
        def pose_error(est, gt):
            diff = np.linalg.inv(gt) @ est
            t_err = np.linalg.norm(diff[:3, 3])
            r_err = np.arccos(np.clip((np.trace(diff[:3, :3]) - 1) / 2, -1, 1))
            return t_err, np.degrees(r_err)
        e_b = pose_error(poses_before[i], gt_poses[i])
        e_a = pose_error(poses_after[i], gt_poses[i])
        print(f"  {i:<4}  {e_b[0]:8.4f}   {e_a[0]:8.4f}   {e_b[1]:8.2f}°  {e_a[1]:8.2f}°")

# ============================================================
# 总结
# ============================================================

print(f"\n完成！")
print(f"  多视角数据集:  {MULTI_DIR}/  ({len(views)} 个 .pcd)")
print(f"  拼接结果:      {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"    {f}")
for f in sorted(os.listdir(MULTI_DIR)):
    print(f"    {f}")
