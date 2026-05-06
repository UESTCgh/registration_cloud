"""
综合演示脚本：按照配准说明.docx的要求，展示所有配准场景的结果。
包含：密度计算、FPFH粗配准+ICP精配准，以及扩展场景。
"""

import open3d as o3d
import numpy as np
import copy
import os
import shutil

# 数据与输出目录
DATA_DIR = "data"
OUT_DIR = "result"

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

def data(path):
    """拼接数据路径"""
    return os.path.join(DATA_DIR, path)

def out(path):
    """拼接输出路径"""
    return os.path.join(OUT_DIR, path)

# ============================================================
# 工具函数
# ============================================================

def preprocess_point_cloud(pcd, voxel_size):
    """采样、法线估计、计算FPFH特征"""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100))
    return pcd_down, pcd_fpfh


def midu(pcd):
    """计算点云密度"""
    point = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    point_size = point.shape[0]
    dd = np.zeros(point_size)
    for i in range(point_size):
        [_, idx, dis] = kdtree.search_knn_vector_3d(point[i], 2)
        dd[i] = dis[1]
    return np.mean(np.sqrt(dd))


def run_fpfh_icp(src, dst, voxel_size, threshold, name="demo",
                 max_iterations=64, max_tuples=1000):
    """
    运行 FPFH 粗配准 + 点到点 ICP 精配准
    返回: src_after_fgr, src_after_icp, fgr_result, icp_result
    """
    distance_threshold = 1.5 * voxel_size

    # 预处理
    src_down, src_fpfh = preprocess_point_cloud(src, voxel_size)
    dst_down, dst_fpfh = preprocess_point_cloud(dst, voxel_size)

    # FGR 粗配准
    fgr_result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        src_down, dst_down, src_fpfh, dst_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold,
            iteration_number=max_iterations,
            maximum_tuple_count=max_tuples))

    src_coarse = copy.deepcopy(src).transform(fgr_result.transformation)

    # 点到点 ICP 精配准
    icp_result = o3d.pipelines.registration.registration_icp(
        src, dst, threshold, fgr_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    src_fine = copy.deepcopy(src).transform(icp_result.transformation)

    return src_coarse, src_fine, fgr_result, icp_result


def save_viewpoint_screenshot(geometries, filename, window_name="View"):
    """使用 headless 渲染保存截图"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1024, height=768, visible=False)
    for g in geometries:
        vis.add_geometry(g)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename, do_render=True)
    vis.destroy_window()
    print(f"  截图已保存: {filename}")


# ============================================================
# 第一步：计算所有点云的密度
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
    fp = data(f)
    if os.path.exists(fp):
        p = o3d.io.read_point_cloud(fp)
        d = midu(p)
        print(f"  {f:25s} → 点数: {len(p.points):8d}  密度: {d:.6f}")


# ============================================================
# 第二步：fpfh_icp.py — 两片不同点云配准
# ============================================================

print("\n" + "=" * 70)
print("第二步：fpfh_icp.py 风格 —— 两片不同位姿的点云直接配准")
print("=" * 70)

# --- 场景1: cloud_bin_0 vs cloud_bin_1 ---
print("\n--- 场景1: cloud_bin_0.pcd + cloud_bin_1.pcd ---")
print("参数: voxel_size=0.05, threshold=0.02")

src = o3d.io.read_point_cloud(data("cloud_bin_0.pcd"))
dst = o3d.io.read_point_cloud(data("cloud_bin_1.pcd"))
src.paint_uniform_color([1, 0.706, 0])
dst.paint_uniform_color([0, 0.651, 0.929])

save_viewpoint_screenshot([src, dst], out("result_bin_before.png"), "cloud_bin 配准前")

src_coarse, src_fine, fgr_r, icp_r = run_fpfh_icp(
    src, dst, voxel_size=0.05, threshold=0.02, name="cloud_bin")

print(f"  FGR粗配准 — fitness: {fgr_r.fitness:.4f}, RMSE: {fgr_r.inlier_rmse:.6f}")
print(f"  ICP精配准 — fitness: {icp_r.fitness:.4f}, RMSE: {icp_r.inlier_rmse:.6f}")
print(f"  精配准矩阵:\n{icp_r.transformation}")

save_viewpoint_screenshot([src_fine, dst], out("result_bin_after.png"), "cloud_bin 精配准后")

# --- 场景2: 1.pcd vs 2.pcd ---
print("\n--- 场景2: 1.pcd + 2.pcd ---")
print("参数: voxel_size=0.005, threshold=0.002")

src = o3d.io.read_point_cloud(data("1.pcd"))
dst = o3d.io.read_point_cloud(data("2.pcd"))
src.paint_uniform_color([1, 0.706, 0])
dst.paint_uniform_color([0, 0.651, 0.929])

save_viewpoint_screenshot([src, dst], out("result_12_before.png"), "1/2 配准前")

src_coarse, src_fine, fgr_r, icp_r = run_fpfh_icp(
    src, dst, voxel_size=0.005, threshold=0.002, name="1_2")

print(f"  FGR粗配准 — fitness: {fgr_r.fitness:.4f}, RMSE: {fgr_r.inlier_rmse:.6f}")
print(f"  ICP精配准 — fitness: {icp_r.fitness:.4f}, RMSE: {icp_r.inlier_rmse:.6f}")
print(f"  精配准矩阵:\n{icp_r.transformation}")

save_viewpoint_screenshot([src_fine, dst], out("result_12_after.png"), "1/2 精配准后")


# ============================================================
# 第三步：peizhun.py — 同一片点云人为变换后配准
# ============================================================

print("\n" + "=" * 70)
print("第三步：peizhun.py 风格 —— 同一片点云人为变换位姿后配准")
print("=" * 70)

# --- 场景3: Dino ---
print("\n--- 场景3: Dino.pcd (人为旋转+平移后配准) ---")
print("参数: voxel_size=0.05, threshold=0.02")

src = o3d.io.read_point_cloud(data("Dino.pcd"))
T = np.eye(4)
T[:3, :3] = src.get_rotation_matrix_from_xyz((np.pi / 3, np.pi / 5, -np.pi / 6))
T[0, 3] = 1.5
T[1, 3] = 3
dst = copy.deepcopy(src).transform(T)

src.paint_uniform_color([0, 1, 0])
dst.paint_uniform_color([0, 0, 1])

print(f"  人为变换矩阵:\n{T}")

save_viewpoint_screenshot([src, dst], out("result_dino_before.png"), "Dino 配准前")

src_coarse, src_fine, fgr_r, icp_r = run_fpfh_icp(
    src, dst, voxel_size=0.05, threshold=0.02, name="Dino")

print(f"  FGR粗配准 — fitness: {fgr_r.fitness:.4f}, RMSE: {fgr_r.inlier_rmse:.6f}")
print(f"  ICP精配准 — fitness: {icp_r.fitness:.4f}, RMSE: {icp_r.inlier_rmse:.6f}")

# 验证：小幅平移拉开看效果
T_verify = np.eye(4)
T_verify[1, 3] = 0.01
src_verify = copy.deepcopy(src_fine).transform(T_verify)
save_viewpoint_screenshot([dst, src_verify], out("result_dino_after.png"), "Dino 精配准后(平移验证)")

# --- 场景4: Elephant ---
print("\n--- 场景4: Elephant.pcd (人为旋转+平移后配准) ---")
print("参数: voxel_size=0.05, threshold=0.02")

src = o3d.io.read_point_cloud(data("Elephant.pcd"))
T = np.eye(4)
T[:3, :3] = src.get_rotation_matrix_from_xyz((np.pi / 3, np.pi / 5, -np.pi / 6))
T[0, 3] = 1.5
T[1, 3] = 3
dst = copy.deepcopy(src).transform(T)

src.paint_uniform_color([0, 1, 0])
dst.paint_uniform_color([0, 0, 1])

save_viewpoint_screenshot([src, dst], out("result_elephant_before.png"), "Elephant 配准前")

src_coarse, src_fine, fgr_r, icp_r = run_fpfh_icp(
    src, dst, voxel_size=0.05, threshold=0.02, name="Elephant")

print(f"  FGR粗配准 — fitness: {fgr_r.fitness:.4f}, RMSE: {fgr_r.inlier_rmse:.6f}")
print(f"  ICP精配准 — fitness: {icp_r.fitness:.4f}, RMSE: {icp_r.inlier_rmse:.6f}")

T_verify = np.eye(4)
T_verify[1, 3] = 0.01
src_verify = copy.deepcopy(src_fine).transform(T_verify)
save_viewpoint_screenshot([dst, src_verify], out("result_elephant_after.png"), "Elephant 精配准后(平移验证)")

# --- 场景5: Chair ---
print("\n--- 场景5: Chair.pcd (人为旋转+平移后配准，大密度) ---")
print("参数: voxel_size=16, threshold=16")

src = o3d.io.read_point_cloud(data("Chair.pcd"))
T = np.eye(4)
T[:3, :3] = src.get_rotation_matrix_from_xyz((np.pi / 3, np.pi / 5, -np.pi / 6))
T[0, 3] = 1.5
T[1, 3] = 3
dst = copy.deepcopy(src).transform(T)

src.paint_uniform_color([0, 1, 0])
dst.paint_uniform_color([0, 0, 1])

save_viewpoint_screenshot([src, dst], out("result_chair_before.png"), "Chair 配准前")

src_coarse, src_fine, fgr_r, icp_r = run_fpfh_icp(
    src, dst, voxel_size=16, threshold=16, max_iterations=300, name="Chair")

print(f"  FGR粗配准 — fitness: {fgr_r.fitness:.4f}, RMSE: {fgr_r.inlier_rmse:.6f}")
print(f"  ICP精配准 — fitness: {icp_r.fitness:.4f}, RMSE: {icp_r.inlier_rmse:.6f}")

T_verify = np.eye(4)
T_verify[1, 3] = 1.0
src_verify = copy.deepcopy(src_fine).transform(T_verify)
save_viewpoint_screenshot([dst, src_verify], out("result_chair_after.png"), "Chair 精配准后(平移验证)")


# ============================================================
# 第四步：tuozhan.py — 密度差异大的点云，预降采样后配准
# ============================================================

print("\n" + "=" * 70)
print("第四步：tuozhan.py 扩展 —— 密度差异大的点云，预降采样后配准")
print("=" * 70)

print("\n--- 场景6: lab2_sfm.ply + lab2_kinect.ply ---")

src1 = o3d.io.read_point_cloud(data("lab2_sfm.ply"))
dst = o3d.io.read_point_cloud(data("lab2_kinect.ply"))

print(f"  lab2_sfm 原始密度: {midu(src1):.6f}")
print(f"  lab2_kinect 原始密度: {midu(dst):.6f}")

# 预降采样 lab2_sfm 使密度与 kinect 相近
src = src1.voxel_down_sample(0.008)
print(f"  lab2_sfm 降采样(0.008)后密度: {midu(src):.6f}")
print("参数: voxel_size=0.03, threshold=0.03")

src.paint_uniform_color([1, 0.706, 0])
dst.paint_uniform_color([0, 0.651, 0.929])

save_viewpoint_screenshot([src, dst], out("result_lab_before.png"), "lab2 配准前")

src_coarse, src_fine, fgr_r, icp_r = run_fpfh_icp(
    src, dst, voxel_size=0.03, threshold=0.03, name="lab2")

print(f"  FGR粗配准 — fitness: {fgr_r.fitness:.4f}, RMSE: {fgr_r.inlier_rmse:.6f}")
print(f"  ICP精配准 — fitness: {icp_r.fitness:.4f}, RMSE: {icp_r.inlier_rmse:.6f}")
print(f"  精配准矩阵:\n{icp_r.transformation}")

save_viewpoint_screenshot([src_fine, dst], out("result_lab_after.png"), "lab2 精配准后")


# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 70)
print("全部演示完成！")
print("=" * 70)
print("\n生成的文件:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
