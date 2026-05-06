"""工具函数：密度计算、点云厚度、预处理、可视化"""

import open3d as o3d
import numpy as np


def estimate_density(pcd):
    """计算点云平均密度（最近邻距离的均方根）"""
    points = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    n = points.shape[0]
    dd = np.zeros(n)
    for i in range(n):
        [_, _, dis] = kdtree.search_knn_vector_3d(points[i], 2)
        dd[i] = dis[1]
    return float(np.mean(np.sqrt(dd)))


def compute_thickness(pcd, k=20):
    """计算点云厚度（局部平面拟合残差）

    对每个点 p_i，用 k 近邻拟合局部平面，计算 p_i 到该平面的距离 d_i。
    返回 {'mean', 'std', 'median', 'max', 'rms'} 五项统计量。
    值越大说明点云越"厚"（噪声大/配准偏差大/表面粗糙）。
    """
    pts = np.asarray(pcd.points)
    n = len(pts)
    if n < k + 3:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "max": 0.0, "rms": 0.0}

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    residuals = np.zeros(n)

    for i in range(n):
        [_, idx, _] = kdtree.search_knn_vector_3d(pts[i], k)
        neighbors = pts[idx]                          # (k, 3)
        centroid = neighbors.mean(axis=0)             # (3,)
        cov = np.cov(neighbors - centroid, rowvar=False)  # (3, 3)
        # 最小特征值对应的特征向量 = 法向量
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]                        # 法向量 (3,)
        # 点到局部平面的距离
        residuals[i] = abs(np.dot(pts[i] - centroid, normal))

    return {
        "mean":   float(np.mean(residuals)),
        "std":    float(np.std(residuals)),
        "median": float(np.median(residuals)),
        "max":    float(np.max(residuals)),
        "rms":    float(np.sqrt(np.mean(residuals ** 2))),
    }


def preprocess(pcd, voxel_size, normal_radius_mult=2.0, normal_max_nn=30,
               feature_radius_mult=5.0, feature_max_nn=100):
    """降采样 → 法线估计 → 计算 FPFH 特征"""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * normal_radius_mult, max_nn=normal_max_nn))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * feature_radius_mult, max_nn=feature_max_nn))
    return pcd_down, fpfh


def save_screenshot(geometries, filename, window_name="View",
                    width=1024, height=768):
    """无窗口渲染并保存截图"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height,
                      visible=False)
    for g in geometries:
        vis.add_geometry(g)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename, do_render=True)
    vis.destroy_window()
