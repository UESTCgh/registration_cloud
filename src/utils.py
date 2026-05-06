"""工具函数：密度计算、预处理、可视化"""

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
