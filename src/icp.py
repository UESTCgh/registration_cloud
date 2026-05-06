"""ICP 精配准"""

import open3d as o3d


def icp_point_to_point(source, target, threshold, init_transform,
                       max_iteration=2000):
    """点到点 ICP"""
    return o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iteration))


def icp_point_to_plane(source, target, threshold, init_transform,
                       voxel_size, max_iteration=3000):
    """点到面 ICP（需要 target 有法线）"""
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2.0, max_nn=30))
    return o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iteration))
