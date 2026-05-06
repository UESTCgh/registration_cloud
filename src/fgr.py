"""Fast Global Registration 基于 FPFH 特征"""

import open3d as o3d

from .utils import preprocess


def fgr_register(source, target, voxel_size,
                 max_iterations=64, max_tuples=1000,
                 normal_radius_mult=2.0, normal_max_nn=30,
                 feature_radius_mult=5.0, feature_max_nn=100):
    """FPFH + Fast Global Registration 粗配准"""
    distance_threshold = 1.5 * voxel_size

    src_down, src_fpfh = preprocess(source, voxel_size,
                                     normal_radius_mult, normal_max_nn,
                                     feature_radius_mult, feature_max_nn)
    dst_down, dst_fpfh = preprocess(target, voxel_size,
                                     normal_radius_mult, normal_max_nn,
                                     feature_radius_mult, feature_max_nn)

    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        src_down, dst_down, src_fpfh, dst_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold,
            iteration_number=max_iterations,
            maximum_tuple_count=max_tuples))
    return result
