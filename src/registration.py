"""配准管线：FPFH 粗配准 + ICP 精配准"""

from .fgr import fgr_register
from .icp import icp_point_to_point


def register(source, target, voxel_size, threshold,
             max_iterations=64, max_tuples=1000,
             icp_max_iterations=2000):
    """完整的 FGR + ICP 配准管线

    Returns:
        result: ICP 配准结果 (transformation, fitness, inlier_rmse)
    """
    # 粗配准
    fgr_result = fgr_register(source, target, voxel_size,
                               max_iterations, max_tuples)

    # 精配准
    icp_result = icp_point_to_point(source, target, threshold,
                                     fgr_result.transformation,
                                     icp_max_iterations)
    return icp_result


def register_with_fallback(source, target, model, voxel_size, threshold,
                           min_fitness=0.3):
    """配准，质量太差时回退到注册到 model（用于增量拼接）

    Returns:
        (result, used_fallback)
    """
    result = register(source, target, voxel_size, threshold)
    if result.fitness >= min_fitness:
        return result, False

    # 回退：注册到累积模型
    result = register(source, model, voxel_size, threshold)
    return result, True
