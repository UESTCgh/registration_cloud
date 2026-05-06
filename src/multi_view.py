"""增量式多视角拼接"""

import copy
import numpy as np
import open3d as o3d

from .registration import register
from .utils import preprocess


class MultiViewStitcher:
    """增量式多视角点云拼接器"""

    def __init__(self, voxel_size, threshold, min_fitness=0.3,
                 downsample_every=3):
        self.voxel_size = voxel_size
        self.threshold = threshold
        self.min_fitness = min_fitness
        self.downsample_every = downsample_every

        self.model = None          # 累积点云
        self.model_down = None     # 降采样模型（用于配准）
        self.views = []            # 原始视角
        self.poses = []            # 各视角 → 参考系 的变换
        self.fallbacks = []        # 是否触发了回退

    def init_with_view(self, view):
        """用第一个视角初始化模型"""
        self.model = copy.deepcopy(view)
        self.model_down = self.model.voxel_down_sample(self.voxel_size)
        self.views = [view]
        self.poses = [np.eye(4)]
        self.fallbacks = [False]

    def add_view(self, view):
        """注册并融合一个新视角：
        1) 先尝试直接配准到累积模型
        2) 若失败，用链式配准求初值，再 ICP 精调到模型"""
        i = len(self.views)
        used_fallback = False

        # 直接配准到模型
        result = register(view, self.model_down,
                          self.voxel_size, self.threshold)
        T_ref = result.transformation

        if result.fitness < self.min_fitness:
            # 回退：先配准到上一帧（重叠大），得到链式初值
            chain = register(view, self.views[i - 1],
                              self.voxel_size, self.threshold)
            T_chain = np.dot(self.poses[i - 1], chain.transformation)

            # 用链式结果做初值，ICP 精调到累积模型
            from .icp import icp_point_to_point
            result = icp_point_to_point(view, self.model_down,
                                         self.threshold, T_chain)
            if result.fitness > chain.fitness:
                T_ref = result.transformation
            else:
                T_ref = T_chain
            used_fallback = True

        self.views.append(view)
        self.poses.append(T_ref)
        self.fallbacks.append(used_fallback)

        # 融入模型
        aligned = copy.deepcopy(view).transform(T_ref)
        self.model += aligned

        # 周期性降采样
        if i % self.downsample_every == 0:
            self.model_down = self.model.voxel_down_sample(self.voxel_size)

        return result

    def finalize(self):
        """最后一次降采样，在拼接前调用"""
        self.model_down = self.model.voxel_down_sample(self.voxel_size)

    def stitched(self, voxel_final_mult=0.5):
        """返回降采样后的最终拼接点云"""
        vs = max(self.voxel_size * voxel_final_mult, 0.002)
        return self.model.voxel_down_sample(vs)

    def intermediate(self, step, voxel_final_mult=0.5):
        """重建第 step 步时的中间模型"""
        partial = o3d.geometry.PointCloud()
        for j in range(min(step + 1, len(self.views))):
            partial += copy.deepcopy(self.views[j]).transform(self.poses[j])
        vs = max(self.voxel_size * voxel_final_mult, 0.002)
        return partial.voxel_down_sample(vs)
