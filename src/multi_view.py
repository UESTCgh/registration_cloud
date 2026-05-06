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
        """注册并融合一个新视角"""
        i = len(self.views)

        # 尝试注册到累积模型
        result = register(view, self.model_down,
                          self.voxel_size, self.threshold)
        used_fallback = False

        if result.fitness < self.min_fitness:
            # 回退到链式配准（注册到上一帧）
            result = register(view, self.views[i - 1],
                              self.voxel_size, self.threshold)
            T_ref = np.dot(self.poses[i - 1], result.transformation)
            used_fallback = True
        else:
            T_ref = result.transformation

        self.views.append(view)
        self.poses.append(T_ref)
        self.fallbacks.append(used_fallback)

        # 融入模型
        aligned = copy.deepcopy(view).transform(T_ref)
        self.model += aligned

        # 周期性降采样（每 N 步）
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
