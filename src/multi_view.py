"""增量式多视角拼接"""

import copy
import numpy as np
import open3d as o3d

from .registration import register
from .utils import preprocess


def _ransac_register(src, dst, voxel_size, threshold):
    src_d, src_f = preprocess(src, voxel_size)
    dst_d, dst_f = preprocess(dst, voxel_size)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, dst_d, src_f, dst_f, True,
        threshold * 3,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold * 3)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return o3d.pipelines.registration.registration_icp(
        src, dst, threshold, result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))


def _try_one(src, dst, voxel_size, threshold):
    return register(src, dst, voxel_size, threshold)


class MultiViewStitcher:
    """增量式多视角点云拼接器"""

    def __init__(self, voxel_size, threshold, min_fitness=0.25,
                 downsample_every=3, max_model_points=2000):
        self.voxel_size = voxel_size
        self.threshold = threshold
        self.min_fitness = min_fitness
        self.downsample_every = downsample_every
        self.max_model_points = max_model_points

        self.model = None
        self.model_down = None
        self.views = []
        self.poses = []
        self.fallbacks = []    # -1=模型直接成功, >=0=以此为参考回退成功

    def _downsample_model(self):
        vs = self.voxel_size
        for _ in range(10):
            md = self.model.voxel_down_sample(vs)
            if len(md.points) <= self.max_model_points:
                break
            vs *= 1.5
        self.model_down = md

    def init_with_view(self, view):
        self.model = copy.deepcopy(view)
        self.views = [view]
        self.poses = [np.eye(4)]
        self.fallbacks = [-1]
        self._downsample_model()

    def add_view(self, view):
        i = len(self.views)
        ref_idx = -1   # -1=模型, >=0=该视角兜底

        scales = [(self.voxel_size, self.threshold),
                  (self.voxel_size * 1.5, self.threshold * 1.5),
                  (self.voxel_size * 2, self.threshold * 2)]

        # 1) 优先配准到累积大模型 — 选最精细尺度中 fitness 达标的
        best, best_fitness = None, -1.0
        for vs, th in scales:
            r = _try_one(view, self.model_down, vs, th)
            if r.fitness >= self.min_fitness:
                best, best_fitness = r, r.fitness
                T_ref = r.transformation
                ref_idx = -1
                break  # 精细尺度达标，不再粗化
            if r.fitness > best_fitness:
                best, best_fitness = r, r.fitness
                T_ref = r.transformation

        # 2) 模型不达标 → 多参考兜底
        if best_fitness < self.min_fitness:
            for j in range(i - 1, max(i - 10, -1), -1):
                for vs, th in scales:
                    r = _try_one(view, self.views[j], vs, th)
                    if r.fitness > best_fitness:
                        best_fitness = r.fitness
                        T_ref = np.dot(self.poses[j], r.transformation)
                        best = r
                        ref_idx = j
                    if r.fitness >= self.min_fitness:
                        break
                if best_fitness >= self.min_fitness:
                    break

        # 3) 还不行 → RANSAC 兜底
        if best_fitness < 0.2:
            for j in range(i - 1, max(i - 5, -1), -1):
                r = _ransac_register(view, self.views[j],
                                     self.voxel_size * 2, self.threshold * 2)
                if r.fitness > best_fitness:
                    best_fitness = r.fitness
                    T_ref = np.dot(self.poses[j], r.transformation)
                    best = r
                    ref_idx = j

        self.views.append(view)
        self.poses.append(T_ref)
        self.fallbacks.append(ref_idx)

        aligned = copy.deepcopy(view).transform(T_ref)
        self.model += aligned

        if i % self.downsample_every == 0:
            self._downsample_model()

        return best, ref_idx

    def finalize(self):
        self._downsample_model()

    def stitched(self, voxel_final_mult=0.5):
        vs = max(self.voxel_size * voxel_final_mult, 0.002)
        return self.model.voxel_down_sample(vs)

    def intermediate(self, step, voxel_final_mult=0.5):
        partial = o3d.geometry.PointCloud()
        for j in range(min(step + 1, len(self.views))):
            partial += copy.deepcopy(self.views[j]).transform(self.poses[j])
        vs = max(self.voxel_size * voxel_final_mult, 0.002)
        return partial.voxel_down_sample(vs)
