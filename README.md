# 点云配准工具集

基于 Open3D 实现的 FPFH 全局粗配准 + ICP 精配准管线，支持两两配准、多视角增量拼接、点云厚度评估。

## 项目结构

```
├── config/
│   ├── default.yaml          # 配准参数预设
│   └── multi_view.yaml       # 多视角拼接参数
├── src/                      # 算法核心库
│   ├── fgr.py                # Fast Global Registration (FPFH特征)
│   ├── icp.py                # ICP 精配准 (point-to-point / point-to-plane)
│   ├── registration.py       # 完整配准管线 (FGR + ICP)
│   ├── multi_view.py         # MultiViewStitcher 增量拼接类
│   └── utils.py              # 密度计算、厚度评估、预处理、截图
├── scripts/
│   ├── pairwise_demo.py      # 两两配准演示（含厚度评估）
│   └── multi_view_demo.py    # 多视角增量拼接演示
├── data/                     # 原始点云数据
│   └── multi/                # 生成的多视角数据集
└── result/                   # 结果输出
    └── multi/                # 多视角拼接结果
```

## 安装

```bash
pip3 install open3d numpy
```

## 快速开始

### 两两配准

```bash
python3 scripts/pairwise_demo.py
```

对 cloud_bin_0+1、1+2、lab2_sfm+kinect 三组真实扫描数据运行配准，并输出：
- 配准前后点云厚度（局部平面拟合残差的 RMS/均值/中位数/标准差/最大值）
- FGR+ICP 的 fitness 和 RMSE
- 变换矩阵
- 配准前后截图

### 多视角增量拼接

```bash
# 默认：完整点云 + 大旋转角 = 低重叠 + 高精度
python3 scripts/multi_view_demo.py

# 部分视图模式（每视角只保留可见表面）
python3 scripts/multi_view_demo.py --hidden --angle_range 20

# 自定义参数
python3 scripts/multi_view_demo.py --pcd Chair.pcd --views 6 --angle_range 30
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--pcd` | Dino.pcd | 源点云 |
| `--views` | 8 | 生成视角数 |
| `--angle_range` | 60 | 相邻视角最大旋转角 (度) |
| `--angle_min` | 0 | 相邻视角最小旋转角 (度) |
| `--trans_range` | 0.3 | 最大平移量 |
| `--noise` | 0.001 | 高斯噪声标准差 |
| `--hidden` | 关闭 | 启用隐藏点去除 |
| `--gen_only` | — | 仅生成数据集 |

## 配准参数经验

| 场景 | 密度范围 | voxel_size | threshold | angle_range |
|---|---|---|---|---|
| 高精度小物体 | <0.001 | 密度 × 8 | 密度 × 3 | 30-45° |
| 通用 | 0.001-0.01 | **密度 × 10** | **密度 × 5** | 45-60° |
| 大场景稀疏 | 0.01-0.1 | 密度 × 10 | 密度 × 5 | 30-40° |
| 超稀疏 | >1.0 | **密度 × 1.5** | 密度 × 1.5 | 20-40° |

> 详细指标见 [doc/metrics.md](doc/metrics.md)

## 点云厚度

对合并后点云的每个点，取 k 个近邻拟合局部平面，计算该点到平面的距离 d_i。厚度定义为：

$$\text{RMS} = \sqrt{\frac{1}{N}\sum_i d_i^2}$$

值越小 = 两片点云贴合越紧密 = 配准质量越好。真实双扫描数据永远有残留厚度（传感器噪声），同源配准则趋近于 0。

## 核心算法

**两两配准管线** (`src/registration.py`):
1. 降采样 + 法线估计 + FPFH 特征计算
2. Fast Global Registration（粗配准）
3. ICP Point-to-Point（精配准）

**增量拼接** (`src/multi_view.py`):
1. view_0 作为初始模型
2. 每来一个新视角：先尝试注册到累积模型
3. 若 fitness 低于阈值 → 回退到链式配准（注册到上一帧），再 ICP 精调到模型
4. 融合后周期性降采样控制模型规模
