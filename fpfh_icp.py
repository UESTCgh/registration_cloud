import open3d as o3d
import numpy as np
import copy

'''
    这个程序是第一次跑通FPFH粗配准+icp精配准，后来也作为了一个标准历程。可以跑通online_source, Dino,Elephant,他们的特点是都比较小，一两毫米左右
'''
#该代码先基于FPFH全局配准，再用点到点ICP
#采样，法线估计，计算FPFH特征
def preprocess_point_cloud(pcd,voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2.0,max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5.0,max_nn=100))
    return (pcd_down,pcd_fpfh)

#初始参数：online_source, Dino,Elephant： 0.05  64  1000  1.5*  0.02，
#人为变换需要np.pi / 2, np.pi / 3, np.pi / 6   Chair在16  300 1000  1.5*  16  成功率高
#12 0.005 300 1000 1.5* 0.002
#bunny
voxel_size=0.005#采样格子大小max 初始0.05
max_iterations=300 # 配准的最大迭代次数 64
max_tuples=1000#最多同名点个数   1000
distance_threshold=1.5*voxel_size# 同名点之间距离阙值

#获取示例数据

src = o3d.io.read_point_cloud("1.pcd")
dst = o3d.io.read_point_cloud("2.pcd")
# dst = copy.deepcopy(dst).translate((0.5, 0, 0)) 
# #对他进行旋转
# T = np.eye(4)
# T[:3, :3] = src.get_rotation_matrix_from_xyz((np.pi / 2, np.pi / 3, np.pi / 6))
# T[0, 3] = 1.5
# T[1, 3] = 3
# print("人为变换矩阵")
# print(T)
# dst= copy.deepcopy(src).transform(T)


#配置颜色
src.paint_uniform_color([0,1,0])
dst.paint_uniform_color([0,0,1])

# 显示配准之前的点云
o3d.visualization.draw([src,dst])

#预处理
src_down,src_fpfh = preprocess_point_cloud(src, voxel_size)
dst_down,dst_fpfh = preprocess_point_cloud(dst ,voxel_size)

#显示预处理点云
#o3d.visualization.draw([src_down,dst_down])
#FGR
result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(src_down, dst_down, src_fpfh, dst_fpfh,
                                                                               o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold,# 同名点之间的最大距离
                                                                                                                                       iteration_number=max_iterations,#最大迭代次数
                                                                                                                                       maximum_tuple_count=max_tuples))# 同名点对个数
print("粗矩阵")
print(result.transformation)

#显示配准后的点云
src1= copy.deepcopy(src).transform(result.transformation)
o3d.visualization.draw([src1, dst])



threshold=0.002# RMSE残差网值，小于该残差闯值，迭代终止0.02 0.002
trans_init=result.transformation



# #点到点的ICP，和点到面只能启用一个！
# result1 = o3d.pipelines.registration.registration_icp(src,dst, threshold,trans_init,o3d.pipelines.registration.TransformationEstimationPointToPoint())
# print(result1)
# print("精矩阵")
# print(result1.transformation)
# src2= copy.deepcopy(src).transform(result1.transformation)
# o3d.visualization.draw([dst,src2])


# #点到面icp
dst.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
print("Apply point-to-plane ICP")
reg_p2l = o3d.pipelines.registration.registration_icp(
    src, dst, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=3000))
print(reg_p2l)
# print("点到面icp精矩阵")
# print(reg_p2l.transformation)
src2= copy.deepcopy(src).transform(reg_p2l.transformation)
o3d.visualization.draw([dst,src2])





# #一次验证，不然重合看不出来
# T = np.eye(4)
# T[:3, :3] = src2.get_rotation_matrix_from_xyz((0, 0, 0))
# T[0, 3] = 0.00
# T[1, 3] = -0.01
# src3= copy.deepcopy(src2).transform(T)
# o3d.visualization.draw([dst,src3])

