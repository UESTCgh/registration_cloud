import open3d as o3d
import numpy as np

#算点云密度
pcd =o3d.io.read_point_cloud("1.pcd")#读取点云数据

# pcd=pcd.uniform_down_sample(50)

point = np.asarray(pcd.points)#然职点警标
kdtree = o3d.geometry.KDTreeFlann(pcd)#立K树引
point_size = point.shape[0]#获眼点的个数
dd = np.zeros(point_size)
for i in range(point_size):
    [_,idx,dis]=kdtree.search_knn_vector_3d(point[i],2)
    dd[i] = dis[1]#获报最近邻点的距离一方

print(pcd)
density = np.mean(np.sqrt(dd))#计算平均密度
print("点云密度: density=",density)
# o3d.visualization.draw_geometries([pcd2,pcd3])