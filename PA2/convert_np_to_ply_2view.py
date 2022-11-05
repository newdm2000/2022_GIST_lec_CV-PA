import open3d as o3d
import numpy as np
import pandas as pd
df = pd.read_csv('./result/2view/3D_Point_Clouds_Two_Views.csv', sep=',', header=None)
point_cloud = np.array(np.asarray(df.values)[1:, 1:], dtype=np.float32)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:, 0:3])
pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6] / 255.0)
o3d.io.write_point_cloud("./result/2view/3D_Point_Clouds_Two_Views.ply", pcd)