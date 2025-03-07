import open3d as o3d
import numpy as np

def filter_and_save_point_cloud(file_path, output_path):
    """
    读取点云文件，删除所有 z 坐标小于 -15 的点，并保存。
    :param file_path: 输入点云文件路径
    :param output_path: 输出处理后的点云文件路径
    """
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    
    # 过滤 z 坐标小于 -15 的点
    filtered_points = points[points[:, 2] >= -15]
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    # 保存处理后的点云
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"处理后的点云已保存至 {output_path}")

def visualize_point_cloud(file_path):
    """
    可视化点云数据。
    :param file_path: 点云文件路径
    """
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])

# 示例：加载并过滤 .ply 文件，然后保存并可视化
filtered_ply_path = "image/filtered_point_cloud.ply"
filter_and_save_point_cloud("image/point_cloud.ply", filtered_ply_path)
visualize_point_cloud(filtered_ply_path)