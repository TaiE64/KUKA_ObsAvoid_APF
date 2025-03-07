import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

### ========== 1. 读取PLY点云 ========== ###
def load_ply_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    print("点云加载成功，包含 {} 个点".format(len(pcd.points)))
    return pcd

### ========== 2. 预处理点云 ========== ###
def preprocess_point_cloud(pcd, voxel_size=0.05):
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    return pcd

### ========== 3. 计算APF势场 ========== ###
def compute_apf(pcd, goal, k_att=20.0, k_rep=1, d0=0.2):
    points = np.asarray(pcd.points)
    U_att = np.zeros(len(points))
    U_rep = np.zeros(len(points))
    
    min_dist = 0.1  # 避免斥力无限大

    for i, p in enumerate(points):
        dist_to_goal = np.linalg.norm(p - goal)
        U_att[i] = 0.5 * k_att * dist_to_goal**2

        for obs in points:
            dist_to_obs = np.linalg.norm(p - obs)
            if dist_to_obs < d0 and dist_to_obs > min_dist:
                U_rep[i] += 0.5 * k_rep * ((1 / (dist_to_obs + min_dist)) - (1 / d0))**2

    return U_att + U_rep, points

### ========== 4. 计算路径 (梯度下降) ========== ###
def compute_path(points, goal, k_att=20.0, k_rep=1, d0=0.2, alpha=0.1, max_iter=2000):
    pos = np.array([0.0, -0.25, -2.9])  # 机器人起始位置
    path = [pos]

    for _ in range(max_iter):
        F_att = -k_att * (pos - goal)
        F_rep = np.zeros(3)

        for obs in points:
            dist_to_obs = np.linalg.norm(pos - obs)
            if dist_to_obs < d0 and dist_to_obs > 0.1:
                F_rep += k_rep * (1/dist_to_obs - 1/d0) * (1/dist_to_obs**2) * (pos - obs)

        alpha = 0.1 / (1 + np.linalg.norm(F_att + F_rep))  # 动态调整步长
        pos = pos + alpha * (F_att + F_rep)
        path.append(pos)

        if np.linalg.norm(pos - goal) < 0.05:  # 终止条件
            break

    return np.array(path)

### ========== 5. 可视化 ========== ###
def visualize_point_cloud(pcd, U_total):
    colors = plt.cm.viridis(U_total / np.max(U_total))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def visualize_path_3d(points, path, goal):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(path)
    line_set.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(path)-1)])
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(path)-1)])

    goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    goal_sphere.paint_uniform_color([0, 1, 0])
    goal_sphere.translate(goal)

    o3d.visualization.draw_geometries([pcd, line_set, goal_sphere])

### ========== 主函数 ========== ###
if __name__ == "__main__":
    file_path = "image\point_cloud.ply"
    goal = np.array([0, 0.25, -2.9])

    pcd = load_ply_point_cloud(file_path)
    pcd = preprocess_point_cloud(pcd, voxel_size=0.05)
    
    U_total, points = compute_apf(pcd, goal)
    path = compute_path(points, goal)

    visualize_point_cloud(pcd, U_total)
    visualize_path_3d(points, path, goal)
