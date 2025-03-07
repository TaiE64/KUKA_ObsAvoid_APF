import pybullet as p
import pybullet_data
import numpy as np
import cv2
import os
import time

# 连接 PyBullet
_ = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 加载地面和箱子
planeUid = p.loadURDF("plane.urdf", useMaximalCoordinates=True)
time.sleep(1)
trayUid = p.loadURDF("tray/tray.urdf", basePosition=[0.5, 0, 0],useFixedBase=1)
p.setGravity(0, 0, -10)

# 相机参数
width = 1080  
height = 720  
fov = 50  
aspect = width / height  
near = 0.01  
far = 20  

cameraPos = [0.5, 0, 2]  # 相机位置
targetPos = [0.5, 0, 0]  # 目标点
cameraupPos = [0, 1, 0]  # 上方向设为z轴方向

# 计算视角和投影矩阵
viewMatrix = p.computeViewMatrix(cameraPos, targetPos, cameraupPos)
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# 获取相机图像
w, h, rgb, depth, seg = p.getCameraImage(width, height, viewMatrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

# 处理 RGB 图像
rgbImg = np.array(rgb, dtype=np.uint8)  
rgbImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGBA2RGB)  

# 创建存储文件夹
save_path = 'image'
os.makedirs(save_path, exist_ok=True)

# 存储 RGB 图像
cv2.imwrite(os.path.join(save_path, 'rgb.jpg'), rgbImg)

# 存储原始深度图（不归一化）
depthImg = np.array(depth, dtype=np.float32)
np.save(os.path.join(save_path, 'depth.npy'), depthImg)

# 等待2秒后断开连接
time.sleep(2)
# input()
p.disconnect()

def depth_to_point_cloud(depth, rgb, view_matrix, projection_matrix):
    """
    将深度图和RGB图转换为点云
    """
    height, width = depth.shape
    fx = width / (2 * np.tan(np.radians(50) / 2))  # 由FOV计算焦距
    fy = fx  
    cx, cy = width / 2, height / 2  

    # 修正 PyBullet 的深度范围转换（重要）
    far = 20
    near = 0.01
    depth = -far * near / (far - (far - near) * depth)  # 深度解码

    # 生成像素网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u, v = u.flatten(), v.flatten()

    # 计算世界坐标
    z = depth.flatten()
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.vstack((x, y, z, np.ones_like(z)))  

    # 计算相机坐标系到世界坐标系的变换
    inv_view_matrix = np.linalg.inv(np.array(view_matrix).reshape(4, 4))
    world_points = inv_view_matrix @ points  
    world_points = world_points[:3, :].T  

    # 归一化颜色
    colors = np.clip(rgb.reshape(-1, 3) / 255.0, 0, 1)  

    return world_points, colors

# 加载 RGB 和深度图
rgb = cv2.imread('image/rgb.jpg')
depth = np.load('image/depth.npy')

# 计算点云
points, colors = depth_to_point_cloud(depth, rgb, viewMatrix, projection_matrix)

# 存储为 PLY 点云格式
ply_header = f'''ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

# 组合点云数据
ply_data = np.hstack((points, (colors * 255).astype(np.uint8)))

# 保存 PLY
save_path = 'image/point_cloud.ply'
with open(save_path, 'w') as f:
    f.write(ply_header)
    np.savetxt(f, ply_data, fmt="%.4f %.4f %.4f %d %d %d")

print(f"点云已保存到 {save_path}")
