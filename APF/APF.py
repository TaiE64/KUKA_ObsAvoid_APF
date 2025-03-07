import pybullet as p
import pybullet_data
import numpy as np
import time
from math import pi
import sys
sys.path.append("c:/Users/33582/Desktop/Obs_avoid_APF/RGB2PLY")  # 添加 APF 目录到路径

import apf2 as apf
# 仿真设置
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setPhysicsEngineParameter(enableConeFriction=1, deterministicOverlappingPairs=1, useSplitImpulse=1, splitImpulsePenetrationThreshold=0.01)
p.setGravity(0, 0, -9.8)

plane = p.loadURDF("plane.urdf")

# 设置工作空间障碍物
obstacle_pos = [0.5, 0, 0]  # 修正障碍物位置
obstacle = p.loadURDF("tray/tray.urdf", obstacle_pos,useFixedBase=1)

# 加载UR5机械臂
robot = p.loadURDF("utils\KUKA\model.urdf", [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
num_joints = p.getNumJoints(robot)
end_effector_index = num_joints - 1

# 机械臂参数配置
joint_indices = [i for i in range(num_joints) if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED]
joint_limits = [(p.getJointInfo(robot, i)[8], p.getJointInfo(robot, i)[9]) for i in joint_indices]

# 目标设置
target_pos = [1, 0, 0.3]
target_orientation = p.getQuaternionFromEuler([0, pi/2, 0])

# 逆运动学求解初始配置
ik_start = [0]*15
ik_solution = p.calculateInverseKinematics(robot, end_effector_index, 
                                        target_pos, target_orientation,
                                        lowerLimits=[-pi]*6,
                                        upperLimits=[pi]*6,
                                        jointDamping=[0.1]*6)

def moving_average_filter(path, window_size=10):
    """
    对路径进行滑动窗口平均滤波
    :param path: (n, 3) numpy 数组，表示路径点
    :param window_size: 滑动窗口大小
    :return: 平滑后的路径 (n, 3)
    """
    smoothed_path = np.zeros_like(path)
    smoothed_path[0]=path[0]
    path=path[1:]
    for i in range(len(path)):
        start = max(0, i - window_size//2)
        end = min(len(path), i + window_size//2 + 1)
        smoothed_path[i+1] = np.mean(path[start:end], axis=0)
    return smoothed_path

def Visualization(points,goal,start):
    path = moving_average_filter(apf.compute_path(points,goal,start))
    apf.visualize_path_3d(points, path, goal)

# 生成初始轨迹（关节空间）
num_waypoints = 15
file_path = "image/filtered_point_cloud.ply"
start=np.array([0, 0, -2])
goal = np.array([0, 0.4, -2])

pcd = apf.load_ply_point_cloud(file_path)
pcd = apf.preprocess_point_cloud(pcd, voxel_size=0.05)
U_total, points = apf.compute_apf(pcd, goal)

Visualization(points,goal,start)

path = apf.compute_path(points,goal,start)
path=apf.pathMapping(path,obstacle_pos)
path = moving_average_filter(path, window_size=5)  # 平滑路径


# 让机械臂沿着 path 运动
for i, pos in enumerate(path):
    target_pos = pos.tolist()  # 转换为 Python 列表
    ik_solution = p.calculateInverseKinematics(robot, end_effector_index, target_pos)

    # 控制关节移动到目标位置
    p.setJointMotorControlArray(robot, joint_indices, 
                                controlMode=p.POSITION_CONTROL, 
                                targetPositions=ik_solution)

    # 让仿真运行一段时间，让机械臂移动
    for _ in range(100):  # 调整步数以控制平滑度
        p.stepSimulation()
        time.sleep(1/240)  # 240Hz 物理仿真步长

    print(f"Step {i+1}/{len(path)}: Target {target_pos}")

print("Path following complete!")

# 保持仿真
time.sleep(2)
input()
p.disconnect()