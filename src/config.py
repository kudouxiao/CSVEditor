import numpy as np
import mujoco

import os
import configparser

# ============ 路径定义 (用户数据) ============
USER_DATA_DIR = os.path.join(os.path.expanduser("~"), ".g1_pro_editor")
os.makedirs(USER_DATA_DIR, exist_ok=True)

CONFIG_FILE = os.path.join(USER_DATA_DIR, "settings.ini")
LICENSE_PATH = os.path.join(USER_DATA_DIR, "g1_editor.lic")

# ============ 动态配置解析 ============
config = configparser.ConfigParser()
if os.path.exists(CONFIG_FILE):
    config.read(CONFIG_FILE)

def set_setting(section, key, value):
    if not config.has_section(section):
        config.add_section(section)
    config.set(section, key, value)
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

def get_setting(section, key, fallback):
    if config.has_section(section) and config.has_option(section, key):
        return config.get(section, key)
    return fallback

# ============ 默认路径 ============
DEFAULT_CSV_PATH = ""
# 默认模型路径可以指向上层目录里的资产
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "unitree_g1", "g1_mocap_29dof.xml"))

# SMPL-X 配置
DEFAULT_SMPLX_DATA_PATH = ""
# 默认优先读取配置文件中的路径，如果没有，则默认为当前目录下（用户自己选定一次后将覆盖此路径）
DEFAULT_SMPLX_DIR = ""
SMPLX_BODY_MODEL_DIR = get_setting("Paths", "SMPLX_BODY_MODEL_DIR", DEFAULT_SMPLX_DIR) 

# ============ BVH 配置 ============
DEFAULT_BVH_PATH = "" 

# 可选值: "SMPL", "BVH", "AUTO" (AUTO = 优先SMPL，没有则找BVH)
REF_LOAD_MODE = "AUTO"

ROBOT_FPS = 30

# ============ 关节定义 ============
CSV_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", 
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint"
]

# Root + Joints 完整列表
ALL_CHANNEL_NAMES = [
    "Root_Pos_X", "Root_Pos_Y", "Root_Pos_Z",
    "Root_Quat_W", "Root_Quat_X", "Root_Quat_Y", "Root_Quat_Z"
] + CSV_JOINT_NAMES

# ============ SMPL 配置 ============
# 骨骼连接关系
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# ============ 渲染配置 ============
RENDER_FLAGS = {
            # 可视化 (mjVIS) - 控制 opt.flags
            "Joints (关节轴)":    (mujoco.mjtVisFlag.mjVIS_JOINT,     False, False),
            "Actuators (执行器)": (mujoco.mjtVisFlag.mjVIS_ACTUATOR,  False, False),
            # MuJoCo 3.x 使用 mjVIS_CONSTRAINT 显示接触和约束
            "Constraints (接触)": (mujoco.mjtVisFlag.mjVIS_CONSTRAINT, False, False), 
            "CoM (质心)":         (mujoco.mjtVisFlag.mjVIS_COM,       False, False),
            "Transparent (半透明)":(mujoco.mjtVisFlag.mjVIS_TRANSPARENT, False, False),
            "Convex Hull (凸包)": (mujoco.mjtVisFlag.mjVIS_CONVEXHULL, False, False),
            "Inertia (惯性框)":   (mujoco.mjtVisFlag.mjVIS_INERTIA,   False, False),
            "Perturbation (扰动)":(mujoco.mjtVisFlag.mjVIS_PERTFORCE, True, False),
            
            # 渲染效果 (mjRND) - 控制 scn.flags
            "Shadows (阴影)":     (mujoco.mjtRndFlag.mjRND_SHADOW,    False,  True),
            "Wireframe (线框)":   (mujoco.mjtRndFlag.mjRND_WIREFRAME, False, True),
            "Reflection (反射)":  (mujoco.mjtRndFlag.mjRND_REFLECTION, True, True),
            "Fog (雾效)":         (mujoco.mjtRndFlag.mjRND_FOG,       False, True),
        }

LABEL_OPTIONS = {
            "None (无)": mujoco.mjtLabel.mjLABEL_NONE,
            "Body Name (刚体名)": mujoco.mjtLabel.mjLABEL_BODY,
            "Geom Name (几何体名)": mujoco.mjtLabel.mjLABEL_GEOM,
            "Joint Name (关节名)": mujoco.mjtLabel.mjLABEL_JOINT,
            "Site Name (位点名)": mujoco.mjtLabel.mjLABEL_SITE,
            "Camera Name (相机名)": mujoco.mjtLabel.mjLABEL_CAMERA,
            "Light Name (灯光名)": mujoco.mjtLabel.mjLABEL_LIGHT,
            "Tendon Name (肌腱名)": mujoco.mjtLabel.mjLABEL_TENDON,
            "Actuator Name (执行器名)": mujoco.mjtLabel.mjLABEL_ACTUATOR,
            "Constraint Name (约束名)": mujoco.mjtLabel.mjLABEL_CONSTRAINT,
            "Selection (仅选中)": mujoco.mjtLabel.mjLABEL_SELECTION,
        }

FRAME_OPTIONS = {
            "None (无)": mujoco.mjtFrame.mjFRAME_NONE,
            "Body (刚体坐标)": mujoco.mjtFrame.mjFRAME_BODY,
            "Geom (几何坐标)": mujoco.mjtFrame.mjFRAME_GEOM,
            "Site (位点坐标)": mujoco.mjtFrame.mjFRAME_SITE,
            "Camera (相机坐标)": mujoco.mjtFrame.mjFRAME_CAMERA,
            "Light (灯光坐标)": mujoco.mjtFrame.mjFRAME_LIGHT,
            "World (世界坐标)": mujoco.mjtFrame.mjFRAME_WORLD,
        }