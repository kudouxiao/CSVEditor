import numpy as np
import mujoco

# ============ 默认路径 ============
DEFAULT_CSV_PATH = "/home/jq/project/CSVEditor/retargeted/right31.csv"
DEFAULT_MODEL_PATH = "/home/jq/project/CSVEditor/assets/unitree_g1/g1_mocap_29dof.xml"

# SMPL-X 配置
DEFAULT_SMPLX_DATA_PATH = "/home/jq/project/CSVEditor/smplx/left91.npz"
SMPLX_BODY_MODEL_DIR = "/home/jq/project/CSVEditor/assets/body_models" 

# ============ BVH 配置 ============
DEFAULT_BVH_PATH = "/home/jq/project/CSVEditor/bvh/left71_cut-zuo.bvh" 

# 可选值: "SMPL", "BVH", "AUTO" (AUTO = 优先SMPL，没有则找BVH)
REF_LOAD_MODE = "BVH"

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