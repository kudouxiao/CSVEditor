import sys
import threading
import time
import os
import numpy as np
import pandas as pd
import mujoco
from scipy.interpolate import PchipInterpolator

# 新增依赖
import torch
import smplx

# UI 库
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QListWidget, QListWidgetItem, QSplitter, QFileDialog, 
                             QSpinBox, QDoubleSpinBox, QGroupBox, QMessageBox, 
                             QAbstractItemView, QFrame, QShortcut, QOpenGLWidget, 
                             QSizePolicy, QCheckBox, QTabWidget, QComboBox, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QSize, QTimer, QPoint
from PyQt5.QtGui import QKeySequence, QMouseEvent, QWheelEvent

# 绘图库
import pyqtgraph as pg

# 主题
try:
    import qdarktheme
except ImportError:
    pass

# ============ 配置区 ============
DEFAULT_CSV_PATH = "/home/jq/project/CSVEditor/retargeted/left91gai.csv"
DEFAULT_MODEL_PATH = "/home/jq/project/CSVEditor/assets/unitree_g1/g1_mocap_29dof.xml"

# SMPL-X 配置
DEFAULT_SMPLX_DATA_PATH = "/home/jq/project/CSVEditor/smplx/left91.npz"
SMPLX_BODY_MODEL_DIR = "/home/jq/project/CSVEditor/assets/body_models" 

# SMPL-X 骨骼连接关系
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# ============ 1. MuJoCo 渲染控件 (修复绘制调用) ============
class MuJoCoWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.data = None
        
        # SMPL-X
        self.smplx_joints = None 
        self.smplx_offset = np.array([0.0, 0.0, 0.0]) 
        self.smplx_frame_scale = 3.0 # 帧率缩放系数
        self.show_smplx = True
        self.current_frame_idx = 0
        
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption() 
        self.scn = None
        self.con = None
        self.pert = mujoco.MjvPerturb()
        self.last_mouse_pos = QPoint()
        
        # === 1. 开关型标志 (Checkbox) ===
        # 格式: "显示名称": (MuJoCo枚举, 默认开启?, 是否渲染标志mjRND)
        self.render_flags = {
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
            "Shadows (阴影)":     (mujoco.mjtRndFlag.mjRND_SHADOW,    True,  True),
            "Wireframe (线框)":   (mujoco.mjtRndFlag.mjRND_WIREFRAME, False, True),
            "Reflection (反射)":  (mujoco.mjtRndFlag.mjRND_REFLECTION, True, True),
            "Fog (雾效)":         (mujoco.mjtRndFlag.mjRND_FOG,       False, True),
        }

        # === 2. 标签显示模式 (mjLABEL) ===
        self.label_options = {
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

        # === 3. 坐标系显示模式 (mjFRAME) ===
        self.frame_options = {
            "None (无)": mujoco.mjtFrame.mjFRAME_NONE,
            "Body (刚体坐标)": mujoco.mjtFrame.mjFRAME_BODY,
            "Geom (几何坐标)": mujoco.mjtFrame.mjFRAME_GEOM,
            "Site (位点坐标)": mujoco.mjtFrame.mjFRAME_SITE,
            "Camera (相机坐标)": mujoco.mjtFrame.mjFRAME_CAMERA,
            "Light (灯光坐标)": mujoco.mjtFrame.mjFRAME_LIGHT,
            "World (世界坐标)": mujoco.mjtFrame.mjFRAME_WORLD,
        }
        
        self.setMinimumHeight(400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)

    def init_mujoco(self, model, data):
        self.model = model
        self.data = data
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat = np.array([0, 0, 1.0]) 
        self.cam.distance = 4.0
        self.cam.azimuth = 90
        self.cam.elevation = -15
        
        mujoco.mjv_defaultOption(self.opt)
        mujoco.mjv_defaultPerturb(self.pert)
        
        if self.model:
            self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.update()

    def set_smplx_data(self, joints):
        self.smplx_joints = joints
        # 自动对齐：将第一帧中心移到 (0, 1, 1)
        if len(joints) > 0:
            # 兼容 Tensor 和 Numpy
            first_frame = joints[0]
            if hasattr(first_frame, 'cpu'): first_frame = first_frame.detach().cpu().numpy()
            
            center = first_frame[0] 
            self.smplx_offset = -center + np.array([0, 1.0, 1.0]) # Y错开1米, Z抬高1米
            print(f"[SMPL] Auto Offset applied: {self.smplx_offset}")
        self.update()

    def set_render_flag(self, flag_name, enabled):
        if flag_name not in self.render_flags: return
        mj_flag, _, is_rnd = self.render_flags[flag_name]
        if is_rnd:
            if self.scn: self.scn.flags[mj_flag] = 1 if enabled else 0
        else:
            self.opt.flags[mj_flag] = 1 if enabled else 0
        self.update()
        
    def set_label_mode(self, mode_name):
        if mode_name in self.label_options: 
            self.opt.label = self.label_options[mode_name]
            self.update()
            
    def set_frame_mode(self, mode_name):
        if mode_name in self.frame_options: 
            self.opt.frame = self.frame_options[mode_name]
            self.update()

    def initializeGL(self):
        if self.model and not self.con:
            try: 
                self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            except Exception as e: 
                print(e)

    def paintGL(self):
        if not self.model or not self.data or not self.scn or not self.con: 
            return
        
        viewport = mujoco.MjrRect(0, 0, self.width(), self.height())
        
        # 1. 更新 MuJoCo 场景
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, self.pert, 
            self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn
        )
        
        # 2. 绘制调试基准点 (绿色大球)
        self.draw_debug_marker()

        # 3. 绘制 SMPL (确保这一步被调用)
        if self.show_smplx and self.smplx_joints is not None:
            self.add_smplx_to_scene()

        # 4. 提交渲染
        mujoco.mjr_render(viewport, self.scn, self.con)

    def draw_debug_marker(self):
        """原点绘制绿球"""
        if self.scn.ngeom >= self.scn.maxgeom: 
            return
        mujoco.mjv_initGeom(
            self.scn.geoms[self.scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0],
            pos=[0, 0, 0],
            mat=np.eye(3).flatten(),
            rgba=[0, 1, 0, 1] # 纯绿
        )
        self.scn.ngeom += 1

    def add_smplx_to_scene(self):
        """
        绘制 SMPL 骨架 (高性能版: 使用 mjv_makeConnector 避免 Python 计算矩阵)
        """
        num_frames = len(self.smplx_joints)
        if num_frames == 0: 
            return
        
        # 计算映射帧: RobotFrame * Scale
        target_frame = int(self.current_frame_idx * self.smplx_frame_scale)
        frame_idx = min(target_frame, num_frames - 1)
        
        joints = self.smplx_joints[frame_idx]
        if hasattr(joints, 'cpu'): 
            joints = joints.detach().cpu().numpy()
        
        # 应用偏移
        joints = joints + self.smplx_offset
        
        # 1. 关节 (红色不透明)
        for j_pos in joints[:22]: 
            if self.scn.ngeom >= self.scn.maxgeom: 
                break
            mujoco.mjv_initGeom(
                self.scn.geoms[self.scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.03, 0, 0], 
                pos=j_pos,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 1] 
            )
            self.scn.ngeom += 1
            
        # 2. 骨骼 (白色半透明胶囊)
        # 使用 mjv_makeConnector 自动计算旋转，性能提升 10 倍
        for i, parent_idx in enumerate(SMPL_PARENTS):
            if parent_idx == -1 or i >= 22: 
                continue
            if self.scn.ngeom >= self.scn.maxgeom: 
                break
            
            p1 = joints[parent_idx]
            p2 = joints[i]
            
            # 核心优化：让 MuJoCo C 引擎计算胶囊的朝向
            mujoco.mjv_connector(
                self.scn.geoms[self.scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                width=0.015, 
                from_=p1,
                to=p2
            )
            # 覆盖颜色
            self.scn.geoms[self.scn.ngeom].rgba = np.array([1, 1, 1, 0.5])
            self.scn.ngeom += 1

    def rotation_matrix_from_vectors(self, vec1, vec2):
        a = vec1 / np.linalg.norm(vec1)
        b = vec2 / np.linalg.norm(vec2)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s < 1e-6: 
            return np.eye(3) 
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    # 鼠标交互
    def mousePressEvent(self, event: QMouseEvent): 
        self.last_mouse_pos = event.pos()
        event.accept()
        
    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.model: 
            return
        dx = event.x()-self.last_mouse_pos.x()
        dy = event.y()-self.last_mouse_pos.y()
        self.last_mouse_pos = event.pos()
        width, height = self.width(), self.height()
        if event.buttons() & Qt.LeftButton:
            if dx != 0: 
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_H, dx/height, 0, self.scn, self.cam)
            if dy != 0: 
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, 0, dy/height, self.scn, self.cam)
        elif event.buttons() & Qt.RightButton:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_V, dx/height, dy/height, self.scn, self.cam)
        elif event.buttons() & Qt.MidButton:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, dx/height, dy/height, self.scn, self.cam)
        self.update()
        event.accept()
        
    def wheelEvent(self, event: QWheelEvent):
        dy = event.angleDelta().y(); 
        if abs(dy) < 1: 
            return
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -dy*0.002, self.scn, self.cam)
        self.update()
        event.accept()
        
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self.data: 
            self.cam.lookat = np.array(self.data.qpos[0:3])
            self.cam.distance = 3.0
            self.update()

# ============ 2. 后端逻辑 ============
class G1Backend(QObject):
    def __init__(self):
        super().__init__()
        self.df = None
        self.df_orig = None
        self.model = None
        self.data = None
        self.smplx_joints = None 
        self.lock = threading.Lock()
        self.joint_mapping = {}
        self.modified_frames = set()
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 50

        self.root_names = [
            "Root_Pos_X", "Root_Pos_Y", "Root_Pos_Z",
            "Root_Quat_W", "Root_Quat_X", "Root_Quat_Y", "Root_Quat_Z"
        ]

        self.csv_joint_names = [
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

        self.all_names = self.root_names + self.csv_joint_names

    def load_data(self, csv_path, model_path):
        try:
            self.df = pd.read_csv(csv_path, header=None)
            if isinstance(self.df.iloc[0, 0], str): 
                self.df = pd.read_csv(csv_path)
            self.df_orig = self.df.copy()
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
            self.model.opt.disableflags = 65535 
            self.data = mujoco.MjData(self.model)
            
            model_j_names = {}
            q_ptr = 0
            for i in range(self.model.njnt):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if self.model.jnt_type[i] == 0: 
                    q_ptr += 7
                    continue
                model_j_names[name] = q_ptr
                q_ptr += 1
            for idx, name in enumerate(self.csv_joint_names):
                simple = name.replace("_joint", "")
                for m_n, addr in model_j_names.items():
                    if simple in m_n or m_n in simple: 
                        self.joint_mapping[idx] = addr
                        break
            
            return True, len(self.df)
        except Exception as e:
            print(f"Load Error: {e}")
            return False, 0

    # === 稳健的 SMPL-X 加载 ===
    def load_smplx_data(self, smplx_file, smplx_body_model_path):
        try:
            print(f"[INFO] Loading SMPL-X: {smplx_file}")
            smplx_data = np.load(smplx_file, allow_pickle=True)
            data_dict = {k: smplx_data[k] for k in smplx_data.files}
            
            gender = str(data_dict.get("gender", "neutral"))
            if isinstance(gender, bytes): 
                gender = gender.decode('utf-8')
            if gender == 'n': 
                gender = 'neutral'

            if not os.path.exists(smplx_body_model_path):
                print(f"[ERROR] SMPL Model Dir not found: {smplx_body_model_path}")
                return False

            body_model = smplx.create(
                smplx_body_model_path,
                "smplx",
                gender=gender,
                use_pca=False,
                dtype=torch.float32 
            )
            
            num_frames = data_dict["pose_body"].shape[0]
            print(f"[INFO] SMPL Frames: {num_frames}")
            
            def to_tensor(x): 
                return torch.tensor(x).float().cpu()
            
            output = body_model(
                betas=to_tensor(data_dict["betas"]).view(1, -1),
                global_orient=to_tensor(data_dict["root_orient"]),
                body_pose=to_tensor(data_dict["pose_body"]),
                transl=to_tensor(data_dict["trans"]),
                left_hand_pose=torch.zeros(num_frames, 45).float(),
                right_hand_pose=torch.zeros(num_frames, 45).float(),
                jaw_pose=torch.zeros(num_frames, 3).float(),
                leye_pose=torch.zeros(num_frames, 3).float(),
                reye_pose=torch.zeros(num_frames, 3).float(),
                return_full_pose=True,
            )
            
            self.smplx_joints = output.joints # (N, 127, 3)
            print(f"[SUCCESS] SMPL-X Loaded. Joints: {self.smplx_joints.shape}")
            return True
        except Exception as e:
            print(f"[ERROR] SMPL Load Exception: {e}")
            import traceback
            traceback.print_exc()
            return str(e)

    def set_frame(self, idx):
        if self.df is None: 
            return
        with self.lock:
            line = pd.to_numeric(self.df.iloc[idx].values, errors='coerce')
            line = np.nan_to_num(line.astype(float))
            p = line[0:3]
            self.data.qpos[0:3] = p * 0.001 if np.any(np.abs(p) > 50) else p
            q = line[3:7]
            wxyz = np.array([q[3], q[0], q[1], q[2]])
            n = np.linalg.norm(wxyz)
            self.data.qpos[3:7] = wxyz/n if n > 1e-4 else [1,0,0,0]
            for c_idx, m_idx in self.joint_mapping.items():
                col = 7 + c_idx
                if col < len(line): 
                    self.data.qpos[m_idx] = line[col]
            mujoco.mj_forward(self.model, self.data)
            
    def snapshot(self):
        if self.df is None: 
            return
        self.undo_stack.append(self.df.copy())
        self.redo_stack.clear()
        if len(self.undo_stack) > self.max_history: 
            self.undo_stack.pop(0)
            
    def undo(self):
        if not self.undo_stack: 
            return False
        self.redo_stack.append(self.df)
        self.df = self.undo_stack.pop()
        return True
        
    def redo(self):
        if not self.redo_stack: 
            return False
        self.undo_stack.append(self.df)
        self.df = self.redo_stack.pop()
        return True

# ============ 3. 曲线编辑器 ============
class CurveEditor(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('#1e1e1e')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setMouseEnabled(x=True, y=False)
        self.region = pg.LinearRegionItem([0, 100], brush=(50, 50, 200, 50))
        self.region.setZValue(10)
        self.addItem(self.region)
        self.current_frame_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#FF5555', width=2), movable=True)
        self.current_frame_line.setZValue(100)
        self.addItem(self.current_frame_line)
        self.curves = {}
        self.selected_joint_idx = None
        self.backend_ref = None
        self.main_window_ref = None
        self.is_editing = False
        self.drag_start_pos = None
        self.drag_start_data = None
        
    def set_backend(self, backend, main_window):
        self.backend_ref = backend
        self.main_window_ref = main_window
        self.current_frame_line.sigDragged.connect(self.on_line_dragged)
        
    def update_curves(self, selected_indices):
        for item in self.curves.values():
            self.removeItem(item)
        self.curves.clear()
        if not self.backend_ref or self.backend_ref.df is None: 
            return
        if len(selected_indices) > 0:
            self.selected_joint_idx = selected_indices[0]
            for idx in selected_indices[1:]:
                # 修改：直接使用 idx，不再 +7
                col = idx
                data = self.backend_ref.df.iloc[:, col].values
                curve = self.plot(data, pen=pg.mkPen((80, 80, 80), width=1))
                curve.setZValue(5)
                self.curves[idx] = curve
            # 修改：直接使用 selected_joint_idx
            col = self.selected_joint_idx
            data = self.backend_ref.df.iloc[:, col].values
            main_curve = self.plot(data, pen=pg.mkPen('#00ffff', width=3))
            main_curve.setZValue(20)
            self.curves[self.selected_joint_idx] = main_curve
            self.autoRange()
        else: 
            self.selected_joint_idx = None
            
    def on_line_dragged(self):
        idx = int(self.current_frame_line.value())
        if self.main_window_ref: 
            self.main_window_ref.update_frame_from_graph(idx)

    def mousePressEvent(self, ev):
        if (ev.modifiers() & Qt.ControlModifier) and self.selected_joint_idx is not None and ev.button() == Qt.LeftButton:
            self.is_editing = True
            ev.accept()
            self.backend_ref.snapshot()
            widget_point = ev.pos()
            scene_point = self.mapToScene(widget_point)
            mouse_point = self.plotItem.vb.mapSceneToView(scene_point)
            self.drag_start_pos = mouse_point.y()
            r_min, r_max = self.region.getRegion()
            s, e = int(r_min), int(r_max)
            s = max(0, s)
            e = min(len(self.backend_ref.df)-1, e)
            # 修改：col = self.selected_joint_idx
            col = self.selected_joint_idx
            self.drag_start_data = self.backend_ref.df.iloc[s:e+1, col].values.copy()
            self.region.setMovable(False)
        else: 
            self.region.setMovable(True)
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.is_editing and not (ev.buttons() & Qt.LeftButton): 
            self.is_editing = False
            self.drag_start_data = None
            self.region.setMovable(True)
            super().mouseMoveEvent(ev)
            return
        if self.is_editing:
            ev.accept()
            widget_point = ev.pos()
            scene_point = self.mapToScene(widget_point)
            mouse_point = self.plotItem.vb.mapSceneToView(scene_point)
            y_curr = mouse_point.y()
            delta_y = y_curr - self.drag_start_pos
            r_min, r_max = self.region.getRegion()
            s, e = int(r_min), int(r_max)
            s = max(0, s)
            e = min(len(self.backend_ref.df)-1, e)
            if s < e and self.drag_start_data is not None:
                length = e - s + 1
                x = np.linspace(-np.pi, np.pi, length)
                weights = (np.cos(x) + 1) / 2 
                new_values = self.drag_start_data + (delta_y * weights)
                # 修改：col = self.selected_joint_idx
                col = self.selected_joint_idx
                self.backend_ref.df.iloc[s:e+1, col] = new_values
                self.backend_ref.modified_frames.update(range(s, e+1))
                self.curves[self.selected_joint_idx].setData(self.backend_ref.df.iloc[:, col].values)
                curr_f = int(self.current_frame_line.value())
                if s <= curr_f <= e: 
                    self.backend_ref.set_frame(curr_f)
                    self.main_window_ref.mujoco_widget.update()
        else: 
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self.is_editing: 
            self.is_editing = False
            self.drag_start_data = None
            self.region.setMovable(True)
            ev.accept()
        else: 
            super().mouseReleaseEvent(ev)

# ============ 4. 主窗口 ============
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("G1 Pro Editor (Diagnostic Build)")
        self.resize(1600, 1000)
        
        # 调试输出：检查路径
        print("="*40)
        print("SYSTEM CHECK:")
        print(f"CSV Path:   {DEFAULT_CSV_PATH} -> {os.path.exists(DEFAULT_CSV_PATH)}")
        print(f"Model Path: {DEFAULT_MODEL_PATH} -> {os.path.exists(DEFAULT_MODEL_PATH)}")
        print(f"SMPL Data:  {DEFAULT_SMPLX_DATA_PATH} -> {os.path.exists(DEFAULT_SMPLX_DATA_PATH)}")
        print(f"SMPL Dir:   {SMPLX_BODY_MODEL_DIR} -> {os.path.exists(SMPLX_BODY_MODEL_DIR)}")
        print("="*40)

        self.backend = G1Backend()
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next_frame)
        
        # 1. 自动加载机器人
        if os.path.exists(DEFAULT_CSV_PATH) and os.path.exists(DEFAULT_MODEL_PATH):
            success, frames = self.backend.load_data(DEFAULT_CSV_PATH, DEFAULT_MODEL_PATH)
            if success:
                self.mujoco_widget.init_mujoco(self.backend.model, self.backend.data)
                self.total_frames = frames
                self.graph.setXRange(0, frames)
                self.graph.region.setRegion([0, frames//5])
                self.update_frame(0)
                self.status_bar.showMessage(f"Loaded Robot Data: {frames} frames.")
        
        # 2. 自动加载 SMPL
        if os.path.exists(DEFAULT_SMPLX_DATA_PATH) and os.path.exists(SMPLX_BODY_MODEL_DIR):
            print("Attempting auto-load of SMPL data...")
            if self.backend.load_smplx_data(DEFAULT_SMPLX_DATA_PATH, SMPLX_BODY_MODEL_DIR):
                self.mujoco_widget.set_smplx_data(self.backend.smplx_joints)
                self.status_bar.showMessage(f"Loaded Robot & SMPL-X Ref")
            else:
                print("Auto-load failed inside backend.")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        top_bar = QHBoxLayout()
        self.btn_undo = QPushButton("[Undo]")
        self.btn_undo.clicked.connect(self.perform_undo)
        self.btn_redo = QPushButton("[Redo]")
        self.btn_redo.clicked.connect(self.perform_redo)
        btn_save = QPushButton("[Save As]")
        btn_save.clicked.connect(self.save_as)
        btn_load_smpl = QPushButton("[Load SMPL-X]")
        btn_load_smpl.clicked.connect(self.load_smplx_ref)
        
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.perform_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self.perform_redo)
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.toggle_play)
        
        top_bar.addWidget(self.btn_undo)
        top_bar.addWidget(self.btn_redo)
        top_bar.addSpacing(20)
        top_bar.addWidget(btn_save)
        top_bar.addWidget(btn_load_smpl)
        top_bar.addStretch()
        layout.addLayout(top_bar)
        
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        left_container = QWidget()
        l_layout = QVBoxLayout(left_container)
        l_layout.setContentsMargins(0,0,0,0)
        self.mujoco_widget = MuJoCoWidget()
        l_layout.addWidget(self.mujoco_widget, stretch=4)
        self.graph = CurveEditor()
        self.graph.set_backend(self.backend, self)
        l_layout.addWidget(self.graph, stretch=3)
        play_ctrl = QHBoxLayout()
        self.btn_prev = QPushButton("[<]")
        self.btn_prev.clicked.connect(lambda: self.jump(-1))
        self.btn_play = QPushButton("[Play/Space]")
        self.btn_play.setStyleSheet("background-color: #44aa44; font-weight: bold; color: white;")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_next = QPushButton("[>]")
        self.btn_next.clicked.connect(lambda: self.jump(1))
        self.lbl_frame = QLabel("0000")
        self.lbl_frame.setFixedWidth(50)
        self.lbl_frame.setStyleSheet("font-size: 14px; font-weight: bold; color: white;")
        play_ctrl.addWidget(self.btn_prev)
        play_ctrl.addWidget(self.btn_play)
        play_ctrl.addWidget(self.btn_next)
        play_ctrl.addSpacing(10)
        play_ctrl.addWidget(self.lbl_frame)
        l_layout.addLayout(play_ctrl)
        splitter.addWidget(left_container)
        
        # Right (Tabs)
        right_tabs = QTabWidget()
        
        # Tab 1: Edit
        tab_edit = QWidget()
        r_layout = QVBoxLayout(tab_edit)
        r_layout.addWidget(QLabel("Channels List (Root + Joints)"))
        self.joint_list = QListWidget()
        self.joint_list.setSelectionMode(QAbstractItemView.SingleSelection)

        for i, name in enumerate(self.backend.all_names): 
            # 给 Root 通道加个颜色区分
            item = QListWidgetItem(f"[{i:02d}] {name.replace('_joint','')}")
            if i < 7: 
                item.setForeground(Qt.cyan) # Root 设为青色
            self.joint_list.addItem(item)

        self.joint_list.itemSelectionChanged.connect(self.on_selection_change)
        r_layout.addWidget(self.joint_list)
        tools_group = QGroupBox("Tools")
        t_layout = QVBoxLayout()
        btn_smooth = QPushButton("[Smooth Region]")
        btn_smooth.clicked.connect(self.apply_smooth)
        t_layout.addWidget(btn_smooth)
        btn_add = QPushButton("[Additive Interpolation]")
        btn_add.clicked.connect(self.apply_additive)
        t_layout.addWidget(btn_add)
        btn_reset = QPushButton("[Reset Region]")
        btn_reset.clicked.connect(self.reset_original)
        t_layout.addWidget(btn_reset)
        tools_group.setLayout(t_layout)
        r_layout.addWidget(tools_group)
        right_tabs.addTab(tab_edit, "Edit")

        # Tab 2: View & SMPL
        tab_view = QWidget()
        v_layout = QVBoxLayout(tab_view)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_w = QWidget()
        scroll_l = QVBoxLayout(scroll_w)
        
        smpl_g = QGroupBox("SMPL-X Reference Settings")
        smpl_l = QVBoxLayout()
        self.chk_smpl_vis = QCheckBox("Show Original Reference")
        self.chk_smpl_vis.setChecked(True)
        self.chk_smpl_vis.stateChanged.connect(lambda s: setattr(self.mujoco_widget, 'show_smplx', s==Qt.Checked) or self.mujoco_widget.update())
        smpl_l.addWidget(self.chk_smpl_vis)

        # 新增：帧率缩放控制
        scale_l = QHBoxLayout()
        scale_l.addWidget(QLabel("Frame Sync (Scale):"))
        sp_scale = QDoubleSpinBox()
        sp_scale.setRange(0.1, 10.0)
        sp_scale.setSingleStep(0.1)
        sp_scale.setValue(3.0)
        sp_scale.valueChanged.connect(lambda v: setattr(self.mujoco_widget, 'smplx_frame_scale', v) or self.mujoco_widget.update())
        scale_l.addWidget(sp_scale)
        smpl_l.addLayout(scale_l)

        
        off_l = QHBoxLayout()
        off_l.addWidget(QLabel("X Offset:"))
        sp_x = QDoubleSpinBox()
        sp_x.setRange(-5, 5)
        sp_x.setSingleStep(0.1)
        sp_x.setValue(0.0)
        sp_x.valueChanged.connect(lambda v: self.update_smpl_offset(0, v))
        off_l.addWidget(sp_x)
        off_l.addWidget(QLabel("Y Offset:"))
        sp_y = QDoubleSpinBox()
        sp_y.setRange(-5, 5)
        sp_y.setSingleStep(0.1)
        sp_y.setValue(1.0) # 默认错开
        sp_y.valueChanged.connect(lambda v: self.update_smpl_offset(1, v))
        off_l.addWidget(sp_y)
        smpl_l.addLayout(off_l)
        smpl_g.setLayout(smpl_l)
        scroll_l.addWidget(smpl_g)
        
        # 1. Labels
        g_lbl = QGroupBox("Display Labels")
        l_g_l = QVBoxLayout()
        combo_lbl = QComboBox()
        for name in self.mujoco_widget.label_options.keys(): 
            combo_lbl.addItem(name)
        combo_lbl.currentTextChanged.connect(self.mujoco_widget.set_label_mode)
        l_g_l.addWidget(combo_lbl)
        g_lbl.setLayout(l_g_l)
        scroll_l.addWidget(g_lbl)

        # 2. Frames
        g_frm = QGroupBox("Display Frames")
        f_g_l = QVBoxLayout()
        combo_frm = QComboBox()
        for name in self.mujoco_widget.frame_options.keys(): 
            combo_frm.addItem(name)
        combo_frm.currentTextChanged.connect(self.mujoco_widget.set_frame_mode)
        f_g_l.addWidget(combo_frm)
        g_frm.setLayout(f_g_l)
        scroll_l.addWidget(g_frm)

        # 3. Render Flags
        g_flags = QGroupBox("Render Flags")
        fl_l = QVBoxLayout()
        for name, (flag, val, _) in self.mujoco_widget.render_flags.items():
            cb = QCheckBox(name)
            cb.setChecked(val)
            cb.stateChanged.connect(lambda s, n=name: self.mujoco_widget.set_render_flag(n, s==Qt.Checked))
            fl_l.addWidget(cb)
        g_flags.setLayout(fl_l)
        scroll_l.addWidget(g_flags)
        
        scroll_l.addStretch()
        scroll.setWidget(scroll_w)
        v_layout.addWidget(scroll)
        right_tabs.addTab(tab_view, "View")
        
        splitter.addWidget(right_tabs)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        self.status_bar = self.statusBar()

    def load_smplx_ref(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load SMPL-X (.npz)", "", "NPZ Files (*.npz)")
        if path:
            self.status_bar.showMessage("Loading SMPL-X model, please wait...")
            QApplication.processEvents()
            
            # 使用默认配置路径作为 Model Dir
            if self.backend.load_smplx_data(path, SMPLX_BODY_MODEL_DIR):
                self.mujoco_widget.set_smplx_data(self.backend.smplx_joints)
                self.status_bar.showMessage(f"SMPL-X Loaded: {len(self.backend.smplx_joints)} frames")
                self.mujoco_widget.smplx_offset = np.array([0.0, 1.0, 0.0])
                self.mujoco_widget.update()
            else:
                QMessageBox.warning(self, "Error", "Failed to load SMPL-X. Check console for details.")

    def update_smpl_offset(self, axis, value):
        self.mujoco_widget.smplx_offset[axis] = value
        self.mujoco_widget.update()

    def toggle_play(self):
        if self.total_frames == 0: 
            return
        if self.is_playing: 
            self.timer.stop()
            self.is_playing = False
            self.btn_play.setText("[Play/Space]")
            self.btn_play.setStyleSheet("background-color: #44aa44; font-weight: bold; color: white;")
        else: 
            self.timer.start(33)
            self.is_playing = True
            self.btn_play.setText("[Pause/Space]")
            self.btn_play.setStyleSheet("background-color: #aa4444; font-weight: bold; color: white;")

    def play_next_frame(self):
        next_idx = self.current_frame + 1
        if next_idx >= self.total_frames: 
            next_idx = 0
        self.update_frame(next_idx)

    def on_selection_change(self):
        items = self.joint_list.selectedIndexes()
        if not items: 
            return
        indices = [i.row() for i in items]
        self.graph.update_curves(indices)

    def update_frame(self, idx):
        self.current_frame = idx
        self.lbl_frame.setText(f"{idx:04d}")
        self.graph.current_frame_line.setValue(idx)
        self.mujoco_widget.current_frame_idx = idx # 同步帧号给渲染器
        self.backend.set_frame(idx)
        self.mujoco_widget.update() 

    def update_frame_from_graph(self, idx):
        idx = max(0, min(self.total_frames-1, idx))
        self.current_frame = idx
        self.lbl_frame.setText(f"{idx:04d}")
        self.mujoco_widget.current_frame_idx = idx
        self.backend.set_frame(idx)
        self.mujoco_widget.update()

    def jump(self, delta):
        new = max(0, min(self.total_frames-1, self.current_frame + delta))
        self.update_frame(new)

    def apply_smooth(self):
        if self.graph.selected_joint_idx is None: 
            return
        self.backend.snapshot()
        r_min, r_max = self.graph.region.getRegion()
        s, e = int(r_min), int(r_max)
        s = max(0, s)
        e = min(self.total_frames-1, e)
        # 修改：直接使用 selected_joint_idx
        col = self.graph.selected_joint_idx 
        self.backend.df.iloc[s:e+1, col] = self.backend.df.iloc[s:e+1, col].rolling(window=5, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        self.graph.update_curves([self.graph.selected_joint_idx])
        self.backend.set_frame(self.current_frame)
        self.mujoco_widget.update()
        self.status_bar.showMessage("Smoothed")

    def apply_additive(self):
        if self.graph.selected_joint_idx is None: 
            return
        self.backend.snapshot()
        r_min, r_max = self.graph.region.getRegion()
        s, e = int(r_min), int(r_max)
        if s >= e: 
            return
        # 修改：直接使用 selected_joint_idx
        col = self.graph.selected_joint_idx
        orig, curr = self.backend.df_orig, self.backend.df
        delta_s = curr.iloc[s, col] - orig.iloc[s, col]
        delta_e = curr.iloc[e, col] - orig.iloc[e, col]
        interp = PchipInterpolator([s, e], [delta_s, delta_e])
        curr.iloc[s:e+1, col] = orig.iloc[s:e+1, col] + interp(np.arange(s, e + 1))
        self.graph.update_curves([self.graph.selected_joint_idx])
        self.backend.set_frame(self.current_frame)
        self.mujoco_widget.update()
        self.status_bar.showMessage("Interpolated")

    def reset_original(self):
        if self.graph.selected_joint_idx is None: 
            return
        self.backend.snapshot()
        r_min, r_max = self.graph.region.getRegion()
        s, e = int(r_min), int(r_max)
        # 修改：直接使用 selected_joint_idx
        col = self.graph.selected_joint_idx
        self.backend.df.iloc[s:e+1, col] = self.backend.df_orig.iloc[s:e+1, col]
        self.graph.update_curves([self.graph.selected_joint_idx])
        self.backend.set_frame(self.current_frame)
        self.mujoco_widget.update()
        self.status_bar.showMessage("Reset")

    def perform_undo(self):
        if self.backend.undo(): 
            self.refresh_ui("Undone")
            
    def perform_redo(self):
        if self.backend.redo(): 
            self.refresh_ui("Redone")
            
    def refresh_ui(self, msg):
        if self.graph.selected_joint_idx is not None: 
            self.graph.update_curves([self.graph.selected_joint_idx])
        self.backend.set_frame(self.current_frame)
        self.mujoco_widget.update()
        self.status_bar.showMessage(msg)
        
    def save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save As", "", "CSV (*.csv)")
        if path: 
            self.backend.df.to_csv(path, index=False, header=False)
            QMessageBox.information(self, "Save", "Saved successfully")

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    if 'qdarktheme' in sys.modules: 
        qdarktheme.setup_theme("dark")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
