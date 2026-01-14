import numpy as np
import mujoco
from PyQt5.QtWidgets import QOpenGLWidget, QSizePolicy
from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtGui import QMouseEvent, QWheelEvent

from src.config import RENDER_FLAGS, SMPL_PARENTS, LABEL_OPTIONS, FRAME_OPTIONS
from src.utils import rotation_matrix_from_vectors

class MuJoCoWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.data = None
        
        self.ref_joints = None 
        self.ref_parents = None
        self.ref_offset = np.array([0.0, 0.0, 0.0]) 
        self.show_ref = True # 改名 show_smplx -> show_ref
        self.current_frame_idx = 0
        self.smplx_frame_scale = 3.0
        
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption() 
        self.scn = None
        self.con = None
        self.pert = mujoco.MjvPerturb()
        self.last_mouse_pos = QPoint()
        
        # Wheel zoom optimization
        self.wheel_accumulator = 0.0
        self.wheel_timer = QTimer()
        self.wheel_timer.setSingleShot(True)
        self.wheel_timer.timeout.connect(self._apply_wheel_zoom)
        
        # === 1. 开关型标志 (Checkbox) ===
        # 格式: "显示名称": (MuJoCo枚举, 默认开启?, 是否渲染标志mjRND)
        self.render_flags = RENDER_FLAGS

        # === 2. 标签显示模式 (mjLABEL) ===
        self.label_options = LABEL_OPTIONS

        # === 3. 坐标系显示模式 (mjFRAME) ===
        self.frame_options = FRAME_OPTIONS
        
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
            # 应用配置文件中的默认渲染标志
            self._apply_default_render_flags()
        self.update()
    
    def _apply_default_render_flags(self):
        """初始化时应用配置文件中的默认渲染标志"""
        for flag_name, (mj_flag, default_val, is_rnd) in self.render_flags.items():
            if is_rnd:
                if self.scn: 
                    self.scn.flags[mj_flag] = 1 if default_val else 0
            else:
                self.opt.flags[mj_flag] = 1 if default_val else 0

    def set_ref_data(self, joints, parents):
        """通用参考数据设置入口"""
        self.ref_joints = joints
        self.ref_parents = parents
        
        # 自动对齐 (通用)
        if len(joints) > 0:
            first_frame = joints[0]
            if hasattr(first_frame, 'cpu'): first_frame = first_frame.detach().cpu().numpy()
            center = first_frame[0] # 假设第0个关节是 Root
            self.ref_offset = -center + np.array([0, 1.0, 1.0]) 
            print(f"[Ref] Auto Offset: {self.ref_offset}")
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
        if mode_name in self.label_options: self.opt.label = self.label_options[mode_name]; self.update()
    def set_frame_mode(self, mode_name):
        if mode_name in self.frame_options: self.opt.frame = self.frame_options[mode_name]; self.update()

    def initializeGL(self):
        if self.model and not self.con:
            try: self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            except Exception as e: print(e)

    def paintGL(self):
        if not self.model or not self.data or not self.scn or not self.con: return
        
        viewport = mujoco.MjrRect(0, 0, self.width(), self.height())
        
        # 1. 更新 MuJoCo 场景
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, self.pert, 
            self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn
        )
        
        # 2. 绘制调试基准点 (绿色大球)
        self.draw_debug_marker()

        # 绘制参考 (Generic)
        if self.show_ref and self.ref_joints is not None:
            self.add_ref_to_scene()

        # 4. 提交渲染
        mujoco.mjr_render(viewport, self.scn, self.con)

    def draw_debug_marker(self):
        """原点绘制绿球"""
        if self.scn.ngeom >= self.scn.maxgeom: return
        mujoco.mjv_initGeom(
            self.scn.geoms[self.scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0],
            pos=[0, 0, 0],
            mat=np.eye(3).flatten(),
            rgba=[0, 1, 0, 1] # 纯绿
        )
        self.scn.ngeom += 1

    def add_ref_to_scene(self):
        """通用骨架绘制函数 (支持 SMPL 和 BVH)"""
        num_frames = len(self.ref_joints)
        if num_frames == 0 or self.ref_parents is None: return
        
        # frame_idx = min(self.current_frame_idx, num_frames - 1) # 简单钳位
        # 如果需要帧率缩放:
        frame_idx = min(int(self.current_frame_idx * self.smplx_frame_scale), num_frames-1)

        joints = self.ref_joints[frame_idx]
        if hasattr(joints, 'cpu'): joints = joints.detach().cpu().numpy()
        
        joints = joints + self.ref_offset
        
        # 1. 画关节
        for j_pos in joints: 
            if self.scn.ngeom >= self.scn.maxgeom: break
            mujoco.mjv_initGeom(
                self.scn.geoms[self.scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.03, 0, 0], 
                pos=j_pos,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 1] 
            )
            self.scn.ngeom += 1
            
        # 2. 画连线 (基于传入的 parents 列表)
        for i, parent_idx in enumerate(self.ref_parents):
            if parent_idx == -1: continue # Root 没有父节点
            if self.scn.ngeom >= self.scn.maxgeom: break
            
            p1 = joints[parent_idx]
            p2 = joints[i]
            
            mujoco.mjv_connector(
                self.scn.geoms[self.scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                width=0.01, # BVH 骨架稍微细一点
                from_=p1,
                to=p2
            )
            self.scn.geoms[self.scn.ngeom].rgba = np.array([1, 1, 1, 0.5])
            self.scn.ngeom += 1

    # 鼠标交互
    def mousePressEvent(self, event: QMouseEvent): self.last_mouse_pos = event.pos(); event.accept()
    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.model: return
        dx = event.x()-self.last_mouse_pos.x(); dy = event.y()-self.last_mouse_pos.y()
        self.last_mouse_pos = event.pos(); width, height = self.width(), self.height()
        if event.buttons() & Qt.LeftButton:
            if dx!=0: mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_H, dx/height, 0, self.scn, self.cam)
            if dy!=0: mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, 0, dy/height, self.scn, self.cam)
        elif event.buttons() & Qt.RightButton:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_V, dx/height, dy/height, self.scn, self.cam)
        elif event.buttons() & Qt.MidButton:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, dx/height, dy/height, self.scn, self.cam)
        self.update(); event.accept()
    def wheelEvent(self, event: QWheelEvent):
        """优化的滚轮缩放：累积事件并批量处理"""
        dy = event.angleDelta().y()
        if abs(dy) < 1:
            return
        
        # 累积滚轮增量
        self.wheel_accumulator += dy
        
        # 立即应用缩放（不等待定时器）以获得即时反馈
        if not self.wheel_timer.isActive():
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -dy*0.002, self.scn, self.cam)
            self.update()
        
        # 重置定时器，延迟最后的更新
        self.wheel_timer.stop()
        self.wheel_timer.start(16)  # ~60fps刷新率
        event.accept()
    
    def _apply_wheel_zoom(self):
        """应用累积的滚轮缩放（定时器回调）"""
        if abs(self.wheel_accumulator) > 0:
            # 清空累积器
            self.wheel_accumulator = 0.0
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self.data: self.cam.lookat = np.array(self.data.qpos[0:3]); self.cam.distance = 3.0; self.update()