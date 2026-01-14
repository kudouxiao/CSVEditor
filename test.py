import sys
import threading
import time
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
from scipy.interpolate import PchipInterpolator, CubicSpline
# 引入信号处理库
from scipy.signal import savgol_filter

# UI 库
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QListWidget, 
                             QSplitter, QFileDialog, QSpinBox, QGroupBox, 
                             QMessageBox, QAbstractItemView, QFrame, QShortcut,
                             QTabWidget, QCheckBox, QComboBox)
from PyQt5.QtCore import Qt, QObject, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence

# 绘图库
import pyqtgraph as pg

# 主题
try:
    import qdarktheme
except ImportError:
    pass

# ============ 配置区 ============
DEFAULT_CSV_PATH = "/home/jq/dance/GMR/retargeted/38_3.csv"
DEFAULT_MODEL_PATH = "/home/jq/dance/GMR/assets/unitree_g1/g1_mocap_29dof.xml"

# ============ 后端逻辑 ============
class G1Backend(QObject):
    def __init__(self):
        super().__init__()
        self.df = None
        self.df_orig = None # 原始数据，用于Ghosting和重置
        self.model = None
        self.data = None
        self.viewer_running = False
        self.lock = threading.Lock()
        self.joint_mapping = {}
        self.modified_frames = set()
        
        # 撤销/重做栈
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 50
        
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

    def load_data(self, csv_path, model_path):
        try:
            self.df = pd.read_csv(csv_path, header=None)
            if isinstance(self.df.iloc[0, 0], str): self.df = pd.read_csv(csv_path)
            self.df_orig = self.df.copy() # 保存副本
            
            self.undo_stack.clear()
            self.redo_stack.clear()
            
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
            self.model.opt.disableflags = 65535 
            self.data = mujoco.MjData(self.model)
            
            self.joint_mapping = {}
            model_j_names = {}
            q_ptr = 0
            for i in range(self.model.njnt):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if self.model.jnt_type[i] == 0: q_ptr += 7; continue
                model_j_names[name] = q_ptr; q_ptr += 1
            for idx, name in enumerate(self.csv_joint_names):
                simple = name.replace("_joint", "")
                for m_n, addr in model_j_names.items():
                    if simple in m_n or m_n in simple: self.joint_mapping[idx] = addr; break
            
            if not self.viewer_running:
                threading.Thread(target=self._viewer_thread, daemon=True).start()
            return True, len(self.df)
        except Exception as e:
            print(f"Load Error: {e}")
            return False, 0

    def _viewer_thread(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as v:
            self.viewer_running = True
            while v.is_running() and self.viewer_running:
                with self.lock: v.sync()
                time.sleep(0.01)

    def set_frame(self, idx):
        if self.df is None: return
        with self.lock:
            line = pd.to_numeric(self.df.iloc[idx].values, errors='coerce')
            line = np.nan_to_num(line.astype(float))
            p = line[0:3]; self.data.qpos[0:3] = p * 0.001 if np.any(np.abs(p) > 50) else p
            q = line[3:7]; wxyz = np.array([q[3], q[0], q[1], q[2]])
            n = np.linalg.norm(wxyz); self.data.qpos[3:7] = wxyz/n if n > 1e-4 else [1,0,0,0]
            for c_idx, m_idx in self.joint_mapping.items():
                col = 7 + c_idx
                if col < len(line): self.data.qpos[m_idx] = line[col]
            mujoco.mj_forward(self.model, self.data)
            
    def snapshot(self):
        if self.df is None: return
        self.undo_stack.append(self.df.copy())
        self.redo_stack.clear()
        if len(self.undo_stack) > self.max_history: self.undo_stack.pop(0)
            
    def undo(self):
        if not self.undo_stack: return False
        self.redo_stack.append(self.df)
        self.df = self.undo_stack.pop()
        return True
        
    def redo(self):
        if not self.redo_stack: return False
        self.undo_stack.append(self.df)
        self.df = self.redo_stack.pop()
        return True

# ============ 曲线编辑器 ============
class CurveEditor(pg.PlotWidget):
    """
    CurveEditor v2.1:
    - 基础绘制
    - 幽灵帧 (Ghosting)
    - 交互式平滑拖拽
    - 锚点样条编辑 (Spline Points) + 自动切线约束(Auto Tangent Clamp)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('#1e1e1e')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setMouseEnabled(x=True, y=False)
        self.getPlotItem().setLabel('bottom', 'Frame Index')
        self.getPlotItem().setLabel('left', 'Angle (Rad)')
        
        # 1. 范围选择器 (Layer 10)
        self.region = pg.LinearRegionItem([0, 100], brush=(50, 50, 200, 50))
        self.region.setZValue(10) 
        self.addItem(self.region)
        
        # 2. 当前帧指示线 (Layer 100)
        self.current_frame_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#FF5555', width=2), movable=True)
        self.current_frame_line.setZValue(100) 
        self.addItem(self.current_frame_line)
        
        # 3. 幽灵曲线 (Layer 0) - 显示原始数据
        self.ghost_curve = self.plot([], pen=pg.mkPen((100, 100, 100), width=1, style=Qt.DashLine))
        self.ghost_curve.setZValue(0)
        self.show_ghost = False
        
        # 4. 样条编辑锚点 (Layer 200) - 用于 Keyframe 编辑模式
        self.scatter_item = pg.ScatterPlotItem(size=15, pen=pg.mkPen('w'), brush=pg.mkBrush('#FFCC00'))
        self.scatter_item.setZValue(200)
        self.addItem(self.scatter_item)
        self.spline_preview_curve = self.plot([], pen=pg.mkPen('#FFCC00', width=2, style=Qt.DashLine)) # 预览线
        self.spline_preview_curve.setZValue(190)

        self.curves = {} 
        self.selected_joint_idx = None
        self.backend_ref = None
        self.main_window_ref = None
        
        # 状态标志
        self.is_editing = False 
        self.drag_start_pos = None
        self.drag_start_data = None 
        
        # 样条编辑模式状态
        self.spline_mode_active = False
        self.spline_anchors_x = []
        self.spline_anchors_y = []
        self.dragged_anchor_index = None
        
        # === 新增：用于切线约束的边界值 ===
        self.spline_boundary_slopes = None # (slope_in, slope_out)

    def set_backend(self, backend, main_window):
        self.backend_ref = backend
        self.main_window_ref = main_window
        self.current_frame_line.sigDragged.connect(self.on_line_dragged)
        self.region.sigRegionChanged.connect(main_window.on_region_changed)

    def set_ghost_visible(self, visible):
        self.show_ghost = visible
        self.update_curves([self.selected_joint_idx] if self.selected_joint_idx is not None else [])

    def update_curves(self, selected_indices):
        # 清除旧曲线
        for item in self.curves.values():
            self.removeItem(item)
        self.curves.clear()
        
        if not self.backend_ref or self.backend_ref.df is None: return
        
        if len(selected_indices) > 0:
            self.selected_joint_idx = selected_indices[0]
            
            # 1. 绘制幽灵帧 (原始数据)
            if self.show_ghost:
                col = 7 + self.selected_joint_idx
                orig_data = self.backend_ref.df_orig.iloc[:, col].values
                self.ghost_curve.setData(orig_data)
            else:
                self.ghost_curve.setData([])

            # 2. 绘制其他未选中的关节 (背景参考)
            for idx in selected_indices[1:]:
                col = 7 + idx
                data = self.backend_ref.df.iloc[:, col].values
                curve = self.plot(data, pen=pg.mkPen((80, 80, 80), width=1))
                curve.setZValue(5)
                self.curves[idx] = curve
            
            # 3. 绘制当前选中的关节 (高亮)
            col = 7 + self.selected_joint_idx
            data = self.backend_ref.df.iloc[:, col].values
            main_curve = self.plot(data, pen=pg.mkPen('#00ffff', width=3))
            main_curve.setZValue(20)
            self.curves[self.selected_joint_idx] = main_curve
            
            self.autoRange()
        else:
            self.selected_joint_idx = None
            self.ghost_curve.setData([])

    def on_line_dragged(self):
        idx = int(self.current_frame_line.value())
        if self.main_window_ref:
            self.main_window_ref.update_frame_from_graph(idx)

    # ============= 样条编辑模式 (Spline Mode) 逻辑 =============
    def start_spline_mode(self, start_frame, end_frame, num_anchors=5):
        """进入样条编辑模式：在区间内生成锚点，并计算边界切线"""
        if self.selected_joint_idx is None: return
        self.spline_mode_active = True
        self.region.setMovable(False)
        
        col = 7 + self.selected_joint_idx
        
        # === 计算边界切线 (Tangents) 用于平滑连接 ===
        # 斜率 = (y_curr - y_prev) / 1
        slope_in = 0.0
        slope_out = 0.0
        
        # 获取 Start 前一帧的斜率
        if start_frame > 0:
            y_s = self.backend_ref.df.iloc[start_frame, col]
            y_prev = self.backend_ref.df.iloc[start_frame - 1, col]
            slope_in = y_s - y_prev
        
        # 获取 End 后一帧的斜率
        total_len = len(self.backend_ref.df)
        if end_frame < total_len - 1:
            y_e = self.backend_ref.df.iloc[end_frame, col]
            y_next = self.backend_ref.df.iloc[end_frame + 1, col]
            slope_out = y_next - y_e
            
        self.spline_boundary_slopes = (slope_in, slope_out)
        # ==========================================
        
        # 生成等间距的锚点 X 坐标
        self.spline_anchors_x = np.linspace(start_frame, end_frame, num_anchors).astype(int)
        # 对应的 Y 坐标 (从当前曲线取值)
        self.spline_anchors_y = self.backend_ref.df.iloc[self.spline_anchors_x, col].values
        
        self.update_spline_visuals()
        
    def update_spline_visuals(self):
        """更新锚点显示和预览曲线 (包含切线约束)"""
        # 1. 更新锚点位置
        spots = [{'pos': (x, y), 'data': i} for i, (x, y) in enumerate(zip(self.spline_anchors_x, self.spline_anchors_y))]
        self.scatter_item.setData(spots=spots)
        
        # 2. 生成样条插值预览
        if len(self.spline_anchors_x) >= 2:
            try:
                # === 关键修改：使用 CubicSpline 并施加边界条件 ===
                # bc_type=((1, slope_in), (1, slope_out)) 表示一阶导数（斜率）固定
                if self.spline_boundary_slopes:
                    cs = CubicSpline(self.spline_anchors_x, self.spline_anchors_y, 
                                     bc_type=((1, self.spline_boundary_slopes[0]), 
                                              (1, self.spline_boundary_slopes[1])))
                else:
                    # 备用（虽然通常不会走到这）
                    cs = PchipInterpolator(self.spline_anchors_x, self.spline_anchors_y)
                
                x_new = np.arange(self.spline_anchors_x[0], self.spline_anchors_x[-1] + 1)
                y_new = cs(x_new)
                self.spline_preview_curve.setData(x_new, y_new)
            except Exception as e:
                print(f"Spline Error: {e}")
                pass

    def apply_spline_to_data(self):
        """将预览曲线应用到真实数据"""
        if not self.spline_mode_active or self.selected_joint_idx is None: return
        
        self.backend_ref.snapshot() # 存档
        
        x_data, y_data = self.spline_preview_curve.getData()
        if x_data is None: return
        
        col = 7 + self.selected_joint_idx
        start = int(x_data[0])
        end = int(x_data[-1])
        
        # 写入
        self.backend_ref.df.iloc[start:end+1, col] = y_data
        self.backend_ref.modified_frames.update(range(start, end+1))
        
        # 退出模式
        self.cancel_spline_mode()
        # 刷新主曲线
        self.update_curves([self.selected_joint_idx])

    def cancel_spline_mode(self):
        """退出样条模式"""
        self.spline_mode_active = False
        self.spline_boundary_slopes = None
        self.region.setMovable(True)
        self.scatter_item.clear()
        self.spline_preview_curve.setData([])
        self.dragged_anchor_index = None

    # ============= 鼠标事件分发 =============
    def mousePressEvent(self, ev):
        if self.spline_mode_active:
            # === 样条模式下的点击 ===
            if ev.button() == Qt.LeftButton:
                pos = self.plotItem.vb.mapSceneToView(ev.pos())
                # 检测是否点击了某个锚点
                idx = -1
                click_x = pos.x()
                click_y = pos.y()
                x_range, y_range = self.viewRange()
                x_tol = (x_range[1] - x_range[0]) * 0.02
                y_tol = (y_range[1] - y_range[0]) * 0.05
                
                for i, (ax, ay) in enumerate(zip(self.spline_anchors_x, self.spline_anchors_y)):
                    if abs(ax - click_x) < x_tol and abs(ay - click_y) < y_tol:
                        idx = i
                        break
                
                if idx != -1:
                    self.dragged_anchor_index = idx
                    ev.accept()
                    return
        else:
            # === 普通模式 (原有逻辑) ===
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
                s = max(0, s); e = min(len(self.backend_ref.df)-1, e)
                col = 7 + self.selected_joint_idx
                self.drag_start_data = self.backend_ref.df.iloc[s:e+1, col].values.copy()
                self.region.setMovable(False)
                return

        if not self.spline_mode_active:
            self.region.setMovable(True) 
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        pos = self.plotItem.vb.mapSceneToView(ev.pos())

        if self.spline_mode_active and self.dragged_anchor_index is not None:
            # === 拖拽锚点 ===
            new_y = pos.y()
            self.spline_anchors_y[self.dragged_anchor_index] = new_y
            self.update_spline_visuals()
            ev.accept()
            return

        if self.is_editing:
            # === 普通整体拖拽 (Sine Wave) ===
            if not (ev.buttons() & Qt.LeftButton):
                self.is_editing = False
                self.drag_start_data = None
                self.region.setMovable(True)
                super().mouseMoveEvent(ev)
                return

            ev.accept()
            y_curr = pos.y()
            delta_y = y_curr - self.drag_start_pos
            
            r_min, r_max = self.region.getRegion()
            s, e = int(r_min), int(r_max)
            s = max(0, s); e = min(len(self.backend_ref.df)-1, e)
            
            if s < e and self.drag_start_data is not None:
                length = e - s + 1
                x = np.linspace(-np.pi, np.pi, length)
                weights = (np.cos(x) + 1) / 2 
                new_values = self.drag_start_data + (delta_y * weights)
                
                col = 7 + self.selected_joint_idx
                self.backend_ref.df.iloc[s:e+1, col] = new_values
                self.backend_ref.modified_frames.update(range(s, e+1))
                
                self.curves[self.selected_joint_idx].setData(self.backend_ref.df.iloc[:, col].values)
                
                curr_f = int(self.current_frame_line.value())
                if s <= curr_f <= e:
                    self.backend_ref.set_frame(curr_f)
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self.spline_mode_active:
            self.dragged_anchor_index = None
            ev.accept()
        elif self.is_editing:
            self.is_editing = False
            self.drag_start_data = None
            self.region.setMovable(True)
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)


# ============ 主窗口 ============
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("G1 Pose Editor v2.1 (Tangent Clamp)")
        self.resize(1600, 950)
        
        self.backend = G1Backend()
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False 
        self.updating_region_from_spin = False 
        
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next_frame)
        
        # 自动加载
        import os
        if os.path.exists(DEFAULT_CSV_PATH) and os.path.exists(DEFAULT_MODEL_PATH):
            success, frames = self.backend.load_data(DEFAULT_CSV_PATH, DEFAULT_MODEL_PATH)
            if success:
                self.total_frames = frames
                self.graph.setXRange(0, frames)
                self.graph.region.setRegion([0, frames//5])
                self.spin_start.setMaximum(frames - 1)
                self.spin_end.setMaximum(frames - 1)
                self.update_frame(0)
                self.status_bar.showMessage(f"Loaded {frames} frames.")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # === 1. 顶部工具栏 (Global Toolbar) ===
        top_bar = QHBoxLayout()
        
        self.btn_undo = QPushButton("↩ 撤销"); self.btn_undo.clicked.connect(self.perform_undo)
        self.btn_redo = QPushButton("↪ 重做"); self.btn_redo.clicked.connect(self.perform_redo)
        
        # Ghosting 开关
        self.chk_ghost = QCheckBox("👻 显示原图 (Ghost)"); 
        self.chk_ghost.setStyleSheet("color: #aaaaaa; font-weight: bold;")
        self.chk_ghost.stateChanged.connect(self.toggle_ghost)

        btn_save = QPushButton("💾 另存为"); btn_save.clicked.connect(self.save_as)
        
        top_bar.addWidget(self.btn_undo)
        top_bar.addWidget(self.btn_redo)
        top_bar.addSpacing(20)
        top_bar.addWidget(self.chk_ghost) # Add here
        top_bar.addStretch()
        top_bar.addWidget(btn_save)
        
        layout.addLayout(top_bar)
        
        # === 2. 主体分割视图 ===
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # 左侧: 曲线编辑器 + 播放条
        left_container = QWidget()
        l_layout = QVBoxLayout(left_container)
        l_layout.setContentsMargins(0,0,0,0)
        
        self.graph = CurveEditor()
        l_layout.addWidget(self.graph)
        
        play_ctrl = QHBoxLayout()
        self.btn_prev = QPushButton("◀"); self.btn_prev.clicked.connect(lambda: self.jump(-1))
        self.btn_play = QPushButton("▶ 播放 (Space)")
        self.btn_play.setStyleSheet("background-color: #44aa44; font-weight: bold; color: white;")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_next = QPushButton("▶"); self.btn_next.clicked.connect(lambda: self.jump(1))
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
        
        # 右侧: 控制面板 (Tabbed)
        right_container = QFrame()
        right_container.setFrameShape(QFrame.StyledPanel)
        r_layout = QVBoxLayout(right_container)
        
        # 关节列表
        lbl_list = QLabel("关节列表")
        r_layout.addWidget(lbl_list)
        self.joint_list = QListWidget()
        self.joint_list.setSelectionMode(QAbstractItemView.SingleSelection)
        for i, name in enumerate(self.backend.csv_joint_names):
            self.joint_list.addItem(f"[{i:02d}] {name.replace('_joint','')}")
        self.joint_list.itemSelectionChanged.connect(self.on_selection_change)
        r_layout.addWidget(self.joint_list)

        # 区间选择器 (Global SpinBoxes)
        range_group = QGroupBox("选中范围")
        range_layout = QHBoxLayout()
        self.spin_start = QSpinBox(); self.spin_start.setRange(0, 99999)
        self.spin_start.valueChanged.connect(self.on_spinbox_changed)
        self.spin_end = QSpinBox(); self.spin_end.setRange(0, 99999)
        self.spin_end.valueChanged.connect(self.on_spinbox_changed)
        range_layout.addWidget(QLabel("Start:"))
        range_layout.addWidget(self.spin_start)
        range_layout.addWidget(QLabel("End:"))
        range_layout.addWidget(self.spin_end)
        range_group.setLayout(range_layout)
        r_layout.addWidget(range_group)

        # 工具 Tab 页
        self.tabs = QTabWidget()
        
        # Tab 1: 基础工具 (连接 & 平滑)
        tab_basic = QWidget()
        tb_layout = QVBoxLayout(tab_basic)
        
        btn_linear = QPushButton("📏 直线连接 (Linear)")
        btn_linear.setStyleSheet("background-color: #d68a00; color: white;")
        btn_linear.clicked.connect(lambda: self.apply_connect("linear"))
        
        btn_sigmoid = QPushButton("🌊 S形连接 (S-Curve)")
        btn_sigmoid.setStyleSheet("background-color: #d66a00; color: white;")
        btn_sigmoid.setToolTip("使用 Cosine 曲线连接，首尾速度更柔顺")
        btn_sigmoid.clicked.connect(lambda: self.apply_connect("sigmoid"))
        
        btn_smooth = QPushButton("💧 选区平滑 (Sav-Gol)")
        btn_smooth.clicked.connect(self.apply_smooth)
        
        tb_layout.addWidget(btn_linear)
        tb_layout.addWidget(btn_sigmoid)
        tb_layout.addSpacing(10)
        tb_layout.addWidget(btn_smooth)
        tb_layout.addStretch()
        
        # Tab 2: 高级工具 (样条编辑)
        tab_adv = QWidget()
        ta_layout = QVBoxLayout(tab_adv)
        
        # === 新增：锚点数量控制 ===
        h_anchor = QHBoxLayout()
        h_anchor.addWidget(QLabel("锚点数量 (Anchor Count):"))
        self.spin_anchor_count = QSpinBox()
        self.spin_anchor_count.setRange(3, 50)
        self.spin_anchor_count.setValue(5)
        h_anchor.addWidget(self.spin_anchor_count)
        ta_layout.addLayout(h_anchor)
        # ==========================
        
        lbl_info = QLabel("高级锚点编辑模式:\n1. 点击 '开始编辑'\n2. 拖动出现的黄色锚点\n3. 系统会自动对齐头尾切线，保证连接平滑。\n4. 点击 '应用' 保存")
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color: gray; font-size: 11px;")
        
        self.btn_spline_start = QPushButton("✏️ 开始锚点编辑")
        self.btn_spline_start.setStyleSheet("background-color: #880088; color: white;")
        self.btn_spline_start.clicked.connect(self.toggle_spline_mode)
        
        self.btn_spline_apply = QPushButton("✅ 应用 (Apply)")
        self.btn_spline_apply.setEnabled(False)
        self.btn_spline_apply.clicked.connect(self.apply_spline)
        
        self.btn_spline_cancel = QPushButton("❌ 取消 (Cancel)")
        self.btn_spline_cancel.setEnabled(False)
        self.btn_spline_cancel.clicked.connect(self.cancel_spline)
        
        ta_layout.addWidget(lbl_info)
        ta_layout.addWidget(self.btn_spline_start)
        ta_layout.addWidget(self.btn_spline_apply)
        ta_layout.addWidget(self.btn_spline_cancel)
        ta_layout.addStretch()

        # Tab 3: 批量 & 修正
        tab_batch = QWidget()
        tc_layout = QVBoxLayout(tab_batch)
        btn_add = QPushButton("✨ 叠加插值 (Additive)")
        btn_add.clicked.connect(self.apply_additive)
        btn_reset = QPushButton("🔄 重置为原始数据")
        btn_reset.clicked.connect(self.reset_original)
        tc_layout.addWidget(btn_add)
        tc_layout.addWidget(btn_reset)
        tc_layout.addStretch()
        
        self.tabs.addTab(tab_basic, "基础")
        self.tabs.addTab(tab_adv, "锚点/样条")
        self.tabs.addTab(tab_batch, "工具")
        
        r_layout.addWidget(self.tabs)
        
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        
        self.status_bar = self.statusBar()
        
        # 快捷键
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.perform_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self.perform_redo)
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.toggle_play)

        # 延迟绑定
        self.graph.set_backend(self.backend, self)

    # === 逻辑实现 ===
    
    def toggle_ghost(self, state):
        self.graph.set_ghost_visible(state == Qt.Checked)

    def on_region_changed(self):
        if self.updating_region_from_spin: return
        r_min, r_max = self.graph.region.getRegion()
        s, e = int(r_min), int(r_max)
        self.spin_start.blockSignals(True)
        self.spin_end.blockSignals(True)
        self.spin_start.setValue(s)
        self.spin_end.setValue(e)
        self.spin_start.blockSignals(False)
        self.spin_end.blockSignals(False)

    def on_spinbox_changed(self):
        s = self.spin_start.value()
        e = self.spin_end.value()
        if s > e: return 
        self.updating_region_from_spin = True
        self.graph.region.setRegion([s, e])
        self.updating_region_from_spin = False

    # --- 样条/锚点编辑 ---
    def toggle_spline_mode(self):
        if self.graph.selected_joint_idx is None:
            QMessageBox.warning(self, "提示", "请先选择关节")
            return
            
        s = self.spin_start.value()
        e = self.spin_end.value()
        
        # 读取用户自定义的锚点数量
        n_anchors = self.spin_anchor_count.value()
        
        if e - s < n_anchors:
            QMessageBox.warning(self, "提示", f"选中区间太短，无法生成 {n_anchors} 个锚点")
            return

        self.btn_spline_start.setEnabled(False)
        self.btn_spline_apply.setEnabled(True)
        self.btn_spline_cancel.setEnabled(True)
        self.spin_anchor_count.setEnabled(False) # 编辑中禁止修改数量
        self.tabs.setCurrentIndex(1) 
        
        self.graph.start_spline_mode(s, e, num_anchors=n_anchors)
        self.status_bar.showMessage(f"进入锚点编辑模式 ({n_anchors} Points)")

    def apply_spline(self):
        self.graph.apply_spline_to_data()
        self.reset_spline_ui()
        self.status_bar.showMessage("样条修改已应用")

    def cancel_spline(self):
        self.graph.cancel_spline_mode()
        self.reset_spline_ui()
        self.status_bar.showMessage("已取消")

    def reset_spline_ui(self):
        self.btn_spline_start.setEnabled(True)
        self.btn_spline_apply.setEnabled(False)
        self.btn_spline_cancel.setEnabled(False)
        self.spin_anchor_count.setEnabled(True)

    # --- 连接功能 ---
    def apply_connect(self, mode="linear"):
        if self.graph.selected_joint_idx is None: return
        s = self.spin_start.value()
        e = self.spin_end.value()
        if s >= e or e >= self.total_frames: return

        self.backend.snapshot()
        col = 7 + self.graph.selected_joint_idx
        
        val_start = self.backend.df.iloc[s, col]
        val_end = self.backend.df.iloc[e, col]
        count = e - s + 1
        
        if mode == "linear":
            new_values = np.linspace(val_start, val_end, count)
        elif mode == "sigmoid":
            # Cosine interpolation: 0 -> 1
            t = np.linspace(0, 1, count)
            # weights: 0 at start, 1 at end, smooth transition
            weights = (1 - np.cos(t * np.pi)) / 2
            new_values = val_start + (val_end - val_start) * weights
            
        self.backend.df.iloc[s:e+1, col] = new_values
        self.backend.modified_frames.update(range(s, e+1))
        
        self.graph.update_curves([self.graph.selected_joint_idx])
        self.backend.set_frame(self.current_frame)
        self.status_bar.showMessage(f"已应用 {mode} 连接")

    # --- 其他原有功能 ---
    def apply_smooth(self):
        if self.graph.selected_joint_idx is None: return
        self.backend.snapshot()
        s, e = self.spin_start.value(), self.spin_end.value()
        if s >= e: return
        col = 7 + self.graph.selected_joint_idx
        data_chunk = self.backend.df.iloc[s:e+1, col].values
        window_len = min(len(data_chunk), 31)
        if window_len % 2 == 0: window_len -= 1
        if window_len >= 3:
            smoothed = savgol_filter(data_chunk, window_len, 3)
            self.backend.df.iloc[s:e+1, col] = smoothed
            self.backend.modified_frames.update(range(s, e+1))
            self.graph.update_curves([self.graph.selected_joint_idx])
            self.backend.set_frame(self.current_frame)
            self.status_bar.showMessage("已平滑")

    def apply_additive(self):
        if self.graph.selected_joint_idx is None: return
        self.backend.snapshot()
        s, e = self.spin_start.value(), self.spin_end.value()
        col = 7 + self.graph.selected_joint_idx
        orig, curr = self.backend.df_orig, self.backend.df
        delta_s = curr.iloc[s, col] - orig.iloc[s, col]
        delta_e = curr.iloc[e, col] - orig.iloc[e, col]
        interp = PchipInterpolator([s, e], [delta_s, delta_e])
        curr.iloc[s:e+1, col] = orig.iloc[s:e+1, col] + interp(np.arange(s, e + 1))
        self.backend.modified_frames.update(range(s, e+1))
        self.graph.update_curves([self.graph.selected_joint_idx])
        self.status_bar.showMessage("已插值")

    def reset_original(self):
        if self.graph.selected_joint_idx is None: return
        self.backend.snapshot()
        s, e = self.spin_start.value(), self.spin_end.value()
        col = 7 + self.graph.selected_joint_idx
        self.backend.df.iloc[s:e+1, col] = self.backend.df_orig.iloc[s:e+1, col]
        self.backend.modified_frames.update(range(s, e+1))
        self.graph.update_curves([self.graph.selected_joint_idx])
        self.status_bar.showMessage("已重置")

    # --- 基础架构 ---
    def toggle_play(self):
        if self.total_frames == 0: return
        if self.is_playing:
            self.timer.stop(); self.is_playing = False
            self.btn_play.setText("▶ 播放 (Space)"); self.btn_play.setStyleSheet("background-color: #44aa44; font-weight: bold; color: white;")
        else:
            self.timer.start(33); self.is_playing = True
            self.btn_play.setText("⏸ 暂停 (Space)"); self.btn_play.setStyleSheet("background-color: #aa4444; font-weight: bold; color: white;")

    def play_next_frame(self):
        next_idx = self.current_frame + 1
        if next_idx >= self.total_frames: next_idx = 0
        self.update_frame(next_idx)

    def on_selection_change(self):
        items = self.joint_list.selectedIndexes()
        if not items: return
        indices = [i.row() for i in items]
        self.graph.update_curves(indices)

    def update_frame(self, idx):
        self.current_frame = idx
        self.lbl_frame.setText(f"{idx:04d}")
        self.graph.current_frame_line.setValue(idx)
        self.backend.set_frame(idx)

    def update_frame_from_graph(self, idx):
        idx = max(0, min(self.total_frames-1, idx))
        self.current_frame = idx
        self.lbl_frame.setText(f"{idx:04d}")
        self.backend.set_frame(idx)

    def jump(self, delta):
        self.update_frame(max(0, min(self.total_frames-1, self.current_frame + delta)))

    def perform_undo(self):
        if self.backend.undo(): 
            self.graph.update_curves([self.graph.selected_joint_idx] if self.graph.selected_joint_idx is not None else [])
            self.backend.set_frame(self.current_frame)
            self.status_bar.showMessage("已撤销")

    def perform_redo(self):
        if self.backend.redo():
            self.graph.update_curves([self.graph.selected_joint_idx] if self.graph.selected_joint_idx is not None else [])
            self.backend.set_frame(self.current_frame)
            self.status_bar.showMessage("已重做")

    def save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "另存为", "", "CSV (*.csv)")
        if path:
            self.backend.df.to_csv(path, index=False, header=False)
            QMessageBox.information(self, "保存", "保存成功")

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    if 'qdarktheme' in sys.modules: qdarktheme.setup_theme("dark")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())