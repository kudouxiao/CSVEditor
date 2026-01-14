import sys
import threading
import time
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
from scipy.interpolate import PchipInterpolator

# UI 库
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QListWidget, QListWidgetItem, QSplitter, QFileDialog, 
                             QSpinBox, QGroupBox, QMessageBox, QAbstractItemView, QFrame,
                             QShortcut)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QSize, QTimer
from PyQt5.QtGui import QKeySequence

# 绘图库
import pyqtgraph as pg

# 主题
try:
    import qdarktheme
except ImportError:
    pass

# ============ 配置区 ============
DEFAULT_CSV_PATH = "/home/jq/project/GMR/retargeted/left71s.csv"
DEFAULT_MODEL_PATH = "/home/jq/project/GMR/assets/unitree_g1/g1_mocap_29dof.xml"

# ============ 后端逻辑 ============
class G1Backend(QObject):
    def __init__(self):
        super().__init__()
        self.df = None
        self.df_orig = None
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
            self.df_orig = self.df.copy()
            
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
    专业级曲线编辑器 (红线版 + 交互修复)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('#1e1e1e')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setMouseEnabled(x=True, y=False)
        self.getPlotItem().setLabel('bottom', 'Frame Index')
        self.getPlotItem().setLabel('left', 'Angle (Rad)')
        
        # 1. 范围选择器 (Layer 10) - 蓝色区域
        self.region = pg.LinearRegionItem([0, 100], brush=(50, 50, 200, 50))
        self.region.setZValue(10) 
        self.addItem(self.region)
        
        # 2. 当前帧指示线 (Layer 100) - 【改为亮红色】
        # pen='#FF5555' 是明亮的红色，在深色背景和蓝色选区上都很显眼
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
        
        if not self.backend_ref or self.backend_ref.df is None: return
        
        if len(selected_indices) > 0:
            self.selected_joint_idx = selected_indices[0]
            for idx in selected_indices[1:]:
                col = 7 + idx
                data = self.backend_ref.df.iloc[:, col].values
                curve = self.plot(data, pen=pg.mkPen((80, 80, 80), width=1))
                curve.setZValue(5)
                self.curves[idx] = curve
            
            col = 7 + self.selected_joint_idx
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

    # === 交互事件 (防粘连) ===
    def mousePressEvent(self, ev):
        if (ev.modifiers() & Qt.ControlModifier) and self.selected_joint_idx is not None and ev.button() == Qt.LeftButton:
            self.is_editing = True
            ev.accept()
            self.backend_ref.snapshot() # 快照
            
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
        if self.is_editing:
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
        self.setWindowTitle("G1 Pose Editor Pro (Play/Pause & Red Line)")
        self.resize(1600, 950)
        
        self.backend = G1Backend()
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False # 播放状态
        
        self.init_ui()
        
        # 初始化播放定时器
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
                self.update_frame(0)
                self.status_bar.showMessage(f"Loaded {frames} frames.")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # === 顶部工具栏 ===
        top_bar = QHBoxLayout()
        
        self.btn_undo = QPushButton("↩ 撤销"); self.btn_undo.clicked.connect(self.perform_undo)
        self.btn_redo = QPushButton("↪ 重做"); self.btn_redo.clicked.connect(self.perform_redo)
        # 绑定快捷键
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.perform_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self.perform_redo)
        # 绑定空格键播放
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.toggle_play)

        btn_save = QPushButton("💾 另存为"); btn_save.clicked.connect(self.save_as)
        
        top_bar.addWidget(self.btn_undo)
        top_bar.addWidget(self.btn_redo)
        top_bar.addSpacing(20)
        top_bar.addWidget(btn_save)
        top_bar.addStretch()
        
        layout.addLayout(top_bar)
        
        # === 分割视图 ===
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        left_container = QWidget()
        l_layout = QVBoxLayout(left_container)
        l_layout.setContentsMargins(0,0,0,0)
        
        self.graph = CurveEditor()
        self.graph.set_backend(self.backend, self)
        l_layout.addWidget(self.graph)
        
        # === 播放控制条 ===
        play_ctrl = QHBoxLayout()
        
        self.btn_prev = QPushButton("◀"); self.btn_prev.clicked.connect(lambda: self.jump(-1))
        
        # 播放/暂停按钮
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
        
        right_container = QFrame()
        right_container.setFrameShape(QFrame.StyledPanel)
        r_layout = QVBoxLayout(right_container)
        
        lbl_list = QLabel("关节列表 (单选编辑)")
        r_layout.addWidget(lbl_list)
        
        self.joint_list = QListWidget()
        self.joint_list.setSelectionMode(QAbstractItemView.SingleSelection)
        for i, name in enumerate(self.backend.csv_joint_names):
            self.joint_list.addItem(f"[{i:02d}] {name.replace('_joint','')}")
        self.joint_list.itemSelectionChanged.connect(self.on_selection_change)
        r_layout.addWidget(self.joint_list)
        
        tools_group = QGroupBox("批量工具")
        t_layout = QVBoxLayout()
        btn_smooth = QPushButton("🌊 选区平滑"); btn_smooth.clicked.connect(self.apply_smooth)
        t_layout.addWidget(btn_smooth)
        btn_add = QPushButton("✨ 叠加插值 (Additive)"); btn_add.clicked.connect(self.apply_additive)
        t_layout.addWidget(btn_add)
        btn_reset = QPushButton("🔄 重置选中区域"); btn_reset.clicked.connect(self.reset_original)
        t_layout.addWidget(btn_reset)
        tools_group.setLayout(t_layout)
        r_layout.addWidget(tools_group)
        
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        
        self.status_bar = self.statusBar()

    # === 播放逻辑 ===
    def toggle_play(self):
        if self.total_frames == 0: return
        
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.btn_play.setText("▶ 播放 (Space)")
            self.btn_play.setStyleSheet("background-color: #44aa44; font-weight: bold; color: white;")
        else:
            # 30 FPS => 33ms interval
            self.timer.start(33)
            self.is_playing = True
            self.btn_play.setText("⏸ 暂停 (Space)")
            self.btn_play.setStyleSheet("background-color: #aa4444; font-weight: bold; color: white;")

    def play_next_frame(self):
        next_idx = self.current_frame + 1
        # 循环播放
        if next_idx >= self.total_frames:
            next_idx = 0
        self.update_frame(next_idx)

    # === 原有逻辑 ===
    def on_selection_change(self):
        items = self.joint_list.selectedIndexes()
        if not items: return
        indices = [i.row() for i in items]
        self.graph.update_curves(indices)

    def update_frame(self, idx):
        self.current_frame = idx
        self.lbl_frame.setText(f"{idx:04d}")
        # 这里只设置位置，不触发 dragged 信号，避免递归
        self.graph.current_frame_line.setValue(idx)
        self.backend.set_frame(idx)

    def update_frame_from_graph(self, idx):
        idx = max(0, min(self.total_frames-1, idx))
        self.current_frame = idx
        self.lbl_frame.setText(f"{idx:04d}")
        self.backend.set_frame(idx)

    def jump(self, delta):
        new = max(0, min(self.total_frames-1, self.current_frame + delta))
        self.update_frame(new)

    def apply_smooth(self):
        if self.graph.selected_joint_idx is None: return
        self.backend.snapshot()
        r_min, r_max = self.graph.region.getRegion()
        s, e = int(r_min), int(r_max)
        s = max(0, s); e = min(self.total_frames-1, e)
        col = 7 + self.graph.selected_joint_idx
        self.backend.df.iloc[s:e+1, col] = self.backend.df.iloc[s:e+1, col].rolling(window=5, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        self.graph.update_curves([self.graph.selected_joint_idx])
        self.backend.set_frame(self.current_frame)
        self.status_bar.showMessage("已平滑")

    def apply_additive(self):
        if self.graph.selected_joint_idx is None: return
        self.backend.snapshot()
        r_min, r_max = self.graph.region.getRegion()
        s, e = int(r_min), int(r_max)
        if s >= e: return
        col = 7 + self.graph.selected_joint_idx
        orig, curr = self.backend.df_orig, self.backend.df
        delta_s = curr.iloc[s, col] - orig.iloc[s, col]
        delta_e = curr.iloc[e, col] - orig.iloc[e, col]
        interp = PchipInterpolator([s, e], [delta_s, delta_e])
        curr.iloc[s:e+1, col] = orig.iloc[s:e+1, col] + interp(np.arange(s, e + 1))
        self.graph.update_curves([self.graph.selected_joint_idx])
        self.backend.set_frame(self.current_frame)
        self.status_bar.showMessage("已插值")

    def reset_original(self):
        if self.graph.selected_joint_idx is None: return
        self.backend.snapshot()
        r_min, r_max = self.graph.region.getRegion()
        s, e = int(r_min), int(r_max)
        col = 7 + self.graph.selected_joint_idx
        self.backend.df.iloc[s:e+1, col] = self.backend.df_orig.iloc[s:e+1, col]
        self.graph.update_curves([self.graph.selected_joint_idx])
        self.backend.set_frame(self.current_frame)
        self.status_bar.showMessage("已重置")

    def perform_undo(self):
        if self.backend.undo(): self.refresh_ui("已撤销")
    def perform_redo(self):
        if self.backend.redo(): self.refresh_ui("已重做")

    def refresh_ui(self, msg):
        if self.graph.selected_joint_idx is not None:
            self.graph.update_curves([self.graph.selected_joint_idx])
        self.backend.set_frame(self.current_frame)
        self.status_bar.showMessage(msg)

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