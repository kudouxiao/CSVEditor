import os
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSplitter, QListWidget, 
                             QAbstractItemView, QFrame, QTabWidget, QScrollArea, 
                             QGroupBox, QCheckBox, QDoubleSpinBox, QComboBox, 
                             QShortcut, QFileDialog, QMessageBox, QListWidgetItem)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from scipy.interpolate import PchipInterpolator

# 导入模块
from src.config import (DEFAULT_CSV_PATH, DEFAULT_MODEL_PATH, 
                        DEFAULT_SMPLX_DATA_PATH, DEFAULT_BVH_PATH, SMPLX_BODY_MODEL_DIR, REF_LOAD_MODE)
from src.core.backend import G1Backend
from src.ui.widgets.mujoco_widget import MuJoCoWidget
from src.ui.widgets.curve_editor import CurveEditor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("G1 Pro Editor (SMPL & BVH Support)")
        self.resize(1600, 1000)
        
        # 调试输出：检查所有路径
        print("="*40)
        print("SYSTEM CHECK:")
        print(f"CSV Path:   {DEFAULT_CSV_PATH} -> {os.path.exists(DEFAULT_CSV_PATH)}")
        print(f"Model Path: {DEFAULT_MODEL_PATH} -> {os.path.exists(DEFAULT_MODEL_PATH)}")
        print(f"SMPL Data:  {DEFAULT_SMPLX_DATA_PATH} -> {os.path.exists(DEFAULT_SMPLX_DATA_PATH)}")
        print(f"SMPL Dir:   {SMPLX_BODY_MODEL_DIR} -> {os.path.exists(SMPLX_BODY_MODEL_DIR)}")
        print(f"BVH Data:   {DEFAULT_BVH_PATH} -> {os.path.exists(DEFAULT_BVH_PATH)}") # 新增 BVH 检查
        print("="*40)

        self.backend = G1Backend()
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next_frame)
        
        # 1. 自动加载机器人 (保持不变)
        if os.path.exists(DEFAULT_CSV_PATH) and os.path.exists(DEFAULT_MODEL_PATH):
            success, frames = self.backend.load_data(DEFAULT_CSV_PATH, DEFAULT_MODEL_PATH)
            if success:
                self.mujoco_widget.init_mujoco(self.backend.model, self.backend.data)
                self.total_frames = frames
                self.graph.setXRange(0, frames)
                self.graph.region.setRegion([0, frames//5])
                self.update_frame(0)
                self.status_bar.showMessage(f"Loaded Robot Data: {frames} frames.")
        
        # 2. 根据配置加载参考动作 (SMPL vs BVH)
        ref_loaded = False
        
        # 定义加载函数以避免重复代码
        def try_load_smpl():
            if os.path.exists(DEFAULT_SMPLX_DATA_PATH) and os.path.exists(SMPLX_BODY_MODEL_DIR):
                print("[Init] Loading SMPL...")
                if self.backend.load_smplx_data(DEFAULT_SMPLX_DATA_PATH, SMPLX_BODY_MODEL_DIR):
                    self.mujoco_widget.set_ref_data(self.backend.ref_joints, self.backend.ref_parents)
                    return True
            return False

        def try_load_bvh():
            if os.path.exists(DEFAULT_BVH_PATH):
                print("[Init] Loading BVH...")
                if self.backend.load_bvh_data(DEFAULT_BVH_PATH):
                    self.mujoco_widget.set_ref_data(self.backend.ref_joints, self.backend.ref_parents)
                    return True
            return False

        # 根据配置决定加载逻辑
        if REF_LOAD_MODE == "SMPL":
            if try_load_smpl():
                self.status_bar.showMessage("Loaded Robot & SMPL Ref")
            else:
                print(f"SMPL Load Failed (Mode: {REF_LOAD_MODE})")

        elif REF_LOAD_MODE == "BVH":
            if try_load_bvh():
                self.status_bar.showMessage("Loaded Robot & BVH Ref")
            else:
                print(f"BVH Load Failed (Mode: {REF_LOAD_MODE})")

        elif REF_LOAD_MODE == "AUTO":
            # 优先 SMPL，失败则尝试 BVH
            if try_load_smpl():
                self.status_bar.showMessage("Loaded Robot & SMPL Ref (Auto)")
            elif try_load_bvh():
                self.status_bar.showMessage("Loaded Robot & BVH Ref (Auto)")
        
        if not ref_loaded:
            print("No reference motion loaded (SMPL/BVH not found).")

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
        btn_load_bvh = QPushButton("[Load BVH]")
        btn_load_bvh.clicked.connect(self.load_bvh_ref)
        
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.perform_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self.perform_redo)
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.toggle_play)
        
        top_bar.addWidget(self.btn_undo)
        top_bar.addWidget(self.btn_redo)
        top_bar.addSpacing(20)
        top_bar.addWidget(btn_save)
        top_bar.addWidget(btn_load_smpl)
        top_bar.addWidget(btn_load_bvh) # 新增按钮
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
        
        smpl_g = QGroupBox("Reference Motion Settings")
        smpl_l = QVBoxLayout()
        self.chk_smpl_vis = QCheckBox("Show Original Reference")
        self.chk_smpl_vis.setChecked(True)
        self.chk_smpl_vis.stateChanged.connect(
            lambda s: setattr(self.mujoco_widget, 'show_ref', s==Qt.Checked) or self.mujoco_widget.update()
        )
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
                self.mujoco_widget.set_ref_data(
                    self.backend.ref_joints, 
                    self.backend.ref_parents
                )
                self.status_bar.showMessage(f"SMPL-X Loaded: {len(self.backend.smplx_joints)} frames")
                self.mujoco_widget.smplx_offset = np.array([0.0, 1.0, 0.0])
                self.mujoco_widget.update()
            else:
                QMessageBox.warning(self, "Error", "Failed to load SMPL-X. Check console for details.")


    def load_bvh_ref(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load BVH", "", "BVH Files (*.bvh)")
        if path:
            self.status_bar.showMessage("Loading BVH...")
            QApplication.processEvents()
            
            result = self.backend.load_bvh_data(path)
            if result is True:
                # 传递数据和父子关系
                self.mujoco_widget.set_ref_data(
                    self.backend.ref_joints, 
                    self.backend.ref_parents
                )
                self.status_bar.showMessage(f"BVH Loaded: {len(self.backend.ref_joints)} frames")
                self.mujoco_widget.smplx_offset = np.array([0.0, 1.0, 1.0]) # 重置偏移
                self.mujoco_widget.update()
            else:
                QMessageBox.warning(self, "Error", f"BVH Load Failed: {result}")


    def update_smpl_offset(self, axis, value):
        self.mujoco_widget.ref_offset[axis] = value
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