import os
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSplitter, QListWidget, 
                             QAbstractItemView, QFrame, QTabWidget, QScrollArea, 
                             QGroupBox, QCheckBox, QDoubleSpinBox, QComboBox, 
                             QShortcut, QFileDialog, QMessageBox, QListWidgetItem, QSpinBox, QApplication, QDialog)
                             # 引入媒体库
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

# 导入模块
from src.config import (DEFAULT_CSV_PATH, DEFAULT_MODEL_PATH, 
                        DEFAULT_SMPLX_DATA_PATH, DEFAULT_BVH_PATH, SMPLX_BODY_MODEL_DIR, REF_LOAD_MODE, ROBOT_FPS)
from src.core.backend import G1Backend
from src.ui.widgets.mujoco_widget import MuJoCoWidget
from src.ui.widgets.curve_editor import CurveEditor
from src.ui.widgets.audio_track import AudioTrack

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("G1 Pro Editor")
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
        self.updating_region_from_spin = False # 防止循环信号
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

        # === 新增：音频播放器 ===
        self.media_player = QMediaPlayer() # 播放器
        self.media_player.setVolume(100) # 设置音量
        self.media_player.error.connect(lambda: print(f"Media Error: {self.media_player.errorString()}"))

    def init_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # === Top Bar ===
        top_bar = QHBoxLayout()
        self.btn_undo = QPushButton("↩ 撤销"); self.btn_undo.clicked.connect(self.perform_undo)
        self.btn_redo = QPushButton("↪ 重做"); self.btn_redo.clicked.connect(self.perform_redo)
        
        self.chk_ghost = QCheckBox("👻 显示原数据(Ghost)")
        self.chk_ghost.stateChanged.connect(self.toggle_ghost)
        
        btn_save = QPushButton("💾 另存为"); btn_save.clicked.connect(self.save_as)
        
        # 增加一个加载音频的按钮
        btn_audio = QPushButton("🎵 加载音乐"); btn_audio.clicked.connect(self.load_audio)
        
        btn_load_smpl = QPushButton("🕺 加载参考"); btn_load_smpl.clicked.connect(self.load_smplx_ref)

        top_bar.addWidget(self.btn_undo); top_bar.addWidget(self.btn_redo); top_bar.addSpacing(10)
        top_bar.addWidget(self.chk_ghost); top_bar.addSpacing(10)
        top_bar.addWidget(btn_audio); top_bar.addStretch()
        top_bar.addWidget(btn_save); top_bar.addWidget(btn_load_smpl)
        layout.addLayout(top_bar)
        
        splitter = QSplitter(Qt.Horizontal); layout.addWidget(splitter)
        
        # === Left ===
        left_container = QWidget(); l_layout = QVBoxLayout(left_container); l_layout.setContentsMargins(0,0,0,0)
        self.mujoco_widget = MuJoCoWidget()
        l_layout.addWidget(self.mujoco_widget, stretch=4)

        self.graph = CurveEditor()
        self.graph.set_backend(self.backend, self)
        l_layout.addWidget(self.graph, stretch=3)

        # 3. Audio Track (Bottom)
        self.audio_track = AudioTrack()
        self.audio_track.set_backend(self.backend)
        self.audio_track.setXLink(self.graph) # 保持 X 轴同步缩放
        self.audio_track.frame_changed.connect(self.update_frame_from_graph)
        l_layout.addWidget(self.audio_track, stretch=0) # stretch=0 固定高度

        # 播放控制
        play_ctrl = QHBoxLayout()
        self.btn_prev = QPushButton("◀"); self.btn_prev.clicked.connect(lambda: self.jump(-1))
        self.btn_play = QPushButton("▶ 播放"); self.btn_play.clicked.connect(self.toggle_play)
        self.btn_next = QPushButton("▶"); self.btn_next.clicked.connect(lambda: self.jump(1))
        self.lbl_frame = QLabel("0000"); self.lbl_frame.setFixedWidth(50)
        play_ctrl.addWidget(self.btn_prev); play_ctrl.addWidget(self.btn_play); play_ctrl.addWidget(self.btn_next); play_ctrl.addWidget(self.lbl_frame)
        l_layout.addLayout(play_ctrl)
        splitter.addWidget(left_container)
        
        # === Right ===
        right_container = QFrame(); right_container.setFrameShape(QFrame.StyledPanel); r_layout = QVBoxLayout(right_container)
        
        # 1. 列表
        r_layout.addWidget(QLabel("通道列表"))
        self.joint_list = QListWidget(); self.joint_list.setSelectionMode(QAbstractItemView.SingleSelection)
        for i, name in enumerate(self.backend.all_names):
            item = QListWidgetItem(f"[{i:02d}] {name.replace('_joint','')}")
            if i < 6: item.setForeground(Qt.cyan)
            self.joint_list.addItem(item)
        self.joint_list.itemSelectionChanged.connect(self.on_selection_change)
        r_layout.addWidget(self.joint_list)
        
        # 2. 范围选择 (全局)
        range_group = QGroupBox("编辑范围"); range_layout = QHBoxLayout()
        self.spin_start = QSpinBox(); self.spin_start.setRange(0, 99999); self.spin_start.valueChanged.connect(self.on_spinbox_changed)
        self.spin_end = QSpinBox(); self.spin_end.setRange(0, 99999); self.spin_end.valueChanged.connect(self.on_spinbox_changed)
        range_layout.addWidget(QLabel("Start:")); range_layout.addWidget(self.spin_start)
        range_layout.addWidget(QLabel("End:")); range_layout.addWidget(self.spin_end)
        range_group.setLayout(range_layout); r_layout.addWidget(range_group)
        
        # 3. 功能 Tabs
        self.tabs = QTabWidget()
        
        # --- Tab 1: Tools ---
        tab_tools = QWidget(); tb_layout = QVBoxLayout(tab_tools)
        
        # A. 帧操作组 (新增)
        g_frame_ops = QGroupBox("帧操作 (Timeline)")
        l_frame_ops = QVBoxLayout()
        
        h_insert = QHBoxLayout()
        h_insert.addWidget(QLabel("数量:"))
        self.spin_frame_count = QSpinBox(); self.spin_frame_count.setRange(1, 1000); self.spin_frame_count.setValue(10)
        h_insert.addWidget(self.spin_frame_count)
        btn_insert = QPushButton("➕ 插入帧 (Insert)")
        btn_insert.setToolTip("在当前光标位置插入 N 帧当前姿态")
        btn_insert.clicked.connect(self.perform_frame_insert)
        h_insert.addWidget(btn_insert)
        
        btn_delete = QPushButton("➖ 删除选区 (Delete)")
        btn_delete.setToolTip("删除蓝色选区内的所有帧")
        btn_delete.clicked.connect(self.perform_frame_delete)
        
        l_frame_ops.addLayout(h_insert); l_frame_ops.addWidget(btn_delete)
        g_frame_ops.setLayout(l_frame_ops)
        tb_layout.addWidget(g_frame_ops)
        
        # B. 曲线生成组
        g_curve_gen = QGroupBox("曲线生成")
        l_curve_gen = QVBoxLayout()
        btn_linear = QPushButton("📏 直线连接 (Linear)"); btn_linear.clicked.connect(lambda: self.apply_connect("linear"))
        btn_sigmoid = QPushButton("🌊 S形连接 (Sigmoid)"); btn_sigmoid.clicked.connect(lambda: self.apply_connect("sigmoid"))
        l_curve_gen.addWidget(btn_linear); l_curve_gen.addWidget(btn_sigmoid)
        g_curve_gen.setLayout(l_curve_gen)
        tb_layout.addWidget(g_curve_gen)
        
        # C. 批处理组
        g_batch = QGroupBox("批处理")
        l_batch = QVBoxLayout()
        btn_smooth = QPushButton("💧 SavGol 平滑"); btn_smooth.clicked.connect(self.apply_smooth_savgol)
        btn_add = QPushButton("✨ 叠加插值 (Additive)"); btn_add.clicked.connect(self.apply_additive)
        btn_mirror = QPushButton("🪞 动作镜像 (Mirror)"); btn_mirror.clicked.connect(self.apply_mirror_action)
        btn_align = QPushButton("🔗 对齐全局坐标 (Align)"); btn_align.clicked.connect(self.align_global_coords)
        btn_align.setToolTip("在当前帧处对齐后续动作，用于拼接两段动作")
        # # === 新增：手动修复四元数按钮 ===
        # btn_fix_quat = QPushButton("🔧 修复四元数 (Fix Quat)")
        # btn_fix_quat.setToolTip("强制归一化并修复 Root 旋转的连续性")
        # btn_fix_quat.clicked.connect(self.apply_quat_fix)
        # ==============================

        btn_reset = QPushButton("🔄 重置选中区域"); btn_reset.clicked.connect(self.reset_original)
        
        # # === 新增：Euler角编辑按钮 ===
        # btn_euler_edit = QPushButton("🔄 Euler角编辑 (Euler Edit)")
        # btn_euler_edit.setToolTip("切换到欧拉角编辑模式（仅适用于Root旋转）")
        # btn_euler_edit.clicked.connect(self.toggle_euler_mode)
        # # ==============================
        
        l_batch.addWidget(btn_smooth)
        l_batch.addWidget(btn_add)
        l_batch.addWidget(btn_mirror)
        # l_batch.addWidget(btn_fix_quat) # 添加到布局
        l_batch.addWidget(btn_align)
        # l_batch.addWidget(btn_euler_edit)  # 添加Euler按钮
        l_batch.addWidget(btn_reset)
        g_batch.setLayout(l_batch)
        tb_layout.addWidget(g_batch)

        tb_layout.addStretch() # 底部留白
        self.tabs.addTab(tab_tools, "🛠️ 工具")
        
        # --- Tab 2: Spline ---
        tab_spline = QWidget(); ts_layout = QVBoxLayout(tab_spline)
        h_anc = QHBoxLayout(); h_anc.addWidget(QLabel("锚点数:")); self.spin_anchor_count = QSpinBox(); self.spin_anchor_count.setRange(3, 50); self.spin_anchor_count.setValue(5); h_anc.addWidget(self.spin_anchor_count)
        ts_layout.addLayout(h_anc)
        
        self.btn_spline_start = QPushButton("✏️ 开始编辑"); self.btn_spline_start.clicked.connect(self.toggle_spline_mode)
        self.btn_spline_apply = QPushButton("✅ 应用"); self.btn_spline_apply.setEnabled(False); self.btn_spline_apply.clicked.connect(self.apply_spline)
        self.btn_spline_cancel = QPushButton("❌ 取消"); self.btn_spline_cancel.setEnabled(False); self.btn_spline_cancel.clicked.connect(self.cancel_spline)
        
        ts_layout.addWidget(self.btn_spline_start)
        ts_layout.addWidget(self.btn_spline_apply)
        ts_layout.addWidget(self.btn_spline_cancel)
        ts_layout.addStretch()
        self.tabs.addTab(tab_spline, "✏️ 样条")
        
        # --- Tab 3: View (原有的) ---
        tab_view = QWidget(); v_layout = QVBoxLayout(tab_view)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll_w = QWidget(); scroll_l = QVBoxLayout(scroll_w)
        
        # Ref Settings
        smpl_g = QGroupBox("Reference Settings"); smpl_l = QVBoxLayout()
        self.chk_ref_vis = QCheckBox("Show Reference"); self.chk_ref_vis.setChecked(True)
        self.chk_ref_vis.stateChanged.connect(lambda s: setattr(self.mujoco_widget, 'show_ref', s==Qt.Checked) or self.mujoco_widget.update())
        smpl_l.addWidget(self.chk_ref_vis)
        
        # Frame Scale
        scale_l = QHBoxLayout(); scale_l.addWidget(QLabel("Scale:")); sp_scale = QDoubleSpinBox(); sp_scale.setRange(0.1, 10.0); sp_scale.setValue(3.0); sp_scale.valueChanged.connect(lambda v: setattr(self.mujoco_widget, 'smplx_frame_scale', v) or self.mujoco_widget.update()); scale_l.addWidget(sp_scale); smpl_l.addLayout(scale_l)
        
        # Offset
        off_l = QHBoxLayout(); off_l.addWidget(QLabel("X:")); sp_x = QDoubleSpinBox(); sp_x.setRange(-5,5); sp_x.valueChanged.connect(lambda v: self.update_smpl_offset(0,v)); off_l.addWidget(sp_x)
        off_l.addWidget(QLabel("Y:")); sp_y = QDoubleSpinBox(); sp_y.setRange(-5,5); sp_y.setValue(1.0); sp_y.valueChanged.connect(lambda v: self.update_smpl_offset(1,v)); off_l.addWidget(sp_y)
        smpl_l.addLayout(off_l); smpl_g.setLayout(smpl_l); scroll_l.addWidget(smpl_g)
        
        # Flags
        g_flags = QGroupBox("Render Flags"); fl_l = QVBoxLayout()
        for name, (flag, val, _) in self.mujoco_widget.render_flags.items():
            cb = QCheckBox(name); cb.setChecked(val)
            cb.stateChanged.connect(lambda s, n=name: self.mujoco_widget.set_render_flag(n, s==Qt.Checked))
            fl_l.addWidget(cb)
        g_flags.setLayout(fl_l); scroll_l.addWidget(g_flags)
        
        scroll.setWidget(scroll_w); v_layout.addWidget(scroll)
        self.tabs.addTab(tab_view, "👀 视图")
        
        r_layout.addWidget(self.tabs)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 5); splitter.setStretchFactor(1, 1)
        self.status_bar = self.statusBar()
        
        # Shortcuts
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.perform_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self.perform_redo)
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.toggle_play)

    # === Interaction Logic ===
    def toggle_ghost(self, state):
        self.graph.set_ghost_visible(state == Qt.Checked)

    def on_region_changed(self):
        if self.updating_region_from_spin: return
        r_min, r_max = self.graph.region.getRegion()
        s, e = int(r_min), int(r_max)
        self.spin_start.setValue(s); self.spin_end.setValue(e)

    def on_spinbox_changed(self):
        s = self.spin_start.value(); e = self.spin_end.value()
        if s > e: return 
        self.updating_region_from_spin = True
        self.graph.region.setRegion([s, e])
        self.updating_region_from_spin = False

    # --- Tool Functions ---
    def apply_connect(self, mode):
        if self.graph.selected_joint_idx is None: return
        self.backend.snapshot()
        s, e = self.spin_start.value(), self.spin_end.value()
        col = self.graph.selected_joint_idx # 直接使用 idx
        
        if s >= e: return
        val_start = self.backend.df.iloc[s, col]
        val_end = self.backend.df.iloc[e, col]
        count = e - s + 1
        
        if mode == "linear":
            new_vals = np.linspace(val_start, val_end, count)
        elif mode == "sigmoid":
            t = np.linspace(0, 1, count)
            w = (1 - np.cos(t * np.pi)) / 2
            new_vals = val_start + (val_end - val_start) * w
            
        self.backend.df.iloc[s:e+1, col] = new_vals
        self.backend.modified_frames.update(range(s, e+1))
        self.refresh_ui(f"Applied {mode} connect")

    def apply_smooth_savgol(self):
        if self.graph.selected_joint_idx is None: return
        self.backend.snapshot()
        s, e = self.spin_start.value(), self.spin_end.value()
        col = self.graph.selected_joint_idx
        
        data_chunk = self.backend.df.iloc[s:e+1, col].values
        # Window size must be odd and <= length
        window_len = min(len(data_chunk), 31)
        if window_len % 2 == 0: window_len -= 1
        
        # Ensure window_len > polyorder
        polyorder = 3
        if window_len <= polyorder:
            # Reduce polyorder to fit the window, or use a minimum
            polyorder = min(polyorder, window_len - 1)
            if polyorder < 0:
                polyorder = 0  # Minimum possible order
        
        if window_len >= 3 and polyorder >= 0:
            smoothed = savgol_filter(data_chunk, window_len, polyorder) # polyorder dynamically adjusted
            self.backend.df.iloc[s:e+1, col] = smoothed
            self.backend.modified_frames.update(range(s, e+1))
            self.refresh_ui("Applied SavGol Smooth")
        else:
            # For very short windows, use a simple average instead of failing
            self.backend.df.iloc[s:e+1, col] = np.mean(data_chunk)
            self.refresh_ui("Applied Average Smooth (too short for SavGol)")

    # --- Spline Functions ---
    def toggle_spline_mode(self):
        if self.graph.selected_joint_idx is None:
            QMessageBox.warning(self, "提示", "请先选择通道")
            return
        
        s = self.spin_start.value(); e = self.spin_end.value()
        n = self.spin_anchor_count.value()
        if e - s < n: return
        
        # Lock UI
        self.btn_spline_start.setEnabled(False)
        self.btn_spline_apply.setEnabled(True)
        self.btn_spline_cancel.setEnabled(True)
        self.tabs.setCurrentIndex(1) # Force switch tab
        
        self.graph.start_spline_mode(s, e, num_anchors=n)
        self.status_bar.showMessage("Spline Mode Active")

    def apply_spline(self):
        self.graph.apply_spline_to_data()
        self.reset_spline_ui()
        self.status_bar.showMessage("Spline Applied")

    def cancel_spline(self):
        self.graph.cancel_spline_mode()
        self.reset_spline_ui()
        self.status_bar.showMessage("Spline Canceled")

    def reset_spline_ui(self):
        self.btn_spline_start.setEnabled(True)
        self.btn_spline_apply.setEnabled(False)
        self.btn_spline_cancel.setEnabled(False)

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
                self.status_bar.showMessage(f"SMPL-X Loaded: {len(self.backend.ref_joints)} frames")
                self.mujoco_widget.ref_offset = np.array([0.0, 1.0, 0.0])
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
                self.mujoco_widget.ref_offset = np.array([0.0, 1.0, 1.0]) # 重置偏移
                self.mujoco_widget.update()
            else:
                QMessageBox.warning(self, "Error", f"BVH Load Failed: {result}")

    def perform_frame_insert(self):
        """在当前红色播放线位置插入帧"""
        count = self.spin_frame_count.value()
        # 获取红色播放线当前所在的位置
        current_idx = self.current_frame 
        
        # 调用后端，并获取插入范围
        insert_range = self.backend.insert_frames(current_idx, count)
        
        if insert_range:
            # 1. 刷新界面结构 (更新总帧数、滑块范围)
            self.refresh_ui_structure(f"已在第 {current_idx} 帧处插入 {count} 帧 (绿色区域)")
            
            # 2. 在曲线图中添加绿色高亮
            start, end = insert_range
            self.graph.add_highlight_region(start, end)
            
            # 3. 将播放头移动到插入段的末尾，方便继续操作
            self.update_frame(end)

    def perform_frame_delete(self):
        """删除当前选区内的帧"""
        r_min, r_max = self.graph.region.getRegion()
        s, e = int(r_min), int(r_max)
        
        if e - s + 1 >= self.backend.df.shape[0]:
            QMessageBox.warning(self, "警告", "不能删除所有帧，至少保留一帧。")
            return

        reply = QMessageBox.question(self, '确认删除', 
                                     f"确定要删除第 {s} 到 {e} 帧吗？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            count = e - s + 1
            if self.backend.delete_frames(s, count):
                # === 修复点 1: 删除操作发生后，立即清除之前的绿色高亮 ===
                # 因为帧数变了，旧的高亮位置已经没有意义了，保留会导致错位
                self.graph.clear_highlights()
                
                self.refresh_ui_structure(f"已删除 {count} 帧")
                
                # 修正光标位置 (防止光标停留在已删除的区域)
                new_idx = min(self.current_frame, self.backend.df.shape[0]-1)
                self.update_frame(new_idx)

    # === 修改刷新函数 (添加清除逻辑) ===
    def refresh_ui_structure(self, status_msg):
        """
        当数据结构（总帧数）发生变化时调用此函数
        """
        self.total_frames = self.backend.df.shape[0]
        
        # 1. 更新 ViewBox 范围
        self.graph.setXRange(0, self.total_frames)
        
        # === 修复点 2: 强制修正蓝色选区的位置，防止越界 ===
        self.graph.limit_region_to_range(self.total_frames)
        
        # 3. 更新 SpinBox 范围
        self.spin_start.setMaximum(self.total_frames - 1)
        self.spin_end.setMaximum(self.total_frames - 1)
        
        # 4. 同步 SpinBox 数值 (因为 limit_region 可能会改变选区，需要反向同步给数字框)
        r_min, r_max = self.graph.region.getRegion()
        self.spin_start.setValue(int(r_min))
        self.spin_end.setValue(int(r_max))
        
        # 5. 强制刷新曲线
        if self.graph.selected_joint_idx is not None:
            self.graph.update_curves([self.graph.selected_joint_idx])
            
        self.status_bar.showMessage(f"{status_msg} (总帧数: {self.total_frames})")

    def update_smpl_offset(self, axis, value):
        self.mujoco_widget.ref_offset[axis] = value
        self.mujoco_widget.update()

    # def toggle_play(self):
    #     if self.total_frames == 0: 
    #         return
    #     if self.is_playing: 
    #         self.timer.stop()
    #         self.is_playing = False
    #         self.btn_play.setText("[Play/Space]")
    #         self.btn_play.setStyleSheet("background-color: #44aa44; font-weight: bold; color: white;")
    #     else: 
    #         self.timer.start(33)
    #         self.is_playing = True
    #         self.btn_play.setText("[Pause/Space]")
    #         self.btn_play.setStyleSheet("background-color: #aa4444; font-weight: bold; color: white;")

    # def play_next_frame(self):
    #     next_idx = self.current_frame + 1
    #     if next_idx >= self.total_frames: 
    #         next_idx = 0
    #     self.update_frame(next_idx)

    def on_selection_change(self):
        items = self.joint_list.selectedIndexes()
        if not items: 
            return
        indices = [i.row() for i in items]
        self.graph.update_curves(indices)

    def update_frame(self, idx):
        self.current_frame = idx
        self.lbl_frame.setText(f"{idx:04d}")
        
        # 1. 更新上方曲线图的红线
        # blockSignals 防止循环调用 (虽然 setValue 通常不触发 dragged，但保险起见)
        self.graph.current_frame_line.blockSignals(True)
        self.graph.current_frame_line.setValue(idx)
        self.graph.current_frame_line.blockSignals(False)
        
        # 2. 更新下方音轨的红线
        self.audio_track.current_line.blockSignals(True)
        self.audio_track.current_line.setValue(idx)
        self.audio_track.current_line.blockSignals(False)
        
        # 3. 更新后端和画面
        self.mujoco_widget.current_frame_idx = idx # 同步帧号给渲染器
        self.backend.set_frame(idx)
        self.mujoco_widget.update() 

    # def update_frame_from_graph(self, idx):
    #     idx = max(0, min(self.total_frames-1, idx))
    #     self.current_frame = idx
    #     self.lbl_frame.setText(f"{idx:04d}")
    #     self.mujoco_widget.current_frame_idx = idx
    #     self.backend.set_frame(idx)
    #     self.mujoco_widget.update()

    def update_frame_from_graph(self, idx):
        idx = max(0, min(self.total_frames-1, idx))
        self.current_frame = idx
        self.lbl_frame.setText(f"{idx:04d}")
        self.mujoco_widget.current_frame_idx = idx # 同步帧号给渲染器
        self.backend.set_frame(idx)
        self.mujoco_widget.update()
        
        # === 新增：拖动时同步声音 (可选) ===
        # 如果你想拖动时听到“滋滋”的声音可以加，但通常建议拖动时不播放，只定位
        # 但我们需要更新播放器的内部指针，这样下次按播放时能接上
        fps = 30.0
        time_ms = int((idx / fps) * 1000)
        if self.media_player.mediaStatus() != QMediaPlayer.NoMedia:
             # 如果正在播放，拖动不应该打断播放逻辑，或者应该暂停播放
             # 这里不做操作，只有在 toggle_play 里的 start 才会用到 setPosition
             pass

    def handle_media_error(self):
        print(f"Media Player Error: {self.media_player.errorString()}")

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

    # === 修改撤销/重做 (清除高亮) ===
    def perform_undo(self):
        if self.backend.undo():
            self.graph.clear_highlights()
            self.refresh_ui_structure("已撤销")
            
            # 获取当前数据的最大有效索引
            max_idx = self.backend.df.shape[0] - 1
            # 如果当前光标位置超过了最大帧数，强行拉回来
            if self.current_frame > max_idx:
                self.current_frame = max_idx
            
            # 使用 update_frame 而不是 backend.set_frame
            # 这样可以同步更新滑块位置、数字显示和物理画面
            self.update_frame(self.current_frame)
            # === 修复结束 ===

    def perform_redo(self):
        if self.backend.redo():
            self.graph.clear_highlights()
            self.refresh_ui_structure("已重做")
            
            # Redo 同理（比如重做了一个“删除帧”的操作，总帧数变少，也需要防越界）
            max_idx = self.backend.df.shape[0] - 1
            if self.current_frame > max_idx:
                self.current_frame = max_idx
                
            self.update_frame(self.current_frame)
            
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

    # === 新增：加载音频槽函数 ===
    def load_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Audio", "", "Audio (*.mp3 *.wav)")
        if path:
            if self.backend.load_audio_data(path):
                self.audio_track.update_waveform() # 刷新音轨显示
                
                # 设置播放器源
                url = QUrl.fromLocalFile(path)
                self.media_player.setMedia(QMediaContent(url))
                self.status_bar.showMessage(f"Audio Loaded: {os.path.basename(path)}")

    # === 修改：播放逻辑 (核心同步) ===
    def toggle_play(self):
        if self.total_frames == 0: return
        
        if self.is_playing:
            self.timer.stop(); self.is_playing = False; self.media_player.pause()
            self.btn_play.setText("▶ 播放 (Space)")
        else:
            # 计算开始播放的位置
            # 机器人当前帧 -> 对应音频的时间点
            audio_start_time = self.backend.get_audio_time_from_frame(self.current_frame, ROBOT_FPS)
            
            # QMediaPlayer 使用毫秒
            start_ms = int(audio_start_time * 1000)
            
            if self.backend.audio_path:
                if start_ms < 0: 
                    # 音乐还没开始 (偏移导致)，延迟播放
                    self.media_player.stop()
                    # 这里可以做一个单次定时器在未来某个时刻 start，简化起见先 stop
                else:
                    self.media_player.setPosition(start_ms)
                    self.media_player.play()
            
            self.timer.start(33); self.is_playing = True
            self.btn_play.setText("⏸ 暂停 (Space)")


    def play_next_frame(self):
        # 这里的逻辑是：以 UI 帧数为准，还是以音乐时间为准？
        # 为了保证音画同步，通常以【音乐时间】为基准校准帧数
        
        if self.media_player.state() == QMediaPlayer.PlayingState:
            # 获取播放器当前时间 (s)
            current_audio_time = self.media_player.position() / 1000.0
            # 反推机器人应该在哪一帧：Frame = (AudioTime + Offset) * FPS
            target_frame = int((current_audio_time + self.backend.audio_offset) * ROBOT_FPS)
            
            # 简单的防跳变保护
            if target_frame > self.current_frame:
                next_idx = target_frame
            else:
                next_idx = self.current_frame + 1
        else:
            # 无音乐或音乐还没开始时的普通播放
            next_idx = self.current_frame + 1

        if next_idx >= self.total_frames: next_idx = 0
        self.update_frame(next_idx)

    def apply_mirror_action(self):
        s = self.spin_start.value()
        e = self.spin_end.value()
        
        # 弹窗确认，因为这是一个大范围修改
        reply = QMessageBox.question(self, '确认镜像', 
                                     f"确定要将第 {s} 到 {e} 帧进行左右镜像吗？\n这将交换左右手脚的数据。",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            if self.backend.apply_mirror(s, e):
                # 刷新曲线和画面
                self.refresh_ui(f"已镜像范围 {s}-{e}")
                # 如果当前选中的是左手，镜像后可能想看右手的情况，
                # 重新绘制当前选中的曲线
                if self.graph.selected_joint_idx is not None:
                    self.graph.update_curves([self.graph.selected_joint_idx])
                    
    def align_global_coords(self):
        """
        在当前播放帧处对齐后续动作的全局坐标
        使用场景：拼接两段动作时，将光标移动到第二段的起始帧，点击此按钮
        """
        current = self.current_frame
        if current <= 0:
            QMessageBox.warning(self, "提示", "请将光标移动到拼接点（第二段动作的起始帧）")
            return
        
        # 确认对话框
        reply = QMessageBox.question(
            self, 
            "确认对齐",
            f"将在帧 {current} 处对齐后续动作的全局坐标。\n\n"
            f"此操作将修改从帧 {current} 到末尾的所有 Root 数据。\n\n是否继续？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.backend.align_global_coordinates(current):
                self.refresh_ui(f"Aligned at frame {current}")
                QMessageBox.information(self, "成功", f"已对齐帧 {current} 后的动作")
            else:
                QMessageBox.warning(self, "错误", "对齐失败，请检查帧范围")


    # === 新增槽函数 ===
    def apply_quat_fix(self):
        self.backend.snapshot()
        self.backend.sanitize_quaternions()
        
        # 如果当前正好选中了四元数通道，刷新一下显示
        if self.graph.selected_joint_idx in [3, 4, 5, 6]:
            self.graph.update_curves([self.graph.selected_joint_idx])
            
        self.backend.set_frame(self.current_frame)
        self.status_bar.showMessage("四元数已清洗 (Normalized & Unwrapped)")

    # 在 MainWindow 类中

    def toggle_euler_mode(self):
        """切换欧拉角编辑模式 (非模态窗口)"""
        if self.graph.selected_joint_idx is None:
            QMessageBox.warning(self, "警告", "请先选择Root旋转通道 (索引3-6)")
            return
            
        if self.graph.selected_joint_idx not in [3, 4, 5, 6]:
            reply = QMessageBox.question(self, "确认", 
                                       "您选择的不是Root四元数通道。\n是否继续？",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No: return
        
        s = self.spin_start.value()
        e = self.spin_end.value()
        if s >= e:
            QMessageBox.warning(self, "提示", "范围无效 (Start >= End)")
            return

        # 1. 获取欧拉角数据
        eulers = self.backend.convert_range_quat_to_euler(s, e)
        if eulers is None: return
        
        # 2. 创建非模态对话框 (保存为成员变量防止被垃圾回收)
        # 注意：这里我们不再使用 exec_() 阻塞
        self.euler_dialog = EulerEditDialog(eulers, s, e, self)
        
        # 3. 智能定位：将弹窗移动到屏幕右下角或右侧，避开 MuJoCo
        # 获取主窗口几何信息
        geo = self.geometry()
        # 尝试移动到主窗口右侧内部，或者右下角
        dialog_x = geo.x() + geo.width() - 1050 # 假设弹窗宽1000，留点边距
        dialog_y = geo.y() + geo.height() - 650
        self.euler_dialog.move(max(geo.x(), dialog_x), max(geo.y(), dialog_y))
        
        self.euler_dialog.show() # 显示窗口，允许主界面继续运行