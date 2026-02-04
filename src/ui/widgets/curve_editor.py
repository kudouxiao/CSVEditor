import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from scipy.interpolate import CubicSpline
from src.config import ROBOT_FPS # 引入帧率用于计算时间

# === 自定义坐标轴类 ===
class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        """显示 帧数 (时间)"""
        strings = []
        for v in values:
            time_sec = v / ROBOT_FPS
            strings.append(f"{int(v)}\n{time_sec:.2f}s")
        return strings


class CurveEditor(pg.PlotWidget):
    """
    CurveEditor v3.4: 全局端点修正 + 局部软选择 + 四元数实时清洗
    """
    def __init__(self, parent=None):
        time_axis = TimeAxisItem(orientation='bottom')
        super().__init__(parent, axisItems={'bottom': time_axis})

        self.setBackground('#1e1e1e')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setMouseEnabled(x=True, y=False) # 锁定Y轴缩放

        self.getPlotItem().setLabel('bottom', 'Frame (Time)')
        self.getPlotItem().setLabel('left', 'Value')

        # === 0. 音频波形曲线 (Layer -100, 最底层) ===
        self.audio_curve = self.plot([], pen=pg.mkPen(None), brush=pg.mkBrush(30, 144, 255, 30)) 
        self.audio_curve.setZValue(-100) 
        self.audio_curve.setFillLevel(0) 
        
        # === 1. 选区 (Layer 10) ===
        self.region = pg.LinearRegionItem([0, 100], brush=(50, 50, 200, 50))
        self.region.setZValue(10)
        self.addItem(self.region)
        
        # === 2. 当前帧线 (Layer 100) ===
        self.current_frame_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#FF5555', width=2), movable=True)
        self.current_frame_line.setZValue(100)
        self.addItem(self.current_frame_line)
        
        # === 3. 辅助线 ===
        # Ghost Curve
        self.ghost_curve = self.plot([], pen=pg.mkPen((100, 100, 100), width=1, style=Qt.DashLine))
        self.ghost_curve.setZValue(0)
        self.show_ghost = False

        # Spline Anchors
        self.scatter_item = pg.ScatterPlotItem(size=15, pen=pg.mkPen('w'), brush=pg.mkBrush('#FFCC00'))
        self.scatter_item.setZValue(200)
        self.addItem(self.scatter_item)
        self.spline_preview_curve = self.plot([], pen=pg.mkPen('#FFCC00', width=2, style=Qt.DashLine))
        self.spline_preview_curve.setZValue(190)

        # 限位线
        limit_pen = pg.mkPen('#FF3333', width=1.5, style=Qt.DashLine)
        self.limit_upper = pg.InfiniteLine(angle=0, pen=limit_pen, movable=False)
        self.limit_lower = pg.InfiniteLine(angle=0, pen=limit_pen, movable=False)
        self.limit_upper.setZValue(50); self.addItem(self.limit_upper); self.limit_upper.hide()
        self.limit_lower.setZValue(50); self.addItem(self.limit_lower); self.limit_lower.hide()
        
        self.curves = {}
        self.selected_joint_idx = None
        self.backend_ref = None
        self.main_window_ref = None
        
        # 交互状态
        self.is_editing = False 
        self.drag_mode = "LOCAL_CENTER"
        self.spline_mode_active = False 
        self.drag_start_pos = None
        self.drag_start_data = None
        
        # Spline 数据
        self.spline_anchors_x = []
        self.spline_anchors_y = []
        self.dragged_anchor_index = None
        self.spline_boundary_slopes = None

        # 高亮列表
        self.insertion_highlights = []

    def set_backend(self, backend, main_window):
        self.backend_ref = backend
        self.main_window_ref = main_window
        self.current_frame_line.sigDragged.connect(self.on_line_dragged)
        self.region.sigRegionChanged.connect(main_window.on_region_changed)

    def set_ghost_visible(self, visible):
        self.show_ghost = visible
        if self.selected_joint_idx is not None:
            self.update_curves([self.selected_joint_idx])

    # === 高亮功能 ===
    def add_highlight_region(self, start, end):
        highlight = pg.LinearRegionItem([start, end], brush=(0, 255, 0, 50), movable=False)
        for line in highlight.lines:
            line.setPen(pg.mkPen(None))
            line.setHoverPen(pg.mkPen(None))
        highlight.setZValue(-10) 
        self.addItem(highlight)
        self.insertion_highlights.append(highlight)

    def clear_highlights(self):
        for item in self.insertion_highlights:
            self.removeItem(item)
        self.insertion_highlights.clear()

    def limit_region_to_range(self, total_frames):
        if total_frames <= 0: return
        r_min, r_max = self.region.getRegion()
        if r_max >= total_frames: r_max = total_frames - 1
        if r_min >= r_max: r_min = max(0, r_max - 10)
        self.region.setRegion([r_min, r_max])

    # === 曲线更新逻辑 ===
    def update_curves(self, selected_indices):
        for item in self.curves.values(): self.removeItem(item)
        self.curves.clear()
        
        if not self.backend_ref or self.backend_ref.df is None: return
        
        if len(selected_indices) > 0:
            self.selected_joint_idx = selected_indices[0]
            
            # 更新限位
            limits = self.backend_ref.get_joint_limits(self.selected_joint_idx)
            if limits is not None:
                self.limit_lower.setPos(limits[0]); self.limit_lower.show()
                self.limit_upper.setPos(limits[1]); self.limit_upper.show()
            else:
                self.limit_lower.hide(); self.limit_upper.hide()

            # 绘制 Ghost
            if self.show_ghost:
                col = self.selected_joint_idx
                orig_data = self.backend_ref.get_original_channel_data(col)
                self.ghost_curve.setData(orig_data)
            else:
                self.ghost_curve.setData([])

            # 绘制背景线 (其他选中的)
            for idx in selected_indices[1:]:
                col = idx
                data = self.backend_ref.get_channel_data(col)
                curve = self.plot(data, pen=pg.mkPen((80, 80, 80), width=1))
                curve.setZValue(5)
                self.curves[idx] = curve
            
            # 绘制主线 (高亮)
            col = self.selected_joint_idx
            data = self.backend_ref.get_channel_data(col)
            main_curve = self.plot(data, pen=pg.mkPen('#00ffff', width=3))
            main_curve.setZValue(20)
            self.curves[self.selected_joint_idx] = main_curve
            
            self.autoRange()
            if limits is not None:
                yr = self.viewRange()[1]
                new_min = min(yr[0], limits[0] - 0.2)
                new_max = max(yr[1], limits[1] + 0.2)
                self.setYRange(new_min, new_max, padding=0)
        else:
            self.selected_joint_idx = None
            self.ghost_curve.setData([])
            self.limit_lower.hide()
            self.limit_upper.hide()

    def on_line_dragged(self):
        idx = int(self.current_frame_line.value())
        if self.main_window_ref:
            self.main_window_ref.update_frame_from_graph(idx)

    # === Spline Mode ===
    def start_spline_mode(self, start_frame, end_frame, num_anchors=5):
        if self.selected_joint_idx is None: return
        self.spline_mode_active = True
        self.region.setMovable(False)
        col = self.selected_joint_idx
        
        # [FIX]: 使用 get_channel_data 代替 df.iloc
        full_data = self.backend_ref.get_channel_data(col) 
        
        slope_in = 0.0; slope_out = 0.0
        if start_frame > 0:
            slope_in = full_data[start_frame] - full_data[start_frame-1]
        
        # 注意: total_len 还是从 df 获取是安全的，因为行数是一致的
        total_len = len(self.backend_ref.df) 
        if end_frame < total_len - 1:
            slope_out = full_data[end_frame+1] - full_data[end_frame]
        self.spline_boundary_slopes = (slope_in, slope_out)
        
        self.spline_anchors_x = np.linspace(start_frame, end_frame, num_anchors).astype(int)
        
        # [FIX]: 获取锚点数据时也要用 get_channel_data
        self.spline_anchors_y = full_data[self.spline_anchors_x] 
        self.update_spline_visuals()

    def update_spline_visuals(self):
        spots = [{'pos': (x, y), 'data': i} for i, (x, y) in enumerate(zip(self.spline_anchors_x, self.spline_anchors_y))]
        self.scatter_item.setData(spots=spots)
        
        if len(self.spline_anchors_x) >= 2:
            try:
                cs = CubicSpline(self.spline_anchors_x, self.spline_anchors_y, 
                                 bc_type=((1, self.spline_boundary_slopes[0]), (1, self.spline_boundary_slopes[1])))
                x_new = np.arange(self.spline_anchors_x[0], self.spline_anchors_x[-1] + 1)
                y_new = cs(x_new)
                self.spline_preview_curve.setData(x_new, y_new)
            except: pass

    def apply_spline_to_data(self):
        if not self.spline_mode_active: return
        self.backend_ref.snapshot()
        
        x_data, y_data = self.spline_preview_curve.getData()
        if x_data is None: return
        
        col = self.selected_joint_idx
        start = int(x_data[0])
        end = int(x_data[-1])
        
        # [FIX]: 使用 set_channel_data 写入数据，自动处理索引偏移
        self.backend_ref.set_channel_data(col, y_data, start, end)
        
        # 实时清洗四元数 (Root Rotation 是 3,4,5)
        # 这里的 col 是 UI 索引，3-5 对应欧拉角，Backend 内部会处理转四元数
        if col in [3, 4, 5]: 
             # set_channel_data 内部已经处理了欧拉转四元数，通常不需要额外清洗，
             # 但为了保险起见，可以调用全局清洗
             pass 
            
        self.backend_ref.modified_frames.update(range(start, end+1))
        self.cancel_spline_mode()
        self.update_curves([self.selected_joint_idx])
        curr = int(self.current_frame_line.value())
        self.backend_ref.set_frame(curr)

    def cancel_spline_mode(self):
        self.spline_mode_active = False
        self.spline_boundary_slopes = None
        self.region.setMovable(True)
        self.scatter_item.clear()
        self.spline_preview_curve.setData([])
        self.dragged_anchor_index = None

    # === Interaction Logic ===
    def mousePressEvent(self, ev):
        # 1. 样条模式点击
        if self.spline_mode_active:
            if ev.button() == Qt.LeftButton:
                pos = self.plotItem.vb.mapSceneToView(ev.pos())
                x_tol = (self.viewRange()[0][1] - self.viewRange()[0][0]) * 0.02
                y_tol = (self.viewRange()[1][1] - self.viewRange()[1][0]) * 0.05
                for i, (ax, ay) in enumerate(zip(self.spline_anchors_x, self.spline_anchors_y)):
                    if abs(ax - pos.x()) < x_tol and abs(ay - pos.y()) < y_tol:
                        self.dragged_anchor_index = i
                        ev.accept()
                        return
                ev.accept() # 拦截未命中的左键
            return
        
        # 2. 软拖拽模式 (Ctrl+Click)
        if (ev.modifiers() & Qt.ControlModifier) and self.selected_joint_idx is not None and ev.button() == Qt.LeftButton:
            self.is_editing = True
            ev.accept()
            self.backend_ref.snapshot()
            
            widget_point = ev.pos()
            scene_point = self.mapToScene(widget_point)
            mouse_point = self.plotItem.vb.mapSceneToView(scene_point)
            
            self.drag_start_pos = mouse_point.y()
            click_x = mouse_point.x()
            
            r_min, r_max = self.region.getRegion()
            total_len = len(self.backend_ref.df)
            col = self.selected_joint_idx
            
            # [FIX]: 获取当前通道的全量数据用于计算 delta
            full_data = self.backend_ref.get_channel_data(col)
            
            # 判断点击位置
            if r_min <= click_x <= r_max:
                # 局部模式
                s_local, e_local = int(r_min), int(r_max)
                s_local = max(0, s_local); e_local = min(total_len-1, e_local)
                region_len = e_local - s_local
                # [FIX]: 使用从 get_channel_data 获取的 full_data
                self.drag_start_data = full_data[s_local:e_local+1].copy()
                if region_len > 0:
                    normalized_x = (click_x - s_local) / region_len
                    if normalized_x < 0.2:
                        self.drag_mode = "LOCAL_LEFT"
                        if self.main_window_ref: self.main_window_ref.status_bar.showMessage("模式: 局部左侧调节")
                    elif normalized_x > 0.8:
                        self.drag_mode = "LOCAL_RIGHT"
                        if self.main_window_ref: self.main_window_ref.status_bar.showMessage("模式: 局部右侧调节")
                    else:
                        self.drag_mode = "LOCAL_CENTER"
                        if self.main_window_ref: self.main_window_ref.status_bar.showMessage("模式: 局部整体调节")
            else:
                # 全局模式
                global_threshold = total_len * 0.05
                if click_x < global_threshold:
                    self.drag_mode = "GLOBAL_START"
                    if self.main_window_ref: self.main_window_ref.status_bar.showMessage("模式: 全局起点调节")
                    self.drag_start_data = full_data.copy()
                elif click_x > (total_len - global_threshold):
                    self.drag_mode = "GLOBAL_END"
                    if self.main_window_ref: self.main_window_ref.status_bar.showMessage("模式: 全局终点调节")
                    self.drag_start_data = full_data.copy()
                else:
                    self.is_editing = False
                    ev.ignore()
                    return
            
            self.region.setMovable(False)
        else:
            if not self.spline_mode_active: self.region.setMovable(True)
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        widget_point = ev.pos()
        scene_point = self.mapToScene(widget_point)
        pos = self.plotItem.vb.mapSceneToView(scene_point)
        
        # Spline Drag
        if self.spline_mode_active:
            if self.dragged_anchor_index is not None:
                self.spline_anchors_y[self.dragged_anchor_index] = pos.y()
                self.update_spline_visuals()
                ev.accept()
                return
            else:
                # 让父类处理平移缩放
                super().mouseMoveEvent(ev)
                return

        # Soft Drag
        if self.is_editing:
            if not (ev.buttons() & Qt.LeftButton):
                self.is_editing = False; self.drag_start_data = None; self.region.setMovable(True); super().mouseMoveEvent(ev); return
            
            ev.accept()
            y_curr = pos.y()
            delta_y = y_curr - self.drag_start_pos
            col = self.selected_joint_idx

            if "LOCAL" in self.drag_mode:
                r_min, r_max = self.region.getRegion()
                s, e = int(r_min), int(r_max)
                s = max(0, s); e = min(len(self.backend_ref.df)-1, e)
                
                if s < e and self.drag_start_data is not None:
                    length = e - s + 1
                    x = np.linspace(0, 1, length)
                    
                    if self.drag_mode == "LOCAL_CENTER": weights = (1 - np.cos(2 * np.pi * x)) / 2
                    elif self.drag_mode == "LOCAL_LEFT": weights = (1 + np.cos(np.pi * x)) / 2
                    elif self.drag_mode == "LOCAL_RIGHT": weights = (1 - np.cos(np.pi * x)) / 2
                    else: weights = np.ones_like(x)

                    new_values = self.drag_start_data + (delta_y * weights)
                    # [FIX]: 使用 set_channel_data 而不是直接修改 df.iloc
                    self.backend_ref.set_channel_data(col, new_values, s, e)
                    self.backend_ref.modified_frames.update(range(s, e+1))
            
            elif "GLOBAL" in self.drag_mode:
                total_len = len(self.backend_ref.df)
                x = np.linspace(0, 1, total_len)
                if self.drag_mode == "GLOBAL_START": weights = 1 - x
                elif self.drag_mode == "GLOBAL_END": weights = x
                if self.drag_start_data is not None:
                    new_values = self.drag_start_data + (delta_y * weights)
                    # [FIX]: 使用 set_channel_data 而不是直接修改 df.iloc
                    self.backend_ref.set_channel_data(col, new_values, 0, total_len-1)
                    self.backend_ref.modified_frames.update(range(0, total_len))

            # === 刷新逻辑 ===
            # [FIX]: 重新获取数据来刷新曲线，确保看到的是 Backend 处理后的结果（尤其是 Root 旋转）
            updated_data = self.backend_ref.get_channel_data(col)
            if self.selected_joint_idx in self.curves:
                self.curves[self.selected_joint_idx].setData(updated_data)
            
            # 刷新机器人
            curr_f = int(self.current_frame_line.value())
            self.backend_ref.set_frame(curr_f)
            if self.main_window_ref: self.main_window_ref.mujoco_widget.update()
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

    def update_audio_visuals(self):
        """当后端加载音频后调用"""
        if self.backend_ref and self.backend_ref.audio_waveform is not None:
            # 绘制波形
            self.audio_curve.setData(
                self.backend_ref.audio_frames, 
                self.backend_ref.audio_waveform
            )
        else:
            self.audio_curve.setData([], [])