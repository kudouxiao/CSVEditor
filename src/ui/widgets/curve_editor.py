import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from scipy.interpolate import CubicSpline

class CurveEditor(pg.PlotWidget):
    """
    CurveEditor v3.2: 全局端点修正 + 局部软选择
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('#1e1e1e')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setMouseEnabled(x=True, y=False)
        self.getPlotItem().setLabel('bottom', 'Frame Index')
        self.getPlotItem().setLabel('left', 'Value')
        
        # 1. 选区 (Layer 10)
        self.region = pg.LinearRegionItem([0, 100], brush=(50, 50, 200, 50))
        self.region.setZValue(10)
        self.addItem(self.region)
        
        # 2. 当前帧线 (Layer 100)
        self.current_frame_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#FF5555', width=2), movable=True)
        self.current_frame_line.setZValue(100)
        self.addItem(self.current_frame_line)
        
        # 3. Ghost Curve
        self.ghost_curve = self.plot([], pen=pg.mkPen((100, 100, 100), width=1, style=Qt.DashLine))
        self.ghost_curve.setZValue(0)
        self.show_ghost = False
        
        # 4. Spline Anchors
        self.scatter_item = pg.ScatterPlotItem(size=15, pen=pg.mkPen('w'), brush=pg.mkBrush('#FFCC00'))
        self.scatter_item.setZValue(200)
        self.addItem(self.scatter_item)
        self.spline_preview_curve = self.plot([], pen=pg.mkPen('#FFCC00', width=2, style=Qt.DashLine))
        self.spline_preview_curve.setZValue(190)

        # === 3. 限位线 (新增) ===
        # 红色虚线，用于指示物理极限
        limit_pen = pg.mkPen('#FF3333', width=1.5, style=Qt.DashLine)
        self.limit_upper = pg.InfiniteLine(angle=0, pen=limit_pen, movable=False)
        self.limit_lower = pg.InfiniteLine(angle=0, pen=limit_pen, movable=False)
        self.limit_upper.setZValue(50)
        self.limit_lower.setZValue(50)
        self.addItem(self.limit_upper)
        self.addItem(self.limit_lower)
        # 默认隐藏
        self.limit_upper.hide()
        self.limit_lower.hide()
        
        self.curves = {}
        self.selected_joint_idx = None
        self.backend_ref = None
        self.main_window_ref = None
        
        # 交互状态
        self.is_editing = False 
        self.drag_mode = "LOCAL_CENTER" # GLOBAL_START, GLOBAL_END, LOCAL_LEFT, LOCAL_RIGHT, LOCAL_CENTER
        self.spline_mode_active = False 
        self.drag_start_pos = None
        self.drag_start_data = None
        
        # Spline 数据
        self.spline_anchors_x = []
        self.spline_anchors_y = []
        self.dragged_anchor_index = None
        self.spline_boundary_slopes = None

    def set_backend(self, backend, main_window):
        self.backend_ref = backend
        self.main_window_ref = main_window
        self.current_frame_line.sigDragged.connect(self.on_line_dragged)
        self.region.sigRegionChanged.connect(main_window.on_region_changed)

    def set_ghost_visible(self, visible):
        self.show_ghost = visible
        if self.selected_joint_idx is not None:
            self.update_curves([self.selected_joint_idx])

    def update_curves(self, selected_indices):
        for item in self.curves.values(): self.removeItem(item)
        self.curves.clear()
        
        if not self.backend_ref or self.backend_ref.df is None: return
        
        if len(selected_indices) > 0:
            self.selected_joint_idx = selected_indices[0]
            
            # === 更新限位显示 ===
            # 调用后端获取限位
            limits = self.backend_ref.get_joint_limits(self.selected_joint_idx)
            
            if limits is not None:
                min_val, max_val = limits
                self.limit_lower.setPos(min_val)
                self.limit_upper.setPos(max_val)
                self.limit_lower.show()
                self.limit_upper.show()
                
                # 可选：如果在样条模式，还可以添加限位标签文本
            else:
                self.limit_lower.hide()
                self.limit_upper.hide()
            # ====================

            # 绘制 Ghost
            if self.show_ghost:
                col = self.selected_joint_idx
                orig_data = self.backend_ref.df_orig.iloc[:, col].values
                self.ghost_curve.setData(orig_data)
            else:
                self.ghost_curve.setData([])

            # 绘制背景线
            for idx in selected_indices[1:]:
                col = idx; data = self.backend_ref.df.iloc[:, col].values
                curve = self.plot(data, pen=pg.mkPen((80, 80, 80), width=1)); curve.setZValue(5); self.curves[idx] = curve
            
            # 绘制主线
            col = self.selected_joint_idx; data = self.backend_ref.df.iloc[:, col].values
            main_curve = self.plot(data, pen=pg.mkPen('#00ffff', width=3)); main_curve.setZValue(20); self.curves[self.selected_joint_idx] = main_curve
            
            # 自动调整视野，确保限位线也在视野内 (如果开启)
            self.autoRange()
            # 稍微扩展 Y 轴以看清限位
            if limits is not None:
                yr = self.viewRange()[1]
                min_y, max_y = yr[0], yr[1]
                # 确保当前视图包含限位
                new_min = min(min_y, limits[0] - 0.2)
                new_max = max(max_y, limits[1] + 0.2)
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

    # === Spline Mode Logic ===
    def start_spline_mode(self, start_frame, end_frame, num_anchors=5):
        if self.selected_joint_idx is None: return
        self.spline_mode_active = True
        self.region.setMovable(False)
        col = self.selected_joint_idx
        
        slope_in = 0.0; slope_out = 0.0
        if start_frame > 0:
            slope_in = self.backend_ref.df.iloc[start_frame, col] - self.backend_ref.df.iloc[start_frame-1, col]
        total_len = len(self.backend_ref.df)
        if end_frame < total_len - 1:
            slope_out = self.backend_ref.df.iloc[end_frame+1, col] - self.backend_ref.df.iloc[end_frame, col]
        self.spline_boundary_slopes = (slope_in, slope_out)
        
        self.spline_anchors_x = np.linspace(start_frame, end_frame, num_anchors).astype(int)
        self.spline_anchors_y = self.backend_ref.df.iloc[self.spline_anchors_x, col].values
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
        
        self.backend_ref.df.iloc[start:end+1, col] = y_data
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

    # === Interaction Logic (核心修改部分) ===
    def mousePressEvent(self, ev):
        # 1. 样条模式点击
        if self.spline_mode_active and ev.button() == Qt.LeftButton:
            pos = self.plotItem.vb.mapSceneToView(ev.pos())
            x_tol = (self.viewRange()[0][1] - self.viewRange()[0][0]) * 0.02
            y_tol = (self.viewRange()[1][1] - self.viewRange()[1][0]) * 0.05
            for i, (ax, ay) in enumerate(zip(self.spline_anchors_x, self.spline_anchors_y)):
                if abs(ax - pos.x()) < x_tol and abs(ay - pos.y()) < y_tol:
                    self.dragged_anchor_index = i
                    ev.accept()
                    return
        
        # 2. 软拖拽模式 (Ctrl+Click)
        if not self.spline_mode_active and (ev.modifiers() & Qt.ControlModifier) and self.selected_joint_idx is not None and ev.button() == Qt.LeftButton:
            self.is_editing = True
            ev.accept()
            self.backend_ref.snapshot()
            
            # 坐标转换
            widget_point = ev.pos()
            scene_point = self.mapToScene(widget_point)
            mouse_point = self.plotItem.vb.mapSceneToView(scene_point)
            self.drag_start_pos = mouse_point.y()
            click_x = mouse_point.x()
            
            # 获取范围信息
            r_min, r_max = self.region.getRegion()
            s_local, e_local = int(r_min), int(r_max)
            total_len = len(self.backend_ref.df)
            col = self.selected_joint_idx
            
            # === 判断全局/局部模式 ===
            # 判定阈值：总长度的 5%
            global_threshold = total_len * 0.05
            
            if click_x < global_threshold:
                # 点击了整段曲线的开头 -> 全局左端点调节
                self.drag_mode = "GLOBAL_START"
                if self.main_window_ref: self.main_window_ref.status_bar.showMessage("模式: 全局起点调节 (终点固定)")
                # 备份整列数据
                self.drag_start_data = self.backend_ref.df.iloc[:, col].values.copy()
                
            elif click_x > (total_len - global_threshold):
                # 点击了整段曲线的结尾 -> 全局右端点调节
                self.drag_mode = "GLOBAL_END"
                if self.main_window_ref: self.main_window_ref.status_bar.showMessage("模式: 全局终点调节 (起点固定)")
                # 备份整列数据
                self.drag_start_data = self.backend_ref.df.iloc[:, col].values.copy()
                
            else:
                # 局部选区模式 (原有逻辑)
                # 进一步细分局部模式
                s_local = max(0, s_local); e_local = min(total_len-1, e_local)
                region_len = e_local - s_local
                
                # 备份选区数据
                self.drag_start_data = self.backend_ref.df.iloc[s_local:e_local+1, col].values.copy()
                
                # 判断在选区内的相对位置
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
            
            self.region.setMovable(False)
        else:
            if not self.spline_mode_active: self.region.setMovable(True)
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        widget_point = ev.pos()
        scene_point = self.mapToScene(widget_point)
        pos = self.plotItem.vb.mapSceneToView(scene_point)
        
        # Spline Drag
        if self.spline_mode_active and self.dragged_anchor_index is not None:
            self.spline_anchors_y[self.dragged_anchor_index] = pos.y()
            self.update_spline_visuals()
            ev.accept()
            return

        # Soft Drag
        if self.is_editing:
            if not (ev.buttons() & Qt.LeftButton): # 安全检查
                self.is_editing = False; self.drag_start_data = None; self.region.setMovable(True); super().mouseMoveEvent(ev); return
            
            ev.accept()
            y_curr = pos.y()
            delta_y = y_curr - self.drag_start_pos
            col = self.selected_joint_idx
            
            # === 根据模式应用修改 ===
            
            if "GLOBAL" in self.drag_mode:
                # 全局模式：操作整个 DataFrame 列
                total_len = len(self.backend_ref.df)
                x = np.linspace(0, 1, total_len)
                
                if self.drag_mode == "GLOBAL_START":
                    # 起点动，终点不动 (线性衰减)
                    weights = 1 - x
                elif self.drag_mode == "GLOBAL_END":
                    # 起点不动，终点动 (线性增加)
                    weights = x
                
                if self.drag_start_data is not None:
                    new_values = self.drag_start_data + (delta_y * weights)
                    self.backend_ref.df.iloc[:, col] = new_values
                    self.backend_ref.modified_frames.update(range(0, total_len))
            
            else:
                # 局部模式：只操作选区
                r_min, r_max = self.region.getRegion()
                s, e = int(r_min), int(r_max)
                s = max(0, s); e = min(len(self.backend_ref.df)-1, e)
                
                if s < e and self.drag_start_data is not None:
                    length = e - s + 1
                    x = np.linspace(0, 1, length)
                    
                    if self.drag_mode == "LOCAL_CENTER":
                        # 钟形曲线
                        weights = (1 - np.cos(2 * np.pi * x)) / 2
                    elif self.drag_mode == "LOCAL_LEFT":
                        # 左动右不动
                        weights = (1 + np.cos(np.pi * x)) / 2
                    elif self.drag_mode == "LOCAL_RIGHT":
                        # 右动左不动
                        weights = (1 - np.cos(np.pi * x)) / 2
                    else:
                        weights = np.ones_like(x)

                    new_values = self.drag_start_data + (delta_y * weights)
                    self.backend_ref.df.iloc[s:e+1, col] = new_values
                    self.backend_ref.modified_frames.update(range(s, e+1))

            # 刷新曲线
            self.curves[self.selected_joint_idx].setData(self.backend_ref.df.iloc[:, col].values)
            
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