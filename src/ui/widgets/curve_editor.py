import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from scipy.interpolate import CubicSpline, PchipInterpolator

class CurveEditor(pg.PlotWidget):
    """
    CurveEditor v3.0: 集成 Ghost, Spline Editing, Tangent Clamp, Soft Drag
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
        
        # 3. Ghost Curve (Layer 0, 灰色虚线) - 新增
        self.ghost_curve = self.plot([], pen=pg.mkPen((100, 100, 100), width=1, style=Qt.DashLine))
        self.ghost_curve.setZValue(0)
        self.show_ghost = False
        
        # 4. Spline Anchors (Layer 200) - 新增
        self.scatter_item = pg.ScatterPlotItem(size=15, pen=pg.mkPen('w'), brush=pg.mkBrush('#FFCC00'))
        self.scatter_item.setZValue(200)
        self.addItem(self.scatter_item)
        self.spline_preview_curve = self.plot([], pen=pg.mkPen('#FFCC00', width=2, style=Qt.DashLine))
        self.spline_preview_curve.setZValue(190)
        
        self.curves = {}
        self.selected_joint_idx = None
        self.backend_ref = None
        self.main_window_ref = None
        
        # 交互状态
        self.is_editing = False # 软拖拽模式
        self.spline_mode_active = False # 样条编辑模式
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
        # 绑定选区变化信号 (需要在 MainWindow 里连接槽函数)
        self.region.sigRegionChanged.connect(main_window.on_region_changed)

    def set_ghost_visible(self, visible):
        self.show_ghost = visible
        # 刷新显示
        if self.selected_joint_idx is not None:
            self.update_curves([self.selected_joint_idx])

    def update_curves(self, selected_indices):
        for item in self.curves.values(): self.removeItem(item)
        self.curves.clear()
        
        if not self.backend_ref or self.backend_ref.df is None: return
        
        if len(selected_indices) > 0:
            self.selected_joint_idx = selected_indices[0]
            
            # 绘制 Ghost (原始数据)
            if self.show_ghost:
                col = self.selected_joint_idx
                orig_data = self.backend_ref.df_orig.iloc[:, col].values
                self.ghost_curve.setData(orig_data)
            else:
                self.ghost_curve.setData([])

            # 绘制背景线
            for idx in selected_indices[1:]:
                col = idx
                data = self.backend_ref.df.iloc[:, col].values
                curve = self.plot(data, pen=pg.mkPen((80, 80, 80), width=1))
                curve.setZValue(5)
                self.curves[idx] = curve
            
            # 绘制主编辑线
            col = self.selected_joint_idx
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

    # === Spline Logic ===
    def start_spline_mode(self, start_frame, end_frame, num_anchors=5):
        if self.selected_joint_idx is None: return
        self.spline_mode_active = True
        self.region.setMovable(False)
        col = self.selected_joint_idx
        
        # 计算边界切线 (保持连续性)
        slope_in = 0.0; slope_out = 0.0
        if start_frame > 0:
            slope_in = self.backend_ref.df.iloc[start_frame, col] - self.backend_ref.df.iloc[start_frame-1, col]
        total_len = len(self.backend_ref.df)
        if end_frame < total_len - 1:
            slope_out = self.backend_ref.df.iloc[end_frame+1, col] - self.backend_ref.df.iloc[end_frame, col]
        self.spline_boundary_slopes = (slope_in, slope_out)
        
        # 生成锚点
        self.spline_anchors_x = np.linspace(start_frame, end_frame, num_anchors).astype(int)
        self.spline_anchors_y = self.backend_ref.df.iloc[self.spline_anchors_x, col].values
        self.update_spline_visuals()

    def update_spline_visuals(self):
        spots = [{'pos': (x, y), 'data': i} for i, (x, y) in enumerate(zip(self.spline_anchors_x, self.spline_anchors_y))]
        self.scatter_item.setData(spots=spots)
        
        if len(self.spline_anchors_x) >= 2:
            try:
                # 使用 CubicSpline 施加边界约束
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
        # 更新机器人姿态
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
        # 1. Spline Anchor Click
        if self.spline_mode_active and ev.button() == Qt.LeftButton:
            pos = self.plotItem.vb.mapSceneToView(ev.pos())
            # 简单的命中检测
            x_tol = (self.viewRange()[0][1] - self.viewRange()[0][0]) * 0.02
            y_tol = (self.viewRange()[1][1] - self.viewRange()[1][0]) * 0.05
            for i, (ax, ay) in enumerate(zip(self.spline_anchors_x, self.spline_anchors_y)):
                if abs(ax - pos.x()) < x_tol and abs(ay - pos.y()) < y_tol:
                    self.dragged_anchor_index = i
                    ev.accept()
                    return
        
        # 2. Soft Drag Mode (Ctrl+Click)
        if not self.spline_mode_active and (ev.modifiers() & Qt.ControlModifier) and self.selected_joint_idx is not None and ev.button() == Qt.LeftButton:
            self.is_editing = True
            ev.accept()
            self.backend_ref.snapshot()
            
            # PyQt5 坐标转换
            widget_point = ev.pos()
            scene_point = self.mapToScene(widget_point)
            mouse_point = self.plotItem.vb.mapSceneToView(scene_point)
            
            self.drag_start_pos = mouse_point.y()
            r_min, r_max = self.region.getRegion()
            s, e = int(r_min), int(r_max)
            s = max(0, s); e = min(len(self.backend_ref.df)-1, e)
            col = self.selected_joint_idx
            self.drag_start_data = self.backend_ref.df.iloc[s:e+1, col].values.copy()
            self.region.setMovable(False)
        else:
            if not self.spline_mode_active: self.region.setMovable(True)
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        # 坐标转换
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
            
            r_min, r_max = self.region.getRegion()
            s, e = int(r_min), int(r_max)
            s = max(0, s); e = min(len(self.backend_ref.df)-1, e)
            
            if s < e and self.drag_start_data is not None:
                length = e - s + 1
                x = np.linspace(-np.pi, np.pi, length)
                weights = (np.cos(x) + 1) / 2 
                new_values = self.drag_start_data + (delta_y * weights)
                
                col = self.selected_joint_idx
                self.backend_ref.df.iloc[s:e+1, col] = new_values
                self.backend_ref.modified_frames.update(range(s, e+1))
                
                self.curves[self.selected_joint_idx].setData(self.backend_ref.df.iloc[:, col].values)
                
                curr_f = int(self.current_frame_line.value())
                if s <= curr_f <= e: 
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