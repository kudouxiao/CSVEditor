import pyqtgraph as pg
from PyQt5.QtCore import Qt

class CurveEditor(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('#1e1e1e'); self.showGrid(x=True, y=True, alpha=0.3)
        self.setMouseEnabled(x=True, y=False)
        self.region = pg.LinearRegionItem([0, 100], brush=(50, 50, 200, 50)); self.region.setZValue(10); self.addItem(self.region)
        self.current_frame_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#FF5555', width=2), movable=True); self.current_frame_line.setZValue(100); self.addItem(self.current_frame_line)
        self.curves = {}; 
        self.selected_joint_idx = None; 
        self.backend_ref = None; self.main_window_ref = None
        self.is_editing = False; 
        self.drag_start_pos = None; 
        self.drag_start_data = None 
    def set_backend(self, backend, main_window):
        self.backend_ref = backend; self.main_window_ref = main_window; self.current_frame_line.sigDragged.connect(self.on_line_dragged)
    def update_curves(self, selected_indices):
        for item in self.curves.values(): self.removeItem(item)
        self.curves.clear()
        if not self.backend_ref or self.backend_ref.df is None: return
        if len(selected_indices) > 0:
            self.selected_joint_idx = selected_indices[0]
            for idx in selected_indices[1:]:
                # 修改：直接使用 idx，不再 +7
                col = idx; data = self.backend_ref.df.iloc[:, col].values
                curve = self.plot(data, pen=pg.mkPen((80, 80, 80), width=1)); curve.setZValue(5); self.curves[idx] = curve
            # 修改：直接使用 selected_joint_idx
            col = self.selected_joint_idx; data = self.backend_ref.df.iloc[:, col].values
            main_curve = self.plot(data, pen=pg.mkPen('#00ffff', width=3)); main_curve.setZValue(20); self.curves[self.selected_joint_idx] = main_curve
            self.autoRange()
        else: self.selected_joint_idx = None
    def on_line_dragged(self):
        idx = int(self.current_frame_line.value()); 
        if self.main_window_ref: self.main_window_ref.update_frame_from_graph(idx)

    def mousePressEvent(self, ev):
        if (ev.modifiers() & Qt.ControlModifier) and self.selected_joint_idx is not None and ev.button() == Qt.LeftButton:
            self.is_editing = True; ev.accept(); self.backend_ref.snapshot()
            widget_point = ev.pos(); scene_point = self.mapToScene(widget_point); mouse_point = self.plotItem.vb.mapSceneToView(scene_point)
            self.drag_start_pos = mouse_point.y()
            r_min, r_max = self.region.getRegion(); s, e = int(r_min), int(r_max)
            s = max(0, s); e = min(len(self.backend_ref.df)-1, e)
            # 修改：col = self.selected_joint_idx
            col = self.selected_joint_idx; self.drag_start_data = self.backend_ref.df.iloc[s:e+1, col].values.copy()
            self.region.setMovable(False)
        else: self.region.setMovable(True); super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.is_editing and not (ev.buttons() & Qt.LeftButton): self.is_editing = False; self.drag_start_data = None; self.region.setMovable(True); super().mouseMoveEvent(ev); return
        if self.is_editing:
            ev.accept(); widget_point = ev.pos(); scene_point = self.mapToScene(widget_point); mouse_point = self.plotItem.vb.mapSceneToView(scene_point)
            y_curr = mouse_point.y(); delta_y = y_curr - self.drag_start_pos
            r_min, r_max = self.region.getRegion(); s, e = int(r_min), int(r_max); s = max(0, s); e = min(len(self.backend_ref.df)-1, e)
            if s < e and self.drag_start_data is not None:
                length = e - s + 1; x = np.linspace(-np.pi, np.pi, length); weights = (np.cos(x) + 1) / 2 
                new_values = self.drag_start_data + (delta_y * weights)
                # 修改：col = self.selected_joint_idx
                col = self.selected_joint_idx; self.backend_ref.df.iloc[s:e+1, col] = new_values
                self.backend_ref.modified_frames.update(range(s, e+1))
                self.curves[self.selected_joint_idx].setData(self.backend_ref.df.iloc[:, col].values)
                curr_f = int(self.current_frame_line.value())
                if s <= curr_f <= e: self.backend_ref.set_frame(curr_f); self.main_window_ref.mujoco_widget.update()
        else: super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self.is_editing: self.is_editing = False; self.drag_start_data = None; self.region.setMovable(True); ev.accept()
        else: super().mouseReleaseEvent(ev)