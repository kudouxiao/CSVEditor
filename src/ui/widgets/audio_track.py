import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtWidgets import QMenu, QAction
from src.config import ROBOT_FPS

# 复用之前的自定义坐标轴，或者直接使用默认
class TimeAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [f"{int(v)}\n{v/ROBOT_FPS:.2f}s" for v in values]

class AudioTrack(pg.PlotWidget):
    """
    AudioTrack v2: 支持波形偏移、双向同步、打点标记
    """
    offset_changed = pyqtSignal(float) # 偏移量改变信号
    frame_changed = pyqtSignal(int)    # 红线拖动信号 (用于同步上方)

    def __init__(self, parent=None):
        axis = TimeAxisItem(orientation='bottom')
        super().__init__(parent, axisItems={'bottom': axis})
        
        self.setBackground('#121212')
        self.showGrid(x=True, y=False, alpha=0.3)
        self.setMouseEnabled(x=True, y=False)
        self.getPlotItem().setLabel('left', 'Audio')
        self.setYRange(-1, 1)
        self.setFixedHeight(150) # 稍微加高一点方便看标记
        
        # 1. 波形曲线
        self.curve = self.plot([], pen=pg.mkPen('#00BFFF', width=1), fillLevel=0, brush=(0, 191, 255, 80))
        
        # 2. 当前帧指示线 (改为可拖动)
        # 设为 movable=True，这样在音轨上也能拖动进度
        self.current_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#FF5555', width=2), movable=True)
        self.current_line.setZValue(100)
        self.current_line.sigDragged.connect(self.on_line_dragged) # 绑定拖动事件
        self.addItem(self.current_line)
        
        # 3. 标记列表
        self.markers = []
        
        self.backend_ref = None
        self.drag_start_x = None
        self.drag_start_offset = 0.0

    def set_backend(self, backend):
        self.backend_ref = backend

    def on_line_dragged(self):
        """当用户拖动音轨上的红线时"""
        idx = int(self.current_line.value())
        self.frame_changed.emit(idx) # 发送信号给 MainWindow

    def update_waveform(self):
        if not self.backend_ref or self.backend_ref.audio_data is None: return
        y = self.backend_ref.audio_data
        sr = self.backend_ref.audio_sr
        # 降采样
        total_frames = int(len(y) / sr * ROBOT_FPS)
        step = max(1, len(y) // (total_frames * 10))
        y_ds = y[::step]
        max_val = np.max(np.abs(y_ds))
        if max_val > 0: y_ds = y_ds / max_val
        x_axis = np.linspace(0, len(y)/sr * ROBOT_FPS, len(y_ds))
        self.curve.setData(x_axis, y_ds)
        self.update_position()

    def update_position(self):
        if self.backend_ref:
            shift_frames = self.backend_ref.audio_offset * ROBOT_FPS
            self.curve.setPos(shift_frames, 0)

    # === 打点标记功能 (Markers) ===
    def add_marker(self, x_pos):
        """在指定位置添加一个绿色标记线"""
        # 标记线：绿色，可移动
        marker = pg.InfiniteLine(pos=x_pos, angle=90, pen=pg.mkPen('#00FF00', width=1.5, style=Qt.DashLine), movable=True)
        marker.setZValue(50)
        
        # 给标记线绑定右键菜单（删除功能）
        # pyqtgraph 的 InfiniteLine 没有直接的右键信号，我们需要稍微 hack 一下或者利用 PlotWidget 的 scene 事件
        # 这里为了简单，我们利用 marker 的 sigPositionChangeFinished 来做简单的交互，或者在 sceneEventFilter 处理
        # 更简单的方案：如果用户按住 Alt 点击，或者右键点击 ViewBox
        
        self.addItem(marker)
        self.markers.append(marker)
        return marker

    def remove_marker(self, marker):
        if marker in self.markers:
            self.removeItem(marker)
            self.markers.remove(marker)

    # === 交互事件处理 ===
    def mousePressEvent(self, ev):
        # 1. Shift + 左键：添加标记
        if ev.modifiers() & Qt.ShiftModifier and ev.button() == Qt.LeftButton:
            ev.accept()
            pos = self.plotItem.vb.mapSceneToView(ev.pos())
            self.add_marker(pos.x())
            return

        # 2. Ctrl + 左键：拖动波形 (Offset)
        if ev.modifiers() & Qt.ControlModifier and ev.button() == Qt.LeftButton:
            ev.accept()
            pos = self.plotItem.vb.mapSceneToView(ev.pos())
            self.drag_start_x = pos.x()
            self.drag_start_offset = self.backend_ref.audio_offset
            return
            
        # 3. 右键点击：检查是否点中了标记 (用于删除)
        if ev.button() == Qt.RightButton:
            pos = self.plotItem.vb.mapSceneToView(ev.pos())
            click_x = pos.x()
            # 查找附近的标记
            x_range = self.viewRange()[0]
            tol = (x_range[1] - x_range[0]) * 0.01 # 1% 的容差
            
            for m in self.markers:
                if abs(m.value() - click_x) < tol:
                    self.remove_marker(m)
                    ev.accept()
                    return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        # 处理波形拖动
        if self.drag_start_x is not None:
            ev.accept()
            pos = self.plotItem.vb.mapSceneToView(ev.pos())
            dx_frames = pos.x() - self.drag_start_x
            dt_seconds = dx_frames / ROBOT_FPS
            new_offset = self.drag_start_offset + dt_seconds
            self.backend_ref.audio_offset = new_offset
            self.update_position()
            self.offset_changed.emit(new_offset)
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self.drag_start_x is not None:
            self.drag_start_x = None
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)