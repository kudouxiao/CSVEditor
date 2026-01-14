# G1 Motion Refinement Tool (G1 Pro Editor)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/Render-MuJoCo-orange)](https://mujoco.org/)

A professional-grade GUI tool designed for refining, editing, and visualizing motion capture data for the **Unitree G1 Humanoid Robot**.

This tool bridges the gap between raw retargeted motion data and physical simulation. It features an embedded **MuJoCo** viewer, a non-linear **Curve Editor**, and **Reference Motion Overlay** (SMPL-X / BVH), allowing for precise "additive editing" of joint trajectories while maintaining original motion details.

---

## ✨ Key Features

### 1. Embedded 3D Visualization
*   **Real-time Rendering**: Powered by MuJoCo 3.x with high-performance OpenGL widgets.
*   **Visual Debugging**: Toggle visibility for Joints, Actuators, Contact Forces, Center of Mass (CoM), and Inertia boxes.
*   **Rendering Modes**: Supports Shadows, Wireframe, Transparency, and Fog.

### 2. Multi-Format Reference Overlay
*   **SMPL-X Support**: Load `.npz` files to visualize the ground truth human motion.
*   **BVH Support**: Load standard `.bvh` motion files.
*   **Auto-Alignment**: Automatic centering and offset adjustment.
*   **Sync Control**: Adjustable frame rate scaling to synchronize reference motion with robot data.

### 3. Professional Curve Editor
*   **Non-Linear Editing**:
    *   **Soft Drag**: Gaussian-weighted dragging (Ctrl+Click) for organic motion modification.
    *   **Spline Mode**: Edit trajectories using Cubic Splines with tangent clamping for perfect continuity.
*   **Ghosting**: View the original curve (Ghost) behind your edits for comparison.
*   **Root Motion**: Full support for editing Root Position (X/Y/Z) and Orientation (Quaternions).

### 4. Advanced Batch Tools
*   **Additive Interpolation**: Modify the overall trend of a motion segment without destroying high-frequency details (jitter/vibration).
*   **Smart Connect**: Linear or Sigmoid (S-Curve) connections between keyframes.
*   **Smoothing**: Savitzky-Golay filtering for noise reduction.
*   **Undo/Redo**: Full history stack (Ctrl+Z / Ctrl+Y).

---

## 🧭 UI Overview & Controls

### Top Bar
- **↩ 撤销 (Undo)**: Undo the last edit (also **Ctrl+Z**).
- **↪ 重做 (Redo)**: Redo the last undone edit (also **Ctrl+Y**).
- **👻 显示原数据 (Ghost)**: Toggle display of the original unedited curve in the curve editor.
- **💾 另存为 (Save As)**: Export the current motion to a new CSV file.
- **🕺 加载参考 (Load Reference)**: Load an external SMPL-X `.npz` reference file.

### Playback Controls (below viewer)
- **◀**: Step to the previous frame.
- **▶ 播放 / 暂停**: Play or pause the animation (also **Space** key).
- **▶**: Step to the next frame.
- **Frame Label**: Shows the current frame index.

### Curve Editor & Channel List
- **通道列表 (Channel List)**: Click an entry to select which channel to edit; root channels are highlighted in cyan.
- **编辑范围 (Edit Range)**: `Start` / `End` spin boxes define the global edit region; they stay in sync with the blue region in the curve editor.
- **Time Scrub**: Drag the red vertical line to preview a specific frame in both the plot and 3D viewer.
- **Soft Edit (Ctrl+Click + Drag)**:
  - Click near the very start/end of the timeline for **GLOBAL_START / GLOBAL_END** modes.
  - Click inside the selected region for **LOCAL_LEFT / LOCAL_RIGHT / LOCAL_CENTER** modes with smooth weighting.

### Tools Tab (🛠️ 工具)
- **📏 直线连接 (Linear)**: Connect start and end of the selected range with a straight line.
- **🌊 S形连接 (Sigmoid)**: Connect with a smooth S-curve.
- **💧 SavGol 平滑**: Apply Savitzky–Golay smoothing to the selected range.
- **✨ 叠加插值 (Additive)**: Perform additive interpolation relative to the original curve.
- **🔄 重置选中区域 (Reset)**: Restore the selected range back to the original data.

### Spline Tab (✏️ 样条)
- **锚点数 (Anchor Count)**: Controls how many spline control points are used.
- **✏️ 开始编辑 (Start Editing)**: Enters spline mode and creates movable anchors.
- **✅ 应用 (Apply)**: Applies the preview spline to the data.
- **❌ 取消 (Cancel)**: Cancels spline editing and restores the original curve.

### View Tab (👀 视图)
- **Reference Settings**:
  - **Show Reference**: Show/hide SMPL-X / BVH reference skeleton.
  - **Scale**: Adjusts temporal scale between robot and reference motion.
  - **X / Y Offset**: Shifts reference skeleton in world space.
- **Render Flags**: Checkboxes for Joints, Actuators, Constraints, CoM, Transparency, Convex Hull, Inertia, Shadows, Wireframe, Reflection, and Fog.

### 3D Viewer Mouse Controls
- **Left Drag**: Orbit the camera around the robot.
- **Right Drag**: Pan the camera.
- **Middle Drag / Mouse Wheel**: Zoom smoothly in or out.
- **Double-click**: Re-focus the camera on the robot root.

## 🛠️ Installation

### Prerequisites
*   Python 3.8 or higher.
*   (Optional) CUDA for accelerated PyTorch/SMPL-X operations.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/G1_Editor_Project.git
cd G1_Editor_Project