# G1 Motion Refinement Tool (G1 Pro Editor)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
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

## 🛠️ Installation

### Prerequisites
*   Python 3.8 or higher.
*   (Optional) CUDA for accelerated PyTorch/SMPL-X operations.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/G1_Editor_Project.git
cd G1_Editor_Project