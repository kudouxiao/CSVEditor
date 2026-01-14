# G1 Motion Refinement Tool

A professional-grade GUI tool designed for refining, editing, and visualizing motion capture data for the **Unitree G1 Humanoid Robot**.

This tool bridges the gap between raw retargeted motion data and physical simulation. It features an embedded **MuJoCo** viewer, a non-linear **Curve Editor**, and **Reference Motion Overlay** (SMPL-X / BVH), allowing for precise "additive editing" of joint trajectories while maintaining original motion details.

---

## ✨ Key Features

*   **Embedded MuJoCo Rendering**: 
    *   Real-time 3D visualization within the application.
    *   Supports Shadows, Wireframe, Transparency, and Contact Force visualization.
    *   Toggle visualization for Joints, Actuators, and Center of Mass (CoM).
*   **Multi-Format Reference Overlay**:
    *   **SMPL-X (.npz)** and **BVH** motion support.
    *   Auto-alignment and "Ghost" skeleton visualization.
    *   Adjustable offset (X/Y/Z) and frame rate scaling for perfect synchronization.
*   **Advanced Curve Editor**:
    *   **Additive Editing**: Modify motion trends without destroying high-frequency details (jitter/vibration).
    *   **Soft Selection**: Gaussian-weighted dragging (Ctrl + Click) for smooth transitions.
    *   **Root Motion Support**: Edit root position and orientation alongside joint angles.
*   **Professional Workflow**:
    *   **Undo/Redo System** (Ctrl+Z / Ctrl+Y) with history stack.
    *   **Batch Tools**: Smoothing, Linear/PCHIP Interpolation, and Global Offsets.
    *   **Timeline Control**: Real-time playback, pause, and frame scrubbing.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.8+
*   CUDA (Optional, for PyTorch acceleration if available)

### Dependencies
Install the required Python packages:

```bash
pip install PyQt5 pyqtgraph mujoco pandas numpy scipy torch smplx qdarktheme