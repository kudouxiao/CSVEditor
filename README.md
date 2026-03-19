# CSVEditor (G1 Pro Editor)

面向 Unitree G1 动作数据的可视化与精修工具。项目提供 MuJoCo 实时预览、曲线编辑、参考动作叠加（SMPL-X / BVH）与音频对齐能力，适用于机器人动作后期调整与拼接。

## 主要功能

- MuJoCo 3D 实时渲染与交互相机控制
- 多通道曲线编辑（含软拖拽、样条编辑、Ghost 对比）
- 批处理工具（平滑、连接、叠加插值、局部重置）
- 参考动作加载（SMPL-X `.npz`、BVH）
- 音频波形显示与动作时间轴同步
- Undo/Redo 与帧级插入/删除

## 项目结构

```text
CSVEditor/
├── main.py
├── requirements.txt
├── assets/
│   ├── unitree_g1/                 # 机器人模型与网格
│   └── body_models/
│       └── smplx/                  # 仅保留目录结构（不包含模型文件）
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── core/
│   │   └── backend.py
│   └── ui/
│       ├── main_window.py
│       └── widgets/
│           ├── mujoco_widget.py
│           ├── curve_editor.py
│           └── audio_track.py
├── bvh/                            # 仅保留目录结构
├── smplx/                          # 仅保留目录结构
└── retargeted/                     # 仅保留目录结构
```

## 环境要求

- Python 3.10（建议，与当前代码环境一致）
- Linux（已在 Conda `gmr` 环境使用）
- 可用 OpenGL 驱动（MuJoCo 渲染需要）

## 安装与运行

### 1. 创建/激活 Conda 环境

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动

```bash
python main.py
```

## 数据与模型说明

为了开源发布，仓库默认不包含以下大文件或私有数据：

- `retargeted/` 下的 CSV/视频数据
- `bvh/` 下的 BVH 数据
- `smplx/` 下的参考动作 `.npz`
- `assets/body_models/smplx/` 下的 SMPL-X body model 文件

你可以在本地把自己的数据放回这些目录后直接运行。

## 配置说明

项目配置位于 `src/config.py`：

- `DEFAULT_MODEL_PATH`：默认机器人模型 XML 路径
- `ROBOT_FPS`：动作帧率
- `REF_LOAD_MODE`：参考动作加载模式（`SMPL` / `BVH` / `AUTO`）

首次运行后，部分用户级设置会写入：

- `~/.g1_pro_editor/settings.ini`
