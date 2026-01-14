import threading
import os
import numpy as np
import pandas as pd
import torch
import smplx
import mujoco
from PyQt5.QtCore import QObject, pyqtSignal

# 导入配置
from src.config import CSV_JOINT_NAMES
from src.utils import parse_bvh # 导入新工具
from src.config import SMPL_PARENTS # 导入默认SMPL关系

# ============ 2. 后端逻辑 ============
class G1Backend(QObject):
    def __init__(self):
        super().__init__()
        self.df = None; self.df_orig = None
        self.model = None; self.data = None

        self.ref_joints = None   # (Frames, Joints, 3)
        self.ref_parents = None  # List[int]
        self.ref_type = "none"   # "smplx" or "bvh"

        self.lock = threading.Lock()
        self.joint_mapping = {}
        self.modified_frames = set()
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 50

        self.root_names = [
            "Root_Pos_X", "Root_Pos_Y", "Root_Pos_Z",
            "Root_Quat_W", "Root_Quat_X", "Root_Quat_Y", "Root_Quat_Z"
        ]

        self.csv_joint_names = CSV_JOINT_NAMES

        self.all_names = self.root_names + self.csv_joint_names

    def load_data(self, csv_path, model_path):
        try:
            self.df = pd.read_csv(csv_path, header=None)
            if isinstance(self.df.iloc[0, 0], str): self.df = pd.read_csv(csv_path)
            self.df_orig = self.df.copy()
            self.undo_stack.clear(); self.redo_stack.clear()
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
            self.model.opt.disableflags = 65535 
            self.data = mujoco.MjData(self.model)
            
            model_j_names = {}
            q_ptr = 0
            for i in range(self.model.njnt):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if self.model.jnt_type[i] == 0: q_ptr += 7; continue
                model_j_names[name] = q_ptr; q_ptr += 1
            for idx, name in enumerate(self.csv_joint_names):
                simple = name.replace("_joint", "")
                for m_n, addr in model_j_names.items():
                    if simple in m_n or m_n in simple: self.joint_mapping[idx] = addr; break
            
            return True, len(self.df)
        except Exception as e:
            print(f"Load Error: {e}")
            return False, 0

    # === 稳健的 SMPL-X 加载 ===
    def load_smplx_data(self, smplx_file, smplx_body_model_path):
        try:
            print(f"[INFO] Loading SMPL-X: {smplx_file}")
            smplx_data = np.load(smplx_file, allow_pickle=True)
            data_dict = {k: smplx_data[k] for k in smplx_data.files}
            
            gender = str(data_dict.get("gender", "neutral"))
            if isinstance(gender, bytes): gender = gender.decode('utf-8')
            if gender == 'n': gender = 'neutral'

            if not os.path.exists(smplx_body_model_path):
                print(f"[ERROR] SMPL Model Dir not found: {smplx_body_model_path}")
                return False

            body_model = smplx.create(
                smplx_body_model_path,
                "smplx",
                gender=gender,
                use_pca=False,
                dtype=torch.float32 
            )
            
            num_frames = data_dict["pose_body"].shape[0]
            print(f"[INFO] SMPL Frames: {num_frames}")
            
            def to_tensor(x): return torch.tensor(x).float().cpu()
            
            output = body_model(
                betas=to_tensor(data_dict["betas"]).view(1, -1),
                global_orient=to_tensor(data_dict["root_orient"]),
                body_pose=to_tensor(data_dict["pose_body"]),
                transl=to_tensor(data_dict["trans"]),
                left_hand_pose=torch.zeros(num_frames, 45).float(),
                right_hand_pose=torch.zeros(num_frames, 45).float(),
                jaw_pose=torch.zeros(num_frames, 3).float(),
                leye_pose=torch.zeros(num_frames, 3).float(),
                reye_pose=torch.zeros(num_frames, 3).float(),
                return_full_pose=True,
            )
            
            self.ref_joints = output.joints # Tensor
            self.ref_parents = SMPL_PARENTS # SMPL 结构固定
            self.ref_type = "smplx"

            print(f"[SUCCESS] SMPL-X Loaded. Joints: {self.ref_joints.shape}")
            return True
        except Exception as e:
            print(f"[ERROR] SMPL Load Exception: {e}")
            import traceback; traceback.print_exc()
            return str(e)

    # === 新增：BVH 加载逻辑 ===
    def load_bvh_data(self, bvh_path):
        try:
            print(f"Loading BVH: {bvh_path}")
            positions, parents = parse_bvh(bvh_path)
            
            self.ref_joints = torch.tensor(positions).float() # 转为 Tensor 统一格式
            self.ref_parents = parents
            self.ref_type = "bvh"
            
            print(f"[SUCCESS] BVH Loaded. Shape: {self.ref_joints.shape}")
            return True
        except Exception as e:
            print(f"BVH Error: {e}")
            import traceback; traceback.print_exc()
            return str(e)

    def set_frame(self, idx):
        if self.df is None: return
        with self.lock:
            line = pd.to_numeric(self.df.iloc[idx].values, errors='coerce')
            line = np.nan_to_num(line.astype(float))
            p = line[0:3]; self.data.qpos[0:3] = p * 0.001 if np.any(np.abs(p) > 50) else p
            q = line[3:7]; wxyz = np.array([q[3], q[0], q[1], q[2]])
            n = np.linalg.norm(wxyz); self.data.qpos[3:7] = wxyz/n if n > 1e-4 else [1,0,0,0]
            for c_idx, m_idx in self.joint_mapping.items():
                col = 7 + c_idx
                if col < len(line): self.data.qpos[m_idx] = line[col]
            mujoco.mj_forward(self.model, self.data)
            
    def snapshot(self):
        if self.df is None: return
        self.undo_stack.append(self.df.copy()); self.redo_stack.clear()
        if len(self.undo_stack) > self.max_history: self.undo_stack.pop(0)
    def undo(self):
        if not self.undo_stack: return False
        self.redo_stack.append(self.df); self.df = self.undo_stack.pop(); return True
    def redo(self):
        if not self.redo_stack: return False
        self.undo_stack.append(self.df); self.df = self.redo_stack.pop(); return True