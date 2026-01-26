import threading
import os
import numpy as np
import pandas as pd
import torch
import smplx
import mujoco
from PyQt5.QtCore import QObject, pyqtSignal
import librosa # 引入音频库

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

        # 音频数据
        self.audio_path = None
        self.audio_data = None # 原始音频数据 (numpy array)
        self.audio_sr = 22050  # 采样率
        self.audio_offset = 0.0 # 音频相对于动作的偏移 (秒) - 负数表示音乐延迟播放
        self.duration = 0.0

        self.lock = threading.Lock()
        self.joint_mapping = {}
        self.joint_id_mapping = {}  # [新增] CSV Index -> Joint ID (用于查限位)
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


    def load_audio_data(self, file_path):
        try:
            print(f"[INFO] Loading Audio: {file_path}")
            # 加载原始音频
            y, sr = librosa.load(file_path, sr=None) # sr=None 保持原始采样率
            
            self.audio_data = y
            self.audio_sr = sr
            self.audio_path = file_path
            self.duration = librosa.get_duration(y=y, sr=sr)
            self.audio_offset = 0.0 # 重置偏移
            
            return True
        except Exception as e:
            print(f"Audio Error: {e}")
            return False
            
    # 计算当前帧对应的音频时间戳 (考虑偏移)
    def get_audio_time_from_frame(self, frame_idx, fps=30.0):
        # 动作时间 = frame / fps
        # 音频应该播放的时间 = 动作时间 - 偏移量
        return (frame_idx / fps) - self.audio_offset


    def load_data(self, csv_path, model_path):
        try:
            self.df = pd.read_csv(csv_path, header=None)
            if isinstance(self.df.iloc[0, 0], str): self.df = pd.read_csv(csv_path)
            self.df_orig = self.df.copy()
            self.undo_stack.clear(); self.redo_stack.clear()
            
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
            self.model.opt.disableflags = 65535 
            self.data = mujoco.MjData(self.model)
            
            # === 建立双重映射 ===
            model_j_names = {} # Name -> Qpos Addr
            model_j_ids = {}   # Name -> Joint ID
            
            q_ptr = 0
            for i in range(self.model.njnt):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                # 跳过自由关节(Root)
                if self.model.jnt_type[i] == 0: 
                    q_ptr += 7
                    continue
                
                model_j_names[name] = q_ptr
                model_j_ids[name] = i # 记录 Joint ID
                q_ptr += 1
            
            self.joint_mapping = {}
            self.joint_id_mapping = {} # [新增]
            
            for idx, name in enumerate(self.csv_joint_names):
                simple = name.replace("_joint", "")
                for m_n, addr in model_j_names.items():
                    if simple in m_n or m_n in simple: 
                        self.joint_mapping[idx] = addr
                        self.joint_id_mapping[idx] = model_j_ids[m_n] # 绑定 ID
                        break
            
            return True, len(self.df)
        except Exception as e:
            print(f"Load Error: {e}")
            return False, 0

    # === [新增] 获取限位函数 ===
    def get_joint_limits(self, ui_index):
        """
        根据 UI 列表索引返回 (min, max) 或 None
        ui_index: 0-6 (Root), 7+ (Joints)
        """
        if self.model is None: return None
        
        # Root 没有限位
        if ui_index < 7: return None
        
        # 映射回 CSV 索引
        csv_idx = ui_index - 7
        
        if csv_idx in self.joint_id_mapping:
            jnt_id = self.joint_id_mapping[csv_idx]
            
            # 检查关节是否开启了限位 (jnt_limited)
            # limited: 0=false, 1=true
            is_limited = self.model.jnt_limited[jnt_id]
            
            if is_limited:
                # jnt_range 形状为 (njnt, 2)
                return self.model.jnt_range[jnt_id]
        
        return None

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


    def insert_frames(self, frame_idx, count=1):
        """在指定帧索引之后插入 count 个当前帧的副本"""
        if self.df is None: return False
        
        self.snapshot() # 1. 记录撤销栈
        
        # 2. 切分数据
        # part1: 0 到 frame_idx (包含当前帧)
        part1 = self.df.iloc[:frame_idx+1]
        # part2: frame_idx+1 到 结束
        part2 = self.df.iloc[frame_idx+1:]
        
        # 3. 构造插入块 (复制当前帧 count 次)
        # 注意：使用 iloc[i:i+1] 保持 DataFrame 格式而不是 Series
        frame_to_copy = self.df.iloc[frame_idx:frame_idx+1]
        new_block = pd.concat([frame_to_copy] * count, ignore_index=True)
        
        # 4. 合并
        self.df = pd.concat([part1, new_block, part2], ignore_index=True).reset_index(drop=True)
        
        part1 = self.df.iloc[:frame_idx+1]
        part2 = self.df.iloc[frame_idx+1:]
        frame_to_copy = self.df.iloc[frame_idx:frame_idx+1]
        new_block = pd.concat([frame_to_copy] * count, ignore_index=True)
        self.df = pd.concat([part1, new_block, part2], ignore_index=True).reset_index(drop=True)
        
        # === 修改：返回插入的范围 (开始帧, 结束帧) ===
        # 插入是从 frame_idx + 1 开始的
        inserted_start = frame_idx + 1
        inserted_end = frame_idx + count
        return (inserted_start, inserted_end)

    def delete_frames(self, start_idx, count=1):
        """删除从 start_idx 开始的 count 帧"""
        if self.df is None or len(self.df) <= 1: return False
        
        self.snapshot()
        
        # 保护边界
        actual_count = min(count, len(self.df) - start_idx)
        if actual_count <= 0: return False
        
        # 删除逻辑：保留头部和尾部，丢弃中间
        # part1: 0 到 start_idx-1
        part1 = self.df.iloc[:start_idx]
        # part2: start_idx + count 到 结束
        part2 = self.df.iloc[start_idx + actual_count:]
        
        self.df = pd.concat([part1, part2], ignore_index=True).reset_index(drop=True)
        
        # 如果删空了，保留至少一帧
        if len(self.df) == 0:
            # 这种情况极少见，除非全选删除了，恢复一帧零姿态
            self.df = pd.DataFrame(np.zeros((1, self.df_orig.shape[1])))
            
        return True
            
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

    def apply_mirror(self, start_frame, end_frame):
        """
        对指定范围内的动作进行左右镜像
        逻辑：
        1. Root: Pos Y 取反, Quat X/Z 取反
        2. Body: 左关节数据 <-> 右关节数据
        3. Sign: Roll/Yaw 关节取反
        """
        if self.df is None: return False
        self.snapshot()
        
        # 1. 准备数据切片
        # 注意：这里必须使用 copy()，否则交换时会互相覆盖
        subset = self.df.iloc[start_frame:end_frame+1].copy()
        
        # === A. Root 处理 (前7列) ===
        # Root Pos: [X, Y, Z] -> Y 取反
        subset.iloc[:, 1] *= -1 
        
        # Root Quat: [W, X, Y, Z] -> X, Z 取反 (假设 MuJoCo 格式 w,x,y,z)
        # 镜像原理：翻转 Roll(x) 和 Yaw(z)，保留 Pitch(y)
        subset.iloc[:, 4] *= -1 # qx
        subset.iloc[:, 6] *= -1 # qz
        
        # === B. 关节处理 (交换 + 取反) ===
        # 建立映射表 (仅在第一次运行时建立，提高性能)
        if not hasattr(self, 'mirror_pairs'):
            self.build_mirror_map()
            
        # 临时存储镜像后的关节数据
        mirrored_joints = subset.iloc[:, 7:].copy()
        
        # 遍历映射表进行交换和取反
        for src_col, dst_col, flip_sign in self.mirror_pairs:
            # 获取原始数据
            src_data = subset.iloc[:, src_col].values
            dst_data = subset.iloc[:, dst_col].values
            
            # 交换并处理符号
            # 目标列 = 源数据 * (符号因子)
            mirrored_joints.iloc[:, dst_col - 7] = src_data * (-1 if flip_sign else 1)
            mirrored_joints.iloc[:, src_col - 7] = dst_data * (-1 if flip_sign else 1)
            
        # 覆盖回 subset
        self.df.iloc[start_frame:end_frame+1, 7:] = mirrored_joints
        
        # 标记修改
        self.modified_frames.update(range(start_frame, end_frame+1))
        return True

    def align_global_coordinates(self, junction_frame):
        """
        自动对齐全局坐标：在指定帧处对齐后续动作的根位置和朝向
        
        参数:
            junction_frame: 拼接点帧索引（后一段动作的起始帧）
        
        功能:
            1. 计算拼接点处的位置和朝向差异
            2. 对后一段动作应用平移和旋转，使其在拼接点处连续
            3. 保持后一段动作的相对运动特征
        """
        if self.df is None or junction_frame <= 0 or junction_frame >= len(self.df):
            return False
        
        self.snapshot()
        
        # 1. 获取拼接点前后两帧的根位置和朝向
        frame_before = junction_frame - 1
        frame_after = junction_frame
        
        # Root Position (X, Y, Z)
        pos_before = self.df.iloc[frame_before, 0:3].values.astype(float)
        pos_after = self.df.iloc[frame_after, 0:3].values.astype(float)
        
        # Root Quaternion (W, X, Y, Z)
        quat_before = self.df.iloc[frame_before, 3:7].values.astype(float)
        quat_after = self.df.iloc[frame_after, 3:7].values.astype(float)
        
        # 2. 计算位置偏移
        pos_offset = pos_before - pos_after
        
        # 3. 计算旋转差异（四元数）
        # q_before = q_offset * q_after
        # q_offset = q_before * conjugate(q_after)
        quat_after_conj = quat_after.copy()
        quat_after_conj[1:] *= -1  # 共轭：保持w，翻转x,y,z
        
        # 四元数乘法: q1 * q2
        def quat_multiply(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])
        
        quat_offset = quat_multiply(quat_before, quat_after_conj)
        
        # 归一化
        quat_offset /= np.linalg.norm(quat_offset)
        
        # 4. 对后续所有帧应用偏移
        for i in range(frame_after, len(self.df)):
            # 应用位置偏移
            self.df.iloc[i, 0:3] += pos_offset
            
            # 应用旋转偏移
            current_quat = self.df.iloc[i, 3:7].values.astype(float)
            new_quat = quat_multiply(quat_offset, current_quat)
            new_quat /= np.linalg.norm(new_quat)  # 归一化
            self.df.iloc[i, 3:7] = new_quat
        
        # 标记修改的帧
        self.modified_frames.update(range(frame_after, len(self.df)))
        
        print(f"[Align] Applied offset: Pos={pos_offset}, Quat={quat_offset}")
        return True

    def build_mirror_map(self):
        """
        自动构建左右关节映射表
        return: list of (src_col_idx, dst_col_idx, needs_flip)
        """
        self.mirror_pairs = []
        processed_indices = set()
        
        for i, name in enumerate(self.csv_joint_names):
            col_idx = 7 + i
            if i in processed_indices: continue
            
            # 1. 处理左右对称关节 (Left <-> Right)
            if "left" in name:
                target_name = name.replace("left", "right")
                try:
                    target_i = self.csv_joint_names.index(target_name)
                    # 确定是否需要取反
                    # 经验法则：Pitch 不反，Roll/Yaw 反
                    needs_flip = ("roll" in name) or ("yaw" in name)
                    
                    self.mirror_pairs.append((col_idx, 7 + target_i, needs_flip))
                    processed_indices.add(i)
                    processed_indices.add(target_i)
                except ValueError:
                    print(f"[Mirror] Warning: No pair found for {name}")
            
            # 2. 处理中轴关节 (Waist)
            elif "waist" in name:
                # 中轴关节不交换，只取反
                # Waist Yaw/Roll 取反，Pitch 不变
                needs_flip = ("roll" in name) or ("yaw" in name)
                if needs_flip:
                    # 自己跟自己换 = 原地取反
                    self.mirror_pairs.append((col_idx, col_idx, True))
                processed_indices.add(i)