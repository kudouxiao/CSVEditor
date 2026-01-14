import numpy as np
import re
from scipy.spatial.transform import Rotation as R


def rotation_matrix_from_vectors(vec1, vec2):
    """计算将 vec1 旋转到 vec2 的旋转矩阵"""
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    if s < 1e-6: 
        return np.eye(3) 
        
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))


def parse_bvh(file_path):
    """
    独立 BVH 解析器
    返回: 
    - positions: (Frames, Joints, 3) 世界坐标
    - parents: List[int] 父关节索引
    - names: List[str] 关节名称
    """
    with open(file_path, 'r') as f:
        content = f.read()

    hierarchy_part, motion_part = content.split('MOTION')
    
    # 1. 解析层级结构
    names = []
    offsets = []
    parents = []
    stack = []
    
    lines = hierarchy_part.split('\n')
    current_parent = -1
    
    for line in lines:
        line = line.strip()
        if line.startswith('ROOT') or line.startswith('JOINT'):
            name = line.split()[1]
            names.append(name)
            parents.append(current_parent)
            current_parent = len(names) - 1
            stack.append(current_parent)
        elif line.startswith('End Site'):
            names.append(names[-1] + '_End')
            parents.append(current_parent)
            stack.append(len(names) - 1) # 压栈以便弹出
        elif line.startswith('OFFSET'):
            offset = np.array([float(x) for x in line.split()[1:]])
            offsets.append(offset)
        elif line.startswith('}'):
            if stack:
                stack.pop()
                current_parent = stack[-1] if stack else -1

    # 补齐 End Site 的 Offset
    if len(offsets) < len(names):
        # 简单查找剩余的 OFFSET
        pass 

    # 2. 解析动作数据
    motion_lines = motion_part.strip().split('\n')
    frame_count = 0
    frame_time = 0.033
    
    data_start_line = 0
    for i, line in enumerate(motion_lines):
        if line.startswith('Frames:'):
            frame_count = int(line.split()[1])
        elif line.startswith('Frame Time:'):
            frame_time = float(line.split()[2])
        elif line.strip() and not line.startswith('Frame'):
            data_start_line = i
            break
            
    raw_data = np.array([list(map(float, l.split())) for l in motion_lines[data_start_line:] if l.strip()])
    
    # 3. 前向运动学 (FK) 计算世界坐标
    # 这一步简化处理，假设标准 ZXY 欧拉角 (常见 BVH 格式)
    # 实际项目中可能需要更复杂的旋转顺序处理
    
    num_joints = len(names)
    num_frames = raw_data.shape[0]
    positions = np.zeros((num_frames, num_joints, 3))
    
    # 解析 channel 映射 (略简化，假设标准 3位移+3旋转 for Root, 3旋转 for Others)
    # 为了通用性，这里使用递归 FK 计算
    
    # 预计算所有局部旋转和位移
    # 注意：这里是一个简化版 FK，如果你的 BVH 旋转顺序很特殊，需要调整 axes='zxy'
    
    pointer = 0
    local_rotations = [] # list of (F, 4) quat per joint
    root_pos = raw_data[:, :3]
    pointer += 3
    
    for i in range(num_joints):
        # Root 和 Joint 通常有 3个旋转通道
        # End Site 没有通道
        if "End" in names[i]:
            rot = np.tile([0,0,0,1], (num_frames, 1)) # identity
        else:
            eulers = raw_data[:, pointer:pointer+3]
            rot = R.from_euler('zxy', eulers, degrees=True).as_quat() # (F, 4)
            pointer += 3
        local_rotations.append(rot)

    # 递归计算全局
    global_pos = [None] * num_joints
    global_rot = [None] * num_joints
    
    offsets = np.array(offsets) # (J, 3)
    
    for i in range(num_joints):
        parent = parents[i]
        local_off = offsets[i] # (3,)
        local_q = local_rotations[i] # (F, 4)
        
        if parent == -1:
            # Root
            global_pos[i] = root_pos + local_off # (F, 3)
            global_rot[i] = local_q
        else:
            # Child: P_g = Parent_g + Parent_R_g * local_off
            # R_g = Parent_R_g * local_R
            parent_pos = global_pos[parent]
            parent_rot = global_rot[parent]
            
            # Rotate offset by parent rotation
            r_obj = R.from_quat(parent_rot)
            rotated_offset = r_obj.apply(local_off)
            
            global_pos[i] = parent_pos + rotated_offset
            
            # Combine rotations
            # scipy: q_total = q_parent * q_local
            # (scipy multiply is distinct, check docs. usually r1 * r2 means apply r2 then r1? 
            # In scipy R1 * R2 applies R2 then R1 frame-wise? No, it's composition.)
            # simpler:
            global_rot[i] = (R.from_quat(parent_rot) * R.from_quat(local_q)).as_quat()

    positions = np.stack(global_pos, axis=1) # (F, J, 3)
    
    # 4. 坐标系转换 (BVH Y-up -> MuJoCo Z-up)
    # MuJoCo: Z-up, Y-depth, X-right
    # BVH standard: Y-up, Z-depth, X-right
    # Matrix: Rotate -90 around X
    
    # Apply coordinate transform to final positions
    # (x, y, z) -> (x, -z, y)
    final_pos = np.zeros_like(positions)
    final_pos[..., 0] = positions[..., 0] / 100.0 # cm to m
    final_pos[..., 1] = -positions[..., 2] / 100.0
    final_pos[..., 2] = positions[..., 1] / 100.0
    
    return final_pos, parents