import argparse
import json
import numpy as np
from Bio.PDB import PDBParser, Selection

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

def get_non_x_indices(a3m_path):
    """从A3M文件读取query序列，识别非X位置的索引"""
    with open(a3m_path, "r") as f:
        lines = f.readlines()
    # 提取第一条序列 (Query)
    query_seq = ""
    for line in lines:
        if line.startswith(">"):
            if query_seq: break
            continue
        query_seq += line.strip()
    
    # 记录非 'X' 且非 '-' 的位置索引 (从0开始)
    non_x_indices = [i for i, char in enumerate(query_seq) if char.upper() != 'X' and char != '-']
    return non_x_indices, query_seq

def get_x_indices(query_seq):
    return [i for i, c in enumerate(query_seq) if c.upper() == "X"]

def _build_chain_backbone_table(pdb_path, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]
    chain = model[chain_id]
    residues = [r for r in chain.get_residues() if r.id[0] == " "]
    seq_chars = []
    full_bb = {}
    for idx, res in enumerate(residues):
        aa = AA3_TO_1.get(res.get_resname().upper(), "X")
        seq_chars.append(aa)
        atoms = {}
        for atom_name in ("N", "CA", "C", "O"):
            if atom_name in res:
                atoms[atom_name] = res[atom_name].get_coord()
        if len(atoms) == 4:
            full_bb[idx] = np.array([atoms["N"], atoms["CA"], atoms["C"], atoms["O"]], dtype=np.float64)
    return "".join(seq_chars), full_bb

def _global_align_map(query_seq, target_seq):
    match_score = 2.0
    mismatch_score = -1.0
    gap_score = -2.0
    m = len(query_seq)
    n = len(target_seq)
    dp = np.zeros((m + 1, n + 1), dtype=np.float64)
    trace = np.zeros((m + 1, n + 1), dtype=np.int8)
    for i in range(1, m + 1):
        dp[i, 0] = dp[i - 1, 0] + gap_score
        trace[i, 0] = 1
    for j in range(1, n + 1):
        dp[0, j] = dp[0, j - 1] + gap_score
        trace[0, j] = 2
    for i in range(1, m + 1):
        qi = query_seq[i - 1]
        for j in range(1, n + 1):
            tj = target_seq[j - 1]
            diag = dp[i - 1, j - 1] + (match_score if qi == tj else mismatch_score)
            up = dp[i - 1, j] + gap_score
            left = dp[i, j - 1] + gap_score
            if diag >= up and diag >= left:
                dp[i, j] = diag
                trace[i, j] = 0
            elif up >= left:
                dp[i, j] = up
                trace[i, j] = 1
            else:
                dp[i, j] = left
                trace[i, j] = 2
    mapping = {}
    i, j = m, n
    while i > 0 or j > 0:
        t = trace[i, j]
        if t == 0 and i > 0 and j > 0:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif t == 1 and i > 0:
            i -= 1
        elif j > 0:
            j -= 1
    return mapping

def get_aligned_backbone_coords(native_pdb, design_pdb, chain_id, query_seq, query_positions):
    native_seq, native_bb = _build_chain_backbone_table(native_pdb, chain_id)
    design_seq, design_bb = _build_chain_backbone_table(design_pdb, chain_id)
    q2n = _global_align_map(query_seq, native_seq)
    q2d = _global_align_map(query_seq, design_seq)
    paired = []
    for q in query_positions:
        if q not in q2n or q not in q2d:
            continue
        ni = q2n[q]
        di = q2d[q]
        if ni not in native_bb or di not in design_bb:
            continue
        paired.append((native_bb[ni], design_bb[di]))
    if not paired:
        return np.array([]), np.array([]), 0
    ref = np.array([p[0] for p in paired], dtype=np.float64)
    pred = np.array([p[1] for p in paired], dtype=np.float64)
    return ref, pred, len(paired)

def extract_backbone_coords(pdb_path, chain_id, target_indices):
    """
    提取特定链、特定残基的主链原子 (N, CA, C, O) 坐标
    注意：这里假设 PDB 残基顺序与 A3M 序列完全一致
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]
    chain = model[chain_id]
    
    # 提取所有残基
    residues = list(chain.get_residues())
    # 过滤标准氨基酸 (去掉水、配体)
    residues = [r for r in residues if r.id[0] == " "]
    
    coords = []
    backbone_atoms = ['N', 'CA', 'C', 'O']
    
    for idx in target_indices:
        if idx >= len(residues):
            continue
        res = residues[idx]
        # 严格提取主链四个原子
        res_coords = []
        for atom_name in backbone_atoms:
            if atom_name in res:
                res_coords.append(res[atom_name].get_coord())
        
        # 只有当4个主链原子都存在时才加入，保证一一对应
        if len(res_coords) == 4:
            coords.append(res_coords)
            
    return np.array(coords) # Shape: (N_res, 4, 3)

def calculate_rmsd(coords_ref, coords_pred):
    """
    使用 Kabsch 算法计算叠加后的主链 RMSD
    $RMSD = \sqrt{\frac{1}{N} \sum_{i=1}^N \|v_i - w_i\|^2}$
    """
    # 展平为 (N_atoms, 3)
    ref = coords_ref.reshape(-1, 3)
    pred = coords_pred.reshape(-1, 3)
    
    # 1. 中心化
    centroid_ref = np.mean(ref, axis=0)
    centroid_pred = np.mean(pred, axis=0)
    ref_c = ref - centroid_ref
    pred_c = pred - centroid_pred
    
    # 2. 奇异值分解 (SVD) 求旋转矩阵
    H = pred_c.T @ ref_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    step = np.eye(3)
    if d < 0:
        step[2, 2] = -1
    
    rotation = Vt.T @ step @ U.T
    
    # 3. 叠加并计算 RMSD
    pred_rotated = (rotation @ pred_c.T).T
    diff = ref_c - pred_rotated
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd

def calculate_lddt(coords_ref, coords_pred, cutoff=15.0):
    """
    计算局部距离差值测试 (lDDT)
    这是评估骨架局部几何准确性的黄金标准，不需要对齐
    """
    ref = coords_ref.reshape(-1, 3)
    pred = coords_pred.reshape(-1, 3)
    n = ref.shape[0]
    
    # 计算距离矩阵
    diff_ref = ref[:, np.newaxis, :] - ref[np.newaxis, :, :]
    dist_ref = np.sqrt(np.sum(diff_ref**2, axis=-1))
    
    diff_pred = pred[:, np.newaxis, :] - pred[np.newaxis, :, :]
    dist_pred = np.sqrt(np.sum(diff_pred**2, axis=-1))
    
    # 仅考虑在 cutoff 范围内的原子对，且排除原子自身
    mask = (dist_ref < cutoff) & (np.eye(n) == 0)
    
    if not np.any(mask):
        return 0.0
    
    # 计算距离偏差
    dist_diff = np.abs(dist_ref - dist_pred)
    
    # lDDT 统计四个阈值: 0.5A, 1A, 2A, 4A
    thresholds = [0.5, 1.0, 2.0, 4.0]
    scores = []
    for t in thresholds:
        preserved = (dist_diff[mask] < t).sum()
        scores.append(preserved / mask.sum())
    
    return np.mean(scores)

def main():
    parser = argparse.ArgumentParser(description="验证非X骨架区域的精度")
    parser.add_argument("--a3m", required=True, help="输入的A3M序列文件")
    parser.add_argument("--native_pdb", required=True, help="天然结构PDB")
    parser.add_argument("--design_pdb", required=True, help="模型生成的PDB")
    parser.add_argument("--chain", default="A", help="要比对的链ID")
    args = parser.parse_args()

    # 1. 确定非X区域
    non_x_indices, full_seq = get_non_x_indices(args.a3m)
    print(f"检测到总序列长度: {len(full_seq)}, 其中非X骨架长度: {len(non_x_indices)}")

    ref_coords, pred_coords, mapped_count = get_aligned_backbone_coords(
        args.native_pdb, args.design_pdb, args.chain, full_seq, non_x_indices
    )
    if mapped_count == 0:
        print("错误：对齐后没有可比较的非X残基，请检查链ID或结构缺失。")
        return

    # 3. 计算指标
    rmsd_val = calculate_rmsd(ref_coords, pred_coords)
    lddt_val = calculate_lddt(ref_coords, pred_coords)

    # 4. 输出结果
    result = {
        "scaffold_length": len(non_x_indices),
        "mapped_scaffold_length": mapped_count,
        "backbone_rmsd": round(float(rmsd_val), 4),
        "backbone_lddt": round(float(lddt_val), 4),
        "status": "PASS" if rmsd_val < 2.0 else "FAIL" # 工业界通常认为骨架RMSD < 2A是基本合格
    }
    
    print("\n--- 骨架精度验证报告 ---")
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
