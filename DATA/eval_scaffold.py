import argparse
import subprocess
import json
import re
import os
import math
from Bio import pairwise2

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

def get_pdb_mapping(pdb_path, a3m_seq):
    """提取 PDB 序列并建立 A3M 索引 -> PDB 残基原子行的映射"""
    pdb_seq = ""
    pdb_residue_data = [] # 存储每个残基的所有原子行
    
    current_res_id = None
    current_res_lines = []
    
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                res_id = (line[21], line[22:26].strip(), line[26]) # chain, res_num, insertion
                if res_id != current_res_id:
                    if current_res_lines:
                        # 记录上一个残基
                        res_name = current_res_lines[0][17:20].strip()
                        pdb_seq += AA3_TO_1.get(res_name.upper(), "X")
                        pdb_residue_data.append(current_res_lines)
                    current_res_id = res_id
                    current_res_lines = [line]
                else:
                    current_res_lines.append(line)
        # 最后一个
        if current_res_lines:
            res_name = current_res_lines[0][17:20].strip()
            pdb_seq += AA3_TO_1.get(res_name.upper(), "X")
            pdb_residue_data.append(current_res_lines)

    # 比对：A3M 里的 X 可以匹配任何东西
    alignments = pairwise2.align.globalms(a3m_seq, pdb_seq, 2, 1, -1, -1)
    best_align = alignments[0]
    aligned_a3m, aligned_pdb = best_align.seqA, best_align.seqB
    
    mapping = {} # a3m_idx -> pdb_residue_lines
    pdb_idx = 0
    a3m_idx = 0
    for i in range(len(aligned_a3m)):
        if aligned_a3m[i] != "-" and aligned_pdb[i] != "-":
            mapping[a3m_idx] = pdb_residue_data[pdb_idx]
        
        if aligned_a3m[i] != "-": a3m_idx += 1
        if aligned_pdb[i] != "-": pdb_idx += 1
            
    return mapping

def run_usalign(design_pdb, native_pdb, usalign_bin="/data/wenhao/USalign/USalign"):
    cmd = [usalign_bin, design_pdb, native_pdb]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception as e:
        raise RuntimeError(f"US-align 运行失败: {e}")

    rmsd, tm_score = None, None
    for line in result.stdout.split('\n'):
        if "RMSD=" in line:
            match = re.search(r"RMSD=\s*([0-9.]+)", line)
            if match: rmsd = float(match.group(1))
        if "TM-score=" in line and ("normalized by length of Reference" in line or "normalized by length of Structure_2" in line):
            match = re.search(r"TM-score=\s*([0-9.]+)", line)
            if match: tm_score = float(match.group(1))
    return rmsd, tm_score

def _parse_ca_coords(pdb_path):
    coords = {}
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                chain = line[21]
                resi = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords[(chain, resi)] = (x, y, z)
    return coords

def run_geometric_lddt(design_pdb, native_pdb, cutoff=15.0):
    design_ca = _parse_ca_coords(design_pdb)
    native_ca = _parse_ca_coords(native_pdb)
    shared = sorted(set(design_ca.keys()) & set(native_ca.keys()), key=lambda k: (k[0], k[1]))
    if len(shared) < 2:
        return None, 0, 0

    thresholds = (0.5, 1.0, 2.0, 4.0)
    residue_scores = []
    neighbor_pairs = 0

    for i, key_i in enumerate(shared):
        ni = native_ca[key_i]
        di = design_ca[key_i]
        neighbors = []
        for j, key_j in enumerate(shared):
            if i == j:
                continue
            nj = native_ca[key_j]
            d_nat = math.dist(ni, nj)
            if d_nat < cutoff:
                neighbors.append((key_j, d_nat))

        if not neighbors:
            continue

        hits = 0
        checks = 0
        for key_j, d_nat in neighbors:
            dj = design_ca[key_j]
            d_mod = math.dist(di, dj)
            delta = abs(d_mod - d_nat)
            for t in thresholds:
                checks += 1
                if delta < t:
                    hits += 1

        residue_scores.append(hits / checks)
        neighbor_pairs += len(neighbors)

    if not residue_scores:
        return None, 0, neighbor_pairs
    return sum(residue_scores) / len(residue_scores), len(residue_scores), neighbor_pairs

def main():
    parser = argparse.ArgumentParser(description="终极 Scaffold 评估 (统一重编号版)")
    parser.add_argument("--design", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--a3m", required=True)
    parser.add_argument("--usalign_bin", default="/data/wenhao/USalign/USalign")
    parser.add_argument("--out", default="scaffold_eval.json")
    args = parser.parse_args()

    # 1. 读取 A3M 理论序列
    a3m_seq = ""
    with open(args.a3m, "r") as f:
        for line in f:
            if not line.startswith(">"): a3m_seq += line.strip(); break

    # 2. 建立两边的映射
    print("🔍 正在同步序列与编号...")
    design_map = get_pdb_mapping(args.design, a3m_seq)
    native_map = get_pdb_mapping(args.native, a3m_seq)

    # 3. 找出共识骨架索引 (非X, 两边都有坐标)
    consensus_indices = sorted([
        i for i in design_map 
        if i in native_map and a3m_seq[i].upper() != 'X'
    ])

    if not consensus_indices:
        print("❌ 错误: 没有找到共同的骨架残基！")
        return

    # 4. 生成统一重编号的临时 PDB
    def write_remapped_pdb(mapping, indices, suffix, pdb_path):
        # 修改保存路径到 /data/wenhao/Protein-Hunter-server/tmp，并添加明确的含义标注
        base_dir = "/data/wenhao/Protein-Hunter-server/tmp"
        # 使用 PDB 文件名作为前缀，避免覆盖
        pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
        filename = f"CHECK_ME_{pdb_name}_{suffix}_scaffold_only_renumbered.pdb"
        path = os.path.join(base_dir, filename)
        
        with open(path, 'w') as f:
            for a3m_idx in indices:
                lines = mapping[a3m_idx]
                abs_num = a3m_idx + 1
                for line in lines:
                    new_line = line[:21] + "A" + f"{abs_num:4}" + line[26:]
                    f.write(new_line)
            f.write("END\n")
        return path

    tmp_design = write_remapped_pdb(design_map, consensus_indices, "DESIGN", args.design)
    tmp_native = write_remapped_pdb(native_map, consensus_indices, "NATIVE", args.native)

    try:
        print(f"✅ 已提取 {len(consensus_indices)} 个共识骨架残基并统一重编号。")
        print(f"📂 临时文件已保存至 (供核对):")
        print(f"   - 预测骨架: {tmp_design}")
        print(f"   - 天然骨架: {tmp_native}")
        
        rmsd, tm_score = run_usalign(tmp_design, tmp_native, args.usalign_bin)
        lddt, lddt_residue_count, lddt_neighbor_pairs = run_geometric_lddt(tmp_design, tmp_native)

        report = {
            "metrics": {"Scaffold_RMSD": rmsd, "Scaffold_TM_score": tm_score, "Scaffold_lDDT": lddt},
            "lddt_method": "geometric_ca_lddt",
            "lddt_cutoff_angstrom": 15.0,
            "lddt_residue_count": lddt_residue_count,
            "lddt_neighbor_pairs": lddt_neighbor_pairs,
            "residues_evaluated": len(consensus_indices),
            "status": "SUCCESS"
        }
        print(f"\n📏 RMSD: {rmsd:.4f} Å | 🧬 TM-score: {tm_score:.4f} | 🎯 lDDT: {lddt if lddt else 'N/A'}")
        with open(args.out, "w") as f: json.dump(report, f, indent=4)
    finally:
        # 为了让用户手动核对，这里不再自动删除文件
        print(f"\n💡 评估完成。请手动核对以上路径下的临时 PDB 文件。")
        # for p in [tmp_design, tmp_native]:
        #     if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    main()
