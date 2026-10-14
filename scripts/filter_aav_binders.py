import os
import glob
import shutil
import numpy as np
from scipy.spatial.distance import cdist
from Bio.PDB import PDBParser
import warnings
from Bio import BiopythonWarning

# 忽略 Biopython 解析 PDB 时的一些常见非致命警告
warnings.simplefilter('ignore', BiopythonWarning)

def get_heavy_atoms_coords(chain, res_list=None):
    """
    提取指定链（或指定残基列表）中所有重原子的坐标
    """
    coords = []
    for res in chain:
        if res_list is not None and res.id[1] not in res_list:
            continue
        for atom in res:
            if atom.element != 'H':
                coords.append(atom.coord)
    return np.array(coords)

# ==================== [核心算法：向量圆锥碰撞检测] ====================
def check_glycan_cone_clash(target_chain, peptide_coords, res_id, 
                            cone_length=10.0, cone_angle_deg=60.0, hard_clash_dist=4.0):
    """
    基于向量的圆锥形糖链位阻判定：
    - 枪管方向 (Axis): CG 指向 ND2
    - 危险区: 位于以 ND2 为顶点、向外延伸 cone_length、张角为 cone_angle_deg 的圆锥内。
    - 绝对死区: 距离 ND2 小于 hard_clash_dist 的任何方向。
    """
    res = None
    try:
        res = target_chain[res_id]
    except KeyError:
        for r in target_chain:
            if r.id[1] == res_id:
                res = r
                break
                
    if res is None:
        return False, "残基不存在"
    
    # 严厉的类型断言：确保你找的位点真的是天冬酰胺 (ASN)
    if res.resname != 'ASN':
        print(f"\n[致命警告] 位点 {res_id} 竟然是 {res.resname}！糖基化只能发生在 ASN 上。")
        return False, f"残基错误({res.resname})"

    # 容错：如果缺原子，退化为保守的球面过滤
    if 'CG' not in res or 'ND2' not in res:
        res_coords = get_heavy_atoms_coords(target_chain, [res_id])
        if len(res_coords) > 0:
            if np.min(cdist(peptide_coords, res_coords)) < 6.0:
                return True, "缺原子退化过滤淘汰 (< 6.0 Å)"
        return False, "安全"

    coord_cg = res['CG'].coord
    coord_nd2 = res['ND2'].coord
    
    # 1. 糖链生长主轴向量及单位化
    axis_vector = coord_nd2 - coord_cg
    axis_norm = np.linalg.norm(axis_vector)
    if axis_norm < 1e-5:
        return False, "向量异常"
    axis_unit = axis_vector / axis_norm
    
    # 2. 圆锥张角的余弦阈值 (例如 60度的余弦是 0.5)
    cos_thresh = np.cos(np.radians(cone_angle_deg))
    
    # 3. 遍历多肽的每一个重原子
    for pep_atom in peptide_coords:
        pep_vector = pep_atom - coord_nd2
        dist = np.linalg.norm(pep_vector)
        
        # [防线 1] 绝对物理碰撞：离氨基酸本身太近 (无论在哪个方向)
        if dist < hard_clash_dist:
            return True, f"绝对碰撞 (距离 {dist:.2f} < {hard_clash_dist} Å)"
            
        # [防线 2] 糖链扫帚圆锥区
        if dist < cone_length:
            pep_unit = pep_vector / dist
            # 计算向量夹角的余弦值
            cos_angle = np.dot(axis_unit, pep_unit)
            
            # 余弦值大于阈值，说明夹角小于设定锥角 -> 掉进圆锥里了！
            if cos_angle > cos_thresh:
                angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                return True, f"落入糖链圆锥 (距离 {dist:.2f} Å < {cone_length}, 偏角 {angle_deg:.1f}° < {cone_angle_deg}°)"
                
    return False, "安全"
# ======================================================================

def evaluate_pdb(pdb_path, target_chain_id, peptide_chain_id, 
                 glycan_sites, bottom_sites):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", pdb_path)
        model = structure[0]
        
        if target_chain_id not in model or peptide_chain_id not in model:
            return False, f"缺失目标链({target_chain_id})或多肽链({peptide_chain_id})"
            
        target_chain = model[target_chain_id]
        peptide_chain = model[peptide_chain_id]
        pep_coords = get_heavy_atoms_coords(peptide_chain)
        
        if len(pep_coords) == 0:
            return False, "多肽链中无有效重原子"

        # 检查 1：糖链圆锥位阻
        for g_site in glycan_sites:
            is_clash, reason = check_glycan_cone_clash(
                target_chain, pep_coords, g_site, 
                cone_length=12.0,      # 糖链伸长范围 10 埃
                cone_angle_deg=60.0,   # 允许 60 度的圆锥半角（覆盖 120 度的立体角扫射）
                hard_clash_dist=4.0    # 距离氨基酸核心 4.0 埃内无差别杀伤
            )
            if is_clash:
                return False, f"位点 {g_site} {reason}"

        # 检查 2：底部细胞膜位阻
        bottom_coords = get_heavy_atoms_coords(target_chain, res_list=bottom_sites)
        if len(bottom_coords) > 0:
            dist_matrix_bottom = cdist(pep_coords, bottom_coords)
            min_dist_bottom = np.min(dist_matrix_bottom)
            if min_dist_bottom < 4.0:
                return False, f"底部撞击 (最短距离 {min_dist_bottom:.2f} Å < 4.0 Å)"

        return True, "通过"
        
    except Exception as e:
        return False, f"解析错误: {str(e)}"

def batch_process(input_dir, target_chain_id, peptide_chain_id, glycan_sites, bottom_sites):
    input_dir = os.path.abspath(input_dir)
    output_dir = input_dir + "_cone_filtered"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[*] 创建输出目录: {output_dir}")
    else:
        print(f"[*] 输出目录已存在: {output_dir}")

    pdb_files = glob.glob(os.path.join(input_dir, "*.pdb"))
    total_files = len(pdb_files)
    
    print(f"[*] 找到 {total_files} 个 PDB 文件，开始 [圆锥空间] 分析...")
    
    pass_count, fail_count = 0, 0
    
    for pdb_path in pdb_files:
        filename = os.path.basename(pdb_path)
        is_valid, reason = evaluate_pdb(
            pdb_path, target_chain_id, peptide_chain_id, glycan_sites, bottom_sites
        )
        
        if is_valid:
            dest_path = os.path.join(output_dir, filename)
            shutil.copy2(pdb_path, dest_path)
            pass_count += 1
            print(f"[通过] {filename}")
        else:
            # 强烈建议在跑的时候保留这一行的打印，看看死在圆锥内的具体参数
            print(f"[淘汰] {filename} -> 原因: {reason}")
            fail_count += 1

    print("\n" + "="*45)
    print(" 圆 锥 算 法 筛 选 完 成")
    print("="*45)
    print(f"输出路径: {output_dir}")
    if total_files > 0:
        print(f"通过率: {pass_count}/{total_files} ({(pass_count/total_files)*100:.1f}%)")

# ================= 配置与运行区 =================
if __name__ == "__main__":
    INPUT_DIRECTORY = "/data/wenhao/Protein-Hunter-server/AAV/boltz_design/4LRH_8aa_iptm"
    PEPTIDE_CHAIN = 'A'
    TARGET_CHAIN = 'B'  
    
    # 请确保这里的编号对应的是真正的 ASN 氨基酸
    GLYCAN_SITES = [40, 132, 172]
    BOTTOM_SITES = list(range(200, 208))

    batch_process(INPUT_DIRECTORY, TARGET_CHAIN, PEPTIDE_CHAIN, GLYCAN_SITES, BOTTOM_SITES)