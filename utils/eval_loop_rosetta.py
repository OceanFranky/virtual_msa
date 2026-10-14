import argparse
import json
import os
import sys

# 尝试导入 PyRosetta
try:
    import pyrosetta as pr
    from pyrosetta.rosetta.core.scoring import ScoreType
except ImportError:
    print("错误: 无法导入 PyRosetta。请确保你已在当前环境中安装并激活了它。")
    sys.exit(1)

def read_a3m_query(a3m_path: str) -> str:
    """读取A3M文件并提取第一条序列(Query)"""
    seq = []
    in_query = False
    with open(a3m_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if in_query:
                    break
                in_query = True
                continue
            if in_query:
                seq.append(line)
    if not seq:
        raise ValueError(f"无法从A3M读取query序列: {a3m_path}")
    return "".join(seq).upper()

def evaluate_loop_energy(a3m_path: str, pdb_path: str, chain_id: str = "A"):
    """
    使用 PyRosetta 评估 'X' 区域 (Loop) 的物理能量。
    """
    # 1. 解析 A3M，找到 X 的位置 (基于 Query 序列，1-based 索引)
    query_seq = read_a3m_query(a3m_path)
    x_positions = [i + 1 for i, aa in enumerate(query_seq) if aa == "X"]
    
    if not x_positions:
        print("警告: 在 A3M query 中没有找到 'X' (Loop) 区域。")
        return None
    
    # 2. 初始化 PyRosetta
    # 静默输出，只输出我们关心的结果
    pr.init("-mute all -ignore_unrecognized_res")
    
    # 3. 加载 PDB
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"找不到 PDB 文件: {pdb_path}")
    pose = pr.pose_from_pdb(pdb_path)
    
    # 4. 设置标准打分函数 (ref2015)
    scorefxn = pr.get_fa_scorefxn()
    # 先给整个 pose 打分，这样每个残基的能量才会被计算和缓存
    total_score = scorefxn(pose)
    
    # 5. 提取 Loop (X) 区域的能量
    loop_energies = []
    total_loop_energy = 0.0
    total_loop_clash = 0.0 # 范德华排斥(原子碰撞)
    
    # 我们假设 PDB 的残基编号与 A3M query 的位置是对齐的 (都是从 1 开始连续)
    # 实际项目中可能需要更复杂的对齐，但针对设计出来的从1开始的PDB，这个通常成立。
    valid_x_count = 0
    
    for seq_pos in x_positions:
        # 将 PDB 链+序列号 转换为 Pose 内部编号 (1 到 N)
        try:
            pose_res_num = pose.pdb_info().pdb2pose(chain_id, seq_pos)
            if pose_res_num == 0: # 没找到该残基
                continue
        except Exception:
            continue
            
        valid_x_count += 1
        
        # 获取该残基的能量
        res_energy = pose.energies().residue_total_energy(pose_res_num)
        
        # 获取具体的碰撞能量 (fa_rep: Full-atom repulsion)
        # 高 fa_rep 意味着原子挤在一起，是非常差的结构
        res_energies_obj = pose.energies().residue_total_energies(pose_res_num)
        fa_rep = res_energies_obj[ScoreType.fa_rep]
        
        loop_energies.append({
            "seq_pos": seq_pos,
            "pose_res_num": pose_res_num,
            "res_name": pose.residue(pose_res_num).name3(),
            "total_energy": float(res_energy),
            "fa_rep_clash": float(fa_rep)
        })
        
        total_loop_energy += res_energy
        total_loop_clash += fa_rep

    if valid_x_count == 0:
        raise ValueError(f"在 PDB 链 {chain_id} 中没有找到任何对应的 Loop (X) 残基。")

    # 6. 计算统计指标
    avg_loop_energy = total_loop_energy / valid_x_count
    
    result = {
        "pdb_file": pdb_path,
        "chain_id": chain_id,
        "total_protein_score": float(total_score),
        "loop_residue_count": valid_x_count,
        "loop_total_energy": float(total_loop_energy),
        "loop_average_energy": float(avg_loop_energy),
        "loop_total_clash_fa_rep": float(total_loop_clash),
        "loop_details": loop_energies
    }
    return result

def main():
    parser = argparse.ArgumentParser(description="使用 PyRosetta 评估 A3M 中 'X' (Loop) 区域的物理能量。")
    parser.add_argument("--a3m", required=True, type=str, help="输入的 A3M 文件路径 (用于提取 X 掩码)")
    parser.add_argument("--pdb", required=True, type=str, help="设计的 PDB 结构文件")
    parser.add_argument("--chain", default="A", type=str, help="需要评估的链 ID (默认: A)")
    parser.add_argument("--json_out", default="", type=str, help="将结果输出到 JSON 文件")
    
    args = parser.parse_args()
    
    try:
        out = evaluate_loop_energy(args.a3m, args.pdb, args.chain)
        if out:
            print(json.dumps(out, ensure_ascii=False, indent=2))
            
            if args.json_out:
                with open(args.json_out, "w") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                print(f"\n结果已保存至: {args.json_out}")
    except Exception as e:
        print(f"评估失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
