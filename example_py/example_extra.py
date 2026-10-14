import os
import shutil
import pandas as pd

def extract_and_copy_pdbs(target_name, output_dir, base_path):
    """
    根据 summary_all_runs.csv 中的条件筛选并复制 PDB 文件。
    """
    # 定义 CSV 文件路径
    target_dir = os.path.join(base_path, target_name)
    csv_file = os.path.join(target_dir, "summary_all_runs.csv")
    
    if not os.path.exists(csv_file):
        print(f"❌ 错误: 找不到 CSV 文件 -> {csv_file}")
        return
        
    # 如果输出文件夹不存在，则创建它
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录已准备: {output_dir}")
    
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)
    
    # 设定阈值（根据 CSV 样本，数值为 0-1 范围）
    iptm_threshold = 0.75
    iplddt_threshold = 0.80
    
    copied_count = 0
    missing_count = 0
    
    # 遍历 CSV 中的每一行 (每个 run)
    for index, row in df.iterrows():
        run_id = int(row['run_id'])
        
        # 动态获取当前 run 有多少个 cycle (通过匹配列名)
        cycle_cols = [col for col in df.columns if col.startswith('cycle_') and col.endswith('_iptm')]
        num_cycles = len(cycle_cols)
        
        # 遍历该 run 的每一个 cycle
        for cycle_id in range(num_cycles):
            iptm_col = f'cycle_{cycle_id}_iptm'
            iplddt_col = f'cycle_{cycle_id}_iplddt'
            
            # 确保列名存在，防止报错
            if iptm_col in df.columns and iplddt_col in df.columns:
                iptm_val = row[iptm_col]
                iplddt_val = row[iplddt_col]
                
                # 跳过空值
                if pd.isna(iptm_val) or pd.isna(iplddt_val):
                    continue
                    
                # 判断条件: iplddt > 0.80 且 iptm > 0.75
                if iplddt_val > iplddt_threshold and iptm_val > iptm_threshold:
                    
                    # 构建 PDB 文件名和路径
                    # 格式参考: 7XLD_binder_run_0_predicted_cycle_2.pdb
                    pdb_filename = f"{target_name}_binder_run_{run_id}_predicted_cycle_{cycle_id}.pdb"
                    pdb_dir = os.path.join(target_dir, "0_protein_hunter_design", f"run_{run_id}")
                    pdb_path = os.path.join(pdb_dir, pdb_filename)
                    
                    # 检查文件是否存在并复制
                    if os.path.exists(pdb_path):
                        dest_path = os.path.join(output_dir, pdb_filename)
                        shutil.copy2(pdb_path, dest_path)
                        print(f"✅ 已复制: {pdb_filename} (iptm: {iptm_val:.3f}, iplddt: {iplddt_val:.3f})")
                        copied_count += 1
                    else:
                        print(f"⚠️ 警告: 满足条件但找不到文件 -> {pdb_path}")
                        missing_count += 1
                        
    print("-" * 50)
    print(f"🎉 任务完成！")
    print(f"总计成功复制文件数: {copied_count}")
    if missing_count > 0:
        print(f"缺失的文件数: {missing_count}")

# ==========================================
# 在这里修改你的参数并运行
# ==========================================
if __name__ == "__main__":
    # 基础目录路径
    BASE_PATH = "/data/wenhao/Protein-Hunter-server/example_myself/VHH"
    
    # 你要处理的目录名称 (例如 "7A6O" 或 "7XLD")
    TARGET_NAME = "7XLD" 
    
    # 你想要将满足条件的 pdb 保存到哪个文件夹？(请修改为实际想要的路径)
    OUTPUT_DIRECTORY = "/data/wenhao/Protein-Hunter-server/example_myself/VHH/Filtered_Good_PDBs_7A6O"
    
    extract_and_copy_pdbs(
        target_name=TARGET_NAME, 
        output_dir=OUTPUT_DIRECTORY, 
        base_path=BASE_PATH
    )