#!/usr/bin/env python3
"""
提取 Protein Hunter 设计结果中的 best_structure.pdb 文件
用法: python extract_best_structures.py --input_dir <input_dir> --output_dir <output_dir> --prefix <prefix>

示例:
    python extract_best_structures.py \
        --input_dir /data/wenhao/Protein-Hunter-server/loop_MSA_design_boltz/1A8V \
        --output_dir /data/wenhao/Protein-Hunter-server/extracted_pdbs/1A8V \
        --prefix 1a8v
"""

import argparse
import os
import re
import shutil
from pathlib import Path


def extract_best_structures(input_dir: str, output_dir: str, prefix: str):
    """
    从 input_dir 中提取所有 run_X 文件夹下的 best_structure.pdb 文件
    
    Args:
        input_dir: 包含 0_protein_hunter_design 文件夹的目录
        output_dir: 输出目录
        prefix: 输出文件名前缀
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    design_dir = input_path / "0_protein_hunter_design"
    if not design_dir.exists():
        print(f"❌ 目录不存在: {design_dir}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    run_dirs = sorted(
        [d for d in design_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda x: int(x.name.replace("run_", ""))
    )
    
    if not run_dirs:
        print(f"❌ 未找到 run_X 目录: {design_dir}")
        return
    
    print(f"📁 找到 {len(run_dirs)} 个 run 目录")
    print(f"📂 输出目录: {output_path}")
    print(f"📝 文件名格式: {prefix}_0.pdb, {prefix}_1.pdb, ...")
    print("-" * 50)
    
    extracted_count = 0
    missing_count = 0
    
    for run_dir in run_dirs:
        run_id = run_dir.name.replace("run_", "")
        
        pattern = re.compile(rf".*_run_{run_id}_best_structure\.pdb$")
        best_pdb_files = list(run_dir.glob("*_best_structure.pdb"))
        
        if best_pdb_files:
            src_file = best_pdb_files[0]
            dst_file = output_path / f"{prefix}_{run_id}.pdb"
            shutil.copy2(src_file, dst_file)
            print(f"✅ {src_file.name} -> {dst_file.name}")
            extracted_count += 1
        else:
            print(f"⚠️ run_{run_id}: 未找到 best_structure.pdb")
            missing_count += 1
    
    print("-" * 50)
    print(f"✅ 提取完成: {extracted_count} 个文件")
    if missing_count > 0:
        print(f"⚠️ 缺失: {missing_count} 个文件")
    print(f"📂 输出目录: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="提取 Protein Hunter 设计结果中的 best_structure.pdb 文件"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入目录 (包含 0_protein_hunter_design 文件夹)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="输出文件名前缀 (如: 1a8v)"
    )
    
    args = parser.parse_args()
    
    extract_best_structures(args.input_dir, args.output_dir, args.prefix)


if __name__ == "__main__":
    main()
