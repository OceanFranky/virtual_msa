#!/usr/bin/env python3
"""
合并多个文件夹下的 best_structure.pdb 文件
用法: python merge_best_structures.py --output_dir <output_dir> --prefix_map <folder1>:<prefix1> <folder2>:<prefix2>

示例:
    python merge_best_structures.py \
        --output_dir /data/wenhao/Protein-Hunter-server/merged_pdbs/7EOW_VHH_B \
        --prefix_map /data/wenhao/Protein-Hunter-server/VHH_boltz/7EOW_VHH_B:vhh \
                     /data/wenhao/Protein-Hunter-server/VHH1_boltz/7EOW_VHH_B:vhh1
"""

import argparse
import os
import shutil
from pathlib import Path


def merge_best_structures(output_dir: str, prefix_map: list):
    """
    合并多个文件夹下的 best_structure.pdb 文件
    
    Args:
        output_dir: 输出目录
        prefix_map: 列表，每个元素为 "文件夹路径:前缀" 格式
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 输出目录: {output_path}")
    print("-" * 60)
    
    total_extracted = 0
    
    for item in prefix_map:
        parts = item.split(":")
        if len(parts) != 2:
            print(f"⚠️ 无效格式: {item} (应为 folder:prefix)")
            continue
        
        folder, prefix = parts
        folder_path = Path(folder)
        design_dir = folder_path / "0_protein_hunter_design"
        
        if not design_dir.exists():
            print(f"⚠️ 目录不存在: {design_dir}")
            continue
        
        print(f"\n📁 处理: {folder}")
        print(f"   前缀: {prefix}")
        
        run_dirs = sorted(
            [d for d in design_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda x: int(x.name.replace("run_", ""))
        )
        
        if not run_dirs:
            print(f"   ⚠️ 未找到 run_X 目录")
            continue
        
        extracted = 0
        for run_dir in run_dirs:
            run_id = run_dir.name.replace("run_", "")
            
            best_pdb_files = list(run_dir.glob("*_best_structure.pdb"))
            
            if best_pdb_files:
                src_file = best_pdb_files[0]
                dst_file = output_path / f"{prefix}_{run_id}.pdb"
                shutil.copy2(src_file, dst_file)
                extracted += 1
                total_extracted += 1
        
        print(f"   ✅ 提取 {extracted} 个文件")
    
    print("-" * 60)
    print(f"✅ 总计提取: {total_extracted} 个文件")
    print(f"📂 输出目录: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="合并多个文件夹下的 best_structure.pdb 文件"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--prefix_map",
        type=str,
        nargs="+",
        required=True,
        help="文件夹和前缀映射，格式: folder1:prefix1 folder2:prefix2"
    )
    
    args = parser.parse_args()
    
    merge_best_structures(args.output_dir, args.prefix_map)


if __name__ == "__main__":
    main()
