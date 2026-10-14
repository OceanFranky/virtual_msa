import argparse
import subprocess
import json
import re
import os
import tempfile

def run_usalign(design_pdb, native_pdb, usalign_bin="USalign"):
    """ 调用 US-align 计算全局 RMSD 和 TM-score """
    print("⏳ 正在运行 US-align 进行全长空间比对...")
    cmd = [usalign_bin, design_pdb, native_pdb]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到 US-align: {usalign_bin}。请检查路径。")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"US-align 运行失败:\n{e.stderr}")

    rmsd, tm_score = None, None
    for line in result.stdout.split('\n'):
        if "RMSD=" in line:
            match = re.search(r"RMSD=\s*([0-9.]+)", line)
            if match: rmsd = float(match.group(1))
        # 提取基于参考结构（Reference 或 Structure_2）长度归一化的 TM-score
        if "TM-score=" in line and ("normalized by length of Reference" in line or "normalized by length of Structure_2" in line):
            match = re.search(r"TM-score=\s*([0-9.]+)", line)
            if match: tm_score = float(match.group(1))
                
    if rmsd is None or tm_score is None:
        raise ValueError("无法解析 US-align 输出，请检查输入的 PDB 是否合法。")
        
    return rmsd, tm_score

def run_ost_lddt(design_pdb, native_pdb):
    """ 调用 OpenStructure 计算全局 lDDT """
    print("⏳ 正在运行 OpenStructure 计算全局 lDDT...")
    fd, temp_json = tempfile.mkstemp(suffix=".json")
    os.close(fd) # 仅借用文件名，关闭句柄让 OST 写入
    
    cmd = [
        "ost", "compare-structures",
        "--model", design_pdb,
        "--reference", native_pdb,
        "--output-json", temp_json
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        with open(temp_json, "r") as f:
            data = json.load(f)
        global_lddt = float(data["lddt"])
    except FileNotFoundError:
        raise FileNotFoundError("找不到 'ost' 命令，请确保已激活 ost_env 环境！")
    except Exception as e:
        print(f"⚠️ OST (lDDT) 运行失败 (通常是由于序列不一致): {e}")
        return None
    finally:
        # 运行完毕后自动清理 OST 生成的临时 JSON
        if os.path.exists(temp_json):
            os.remove(temp_json)
            
    return global_lddt

def main():
    parser = argparse.ArgumentParser(description="全长蛋白质结构金标准评估 Pipeline")
    parser.add_argument("--design", required=True, help="大模型生成的完整预测 PDB 文件路径")
    parser.add_argument("--native", required=True, help="参考的天然 PDB 文件路径（即使有缺失也可以）")
    parser.add_argument("--usalign_bin", default="USalign", help="US-align 可执行文件路径")
    parser.add_argument("--out", default="full_eval.json", help="输出的 JSON 报告名")
    
    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.design):
        print(f"❌ 错误: 找不到设计 PDB 文件: {args.design}")
        return
    if not os.path.exists(args.native):
        print(f"❌ 错误: 找不到参考 PDB 文件: {args.native}")
        return

    try:
        # 直接将完整 PDB 扔给金标准双测
        rmsd, tm_score = run_usalign(args.design, args.native, args.usalign_bin)
        lddt = run_ost_lddt(args.design, args.native)

        # 汇总并打印结果
        report = {
            "design_file": args.design,
            "native_file": args.native,
            "metrics": {
                "Global_RMSD": round(rmsd, 4) if rmsd is not None else None, 
                "Global_TM_score": round(tm_score, 4) if tm_score is not None else None, 
                "Global_lDDT": round(lddt, 4) if lddt is not None else None
            },
            "status": "SUCCESS"
        }

        print("\n" + "="*45)
        print("🏆 全长结构 (Full-length) 评估完成 🏆")
        print("="*45)
        if rmsd is not None: print(f"📏 全局 RMSD    : {rmsd:.4f} Å")
        if tm_score is not None: print(f"🧬 全局 TM-score: {tm_score:.4f}")
        if lddt is not None: print(f"🎯 全局 lDDT    : {lddt:.4f}")
        else: print("🎯 全局 lDDT    : N/A (序列不匹配)")
        print("="*45)

        with open(args.out, "w") as f:
            json.dump(report, f, indent=4)
        print(f"💾 评估报告已保存至: {args.out}\n")

    except Exception as e:
        print(f"\n❌ 评估失败: {e}")

if __name__ == "__main__":
    main()