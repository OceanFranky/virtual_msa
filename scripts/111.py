import csv
import os
import shutil


def _resolve_pdb_path(target_dir: str, target_name: str, run_id: int, cycle_id: int) -> str:
    """
    不同设计管线命名可能是 {pdb}_binder_run_* 或 {pdb}_protein_run_*，按存在性尝试。
    """
    run_sub = os.path.join(target_dir, "0_protein_hunter_design", f"run_{run_id}")
    stems = (
        f"{target_name}_binder_run_{run_id}_predicted_cycle_{cycle_id}.pdb",
        f"{target_name}_protein_run_{run_id}_predicted_cycle_{cycle_id}.pdb",
    )
    for name in stems:
        p = os.path.join(run_sub, name)
        if os.path.exists(p):
            return p
    return os.path.join(run_sub, stems[0])


def _parse_float(val):
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_and_copy_pdbs(target_name, output_dir, base_path):
    """
    根据 summary_all_runs.csv 中的条件筛选并复制 PDB 文件。
    """
    target_dir = os.path.join(base_path, target_name)
    csv_file = os.path.join(target_dir, "summary_all_runs.csv")

    if not os.path.exists(csv_file):
        print(f"❌ 错误: 找不到 CSV 文件 -> {csv_file}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录已准备: {output_dir}")

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    iptm_threshold = 0.75
    iplddt_threshold = 0.80

    cycle_iptm_cols = [
        c for c in fieldnames if c.startswith("cycle_") and c.endswith("_iptm")
    ]
    num_cycles = len(cycle_iptm_cols)

    copied_count = 0
    missing_count = 0

    for row in rows:
        run_id = int(row["run_id"])

        for cycle_id in range(num_cycles):
            iptm_col = f"cycle_{cycle_id}_iptm"
            iplddt_col = f"cycle_{cycle_id}_iplddt"

            if iptm_col not in row or iplddt_col not in row:
                continue

            iptm_val = _parse_float(row[iptm_col])
            iplddt_val = _parse_float(row[iplddt_col])

            if iptm_val is None or iplddt_val is None:
                continue

            if iplddt_val > iplddt_threshold and iptm_val > iptm_threshold:
                pdb_path = _resolve_pdb_path(target_dir, target_name, run_id, cycle_id)
                pdb_filename = os.path.basename(pdb_path)

                if os.path.exists(pdb_path):
                    dest_path = os.path.join(output_dir, pdb_filename)
                    shutil.copy2(pdb_path, dest_path)
                    print(
                        f"✅ 已复制: {pdb_filename} "
                        f"(iptm: {iptm_val:.3f}, iplddt: {iplddt_val:.3f})"
                    )
                    copied_count += 1
                else:
                    print(f"⚠️ 警告: 满足条件但找不到文件 -> {pdb_path}")
                    missing_count += 1

    print("-" * 50)
    print("🎉 任务完成！")
    print(f"总计成功复制文件数: {copied_count}")
    if missing_count > 0:
        print(f"缺失的文件数: {missing_count}")


# ==========================================
# 在这里修改你的参数并运行
# ==========================================
if __name__ == "__main__":
    BASE_PATH = "/data/wenhao/Protein-Hunter-server/example_myself/VHH"
    TARGET_NAME = "7XTV"
    OUTPUT_DIRECTORY = (
        "/data/wenhao/Protein-Hunter-server/example_myself/iptm/"
        f"Filtered_Good_PDBs_{TARGET_NAME}"
    )

    extract_and_copy_pdbs(
        target_name=TARGET_NAME,
        output_dir=OUTPUT_DIRECTORY,
        base_path=BASE_PATH,
    )
