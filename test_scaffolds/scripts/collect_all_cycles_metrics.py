#!/usr/bin/env python3
"""对 test_scaffords_5.20 中每个蛋白 run_0 的全部 cycle PDB 计算 RMSD/TM，并汇总 pLDDT（非 best 挑选）。"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path("/data/wenhao/Protein-Hunter-server")
ROOT = REPO / "test_scaffords_5.20"
SCAFFOLDS = REPO / "test_scaffolds"
EVAL = REPO / "DATA" / "eval_scaffold.py"
USALIGN = Path("/data/wenhao/USalign/USalign")
PY = "/data/wenhao/miniconda3/envs/proteinhunter/bin/python"

DEFAULT_IDS = ["8G52", "8G53", "8G64", "8GAC", "8GBE"]
TOP10_IDS = [
    "8U0X", "8IQC", "8BAQ", "8Q8I", "8BAU", "8R2C", "8SV7", "8EO5", "8BUX",
]
CYCLE_RE = re.compile(r"_predicted_cycle_(\d+)\.pdb$")


def eval_one(design: Path, native: Path, a3m: Path, out_json: Path) -> dict:
    subprocess.run(
        [PY, str(EVAL), "--design", str(design), "--native", str(native),
         "--a3m", str(a3m), "--out", str(out_json), "--usalign_bin", str(USALIGN)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(out_json.read_text()).get("metrics", {})


def plddt_from_summary(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    row = df.iloc[0]
    rows = []
    for c in range(11):
        col = f"cycle_{c}_plddt"
        if col in row.index and pd.notna(row[col]):
            rows.append({"cycle": c, "plddt": float(row[col])})
    return pd.DataFrame(rows)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--root", type=Path, default=None, help="结果根目录，默认 test_scaffords_5.20")
    ap.add_argument("--tag", default="", help="输出文件名后缀，如 top10_mpnn1500")
    ap.add_argument("--mean-label", default="平均")
    args = ap.parse_args()
    ids = args.ids if args.ids else DEFAULT_IDS
    root = args.root if args.root else ROOT
    suffix = f"_{args.tag}" if args.tag else ""

    detail_rows: list[dict] = []
    summary_rows: list[dict] = []

    for pid in ids:
        run_dir = root / pid / "0_protein_hunter_design" / "run_0"
        native = SCAFFOLDS / pid / f"{pid}.pdb"
        a3m = root / pid / f"{pid}.a3m"
        summary_csv = root / pid / "summary_all_runs.csv"
        eval_dir = root / pid / "eval_all_cycles"
        eval_dir.mkdir(parents=True, exist_ok=True)

        pdfs = sorted(
            [p for p in run_dir.glob(f"{pid}_run_0_predicted_cycle_*.pdb") if CYCLE_RE.search(p.name)],
            key=lambda p: int(CYCLE_RE.search(p.name).group(1)),
        )
        if not pdfs:
            print(f"warn: no cycle pdbs for {pid}", file=sys.stderr)
            continue

        plddt_df = plddt_from_summary(summary_csv)

        for pdb in pdfs:
            cyc = int(CYCLE_RE.search(pdb.name).group(1))
            out_json = eval_dir / f"{pid}_cycle_{cyc}_eval.json"
            if out_json.exists():
                report = json.loads(out_json.read_text())
                m = report.get("metrics", {})
            else:
                m = eval_one(pdb, native, a3m, out_json)
                report = json.loads(out_json.read_text())
            plddt_row = plddt_df[plddt_df["cycle"] == cyc]
            plddt = float(plddt_row["plddt"].iloc[0]) if len(plddt_row) else None
            detail_rows.append({
                "PDB": pid,
                "cycle": cyc,
                "structure": pdb.name,
                "RMSD (Å)": m.get("Scaffold_RMSD"),
                "pLDDT": plddt,
                "TM-score": m.get("Scaffold_TM_score"),
                "lDDT (vs native)": m.get("Scaffold_lDDT"),
                "评估残基数": report.get("residues_evaluated"),
            })

        ddf = pd.DataFrame([r for r in detail_rows if r["PDB"] == pid])
        n_res = int(round(ddf["评估残基数"].astype(float).mean()))
        summary_rows.append({
            "PDB ID": pid,
            "RMSD (Å)": round(ddf["RMSD (Å)"].astype(float).mean(), 4),
            "TM-score": round(ddf["TM-score"].astype(float).mean(), 4),
            "lDDT": round(ddf["pLDDT"].astype(float).mean(), 4),
            "评估残基数": n_res,
            "参与平均的 cycle 数": len(ddf),
        })
        print(f"{pid}: evaluated {len(ddf)} cycles (0–10)")

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    num_cols = ["RMSD (Å)", "TM-score", "lDDT"]
    mean_row = {
        "PDB ID": args.mean_label,
        "评估残基数": "—",
        "参与平均的 cycle 数": "—",
    }
    for c in num_cols:
        mean_row[c] = round(summary[c].astype(float).mean(), 4)
    summary = pd.concat([summary, pd.DataFrame([mean_row])], ignore_index=True)

    detail_path = root / f"metrics_all_cycles_detail{suffix}.csv"
    summary_path = root / f"metrics_all_cycles_summary{suffix}.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== 汇总（全部 cycle 0–10 平均，非 best）===")
    print(summary.to_string(index=False))
    print(f"\n明细: {detail_path}")
    print(f"汇总: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# 示例:
# python collect_all_cycles_metrics.py --tag top10_mpnn1500 \
#   --ids 8U0X 8IQC 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX --mean-label 平均
