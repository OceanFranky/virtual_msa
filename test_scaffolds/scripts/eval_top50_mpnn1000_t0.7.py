#!/usr/bin/env python3
"""Top50 × mpnn1000_t0.7 Boltz：RMSD / TM / lDDT（cycle 0–10 平均）。"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path("/data/wenhao/Protein-Hunter-server")
SCAFFOLDS = REPO / "test_scaffolds"
EVAL = REPO / "DATA" / "eval_scaffold.py"
USALIGN = Path("/data/wenhao/USalign/USalign")
PY = "/data/wenhao/miniconda3/envs/proteinhunter/bin/python"
CSV = REPO / "top50_from_top100_prefer_short.csv"

DIR_SUFFIX = "top50_mpnn1000_t0.7"
MSA_TAG = "_mpnn1000_t0.7"
RESULTS = REPO / f"results_boltz_{DIR_SUFFIX}"
EVAL_ROOT = REPO / f"results_eval_boltz_{DIR_SUFFIX}"
OUT_DIR = REPO / f"results_boltz_{DIR_SUFFIX}_eval"

CYCLE_RE = re.compile(r"_predicted_cycle_(\d+)\.pdb$")


def load_top50_ids() -> list[str]:
    ids = []
    with CSV.open() as f:
        for row in csv.DictReader(f):
            ids.append(row["PDB_ID"].strip())
    return ids


def eval_one(design: Path, native: Path, a3m: Path, out_json: Path) -> dict:
    subprocess.run(
        [
            PY,
            str(EVAL),
            "--design",
            str(design),
            "--native",
            str(native),
            "--a3m",
            str(a3m),
            "--out",
            str(out_json),
            "--usalign_bin",
            str(USALIGN),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out_json.read_text()).get("metrics", {})


def collect(ids: list[str], force: bool) -> pd.DataFrame:
    rows: list[dict] = []
    for pid in ids:
        run_dir = RESULTS / pid / "0_protein_hunter_design" / "run_0"
        if not run_dir.is_dir():
            print(f"  skip {pid}: no run_0", file=sys.stderr)
            continue
        native = SCAFFOLDS / pid / f"{pid}.pdb"
        a3m = SCAFFOLDS / pid / f"{pid}{MSA_TAG}.a3m"
        if not native.exists() or not a3m.exists():
            print(f"  skip {pid}: missing native/a3m", file=sys.stderr)
            continue
        ed = EVAL_ROOT / pid
        ed.mkdir(parents=True, exist_ok=True)
        pdbs = sorted(
            [
                p
                for p in run_dir.glob(f"{pid}_run_0_predicted_cycle_*.pdb")
                if CYCLE_RE.search(p.name)
            ],
            key=lambda p: int(CYCLE_RE.search(p.name).group(1)),
        )
        if not pdbs:
            print(f"  skip {pid}: no cycle pdbs", file=sys.stderr)
            continue
        for pdb in pdbs:
            cyc = int(CYCLE_RE.search(pdb.name).group(1))
            out_json = ed / f"{pid}_cycle_{cyc}_eval.json"
            if out_json.exists() and not force:
                m = json.loads(out_json.read_text()).get("metrics", {})
            else:
                try:
                    m = eval_one(pdb, native, a3m, out_json)
                except subprocess.CalledProcessError as e:
                    print(f"  eval fail {pid} c{cyc}: {e.stderr[:200]}", file=sys.stderr)
                    continue
            rows.append(
                {
                    "pdb_id": pid,
                    "native_pdb": str(native),
                    "a3m": str(a3m),
                    "design_pdb": str(pdb),
                    "cycle": cyc,
                    "Scaffold_RMSD": m.get("Scaffold_RMSD"),
                    "Scaffold_TM_score": m.get("Scaffold_TM_score"),
                    "Scaffold_lDDT": m.get("Scaffold_lDDT"),
                }
            )
        n = len([r for r in rows if r["pdb_id"] == pid])
        print(f"  {pid}: {n} cycles")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ids = load_top50_ids()
    print(f"Top50 CSV: {len(ids)} ids, evaluating t0.7 Boltz...")
    detail = collect(ids, args.force)
    if detail.empty:
        print("no data", file=sys.stderr)
        return 1

    detail.to_csv(args.out_dir / "detail_all_cycles.csv", index=False)

    per_pdb = (
        detail.groupby("pdb_id", as_index=False)
        .agg(
            n_cycles=("cycle", "count"),
            RMSD_mean=("Scaffold_RMSD", "mean"),
            TM_mean=("Scaffold_TM_score", "mean"),
            lDDT_mean=("Scaffold_lDDT", "mean"),
            RMSD_min=("Scaffold_RMSD", "min"),
            TM_max=("Scaffold_TM_score", "max"),
            lDDT_max=("Scaffold_lDDT", "max"),
        )
        .sort_values("RMSD_mean")
    )
    per_pdb.to_csv(args.out_dir / "per_pdb_mean.csv", index=False)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    per_pdb.to_csv(EVAL_ROOT / "summary_metrics.csv", index=False)

    summary = {
        "n_pdb": len(per_pdb),
        "n_cycles_per_pdb_expected": 11,
        "RMSD_mean": round(per_pdb["RMSD_mean"].mean(), 4),
        "TM_mean": round(per_pdb["TM_mean"].mean(), 4),
        "lDDT_mean": round(per_pdb["lDDT_mean"].mean(), 4),
        "RMSD_median": round(per_pdb["RMSD_mean"].median(), 4),
        "TM_median": round(per_pdb["TM_mean"].median(), 4),
        "lDDT_median": round(per_pdb["lDDT_mean"].median(), 4),
    }
    pd.DataFrame([summary]).to_csv(args.out_dir / "group_summary.csv", index=False)

    with (args.out_dir / "report.md").open("w") as f:
        f.write("# Top50 mpnn1000_t0.7 Boltz 评测（cycle 0–10 平均）\n\n")
        f.write(f"- 结果: `{RESULTS}`\n")
        f.write(f"- 完成 PDB: {summary['n_pdb']}/50（缺 MSA: 8B2S, 8QOH）\n\n")
        f.write("## 组汇总\n\n")
        f.write(pd.DataFrame([summary]).to_csv(index=False))
        f.write("\n\n## per PDB\n\n")
        f.write(per_pdb.to_csv(index=False))

    print("\n=== Top50 t0.7 组汇总 ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n输出: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
