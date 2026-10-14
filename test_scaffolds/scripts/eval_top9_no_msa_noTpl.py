#!/usr/bin/env python3
"""Top9 无 MSA / 无 template Boltz：RMSD / TM / lDDT（对齐用 native {ID}.a3m）。"""

from __future__ import annotations

import argparse
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

TOP9 = [
    "8U0X", "8IQC", "8BAQ", "8Q8I", "8BAU", "8R2C", "8SV7", "8EO5", "8BUX",
]

RESULTS = REPO / "results_boltz_top9_no_msa_noTpl"
EVAL_OUT = REPO / "results_eval_boltz_top9_no_msa_noTpl"
CYCLE_RE = re.compile(r"_predicted_cycle_(\d+)\.pdb$")


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ids", nargs="*", default=TOP9)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "results_boltz_top9_eval_no_msa_noTpl",
    )
    args = ap.parse_args()

    rows: list[dict] = []
    for pid in args.ids:
        run_dir = RESULTS / pid / "0_protein_hunter_design" / "run_0"
        if not run_dir.is_dir():
            print(f"skip {pid}: no run_0", file=sys.stderr)
            continue
        native = SCAFFOLDS / pid / f"{pid}.pdb"
        a3m = SCAFFOLDS / pid / f"{pid}.a3m"
        if not native.exists() or not a3m.exists():
            print(f"skip {pid}: missing native/a3m", file=sys.stderr)
            continue

        ed = EVAL_OUT / pid
        ed.mkdir(parents=True, exist_ok=True)
        pdbs = sorted(
            [
                p
                for p in run_dir.glob(f"{pid}_run_0_predicted_cycle_*.pdb")
                if CYCLE_RE.search(p.name)
            ],
            key=lambda p: int(CYCLE_RE.search(p.name).group(1)),
        )
        for pdb in pdbs:
            cyc = int(CYCLE_RE.search(pdb.name).group(1))
            out_json = ed / f"{pid}_cycle_{cyc}_eval.json"
            if out_json.exists() and not args.force:
                m = json.loads(out_json.read_text()).get("metrics", {})
            else:
                try:
                    m = eval_one(pdb, native, a3m, out_json)
                except subprocess.CalledProcessError as e:
                    print(f"eval fail {pid} c{cyc}: {e}", file=sys.stderr)
                    continue
            rows.append(
                {
                    "pdb_id": pid,
                    "condition": "no_msa_noTpl",
                    "native_pdb": str(native.resolve()),
                    "a3m": str(a3m.resolve()),
                    "design_pdb": str(pdb.resolve()),
                    "cycle": cyc,
                    "Scaffold_RMSD": m.get("Scaffold_RMSD"),
                    "Scaffold_TM_score": m.get("Scaffold_TM_score"),
                    "Scaffold_lDDT": m.get("Scaffold_lDDT"),
                }
            )
        print(f"  {pid}: {len([r for r in rows if r['pdb_id']==pid])} cycles")

    if not rows:
        print("no data", file=sys.stderr)
        return 1

    detail = pd.DataFrame(rows)
    per_pdb = (
        detail.groupby("pdb_id", as_index=False)
        .agg(
            n_cycles=("cycle", "count"),
            RMSD_mean=("Scaffold_RMSD", "mean"),
            TM_mean=("Scaffold_TM_score", "mean"),
            lDDT_mean=("Scaffold_lDDT", "mean"),
        )
    )
    group = pd.DataFrame(
        [
            {
                "condition": "no_msa_noTpl",
                "n_pdb": len(per_pdb),
                "RMSD_mean": round(per_pdb["RMSD_mean"].mean(), 4),
                "TM_mean": round(per_pdb["TM_mean"].mean(), 4),
                "lDDT_mean": round(per_pdb["lDDT_mean"].mean(), 4),
            }
        ]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out_dir / "detail_all_cycles.csv", index=False)
    per_pdb.to_csv(args.out_dir / "per_pdb_mean.csv", index=False)
    group.to_csv(args.out_dir / "group_summary.csv", index=False)
    group.to_csv(EVAL_OUT / "summary_metrics.csv", index=False)

    print("\n=== no_msa no_template (Top9 平均) ===")
    print(group.to_string(index=False))
    print(f"\nWrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
