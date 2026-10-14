#!/usr/bin/env python3
"""Top9 × mpnn1000_t2.0 Boltz：cycle 0–10 平均 RMSD / TM / lDDT（同 t0.01 目录结构）。"""

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

DIR_SUFFIX = "mpnn1000_t2.0"
MSA_TAG = "_mpnn1000_t2.0"
TEMP_LABEL = "t2.0"

CYCLE_RE = re.compile(r"_predicted_cycle_(\d+)\.pdb$")


def results_root() -> Path:
    return REPO / f"results_boltz_top9_{DIR_SUFFIX}"


def eval_root() -> Path:
    return REPO / f"results_eval_boltz_top9_{DIR_SUFFIX}"


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
    root = results_root()
    rows: list[dict] = []

    for pid in ids:
        run_dir = root / pid / "0_protein_hunter_design" / "run_0"
        if not run_dir.is_dir():
            print(f"  skip {pid}: no run_0", file=sys.stderr)
            continue

        native = SCAFFOLDS / pid / f"{pid}.pdb"
        a3m = SCAFFOLDS / pid / f"{pid}{MSA_TAG}.a3m"
        if not native.exists() or not a3m.exists():
            print(f"  skip {pid}: missing native/a3m", file=sys.stderr)
            continue

        ed = eval_root() / pid
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
                    print(f"  eval fail {pid} c{cyc}: {e}", file=sys.stderr)
                    continue

            rows.append(
                {
                    "pdb_id": pid,
                    "temp": TEMP_LABEL,
                    "native_pdb": str(native.resolve()),
                    "a3m": str(a3m.resolve()),
                    "design_pdb": str(pdb.resolve()),
                    "cycle": cyc,
                    "Scaffold_RMSD": m.get("Scaffold_RMSD"),
                    "Scaffold_TM_score": m.get("Scaffold_TM_score"),
                    "Scaffold_lDDT": m.get("Scaffold_lDDT"),
                }
            )
        n = len([r for r in rows if r["pdb_id"] == pid])
        print(f"  {pid}: {n} cycles")

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ids", nargs="*", default=TOP9)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "results_boltz_top9_eval_mpnn1000_t2.0",
    )
    args = ap.parse_args()

    detail = collect(args.ids, args.force)
    if detail.empty:
        print("no eval rows", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
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
                "temp": TEMP_LABEL,
                "n_pdb": len(per_pdb),
                "RMSD_mean": per_pdb["RMSD_mean"].mean(),
                "TM_mean": per_pdb["TM_mean"].mean(),
                "lDDT_mean": per_pdb["lDDT_mean"].mean(),
            }
        ]
    )

    detail.to_csv(args.out_dir / "detail_all_cycles.csv", index=False)
    per_pdb.to_csv(args.out_dir / "per_pdb_mean.csv", index=False)
    group.to_csv(args.out_dir / "group_summary.csv", index=False)

    # 与 t0.01 同结构的 summary_metrics
    metrics_path = eval_root() / "summary_metrics.csv"
    group.to_csv(metrics_path, index=False)

    print(group.to_string(index=False))
    print(f"\nWrote {args.out_dir}")
    print(f"Eval JSONs under {eval_root()}")


if __name__ == "__main__":
    main()
