#!/usr/bin/env python3
"""Top9 × mpnn1000 全采样温度 Boltz：RMSD / TM / lDDT（cycle 0–10 平均）。"""

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

# (显示温度, results 目录名后缀, MSA a3m 后缀)
# 按采样温度从低到高
TEMP_GROUPS: list[tuple[str, str, str]] = [
    ("t0.01", "mpnn1000_t0.01", "_mpnn1000_t0.01"),
    ("t0.1", "mpnn1000", "_mpnn1000_top"),  # MPNN T=0.1 top
    ("t0.2", "mpnn1000_t0.2", "_mpnn1000_t0.2"),
    ("t0.5", "mpnn1000_t0.5", "_mpnn1000_t0.5"),
    ("t0.7", "mpnn1000_t0.7", "_mpnn1000_t0.7"),
    ("t1.0", "mpnn1000_t1.0", "_mpnn1000_t1.0"),
    ("t1.5", "mpnn1000_t1.5", "_mpnn1000_t1.5"),
    ("t2.0", "mpnn1000_t2.0", "_mpnn1000_t2.0"),
]

CYCLE_RE = re.compile(r"_predicted_cycle_(\d+)\.pdb$")


def results_root(dir_suffix: str) -> Path:
    return REPO / f"results_boltz_top9_{dir_suffix}"


def eval_root(dir_suffix: str) -> Path:
    return REPO / f"results_eval_boltz_top9_{dir_suffix}"


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


def collect_group(
    temp_label: str,
    dir_suffix: str,
    msa_tag: str,
    ids: list[str],
    force: bool,
) -> pd.DataFrame:
    root = results_root(dir_suffix)
    rows: list[dict] = []

    for pid in ids:
        run_dir = root / pid / "0_protein_hunter_design" / "run_0"
        if not run_dir.is_dir():
            print(f"  skip {pid}: no run_0 ({temp_label})", file=sys.stderr)
            continue

        native = SCAFFOLDS / pid / f"{pid}.pdb"
        a3m = SCAFFOLDS / pid / f"{pid}{msa_tag}.a3m"
        if not native.exists() or not a3m.exists():
            print(f"  skip {pid}: missing native/a3m {a3m}", file=sys.stderr)
            continue

        ed = eval_root(dir_suffix) / pid
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
                    "temp": temp_label,
                    "dir_suffix": dir_suffix,
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
        print(f"  {temp_label} {pid}: {n} cycles")

    return pd.DataFrame(rows)


def per_pdb_summary(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby(["pdb_id", "temp"], as_index=False)
        .agg(
            n_cycles=("cycle", "count"),
            RMSD_mean=("Scaffold_RMSD", "mean"),
            TM_mean=("Scaffold_TM_score", "mean"),
            lDDT_mean=("Scaffold_lDDT", "mean"),
        )
    )


def build_wide(per_pdb: pd.DataFrame, metric: str, col: str) -> pd.DataFrame:
    order = [t for t, _, _ in TEMP_GROUPS]
    rows = []
    for pid in sorted(per_pdb["pdb_id"].unique()):
        row = {"pdb_id": pid}
        for temp in order:
            sub = per_pdb[(per_pdb["pdb_id"] == pid) & (per_pdb["temp"] == temp)]
            key = temp.replace(".", "_")
            if sub.empty:
                row[metric + "_" + key] = None
            else:
                row[metric + "_" + key] = round(float(sub.iloc[0][col]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def build_group_summary(per_pdb: pd.DataFrame) -> pd.DataFrame:
    order = [t for t, _, _ in TEMP_GROUPS]
    rows = []
    for temp in order:
        sub = per_pdb[per_pdb["temp"] == temp]
        if sub.empty:
            continue
        rows.append(
            {
                "temp": temp,
                "n_pdb": len(sub),
                "RMSD_mean": round(sub["RMSD_mean"].mean(), 4),
                "TM_mean": round(sub["TM_mean"].mean(), 4),
                "lDDT_mean": round(sub["lDDT_mean"].mean(), 4),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=TOP9)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "results_boltz_top9_eval_mpnn1000_temp",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_parts: list[pd.DataFrame] = []

    for temp_label, dir_suffix, msa_tag in TEMP_GROUPS:
        print(f"\n=== {temp_label}  ({dir_suffix}) ===")
        df = collect_group(temp_label, dir_suffix, msa_tag, args.ids, args.force)
        if not df.empty:
            all_parts.append(df)
            df.to_csv(args.out_dir / f"detail_{temp_label}.csv", index=False)
            eval_root(dir_suffix).mkdir(parents=True, exist_ok=True)
            df.to_csv(eval_root(dir_suffix) / "summary_metrics.csv", index=False)

    if not all_parts:
        print("no data", file=sys.stderr)
        return 1

    detail = pd.concat(all_parts, ignore_index=True)
    detail.to_csv(args.out_dir / "detail_all_temps_all_cycles.csv", index=False)

    per_pdb = per_pdb_summary(detail)
    per_pdb.to_csv(args.out_dir / "per_pdb_per_temp_mean.csv", index=False)

    group = build_group_summary(per_pdb)
    group.to_csv(args.out_dir / "compare_summary_by_temp.csv", index=False)

    for metric, col in [
        ("RMSD", "RMSD_mean"),
        ("TM", "TM_mean"),
        ("lDDT", "lDDT_mean"),
    ]:
        build_wide(per_pdb, metric, col).to_csv(
            args.out_dir / f"compare_wide_{metric}.csv", index=False
        )

    wide = per_pdb.pivot(index="pdb_id", columns="temp", values="RMSD_mean")
    wide = wide[[t for t, _, _ in TEMP_GROUPS if t in wide.columns]]
    wide.round(4).to_csv(args.out_dir / "pivot_RMSD.csv")

    report = args.out_dir / "compare_report.md"
    with report.open("w") as f:
        f.write("# Top9 mpnn1000 采样温度对比（cycle 0–10 平均）\n\n")
        f.write("- t0.1 = `results_boltz_top9_mpnn1000` + `_mpnn1000_top` MSA\n")
        f.write("- 全温度 = `results_boltz_top9_mpnn1000` / `mpnn1000_t*`（t0.01–t2.0）\n\n")
        f.write("## 组平均\n\n")
        f.write(group.to_csv(index=False))
        f.write("\n\n## per PDB × temp\n\n")
        f.write(per_pdb.to_csv(index=False))

    print("\n=== 按采样温度组汇总 ===")
    print(group.to_string(index=False))
    print(f"\n输出: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
