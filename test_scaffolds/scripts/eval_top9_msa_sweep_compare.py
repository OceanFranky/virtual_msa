#!/usr/bin/env python3
"""Top9 × mpnn MSA：对 run_0 全部 cycle PDB（非 best）算 RMSD / TM / lDDT，并做组间对比表。"""

from __future__ import annotations

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
MSA_NS = [250, 1000, 1500, 2000, 3000, 5000]
CYCLE_RE = re.compile(r"_predicted_cycle_(\d+)\.pdb$")


def resolve_a3m(scaffold_dir: Path, pid: str, msa_n: int) -> Path:
    if msa_n == 3000:
        for tag in ("_mpnn3000_top", "_mpnn3000_unique"):
            p = scaffold_dir / f"{pid}{tag}.a3m"
            if p.exists():
                return p
        return scaffold_dir / f"{pid}_mpnn3000_unique.a3m"
    if msa_n == 5000:
        return scaffold_dir / f"{pid}_mpnn5000_unique.a3m"
    return scaffold_dir / f"{pid}_mpnn{msa_n}_top.a3m"


def results_root(msa_n: int) -> Path:
    return REPO / f"results_boltz_top9_mpnn{msa_n}"


def eval_dir_for_msa(msa_n: int) -> Path:
    """与 boltz_auto_pipeline 一致：results_eval_boltz_top9_mpnn{N}_mpnn{N}"""
    return REPO / f"results_eval_boltz_top9_mpnn{msa_n}_mpnn{msa_n}"


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


def collect_msa_group(msa_n: int, ids: list[str], force: bool = False) -> pd.DataFrame:
    root = results_root(msa_n)
    rows: list[dict] = []

    for pid in ids:
        run_dir = root / pid / "0_protein_hunter_design" / "run_0"
        if not run_dir.is_dir():
            print(f"  skip {pid}: no run_0 ({msa_n})", file=sys.stderr)
            continue

        native = SCAFFOLDS / pid / f"{pid}.pdb"
        a3m = resolve_a3m(SCAFFOLDS / pid, pid, msa_n)
        if not native.exists() or not a3m.exists():
            print(f"  skip {pid}: missing native/a3m", file=sys.stderr)
            continue

        eval_dir = eval_dir_for_msa(msa_n) / pid
        eval_dir.mkdir(parents=True, exist_ok=True)

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
            out_json = eval_dir / f"{pid}_run_0_cycle_{cyc}_eval.json"
            if out_json.exists() and not force:
                m = json.loads(out_json.read_text()).get("metrics", {})
            else:
                try:
                    m = eval_one(pdb, native, a3m, out_json)
                except subprocess.CalledProcessError as e:
                    print(f"  eval fail {pid} cycle {cyc}: {e}", file=sys.stderr)
                    continue

            rows.append(
                {
                    "pdb_id": pid,
                    "msa_n": msa_n,
                    "run_id": f"0_cycle_{cyc}",
                    "cycle": cyc,
                    "design_pdb": str(pdb),
                    "native_pdb": str(native),
                    "a3m": str(a3m),
                    "eval_json": str(out_json),
                    "structure": pdb.name,
                    "Scaffold_RMSD": m.get("Scaffold_RMSD"),
                    "Scaffold_TM_score": m.get("Scaffold_TM_score"),
                    "Scaffold_lDDT": m.get("Scaffold_lDDT"),
                    "eval_error": "",
                }
            )
        print(f"  mpnn{msa_n} {pid}: {len([r for r in rows if r['pdb_id']==pid])} cycles")

    return pd.DataFrame(rows)


def write_msa_summary_csv(msa_n: int, detail: pd.DataFrame) -> Path:
    """写入 results_eval_boltz_top9_mpnn{N}_mpnn{N}/summary.csv"""
    eval_root = eval_dir_for_msa(msa_n)
    eval_root.mkdir(parents=True, exist_ok=True)
    cols = [
        "pdb_id",
        "msa_n",
        "run_id",
        "design_pdb",
        "native_pdb",
        "a3m",
        "eval_json",
        "Scaffold_RMSD",
        "Scaffold_TM_score",
        "Scaffold_lDDT",
        "eval_error",
    ]
    out = eval_root / "summary.csv"
    detail[cols].to_csv(out, index=False)
    return out


def per_pdb_summary(detail: pd.DataFrame) -> pd.DataFrame:
    g = detail.groupby(["pdb_id", "msa_n"], as_index=False)
    return g.agg(
        n_cycles=("cycle", "count"),
        RMSD_mean=("Scaffold_RMSD", "mean"),
        RMSD_std=("Scaffold_RMSD", "std"),
        TM_mean=("Scaffold_TM_score", "mean"),
        TM_std=("Scaffold_TM_score", "std"),
        lDDT_mean=("Scaffold_lDDT", "mean"),
        lDDT_std=("Scaffold_lDDT", "std"),
    )


def build_wide_compare(per_pdb: pd.DataFrame) -> pd.DataFrame:
    msa_list = sorted(per_pdb["msa_n"].unique())
    rows = []
    for pid in sorted(per_pdb["pdb_id"].unique()):
        row = {"pdb_id": pid}
        for msa_n in msa_list:
            sub = per_pdb[(per_pdb["pdb_id"] == pid) & (per_pdb["msa_n"] == msa_n)]
            prefix = f"mpnn{msa_n}"
            if sub.empty:
                row[f"{prefix}_n"] = 0
                row[f"{prefix}_RMSD"] = None
                row[f"{prefix}_TM"] = None
                row[f"{prefix}_lDDT"] = None
            else:
                r = sub.iloc[0]
                row[f"{prefix}_n"] = int(r["n_cycles"])
                row[f"{prefix}_RMSD"] = round(r["RMSD_mean"], 4)
                row[f"{prefix}_TM"] = round(r["TM_mean"], 4)
                row[f"{prefix}_lDDT"] = round(r["lDDT_mean"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def build_group_summary(per_pdb: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for msa_n in sorted(per_pdb["msa_n"].unique()):
        sub = per_pdb[per_pdb["msa_n"] == msa_n]
        if sub.empty:
            continue
        rows.append(
            {
                "msa_n": msa_n,
                "n_pdb": len(sub),
                "RMSD_mean": round(sub["RMSD_mean"].mean(), 4),
                "TM_mean": round(sub["TM_mean"].mean(), 4),
                "lDDT_mean": round(sub["lDDT_mean"].mean(), 4),
                "RMSD_std_across_pdb": round(sub["RMSD_mean"].std(), 4)
                if len(sub) > 1
                else 0,
            }
        )
    return pd.DataFrame(rows)


def write_pivots(per_pdb: pd.DataFrame, out_dir: Path) -> None:
    for metric, col in [
        ("RMSD", "RMSD_mean"),
        ("TM", "TM_mean"),
        ("lDDT", "lDDT_mean"),
    ]:
        p = per_pdb.pivot(index="pdb_id", columns="msa_n", values=col)
        p.columns = [f"mpnn{c}" for c in p.columns]
        p.sort_index().to_csv(out_dir / f"pivot_{metric}.csv")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=TOP9)
    ap.add_argument("--msa", nargs="*", type=int, default=MSA_NS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results_boltz_top9_eval")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_detail = []

    for msa_n in args.msa:
        print(f"\n=== mpnn{msa_n} ===")
        df = collect_msa_group(msa_n, args.ids, force=args.force)
        if not df.empty:
            all_detail.append(df)
            df.to_csv(args.out_dir / f"detail_mpnn{msa_n}_all_cycles.csv", index=False)
            summary_path = write_msa_summary_csv(msa_n, df)
            print(f"  summary: {summary_path}")

    if not all_detail:
        print("no data", file=sys.stderr)
        return 1

    detail = pd.concat(all_detail, ignore_index=True)
    detail.to_csv(args.out_dir / "detail_all_msa_all_cycles.csv", index=False)

    per_pdb = per_pdb_summary(detail)
    per_pdb.to_csv(args.out_dir / "per_pdb_per_msa_mean.csv", index=False)

    wide = build_wide_compare(per_pdb)
    wide.to_csv(args.out_dir / "compare_wide_pdb_x_msa.csv", index=False)

    group = build_group_summary(per_pdb)
    group.to_csv(args.out_dir / "compare_summary_by_msa.csv", index=False)

    write_pivots(per_pdb, args.out_dir)

    report = args.out_dir / "compare_report.md"
    with report.open("w") as f:
        f.write("# Top9 MSA 对比（全 cycle 0–10 平均，非 best）\n\n")
        f.write("## 按 MSA 组汇总\n\n")
        f.write(group.to_csv(index=False))
        f.write("\n## 宽表\n\n")
        f.write(wide.to_csv(index=False))

    print("\n=== 按 MSA 组汇总 ===")
    print(group.to_string(index=False))
    print(f"\n输出: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
