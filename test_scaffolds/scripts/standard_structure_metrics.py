#!/usr/bin/env python3
"""
标准结构指标（不依赖 eval_scaffold / collect 脚本）：
  - RMSD / TM-score：US-align，TM 以 reference(native) 长度归一化
  - lDDT：Cα 几何 lDDT（15Å 邻域，阈值 0.5/1/2/4Å，与 AlphaFold 惯例一致）
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
from pathlib import Path

IDS = ["8U0X", "8IQC", "8BAQ", "8Q8I", "8BAU", "8R2C", "8SV7", "8EO5", "8BUX"]
CYCLE_RE = re.compile(r"_predicted_cycle_(\d+)\.pdb$")
USALIGN = Path("/data/wenhao/USalign/USalign")
NATIVE_ROOT = Path("/data/wenhao/Protein-Hunter-server/test_scaffolds")


def parse_ca(pdb: Path) -> dict[tuple[str, int], tuple[float, float, float]]:
    ca: dict[tuple[str, int], tuple[float, float, float]] = {}
    with pdb.open() as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            key = (line[21], int(line[22:26]))
            ca[key] = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
    return ca


def usalign_metrics(design: Path, native: Path) -> tuple[float, float]:
    proc = subprocess.run(
        [str(USALIGN), str(design), str(native)],
        capture_output=True,
        text=True,
        check=True,
    )
    rmsd = None
    tm = None
    for line in proc.stdout.splitlines():
        if "RMSD=" in line and "Seq_ID" in line:
            m = re.search(r"RMSD=\s*([0-9.]+)", line)
            if m:
                rmsd = float(m.group(1))
        if "TM-score=" in line and "Structure_2" in line:
            m = re.search(r"TM-score=\s*([0-9.]+)", line)
            if m:
                tm = float(m.group(1))
    if rmsd is None or tm is None:
        raise RuntimeError(f"US-align parse failed:\n{proc.stdout}")
    return rmsd, tm


def geometric_lddt(
    design_ca: dict, native_ca: dict, cutoff: float = 15.0
) -> tuple[float, int]:
    shared = sorted(set(design_ca) & set(native_ca), key=lambda k: (k[0], k[1]))
    if len(shared) < 2:
        return float("nan"), 0
    thresholds = (0.5, 1.0, 2.0, 4.0)
    scores: list[float] = []
    for i, ki in enumerate(shared):
        ni, di = native_ca[ki], design_ca[ki]
        nbrs = []
        for j, kj in enumerate(shared):
            if i == j:
                continue
            d_ref = math.dist(ni, native_ca[kj])
            if d_ref < cutoff:
                nbrs.append((kj, d_ref))
        if not nbrs:
            continue
        hits = checks = 0
        for kj, d_ref in nbrs:
            d_mod = math.dist(di, design_ca[kj])
            delta = abs(d_mod - d_ref)
            for t in thresholds:
                checks += 1
                if delta < t:
                    hits += 1
        scores.append(hits / checks)
    if not scores:
        return float("nan"), 0
    return sum(scores) / len(scores), len(scores)


def eval_structure(design: Path, native: Path) -> dict:
    d_ca, n_ca = parse_ca(design), parse_ca(native)
    rmsd, tm = usalign_metrics(design, native)
    lddt, n_lddt = geometric_lddt(d_ca, n_ca)
    return {
        "RMSD": rmsd,
        "TM_score": tm,
        "lDDT": lddt,
        "n_common_ca": len(set(d_ca) & set(n_ca)),
        "n_lddt_res": n_lddt,
    }


def collect_group(root: Path) -> list[dict]:
    rows = []
    for pid in IDS:
        native = NATIVE_ROOT / pid / f"{pid}.pdb"
        run_dir = root / pid / "0_protein_hunter_design/run_0"
        pdbs = sorted(
            [p for p in run_dir.glob(f"{pid}_run_0_predicted_cycle_*.pdb") if CYCLE_RE.search(p.name)],
            key=lambda p: int(CYCLE_RE.search(p.name).group(1)),
        )
        per = []
        for pdb in pdbs:
            cyc = int(CYCLE_RE.search(pdb.name).group(1))
            m = eval_structure(pdb, native)
            per.append({"cycle": cyc, **m})
        if not per:
            continue
        rows.append({
            "PDB ID": pid,
            "RMSD": round(sum(x["RMSD"] for x in per) / len(per), 4),
            "TM_score": round(sum(x["TM_score"] for x in per) / len(per), 4),
            "lDDT": round(sum(x["lDDT"] for x in per) / len(per), 4),
            "n_cycles": len(per),
            "n_common_ca": per[0]["n_common_ca"],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-tpl", type=Path,
                    default=Path("/data/wenhao/Protein-Hunter-server/test_scaffords_5.20_msa500_nativeQ_withTpl"))
    ap.add_argument("--no-tpl", type=Path,
                    default=Path("/data/wenhao/Protein-Hunter-server/test_scaffords_5.20_msa500_nativeQ_noTpl"))
    ap.add_argument("--out", type=Path,
                    default=Path("/data/wenhao/Protein-Hunter-server/test_scaffords_5.20_msa500_nativeQ_compare_standard.csv"))
    args = ap.parse_args()

    import pandas as pd

    a = pd.DataFrame(collect_group(args.with_tpl)).rename(
        columns=lambda c: c if c == "PDB ID" else f"{c}_withTpl"
    )
    b = pd.DataFrame(collect_group(args.no_tpl)).rename(
        columns=lambda c: c if c == "PDB ID" else f"{c}_noTpl"
    )
    cmp = a.merge(b, on="PDB ID", how="outer")
    for m in ("RMSD", "TM_score", "lDDT"):
        cmp[f"d{m}"] = (cmp[f"{m}_noTpl"] - cmp[f"{m}_withTpl"]).round(4)

    mean = {"PDB ID": "平均"}
    for c in cmp.columns:
        if c == "PDB ID":
            continue
        if cmp[c].dtype.kind in "fi":
            mean[c] = round(cmp[c].astype(float).mean(), 4)
    cmp = pd.concat([cmp, pd.DataFrame([mean])], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmp.to_csv(args.out, index=False)
    print(cmp.to_string(index=False))
    print(f"\nWrote {args.out}")
    print("\nMethods: RMSD/TM=US-align(design,native); lDDT=geometric CA lDDT 15Å 0.5/1/2/4Å")
    print(f"Native ref: {NATIVE_ROOT}/{{ID}}/{{ID}}.pdb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
