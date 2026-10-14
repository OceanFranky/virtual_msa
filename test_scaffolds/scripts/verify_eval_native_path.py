#!/usr/bin/env python3
"""核对评测是否使用 test_scaffolds/{ID}/{ID}.pdb，并打印每 cycle RMSD。"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO = Path("/data/wenhao/Protein-Hunter-server")
SCAFFOLDS = REPO / "test_scaffolds"
EVAL = REPO / "DATA/eval_scaffold.py"
PY = "/data/wenhao/miniconda3/envs/proteinhunter/bin/python"
USALIGN = "/data/wenhao/USalign/USalign"

TEMP_GROUPS = [
    ("t0.1", "mpnn1000", "_mpnn1000_top"),
    ("t0.01", "mpnn1000_t0.01", "_mpnn1000_t0.01"),
    ("t0.2", "mpnn1000_t0.2", "_mpnn1000_t0.2"),
    ("t0.5", "mpnn1000_t0.5", "_mpnn1000_t0.5"),
    ("t1.0", "mpnn1000_t1.0", "_mpnn1000_t1.0"),
]
CYCLE_RE = re.compile(r"_predicted_cycle_(\d+)\.pdb$")


def main() -> None:
    pid = "8U0X"
    native = SCAFFOLDS / pid / f"{pid}.pdb"
    assert native.exists(), native
    print(f"NATIVE (fixed): {native}\n")

    rows = []
    for temp, dir_suf, tag in TEMP_GROUPS:
        root = REPO / f"results_boltz_top9_{dir_suf}"
        a3m = SCAFFOLDS / pid / f"{pid}{tag}.a3m"
        run_dir = root / pid / "0_protein_hunter_design/run_0"
        for pdb in sorted(run_dir.glob(f"{pid}_run_0_predicted_cycle_*.pdb")):
            m = CYCLE_RE.search(pdb.name)
            if not m:
                continue
            cyc = int(m.group(1))
            out = Path(f"/tmp/verify_{pid}_{temp}_c{cyc}.json")
            subprocess.run(
                [
                    PY,
                    str(EVAL),
                    "--design",
                    str(pdb),
                    "--native",
                    str(native),
                    "--a3m",
                    str(a3m),
                    "--out",
                    str(out),
                    "--usalign_bin",
                    USALIGN,
                ],
                check=True,
                capture_output=True,
            )
            met = json.loads(out.read_text())["metrics"]
            rows.append(
                (temp, cyc, met["Scaffold_RMSD"], met["Scaffold_TM_score"], met["Scaffold_lDDT"])
            )

    print(f"{'temp':<8} {'cyc':>3} {'RMSD':>8} {'TM':>8} {'lDDT':>8}")
    print("-" * 40)
    for temp, cyc, r, t, l in rows:
        print(f"{temp:<8} {cyc:>3} {r:>8.4f} {t:>8.4f} {l:>8.4f}")

    from collections import defaultdict

    acc = defaultdict(list)
    for temp, cyc, r, t, l in rows:
        acc[temp].append(r)
    print("\n=== mean RMSD (8U0X, cycles 0-10) ===")
    for temp, _, _ in TEMP_GROUPS:
        vals = acc[temp]
        print(f"{temp}: {sum(vals)/len(vals):.4f}  n={len(vals)}")


if __name__ == "__main__":
    main()
