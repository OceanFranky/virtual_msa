#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC_TSV = ROOT / "protein-ligand" / "pl_external_cases.tsv"
PL_EXTERNAL = Path(__file__).resolve().parent / "pl_external_to_scaffolds.py"
PYTHON_BIN = Path("/home/yumo/miniconda3/envs/proteinhunter/bin/python")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the current built-in Protein-Ligand selection set through the full external-case flow."
    )
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    args = ap.parse_args()

    if not SPEC_TSV.exists():
        raise RuntimeError(f"missing built-in PL spec TSV: {SPEC_TSV}")

    subprocess.run(
        [
            str(PYTHON_BIN),
            str(PL_EXTERNAL),
            "--spec-tsv",
            str(SPEC_TSV),
            "--gpu",
            str(args.gpu),
        ],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
