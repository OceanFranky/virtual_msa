#!/usr/bin/env python3
"""
Batch-run binding-ddg-predictor on iptm_select subfolders (wt_for_ddG.pdb + mut_for_ddG.pdb),
rank by predicted ddG, keep only ddG < 0 (favorable), and copy original designed PDBs from
Filtered_Good_PDBs_<case> into example_myself/ddg/<case>/<stem>/.

Convention: more negative ddG = stronger predicted binding improvement (sort ascending).
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

DDG_LINE = re.compile(r"Predicted ddG:\s*([+-]?\d+(?:\.\d+)?)", re.I)


def default_filtered_dir(select_dir: Path) -> Path:
    """.../iptm_select/<case> -> .../iptm/Filtered_Good_PDBs_<case>"""
    case = select_dir.name
    base = select_dir.parent.parent  # example_myself
    return base / "iptm" / f"Filtered_Good_PDBs_{case}"


def default_ddg_root(select_dir: Path) -> Path:
    return select_dir.parent.parent / "ddg"


def run_predict(
    python_exe: str,
    predictor_root: Path,
    predict_script: Path,
    model_path: Path,
    wt_pdb: Path,
    mut_pdb: Path,
    device: str,
) -> Tuple[Optional[float], str]:
    cmd = [
        python_exe,
        str(predict_script),
        str(wt_pdb),
        str(mut_pdb),
        "--model",
        str(model_path),
        "--device",
        device,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(predictor_root),
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        return None, f"predict exit {proc.returncode}: {out[-2000:]}"
    m = DDG_LINE.search(out)
    if not m:
        return None, f"no ddG in output: {out[-2000:]}"
    return float(m.group(1)), out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--select-dir",
        type=Path,
        required=True,
        help="e.g. .../iptm_select/1I8L (subdirs with wt_for_ddG.pdb / mut_for_ddG.pdb)",
    )
    ap.add_argument(
        "--filtered-dir",
        type=Path,
        default=None,
        help="Original designed PDBs (default: .../iptm/Filtered_Good_PDBs_<case>)",
    )
    ap.add_argument(
        "--ddg-root",
        type=Path,
        default=None,
        help="Output root (default: sibling of iptm_select -> ddg)",
    )
    ap.add_argument(
        "--predictor-root",
        type=Path,
        default=Path("/data/wenhao/binding-ddg-predictor-main"),
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=None,
        help="model.pt (default: <predictor-root>/data/model.pt)",
    )
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument(
        "--max-n",
        type=int,
        default=0,
        help="If >0, only process first N valid subdirs (for testing)",
    )
    args = ap.parse_args()

    select_dir = args.select_dir.resolve()
    if not select_dir.is_dir():
        print(f"Not a directory: {select_dir}", file=sys.stderr)
        sys.exit(1)

    case_id = select_dir.name
    filtered_dir = (
        args.filtered_dir.resolve()
        if args.filtered_dir
        else default_filtered_dir(select_dir)
    )
    ddg_root = args.ddg_root.resolve() if args.ddg_root else default_ddg_root(select_dir)
    predictor_root = args.predictor_root.resolve()
    model_path = (
        args.model.resolve()
        if args.model
        else (predictor_root / "data" / "model.pt")
    )
    predict_script = predictor_root / "scripts" / "predict.py"

    if not predict_script.is_file():
        print(f"Missing {predict_script}", file=sys.stderr)
        sys.exit(1)
    if not model_path.is_file():
        print(f"Missing model {model_path}", file=sys.stderr)
        sys.exit(1)

    subdirs = sorted(
        [p for p in select_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )
    rows: List[Tuple[str, float]] = []
    errors: List[Tuple[str, str]] = []

    n_done = 0
    for sub in subdirs:
        wt = sub / "wt_for_ddG.pdb"
        mut = sub / "mut_for_ddG.pdb"
        if not wt.is_file() or not mut.is_file():
            continue
        stem = sub.name
        ddG, msg = run_predict(
            args.python,
            predictor_root,
            predict_script,
            model_path,
            wt,
            mut,
            args.device,
        )
        if ddG is None:
            errors.append((stem, msg))
            print(f"[SKIP] {stem}: {msg[:200]}", file=sys.stderr)
            continue
        rows.append((stem, ddG))
        print(f"[OK] {stem}: ddG={ddG:.4f}")
        n_done += 1
        if args.max_n > 0 and n_done >= args.max_n:
            break

    favorable = [(s, g) for s, g in rows if g < 0]
    favorable.sort(key=lambda x: x[1])

    out_case = ddg_root / case_id
    out_case.mkdir(parents=True, exist_ok=True)

    rank_csv = out_case / "ddg_rankings_favorable.csv"
    with rank_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "stem", "predicted_ddg", "source_designed_pdb", "copied_ok"])
        for i, (stem, g) in enumerate(favorable, start=1):
            src = filtered_dir / f"{stem}.pdb"
            dest_dir = out_case / stem
            copied = False
            if src.is_file():
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest_dir / src.name)
                copied = True
            else:
                print(f"[WARN] missing Filtered_Good PDB: {src}", file=sys.stderr)
            w.writerow(
                [
                    i,
                    stem,
                    f"{g:.6f}",
                    str(src) if src.is_file() else "",
                    "yes" if copied else "no",
                ]
            )

    err_path = out_case / "ddg_predict_errors.txt"
    if errors:
        err_path.write_text(
            "\n\n".join(f"{stem}\n{msg}" for stem, msg in errors), encoding="utf-8"
        )

    print(
        f"Wrote {rank_csv} ({len(favorable)} favorable ddG<0 / {len(rows)} total scored). "
        f"Designed PDBs under {out_case}/<stem>/"
    )
    if errors:
        print(f"See {err_path} for {len(errors)} predict failures.", file=sys.stderr)


if __name__ == "__main__":
    main()
