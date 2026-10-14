#!/usr/bin/env python3
"""
根据 summary_high_iptm.csv 中的 pdb_filename，把所有涉及到的 PDB 复制到
「run 目录名 + _iptm」文件夹（与 run 目录同级）。

示例（默认路径可按需改）:
  python3 collect_pdbs_from_summary_high_iptm.py \\
    /path/to/4LRH_run1/summary_high_iptm.csv

输出目录: /path/to/boltz_design/4LRH_run1_iptm/

查找顺序: <run_dir>/high_iptm_pdb/<name> → <run_dir>/<name> → rglob  basename。
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


def build_basename_index(run_dir: Path) -> dict[str, Path]:
    """basename -> first matching path under run_dir."""
    idx: dict[str, Path] = {}
    for p in run_dir.rglob("*.pdb"):
        name = p.name
        if name not in idx:
            idx[name] = p
    return idx


def resolve_pdb(run_dir: Path, name: str, index: dict[str, Path]) -> Path | None:
    if not name.strip():
        return None
    pri = run_dir / "high_iptm_pdb" / name
    if pri.is_file():
        return pri
    sec = run_dir / name
    if sec.is_file():
        return sec
    return index.get(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=Path(
            "/data/wenhao/Protein-Hunter-server/AAV/boltz_design/"
            "4LRH_run1/summary_high_iptm.csv"
        ),
        help="summary_high_iptm.csv 路径",
    )
    ap.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录（默认: boltz_design/<run名>_iptm）",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要复制的文件，不写入",
    )
    args = ap.parse_args()

    csv_path = args.csv_path.resolve()
    if not csv_path.is_file():
        print(f"找不到 CSV: {csv_path}", file=sys.stderr)
        sys.exit(1)

    run_dir = csv_path.parent
    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
    else:
        out_dir = run_dir.parent / f"{run_dir.name}_iptm"

    names: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "pdb_filename" not in reader.fieldnames:
            print(
                "CSV 需包含列 pdb_filename；当前列: "
                f"{reader.fieldnames}",
                file=sys.stderr,
            )
            sys.exit(1)
        for row in reader:
            fn = (row.get("pdb_filename") or "").strip()
            if fn:
                names.append(fn)

    unique = sorted(set(names))
    index = build_basename_index(run_dir)

    missing: list[str] = []
    copied = 0

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for name in unique:
        src = resolve_pdb(run_dir, name, index)
        if src is None or not src.is_file():
            missing.append(name)
            continue
        dst = out_dir / name
        if args.dry_run:
            print(f"would copy: {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
            copied += 1

    print(f"CSV: {csv_path}")
    print(f"输出目录: {out_dir}")
    print(f"唯一 pdb_filename 数: {len(unique)}")
    print(f"成功复制: {copied}")
    if missing:
        print(f"缺失 {len(missing)} 个:", file=sys.stderr)
        for m in missing[:20]:
            print(f"  {m}", file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... 另有 {len(missing) - 20} 个", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
