#!/usr/bin/env python3
"""
从 PDB 中提取指定链，写出 <原文件名去掉.pdb>_binder.pdb（默认提取链 E；与本仓库 WT *_clean.pdb 约定一致）。

示例:
  python3 extract_chain_to_binder.py /path/to/9MPJ_case_1_clean.pdb
  python3 extract_chain_to_binder.py foo.pdb -c B -o /tmp/foo_binder.pdb

仅依赖标准库；链 ID 使用 PDB 固定列第 22 列（与 BioPython 一致）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def pdb_chain_id(line: str) -> str | None:
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return None
    if len(line) <= 21:
        return None
    return line[21]


def collect_chains(path: Path) -> set[str]:
    chains: set[str] = set()
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            cid = pdb_chain_id(line)
            if cid is not None and cid.strip():
                chains.add(cid)
    return chains


def extract_chain(in_path: Path, chain_id: str, out_path: Path | None) -> Path:
    if out_path is None:
        out_path = in_path.parent / f"{in_path.stem}_binder.pdb"

    preamble: list[str] = []
    atoms: list[str] = []
    seen_coord = False

    with in_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                seen_coord = True
                if len(line) > 21 and line[21] == chain_id:
                    atoms.append(line)
            elif not seen_coord:
                preamble.append(line)

    if not atoms:
        avail = sorted(collect_chains(in_path))
        raise SystemExit(
            f"链 {chain_id!r} 无坐标行。文件中的链: {avail}\n"
            f"可用: python3 {Path(__file__).name} {in_path} -c <链ID>"
        )

    last = atoms[-1]
    try:
        ter_serial = int(last[6:11]) + 1
    except ValueError:
        ter_serial = 99999
    # 末行 17:27 为 resname + chain + resseq + icode（PDB 经典列）
    ter_line = f"TER   {ter_serial:5d}      {last[17:27]}\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for line in preamble:
            out.write(line)
        for line in atoms:
            out.write(line)
        out.write(ter_line)
        out.write("END\n")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="提取 PDB 中一条链为 *_binder.pdb")
    ap.add_argument("pdb", type=Path, help="输入 PDB")
    ap.add_argument(
        "-c",
        "--chain",
        default="E",
        help="链 ID（单字符，默认 E）",
    )
    ap.add_argument("-o", "--output", type=Path, default=None, help="输出路径（默认与输入同目录）")
    args = ap.parse_args()
    inp = args.pdb.resolve()
    if not inp.is_file():
        print(f"找不到文件: {inp}", file=sys.stderr)
        sys.exit(1)
    out = extract_chain(inp, args.chain, args.output.resolve() if args.output else None)
    print(out)


if __name__ == "__main__":
    main()
