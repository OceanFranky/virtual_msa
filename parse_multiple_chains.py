#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_chain_a_residues(pdb_path: Path) -> int:
    """从 PDB 里粗略数 chain A 的唯一 residue number，生成一个长度匹配的占位序列。"""
    seen: set[int] = set()
    for line in pdb_path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) < 26:
            continue
        chain_id = (line[21] or " ").strip() or "A"
        if chain_id != "A":
            continue
        try:
            resseq = int(line[22:26].strip())
        except Exception:
            continue
        seen.add(resseq)
    return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as out:
        pdb_paths = sorted(p for p in args.input_path.glob("*.pdb") if not p.name.endswith("_mask.pdb"))
        for pdb_path in pdb_paths:
            name = pdb_path.stem.upper()
            L = count_chain_a_residues(pdb_path)
            if L <= 0:
                # 保持和主流程对接：返回 seq 为空会被 cmd_prepare 过滤掉
                out.write(json.dumps({"name": name, "seq": ""}) + "\n")
                continue
            out.write(json.dumps({"name": name, "seq": "A" * L}) + "\n")


if __name__ == "__main__":
    main()

