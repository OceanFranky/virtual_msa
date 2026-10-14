#!/usr/bin/env python3
"""合并多个 MPNN a3m（仅 designed 行）为一个 raw a3m。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from split_mpnn_a3m_by_score import read_a3m, write_a3m  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-a3m", type=Path, action="append", required=True)
    ap.add_argument("--out-a3m", type=Path, required=True)
    args = ap.parse_args()

    if not args.src_a3m:
        raise SystemExit("need at least one --src-a3m")

    q_name, q_seq, _ = read_a3m(args.src_a3m[0].resolve())
    all_records: list[tuple[float, str, str]] = []

    for src in args.src_a3m:
        _, _, recs = read_a3m(src.resolve())
        all_records.extend(recs)

    args.out_a3m.parent.mkdir(parents=True, exist_ok=True)
    write_a3m(args.out_a3m, q_name, q_seq, all_records, len(all_records))
    print(
        f"wrote {args.out_a3m}  query={q_name}  designed={len(all_records)}  "
        f"sources={len(args.src_a3m)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
