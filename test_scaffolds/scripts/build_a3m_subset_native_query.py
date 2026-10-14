#!/usr/bin/env python3
"""
从 MPNN 大库 a3m 按 score 取 top-N 条，query 固定为原生 {ID}.a3m 第一行序列，写出新 a3m + npz。

示例:
  python build_a3m_subset_native_query.py --ids 8BAQ 8U0X --top 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from split_mpnn_a3m_by_score import (  # noqa: E402
    read_a3m,
    write_a3m,
    convert_npz,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/data/wenhao/Protein-Hunter-server/test_scaffolds"))
    ap.add_argument("--repo", type=Path, default=Path("/data/wenhao/Protein-Hunter-server"))
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--top", type=int, default=500)
    ap.add_argument(
        "--msa-src",
        default="_mpnn3000_unique",
        help="MSA 序列来源后缀，如 _mpnn3000_unique / _mpnn1000_top",
    )
    ap.add_argument(
        "--query-suffix",
        default="",
        help="原生 query 来源；空表示 {ID}.a3m",
    )
    ap.add_argument(
        "--out-suffix",
        default=None,
        help="输出后缀，默认 _msa{N}_nativeQ",
    )
    ap.add_argument("--python", default="/data/wenhao/miniconda3/envs/proteinhunter/bin/python")
    args = ap.parse_args()

    out_suffix = args.out_suffix or f"_msa{args.top}_nativeQ"
    query_suffix = args.query_suffix

    for cid in args.ids:
        case = args.root / cid
        msa_src = case / f"{cid}{args.msa_src}.a3m"
        query_path = case / f"{cid}{query_suffix}.a3m"
        if not msa_src.exists():
            print(f"SKIP {cid}: missing {msa_src}", file=sys.stderr)
            continue
        if not query_path.exists():
            print(f"SKIP {cid}: missing native query {query_path}", file=sys.stderr)
            continue

        q_name, q_seq_native, records = read_a3m(msa_src)
        _, q_seq_ref, _ = read_a3m(query_path)
        # query 名与序列均用原生 a3m
        q_name_ref = read_a3m(query_path)[0]
        ranked = sorted(records, key=lambda x: x[0], reverse=True)
        if len(ranked) < args.top:
            print(f"WARN {cid}: only {len(ranked)} MSA seqs (< {args.top})", file=sys.stderr)

        stamp = f"{cid}{out_suffix}"
        out_a3m = case / f"{stamp}.a3m"
        out_npz = case / f"{stamp}.npz"
        n = write_a3m(out_a3m, q_name_ref, q_seq_ref, ranked, args.top)
        convert_npz(out_a3m, out_npz, stamp, args.repo, args.python)
        same = q_seq_native == q_seq_ref
        print(
            f"{cid}: native_query len={len(q_seq_ref)}  "
            f"msa_top{n} from {msa_src.name}  "
            f"query_same_as_msa_src={same}  -> {out_a3m.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
