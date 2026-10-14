#!/usr/bin/env python3
"""将 a3m 首条 query 序列替换为等长全 X，另存 a3m + npz（不覆盖源文件）。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# 复用 split 脚本的 npz 转换
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from split_mpnn_a3m_by_score import read_a3m, write_a3m, convert_npz  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/data/wenhao/Protein-Hunter-server/test_scaffolds"))
    ap.add_argument("--repo", type=Path, default=Path("/data/wenhao/Protein-Hunter-server"))
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument(
        "--src-suffix",
        default="_mpnn250_top",
        help="源 a3m 后缀；空字符串表示 {ID}.a3m（如 8ADD.a3m）",
    )
    ap.add_argument("--src-root", type=Path, default=None, help="源目录，默认 --root")
    ap.add_argument(
        "--out-suffix",
        default=None,
        help="输出后缀，默认 {src-suffix}_queryX",
    )
    ap.add_argument("--python", default="/data/wenhao/miniconda3/envs/proteinhunter/bin/python")
    args = ap.parse_args()

    out_suffix = args.out_suffix or f"{args.src_suffix}_queryX"
    src_root = args.src_root or args.root
    for cid in args.ids:
        case_dir = args.root / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        src = src_root / cid / f"{cid}{args.src_suffix}.a3m"
        if not src.exists():
            print(f"SKIP {cid}: missing {src}", file=sys.stderr)
            continue
        query_name, _old_q, records = read_a3m(src)
        query_x = "X" * len(_old_q)
        stamp = f"{cid}{out_suffix}"
        out_a3m = case_dir / f"{stamp}.a3m"
        out_npz = case_dir / f"{stamp}.npz"
        n = write_a3m(out_a3m, query_name, query_x, records, len(records))
        convert_npz(out_a3m, out_npz, stamp, args.repo, args.python)
        print(
            f"{cid}: query {len(_old_q)}aa -> all-X  "
            f"designed={n}  -> {out_a3m.name}  {out_npz.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
