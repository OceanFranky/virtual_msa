#!/usr/bin/env python3
"""MPNN 短序列 a3m → 以原生 {ID}.a3m query 为列、按 PDB 缺失位补 gap 的 MSA。"""

from __future__ import annotations

import argparse
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from rebuild_a3m_gap_aligned import (  # noqa: E402
    load_query,
    project_to_full,
    ungap,
)
from split_mpnn_a3m_by_score import convert_npz, read_a3m, write_a3m


def infer_observed(q_full: str, obs_old: str, case_dir: Path, case_id: str) -> str:
    observed = ungap(obs_old)
    if observed and len(observed) <= len(q_full) and len(observed) >= len(q_full) * 0.5:
        return observed

    native = case_dir / f"{case_id}.a3m"
    if native.exists():
        lines = native.read_text().splitlines()
        if len(lines) > 3:
            obs2 = ungap(lines[3])
            if obs2 and len(obs2) <= len(q_full):
                return obs2

    sm = SequenceMatcher(None, q_full, ungap(obs_old), autojunk=False)
    pos = sm.find_longest_match(0, len(q_full), 0, len(ungap(obs_old)))
    if pos.size > 30:
        return q_full[pos.a : pos.a + pos.size]
    raise SystemExit(f"{case_id}: cannot infer observed (design) region")


def align_a3m(
    case_dir: Path,
    case_id: str,
    src_a3m: Path,
    out_a3m: Path,
    target: int,
    repo: Path,
    python: str,
    write_npz: bool,
) -> dict:
    native_a3m = case_dir / f"{case_id}.a3m"
    if not native_a3m.exists():
        raise FileNotFoundError(f"missing native a3m: {native_a3m}")

    q_name, q_full = load_query(case_dir, case_id)
    _, obs_old, records = read_a3m(src_a3m)
    observed = infer_observed(q_full, obs_old, case_dir, case_id)

    aligned: list[tuple[float, str, str]] = []
    seen: set[str] = set()
    for score, raw_h, des in records:
        des_u = ungap(des)
        if not des_u or des_u in seen:
            continue
        if len(des_u) != len(observed):
            if abs(len(des_u) - len(observed)) > 3:
                continue
            des_u = (des_u + observed)[: len(observed)]
        try:
            ali = project_to_full(des_u, observed, q_full)
        except ValueError:
            continue
        if len(ali) != len(q_full):
            continue
        seen.add(des_u)
        aligned.append((score, raw_h, ali))
        if len(aligned) >= target:
            break

    if len(aligned) < target:
        raise SystemExit(
            f"{case_id}: only {len(aligned)} gap-aligned unique (need {target})"
        )

    ranked = sorted(aligned, key=lambda x: x[0], reverse=True)[:target]
    write_a3m(out_a3m, q_name, q_full, ranked, target)

    out_npz = out_a3m.with_suffix(".npz")
    if write_npz:
        stamp = out_a3m.stem
        convert_npz(out_a3m, out_npz, stamp, repo, python)

    return {
        "case_id": case_id,
        "qlen": len(q_full),
        "obs_len": len(observed),
        "n": len(ranked),
        "out_a3m": str(out_a3m),
        "gaps_in_query": q_full.count("-"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", type=Path, required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--src-a3m", type=Path, required=True)
    ap.add_argument("--out-a3m", type=Path, required=True)
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--repo", type=Path, default=Path("/data/wenhao/Protein-Hunter-server"))
    ap.add_argument("--python", default="/data/wenhao/miniconda3/envs/proteinhunter/bin/python")
    ap.add_argument("--no-npz", action="store_true")
    args = ap.parse_args()

    r = align_a3m(
        args.case_dir.resolve(),
        args.case_id,
        args.src_a3m.resolve(),
        args.out_a3m.resolve(),
        args.target,
        args.repo.resolve(),
        args.python,
        write_npz=not args.no_npz,
    )
    print(
        f"OK {r['case_id']} n={r['n']} qlen={r['qlen']} obs={r['obs_len']} "
        f"query_gaps={r['gaps_in_query']} -> {r['out_a3m']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
