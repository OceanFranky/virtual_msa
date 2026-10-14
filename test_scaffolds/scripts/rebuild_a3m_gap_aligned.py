#!/usr/bin/env python3
"""
将 MPNN MSA 重建为：query = 原生 {ID}.a3m 全长序列；MSA 行按 PDB 缺失区 gap 对齐（-）。

规则：
  - query 优先 {ID}.a3m 第一行（无 gap）；无 a3m 时用 {ID}.fasta
  - observed = mpnn3000_unique 原 query（去 gap，= 设计区）
  - 每条设计序列 project 到全长列
  - 覆盖 mpnn3000_unique + top 250/500/1000/1500/2000 + npz
"""

from __future__ import annotations

import argparse
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from split_mpnn_a3m_by_score import (  # noqa: E402
    SCORE_RE,
    convert_npz,
    format_header,
    read_a3m,
    write_a3m,
)

TOP_NS = (250, 500, 1000, 1500, 2000)


def ungap(seq: str) -> str:
    return seq.replace("-", "").replace(".", "").upper()


def read_fasta_seq(path: Path) -> str:
    parts: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            continue
        parts.append(line.strip())
    return ungap("".join(parts))


def load_query(case_dir: Path, case_id: str) -> tuple[str, str]:
    native_a3m = case_dir / f"{case_id}.a3m"
    fasta = case_dir / f"{case_id}.fasta"
    if native_a3m.exists():
        name, seq, _ = read_a3m(native_a3m)
        return name, ungap(seq)
    if fasta.exists():
        return case_id, read_fasta_seq(fasta)
    raise FileNotFoundError(f"{case_id}: no {case_id}.a3m or .fasta")


def project_to_full(designed: str, observed: str, full: str) -> str:
    if len(designed) != len(observed):
        raise ValueError(
            f"len(designed)={len(designed)} != len(observed)={len(observed)}"
        )
    chars = ["-"] * len(full)
    for tag, i1, i2, j1, j2 in SequenceMatcher(
        a=full, b=observed, autojunk=False
    ).get_opcodes():
        if tag == "delete":
            continue
        overlap = min(i2 - i1, j2 - j1)
        for off in range(overlap):
            chars[i1 + off] = designed[j1 + off]
    return "".join(chars)


def rebuild_case(
    case_dir: Path,
    case_id: str,
    repo: Path,
    python: str,
    dry_run: bool,
) -> dict:
    src_mpnn = case_dir / f"{case_id}_mpnn3000_unique.a3m"
    if not src_mpnn.exists():
        return {"id": case_id, "status": "skip_no_mpnn3000"}

    q_name, q_full = load_query(case_dir, case_id)
    _, obs_old, records = read_a3m(src_mpnn)
    observed = ungap(obs_old)

    # 已是 gap 对齐版本：query=全长，MSA 行含 '-'
    already = (
        len(observed) >= len(q_full) * 0.95
        and len(records) > 0
        and "-" in records[0][2]
        and len(records[0][2]) == len(q_full)
    )

    aligned_records: list[tuple[float, str, str]] = []
    if already:
        for score, raw_h, seq in records:
            if len(seq) == len(q_full):
                aligned_records.append((score, raw_h, seq))
    else:
        if not observed or len(observed) > len(q_full) * 0.95:
            native = case_dir / f"{case_id}.a3m"
            if native.exists():
                nlines = native.read_text().splitlines()
                if len(nlines) > 3:
                    observed = ungap(nlines[3])
            if not observed or len(observed) >= len(q_full):
                sm = SequenceMatcher(None, q_full, obs_old)
                pos = sm.find_longest_match(0, len(q_full), 0, len(obs_old))
                if pos.size > 50:
                    observed = q_full[pos.a : pos.a + pos.size]
        if not observed:
            return {"id": case_id, "status": "skip_empty_observed"}

        for score, raw_h, des in records:
            des_u = ungap(des)
            if len(des_u) != len(observed):
                if abs(len(des_u) - len(observed)) > 3:
                    continue
                des_u = (des_u + observed)[: len(observed)]
            try:
                ali = project_to_full(des_u, observed, q_full)
            except ValueError:
                continue
            aligned_records.append((score, raw_h, ali))

    if len(aligned_records) < 100:
        return {"id": case_id, "status": f"skip_too_few_{len(aligned_records)}"}

    ranked = sorted(aligned_records, key=lambda x: x[0], reverse=True)

    if dry_run:
        return {"id": case_id, "status": "ok_dry", "n": len(ranked), "qlen": len(q_full)}

    # mpnn3000_unique
    tag3k = "_mpnn3000_unique"
    out3k = case_dir / f"{case_id}{tag3k}.a3m"
    write_a3m(out3k, q_name, q_full, ranked, len(ranked))
    convert_npz(out3k, case_dir / f"{case_id}{tag3k}.npz", f"{case_id}{tag3k}", repo, python)

    for n in TOP_NS:
        if n > len(ranked):
            continue
        tag = f"_mpnn{n}_top"
        out_a3m = case_dir / f"{case_id}{tag}.a3m"
        write_a3m(out_a3m, q_name, q_full, ranked, n)
        convert_npz(out_a3m, case_dir / f"{case_id}{tag}.npz", f"{case_id}{tag}", repo, python)

    return {
        "id": case_id,
        "status": "ok",
        "n": len(ranked),
        "qlen": len(q_full),
        "obs_len": len(observed),
    }


def iter_cases(root: Path):
    if not root.is_dir():
        return
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        cid = d.name
        if not (d / f"{cid}_mpnn3000_unique.a3m").exists():
            continue
        yield d, cid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "/data/wenhao/Protein-Hunter-server/11",
        "/data/wenhao/Protein-Hunter-server/test_scaffolds",
    ])
    ap.add_argument("--repo", type=Path, default=Path("/data/wenhao/Protein-Hunter-server"))
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--python", default="/data/wenhao/miniconda3/envs/proteinhunter/bin/python")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args()

    log_path = args.log or (args.repo / "test_scaffolds/scripts/rebuild_a3m_gap_aligned.log")
    rows = []
    for root_s in args.roots:
        root = Path(root_s)
        print(f"\n=== root {root} ===", flush=True)
        for case_dir, cid in iter_cases(root):
            if args.ids and cid not in args.ids:
                continue
            try:
                r = rebuild_case(case_dir, cid, args.repo, args.python, args.dry_run)
                r["root"] = str(root)
                rows.append(r)
                print(f"  {cid}: {r['status']}", flush=True)
            except Exception as e:
                rows.append({"root": str(root), "id": cid, "status": f"ERR {e}"})
                print(f"  {cid}: ERR {e}", flush=True)

    import pandas as pd
    pd.DataFrame(rows).to_csv(log_path, index=False)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"\nDONE ok={ok} total={len(rows)} log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
