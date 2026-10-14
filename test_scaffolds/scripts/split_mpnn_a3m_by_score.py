#!/usr/bin/env python3
"""
从 mpnn3000_unique.a3m 按 score 降序切分 top-N 子集并转 npz。

score = LigandMPNN overall_confidence，越高越好。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCORE_RE = re.compile(r"score=([0-9.]+)")
TOP_NS = (250, 500, 1000, 1500, 2000)


def read_a3m(path: Path) -> tuple[str, str, list[tuple[float, str, str]]]:
    """返回 (query_name, query_seq, [(score, raw_header, seq), ...])"""
    lines = path.read_text().splitlines()
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"invalid a3m: {path}")
    query_name = lines[0][1:].strip()
    query_seq = lines[1].strip() if len(lines) > 1 else ""
    records: list[tuple[float, str, str]] = []
    i = 2
    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue
        header = lines[i][1:]
        i += 1
        seq_parts: list[str] = []
        while i < len(lines) and not lines[i].startswith(">"):
            seq_parts.append(lines[i].strip())
            i += 1
        seq = "".join(seq_parts)
        m = SCORE_RE.search(header)
        score = float(m.group(1)) if m else 0.0
        records.append((score, header, seq))
    return query_name, query_seq, records


def format_header(raw_header: str, sample_idx: int, score: float) -> str:
    temp_m = re.search(r"T=([0-9.]+)", raw_header)
    temp = temp_m.group(1) if temp_m else "0.1"
    rec_m = re.search(r"seq_rec=([0-9.]+)", raw_header)
    rec = float(rec_m.group(1)) if rec_m else 0.0
    return (
        f"T={temp}, sample={sample_idx}, score={score:.4f}, "
        f"global_score={score:.4f}, seq_recovery={rec:.4f}"
    )


def write_a3m(
    path: Path,
    query_name: str,
    query_seq: str,
    ranked: list[tuple[float, str, str]],
    n_take: int,
) -> int:
    subset = ranked[:n_take]
    out_lines = [f">{query_name}", query_seq]
    for idx, (score, raw_h, seq) in enumerate(subset, start=1):
        out_lines.append(f">{format_header(raw_h, idx, score)}")
        out_lines.append(seq)
    path.write_text("\n".join(out_lines) + "\n")
    return len(subset)


def convert_npz(a3m_path: Path, npz_path: Path, stamp: str, repo: Path, python: str) -> None:
    converter = repo / "protein_sequence_design_tool" / "convert_a3m_to_npz_local.py"
    tmp = Path(f"/tmp/npz_conv_{stamp}")
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    case_dir = tmp / stamp
    case_dir.mkdir(parents=True)
    import shutil
    shutil.copy2(a3m_path, case_dir / f"{stamp}.a3m")
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(repo / "boltz_ph" / "src")}
    subprocess.run(
        [python, str(converter), "--scaffold-root", str(tmp)],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    shutil.copy2(case_dir / f"{stamp}.npz", npz_path)
    shutil.rmtree(tmp)


def process_one(
    case_dir: Path,
    case_id: str,
    src_tag: str,
    top_ns: tuple[int, ...],
    repo: Path,
    python: str,
    write_csv: bool,
) -> dict:
    src_a3m = case_dir / f"{case_id}{src_tag}.a3m"
    if not src_a3m.exists():
        raise FileNotFoundError(src_a3m)

    query_name, query_seq, records = read_a3m(src_a3m)
    ranked = sorted(records, key=lambda x: (-x[0], x[2]))
    n = len(ranked)
    if n < max(top_ns):
        raise ValueError(f"{case_id}: only {n} designed seqs, need {max(top_ns)}")

    if write_csv:
        rows = []
        for rank, (score, raw_h, seq) in enumerate(ranked, start=1):
            rec_m = re.search(r"seq_rec=([0-9.]+)", raw_h)
            rows.append({
                "rank": rank,
                "score": score,
                "seq_recovery": float(rec_m.group(1)) if rec_m else None,
                "seq_len": len(seq),
                "sequence": seq,
            })
        pd.DataFrame(rows).to_csv(case_dir / f"{case_id}{src_tag}_scores.csv", index=False)

    result = {"case_id": case_id, "n_total": n}
    for k in top_ns:
        tag = f"_mpnn{k}_top"
        a3m_out = case_dir / f"{case_id}{tag}.a3m"
        npz_out = case_dir / f"{case_id}{tag}.npz"
        stamp = f"{case_id}{tag}"
        n_written = write_a3m(a3m_out, query_name, query_seq, ranked, k)
        convert_npz(a3m_out, npz_out, stamp, repo, python)
        scores = [r[0] for r in ranked[:k]]
        result[f"top{k}_n"] = n_written
        result[f"top{k}_score_max"] = scores[0]
        result[f"top{k}_score_min"] = scores[-1]
        print(f"  {case_id}{tag}: n={n_written}  score [{scores[-1]:.4f}, {scores[0]:.4f}]")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/data/wenhao/Protein-Hunter-server/test_scaffolds"))
    ap.add_argument("--repo", type=Path, default=Path("/data/wenhao/Protein-Hunter-server"))
    ap.add_argument("--src-tag", default="_mpnn3000_unique")
    ap.add_argument("--ids", nargs="*", default=[
        "8U0X", "8IQC", "8ADD", "8BAQ", "8Q8I", "8BAU", "8R2C", "8SV7", "8EO5", "8BUX",
    ])
    ap.add_argument("--top", type=int, nargs="*", default=list(TOP_NS))
    ap.add_argument("--python", default="/data/wenhao/miniconda3/envs/proteinhunter/bin/python")
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    top_ns = tuple(sorted(set(args.top)))
    summary_rows = []
    for cid in args.ids:
        case_dir = args.root / cid
        print(f"==> {cid}")
        row = process_one(
            case_dir, cid, args.src_tag, top_ns, args.repo, args.python, not args.no_csv
        )
        summary_rows.append(row)

    if len(args.ids) == 1:
        summary_path = args.root / args.ids[0] / f"{args.ids[0]}_mpnn_subset_summary.csv"
    else:
        summary_path = args.root / "mpnn_subset_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nDONE summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
