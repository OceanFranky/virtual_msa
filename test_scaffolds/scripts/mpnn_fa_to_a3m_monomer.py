#!/usr/bin/env python3
"""LigandMPNN seqs/*.fa → monomer a3m (full chain, dedupe, fixed unique count)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def read_fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq_parts: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_parts)))
            header = line[1:]
            seq_parts = []
        else:
            seq_parts.append(line)
    if header is not None:
        records.append((header, "".join(seq_parts)))
    return records


def extract_protein_seq(seq: str) -> str:
    if ":" in seq:
        return seq.split(":")[-1]
    return seq


def parse_mpnn_header(header: str, sample_idx: int) -> str:
    temp_m = re.search(r"T=([0-9.]+)", header)
    temp = temp_m.group(1) if temp_m else "0.1"
    conf_m = re.search(r"overall_confidence=([0-9.]+)", header)
    rec_m = re.search(r"seq_rec=([0-9.]+)", header)
    conf = float(conf_m.group(1)) if conf_m else 0.0
    rec = float(rec_m.group(1)) if rec_m else 0.0
    score = conf if conf > 0 else 1.0
    return (
        f"T={temp}, sample={sample_idx}, score={score:.4f}, "
        f"global_score={score:.4f}, seq_recovery={rec:.4f}"
    )


def read_query_seq(fasta_path: Path, query_name: str) -> str:
    for header, seq in read_fasta_records(fasta_path):
        name = header.split(",")[0].strip()
        if name == query_name or header.startswith(query_name):
            return extract_protein_seq(seq).upper()
    raise SystemExit(f"query {query_name} not found in {fasta_path}")


def native_seq_from_fa(fa_path: Path) -> str:
    records = read_fasta_records(fa_path.resolve())
    if not records:
        raise SystemExit(f"empty fa: {fa_path}")
    return extract_protein_seq(records[0][1]).upper()


def collect_unique_from_fas(
    fa_paths: list[Path],
    target: int,
) -> tuple[list[tuple[str, str]], int]:
    seen: set[str] = set()
    designed: list[tuple[str, str]] = []
    raw_designed = 0
    for fa in fa_paths:
        records = read_fasta_records(fa.resolve())
        for header, seq in records[1:]:
            raw_designed += 1
            protein = extract_protein_seq(seq).upper()
            if not protein:
                continue
            if protein in seen:
                continue
            seen.add(protein)
            designed.append((header, protein))
            if len(designed) >= target:
                return designed[:target], raw_designed
    return designed, raw_designed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa", type=Path, action="append", required=True, help="one or more MPNN fa")
    ap.add_argument("--fasta", type=Path, default=None, help="native query fasta (fallback)")
    ap.add_argument(
        "--query-from-fa",
        type=Path,
        default=None,
        help="use chain-0 sequence from this MPNN fa as a3m query (matches design length)",
    )
    ap.add_argument("--out-a3m", type=Path, required=True)
    ap.add_argument("--query-name", required=True)
    ap.add_argument("--target-seqs", type=int, default=3000)
    args = ap.parse_args()

    fa_paths = [p.resolve() for p in args.fa]
    if args.query_from_fa:
        query = native_seq_from_fa(args.query_from_fa.resolve())
    elif args.fasta:
        query = read_query_seq(args.fasta.resolve(), args.query_name)
    else:
        query = native_seq_from_fa(fa_paths[0])

    designed, raw_n = collect_unique_from_fas(fa_paths, args.target_seqs)
    if len(designed) < args.target_seqs:
        raise SystemExit(
            f"only {len(designed)} unique sequences (raw designed rows scanned={raw_n}), "
            f"need {args.target_seqs}; run more MPNN batches"
        )

    lines = [f">{args.query_name}", query]
    for i, (header, protein) in enumerate(designed, start=1):
        lines.append(f">{parse_mpnn_header(header, i)}")
        lines.append(protein)

    args.out_a3m.parent.mkdir(parents=True, exist_ok=True)
    args.out_a3m.write_text("\n".join(lines) + "\n")
    n_hdr = len(designed) + 1
    print(
        f"wrote {args.out_a3m}  records={n_hdr}  unique_designed={len(designed)}  "
        f"raw_scanned={raw_n}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
