#!/usr/bin/env python3
"""Select top-N MSA sequences with the fewest alignment gaps (after a3m normalization)."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

LOGGER = logging.getLogger("select_a3m")


def normalize_a3m_sequence(raw: str) -> str:
    return re.sub(r"[a-z]", "", raw)


def read_a3m(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text().splitlines()
    records: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue
        header = lines[i][1:].strip()
        i += 1
        chunks: list[str] = []
        while i < len(lines) and not lines[i].startswith(">"):
            chunks.append(lines[i].strip())
            i += 1
        records.append((header, normalize_a3m_sequence("".join(chunks))))
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def parse_identity_score(header: str) -> float:
    """HH-suite a3m: >id score ... — use 2nd field as identity when numeric."""
    parts = header.split()
    if len(parts) >= 2:
        try:
            return float(parts[1])
        except ValueError:
            pass
    return 0.0


def select_low_gap(
    records: list[tuple[str, str]],
    n_select: int,
    *,
    include_query: bool = True,
) -> list[tuple[str, str]]:
    query_name, query_seq = records[0]
    L = len(query_seq)
    candidates: list[tuple[int, float, str, str]] = []

    pool = records[1:] if include_query else records
    for header, seq in pool:
        if len(seq) != L:
            continue
        gaps = seq.count("-")
        ident = parse_identity_score(header)
        candidates.append((gaps, -ident, header, seq))

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    picked = candidates[:n_select]

    out: list[tuple[str, str]] = []
    if include_query:
        out.append((query_name, query_seq))
    out.extend((h, s) for _, _, h, s in picked)
    return out


def write_a3m(records: list[tuple[str, str]], path: Path) -> None:
    lines: list[str] = []
    for name, seq in records:
        lines.append(f">{name}")
        lines.append(seq)
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Select low-gap sequences from A3M.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "-n",
        type=int,
        default=1000,
        help="Number of homologs to select (query is always kept)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    records = read_a3m(args.input)
    selected = select_low_gap(records, args.n)
    write_a3m(selected, args.output)

    gaps = [s.count("-") for _, s in selected]
    LOGGER.info(
        "Wrote %s: %d sequences (1 query + %d homologs), gaps min/med/max = %d/%d/%d",
        args.output,
        len(selected),
        len(selected) - 1,
        min(gaps),
        sorted(gaps)[len(gaps) // 2],
        max(gaps),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
