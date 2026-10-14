#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", required=True, type=Path)
    parser.add_argument("--out_folder", required=True, type=Path)
    parser.add_argument("--num_seq_per_target", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--sampling_temp", default="0.1 0.2 0.3 0.4 0.5")
    parser.add_argument("--model_name", default="v_48_020")
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    seq_dir = args.out_folder / "seqs"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # 本 stub 只保证输出格式被上游 cmd_project 读取即可（至少 2 条记录）。
    entries: list[dict] = []
    with args.jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    for entry in entries:
        name = (entry.get("name") or "").upper()
        seq = entry.get("seq") or ""
        if not name or not seq:
            continue

        fasta_path = seq_dir / f"{name}.fa"
        # 第一条作为 query，第二条作为后续 projected_records 至少需要 1 条（records[1:]）。
        fasta_path.write_text(f">{name}\n{seq}\n>{name}_sample_1\n{seq}\n", encoding="utf-8")


if __name__ == "__main__":
    main()

