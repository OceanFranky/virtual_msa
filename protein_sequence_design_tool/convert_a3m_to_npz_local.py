#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from boltz.data.parse.a3m import parse_a3m


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert scaffold A3M files to Boltz NPZ files locally.")
    ap.add_argument("--scaffold-root", type=Path, required=True)
    ap.add_argument("--name-list", type=Path, default=None)
    return ap.parse_args()


def load_selected_names(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def main() -> int:
    args = parse_args()
    selected = load_selected_names(args.name_list.resolve() if args.name_list else None)
    root = args.scaffold_root.resolve()
    ok = 0
    failed = 0
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = case_dir.name
        if selected is not None and name not in selected:
            continue
        a3m = case_dir / f"{name}.a3m"
        npz = case_dir / f"{name}.npz"
        if not a3m.exists():
            print(f"failed {name}: missing {a3m}")
            failed += 1
            continue
        try:
            msa = parse_a3m(a3m, taxonomy=None, max_seqs=None)
            msa.dump(npz)
        except Exception as exc:
            print(f"failed {name}: {exc}")
            failed += 1
            continue
        ok += 1
    print(f"npz_count {ok}")
    print(f"failed_count {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
