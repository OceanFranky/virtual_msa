#!/usr/bin/env python3
"""从 MPNN a3m 表头解析 score，汇总 min/max/mean/median/p10/p90。"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

SCORE_RE = re.compile(r"score=([0-9.]+)")


def scores_from_a3m(path: Path) -> list[float]:
    scores: list[float] = []
    for line in path.read_text().splitlines():
        if not line.startswith(">"):
            continue
        if line.startswith(">T=") or "score=" in line:
            m = SCORE_RE.search(line)
            if m:
                scores.append(float(m.group(1)))
    return scores


def summarize(scores: list[float]) -> dict[str, float | int]:
    if not scores:
        return {
            "n": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }
    arr = np.asarray(scores, dtype=float)
    return {
        "n": len(arr),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/data/wenhao/Protein-Hunter-server"))
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--ids", nargs="+", default=None)
    ap.add_argument(
        "--temps",
        nargs="+",
        default=["0.01", "0.1", "0.2", "0.5", "0.7", "1.0", "1.5", "2.0"],
    )
    ap.add_argument("--msa-n", type=int, default=50)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-detail-csv", type=Path, default=None)
    args = ap.parse_args()

    root = args.root or args.repo / "test_scaffolds"
    ids = args.ids or [
        "8U0X",
        "8IQC",
        "8BAQ",
        "8Q8I",
        "8BAU",
        "8R2C",
        "8SV7",
        "8EO5",
        "8BUX",
    ]

    def t_tag(temp: str) -> str:
        mapping = {
            "2.0": "t2.0",
            "2": "t2.0",
            "1.5": "t1.5",
            "1.0": "t1.0",
            "1": "t1.0",
            "0.7": "t0.7",
            "0.5": "t0.5",
            "0.2": "t0.2",
            "0.1": "t0.1",
            "0.01": "t0.01",
        }
        return mapping.get(temp, f"t{temp}")

    rows = []
    detail_rows = []
    for temp in args.temps:
        tag = t_tag(temp)
        temp_group = tag
        all_scores: list[float] = []
        n_pdb = 0
        for pid in ids:
            a3m = root / pid / f"{pid}_mpnn{args.msa_n}_{tag}.a3m"
            if not a3m.is_file():
                print(f"WARN missing: {a3m}")
                continue
            scores = scores_from_a3m(a3m)
            n_pdb += 1
            all_scores.extend(scores)
            st = summarize(scores)
            detail_rows.append(
                {
                    "pdb_id": pid,
                    "temp_group": temp_group,
                    "temperature": temp,
                    "n_seq": st["n"],
                    **{k: st[k] for k in ("min", "max", "mean", "median", "p10", "p90")},
                }
            )
        st_all = summarize(all_scores)
        rows.append(
            {
                "temp_group": temp_group,
                "n_pdb": n_pdb,
                "n_seq_total": st_all["n"],
                "score_min": st_all["min"],
                "score_max": st_all["max"],
                "score_mean": st_all["mean"],
                "score_median": st_all["median"],
                "score_p10": st_all["p10"],
                "score_p90": st_all["p90"],
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "temp_group",
        "n_pdb",
        "n_seq_total",
        "score_min",
        "score_max",
        "score_mean",
        "score_median",
        "score_p10",
        "score_p90",
    ]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    if args.out_detail_csv:
        args.out_detail_csv.parent.mkdir(parents=True, exist_ok=True)
        dfields = [
            "pdb_id",
            "temp_group",
            "temperature",
            "n_seq",
            "min",
            "max",
            "mean",
            "median",
            "p10",
            "p90",
        ]
        with args.out_detail_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=dfields)
            w.writeheader()
            w.writerows(detail_rows)

    print(f"Wrote {args.out_csv} ({len(rows)} temps)")
    for r in rows:
        print(
            f"  {r['temp_group']:6s}  n={r['n_seq_total']:4d}  "
            f"min={r['score_min']:.3f}  max={r['score_max']:.3f}  "
            f"mean={r['score_mean']:.3f}  median={r['score_median']:.3f}  "
            f"p10={r['score_p10']:.3f}  p90={r['score_p90']:.3f}"
        )


if __name__ == "__main__":
    main()
