#!/usr/bin/env python3
"""
在单进程内加载一次 binding-ddg-predictor 模型，对多对 wt/mut PDB 批量预测 ddG。

用法:
  python batch_predict_8ye4_ddg.py --pairs-tsv pairs.tsv --out results.tsv --device cpu

pairs.tsv 两列: wt_pdb\\tmut_pdb
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddg-root", type=Path, required=True, help="binding-ddg-predictor 仓库根目录")
    ap.add_argument("--pairs-tsv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--model", type=str, default="")
    args = ap.parse_args()

    root = args.ddg_root.resolve()
    sys.path.insert(0, str(root))
    import torch
    from models.predictor import DDGPredictor
    from utils.data import load_wt_mut_pdb_pair
    from utils.misc import recursive_to

    model_path = Path(args.model) if args.model else root / "data" / "model.pt"
    if not model_path.is_file():
        print(f"Missing model: {model_path}", file=sys.stderr)
        return 1

    ckpt = torch.load(str(model_path), map_location=args.device, weights_only=False)
    config = ckpt["config"]
    weight = ckpt["model"]
    model = DDGPredictor(config.model).to(args.device)
    model.load_state_dict(weight)
    model.eval()

    rows = []
    with args.pairs_tsv.open() as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if len(parts) < 2:
                continue
            wt_p, mut_p = parts[0].strip(), parts[1].strip()
            if not wt_p or not mut_p:
                continue
            stem = Path(mut_p).parent.name
            try:
                batch = load_wt_mut_pdb_pair(wt_p, mut_p)
                batch = recursive_to(batch, args.device)
                with torch.no_grad():
                    pred = model(batch["wt"], batch["mut"])
                ddg = float(pred.item())
                rows.append((stem, wt_p, mut_p, ddg, "ok"))
            except Exception as e:
                rows.append((stem, wt_p, mut_p, "", f"error:{e}"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["stem", "wt_pdb", "mut_pdb", "ddg", "status"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
