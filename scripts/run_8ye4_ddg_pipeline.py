#!/usr/bin/env python3
"""
8YE4 设计体 ddG 全流程：
  1) 用 prepare_tcr_8ye4_ddg_pairs 生成 wt/mut 两链 PDB
  2) binding-ddg-predictor 单进程批量推理
  3) 预测 ddG < 0 的条目排序，取前 50，复制原始设计 PDB 到输出目录

输出目录（默认）:
  .../TCR/8YE4_ddg_results/
    all_predictions.tsv
    negative_ddg_sorted.tsv   # 仅 ddG < 0，升序（越负越靠前）
    top50_manifest.tsv
    top50_original_pdbs/        # 原始 8YE4/*.pdb 副本
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
import sys
from pathlib import Path


def _load_prepare():
    here = Path(__file__).resolve().parent
    path = here / "prepare_tcr_8ye4_ddg_pairs.py"
    spec = importlib.util.spec_from_file_location("prepare_tcr_8ye4_ddg_pairs", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--design-dir",
        type=Path,
        default=Path("/data/wenhao/Protein-Hunter-server/example_myself/TCR/8YE4"),
    )
    ap.add_argument(
        "--clean",
        type=Path,
        default=Path(
            "/data/wenhao/Protein-Hunter-server/example_myself/TCR/8YE4_H_CDR3b/8YE4_H_CDR3b_clean.pdb"
        ),
    )
    ap.add_argument(
        "--ddg-repo",
        type=Path,
        default=Path("/data/wenhao/binding-ddg-predictor-main"),
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("/data/wenhao/Protein-Hunter-server/example_myself/TCR/8YE4_ddg_results"),
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cuda | cpu | auto（有 GPU 则用 cuda）",
    )
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    prep = _load_prepare()
    design_dir = args.design_dir.resolve()
    clean = args.clean.resolve()
    ddg_repo = args.ddg_repo.resolve()
    out_root = args.out_root.resolve()
    prep_root = out_root / "prepared_pairs"
    out_root.mkdir(parents=True, exist_ok=True)
    prep_root.mkdir(parents=True, exist_ok=True)

    pdbs = sorted(design_dir.glob("*.pdb"))
    if not pdbs:
        print(f"No PDB in {design_dir}", file=sys.stderr)
        return 1

    pairs_lines: list[tuple[str, str]] = []
    prep_errors: list[tuple[str, str]] = []
    for pdb in pdbs:
        stem = pdb.stem
        odir = prep_root / stem
        try:
            wt_p, mut_p = prep.process_one(clean, pdb, odir)
            pairs_lines.append((str(wt_p), str(mut_p)))
        except Exception as e:
            prep_errors.append((pdb.name, str(e)))
            print(f"[prep-skip] {pdb.name}: {e}", file=sys.stderr)

    pairs_tsv = out_root / "pairs_for_predict.tsv"
    with pairs_tsv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for a, b in pairs_lines:
            w.writerow([a, b])
    print(f"Prepared {len(pairs_lines)} / {len(pdbs)} pairs -> {prep_root}", file=sys.stderr)

    # batch predict (single model load)
    sys.path.insert(0, str(ddg_repo))
    import torch
    from models.predictor import DDGPredictor
    from utils.data import load_wt_mut_pdb_pair
    from utils.misc import recursive_to

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)

    model_path = ddg_repo / "data" / "model.pt"
    if not model_path.is_file():
        print(f"Missing {model_path}", file=sys.stderr)
        return 1

    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    model = DDGPredictor(ckpt["config"].model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_rows: list[list] = []
    for wt_p, mut_p in pairs_lines:
        stem = Path(mut_p).parent.name
        try:
            batch = load_wt_mut_pdb_pair(wt_p, mut_p)
            batch = recursive_to(batch, device)
            with torch.no_grad():
                pred = model(batch["wt"], batch["mut"])
            ddg = float(pred.item())
            all_rows.append([stem, wt_p, mut_p, f"{ddg:.4f}", "ok"])
        except Exception as e:
            all_rows.append([stem, wt_p, mut_p, "", f"error:{e}"])

    all_tsv = out_root / "all_predictions.tsv"
    with all_tsv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["stem", "wt_pdb", "mut_pdb", "ddg", "status"])
        w.writerows(all_rows)

    neg = [r for r in all_rows if r[4] == "ok" and r[3] != "" and float(r[3]) < 0.0]
    neg.sort(key=lambda r: float(r[3]))

    neg_tsv = out_root / "negative_ddg_sorted.tsv"
    with neg_tsv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["stem", "wt_pdb", "mut_pdb", "ddg", "status"])
        w.writerows(neg)

    top = neg[: args.top_k]
    top_dir = out_root / "top50_original_pdbs"
    top_dir.mkdir(parents=True, exist_ok=True)
    for r in top:
        stem = r[0]
        src = design_dir / f"{stem}.pdb"
        if src.is_file():
            shutil.copy2(src, top_dir / src.name)
        else:
            print(f"[warn] missing source {src}", file=sys.stderr)

    man = out_root / "top50_manifest.tsv"
    with man.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rank", "stem", "ddg", "original_pdb"])
        for i, r in enumerate(top, start=1):
            w.writerow([i, r[0], r[3], str(design_dir / (r[0] + ".pdb"))])

    err_log = out_root / "prep_errors.tsv"
    with err_log.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["file", "error"])
        w.writerows(prep_errors)

    print(
        f"Done. predictions={len(all_rows)}, negative={len(neg)}, "
        f"top{args.top_k}_copied={len(list(top_dir.glob('*.pdb')))}, out={out_root}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
