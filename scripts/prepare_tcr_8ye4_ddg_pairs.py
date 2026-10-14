#!/usr/bin/env python3
"""
为 TCR 8YE4_H_CDR3b 设计体准备 binding-ddg-predictor 所需的 WT / Mut 两链 PDB。

WT 参考：8YE4_H_CDR3b_clean.pdb（auth 链）
  - Target（输出链 A）= 链 G + 链 A + 链 E 按顺序拼接
  - Binder（输出链 B）= 链 H

设计 PDB（AlphaFold/Protein-Hunter 常见布局）：
  - Target = 链 B + 链 C + 链 D
  - Binder = 链 A

用与 prepare_binding_ddg_pairs.py 相同的 PairwiseAligner：以 WT 序列为锚，
去掉设计 target 中相对 WT 的多余插入；若设计缺失 WT 残基则报错跳过。

用法:
  python3 prepare_tcr_8ye4_ddg_pairs.py --clean path/to/8YE4_H_CDR3b_clean.pdb \\
      --designed path/to/design.pdb --out-dir path/to/out/design_stem/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from Bio import Align
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from Bio.SeqUtils import seq1


def ordered_standard_residues(chain: Chain) -> List[Residue]:
    residues: List[Residue] = []
    for res in chain:
        if res.id[0].strip():
            continue
        if not res.has_id("CA"):
            continue
        try:
            seq1(res.get_resname())
        except Exception:
            continue
        residues.append(res)
    return residues


def map_wt_to_designed_indices(wt_seq: str, des_seq: str) -> Tuple[List[int], List[int]]:
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.target_open_gap_score = -40
    aligner.target_extend_gap_score = -4
    aligner.query_open_gap_score = -2
    aligner.query_extend_gap_score = -0.2
    alns = aligner.align(wt_seq, des_seq)
    if not alns:
        raise RuntimeError("Alignment failed (empty).")
    aln = alns[0]
    idx_w: List[int] = []
    idx_d: List[int] = []
    for (w0, w1), (d0, d1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(w1 - w0):
            idx_w.append(w0 + k)
            idx_d.append(d0 + k)
    if len(idx_w) != len(wt_seq):
        raise ValueError(
            f"Designed sequence missing residues vs WT target: mapped "
            f"{len(idx_w)} / {len(wt_seq)}."
        )
    return idx_w, idx_d


def residue_copy_renumbered(res: Residue, new_index: int, new_chain: str) -> Residue:
    one = Residue((" ", new_index, " "), res.resname, res.segid)
    for atom in res:
        na = atom.copy()
        na.set_parent(None)
        one.add(na)
    one.set_parent(None)
    return one


def write_two_chain_pdb(
    path: Path,
    residues_target: Sequence[Residue],
    residues_binder: Sequence[Residue],
) -> None:
    struct = Structure("ddg")
    model = Model(0)
    ch_a = Chain("A")
    ch_b = Chain("B")
    for ri, res in enumerate(residues_target, start=1):
        ch_a.add(residue_copy_renumbered(res, ri, "A"))
    for ri, res in enumerate(residues_binder, start=1):
        ch_b.add(residue_copy_renumbered(res, ri, "B"))
    model.add(ch_a)
    model.add(ch_b)
    struct.add(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(struct)
    io.save(str(path))


def concat_target_residues(model: Model, chain_ids: Sequence[str]) -> List[Residue]:
    out: List[Residue] = []
    for cid in chain_ids:
        if cid not in model:
            raise KeyError(f"Chain {cid!r} not in model.")
        out.extend(ordered_standard_residues(model[cid]))
    return out


def process_one(clean_pdb: Path, designed_pdb: Path, out_dir: Path) -> Tuple[Path, Path]:
    parser = PDBParser(QUIET=True)
    wt_s = parser.get_structure("wt", str(clean_pdb))
    mut_s = parser.get_structure("mut", str(designed_pdb))
    wt_m = next(wt_s.get_models())
    mut_m = next(mut_s.get_models())

    # WT: target = G+A+E, binder = H
    wt_t_res = concat_target_residues(wt_m, ("G", "A", "E"))
    wt_b_res = ordered_standard_residues(wt_m["H"])

    # Design: target = B+C+D, binder = A
    mut_t_res = concat_target_residues(mut_m, ("B", "C", "D"))
    mut_b_res = ordered_standard_residues(mut_m["A"])

    wt_t_seq = "".join(seq1(r.get_resname()) for r in wt_t_res)
    wt_b_seq = "".join(seq1(r.get_resname()) for r in wt_b_res)
    mut_t_seq = "".join(seq1(r.get_resname()) for r in mut_t_res)
    mut_b_seq = "".join(seq1(r.get_resname()) for r in mut_b_res)

    tr, tq = map_wt_to_designed_indices(wt_t_seq, mut_t_seq)
    br, bq = map_wt_to_designed_indices(wt_b_seq, mut_b_seq)

    wt_t_f = [wt_t_res[i] for i in tr]
    mut_t_f = [mut_t_res[j] for j in tq]
    wt_b_f = [wt_b_res[i] for i in br]
    mut_b_f = [mut_b_res[j] for j in bq]

    if len(wt_t_f) != len(mut_t_f) or len(wt_b_f) != len(mut_b_f):
        raise AssertionError("Length mismatch after alignment.")

    out_dir.mkdir(parents=True, exist_ok=True)
    wt_out = out_dir / "wt_for_ddG.pdb"
    mut_out = out_dir / "mut_for_ddG.pdb"
    write_two_chain_pdb(wt_out, wt_t_f, wt_b_f)
    write_two_chain_pdb(mut_out, mut_t_f, mut_b_f)
    return wt_out, mut_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean", type=Path, required=True)
    ap.add_argument("--designed", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    try:
        wt, mut = process_one(args.clean.resolve(), args.designed.resolve(), args.out_dir.resolve())
        print(f"[OK] {args.designed.name} -> {wt} , {mut}")
        return 0
    except Exception as e:
        print(f"[FAIL] {args.designed.name}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
