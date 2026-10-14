#!/usr/bin/env python3
"""
Prepare wild-type / designed PDB pairs for binding-ddg-predictor.

- Reads WT from <case_dir>/<case>_clean.pdb and role definitions from <case>.fasta
  (records named *binder* / *target* in headers).
- Maps chains in WT and designed PDBs to binder vs target by best match to FASTA.
- Reorders output to chain A = target, chain B = binder (same in both files).
- Trims **insertions in the designed structure** with a global alignment that heavily
  penalizes gaps in the WT sequence (Bio.Align.PairwiseAligner): every WT residue is
  kept; extra designed residues are dropped. If the design is missing WT residues,
  the script errors (so you can fix the structure).

Requires: biopython
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from Bio import Align
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.SeqUtils import seq1


def read_fasta_records(path: Path) -> Dict[str, str]:
    """Return {record_key: sequence} where key is the first token after '>'."""
    recs: Dict[str, str] = {}
    cur_key: str | None = None
    buf: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_key is not None:
                recs[cur_key] = "".join(buf).replace(" ", "")
            cur_key = line[1:].split()[0]
            buf = []
        else:
            buf.append(line)
    if cur_key is not None:
        recs[cur_key] = "".join(buf).replace(" ", "")
    return recs


def pick_binder_target_seq(recs: Dict[str, str]) -> Tuple[str, str]:
    binder_key = next((k for k in recs if re.search(r"binder", k, re.I)), None)
    target_key = next((k for k in recs if re.search(r"target", k, re.I)), None)
    if not binder_key or not target_key:
        raise ValueError(
            f"FASTA must contain one *binder* and one *target* record; got keys: {list(recs)}"
        )
    return recs[binder_key], recs[target_key]


def chain_aa_sequences(structure) -> Dict[str, str]:
    """One-letter sequences for standard residues with CA, in chain iteration order."""
    out: Dict[str, str] = {}
    model = next(structure.get_models())
    for ch in model:
        chars: List[str] = []
        for res in ch:
            if res.id[0].strip():
                continue
            if not res.has_id("CA"):
                continue
            try:
                chars.append(seq1(res.get_resname()))
            except Exception:
                continue
        out[ch.id] = "".join(chars)
    return out


def ordered_standard_residues(chain) -> List[Residue]:
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
    """
    Align WT (target in aligner) to designed (query). Heavy gap penalty on WT so
    extra residues in the design are treated as insertions and skipped; WT positions
    are preserved 1:1 when possible.
    """
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
            f"Designed sequence is missing residues relative to WT: mapped "
            f"{len(idx_w)} / {len(wt_seq)} positions. Check sequences or gap penalties."
        )
    return idx_w, idx_d


def _align_score(ref: str, query: str) -> float:
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.target_open_gap_score = -40
    aligner.target_extend_gap_score = -4
    aligner.query_open_gap_score = -2
    aligner.query_extend_gap_score = -0.2
    alns = aligner.align(ref, query)
    return float(alns[0].score) if alns else -1e9


def assign_roles_two_chain(
    seqs: Dict[str, str], binder_ref: str, target_ref: str
) -> Tuple[str, str]:
    """Return (chain_id_binder, chain_id_target) for a two-chain complex."""
    ids = sorted(seqs.keys())
    if len(ids) != 2:
        raise ValueError(f"Expected exactly 2 protein chains, got {ids}.")

    c0, c1 = ids[0], ids[1]
    # Prefer assignment where each chain best matches its FASTA role.
    s01 = _align_score(binder_ref, seqs[c0]) + _align_score(target_ref, seqs[c1])
    s10 = _align_score(binder_ref, seqs[c1]) + _align_score(target_ref, seqs[c0])
    if s01 >= s10:
        return c0, c1
    return c1, c0


def residue_copy_renumbered(res: Residue, new_index: int, new_chain: str) -> Residue:
    """Deep-enough copy with new chain and residue number (no insertion code)."""
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
    """Write chain A = target, chain B = binder."""
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


def process_one(
    case_dir: Path,
    designed_pdb: Path,
    out_root: Path,
) -> Tuple[Path, Path]:
    case_id = case_dir.name
    clean_pdb = case_dir / f"{case_id}_clean.pdb"
    fasta = case_dir / f"{case_id}.fasta"
    if not clean_pdb.is_file():
        raise FileNotFoundError(clean_pdb)
    if not fasta.is_file():
        raise FileNotFoundError(fasta)

    recs = read_fasta_records(fasta)
    binder_fa, target_fa = pick_binder_target_seq(recs)

    parser = PDBParser(QUIET=True)
    wt_s = parser.get_structure("wt", clean_pdb)
    mut_s = parser.get_structure("mut", designed_pdb)

    wt_seqs = chain_aa_sequences(wt_s)
    mut_seqs = chain_aa_sequences(mut_s)

    wt_b_ch, wt_t_ch = assign_roles_two_chain(wt_seqs, binder_fa, target_fa)
    mut_b_ch, mut_t_ch = assign_roles_two_chain(mut_seqs, binder_fa, target_fa)

    wt_model = next(wt_s.get_models())
    mut_model = next(mut_s.get_models())
    wt_b_chain = wt_model[wt_b_ch]
    wt_t_chain = wt_model[wt_t_ch]
    mut_b_chain = mut_model[mut_b_ch]
    mut_t_chain = mut_model[mut_t_ch]

    # Trim using actual WT PDB sequences as reference (per user request).
    wt_b_seq = wt_seqs[wt_b_ch]
    wt_t_seq = wt_seqs[wt_t_ch]
    mut_b_seq = mut_seqs[mut_b_ch]
    mut_t_seq = mut_seqs[mut_t_ch]

    br, bq = map_wt_to_designed_indices(wt_b_seq, mut_b_seq)
    tr, tq = map_wt_to_designed_indices(wt_t_seq, mut_t_seq)

    wt_b_ord = ordered_standard_residues(wt_b_chain)
    mut_b_ord = ordered_standard_residues(mut_b_chain)
    wt_t_ord = ordered_standard_residues(wt_t_chain)
    mut_t_ord = ordered_standard_residues(mut_t_chain)

    wt_b_res = [wt_b_ord[i] for i in br]
    mut_b_res = [mut_b_ord[j] for j in bq]
    wt_t_res = [wt_t_ord[i] for i in tr]
    mut_t_res = [mut_t_ord[j] for j in tq]

    if len(wt_b_res) != len(mut_b_res):
        raise AssertionError("Binder length mismatch after filter.")
    if len(wt_t_res) != len(mut_t_res):
        raise AssertionError("Target length mismatch after filter.")

    out_dir = out_root / designed_pdb.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    wt_out = out_dir / "wt_for_ddG.pdb"
    mut_out = out_dir / "mut_for_ddG.pdb"

    # A = target, B = binder
    write_two_chain_pdb(wt_out, wt_t_res, wt_b_res)
    write_two_chain_pdb(mut_out, mut_t_res, mut_b_res)

    n_a = len(wt_t_res)
    n_b = len(wt_b_res)
    print(
        f"[OK] {designed_pdb.name} -> {out_dir} "
        f"(target={n_a} res, binder={n_b} res; total {n_a + n_b})"
    )
    return wt_out, mut_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--wt-case-dir",
        type=Path,
        required=True,
        help="e.g. .../WT/7Z1A (must contain <case>_clean.pdb and <case>.fasta)",
    )
    ap.add_argument(
        "--iptm-dir",
        type=Path,
        required=True,
        help="Directory containing designed complex PDB files",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="e.g. .../iptm_select (each input gets a subfolder by PDB stem)",
    )
    ap.add_argument(
        "--glob",
        default="*.pdb",
        help="Match pattern inside iptm-dir (default: *.pdb)",
    )
    args = ap.parse_args()

    case_dir = args.wt_case_dir.resolve()
    iptm = args.iptm_dir.resolve()
    out_root = args.out_root.resolve()

    pdbs = sorted(iptm.glob(args.glob))
    if not pdbs:
        print(f"No files matching {args.glob!r} in {iptm}", file=sys.stderr)
        sys.exit(1)

    ok = 0
    skipped = 0
    for pdb in pdbs:
        try:
            process_one(case_dir, pdb, out_root)
            ok += 1
        except Exception as e:
            skipped += 1
            print(f"[SKIP] {pdb.name}: {e}", file=sys.stderr)

    print(f"Done. OK={ok}, skipped={skipped}, total={len(pdbs)}")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
