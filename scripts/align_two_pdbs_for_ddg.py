#!/usr/bin/env python3
"""
Align two complex PDBs for binding-ddg-predictor: match chains by sequence, then
trim residues with a global alignment (same scorer as prepare_binding_ddg_pairs.py:
heavy gap penalty on the reference / target side so insertions in the other
structure are discouraged).

By default, only **aligned residue pairs** (columns where both sequences have an
amino acid) are kept; extra residues on either side are dropped. Use --strict-ref
to require the second structure to cover every reference residue (Rosetta-style).

Input: reference PDB (wild-type) first, second structure second.
Output: two PDBs with chains renamed A, B, ... in consistent order and matching
residue counts per chain.

Requires: biopython
"""

from __future__ import annotations

import argparse
import itertools
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


def ordered_standard_residues(chain) -> List[Residue]:
    """Standard residues with full backbone N, CA, C (same filter as binding-ddg-predictor)."""
    residues: List[Residue] = []
    for res in chain:
        if res.id[0].strip():
            continue
        if not (res.has_id("N") and res.has_id("CA") and res.has_id("C")):
            continue
        try:
            seq1(res.get_resname())
        except Exception:
            continue
        residues.append(res)
    return residues


def chain_aa_sequences_from_ordered(ordered_by_chain: Dict[str, List[Residue]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for cid, residues in ordered_by_chain.items():
        chars: List[str] = []
        for res in residues:
            try:
                chars.append(seq1(res.get_resname()))
            except Exception:
                continue
        out[cid] = "".join(chars)
    return out


def chain_ordered_maps(structure) -> Tuple[Dict[str, List[Residue]], Dict[str, str]]:
    model = next(structure.get_models())
    ordered: Dict[str, List[Residue]] = {}
    for ch in model:
        ordered[ch.id] = ordered_standard_residues(ch)
    seqs = chain_aa_sequences_from_ordered(ordered)
    return ordered, seqs


def map_ref_to_other_indices(
    ref_seq: str,
    other_seq: str,
    *,
    strict_ref: bool,
) -> Tuple[List[int], List[int]]:
    """
    Global alignment: ref = aligner target, other = query.
    Returns index lists into ref_seq / other_seq for each **matched pair**
    (ungapped blocks in the optimal alignment).

    If strict_ref: require len(pairs) == len(ref_seq) (other must cover every ref
    residue). Otherwise: keep the common aligned core only (both may lose ends /
    internal positions that align to gaps).
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.target_open_gap_score = -40
    aligner.target_extend_gap_score = -4
    aligner.query_open_gap_score = -2
    aligner.query_extend_gap_score = -0.2
    alns = aligner.align(ref_seq, other_seq)
    if not alns:
        raise RuntimeError("Alignment failed (empty).")
    aln = alns[0]
    idx_r: List[int] = []
    idx_o: List[int] = []
    for (r0, r1), (o0, o1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(r1 - r0):
            idx_r.append(r0 + k)
            idx_o.append(o0 + k)
    if not idx_r:
        raise ValueError("No aligned residue pairs for one chain (empty alignment).")
    if strict_ref and len(idx_r) != len(ref_seq):
        raise ValueError(
            f"Second structure is missing residues vs reference (--strict-ref): "
            f"mapped {len(idx_r)} / {len(ref_seq)} positions for one chain."
        )
    return idx_r, idx_o


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


def best_chain_permutation(
    ref_seqs: Dict[str, str], other_seqs: Dict[str, str]
) -> Dict[str, str]:
    """
    Map each reference chain id -> other chain id by maximizing sum of pairwise
    alignment scores (exhaustive permutations; fine for typical 2–6 chains).
    """
    ids_ref = sorted(ref_seqs.keys())
    ids_other = list(other_seqs.keys())
    if len(ids_ref) != len(ids_other):
        raise ValueError(
            f"Chain count mismatch: reference has {sorted(ref_seqs.keys())}, "
            f"other has {sorted(other_seqs.keys())}."
        )
    n = len(ids_ref)
    if n == 0:
        raise ValueError("No protein chains with N/CA/C found.")
    if n > 8:
        raise ValueError(f"Too many chains ({n}); raise limit or use a dedicated tool.")

    best_perm: Tuple[int, ...] | None = None
    best_score = -1e100
    for perm in itertools.permutations(range(n)):
        s = sum(
            _align_score(ref_seqs[ids_ref[i]], other_seqs[ids_other[perm[i]]])
            for i in range(n)
        )
        if s > best_score:
            best_score = s
            best_perm = perm
    assert best_perm is not None
    return {ids_ref[i]: ids_other[best_perm[i]] for i in range(n)}


def residue_copy_renumbered(res: Residue, new_index: int) -> Residue:
    one = Residue((" ", new_index, " "), res.resname, res.segid)
    for atom in res:
        na = atom.copy()
        na.set_parent(None)
        one.add(na)
    one.set_parent(None)
    return one


def write_renumbered_pdb(
    path: Path,
    chains_residues: Sequence[Sequence[Residue]],
    chain_ids: Sequence[str],
) -> None:
    if len(chains_residues) != len(chain_ids):
        raise ValueError("chains_residues and chain_ids length mismatch.")
    struct = Structure("aln")
    model = Model(0)
    for cid, residues in zip(chain_ids, chains_residues):
        ch = Chain(cid)
        for ri, res in enumerate(residues, start=1):
            ch.add(residue_copy_renumbered(res, ri))
        model.add(ch)
    struct.add(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(struct)
    io.save(str(path))


def _output_chain_labels(n: int) -> List[str]:
    labels = [chr(ord("A") + i) for i in range(min(n, 26))]
    if n > 26:
        raise ValueError("More than 26 chains not supported for output naming.")
    return labels


def run_align(
    ref_path: Path,
    other_path: Path,
    out_ref: Path,
    out_other: Path,
    *,
    strict_ref: bool,
) -> None:
    parser = PDBParser(QUIET=True)
    ref_s = parser.get_structure("ref", ref_path)
    oth_s = parser.get_structure("oth", other_path)

    ref_ord, ref_seqs = chain_ordered_maps(ref_s)
    oth_ord, oth_seqs = chain_ordered_maps(oth_s)

    mapping = best_chain_permutation(ref_seqs, oth_seqs)
    ids_ref_sorted = sorted(ref_seqs.keys())
    out_labels = _output_chain_labels(len(ids_ref_sorted))

    ref_chains_out: List[List[Residue]] = []
    oth_chains_out: List[List[Residue]] = []

    for rid in ids_ref_sorted:
        oid = mapping[rid]
        rseq = ref_seqs[rid]
        oseq = oth_seqs[oid]
        ir, io_ = map_ref_to_other_indices(rseq, oseq, strict_ref=strict_ref)
        ref_res_list = ref_ord[rid]
        oth_res_list = oth_ord[oid]

        ref_pick = [ref_res_list[i] for i in ir]
        oth_pick = [oth_res_list[j] for j in io_]
        if len(ref_pick) != len(oth_pick):
            raise AssertionError(f"Length mismatch after align for chains {rid}->{oid}.")
        if not strict_ref:
            dr, do = len(rseq) - len(ir), len(oseq) - len(io_)
            if dr or do:
                print(
                    f"  trim chain ref {rid} vs other {oid}: "
                    f"kept {len(ir)} pairs (dropped {dr} ref / {do} other positions)",
                    file=sys.stderr,
                )
        ref_chains_out.append(ref_pick)
        oth_chains_out.append(oth_pick)

    write_renumbered_pdb(out_ref, ref_chains_out, out_labels)
    write_renumbered_pdb(out_other, oth_chains_out, out_labels)

    total = sum(len(c) for c in ref_chains_out)
    print(
        f"[OK] chains {ids_ref_sorted} -> {out_labels} (via map {mapping}); "
        f"{total} residues per file"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "ref_pdb",
        type=Path,
        help="Reference / wild-type complex PDB",
    )
    ap.add_argument(
        "other_pdb",
        type=Path,
        help="Second structure (mutant / variant) PDB",
    )
    ap.add_argument(
        "--out-ref",
        type=Path,
        default=None,
        help="Output path for aligned reference (default: <ref_stem>_ddg_aligned.pdb)",
    )
    ap.add_argument(
        "--out-other",
        type=Path,
        default=None,
        help="Output path for aligned second structure (default: <other_stem>_ddg_aligned.pdb)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="If set, write defaults into this directory",
    )
    ap.add_argument(
        "--strict-ref",
        action="store_true",
        help=(
            "Require the second PDB to cover every reference residue per chain "
            "(same as prepare_binding_ddg_pairs.py); otherwise error if lengths differ."
        ),
    )
    args = ap.parse_args()

    ref_path = args.ref_pdb.resolve()
    other_path = args.other_pdb.resolve()
    if not ref_path.is_file():
        print(f"Missing {ref_path}", file=sys.stderr)
        sys.exit(1)
    if not other_path.is_file():
        print(f"Missing {other_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir.resolve() if args.out_dir else None
    if args.out_ref:
        out_ref = args.out_ref.resolve()
    elif out_dir:
        out_ref = out_dir / f"{ref_path.stem}_ddg_aligned.pdb"
    else:
        out_ref = ref_path.parent / f"{ref_path.stem}_ddg_aligned.pdb"

    if args.out_other:
        out_other = args.out_other.resolve()
    elif out_dir:
        out_other = out_dir / f"{other_path.stem}_ddg_aligned.pdb"
    else:
        out_other = other_path.parent / f"{other_path.stem}_ddg_aligned.pdb"

    try:
        run_align(
            ref_path,
            other_path,
            out_ref,
            out_other,
            strict_ref=args.strict_ref,
        )
    except Exception as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  wrote {out_ref}")
    print(f"  wrote {out_other}")


if __name__ == "__main__":
    main()
