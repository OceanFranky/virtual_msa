#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PP_ROOT = ROOT / "protein-protein"
SCAFFOLD_ROOT = ROOT / "scaffolds"
OUT_ROOT = SCAFFOLD_ROOT / "protein-protein"
WORK_ROOT = SCAFFOLD_ROOT / "_runs" / "protein-protein_builtin"

PROTEIN_SEQ_DESIGN_REPO = Path("/mnt/sdc/yumo/proteinSEQ_DESIGN/Protein Sequence Design")
PARSE_SCRIPT = PROTEIN_SEQ_DESIGN_REPO / "helper_scripts" / "parse_multiple_chains.py"
SEQ_DESIGN_SCRIPT = PROTEIN_SEQ_DESIGN_REPO / "protein_seq_design_run.py"
SEQ_DESIGN_ENV_PY = Path("/home/yumo/miniconda3/envs/seq_design/bin/python")
BOLTZ_ENV_PY = Path("/home/yumo/miniconda3/envs/proteinhunter/bin/python")
CONVERTER = Path(__file__).resolve().parent / "convert_a3m_to_npz_local.py"
TOOL = Path(__file__).resolve().parent / "protein_sequence_design_tool.py"

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "KCX": "K", "MHO": "M", "LLP": "K",
    "DLE": "L", "DAR": "R", "DAS": "S", "DSG": "G", "DSN": "N",
    "DGL": "G", "DIL": "I", "DAL": "A", "DVA": "V", "DPN": "P",
    "DPR": "P", "DLY": "Y",
}


@dataclass
class PPCase:
    case_id: str
    export_name: str
    folder: str
    complex_pdb: str | None
    antigen_pdb: str | None
    binder_pdb: str | None
    binder_chain: str | None
    antigen_chains: tuple[str, ...]
    query_mask_ranges: tuple[tuple[int, int], ...]
    blocked_reason: str | None = None


def get_cases() -> list[PPCase]:
    return [
        PPCase("PP-1", "8VSJ", "PP-1_TRACeR-II", "8VSJ_complex.pdb", "antigen.pdb", "binder_reference.pdb", "C", ("A", "B", "P"), ((12, 16),)),
        PPCase("PP-2", "8FRE", "PP-2_RBL_buttressed_loop", None, None, None, None, (), (), "No public peptide-target structure is available, so CASE_ID_antigen.pdb cannot be generated authoritatively."),
        PPCase("PP-3", "6M0J", "PP-3_tFold_antibody", None, None, None, None, (), (), "No public designed antibody-antigen complex structure is available, so CASE_ID_clean.pdb and CASE_ID_mask.pdb cannot be generated authoritatively."),
        PPCase("PP-4", "9HAC", "PP-4_BindCraft", None, None, None, None, (), (), "No authoritative exact binder-side mask interval has been fixed for the built-in control case."),
        PPCase("PP-5", "9H71", "PP-5_beta_pairing_RFdiffusion", "9H71_complex.pdb", "antigen.pdb", "binder_reference.pdb", "C", ("A",), ((21, 28),)),
        PPCase("PP-6", "8T5E", "PP-6_helical_peptide_binder", "8T5E_complex.pdb", "antigen.pdb", "binder_reference.pdb", "A", ("B",), ((65, 78),)),
        PPCase("PP-7", "8GQP", "PP-7_heterochiral_D_binder", None, None, None, None, (), (), "Current local Protein Sequence Design preprocessing does not reliably support this D-amino-acid binder scaffold."),
    ]


def parse_seqres(path: Path) -> dict[str, list[str]]:
    seqres: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("SEQRES"):
            chain = line[11].strip()
            seqres.setdefault(chain, []).extend(line[19:].split())
    return seqres


def seq3_to_seq1(seq3: list[str]) -> str:
    return "".join(AA3_TO_1.get(x.upper(), "X") for x in seq3)


def read_fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq_lines: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_lines)))
            header = line[1:]
            seq_lines = []
        else:
            seq_lines.append(line)
    if header is not None:
        records.append((header, "".join(seq_lines)))
    return records


def iter_polymer_residues(path: Path, keep_chains: tuple[str, ...]) -> list[tuple[str, int, str]]:
    residues = []
    seen = set()
    for raw in path.read_text().splitlines():
        if not raw.startswith(("ATOM", "HETATM")):
            continue
        line = raw if len(raw) >= 80 else raw.ljust(80)
        chain = line[21].strip()
        if chain not in keep_chains:
            continue
        resname = line[17:20].strip().upper()
        if resname not in AA3_TO_1:
            continue
        resseq = int(line[22:26].strip())
        key = (chain, resseq, line[26], resname)
        if key in seen:
            continue
        seen.add(key)
        residues.append((chain, resseq, resname))
    return residues


def residue_seq1(residues: list[tuple[str, int, str]]) -> str:
    return "".join(AA3_TO_1.get(resname.upper(), "X") for _, _, resname in residues)


def full_sequence_from_seqres_or_atoms(path: Path, chain_name: str) -> str:
    seqres = parse_seqres(path)
    if chain_name in seqres and seqres[chain_name]:
        return seq3_to_seq1(seqres[chain_name])
    return residue_seq1(iter_polymer_residues(path, (chain_name,)))


def full_sequence_for_chains(path: Path, chains: tuple[str, ...]) -> str:
    seqres = parse_seqres(path)
    parts: list[str] = []
    for chain_name in chains:
        if chain_name in seqres and seqres[chain_name]:
            parts.append(seq3_to_seq1(seqres[chain_name]))
            continue
        parts.append(residue_seq1(iter_polymer_residues(path, (chain_name,))))
    return "".join(parts)


def map_resseq_to_seqpos(full_seq: str, residues: list[tuple[str, int, str]]) -> dict[tuple[str, int], int]:
    obs_seq = residue_seq1(residues)
    matcher = SequenceMatcher(a=full_seq, b=obs_seq, autojunk=False)
    mapping: dict[tuple[str, int], int] = {}
    obs_index_to_key = [(chain, resseq) for chain, resseq, _ in residues]
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            continue
        overlap = min(i2 - i1, j2 - j1)
        for off in range(overlap):
            mapping[obs_index_to_key[j1 + off]] = i1 + off + 1
    return mapping


def write_filtered_pdb(src: Path, dst: Path, keep_chains: tuple[str, ...], drop_residues: set[tuple[str, int]] | None = None) -> None:
    drop_residues = drop_residues or set()
    out = []
    for raw in src.read_text().splitlines():
        if not raw.startswith(("ATOM", "HETATM", "TER", "END")):
            continue
        line = raw if len(raw) >= 80 else raw.ljust(80)
        if line.startswith(("ATOM", "HETATM", "TER")):
            chain = line[21].strip()
            if chain not in keep_chains:
                continue
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip().upper()
                if resname not in AA3_TO_1:
                    continue
                resseq = int(line[22:26].strip())
                if (chain, resseq) in drop_residues:
                    continue
                line = "ATOM  " + line[6:]
        if not line.startswith("END"):
            out.append(line.rstrip() + "\n")
    out.append("END\n")
    dst.write_text("".join(out))


def renumber_binder_to_a(src: Path, chain_name: str, dst: Path) -> tuple[str, dict[int, int]]:
    out_lines = []
    seq = []
    current_key = None
    new_resseq = 0
    source_resseq_to_query: dict[int, int] = {}
    for raw in src.read_text().splitlines():
        if not raw.startswith(("ATOM", "HETATM")):
            continue
        line = raw if len(raw) >= 80 else raw.ljust(80)
        chain = line[21].strip()
        if chain != chain_name:
            continue
        resname = line[17:20].strip().upper()
        if resname not in AA3_TO_1:
            continue
        source_resseq = int(line[22:26].strip())
        key = (source_resseq, line[26], resname)
        if key != current_key:
            current_key = key
            new_resseq += 1
            seq.append(AA3_TO_1[resname])
            source_resseq_to_query[source_resseq] = new_resseq
        line = f"ATOM  {line[6:21]}A{new_resseq:4d}{line[26:]}"
        out_lines.append(line.rstrip() + "\n")
    out_lines.append("END\n")
    dst.write_text("".join(out_lines))
    return "".join(seq), source_resseq_to_query


def write_single_fasta(path: Path, name: str, seq: str) -> None:
    path.write_text(f">{name}\n{seq}\n")


def copy_a3m_with_query_name(src: Path, dst: Path, query_name: str) -> None:
    lines = src.read_text().splitlines()
    for idx, line in enumerate(lines):
        if line.startswith(">"):
            lines[idx] = f">{query_name}"
            break
    dst.write_text("\n".join(lines) + "\n")


def project_sequence_to_full_length(designed_sequence: str, observed_sequence: str, full_length_sequence: str) -> str:
    if len(designed_sequence) != len(observed_sequence):
        raise RuntimeError(
            f"designed sequence length {len(designed_sequence)} != observed sequence length {len(observed_sequence)}"
        )
    chars = ["-"] * len(full_length_sequence)
    matcher = SequenceMatcher(a=full_length_sequence, b=observed_sequence, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            continue
        overlap = min(i2 - i1, j2 - j1)
        for offset in range(overlap):
            chars[i1 + offset] = designed_sequence[j1 + offset]
    return "".join(chars)


def project_positions_to_full_length(
    observed_positions: set[int],
    observed_sequence: str,
    full_length_sequence: str,
) -> set[int]:
    full_positions: set[int] = set()
    matcher = SequenceMatcher(a=full_length_sequence, b=observed_sequence, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            continue
        overlap = min(i2 - i1, j2 - j1)
        for offset in range(overlap):
            observed_pos = j1 + offset + 1
            if observed_pos in observed_positions:
                full_positions.add(i1 + offset + 1)
    return full_positions


def apply_mask_ranges_to_query(
    full_length_sequence: str,
    observed_sequence: str,
    query_mask_ranges: tuple[tuple[int, int], ...],
) -> str:
    observed_positions: set[int] = set()
    for start, end in query_mask_ranges:
        observed_positions.update(range(start, end + 1))
    full_mask_positions = project_positions_to_full_length(observed_positions, observed_sequence, full_length_sequence)
    chars = list(full_length_sequence)
    for pos in full_mask_positions:
        chars[pos - 1] = "X"
    return "".join(chars)


def export_full_length_a3m(
    src: Path,
    dst: Path,
    query_name: str,
    observed_sequence: str,
    full_length_sequence: str,
    query_mask_ranges: tuple[tuple[int, int], ...],
) -> None:
    records = read_fasta_records(src)
    if not records:
        raise RuntimeError(f"no A3M records found in {src}")
    with dst.open("w") as handle:
        query_sequence = apply_mask_ranges_to_query(full_length_sequence, observed_sequence, query_mask_ranges)
        handle.write(f">{query_name}\n{query_sequence}\n")
        for header, seq in records[1:]:
            projected = project_sequence_to_full_length(seq, observed_sequence, full_length_sequence)
            handle.write(f">{header}\n{projected}\n")


def write_selected_loops_tsv(path: Path, cases: list[tuple[str, tuple[tuple[int, int], ...]]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_name", "loop_start", "loop_end"], delimiter="\t")
        w.writeheader()
        for case_id, ranges in cases:
            for start, end in ranges:
                w.writerow({"sample_name": case_id, "loop_start": start, "loop_end": end})


def combine_fasta(dst: Path, case_id: str, binder_seq: str, antigen_seq: str) -> None:
    dst.write_text(f">{case_id}_binder\n{binder_seq}\n>{case_id}_target\n{antigen_seq}\n")


def export_final(
    case: PPCase,
    binder_observed_seq: str,
    binder_full_seq: str,
    antigen_seq: str,
    source_mask_residues: set[tuple[str, int]],
) -> None:
    src_dir = PP_ROOT / case.folder
    src_scaffold = WORK_ROOT / "scaffolds" / case.case_id
    dst_dir = OUT_ROOT / case.export_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    write_filtered_pdb(src_dir / case.complex_pdb, dst_dir / f"{case.export_name}_clean.pdb", (case.binder_chain, *case.antigen_chains))
    write_filtered_pdb(src_dir / case.antigen_pdb, dst_dir / f"{case.export_name}_antigen.pdb", case.antigen_chains)
    write_filtered_pdb(src_dir / case.binder_pdb, dst_dir / f"{case.export_name}_mask.pdb", (case.binder_chain,), source_mask_residues)
    export_full_length_a3m(
        src_scaffold / f"{case.case_id}.a3m",
        dst_dir / f"{case.export_name}.a3m",
        case.export_name,
        binder_observed_seq,
        binder_full_seq,
        case.query_mask_ranges,
    )
    combine_fasta(dst_dir / f"{case.export_name}.fasta", case.export_name, binder_full_seq, antigen_seq)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run local Protein Sequence Design delivery flow for built-in Protein-Protein cases.")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    cases = get_cases()
    final_root = OUT_ROOT
    final_root.mkdir(parents=True, exist_ok=True)
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    input_dir = WORK_ROOT / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    loops_for_tool: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    prepared: list[tuple[PPCase, str, str, str, set[tuple[str, int]]]] = []

    for case in cases:
        dst_dir = final_root / case.export_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        if case.blocked_reason:
            (dst_dir / "BLOCKED.txt").write_text(case.blocked_reason + "\n")
            print(f"{case.folder}\tblocked:{case.blocked_reason}")
            continue
        src_dir = PP_ROOT / case.folder
        binder_observed_seq, source_to_query = renumber_binder_to_a(src_dir / case.binder_pdb, case.binder_chain, input_dir / f"{case.case_id}.pdb")
        binder_full_seq = full_sequence_for_chains(src_dir / case.binder_pdb, (case.binder_chain,))
        write_single_fasta(input_dir / f"{case.case_id}.fasta", case.case_id, binder_observed_seq)

        antigen_seq = full_sequence_for_chains(src_dir / case.antigen_pdb, case.antigen_chains)

        source_mask = set()
        for qstart, qend in case.query_mask_ranges:
            loops_for_tool.append((case.case_id, ((qstart, qend),)))
            for source_resseq, query_pos in source_to_query.items():
                if qstart <= query_pos <= qend:
                    source_mask.add((case.binder_chain, source_resseq))
        prepared.append((case, binder_observed_seq, binder_full_seq, antigen_seq, source_mask))

    loops_tsv = WORK_ROOT / "selected_loops.tsv"
    write_selected_loops_tsv(loops_tsv, loops_for_tool)

    subprocess.run(
        [
            str(SEQ_DESIGN_ENV_PY),
            str(TOOL),
            "build-delivery",
            "--input-dir", str(input_dir),
            "--work-dir", str(WORK_ROOT),
            "--parse-script", str(PARSE_SCRIPT),
            "--protein-seq-design-script", str(SEQ_DESIGN_SCRIPT),
            "--protein-seq-design-python", str(SEQ_DESIGN_ENV_PY),
            "--npz-converter", str(CONVERTER),
            "--npz-python", str(BOLTZ_ENV_PY),
            "--selected-loops-tsv", str(loops_tsv),
        ],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu},
    )

    exported_names_path = WORK_ROOT / "exported_names.txt"
    exported_names_path.write_text("".join(f"{case.export_name}\n" for case, *_ in prepared))

    for case, binder_observed_seq, binder_full_seq, antigen_seq, source_mask in prepared:
        export_final(case, binder_observed_seq, binder_full_seq, antigen_seq, source_mask)
        print(f"{case.folder}\tok")

    subprocess.run(
        [
            str(BOLTZ_ENV_PY),
            str(CONVERTER),
            "--scaffold-root",
            str(OUT_ROOT),
            "--name-list",
            str(exported_names_path),
        ],
        check=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
