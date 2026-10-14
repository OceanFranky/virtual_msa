#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import gemmi


ROOT = Path(__file__).resolve().parents[2]
SCAFFOLD_ROOT = ROOT / "scaffolds"
OUT_ROOT = SCAFFOLD_ROOT / "protein-ligand"
WORK_ROOT = SCAFFOLD_ROOT / "_runs" / "protein-ligand_external"

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
}


@dataclass
class ExternalPLCase:
    case_id: str
    export_name: str
    protein_structure: Path
    protein_chain: str
    mask_ranges: tuple[tuple[int, int], ...]
    ligand_sdf: Path
    ligand_smiles: Path
    protein_full_fasta: Path | None = None


def parse_mask_ranges(value: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    if not value.strip():
        return tuple(ranges)
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            ranges.append((int(start_s), int(end_s)))
        else:
            pos = int(token)
            ranges.append((pos, pos))
    return tuple(ranges)


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


def read_single_fasta(path: Path) -> str:
    records = read_fasta_records(path)
    if len(records) != 1:
        raise RuntimeError(f"expected exactly one FASTA record in {path}")
    return records[0][1]


def structure_to_pdb(path: Path) -> str:
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    return st.make_pdb_string()


def parse_seqres_from_pdb_string(pdb_string: str) -> dict[str, list[str]]:
    seqres: dict[str, list[str]] = {}
    for line in pdb_string.splitlines():
        if line.startswith("SEQRES"):
            chain = line[11].strip()
            seqres.setdefault(chain, []).extend(line[19:].split())
    return seqres


def seq3_to_seq1(seq3: list[str]) -> str:
    return "".join(AA3_TO_1.get(x.upper(), "X") for x in seq3)


def iter_polymer_residues_from_pdb_string(pdb_string: str, chain_name: str) -> list[tuple[int, str]]:
    residues: list[tuple[int, str]] = []
    seen = set()
    for raw in pdb_string.splitlines():
        if not raw.startswith(("ATOM", "HETATM")):
            continue
        line = raw if len(raw) >= 80 else raw.ljust(80)
        chain = line[21].strip()
        if chain != chain_name:
            continue
        resname = line[17:20].strip().upper()
        if resname not in AA3_TO_1:
            continue
        resseq = int(line[22:26].strip())
        key = (resseq, line[26], resname)
        if key in seen:
            continue
        seen.add(key)
        residues.append((resseq, resname))
    return residues


def residue_seq1(residues: list[tuple[int, str]]) -> str:
    return "".join(AA3_TO_1.get(resname.upper(), "X") for _, resname in residues)


def full_sequence_from_pdb_string(pdb_string: str, chain_name: str) -> str:
    seqres = parse_seqres_from_pdb_string(pdb_string)
    if chain_name in seqres and seqres[chain_name]:
        return seq3_to_seq1(seqres[chain_name])
    return residue_seq1(iter_polymer_residues_from_pdb_string(pdb_string, chain_name))


def renumber_chain_to_a_from_pdb_string(
    pdb_string: str,
    chain_name: str,
    dst: Path,
) -> tuple[str, dict[int, int]]:
    out_lines: list[str] = []
    seq: list[str] = []
    current_key = None
    new_resseq = 0
    source_to_query: dict[int, int] = {}
    for raw in pdb_string.splitlines():
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
            source_to_query[source_resseq] = new_resseq
        line = f"ATOM  {line[6:21]}A{new_resseq:4d}{line[26:]}"
        out_lines.append(line.rstrip() + "\n")
    out_lines.append("END\n")
    dst.write_text("".join(out_lines))
    return "".join(seq), source_to_query


def write_selected_loops_tsv(path: Path, rows: list[dict[str, int | str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_name", "loop_start", "loop_end"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_clean_or_mask_from_pdb_string(
    pdb_string: str,
    chain_name: str,
    dst: Path,
    drop_resseq: set[int] | None = None,
) -> None:
    drop_resseq = drop_resseq or set()
    out: list[str] = []
    for raw in pdb_string.splitlines():
        if not raw.startswith(("ATOM", "HETATM")):
            continue
        line = raw if len(raw) >= 80 else raw.ljust(80)
        chain = line[21].strip()
        if chain != chain_name:
            continue
        resname = line[17:20].strip().upper()
        if resname not in AA3_TO_1:
            continue
        resseq = int(line[22:26].strip())
        if resseq in drop_resseq:
            continue
        out.append(("ATOM  " + line[6:]).rstrip() + "\n")
    out.append("END\n")
    dst.write_text("".join(out))


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


def load_cases(path: Path) -> list[ExternalPLCase]:
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    required = {"case_id", "protein_structure", "protein_chain", "mask_ranges", "ligand_sdf", "ligand_smiles"}
    if not rows:
        raise RuntimeError("external PL spec TSV is empty")
    if not required.issubset(rows[0].keys()):
        raise RuntimeError(f"external PL spec TSV must contain columns: {sorted(required)}")

    cases: list[ExternalPLCase] = []
    for row in rows:
        cases.append(
            ExternalPLCase(
                case_id=row["case_id"].strip().upper(),
                export_name=(row.get("export_name") or row["case_id"]).strip(),
                protein_structure=Path(row["protein_structure"]).resolve(),
                protein_chain=row["protein_chain"].strip(),
                mask_ranges=parse_mask_ranges(row["mask_ranges"]),
                ligand_sdf=Path(row["ligand_sdf"]).resolve(),
                ligand_smiles=Path(row["ligand_smiles"]).resolve(),
                protein_full_fasta=Path(row["protein_full_fasta"]).resolve() if row.get("protein_full_fasta") else None,
            )
        )
    return cases


def prepare_external_inputs(
    cases: list[ExternalPLCase],
    input_dir: Path,
    loops_tsv: Path,
    meta_jsonl: Path,
) -> list[dict[str, object]]:
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    loop_rows: list[dict[str, int | str]] = []
    meta_rows: list[dict[str, object]] = []
    for case in cases:
        pdb_string = structure_to_pdb(case.protein_structure)
        observed_seq, source_to_query = renumber_chain_to_a_from_pdb_string(
            pdb_string,
            case.protein_chain,
            input_dir / f"{case.case_id}.pdb",
        )
        (input_dir / f"{case.case_id}.fasta").write_text(f">{case.case_id}\n{observed_seq}\n")

        full_seq = (
            read_single_fasta(case.protein_full_fasta)
            if case.protein_full_fasta
            else full_sequence_from_pdb_string(pdb_string, case.protein_chain)
        )

        observed_positions: list[int] = []
        for start, end in case.mask_ranges:
            for resseq in range(start, end + 1):
                query_pos = source_to_query.get(resseq)
                if query_pos is not None:
                    observed_positions.append(query_pos)
        observed_positions = sorted(set(observed_positions))

        query_mask_ranges: list[tuple[int, int]] = []
        if observed_positions:
            start = end = observed_positions[0]
            for pos in observed_positions[1:]:
                if pos == end + 1:
                    end = pos
                else:
                    query_mask_ranges.append((start, end))
                    start = end = pos
            query_mask_ranges.append((start, end))

        for start, end in query_mask_ranges:
            loop_rows.append({"sample_name": case.case_id, "loop_start": start, "loop_end": end})

        meta_rows.append(
            {
                "case_id": case.case_id,
                "export_name": case.export_name,
                "protein_structure": str(case.protein_structure),
                "protein_chain": case.protein_chain,
                "observed_seq": observed_seq,
                "full_seq": full_seq,
                "query_mask_ranges": query_mask_ranges,
                "mask_ranges": [list(x) for x in case.mask_ranges],
                "ligand_sdf": str(case.ligand_sdf),
                "ligand_smiles": str(case.ligand_smiles),
            }
        )

    write_selected_loops_tsv(loops_tsv, loop_rows)
    with meta_jsonl.open("w") as handle:
        for row in meta_rows:
            handle.write(json.dumps(row) + "\n")
    return meta_rows


def export_external_results(meta_rows: list[dict[str, object]], exported_names_path: Path) -> None:
    exported_names_path.write_text("".join(f"{row['export_name']}\n" for row in meta_rows))
    for row in meta_rows:
        case_id = str(row["case_id"])
        export_name = str(row["export_name"])
        protein_structure = Path(str(row["protein_structure"]))
        protein_chain = str(row["protein_chain"])
        mask_ranges = [tuple(x) for x in row["mask_ranges"]]
        query_mask_ranges = tuple(tuple(x) for x in row["query_mask_ranges"])

        source_mask = {
            start + offset
            for start, end in mask_ranges
            for offset in range(end - start + 1)
        }
        pdb_string = structure_to_pdb(protein_structure)

        dst_dir = OUT_ROOT / export_name
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        write_clean_or_mask_from_pdb_string(pdb_string, protein_chain, dst_dir / f"{export_name}_clean.pdb")
        write_clean_or_mask_from_pdb_string(pdb_string, protein_chain, dst_dir / f"{export_name}_mask.pdb", source_mask)
        export_full_length_a3m(
            WORK_ROOT / "scaffolds" / case_id / f"{case_id}.a3m",
            dst_dir / f"{export_name}.a3m",
            export_name,
            str(row["observed_seq"]),
            str(row["full_seq"]),
            query_mask_ranges,
        )
        (dst_dir / f"{export_name}.fasta").write_text(f">{export_name}\n{row['full_seq']}\n")
        shutil.copyfile(str(row["ligand_sdf"]), dst_dir / "ligand.sdf")
        shutil.copyfile(str(row["ligand_smiles"]), dst_dir / "ligand_smiles.txt")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the full Protein-Ligand external-case delivery flow from a TSV spec.")
    ap.add_argument("--spec-tsv", type=Path, required=True, help="TSV with case_id, protein_structure, protein_chain, mask_ranges, ligand_sdf, ligand_smiles, and optional protein_full_fasta/export_name")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    args = ap.parse_args()

    cases = load_cases(args.spec_tsv.resolve())
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    input_dir = WORK_ROOT / "input"
    loops_tsv = WORK_ROOT / "selected_loops.tsv"
    meta_jsonl = WORK_ROOT / "export_meta.jsonl"
    meta_rows = prepare_external_inputs(cases, input_dir, loops_tsv, meta_jsonl)

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
    export_external_results(meta_rows, exported_names_path)
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

    print(OUT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
