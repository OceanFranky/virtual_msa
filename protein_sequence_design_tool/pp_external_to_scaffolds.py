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
OUT_ROOT = SCAFFOLD_ROOT / "protein-protein"
WORK_ROOT = SCAFFOLD_ROOT / "_runs" / "protein-protein_external"

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
class ExternalPPCase:
    case_id: str
    export_name: str
    complex_cif: Path
    binder_chains: tuple[str, ...]
    antigen_chains: tuple[str, ...]
    mask_ranges: tuple[tuple[str, int, int], ...]
    binder_full_fasta: Path | None = None
    antigen_full_fasta: Path | None = None


def parse_chain_list(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())


def parse_mask_spec(value: str, default_chain: str | None) -> tuple[tuple[str, int, int], ...]:
    residues: list[tuple[str, int, int]] = []
    if not value.strip():
        return tuple(residues)
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            chain, rng = token.split(":", 1)
            chain = chain.strip()
        else:
            if default_chain is None:
                raise ValueError(f"mask token '{token}' missing chain and no default binder chain available")
            chain = default_chain
            rng = token
        if "-" in rng:
            start_s, end_s = rng.split("-", 1)
            residues.append((chain, int(start_s), int(end_s)))
        else:
            pos = int(rng)
            residues.append((chain, pos, pos))
    return tuple(residues)


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


def iter_polymer_residues_from_pdb_string(pdb_string: str, keep_chains: tuple[str, ...]) -> list[tuple[str, int, str]]:
    residues: list[tuple[str, int, str]] = []
    seen = set()
    for raw in pdb_string.splitlines():
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


def full_sequence_for_chains_from_pdb_string(pdb_string: str, chains: tuple[str, ...]) -> str:
    seqres = parse_seqres_from_pdb_string(pdb_string)
    parts: list[str] = []
    for chain_name in chains:
        if chain_name in seqres and seqres[chain_name]:
            parts.append(seq3_to_seq1(seqres[chain_name]))
            continue
        parts.append(residue_seq1(iter_polymer_residues_from_pdb_string(pdb_string, (chain_name,))))
    return "".join(parts)


def renumber_selected_chains_to_a(
    pdb_string: str,
    keep_chains: tuple[str, ...],
    dst: Path,
) -> tuple[str, dict[tuple[str, int], int]]:
    out_lines: list[str] = []
    seq: list[str] = []
    current_key = None
    new_resseq = 0
    source_to_query: dict[tuple[str, int], int] = {}
    for raw in pdb_string.splitlines():
        if not raw.startswith(("ATOM", "HETATM")):
            continue
        line = raw if len(raw) >= 80 else raw.ljust(80)
        chain = line[21].strip()
        if chain not in keep_chains:
            continue
        resname = line[17:20].strip().upper()
        if resname not in AA3_TO_1:
            continue
        source_resseq = int(line[22:26].strip())
        key = (chain, source_resseq, line[26], resname)
        if key != current_key:
            current_key = key
            new_resseq += 1
            seq.append(AA3_TO_1[resname])
            source_to_query[(chain, source_resseq)] = new_resseq
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


def write_filtered_pdb_from_string(
    pdb_string: str,
    dst: Path,
    keep_chains: tuple[str, ...],
    drop_residues: set[tuple[str, int]] | None = None,
) -> None:
    drop_residues = drop_residues or set()
    out: list[str] = []
    for line in pdb_string.splitlines():
        if not line.startswith(("ATOM", "HETATM", "TER", "END")):
            continue
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
        if not line.startswith("END"):
            out.append(line.rstrip() + "\n")
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


def load_cases(path: Path) -> list[ExternalPPCase]:
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    required = {"case_id", "complex_cif", "binder_chains", "antigen_chains", "mask_ranges"}
    if not rows:
        raise RuntimeError("external PP spec TSV is empty")
    if not required.issubset(rows[0].keys()):
        raise RuntimeError(f"external PP spec TSV must contain columns: {sorted(required)}")

    cases: list[ExternalPPCase] = []
    for row in rows:
        case_id = row["case_id"].strip().upper()
        binder_chains = parse_chain_list(row["binder_chains"])
        antigen_chains = parse_chain_list(row["antigen_chains"])
        default_chain = binder_chains[0] if len(binder_chains) == 1 else None
        cases.append(
            ExternalPPCase(
                case_id=case_id,
                export_name=(row.get("export_name") or case_id).strip(),
                complex_cif=Path(row["complex_cif"]).resolve(),
                binder_chains=binder_chains,
                antigen_chains=antigen_chains,
                mask_ranges=parse_mask_spec(row["mask_ranges"], default_chain),
                binder_full_fasta=Path(row["binder_full_fasta"]).resolve() if row.get("binder_full_fasta") else None,
                antigen_full_fasta=Path(row["antigen_full_fasta"]).resolve() if row.get("antigen_full_fasta") else None,
            )
        )
    return cases


def prepare_external_inputs(
    cases: list[ExternalPPCase],
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
        pdb_string = structure_to_pdb(case.complex_cif)
        binder_observed_seq, source_to_query = renumber_selected_chains_to_a(
            pdb_string,
            case.binder_chains,
            input_dir / f"{case.case_id}.pdb",
        )
        (input_dir / f"{case.case_id}.fasta").write_text(f">{case.case_id}\n{binder_observed_seq}\n")

        binder_full_seq = (
            read_single_fasta(case.binder_full_fasta)
            if case.binder_full_fasta
            else full_sequence_for_chains_from_pdb_string(pdb_string, case.binder_chains)
        )
        antigen_full_seq = (
            read_single_fasta(case.antigen_full_fasta)
            if case.antigen_full_fasta
            else full_sequence_for_chains_from_pdb_string(pdb_string, case.antigen_chains)
        )

        observed_positions: list[int] = []
        for chain, start, end in case.mask_ranges:
            for resseq in range(start, end + 1):
                query_pos = source_to_query.get((chain, resseq))
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
                "cif_path": str(case.complex_cif),
                "binder_chains": ",".join(case.binder_chains),
                "antigen_chains": ",".join(case.antigen_chains),
                "binder_observed_seq": binder_observed_seq,
                "binder_full_seq": binder_full_seq,
                "antigen_full_seq": antigen_full_seq,
                "query_mask_ranges": query_mask_ranges,
                "mask_ranges": [list(x) for x in case.mask_ranges],
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
        binder_chains = tuple(x for x in str(row["binder_chains"]).split(",") if x)
        antigen_chains = tuple(x for x in str(row["antigen_chains"]).split(",") if x)
        mask_ranges = [tuple(x) for x in row["mask_ranges"]]
        query_mask_ranges = tuple(tuple(x) for x in row["query_mask_ranges"])

        pdb_string = structure_to_pdb(Path(str(row["cif_path"])))
        mask_residues = {
            (chain, start + offset)
            for chain, start, end in mask_ranges
            for offset in range(end - start + 1)
        }

        dst_dir = OUT_ROOT / export_name
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        write_filtered_pdb_from_string(pdb_string, dst_dir / f"{export_name}_clean.pdb", binder_chains + antigen_chains)
        write_filtered_pdb_from_string(pdb_string, dst_dir / f"{export_name}_antigen.pdb", antigen_chains)
        write_filtered_pdb_from_string(pdb_string, dst_dir / f"{export_name}_mask.pdb", binder_chains, mask_residues)
        (dst_dir / f"{export_name}.fasta").write_text(
            f">{export_name}_binder\n{row['binder_full_seq']}\n>{export_name}_target\n{row['antigen_full_seq']}\n"
        )
        export_full_length_a3m(
            WORK_ROOT / "scaffolds" / case_id / f"{case_id}.a3m",
            dst_dir / f"{export_name}.a3m",
            export_name,
            str(row["binder_observed_seq"]),
            str(row["binder_full_seq"]),
            query_mask_ranges,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the full Protein-Protein external-case delivery flow from a TSV spec.")
    ap.add_argument("--spec-tsv", type=Path, required=True, help="TSV with case_id, complex_cif, binder_chains, antigen_chains, mask_ranges, and optional binder_full_fasta/antigen_full_fasta/export_name")
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
