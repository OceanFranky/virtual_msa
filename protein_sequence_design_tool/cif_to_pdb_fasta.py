#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gemmi

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one-level raw mmCIF directories into paired Protein Sequence Design .pdb + .fasta inputs.",
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Root containing one-level sample subdirectories with .cif files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Flat output directory for paired <sample>.pdb and <sample>.fasta.")
    parser.add_argument("--preferred-auth-chain", default="A")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional TSV manifest path.")
    return parser.parse_args()


def is_amino_acid_name(name: str) -> bool:
    tab = gemmi.find_tabulated_residue(name)
    return bool(tab) and tab.is_amino_acid()


def normalize_icode(value: str) -> str:
    return "" if value in {".", "?", " "} else value


def mon_id_to_one_letter(mon_id: str) -> str:
    return AA3_TO_1.get(mon_id.upper(), "X")


def wrap_fasta(seq: str, width: int = 80) -> str:
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width)) + "\n"


def column_rows(block: gemmi.cif.Block, tags: list[str]) -> list[dict[str, str]]:
    cols = {tag: list(block.find_values(tag)) for tag in tags}
    lengths = {len(values) for values in cols.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"inconsistent loop lengths for tags: {tags}")
    row_count = next(iter(lengths))
    rows: list[dict[str, str]] = []
    for idx in range(row_count):
        row: dict[str, str] = {}
        for tag in tags:
            row[tag.rsplit(".", 1)[-1]] = cols[tag][idx]
        rows.append(row)
    return rows


def dedupe_scheme_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_seq_id: dict[int, dict[str, str]] = {}
    for row in rows:
        seq_id = int(row["seq_id"])
        current = by_seq_id.get(seq_id)
        if current is None:
            by_seq_id[seq_id] = row
            continue
        if mon_id_to_one_letter(current["mon_id"]) == "X" and mon_id_to_one_letter(row["mon_id"]) != "X":
            by_seq_id[seq_id] = row
    return [by_seq_id[idx] for idx in sorted(by_seq_id)]


def choose_target_asym_id(block: gemmi.cif.Block, preferred_auth_chain: str) -> tuple[str, str]:
    scheme_rows = column_rows(
        block,
        [
            "_pdbx_poly_seq_scheme.asym_id",
            "_pdbx_poly_seq_scheme.pdb_strand_id",
            "_pdbx_poly_seq_scheme.seq_id",
            "_pdbx_poly_seq_scheme.mon_id",
        ],
    )
    protein_rows = [row for row in scheme_rows if row["seq_id"] not in {".", "?"} and is_amino_acid_name(row["mon_id"])]
    if not protein_rows:
        raise RuntimeError("no protein polymer rows found in _pdbx_poly_seq_scheme")
    preferred = [row for row in protein_rows if row["pdb_strand_id"] == preferred_auth_chain]
    if preferred:
        row = preferred[0]
        return row["asym_id"], row["pdb_strand_id"]
    first = protein_rows[0]
    return first["asym_id"], first["pdb_strand_id"]


def build_full_length_sequence(block: gemmi.cif.Block, target_asym_id: str) -> tuple[str, list[dict[str, str]]]:
    scheme_rows = [
        row
        for row in column_rows(
            block,
            [
                "_pdbx_poly_seq_scheme.asym_id",
                "_pdbx_poly_seq_scheme.pdb_strand_id",
                "_pdbx_poly_seq_scheme.seq_id",
                "_pdbx_poly_seq_scheme.mon_id",
            ],
        )
        if row["asym_id"] == target_asym_id and row["seq_id"] not in {".", "?"}
    ]
    scheme_rows.sort(key=lambda row: int(row["seq_id"]))
    scheme_rows = dedupe_scheme_rows(scheme_rows)
    seq = "".join(mon_id_to_one_letter(row["mon_id"]) for row in scheme_rows)
    if not seq:
        raise RuntimeError(f"no sequence rows found for asym_id={target_asym_id}")
    return seq, scheme_rows


def load_allowed_residue_index_map(block: gemmi.cif.Block, target_asym_id: str) -> dict[tuple[str, str, str, str], int]:
    atom_rows = column_rows(
        block,
        [
            "_atom_site.label_asym_id",
            "_atom_site.auth_asym_id",
            "_atom_site.label_seq_id",
            "_atom_site.auth_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.label_comp_id",
        ],
    )
    allowed: dict[tuple[str, str, str, str], int] = {}
    for row in atom_rows:
        if row["label_asym_id"] != target_asym_id:
            continue
        if row["label_seq_id"] in {".", "?"}:
            continue
        if not is_amino_acid_name(row["label_comp_id"]):
            continue
        key = (
            row["auth_asym_id"],
            row["auth_seq_id"],
            normalize_icode(row["pdbx_PDB_ins_code"]),
            row["label_comp_id"],
        )
        allowed[key] = int(row["label_seq_id"])
    return allowed


def clean_and_renumber_pdb(cif_path: Path, block: gemmi.cif.Block, target_asym_id: str, out_pdb_path: Path) -> int:
    structure = gemmi.read_structure(str(cif_path))
    allowed = load_allowed_residue_index_map(block, target_asym_id)
    if not allowed:
        raise RuntimeError(f"no amino-acid ATOM residues found for asym_id={target_asym_id}")

    cleaned = gemmi.Structure()
    cleaned.name = structure.name
    cleaned.cell = structure.cell
    cleaned.spacegroup_hm = structure.spacegroup_hm

    kept = 0
    for model in structure:
        # gemmi.Model() expects the model name string in current gemmi builds.
        new_model = gemmi.Model(str(getattr(model, "name", "1")))
        new_chain = gemmi.Chain("A")
        for chain in model:
            for residue in chain:
                if not is_amino_acid_name(residue.name):
                    continue
                key = (
                    chain.name,
                    str(residue.seqid.num),
                    normalize_icode(residue.seqid.icode),
                    residue.name,
                )
                label_seq_id = allowed.get(key)
                if label_seq_id is None:
                    continue
                new_residue = residue.clone()
                new_residue.het_flag = "A"
                new_residue.seqid = gemmi.SeqId(label_seq_id, " ")
                new_chain.add_residue(new_residue)
                kept += 1
        if len(new_chain):
            new_model.add_chain(new_chain)
        if len(new_model):
            cleaned.add_model(new_model)

    if kept == 0:
        raise RuntimeError("no residues kept after cleaning")

    cleaned.write_pdb(str(out_pdb_path))
    kept_lines: list[str] = []
    with out_pdb_path.open() as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "TER   ", "END")):
                kept_lines.append(line)
    out_pdb_path.write_text("".join(kept_lines))
    return kept


def find_cif_path(sample_dir: Path) -> Path:
    preferred = sample_dir / f"{sample_dir.name.upper()}.cif"
    if preferred.exists():
        return preferred
    cif_paths = sorted(sample_dir.glob("*.cif"))
    if len(cif_paths) == 1:
        return cif_paths[0]
    raise RuntimeError(f"expected exactly one .cif in {sample_dir}")


def process_one(sample_dir: Path, output_dir: Path, preferred_auth_chain: str) -> dict[str, str | int]:
    sample_name = sample_dir.name.upper()
    cif_path = find_cif_path(sample_dir)
    block = gemmi.cif.read(str(cif_path)).sole_block()
    target_asym_id, resolved_auth_chain = choose_target_asym_id(block, preferred_auth_chain)
    full_length_sequence, scheme_rows = build_full_length_sequence(block, target_asym_id)

    pdb_path = output_dir / f"{sample_name}.pdb"
    fasta_path = output_dir / f"{sample_name}.fasta"
    kept = clean_and_renumber_pdb(cif_path, block, target_asym_id, pdb_path)
    fasta_path.write_text(f">{sample_name}|auth_chain={resolved_auth_chain}|asym_id={target_asym_id}\n{wrap_fasta(full_length_sequence)}")

    return {
        "sample_name": sample_name,
        "auth_chain": resolved_auth_chain,
        "asym_id": target_asym_id,
        "full_length": len(full_length_sequence),
        "kept_observed": kept,
        "missing": len(full_length_sequence) - kept,
        "scheme_rows": len(scheme_rows),
        "source_cif": str(cif_path),
        "output_pdb": str(pdb_path),
        "output_fasta": str(fasta_path),
    }


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest.resolve() if args.manifest else output_dir / "prepared_targets.tsv"

    rows = []
    for sample_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        rows.append(process_one(sample_dir, output_dir, args.preferred_auth_chain))

    fieldnames = [
        "sample_name",
        "auth_chain",
        "asym_id",
        "full_length",
        "kept_observed",
        "missing",
        "scheme_rows",
        "source_cif",
        "output_pdb",
        "output_fasta",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"prepared_count {len(rows)}")
    print(f"output_dir {output_dir}")
    print(f"manifest_tsv {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
