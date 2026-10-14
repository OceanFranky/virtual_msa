#!/usr/bin/env python3
"""Evaluate scaffold CA RMSD for standard_examples_5.30 design outputs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
from Bio import pairwise2


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
}


def read_query_sequence(a3m_path: Path) -> str:
    with a3m_path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                continue
            return line.strip().replace("-", "")
    raise ValueError(f"No query sequence found in {a3m_path}")


def read_residues(pdb_path: Path) -> tuple[str, list[dict[str, np.ndarray]]]:
    sequence = []
    residues: list[dict[str, np.ndarray]] = []
    current_id = None
    current_name = None
    current_atoms: dict[str, np.ndarray] = {}

    def flush() -> None:
        if current_name is None:
            return
        sequence.append(AA3_TO_1.get(current_name.upper(), "X"))
        residues.append(current_atoms.copy())

    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            residue_id = (line[21], line[22:26].strip(), line[26])
            if residue_id != current_id:
                flush()
                current_id = residue_id
                current_name = line[17:20].strip()
                current_atoms = {}
            atom = line[12:16].strip()
            current_atoms[atom] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            )
    flush()
    return "".join(sequence), residues


def map_query_to_residues(query_sequence: str, pdb_path: Path) -> dict[int, dict[str, np.ndarray]]:
    pdb_sequence, residues = read_residues(pdb_path)
    alignments = pairwise2.align.globalms(query_sequence, pdb_sequence, 2, 1, -1, -1)
    if not alignments:
        raise ValueError(f"Could not align {pdb_path} to query sequence")

    aligned_query, aligned_pdb = alignments[0].seqA, alignments[0].seqB
    query_index = 0
    pdb_index = 0
    mapping: dict[int, dict[str, np.ndarray]] = {}
    for q_char, p_char in zip(aligned_query, aligned_pdb):
        if q_char != "-" and p_char != "-":
            mapping[query_index] = residues[pdb_index]
        if q_char != "-":
            query_index += 1
        if p_char != "-":
            pdb_index += 1
    return mapping


def kabsch_fit(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mobile_centered = mobile - mobile.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    covariance = mobile_centered.T @ target_centered
    v, _, wt = np.linalg.svd(covariance)
    if np.linalg.det(v @ wt) < 0:
        v[:, -1] *= -1
    rotation = v @ wt
    return rotation, mobile.mean(axis=0), target.mean(axis=0)


def apply_transform(coords: np.ndarray, rotation: np.ndarray, mobile_center: np.ndarray, target_center: np.ndarray) -> np.ndarray:
    return (coords - mobile_center) @ rotation + target_center


def kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    rotation, mobile_center, target_center = kabsch_fit(mobile, target)
    fitted = apply_transform(mobile, rotation, mobile_center, target_center)
    diff = fitted - target
    return float(math.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def heavy_hetatm_coords(pdb_path: Path, residue_name: str | None = None) -> np.ndarray:
    coords = []
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if resname in {"HOH", "WAT"}:
                continue
            if residue_name is not None and resname != residue_name:
                continue
            element = line[76:78].strip() or line[12:16].strip()[0]
            if element.upper() == "H":
                continue
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(coords, dtype=float)


def native_ligand_coords(native_pdb: Path, expected_atoms: int) -> np.ndarray:
    by_residue: dict[tuple[str, str, str, str], list[list[float]]] = {}
    with native_pdb.open() as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if resname in {"HOH", "WAT"}:
                continue
            element = line[76:78].strip() or line[12:16].strip()[0]
            if element.upper() == "H":
                continue
            key = (line[21], line[22:26].strip(), line[26], resname)
            by_residue.setdefault(key, []).append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    exact = [coords for coords in by_residue.values() if len(coords) == expected_atoms]
    if exact:
        return np.array(exact[0], dtype=float)
    if not by_residue:
        return np.empty((0, 3), dtype=float)
    return np.array(max(by_residue.values(), key=len), dtype=float)


def scaffold_and_ligand_metrics(
    design_pdb: Path, native_pdb: Path, a3m_path: Path
) -> tuple[float, int, float | str, float | str, int | str]:
    query_sequence = read_query_sequence(a3m_path)
    design_map = map_query_to_residues(query_sequence, design_pdb)
    native_map = map_query_to_residues(query_sequence, native_pdb)

    indices = [
        i
        for i, aa in enumerate(query_sequence)
        if aa.upper() != "X"
        and i in design_map
        and i in native_map
        and "CA" in design_map[i]
        and "CA" in native_map[i]
    ]
    if not indices:
        raise ValueError(f"No shared scaffold CA atoms for {design_pdb}")

    design_ca = np.vstack([design_map[i]["CA"] for i in indices])
    native_ca = np.vstack([native_map[i]["CA"] for i in indices])
    rotation, mobile_center, target_center = kabsch_fit(design_ca, native_ca)
    fitted_ca = apply_transform(design_ca, rotation, mobile_center, target_center)
    ca_diff = fitted_ca - native_ca
    scaffold_rmsd = float(math.sqrt(np.mean(np.sum(ca_diff * ca_diff, axis=1))))

    design_ligand = heavy_hetatm_coords(design_pdb)
    native_ligand = native_ligand_coords(native_pdb, len(design_ligand))
    ligand_rmsd: float | str = ""
    ligand_centroid_shift: float | str = ""
    ligand_atoms: int | str = ""
    if len(design_ligand) and len(design_ligand) == len(native_ligand):
        fitted_ligand = apply_transform(design_ligand, rotation, mobile_center, target_center)
        ligand_diff = fitted_ligand - native_ligand
        ligand_rmsd = float(math.sqrt(np.mean(np.sum(ligand_diff * ligand_diff, axis=1))))
        ligand_centroid_shift = float(np.linalg.norm(fitted_ligand.mean(axis=0) - native_ligand.mean(axis=0)))
        ligand_atoms = len(design_ligand)

    return scaffold_rmsd, len(indices), ligand_rmsd, ligand_centroid_shift, ligand_atoms


def summarize(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        return {
            "n_models": 0,
            "mean_rmsd": "",
            "median_rmsd": "",
            "min_rmsd": "",
            "max_rmsd": "",
            "std_rmsd": "",
        }
    arr = np.array(values, dtype=float)
    return {
        "n_models": len(values),
        "mean_rmsd": round(float(arr.mean()), 4),
        "median_rmsd": round(float(np.median(arr)), 4),
        "min_rmsd": round(float(arr.min()), 4),
        "max_rmsd": round(float(arr.max()), 4),
        "std_rmsd": round(float(arr.std(ddof=0)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument(
        "--source",
        choices=("high_iptm", "all_cycles"),
        default="high_iptm",
        help="high_iptm uses summary_high_iptm.csv; all_cycles uses every predicted_cycle PDB.",
    )
    args = parser.parse_args()

    per_model_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []

    for group in ("no_msa", "with_msa"):
        group_dir = args.design_root / group
        for case_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            pdb_id = case_dir.name
            summary_path = case_dir / "summary_high_iptm.csv"
            all_runs_path = case_dir / "summary_all_runs.csv"
            native_pdb = args.native_root / pdb_id / f"{pdb_id}_clean.pdb"
            a3m_path = args.native_root / pdb_id / f"{pdb_id}.a3m"
            rmsd_values: list[float] = []
            status = "ok"

            if args.source == "high_iptm" and not summary_path.exists():
                status = "missing_summary_high_iptm"
            elif not native_pdb.exists() or not a3m_path.exists():
                status = "missing_native_or_a3m"
            else:
                metric_lookup: dict[tuple[int, int], dict[str, str]] = {}
                if all_runs_path.exists():
                    with all_runs_path.open(newline="") as handle:
                        for run_row in csv.DictReader(handle):
                            run_id = int(run_row["run_id"])
                            for cycle in range(0, 11):
                                key = (run_id, cycle)
                                metric_lookup[key] = {
                                    "iptm": run_row.get(f"cycle_{cycle}_iptm", ""),
                                    "plddt": run_row.get(f"cycle_{cycle}_plddt", ""),
                                    "iplddt": run_row.get(f"cycle_{cycle}_iplddt", ""),
                                    "sequence": run_row.get(f"cycle_{cycle}_seq", ""),
                                }

                if args.source == "high_iptm":
                    design_rows = []
                    with summary_path.open(newline="") as handle:
                        for row in csv.DictReader(handle):
                            design_rows.append(
                                {
                                    "run_id": row.get("run_id", ""),
                                    "cycle": row.get("cycle", ""),
                                    "iptm": row.get("iptm", ""),
                                    "plddt": row.get("plddt", ""),
                                    "iplddt": row.get("iplddt", ""),
                                    "sequence": row.get("sequence", ""),
                                    "pdb_filename": row["pdb_filename"],
                                    "design_pdb": case_dir / "high_iptm_pdb" / row["pdb_filename"],
                                }
                            )
                else:
                    design_rows = []
                    pdb_paths = sorted(case_dir.glob("0_protein_hunter_design/run_*/*predicted_cycle_*.pdb"))
                    if not pdb_paths:
                        status = "missing_predicted_cycle_pdb"
                    for design_pdb in pdb_paths:
                        match = re.search(r"_run_(\d+)_predicted_cycle_(\d+)\.pdb$", design_pdb.name)
                        if not match:
                            continue
                        run_id = int(match.group(1))
                        cycle = int(match.group(2))
                        metrics = metric_lookup.get((run_id, cycle), {})
                        design_rows.append(
                            {
                                "run_id": str(run_id),
                                "cycle": str(cycle),
                                "iptm": metrics.get("iptm", ""),
                                "plddt": metrics.get("plddt", ""),
                                "iplddt": metrics.get("iplddt", ""),
                                "sequence": metrics.get("sequence", ""),
                                "pdb_filename": design_pdb.name,
                                "design_pdb": design_pdb,
                            }
                        )

                for row in design_rows:
                        pdb_filename = row["pdb_filename"]
                        design_pdb = Path(row["design_pdb"])
                        out_row: dict[str, object] = {
                            "group": group,
                            "pdb_id": pdb_id,
                            "run_id": row.get("run_id", ""),
                            "cycle": row.get("cycle", ""),
                            "iptm": row.get("iptm", ""),
                            "plddt": row.get("plddt", ""),
                            "iplddt": row.get("iplddt", ""),
                            "pdb_filename": pdb_filename,
                            "design_pdb": str(design_pdb),
                            "native_pdb": str(native_pdb),
                        }
                        try:
                            rmsd, n_ca, ligand_rmsd, ligand_shift, ligand_atoms = scaffold_and_ligand_metrics(
                                design_pdb, native_pdb, a3m_path
                            )
                            out_row["scaffold_ca_rmsd"] = round(rmsd, 4)
                            out_row["n_scaffold_ca"] = n_ca
                            out_row["ligand_rmsd"] = round(ligand_rmsd, 4) if ligand_rmsd != "" else ""
                            out_row["ligand_centroid_shift"] = round(ligand_shift, 4) if ligand_shift != "" else ""
                            out_row["ligand_heavy_atoms"] = ligand_atoms
                            out_row["status"] = "ok"
                            rmsd_values.append(rmsd)
                        except Exception as exc:
                            out_row["scaffold_ca_rmsd"] = ""
                            out_row["n_scaffold_ca"] = ""
                            out_row["ligand_rmsd"] = ""
                            out_row["ligand_centroid_shift"] = ""
                            out_row["ligand_heavy_atoms"] = ""
                            out_row["status"] = f"error: {exc}"
                        per_model_rows.append(out_row)

            case_row: dict[str, object] = {"group": group, "pdb_id": pdb_id, "status": status}
            case_row.update(summarize(rmsd_values))
            case_rows.append(case_row)

    per_model_path = args.out_prefix.with_name(args.out_prefix.name + "_per_model.csv")
    per_case_path = args.out_prefix.with_name(args.out_prefix.name + "_per_case.csv")
    per_model_fields = [
        "group",
        "pdb_id",
        "run_id",
        "cycle",
        "iptm",
        "plddt",
        "iplddt",
        "scaffold_ca_rmsd",
        "n_scaffold_ca",
        "ligand_rmsd",
        "ligand_centroid_shift",
        "ligand_heavy_atoms",
        "status",
        "pdb_filename",
        "design_pdb",
        "native_pdb",
    ]
    per_case_fields = [
        "group",
        "pdb_id",
        "status",
        "n_models",
        "mean_rmsd",
        "median_rmsd",
        "min_rmsd",
        "max_rmsd",
        "std_rmsd",
    ]

    with per_model_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_model_fields)
        writer.writeheader()
        writer.writerows(per_model_rows)
    with per_case_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_case_fields)
        writer.writeheader()
        writer.writerows(case_rows)

    print(f"Wrote {per_model_path}")
    print(f"Wrote {per_case_path}")


if __name__ == "__main__":
    main()
