#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import json
import shutil
import subprocess
import sys
from pathlib import Path

PL_BUILTIN_BUILDER = Path(__file__).resolve().parent / "pl_builtin_to_scaffolds.py"
PP_BUILTIN_BUILDER = Path(__file__).resolve().parent / "pp_builtin_to_scaffolds.py"
PP_EXTERNAL_BUILDER = Path(__file__).resolve().parent / "pp_external_to_scaffolds.py"
PL_EXTERNAL_BUILDER = Path(__file__).resolve().parent / "pl_external_to_scaffolds.py"


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


def normalize_pdb(src: Path, dst: Path) -> None:
    lines_out: list[str] = []
    for raw in src.read_text().splitlines():
        if not raw.startswith("ATOM"):
            continue
        line = raw
        if len(line) < 80:
            line = line.ljust(80)
        if line[21] == " ":
            line = f"{line[:21]}A{line[22:]}"
        lines_out.append(line.rstrip() + "\n")
    lines_out.append("END\n")
    dst.write_text("".join(lines_out))


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


def iter_unique_pdb_resseqs(path: Path) -> list[int]:
    residues: list[int] = []
    seen = set()
    for raw in path.read_text().splitlines():
        if not raw.startswith("ATOM"):
            continue
        key = (raw[21], raw[22:26], raw[26], raw[17:20])
        if key in seen:
            continue
        seen.add(key)
        residues.append(int(raw[22:26].strip()))
    return residues


def infer_sequence_length(pdb_path: Path) -> int:
    fasta_path = pdb_path.with_suffix(".fasta")
    if fasta_path.exists():
        records = read_fasta_records(fasta_path)
        if len(records) == 1:
            return len(records[0][1])
    residues = iter_unique_pdb_resseqs(pdb_path)
    if not residues:
        raise RuntimeError(f"no ATOM residues found in {pdb_path}")
    return max(residues)


def read_single_fasta(path: Path) -> str:
    records = read_fasta_records(path)
    if len(records) != 1:
        raise RuntimeError(f"expected exactly one fasta record in {path}")
    return records[0][1]


def parse_pdb_residue_records(path: Path) -> dict[int, str]:
    residues: dict[int, str] = {}
    seen = set()
    for raw in path.read_text().splitlines():
        if not raw.startswith("ATOM"):
            continue
        resseq = int(raw[22:26].strip())
        resname = raw[17:20].strip().upper()
        key = (raw[21], resseq, raw[26], resname)
        if key in seen:
            continue
        seen.add(key)
        residues[resseq] = resname
    return residues


def write_mapping_tsv(fasta_sequence: str, pdb_path: Path, out_path: Path) -> None:
    residues = parse_pdb_residue_records(pdb_path)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seq_id",
                "fasta_aa",
                "kept_in_cleaned_pdb",
                "cleaned_pdb_chain_id",
                "cleaned_pdb_resseq",
                "cleaned_pdb_icode",
                "cleaned_pdb_resname",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for seq_id, aa in enumerate(fasta_sequence, start=1):
            resname = residues.get(seq_id, "")
            writer.writerow(
                {
                    "seq_id": seq_id,
                    "fasta_aa": aa,
                    "kept_in_cleaned_pdb": 1 if seq_id in residues else 0,
                    "cleaned_pdb_chain_id": "A" if seq_id in residues else "",
                    "cleaned_pdb_resseq": seq_id if seq_id in residues else "",
                    "cleaned_pdb_icode": "",
                    "cleaned_pdb_resname": resname,
                }
            )


def l40_policy_for_length(length: int) -> dict[str, int | str]:
    if length <= 700:
        return {
            "batch_size": 100,
            "num_seq_per_target": 100,
            "n_temps": 5,
            "sampling_temp": "0.1 0.2 0.3 0.4 0.5",
            "strategy": "group targets inside the same 100-aa bin and run together",
        }
    if length <= 1100:
        return {
            "batch_size": 10,
            "num_seq_per_target": 100,
            "n_temps": 5,
            "sampling_temp": "0.1 0.2 0.3 0.4 0.5",
            "strategy": "group targets inside the same 100-aa bin and run together; lower batch for long chains",
        }
    return {
        "batch_size": 1,
        "num_seq_per_target": 100,
        "n_temps": 5,
        "sampling_temp": "0.1 0.2 0.3 0.4 0.5",
        "strategy": "run targets one by one; xlong chains exhaust L40 memory quickly",
    }


def load_mapping(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    rows.sort(key=lambda row: int(row["seq_id"]))
    return rows


def load_parsed_sequences(parsed_jsonl: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    with parsed_jsonl.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                parsed[row["name"].upper()] = row["seq"]
    return parsed


def load_selected_loops(path: Path) -> dict[str, list[tuple[int, int]]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for row in csv.DictReader(path.open(), delimiter="\t"):
        sample_name = (row.get("sample_name") or row.get("pdb") or "").upper()
        if not sample_name:
            continue
        grouped.setdefault(sample_name, []).append((int(row["loop_start"]), int(row["loop_end"])))
    for sample_name in grouped:
        grouped[sample_name].sort()
    return grouped


def build_observed_seq_ids(mapping_rows: list[dict[str, str]]) -> list[int]:
    return [int(row["seq_id"]) for row in mapping_rows if row["kept_in_cleaned_pdb"] == "1"]


def project_sequence_to_full_length(
    designed_sequence: str,
    parsed_sequence: str,
    mapping_rows: list[dict[str, str]],
    full_length_sequence: str,
) -> str:
    if len(designed_sequence) != len(parsed_sequence):
        raise RuntimeError(
            f"designed sequence length {len(designed_sequence)} != parsed sequence length {len(parsed_sequence)}"
        )
    observed_seq_ids = {int(row["seq_id"]) for row in mapping_rows if row["kept_in_cleaned_pdb"] == "1"}
    chars = ["-"] * len(full_length_sequence)
    matcher = SequenceMatcher(a=full_length_sequence, b=parsed_sequence, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            continue
        overlap = min(i2 - i1, j2 - j1)
        for offset in range(overlap):
            seq_id = i1 + offset + 1
            if seq_id in observed_seq_ids:
                chars[seq_id - 1] = designed_sequence[j1 + offset]
    return "".join(chars)


def apply_loop_mask(sequence: str, loop_ranges: list[tuple[int, int]], fill_char: str) -> str:
    chars = list(sequence)
    for start, end in loop_ranges:
        for idx in range(start - 1, end):
            if 0 <= idx < len(chars):
                chars[idx] = fill_char
    return "".join(chars)


def residue_in_ranges(chain_id: str, resseq: int, loop_ranges: list[tuple[int, int]]) -> bool:
    if chain_id != "A":
        return False
    for start, end in loop_ranges:
        if start <= resseq <= end:
            return True
    return False


def write_masked_pdb(src_pdb: Path, dst_pdb: Path, loop_ranges: list[tuple[int, int]]) -> None:
    lines_out: list[str] = []
    for raw in src_pdb.read_text().splitlines():
        if raw.startswith("ATOM"):
            line = raw if len(raw) >= 80 else raw.ljust(80)
            chain_id = line[21]
            resseq = int(line[22:26].strip())
            if residue_in_ranges(chain_id, resseq, loop_ranges):
                continue
            lines_out.append(line.rstrip() + "\n")
            continue
        if raw.startswith("END"):
            continue
    lines_out.append("END\n")
    dst_pdb.write_text("".join(lines_out))


def cmd_prepare(args: argparse.Namespace) -> int:
    rows = list(csv.DictReader(args.selection_tsv.open(), delimiter="\t"))
    required = {"sample_name", "sequence_len", "reference_pdb"}
    if not rows:
        raise RuntimeError("selection TSV is empty")
    if not required.issubset(rows[0].keys()):
        raise RuntimeError(f"selection TSV must contain columns {sorted(required)}")

    run_root = args.run_root.resolve()
    pdb_inputs = run_root / "pdb_inputs"
    prep = run_root / "prep"
    pdb_inputs.mkdir(parents=True, exist_ok=True)
    prep.mkdir(parents=True, exist_ok=True)

    copied_rows: list[dict[str, str]] = []
    for row in rows:
        sample_name = row["sample_name"].upper()
        src = Path(row["reference_pdb"]).resolve()
        dst = pdb_inputs / f"{sample_name}.pdb"
        normalize_pdb(src, dst)
        copied_rows.append(
            {
                "sample_name": sample_name,
                "sequence_len": row["sequence_len"],
                "source_pdb": str(src),
                "prepared_pdb": str(dst),
            }
        )

    parsed_jsonl = prep / "parsed_pdbs.jsonl"
    subprocess.run(
        [
            str(args.python_bin.resolve()),
            str(args.parse_script.resolve()),
            "--input_path",
            str(pdb_inputs),
            "--output_path",
            str(parsed_jsonl),
        ],
        check=True,
    )

    entries = []
    with parsed_jsonl.open() as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))

    valid_jsonl = prep / "valid_targets.jsonl"
    rejected_tsv = prep / "rejected_targets.tsv"
    summary_tsv = prep / "prepared_targets.tsv"
    valid_names = set()
    with valid_jsonl.open("w") as out:
        for entry in entries:
            if len(entry.get("seq", "")) > 0:
                out.write(json.dumps(entry) + "\n")
                valid_names.add(entry["name"].upper())

    with rejected_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_name", "reason"], delimiter="\t")
        writer.writeheader()
        for row in copied_rows:
            if row["sample_name"] not in valid_names:
                writer.writerow({"sample_name": row["sample_name"], "reason": "parse_zero_seq"})

    with summary_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_name", "sequence_len", "source_pdb", "prepared_pdb", "parse_status"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in copied_rows:
            writer.writerow(
                {
                    **row,
                    "parse_status": "valid" if row["sample_name"] in valid_names else "parse_zero_seq",
                }
            )

    print(f"requested_targets {len(copied_rows)}")
    print(f"valid_targets {len(valid_names)}")
    print(f"run_root {run_root}")
    print(f"valid_jsonl {valid_jsonl}")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    buckets = [
        ("short", 0, 350, args.short_batch_size),
        ("medium", 351, 700, args.medium_batch_size),
        ("long", 701, 1100, args.long_batch_size),
        ("xlong", 1101, 1000000, args.xlong_batch_size),
    ]

    def bucket_for_length(length: int) -> tuple[str, int]:
        for name, lo, hi, batch_size in buckets:
            if lo <= length <= hi:
                return name, batch_size
        raise RuntimeError(f"no bucket for length {length}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {}
    rows = []
    try:
        for name, _, _, _ in buckets:
            handles[name] = (out_dir / f"{name}.jsonl").open("w")
        with args.input_jsonl.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                length = len(entry["seq"])
                bucket, batch_size = bucket_for_length(length)
                handles[bucket].write(json.dumps(entry) + "\n")
                rows.append({"name": entry["name"], "length": length, "bucket": bucket, "batch_size": batch_size})
    finally:
        for handle in handles.values():
            handle.close()

    with (out_dir / "bucket_summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "length", "bucket", "batch_size"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    counts = {name: 0 for name, _, _, _ in buckets}
    for row in rows:
        counts[row["bucket"]] += 1
    for name, _, _, batch_size in buckets:
        print(f"bucket {name} count={counts[name]} batch_size={batch_size} path={out_dir / f'{name}.jsonl'}")
    return 0


def cmd_plan_from_pdb_dir(args: argparse.Namespace) -> int:
    pdb_dir = args.pdb_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    plan_rows: list[dict[str, str]] = []
    grouped: dict[int, list[dict[str, str]]] = {}
    for pdb_path in sorted(pdb_dir.rglob("*.pdb")):
        if pdb_path.name.endswith("_mask.pdb"):
            continue
        sample_name = pdb_path.stem
        length = infer_sequence_length(pdb_path)
        bin_start = ((length - 1) // 100) * 100 + 1
        bin_end = bin_start + 99
        policy = l40_policy_for_length(length)
        row = {
            "sample_name": sample_name,
            "pdb_path": str(pdb_path),
            "sequence_len": str(length),
            "bin_start": str(bin_start),
            "bin_end": str(bin_end),
            "batch_size": str(policy["batch_size"]),
            "num_seq_per_target": str(policy["num_seq_per_target"]),
            "n_temps": str(policy["n_temps"]),
            "sampling_temp": str(policy["sampling_temp"]),
            "strategy": str(policy["strategy"]),
        }
        rows.append(row)
        grouped.setdefault(bin_start, []).append(row)

    inventory_tsv = out_dir / "pdb_inventory.tsv"
    with inventory_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [], delimiter="\t")
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    group_dir = out_dir / "target_groups"
    group_dir.mkdir(parents=True, exist_ok=True)

    for bin_start in sorted(grouped):
        bin_rows = grouped[bin_start]
        first = bin_rows[0]
        group_label = f"group_{bin_start:04d}_{int(first['bin_end']):04d}"
        target_names = [r["sample_name"] for r in sorted(bin_rows, key=lambda r: (int(r["sequence_len"]), r["sample_name"]))]
        group_json = group_dir / f"{group_label}.json"
        group_json.write_text(
            json.dumps(
                {
                    "group_label": group_label,
                    "bin_start": int(first["bin_start"]),
                    "bin_end": int(first["bin_end"]),
                    "target_count": len(target_names),
                    "target_names": target_names,
                    "batch_size": int(first["batch_size"]),
                    "num_seq_per_target": int(first["num_seq_per_target"]),
                    "n_temps": int(first["n_temps"]),
                    "sampling_temp": first["sampling_temp"],
                    "strategy": first["strategy"],
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n"
        )
        plan_rows.append(
            {
                "group_label": group_label,
                "bin_label": f"{bin_start:04d}-{int(first['bin_end']):04d}",
                "target_count": str(len(bin_rows)),
                "min_length": str(min(int(r["sequence_len"]) for r in bin_rows)),
                "max_length": str(max(int(r["sequence_len"]) for r in bin_rows)),
                "batch_size": first["batch_size"],
                "num_seq_per_target": first["num_seq_per_target"],
                "n_temps": first["n_temps"],
                "sampling_temp": first["sampling_temp"],
                "strategy": first["strategy"],
                "target_names": ",".join(target_names),
                "group_file": str(group_json),
            }
        )

    plan_tsv = out_dir / "l40_100aa_plan.tsv"
    with plan_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plan_rows[0].keys()) if plan_rows else [], delimiter="\t")
        if plan_rows:
            writer.writeheader()
            writer.writerows(plan_rows)

    print(f"pdb_count {len(rows)}")
    print(f"inventory_tsv {inventory_tsv}")
    print(f"plan_tsv {plan_tsv}")
    print(f"group_dir {group_dir}")
    return 0


def cmd_init_scaffolds(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.resolve()
    scaffold_root = args.scaffold_root.resolve()
    scaffold_root.mkdir(parents=True, exist_ok=True)

    pdb_paths = sorted(p for p in input_dir.rglob("*.pdb") if not p.name.endswith("_mask.pdb"))
    created = 0
    for pdb_path in pdb_paths:
        sample_name = pdb_path.stem
        fasta_path = pdb_path.with_suffix(".fasta")
        if not fasta_path.exists():
            continue
        fasta_sequence = read_single_fasta(fasta_path)
        sample_dir = scaffold_root / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        reference_pdb = sample_dir / f"{sample_name}.pdb"
        mask_pdb = sample_dir / f"{sample_name}_mask.pdb"
        out_fasta = sample_dir / f"{sample_name}.fasta"
        mapping_tsv = sample_dir / f"{sample_name}.mapping.tsv"

        normalize_pdb(pdb_path, reference_pdb)
        shutil.copyfile(reference_pdb, mask_pdb)
        out_fasta.write_text(f">{sample_name}\n{fasta_sequence}\n")
        write_mapping_tsv(fasta_sequence, reference_pdb, mapping_tsv)
        created += 1

    print(f"created_scaffolds {created}")
    print(f"scaffold_root {scaffold_root}")
    return 0


def cmd_run_bucket(args: argparse.Namespace) -> int:
    cmd = [
        str(args.python_bin.resolve()),
        str(args.protein_seq_design_script.resolve()),
        "--jsonl_path",
        str(args.jsonl_path.resolve()),
        "--out_folder",
        str(args.out_folder.resolve()),
        "--num_seq_per_target",
        str(args.num_seq_per_target),
        "--batch_size",
        str(args.batch_size),
        "--sampling_temp",
        args.sampling_temp,
        "--model_name",
        args.model_name,
        "--seed",
        str(args.seed),
    ]
    subprocess.run(cmd, check=True)
    return 0


def cmd_mask_pdbs(args: argparse.Namespace) -> int:
    scaffold_root = args.scaffold_root.resolve()
    selected_loops = load_selected_loops(args.selected_loops_tsv.resolve())
    updated = 0
    for scaffold_dir in sorted(path for path in scaffold_root.iterdir() if path.is_dir()):
        sample_name = scaffold_dir.name.upper()
        reference_pdb = scaffold_dir / f"{sample_name}.pdb"
        mask_pdb = scaffold_dir / f"{sample_name}_mask.pdb"
        loop_ranges = selected_loops.get(sample_name, [])
        if not reference_pdb.exists():
            continue
        if loop_ranges:
            write_masked_pdb(reference_pdb, mask_pdb, loop_ranges)
        else:
            shutil.copyfile(reference_pdb, mask_pdb)
        updated += 1
    print(f"updated_mask_pdb {updated}")
    print(f"scaffold_root {scaffold_root}")
    return 0


def cmd_project(args: argparse.Namespace) -> int:
    scaffold_root = args.scaffold_root.resolve()
    parsed_sequences = load_parsed_sequences(args.parsed_jsonl.resolve())
    selected_loops = load_selected_loops(args.selected_loops_tsv.resolve()) if args.selected_loops_tsv else {}
    ok = 0
    failed = 0
    for fa_path in sorted(args.seq_design_dir.resolve().glob("*.fa")):
        sample_name = fa_path.stem.upper()
        scaffold_dir = scaffold_root / sample_name
        fasta_path = scaffold_dir / f"{sample_name}.fasta"
        mapping_path = scaffold_dir / f"{sample_name}.mapping.tsv"
        a3m_path = scaffold_dir / f"{sample_name}.a3m"
        if not fasta_path.exists() or not mapping_path.exists():
            continue
        parsed_sequence = parsed_sequences.get(sample_name)
        if parsed_sequence is None:
            print(f"failed {sample_name}: missing parsed sequence")
            failed += 1
            continue
        try:
            full_length_records = read_fasta_records(fasta_path)
            if len(full_length_records) != 1:
                raise RuntimeError(f"expected one full-length fasta record in {fasta_path}")
            full_length_sequence = full_length_records[0][1]
            mapping_rows = load_mapping(mapping_path)
            records = read_fasta_records(fa_path)
            if len(records) < 2:
                raise RuntimeError("no sampled sequences")
            projected_records = []
            for header, seq in records[1:]:
                projected = project_sequence_to_full_length(seq, parsed_sequence, mapping_rows, full_length_sequence)
                if sample_name in selected_loops:
                    projected = apply_loop_mask(projected, selected_loops[sample_name], "-")
                projected_records.append((header, projected))
        except Exception as exc:
            print(f"failed {sample_name}: {exc}")
            failed += 1
            continue

        query_sequence = full_length_sequence
        if sample_name in selected_loops:
            query_sequence = apply_loop_mask(query_sequence, selected_loops[sample_name], "X")
        with a3m_path.open("w") as handle:
            handle.write(f">{sample_name}\n{query_sequence}\n")
            for header, seq in projected_records:
                handle.write(f">{header}\n{seq}\n")
        ok += 1

    print(f"a3m_count {ok}")
    print(f"failed_count {failed}")
    return 0 if failed == 0 else 1


def cmd_rebuild_npz(args: argparse.Namespace) -> int:
    subprocess.run(
        [
            str(args.python_bin.resolve()),
            str(args.converter.resolve()),
            "--scaffold-root",
            str(args.scaffold_root.resolve()),
        ],
        check=True,
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    scaffold_root = args.scaffold_root.resolve()
    fasta_ok = 0
    a3m_ok = 0
    bad = 0
    for scaffold_dir in sorted(path for path in scaffold_root.iterdir() if path.is_dir()):
        sample_name = scaffold_dir.name
        fasta_path = scaffold_dir / f"{sample_name}.fasta"
        a3m_path = scaffold_dir / f"{sample_name}.a3m"
        mapping_path = scaffold_dir / f"{sample_name}.mapping.tsv"
        npz_path = scaffold_dir / f"{sample_name}.npz"
        if not (fasta_path.exists() and a3m_path.exists() and mapping_path.exists() and npz_path.exists()):
            print(f"missing_core {sample_name}")
            bad += 1
            continue
        fasta_records = read_fasta_records(fasta_path)
        a3m_records = read_fasta_records(a3m_path)
        if len(fasta_records) != 1 or len(a3m_records) < 2:
            print(f"bad_records {sample_name}")
            bad += 1
            continue
        fasta_seq = fasta_records[0][1]
        query_seq = a3m_records[0][1]
        selected_loops = load_selected_loops(args.selected_loops_tsv.resolve()) if args.selected_loops_tsv else {}
        expected_query = fasta_seq
        if sample_name.upper() in selected_loops:
            expected_query = apply_loop_mask(fasta_seq, selected_loops[sample_name.upper()], "X")
        if expected_query != query_seq:
            print(f"query_mismatch {sample_name}")
            bad += 1
            continue
        fasta_ok += 1
        mapping_rows = load_mapping(mapping_path)
        observed_seq_ids = set(build_observed_seq_ids(mapping_rows))
        loop_positions = set()
        for start, end in selected_loops.get(sample_name.upper(), []):
            loop_positions.update(range(start, end + 1))
        for header, seq in a3m_records[1:]:
            if len(seq) != len(fasta_seq):
                print(f"sample_len_mismatch {sample_name} {header}")
                bad += 1
                break
            for idx, aa in enumerate(seq, start=1):
                if idx in loop_positions and aa != "-":
                    print(f"loop_not_masked {sample_name} pos={idx} header={header}")
                    bad += 1
                    break
                if idx not in observed_seq_ids and aa != "-":
                    print(f"unobserved_not_gap {sample_name} pos={idx} header={header}")
                    bad += 1
                    break
            else:
                continue
            break
        else:
            a3m_ok += 1
    print(f"query_equals_fasta {fasta_ok}")
    print(f"projected_a3m_ok {a3m_ok}")
    print(f"bad {bad}")
    return 0 if bad == 0 else 1


def cmd_build_delivery(args: argparse.Namespace) -> int:
    work_dir = args.work_dir.resolve()
    scaffold_root = work_dir / "scaffolds"
    run_root = work_dir / "proteinseq_design_run"
    scaffold_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    init_args = argparse.Namespace(input_dir=args.input_dir, scaffold_root=scaffold_root)
    cmd_init_scaffolds(init_args)

    if args.selected_loops_tsv:
        mask_args = argparse.Namespace(
            scaffold_root=scaffold_root,
            selected_loops_tsv=args.selected_loops_tsv,
        )
        cmd_mask_pdbs(mask_args)

    selection_tsv = work_dir / "selection.tsv"
    rows = []
    for sample_dir in sorted(p for p in scaffold_root.iterdir() if p.is_dir()):
        sample_name = sample_dir.name
        fasta_seq = read_single_fasta(sample_dir / f"{sample_name}.fasta")
        reference_pdb = sample_dir / f"{sample_name}_mask.pdb"
        if not reference_pdb.exists():
            reference_pdb = sample_dir / f"{sample_name}.pdb"
        rows.append(
            {
                "sample_name": sample_name,
                "sequence_len": len(fasta_seq),
                "reference_pdb": str(reference_pdb),
            }
        )
    with selection_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_name", "sequence_len", "reference_pdb"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    prepare_args = argparse.Namespace(
        selection_tsv=selection_tsv,
        run_root=run_root,
        parse_script=args.parse_script,
        python_bin=args.protein_seq_design_python,
    )
    cmd_prepare(prepare_args)

    split_args = argparse.Namespace(
        input_jsonl=run_root / "prep" / "valid_targets.jsonl",
        out_dir=run_root / "prep" / "buckets",
        short_batch_size=100,
        medium_batch_size=100,
        long_batch_size=10,
        xlong_batch_size=1,
    )
    cmd_split(split_args)

    buckets = [
        ("short", 100),
        ("medium", 100),
        ("long", 10),
        ("xlong", 1),
    ]
    for bucket_name, batch_size in buckets:
        jsonl_path = run_root / "prep" / "buckets" / f"{bucket_name}.jsonl"
        if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
            continue
        subprocess.run(
            [
                str(args.protein_seq_design_python.resolve()),
                str(args.protein_seq_design_script.resolve()),
                "--jsonl_path",
                str(jsonl_path),
                "--out_folder",
                str(run_root / "raw_output"),
                "--num_seq_per_target",
                "100",
                "--batch_size",
                str(batch_size),
                "--sampling_temp",
                "0.1 0.2 0.3 0.4 0.5",
                "--model_name",
                "v_48_020",
                "--seed",
                "37",
            ],
            check=True,
        )

    project_args = argparse.Namespace(
        scaffold_root=scaffold_root,
        seq_design_dir=run_root / "raw_output" / "seqs",
        parsed_jsonl=run_root / "prep" / "parsed_pdbs.jsonl",
        selected_loops_tsv=args.selected_loops_tsv,
    )
    cmd_project(project_args)

    npz_args = argparse.Namespace(
        scaffold_root=scaffold_root,
        converter=args.npz_converter,
        python_bin=args.npz_python,
    )
    cmd_rebuild_npz(npz_args)

    verify_args = argparse.Namespace(scaffold_root=scaffold_root, selected_loops_tsv=args.selected_loops_tsv)
    cmd_verify(verify_args)

    print(f"delivery_root {work_dir}")
    print(f"scaffold_root {scaffold_root}")
    return 0


def cmd_hunter_pl_builtin_to_scaffolds(args: argparse.Namespace) -> int:
    subprocess.run(
        [
            str(args.python_bin.resolve()),
            str(PL_BUILTIN_BUILDER.resolve()),
            "--gpu",
            str(args.gpu),
        ],
        check=True,
    )
    return 0


def cmd_hunter_pp_builtin_to_scaffolds(args: argparse.Namespace) -> int:
    subprocess.run(
        [
            str(args.python_bin.resolve()),
            str(PP_BUILTIN_BUILDER.resolve()),
            "--gpu",
            str(args.gpu),
        ],
        check=True,
    )
    return 0


def cmd_hunter_pp_external_to_scaffolds(args: argparse.Namespace) -> int:
    subprocess.run(
        [
            str(args.python_bin.resolve()),
            str(PP_EXTERNAL_BUILDER.resolve()),
            "--spec-tsv",
            str(args.spec_tsv.resolve()),
            "--gpu",
            str(args.gpu),
        ],
        check=True,
    )
    return 0


def cmd_hunter_pl_external_to_scaffolds(args: argparse.Namespace) -> int:
    subprocess.run(
        [
            str(args.python_bin.resolve()),
            str(PL_EXTERNAL_BUILDER.resolve()),
            "--spec-tsv",
            str(args.spec_tsv.resolve()),
            "--gpu",
            str(args.gpu),
        ],
        check=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reusable Protein Sequence Design delivery pipeline helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan-from-pdb-dir", help="Scan a PDB directory and emit a 100-aa L40 run plan")
    p.add_argument("--pdb-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=cmd_plan_from_pdb_dir)

    p = sub.add_parser("init-scaffolds", help="Initialize scaffold directories from paired .pdb and .fasta files")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--scaffold-root", type=Path, required=True)
    p.set_defaults(func=cmd_init_scaffolds)

    p = sub.add_parser("prepare", help="Normalize PDBs and run Protein Sequence Design parse_multiple_chains.py")
    p.add_argument("--selection-tsv", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--parse-script", type=Path, required=True)
    p.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("split", help="Split parsed JSONL into length buckets")
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--short-batch-size", type=int, default=100)
    p.add_argument("--medium-batch-size", type=int, default=100)
    p.add_argument("--long-batch-size", type=int, default=100)
    p.add_argument("--xlong-batch-size", type=int, default=100)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("run-bucket", help="Run Protein Sequence Design on one bucket JSONL")
    p.add_argument("--jsonl-path", type=Path, required=True)
    p.add_argument("--out-folder", type=Path, required=True)
    p.add_argument("--protein-seq-design-script", type=Path, required=True)
    p.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    p.add_argument("--num-seq-per-target", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--sampling-temp", default="0.1 0.2 0.3 0.4 0.5")
    p.add_argument("--model-name", default="v_48_020")
    p.add_argument("--seed", type=int, default=37)
    p.set_defaults(func=cmd_run_bucket)

    p = sub.add_parser("mask-pdbs", help="Rewrite *_mask.pdb by removing selected loop residues")
    p.add_argument("--scaffold-root", type=Path, required=True)
    p.add_argument("--selected-loops-tsv", type=Path, required=True)
    p.set_defaults(func=cmd_mask_pdbs)

    p = sub.add_parser("project", help="Project Protein Sequence Design sampled sequences back to full-length A3M")
    p.add_argument("--scaffold-root", type=Path, required=True)
    p.add_argument("--seq-design-dir", type=Path, required=True)
    p.add_argument("--parsed-jsonl", type=Path, required=True)
    p.add_argument("--selected-loops-tsv", type=Path)
    p.set_defaults(func=cmd_project)

    p = sub.add_parser("rebuild-npz", help="Rebuild NPZ files from A3M")
    p.add_argument("--scaffold-root", type=Path, required=True)
    p.add_argument("--converter", type=Path, required=True)
    p.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    p.set_defaults(func=cmd_rebuild_npz)

    p = sub.add_parser("verify", help="Verify full-length projection semantics")
    p.add_argument("--scaffold-root", type=Path, required=True)
    p.add_argument("--selected-loops-tsv", type=Path)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("build-delivery", help="Run the full delivery pipeline from paired .pdb and .fasta inputs")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--parse-script", type=Path, required=True)
    p.add_argument("--protein-seq-design-script", type=Path, required=True)
    p.add_argument("--protein-seq-design-python", type=Path, default=Path(sys.executable))
    p.add_argument("--npz-converter", type=Path, required=True)
    p.add_argument("--npz-python", type=Path, default=Path(sys.executable))
    p.add_argument("--selected-loops-tsv", type=Path)
    p.set_defaults(func=cmd_build_delivery)

    p = sub.add_parser("hunter-pl-builtins-to-scaffolds", help="Run local Protein Sequence Design flow for the built-in Protein-Ligand cases")
    p.add_argument("--gpu", default="0")
    p.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    p.set_defaults(func=cmd_hunter_pl_builtin_to_scaffolds)

    p = sub.add_parser("hunter-pl-external-to-scaffolds", help="Run the full Protein-Ligand external-case flow from a TSV spec")
    p.add_argument("--spec-tsv", type=Path, required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    p.set_defaults(func=cmd_hunter_pl_external_to_scaffolds)

    p = sub.add_parser("hunter-pp-builtins-to-scaffolds", help="Run local Protein Sequence Design flow for the built-in Protein-Protein cases")
    p.add_argument("--gpu", default="0")
    p.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    p.set_defaults(func=cmd_hunter_pp_builtin_to_scaffolds)

    p = sub.add_parser("hunter-pp-external-to-scaffolds", help="Run the full Protein-Protein external-case flow from a TSV spec")
    p.add_argument("--spec-tsv", type=Path, required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    p.set_defaults(func=cmd_hunter_pp_external_to_scaffolds)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
