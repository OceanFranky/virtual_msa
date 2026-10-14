import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def _split(value: str):
    if not value:
        return []
    if ":" in value:
        parts = value.split(":")
    else:
        parts = value.split(",")
    return [x.strip() for x in parts if x.strip()]


def read_first_a3m_seq(a3m_path: Path) -> str:
    with open(a3m_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            return line
    raise ValueError(f"No sequence found in a3m: {a3m_path}")


def find_best_cif(results_root: Path, name: str):
    run_root = results_root / name
    run_dirs = sorted([p for p in run_root.glob("run_*") if p.is_dir()])
    for rd in run_dirs:
        p = rd / "best.cif"
        if p.exists():
            return str(p)
    p = run_root / "best" / "best.cif"
    if p.exists():
        return str(p)
    return None


def run_one(args, repo_root: Path, subdir: Path):
    name = subdir.name
    a3m = subdir / f"{name}.a3m"
    if not a3m.exists():
        return None, f"missing file: {a3m}"
    seq = read_first_a3m_seq(a3m)

    design_py = repo_root / "Protenix_ph" / "design.py"
    cmd = [
        sys.executable,
        str(design_py),
        "--name",
        name,
        "--mode",
        args.mode,
        "--seq",
        seq,
        "--num_designs",
        str(args.num_designs),
        "--num_cycles",
        str(args.num_cycles),
        "--gpu_id",
        str(args.gpu_id),
        "--binder_msa_mode",
        args.binder_msa_mode,
        "--target_msa_mode",
        args.target_msa_mode,
        "--msa_max_seqs",
        str(args.msa_max_seqs),
        "--model_name",
        args.model_name,
    ]

    if args.auto_constraint_model:
        cmd.append("--auto_constraint_model")

    # target inputs
    cmd.extend(["--target_type", args.target_type])
    if args.target_seq:
        cmd.extend(["--target_seq", args.target_seq])
    if args.target_seqs:
        cmd.extend(["--target_seqs", args.target_seqs])
    if args.target_count:
        cmd.extend(["--target_count", str(args.target_count)])
    if args.target_id:
        cmd.extend(["--target_id", args.target_id])
    if args.protein_seqs:
        cmd.extend(["--protein_seqs", args.protein_seqs])
    if args.ligand_ccd:
        cmd.extend(["--ligand_ccd", args.ligand_ccd])
    if args.ligand_smiles:
        cmd.extend(["--ligand_smiles", args.ligand_smiles])
    if args.ligand_file:
        cmd.extend(["--ligand_file", args.ligand_file])

    # template + msa paths
    if args.binder_msa:
        cmd.extend(["--binder_msa", args.binder_msa])
    if args.target_msa:
        cmd.extend(["--target_msa", args.target_msa])
    if args.template_path:
        cmd.extend(["--template_path", args.template_path])
    if args.target_template_path:
        cmd.extend(["--target_template_path", args.target_template_path])
    if args.binder_template_path:
        cmd.extend(["--binder_template_path", args.binder_template_path])

    # constraints
    if args.constraint_json:
        cmd.extend(["--constraint_json", args.constraint_json])
    if args.constraint_inline_json:
        cmd.extend(["--constraint_inline_json", args.constraint_inline_json])
    if args.pocket_binder_entity:
        cmd.extend(["--pocket_binder_entity", args.pocket_binder_entity])
    if args.pocket_contact_residues:
        cmd.extend(["--pocket_contact_residues", args.pocket_contact_residues])
        cmd.extend(["--pocket_max_distance", str(args.pocket_max_distance)])
    if args.contact_constraints:
        cmd.extend(["--contact_constraints", args.contact_constraints])

    if args.fix_all_if_no_x:
        cmd.append("--fix_all_if_no_x")

    if args.save_root:
        cmd.extend(["--save_dir", str(Path(args.save_root).resolve() / name)])

    if args.dry_run:
        print(" ".join(cmd))
        return None, "dry_run"

    subprocess.run(cmd, check=True, cwd=str(repo_root))
    results_root = Path(args.save_root).resolve() if args.save_root else (repo_root / "results_protenix")
    return find_best_cif(results_root, name), None


def run_batch(args):
    repo_root = Path(__file__).resolve().parents[1]
    input_root = Path(args.root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"input root not found: {input_root}")

    records = []
    for subdir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        name = subdir.name
        best_cif = None
        error = ""
        try:
            best_cif, error = run_one(args, repo_root, subdir)
        except Exception as e:
            error = str(e)
        records.append({"name": name, "best_cif": best_cif or "", "error": error or ""})

    out_csv = Path(args.output_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "best_cif", "error"])
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"Batch finished. Summary: {out_csv}")


def build_parser():
    p = argparse.ArgumentParser(description="Batch runner for Protenix Protein-Hunter pipeline.")
    p.add_argument("--root", required=True, type=str, help="Input root containing per-case subdirs with <name>.a3m")
    p.add_argument("--mode", default="binder", choices=["binder", "unconditional"])
    p.add_argument("--num_designs", default=1, type=int)
    p.add_argument("--num_cycles", default=1, type=int)
    p.add_argument("--gpu_id", default=0, type=int)
    p.add_argument("--save_root", default="", type=str)
    p.add_argument("--output_csv", default="./protenix_batch_summary.csv", type=str)
    p.add_argument("--dry_run", action="store_true", default=False)

    # target
    p.add_argument("--target_type", default="protein", choices=["protein", "ligand", "dna", "rna", "ion"])
    p.add_argument("--target_seq", default="", type=str)
    p.add_argument("--target_seqs", default="", type=str)
    p.add_argument("--target_count", default=1, type=int)
    p.add_argument("--target_id", default="", type=str)
    p.add_argument("--protein_seqs", default="", type=str)
    p.add_argument("--ligand_ccd", default="", type=str)
    p.add_argument("--ligand_smiles", default="", type=str)
    p.add_argument("--ligand_file", default="", type=str)

    # msa/template
    p.add_argument("--binder_msa_mode", default="precomputed", choices=["empty", "precomputed", "mmseqs"])
    p.add_argument("--target_msa_mode", default="empty", choices=["empty", "precomputed", "mmseqs"])
    p.add_argument("--binder_msa", default="", type=str)
    p.add_argument("--target_msa", default="", type=str)
    p.add_argument("--msa_max_seqs", default=500, type=int)
    p.add_argument("--template_path", default="", type=str)
    p.add_argument("--target_template_path", default="", type=str)
    p.add_argument("--binder_template_path", default="", type=str)

    # constraints/model
    p.add_argument("--model_name", default="protenix_base_default_v1.0.0", type=str)
    p.add_argument("--auto_constraint_model", action="store_true", default=False)
    p.add_argument("--constraint_json", default="", type=str)
    p.add_argument("--constraint_inline_json", default="", type=str)
    p.add_argument("--pocket_binder_entity", default="", type=str)
    p.add_argument("--pocket_contact_residues", default="", type=str)
    p.add_argument("--pocket_max_distance", default=6.0, type=float)
    p.add_argument("--contact_constraints", default="", type=str)
    p.add_argument("--fix_all_if_no_x", action="store_true", default=False)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_batch(args)
