import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def read_first_a3m_seq(a3m_path):
    with open(a3m_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            return line
    raise ValueError(f"No sequence found in a3m: {a3m_path}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_file(src, dst, overwrite):
    if os.path.exists(dst) and not overwrite:
        return
    shutil.copy2(src, dst)


def prepare_test_flow(args):
    root = Path(args.root).resolve()
    pdb_id = args.pdb_id
    subdir = root / pdb_id
    ensure_dir(subdir)

    source_pdb = Path(args.source_pdb).resolve()
    source_a3m = Path(args.source_a3m).resolve()
    source_npz = Path(args.source_npz).resolve()
    if not source_pdb.exists():
        raise FileNotFoundError(f"source_pdb not found: {source_pdb}")
    if not source_a3m.exists():
        raise FileNotFoundError(f"source_a3m not found: {source_a3m}")
    if not source_npz.exists():
        raise FileNotFoundError(f"source_npz not found: {source_npz}")

    dest_pdb = subdir / f"{pdb_id}.pdb"
    dest_mask = subdir / f"{pdb_id}_mask.pdb"
    dest_a3m = subdir / f"{pdb_id}.a3m"
    dest_npz = subdir / f"{pdb_id}.npz"
    dest_fasta = subdir / f"{pdb_id}.fasta"

    copy_file(source_pdb, dest_pdb, args.overwrite)
    copy_file(source_pdb, dest_mask, args.overwrite)
    copy_file(source_a3m, dest_a3m, args.overwrite)
    copy_file(source_npz, dest_npz, args.overwrite)

    seq = read_first_a3m_seq(dest_a3m)
    if args.make_fasta:
        with open(dest_fasta, "w") as f:
            f.write(f">{pdb_id}\n{seq}\n")

    print(f"Prepared test flow at: {subdir}")


def parse_best_run_id(summary_csv_path):
    best_run_id = None
    best_iptm = None
    with open(summary_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get("run_id")
            if not run_id:
                continue
            cycle_iptms = []
            for key, value in row.items():
                if not key.endswith("_iptm") or key == "run_id":
                    continue
                if value in ("", None):
                    continue
                try:
                    cycle_iptms.append(float(value))
                except ValueError:
                    continue
            if not cycle_iptms:
                continue
            run_best_iptm = max(cycle_iptms)
            if best_iptm is None or run_best_iptm > best_iptm:
                best_iptm = run_best_iptm
                best_run_id = run_id
    return best_run_id


def find_best_cif(results_root, name):
    base_dir = Path(results_root) / name
    summary_csv = base_dir / "summary_all_runs.csv"
    if summary_csv.exists():
        run_id = parse_best_run_id(summary_csv)
        if run_id:
            candidate = base_dir / run_id / "best.cif"
            if candidate.exists():
                return str(candidate)
    run_dirs = sorted(base_dir.glob("run_*"))
    for run_dir in run_dirs:
        candidate = run_dir / "best.cif"
        if candidate.exists():
            return str(candidate)
    return None


def convert_best_cif_to_pdb(repo_root, cif_path):
    sys.path.insert(0, str(repo_root))
    from utils.convert import convert_cif_to_pdb

    cif_path = Path(cif_path).resolve()
    pdb_path = cif_path.with_suffix(".pdb")
    ok = convert_cif_to_pdb(str(cif_path), str(pdb_path))
    if not ok or not pdb_path.exists():
        raise RuntimeError(f"Failed to convert CIF to PDB: {cif_path}")
    return str(pdb_path)


def copy_results_if_needed(src_root, dst_root, name):
    if not dst_root:
        return
    src = src_root / name
    dst = Path(dst_root).resolve() / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return


def run_design(args, subdir, design_py, repo_root):
    name = subdir.name
    a3m_path = subdir / f"{name}.a3m"
    pdb_path = subdir / f"{name}.pdb"
    mask_path = subdir / f"{name}_mask.pdb"

    required = [a3m_path, pdb_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return None, None, f"missing files: {', '.join(missing)}"

    seq = read_first_a3m_seq(a3m_path)
    template_path = None
    if args.use_template:
        template_path = mask_path if args.use_mask_template and mask_path.exists() else pdb_path

    cmd = [
        sys.executable,
        str(design_py),
        "--jobname",
        name,
        "--seq",
        seq,
        "--n_trials",
        str(args.n_trials),
        "--n_cycles",
        str(args.n_cycles),
        "--n_recycles",
        str(args.n_recycles),
        "--n_diff_steps",
        str(args.n_diff_steps),
        "--gpu_id",
        str(args.gpu_id),
        "--hysteresis_mode",
        args.hysteresis_mode,
        "--min_protein_length",
        str(args.min_protein_length),
        "--max_protein_length",
        str(args.max_protein_length),
        "--temperature",
        str(args.temperature),
    ]

    if args.repredict:
        cmd.append("--repredict")
    if template_path:
        cmd.extend(["--binder_template_pdb", str(template_path)])
    if args.use_msa:
        cmd.extend(["--binder_msa", str(a3m_path)])
    if args.fix_all_if_no_x:
        cmd.append("--fix_all_if_no_x")
    if args.plot:
        cmd.append("--plot")

    if args.dry_run:
        print(" ".join(cmd))
        return None, None, "dry_run"

    subprocess.run(cmd, check=True, cwd=str(repo_root))
    chai_results_root = repo_root / "results_chai"
    copy_results_if_needed(chai_results_root, args.results_root, name)
    search_root = Path(args.results_root).resolve() if args.results_root else chai_results_root
    best_cif = find_best_cif(search_root, name)
    best_pdb = convert_best_cif_to_pdb(repo_root, best_cif) if best_cif else None
    return best_cif, best_pdb, None


def run_eval(args, design_pdb, native_pdb, a3m_path, eval_scaffold_py):
    eval_dir = Path(args.eval_dir).resolve()
    ensure_dir(eval_dir)
    out_json = eval_dir / f"{Path(native_pdb).stem}_eval.json"

    tmp_dir = Path("/data/wenhao/Protein-Hunter-server/tmp")
    try:
        ensure_dir(tmp_dir)
    except Exception:
        return None, "cannot create /data/wenhao/Protein-Hunter-server/tmp for eval_scaffold.py"

    cmd = [
        sys.executable,
        str(eval_scaffold_py),
        "--design",
        str(design_pdb),
        "--native",
        str(native_pdb),
        "--a3m",
        str(a3m_path),
        "--out",
        str(out_json),
        "--usalign_bin",
        str(args.usalign_bin),
    ]
    subprocess.run(cmd, check=True)
    return str(out_json), None


def run_batch(args):
    repo_root = Path(__file__).resolve().parents[1]
    design_py = repo_root / "chai_ph" / "design.py"
    eval_scaffold_py = repo_root / "DATA" / "eval_scaffold.py"
    input_root = Path(args.root).resolve()

    records = []
    for subdir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        name = subdir.name
        a3m_path = subdir / f"{name}.a3m"
        native_pdb = subdir / f"{name}.pdb"
        best_cif = None
        best_pdb = None
        eval_json = None
        design_error = ""
        eval_error = ""

        try:
            best_cif, best_pdb, design_error = run_design(args, subdir, design_py, repo_root)
        except Exception as e:
            design_error = str(e)

        if best_pdb and not args.skip_eval:
            try:
                eval_json, eval_error = run_eval(
                    args, best_pdb, native_pdb, a3m_path, eval_scaffold_py
                )
            except Exception as e:
                eval_error = str(e)

        record = {
            "pdb_id": name,
            "best_cif": best_cif or "",
            "design_pdb": best_pdb or "",
            "native_pdb": str(native_pdb) if native_pdb.exists() else "",
            "a3m": str(a3m_path) if a3m_path.exists() else "",
            "eval_json": eval_json or "",
            "design_error": design_error or "",
            "eval_error": eval_error or "",
        }

        if eval_json and os.path.exists(eval_json):
            with open(eval_json, "r") as f:
                data = json.load(f)
            metrics = data.get("metrics", {})
            record.update(
                {
                    "Scaffold_RMSD": metrics.get("Scaffold_RMSD"),
                    "Scaffold_TM_score": metrics.get("Scaffold_TM_score"),
                    "Scaffold_lDDT": metrics.get("Scaffold_lDDT"),
                }
            )

        records.append(record)

    eval_dir = Path(args.eval_dir).resolve()
    ensure_dir(eval_dir)
    summary_csv = eval_dir / "summary.csv"
    fieldnames = [
        "pdb_id",
        "best_cif",
        "design_pdb",
        "native_pdb",
        "a3m",
        "eval_json",
        "Scaffold_RMSD",
        "Scaffold_TM_score",
        "Scaffold_lDDT",
        "design_error",
        "eval_error",
    ]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"Batch finished. Summary: {summary_csv}")


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--root", default="/data/wenhao/Protein-Hunter-server/test_scaffolds")
    prepare_parser.add_argument("--pdb_id", default="1TAQ")
    prepare_parser.add_argument(
        "--source_pdb",
        default="/data/wenhao/Protein-Hunter-server/DATA/pdb/1TAQ_clean.pdb",
    )
    prepare_parser.add_argument(
        "--source_a3m",
        default="/data/wenhao/Protein-Hunter-server/MSA/1TAQ_full_length.a3m",
    )
    prepare_parser.add_argument(
        "--source_npz",
        default="/data/wenhao/Protein-Hunter-server/MSA/1TAQ_full_length.npz",
    )
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.add_argument("--make_fasta", action="store_true", default=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--root", required=True)
    run_parser.add_argument("--results_root", default="")
    run_parser.add_argument("--eval_dir", default="/data/wenhao/Protein-Hunter-server/results_eval_chai")
    run_parser.add_argument("--n_trials", default=1, type=int)
    run_parser.add_argument("--n_cycles", default=5, type=int)
    run_parser.add_argument("--n_recycles", default=3, type=int)
    run_parser.add_argument("--n_diff_steps", default=200, type=int)
    run_parser.add_argument("--gpu_id", default=0, type=int)
    run_parser.add_argument("--min_protein_length", default=90, type=int)
    run_parser.add_argument("--max_protein_length", default=150, type=int)
    run_parser.add_argument("--temperature", default=0.1, type=float)
    run_parser.add_argument("--hysteresis_mode", default="none", choices=["templates", "esm", "partial_diffusion", "none"], help="Hysteresis mode: esm (uses ESM embeddings), templates (uses templates), none (no hysteresis), partial_diffusion")
    run_parser.add_argument("--use_template", type=str2bool, default=True)
    run_parser.add_argument("--use_mask_template", type=str2bool, default=True)
    run_parser.add_argument("--use_msa", type=str2bool, default=True)
    run_parser.add_argument("--repredict", type=str2bool, default=True)
    run_parser.add_argument("--plot", type=str2bool, default=False)
    run_parser.add_argument("--fix_all_if_no_x", action="store_true", default=False)
    run_parser.add_argument("--skip_eval", action="store_true", default=False)
    run_parser.add_argument("--dry_run", action="store_true", default=False)
    run_parser.add_argument("--usalign_bin", default="/data/wenhao/USalign/USalign", help="Path to USalign binary")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_test_flow(args)
    elif args.command == "run":
        run_batch(args)


if __name__ == "__main__":
    main()
