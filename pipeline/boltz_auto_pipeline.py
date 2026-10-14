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
            if not line:
                continue
            if line.startswith(">"):
                continue
            return line
    raise ValueError(f"No sequence found in a3m: {a3m_path}")


def resolve_msa_paths(subdir, name, msa_n=None, msa_tag=None):
    """
    解析 MSA / 设计序列用的 a3m 与 npz。

    - 指定 --msa_tag：使用 {name}{tag}.a3m / .npz（如 _mpnn1000_t1.0）
    - 指定 --msa_n N：使用 {name}_mpnn{N}_top.a3m / .npz（不用 {name}.a3m）
    - 未指定：使用 {name}.a3m / {name}.npz
    """
    subdir = Path(subdir)
    explicit_tag = (msa_tag or "").strip()
    if explicit_tag:
        tag = explicit_tag if explicit_tag.startswith("_") else f"_{explicit_tag}"
        a3m_path = subdir / f"{name}{tag}.a3m"
        npz_path = subdir / f"{name}{tag}.npz"
        return a3m_path, npz_path, tag
    if msa_n is not None:
        n = int(msa_n)
        if n == 3000:
            for tag in ("_mpnn3000_top", "_mpnn3000_unique"):
                a3m_path = subdir / f"{name}{tag}.a3m"
                npz_path = subdir / f"{name}{tag}.npz"
                if a3m_path.exists() and npz_path.exists():
                    return a3m_path, npz_path, tag
            tag = "_mpnn3000_unique"
            return subdir / f"{name}{tag}.a3m", subdir / f"{name}{tag}.npz", tag
        if n == 5000:
            tag = "_mpnn5000_unique"
            return subdir / f"{name}{tag}.a3m", subdir / f"{name}{tag}.npz", tag
        tag = f"_mpnn{n}_top"
        a3m_path = subdir / f"{name}{tag}.a3m"
        npz_path = subdir / f"{name}{tag}.npz"
        return a3m_path, npz_path, tag
    return subdir / f"{name}.a3m", subdir / f"{name}.npz", ""


def effective_msa_max_seqs(args):
    """未显式传 --msa_max_seqs 时，与 --msa_n / --msa_tag 对齐。"""
    if getattr(args, "msa_max_seqs", None) is not None:
        return args.msa_max_seqs
    tag = (getattr(args, "msa_tag", None) or "").strip()
    if "mpnn1000" in tag:
        return 1000
    if getattr(args, "msa_n", None) is not None:
        return int(args.msa_n)
    return 500


def resolve_binder_template_path(args, subdir, name):
    """
    默认不使用模板。仅当 --use_template true 或显式 --binder_template_path 时返回路径。
    """
    explicit = (getattr(args, "binder_template_path", "") or "").strip()
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"binder_template_path not found: {p}")
        return p
    if not args.use_template:
        return None
    mask_path = subdir / f"{name}_mask.pdb"
    pdb_path = subdir / f"{name}.pdb"
    if args.use_mask_template and mask_path.exists():
        return mask_path
    if pdb_path.exists():
        return pdb_path
    raise FileNotFoundError(
        f"use_template=true but no template in {subdir} "
        f"(expected {name}_mask.pdb or {name}.pdb)"
    )


def iter_case_subdirs(input_root, ids=None):
    """遍历输入根目录下的案例子目录；可选 --ids 过滤。"""
    input_root = Path(input_root)
    if ids:
        for name in ids:
            subdir = input_root / name
            if subdir.is_dir():
                yield subdir
            else:
                print(f"⚠️ skip missing id dir: {subdir}")
        return
    for subdir in sorted(input_root.iterdir()):
        if subdir.is_dir():
            yield subdir


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
            iptm = row.get("best_iptm")
            if run_id is None or iptm is None:
                continue
            try:
                iptm_val = float(iptm)
            except ValueError:
                continue
            if best_iptm is None or iptm_val > best_iptm:
                best_iptm = iptm_val
                best_run_id = run_id
    return best_run_id


def find_all_structures(results_root, name):
    """查找所有 run_x 文件夹下的 best_structure.pdb 文件"""
    base_dir = Path(results_root) / name
    search_root = base_dir / "0_protein_hunter_design"
    structures = []
    
    if not search_root.exists():
        return structures
    
    for run_dir in sorted(search_root.iterdir()):
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            run_id = run_dir.name.replace("run_", "")
            best_pdb = run_dir / f"{name}_run_{run_id}_best_structure.pdb"
            if best_pdb.exists():
                structures.append((run_id, str(best_pdb)))
    
    return structures

def find_best_structure(results_root, name):
    """保留原函数以兼容旧代码，返回最佳结构"""
    structures = find_all_structures(results_root, name)
    if structures:
        return structures[0][1]
    return None


def run_design(args, subdir, design_py, results_root):
    name = subdir.name
    msa_n = getattr(args, "msa_n", None)
    msa_tag_arg = getattr(args, "msa_tag", None)
    a3m_path, npz_path, msa_tag = resolve_msa_paths(
        subdir, name, msa_n, msa_tag_arg
    )

    missing = [str(p) for p in [a3m_path, npz_path] if not p.exists()]
    if missing:
        hint = (
            f" (expected mpnn top MSA for --msa_n {msa_n})"
            if msa_n is not None
            else ""
        )
        return None, f"missing files: {', '.join(missing)}{hint}"

    try:
        template_path = resolve_binder_template_path(args, subdir, name)
    except FileNotFoundError as e:
        return None, str(e)

    seq = read_first_a3m_seq(a3m_path)

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
        "--msa_mode",
        "single",
        "--min_protein_length",
        str(args.min_protein_length),
        "--max_protein_length",
        str(args.max_protein_length),
    ]
    if args.target_template_path:
        cmd.extend(["--target_template_path", args.target_template_path])
    if getattr(args, "protein_seqs", "").strip():
        cmd.extend(["--protein_seqs", args.protein_seqs.strip()])
    if getattr(args, "ligand_ccd", "").strip():
        cmd.extend(["--ligand_ccd", args.ligand_ccd.strip()])
    if getattr(args, "ligand_smiles", "").strip():
        cmd.extend(["--ligand_smiles", args.ligand_smiles.strip()])
    if getattr(args, "ligand_contact_pairs", "").strip():
        cmd.extend(["--ligand_contact_pairs", args.ligand_contact_pairs.strip()])
    if hasattr(args, "ligand_contact_max_distance"):
        cmd.extend(
            ["--ligand_contact_max_distance", str(args.ligand_contact_max_distance)]
        )
    if getattr(args, "ligand_contact_force", False):
        cmd.append("--ligand_contact_force")
    if template_path:
        cmd.extend(["--binder_template_path", str(template_path)])
    if getattr(args, "template_cif_chain_id", "").strip():
        cmd.extend(["--template_cif_chain_id", args.template_cif_chain_id.strip()])
    if getattr(args, "target_template_cif_chain_id", "").strip():
        cmd.extend(
            ["--target_template_cif_chain_id", args.target_template_cif_chain_id.strip()]
        )
    if getattr(args, "binder_template_cif_chain_id", "").strip():
        cmd.extend(
            ["--binder_template_cif_chain_id", args.binder_template_cif_chain_id.strip()]
        )
    if getattr(args, "target_template_force", False):
        cmd.append("--target_template_force")
    if getattr(args, "binder_template_force", False):
        cmd.append("--binder_template_force")
    if hasattr(args, "target_template_force_threshold"):
        cmd.extend(
            [
                "--target_template_force_threshold",
                str(args.target_template_force_threshold),
            ]
        )
    if hasattr(args, "binder_template_force_threshold"):
        cmd.extend(
            [
                "--binder_template_force_threshold",
                str(args.binder_template_force_threshold),
            ]
        )
    if args.fix_all_if_no_x:

        cmd.append("--fix_all_if_no_x")

    # boltz_ph: --unconditional_msa_mode 仅用于 unconditional；与靶标 --msa_mode 独立
    if args.mode == "unconditional":
        cmd.extend(
            ["--unconditional_msa_mode", getattr(args, "unconditional_msa_mode", "empty")]
        )

    # 本地准备的 .npz 作为 --binder_msa：binder 模式照旧；unconditional 下 mmseqs 时不传（走在线搜索）
    if args.mode == "binder":
        if args.use_msa:
            cmd.extend(["--binder_msa", str(npz_path)])
    else:
        umode = getattr(args, "unconditional_msa_mode", "empty")
        if umode == "mmseqs":
            pass
        elif umode == "precomputed":
            if args.use_msa:
                cmd.extend(["--binder_msa", str(npz_path)])
        else:
            # empty：兼容旧行为 — use_msa 时仍传 npz，design 中等效 precomputed
            if args.use_msa:
                cmd.extend(["--binder_msa", str(npz_path)])

    if args.ccd_path:
        cmd.extend(["--ccd_path", str(args.ccd_path)])
    if args.boltz_model_path:
        cmd.extend(["--boltz_model_path", str(args.boltz_model_path)])
    if results_root:
        save_dir = str(Path(results_root) / name)
        cmd.extend(["--save_dir", save_dir])
    if args.plot:
        cmd.append("--plot")
    
    msa_max = effective_msa_max_seqs(args)
    cmd.extend(["--msa_max_seqs", str(msa_max)])

    if args.dry_run:
        print(" ".join(cmd))
        return None, "dry_run"

    try:
        subprocess.run(cmd, check=True)
        return find_best_structure(results_root or "/data/wenhao/Protein-Hunter-server/results_boltz", name), None
    except Exception as e:
        # Catch all exceptions and continue to next PDB
        error_msg = str(e)
        print(f"⚠️ Error for {name}: {error_msg}")
        return None, error_msg


def run_eval(args, design_pdb, native_pdb, a3m_path, eval_scaffold_py, run_id=None):
    eval_dir = Path(args.eval_dir).resolve()
    pdb_name = Path(native_pdb).stem
    
    pdb_eval_dir = eval_dir / pdb_name
    ensure_dir(pdb_eval_dir)
    
    if run_id is not None:
        out_json = pdb_eval_dir / f"{pdb_name}_run_{run_id}_eval.json"
    else:
        out_json = pdb_eval_dir / f"{pdb_name}_eval.json"

    tmp_dir = Path(__file__).resolve().parents[1] / "tmp"
    try:
        ensure_dir(tmp_dir)
    except Exception:
        return None, f"cannot create {tmp_dir} for eval_scaffold.py"

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
    try:
        subprocess.run(cmd, check=True)
        return str(out_json), None
    except Exception as e:
        return None, str(e)


def _process_one_case(
    args,
    subdir,
    name,
    msa_n,
    design_py,
    eval_scaffold_py,
    results_root,
    records,
):
    design_error = None
    a3m_path, _, _ = resolve_msa_paths(
        subdir, name, msa_n, getattr(args, "msa_tag", None)
    )
    native_pdb = subdir / f"{name}.pdb"

    try:
        design_pdb, design_error = run_design(args, subdir, design_py, results_root)
    except Exception as e:
        design_error = str(e)
        print(f"⚠️ Design error for {name}: {design_error}")

    if not args.skip_eval and not design_error:
        all_structures = find_all_structures(
            results_root or "/data/wenhao/Protein-Hunter-server/results_boltz", name
        )

        if not all_structures:
            record = {
                "pdb_id": name,
                "msa_n": msa_n if msa_n is not None else "",
                "run_id": "",
                "design_pdb": "",
                "native_pdb": str(native_pdb) if native_pdb.exists() else "",
                "a3m": str(a3m_path) if a3m_path.exists() else "",
                "eval_json": "",
                "Scaffold_RMSD": "",
                "Scaffold_TM_score": "",
                "Scaffold_lDDT": "",
                "design_error": "No structures found",
                "eval_error": "",
            }
            records.append(record)
        else:
            for run_id, design_pdb_path in all_structures:
                eval_json = None
                eval_error = None

                try:
                    eval_json, eval_error = run_eval(
                        args, design_pdb_path, native_pdb, a3m_path, eval_scaffold_py, run_id
                    )
                except Exception as e:
                    eval_error = str(e)
                    print(f"⚠️ Eval error for {name} run_{run_id}: {eval_error}")

                record = {
                    "pdb_id": name,
                    "msa_n": msa_n if msa_n is not None else "",
                    "run_id": run_id,
                    "design_pdb": design_pdb_path or "",
                    "native_pdb": str(native_pdb) if native_pdb.exists() else "",
                    "a3m": str(a3m_path) if a3m_path.exists() else "",
                    "eval_json": eval_json or "",
                    "design_error": "",
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
    else:
        record = {
            "pdb_id": name,
            "msa_n": msa_n if msa_n is not None else "",
            "run_id": "",
            "design_pdb": "",
            "native_pdb": str(native_pdb) if native_pdb.exists() else "",
            "a3m": str(a3m_path) if a3m_path.exists() else "",
            "eval_json": "",
            "Scaffold_RMSD": "",
            "Scaffold_TM_score": "",
            "Scaffold_lDDT": "",
            "design_error": design_error or "",
            "eval_error": "",
        }
        records.append(record)


def _write_batch_summary(args, records):
    eval_dir = Path(args.eval_dir).resolve()
    if getattr(args, "msa_n", None) is not None:
        eval_dir = eval_dir.parent / f"{eval_dir.name}_mpnn{args.msa_n}"
    ensure_dir(eval_dir)
    summary_csv = eval_dir / "summary.csv"
    fieldnames = [
        "pdb_id",
        "msa_n",
        "run_id",
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


def run_batch(args):
    repo_root = Path(__file__).resolve().parents[1]
    design_py = repo_root / "boltz_ph" / "design.py"
    eval_scaffold_py = repo_root / "DATA" / "eval_scaffold.py"
    input_root = Path(args.root).resolve()
    results_root = args.results_root

    ids = getattr(args, "ids", None) or None
    msa_n = getattr(args, "msa_n", None)
    records = []
    for subdir in iter_case_subdirs(input_root, ids):
        name = subdir.name
        try:
            _process_one_case(
                args,
                subdir,
                name,
                msa_n,
                design_py,
                eval_scaffold_py,
                results_root,
                records,
            )
        except Exception as e:
            print(f"⚠️ Unhandled error for {name}, continuing batch: {e}")
            records.append(
                {
                    "pdb_id": name,
                    "msa_n": msa_n if msa_n is not None else "",
                    "run_id": "",
                    "design_pdb": "",
                    "native_pdb": "",
                    "a3m": "",
                    "eval_json": "",
                    "Scaffold_RMSD": "",
                    "Scaffold_TM_score": "",
                    "Scaffold_lDDT": "",
                    "design_error": f"batch_exception: {e}",
                    "eval_error": "",
                }
            )

    _write_batch_summary(args, records)


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
    run_parser.add_argument(
        "--root",
        required=True,
        help="案例根目录，如 /data/wenhao/Protein-Hunter-server/test_scaffolds；"
        "其下每个子文件夹（7PV5/）为一例，无需再 prepare。",
    )
    run_parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="只跑指定 PDB ID（子目录名）；默认遍历 --root 下全部子目录。",
    )
    run_parser.add_argument(
        "--msa_n",
        type=int,
        default=None,
        help="MPNN top MSA 条数，对应 {id}_mpnn{N}_top.a3m/.npz（如 500 → 7PV5_mpnn500_top）。"
        "指定后不使用 {id}.a3m。未指定则用 {id}.a3m / {id}.npz。",
    )
    run_parser.add_argument(
        "--msa_tag",
        default=None,
        type=str,
        help="MSA 文件名后缀，如 _mpnn1000_t1.0 → {id}_mpnn1000_t1.0.a3m/.npz",
    )
    run_parser.add_argument("--results_root", default="/data/wenhao/Protein-Hunter-server/results_boltz")
    run_parser.add_argument("--eval_dir", default="/data/wenhao/Protein-Hunter-server/results_eval_boltz")
    run_parser.add_argument("--num_designs", default=1, type=int)
    run_parser.add_argument("--num_cycles", default=5, type=int)
    run_parser.add_argument("--gpu_id", default=0, type=int)
    run_parser.add_argument("--min_protein_length", default=90, type=int)
    run_parser.add_argument("--max_protein_length", default=150, type=int)
    run_parser.add_argument("--mode", default="unconditional", choices=["binder", "unconditional"], help="Design mode: binder (uses template strictly) or unconditional (more flexible)")
    run_parser.add_argument(
        "--use_template",
        type=str2bool,
        default=False,
        help="使用子目录内 {id}_mask.pdb 或 {id}.pdb 作为 binder 模板；默认关闭。",
    )
    run_parser.add_argument(
        "--binder_template_path",
        default="",
        type=str,
        help="显式指定 binder 模板 PDB；设置后一定会传入 design（无需 --use_template）。",
    )
    run_parser.add_argument("--use_mask_template", type=str2bool, default=True)
    run_parser.add_argument("--use_msa", type=str2bool, default=True)
    run_parser.add_argument(
        "--unconditional_msa_mode",
        default="empty",
        choices=["empty", "mmseqs", "precomputed"],
        help="仅当 --mode unconditional 时传给 boltz_ph design.py。"
        "empty+--use_msa：仍把各 PDB 目录下的 .npz 作为 binder MSA（与改前一致）。"
        "mmseqs：对当前设计序列走 ColabFold/MMseqs2，不再传入本地 npz。"
        "precomputed：显式使用本地 npz，需 --use_msa true。",
    )
    run_parser.add_argument(
        "--target_template_path", 
        default="", 
        type=str, 
        help="Target 链（受体）的初始结构 PDB 路径"
    )
    run_parser.add_argument(
        "--protein_seqs",
        default="",
        type=str,
        help="靶标蛋白序列（逻辑链 B）；与 design.py 一致。unconditional+单条 target 时用于 A+B 二聚体。",
    )
    run_parser.add_argument(
        "--ligand_ccd",
        default="",
        type=str,
        help="小分子靶标的 CCD ID（例如 SAM）。与 --ligand_smiles 二选一。",
    )
    run_parser.add_argument(
        "--ligand_smiles",
        default="",
        type=str,
        help="小分子靶标的 SMILES。与 --ligand_ccd 二选一。",
    )
    run_parser.add_argument(
        "--ligand_contact_pairs",
        default="",
        type=str,
        help='小分子接触约束，例如 "A:120-L:5,A:121-L:8"。',
    )
    run_parser.add_argument(
        "--ligand_contact_max_distance",
        default=4.5,
        type=float,
        help="小分子接触约束的最大距离（埃）。",
    )
    run_parser.add_argument(
        "--ligand_contact_force",
        action="store_true",
        default=False,
        help="是否对小分子接触约束启用 force。",
    )
    run_parser.add_argument(
        "--template_cif_chain_id",
        default="",
        type=str,
        help="靶标模板在结构文件中的源链 ID（与 target_template_path 对齐，可逗号/冒号分隔）",
    )
    run_parser.add_argument(
        "--target_template_cif_chain_id",
        default="",
        type=str,
        help="若设置则优先于 --template_cif_chain_id，用于靶标模板源链",
    )
    run_parser.add_argument(
        "--binder_template_cif_chain_id",
        default="",
        type=str,
        help="binder 模板在结构文件中的源链 ID（映射到逻辑链 A）",
    )
    run_parser.add_argument(
        "--target_template_force",
        action="store_true",
        default=False,
        help="靶标模板 Boltz force；与 fix_all_if_no_x 独立（二者任一均可触发靶标模板 force）",
    )
    run_parser.add_argument(
        "--target_template_force_threshold",
        default=0.0,
        type=float,
        help="靶标模板 force 模式下传给 Boltz 的阈值。",
    )
    run_parser.add_argument(
        "--binder_template_force",
        action="store_true",
        default=False,
        help="binder 模板 Boltz force",
    )
    run_parser.add_argument(
        "--binder_template_force_threshold",
        default=0.0,
        type=float,
        help="binder 模板 force 模式下传给 Boltz 的阈值。",
    )
    run_parser.add_argument(
        "--msa_max_seqs",
        default=None,
        type=int,
        help="动态 MSA 最多使用的序列数。默认与 --msa_n 相同，否则 500。",
    )
    run_parser.add_argument("--ccd_path", default="")
    run_parser.add_argument("--boltz_model_path", default="")
    run_parser.add_argument("--fix_all_if_no_x", action="store_true", default=False)
    run_parser.add_argument("--skip_eval", action="store_true", default=False)
    run_parser.add_argument("--dry_run", action="store_true", default=False)
    run_parser.add_argument("--plot", action="store_true", default=False, help="Plot cycles figs per run (requires matplotlib)")
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
