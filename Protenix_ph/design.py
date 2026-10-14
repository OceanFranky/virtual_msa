import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Protenix design CLI (Protein Hunter compatible arguments)"
    )
    parser.add_argument("--name", default="job", type=str)
    parser.add_argument("--save_dir", default="", type=str)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--num_designs", default=1, type=int)
    parser.add_argument("--num_cycles", default=5, type=int)
    parser.add_argument("--mode", default="unconditional", choices=["binder", "unconditional"], type=str)
    parser.add_argument("--cyclic", action="store_true", default=False)
    parser.add_argument("--fix_all_if_no_x", action="store_true", default=False)
    parser.add_argument("--seq", default="", type=str)
    parser.add_argument("--min_protein_length", default=100, type=int)
    parser.add_argument("--max_protein_length", default=150, type=int)
    parser.add_argument("--protein_seqs", default="", type=str)
    parser.add_argument(
        "--target_type",
        default="protein",
        choices=["protein", "ligand", "dna", "rna", "ion"],
        type=str,
        help="Target entity type for binder mode.",
    )
    parser.add_argument(
        "--target_seq",
        default="",
        type=str,
        help="Single target sequence for protein/dna/rna target.",
    )
    parser.add_argument(
        "--target_seqs",
        default="",
        type=str,
        help="Multiple target sequences split by comma or colon.",
    )
    parser.add_argument("--target_count", default=1, type=int)
    parser.add_argument(
        "--target_id",
        default="",
        type=str,
        help="Optional target chain IDs split by comma/colon.",
    )
    parser.add_argument("--ligand_ccd", default="", type=str)
    parser.add_argument("--ligand_smiles", default="", type=str)
    parser.add_argument("--ligand_file", default="", type=str)
    parser.add_argument("--template_path", default="", type=str)
    parser.add_argument("--template_cif_chain_id", default="", type=str)
    parser.add_argument(
        "--target_template_path",
        default="",
        type=str,
        help="Target template path(s), split by comma/colon for multi-target protein.",
    )
    parser.add_argument("--binder_template_path", default="", type=str)
    parser.add_argument("--binder_template_cif_chain_id", default="", type=str)
    parser.add_argument("--binder_msa", default="", type=str)
    parser.add_argument(
        "--target_msa",
        default="",
        type=str,
        help="Target precomputed MSA path for protein/rna targets.",
    )
    parser.add_argument(
        "--binder_msa_mode",
        default="precomputed",
        choices=["empty", "precomputed", "mmseqs"],
        type=str,
    )
    parser.add_argument(
        "--target_msa_mode",
        default="empty",
        choices=["empty", "precomputed", "mmseqs"],
        type=str,
    )
    parser.add_argument("--msa_max_seqs", default=500, type=int)
    parser.add_argument(
        "--model_name",
        default="protenix_base_default_v1.0.0",
        type=str,
        help="Protenix model name passed to get_default_runner.",
    )
    parser.add_argument(
        "--auto_constraint_model",
        action="store_true",
        default=False,
        help="Auto switch to protenix_base_constraint_v0.5.0 when constraints exist.",
    )
    parser.add_argument(
        "--constraint_json",
        default="",
        type=str,
        help="Path to a JSON file containing Protenix constraint object.",
    )
    parser.add_argument(
        "--constraint_inline_json",
        default="",
        type=str,
        help="Inline JSON string for Protenix constraint object.",
    )
    parser.add_argument(
        "--pocket_binder_entity",
        default="",
        type=str,
        help="Pocket binder entity spec: entity:copy (e.g. 2:1).",
    )
    parser.add_argument(
        "--pocket_contact_residues",
        default="",
        type=str,
        help="Pocket residues: entity:copy:position,... (e.g. 1:1:120,1:1:121).",
    )
    parser.add_argument("--pocket_max_distance", default=6.0, type=float)
    parser.add_argument(
        "--contact_constraints",
        default="",
        type=str,
        help=(
            "Contact constraints list split by comma. "
            "Format: e1:c1:p1[:a1]-e2:c2:p2[:a2]-max[:min]. "
            "Example: 1:1:169:CA-2:1:1:C5-6:0"
        ),
    )
    parser.add_argument(
        "--protein_hunter_root",
        default="",
        type=str,
        help="Protein-Hunter 根目录（可选）。为空时自动推断。",
    )
    parser.add_argument(
        "--ligand_mpnn_run_py",
        default="",
        type=str,
        help="LigandMPNN run.py 路径（可选）。为空时根据 protein_hunter_root 自动推断。",
    )
    parser.add_argument(
        "--mpnn_model_type",
        default="auto",
        choices=["auto", "protein_mpnn", "soluble_mpnn", "ligand_mpnn"],
        type=str,
    )
    parser.add_argument("--omit_AA", default="", type=str)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--alanine_bias_start", default=-0.5, type=float)
    parser.add_argument("--alanine_bias_end", default=-0.1, type=float)
    parser.add_argument("--alanine_bias", action="store_true")
    return parser.parse_args()


def print_args(args):
    print("=" * 40)
    print("Design Configuration:")
    for k, v in vars(args).items():
        print(f"{k:30}: {v}")
    print("=" * 40)


def main():
    args = parse_args()
    print_args(args)
    if not args.save_dir:
        args.save_dir = f"./results_protenix/{args.name}"

    # Run pipeline
    from pipeline import ProteinHunter_Protenix

    runner = ProteinHunter_Protenix(args)
    runner.run()


if __name__ == "__main__":
    main()
