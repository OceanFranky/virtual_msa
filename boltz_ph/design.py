
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BOLTZ_SRC = os.path.join(os.path.dirname(__file__), "src")
for path in (BOLTZ_SRC, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Keep parse_args() here for CLI functionality
def parse_args():
    parser = argparse.ArgumentParser(
        description="Boltz protein design with cycle optimization"
    )
    # --- Existing Arguments (omitted for brevity, keep all original args) ---
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--grad_enabled", action="store_true", default=False)
    parser.add_argument("--name", default="target_name_is_missing", type=str)
    parser.add_argument(
        "--mode", default="binder", choices=["binder", "unconditional"], type=str
    )
    parser.add_argument("--num_designs", default=50, type=int)
    parser.add_argument("--num_cycles", default=5, type=int)
    parser.add_argument("--cyclic", action="store_true", default=False, help="Enable cyclic peptide design.")
    parser.add_argument("--min_protein_length", default=100, type=int)
    parser.add_argument("--max_protein_length", default=150, type=int)
    parser.add_argument("--seq", default="", type=str)
    parser.add_argument("--fix_all_if_no_x", action="store_true", default=False, 
                        help="如果没有 X，是否固定整个输入序列。默认 False（即全变化）。")
    parser.add_argument("--refiner_mode", action="store_true", default=False)
    parser.add_argument(
        "--protein_seqs",
        default="",
        type=str,
    )
    parser.add_argument("--msa_mode", default="mmseqs", choices=["single", "mmseqs"], type=str)

    parser.add_argument("--ligand_smiles", default="", type=str)
    parser.add_argument("--ligand_ccd", default="", type=str)
    parser.add_argument(
        "--ligand_specs",
        default="",
        type=str,
        help=(
            "Multiple ligands separated by semicolon. Format: "
            "CHAIN:ccd:CCD_ID or CHAIN:smiles:SMILES, e.g. "
            "'B:ccd:NAP;C:ccd:CPM'. Explicit chain IDs are used in "
            "--ligand_contact_pairs."
        ),
    )
    parser.add_argument(
        "--nucleic_type", default="dna", choices=["dna", "rna"], type=str
    )
    parser.add_argument("--nucleic_seq", default="", type=str)
    parser.add_argument(
        "--template_path", default="", type=str
    )  #pdb code or path(s) to .cif/.pdb, multiple allowed separated by colon or comma
    parser.add_argument(
        "--template_cif_chain_id",
        default="",
        type=str,
        help="靶标模板在 PDB/mmCIF 文件中的源链 ID；多条 target 时用逗号或冒号分隔，与 "
        "--target_template_path/--template_path 路径顺序一一对应；仅一条时可广播到所有靶标。",
    )
    parser.add_argument(
        "--binder_template_path", default="", type=str
    )
    parser.add_argument(
        "--target_template_path", 
        default="", 
        type=str, 
        help="Target 链（受体）的初始结构 PDB 路径"
    )
    parser.add_argument(
        "--binder_template_cif_chain_id",
        default="",
        type=str,
        help="binder 模板在 PDB/mmCIF 中的源链 ID（映射到逻辑链 A）。",
    )
    parser.add_argument(
        "--target_template_cif_chain_id",
        default="",
        type=str,
        help="与 --target_template_path 对齐的源链 ID；若设置则优先于 --template_cif_chain_id。",
    )
    parser.add_argument(
        "--target_template_force",
        action="store_true",
        default=False,
        help="对靶标模板施加 Boltz force（强约束）；与 --fix_all_if_no_x 独立，二者任一开启即 force。",
    )
    parser.add_argument(
        "--target_template_force_threshold",
        default=0.0,
        type=float,
        help="当 target template force=True 时传给 Boltz 的模板阈值。",
    )
    parser.add_argument(
        "--binder_template_force",
        action="store_true",
        default=False,
        help="对 binder 模板施加 Boltz force（强约束）。",
    )
    parser.add_argument(
        "--binder_template_force_threshold",
        default=0.0,
        type=float,
        help="当 binder template force=True 时传给 Boltz 的模板阈值。",
    )
    parser.add_argument("--diffuse_steps", default=200, type=int)
    parser.add_argument("--recycling_steps", default=3, type=int)
    parser.add_argument("--boltz_model_version", default="boltz2", type=str)
    parser.add_argument(
        "--boltz_model_path",
        default="/data/wenhao/.boltz/boltz2_conf.ckpt",
        type=str,
    )
    parser.add_argument("--ccd_path", default="/data/wenhao/.boltz/mols", type=str)
    parser.add_argument("--randomly_kill_helix_feature", action="store_true", default=False)
    parser.add_argument("--negative_helix_constant", default=0.2, type=float)
    parser.add_argument("--logmd", action="store_true", default=False)
    parser.add_argument("--save_dir", default="", type=str)
    parser.add_argument("--omit_AA", default="C", type=str)
    parser.add_argument(
        "--mpnn_model_type",
        default="auto",
        choices=["auto", "protein_mpnn", "soluble_mpnn", "ligand_mpnn"],
        type=str,
    )
    parser.add_argument("--exclude_P", action="store_true", default=False)
    parser.add_argument("--percent_X", default=90, type=int)
    parser.add_argument(
        "--binder_msa",
        default="",
        type=str,
        help="Optional path to a precomputed Boltz MSA .npz for chain A (binder/unconditional). "
        "With --mode unconditional --unconditional_msa_mode precomputed, required; "
        "with binder mode, enables dynamic template MSA when set.",
    )
    parser.add_argument(
        "--unconditional_msa_mode",
        default="empty",
        choices=["empty", "mmseqs", "precomputed"],
        type=str,
        help="仅用于 --mode unconditional：empty=无MSA；mmseqs=对当前设计序列跑 ColabFold/MMseqs2 搜索（与靶标的 --msa_mode 独立）；"
        "precomputed=使用 --binder_msa 的 .npz 做动态模板 MSA（与条件设计链一致）。"
        "若保持默认 empty 但仍提供 --binder_msa，则仍按 precomputed 行为兼容旧用法。",
    )
    parser.add_argument(
        "--msa_max_seqs",
        default=500,
        type=int,
        help="MSA 搜索时保留的最大序列数。默认 500。增大可获取更丰富的进化信息，但内存占用增加。",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot cycles figs per run (requires matplotlib)",
    )

    parser.add_argument(
        "--contact_residues", 
        default="", 
        type=str,
        help="Specify which target chain residues must contact the binder (currently only supports protein contacts). For more than two chains, separate by |, e.g., '1,2,5,10 | 3,5,10'."
    )
    parser.add_argument(
        "--ligand_contact_pairs",
        default="",
        type=str,
        help='Ligand contact constraints, e.g. "A:120-L:5,A:121-L:8". '
        "Each pair is token1-token2; token format CHAIN:RES_OR_ATOM. "
        "Use L as ligand-chain alias. For SMILES ligands, numeric token "
        "(e.g. L:5) will be mapped to canonical atom name automatically.",
    )
    parser.add_argument(
        "--ligand_pocket_constraints",
        default="",
        type=str,
        help=(
            "Ligand pocket constraints separated by semicolon. Format: "
            "LIGCHAIN:PROTCHAIN:RES,RES,...; e.g. 'B:A:36,56;C:A:126,136'."
        ),
    )
    parser.add_argument(
        "--ligand_contact_max_distance",
        default=4.5,
        type=float,
        help="Max distance (Angstrom) for ligand contact constraints.",
    )
    parser.add_argument(
        "--ligand_contact_force",
        action="store_true",
        default=False,
        help="Use force=True for ligand contact constraints.",
    )
    parser.add_argument(
        "--ligand_contact_weight",
        default=1.0,
        type=float,
        help="Weight multiplier for ligand contact constraints.",
    )
    parser.add_argument(
        "--ligand_pocket_weight",
        default=1.0,
        type=float,
        help="Weight multiplier for forced ligand pocket constraints.",
    )

    parser.add_argument(
        "--no_contact_filter",
        action="store_true",
        help="Do not filter or restart for unbound contact residues at cycle 0",
    )
    parser.add_argument("--max_contact_filter_retries", default=6, type=int)
    parser.add_argument("--contact_cutoff", default=15.0, type=float)

    parser.add_argument(
        "--alphafold_dir", default=os.path.expanduser("~/alphafold3"), type=str
    )
    parser.add_argument("--af3_docker_name", default="alphafold3_yc", type=str)
    parser.add_argument(
        "--af3_database_settings", default="~/alphafold3/alphafold3_data_save", type=str
    )
    parser.add_argument(
        "--hmmer_path",
        default="~/.conda/envs/alphafold3_venv",
        type=str,
    )
    parser.add_argument("--use_alphafold3_validation", action="store_true", default=False)
    parser.add_argument("--use_msa_for_af3", action="store_true")
    parser.add_argument("--work_dir", default="", type=str)

    # temp and bias params
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--alanine_bias_start", default=-0.5, type=float)
    parser.add_argument("--alanine_bias_end", default=-0.1, type=float)
    parser.add_argument("--alanine_bias", action="store_true")
    parser.add_argument("--high_iptm_threshold", default=0.8, type=float)
    parser.add_argument("--high_plddt_threshold", default=0.8, type=float)
    # --- End Existing Arguments ---

    return parser.parse_args()

def print_args(args):
    print("="*40)
    print("Design Configuration:")
    for k, v in vars(args).items():
        print(f"{k:30}: {v}")
    print("="*40)

def main():
    args = parse_args()
    # Pretty print each argument in a row for better visualization
    print_args(args)
    from pipeline import ProteinHunter_Boltz
    protein_hunter = ProteinHunter_Boltz(args)
    protein_hunter.run_pipeline()

if __name__ == "__main__":
    main()
