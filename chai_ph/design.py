# SPDX-License-Identifier: MIT
#
# Copyright (c) Protein-Hunter authors
#   (upstream: Protein-Hunter-main/chai_ph; project README: MIT License)
# Copyright (c) 2024 Justas Dauparas
#   (LigandMPNN components used by the pipeline; see LigandMPNN/LICENSE)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# -----------------------------------------------------------------------------
# 相对上游 Protein-Hunter 项目 chai_ph/design.py 的修改摘要：
# - 新增 CLI：--fix_all_if_no_x（无 X 时是否固定整条输入序列）。
# - 新增 MSA：--binder_msa、--binder_msa_mode；--target_msa、--target_msa_mode；
#   --msa_max_seqs。
# - 新增 binder 结构模板：--binder_template_pdb、--binder_template_chain。
# - 新增 MPNN：--mpnn_model_type（含 ligand_mpnn；上游版本无此参数）。
# -----------------------------------------------------------------------------

import os
import argparse
from pipeline import ProteinHunter_Chai


def parse_args():
    """Parses command-line arguments for ProteinHunter_Chai."""
    parser = argparse.ArgumentParser(
        description="ProteinHunter input options for structure prediction and design cycles."
    )

    # --- Job and Sequence Settings ---
    sequence_group = parser.add_argument_group("Sequence and Job Settings")
    sequence_group.add_argument(
        "--jobname", type=str, default="test", help="Job name for output files and folders."
    )
    sequence_group.add_argument(
        "--percent_X",
        type=int,
        default=80,
        help="Percentage of 'X' residues in the initial sequence (0, 50, or 100).",
    )
    sequence_group.add_argument(
        "--seq", type=str, default="", help="Input sequence for the binder chain (optional). If empty, sequence is randomly sampled."
    )
    sequence_group.add_argument(
        "--fix_all_if_no_x", action="store_true", default=False,
        help="如果没有 X，是否固定整个输入序列。默认 False。"
    )
    sequence_group.add_argument(
        "--min_protein_length", type=int, default=100, help="Minimum length of the designed protein chain."
    )
    sequence_group.add_argument(
        "--max_protein_length", type=int, default=150, help="Maximum length of the designed protein chain."
    )
    sequence_group.add_argument(
        "--binder_seq", default="", type=str, help="Initial sequence for the binder chain."
    )
    sequence_group.add_argument(
        "--binder_msa",
        type=str,
        default="",
        help="Optional MSA file for binder chain (A).",
    )
    sequence_group.add_argument(
        "--binder_msa_mode",
        type=str,
        default="precomputed",
        choices=["empty", "precomputed", "mmseqs"],
        help="Binder MSA source: empty|precomputed|mmseqs.",
    )
    sequence_group.add_argument(
        "--refiner_mode", action="store_true", default=False, help="Enable refiner mode."
    )
    sequence_group.add_argument(
        "--target_seq",
        type=str,
        default="",
        help="Target sequence (protein) or SMILES (ligand) for binder design (optional).",
    )
    sequence_group.add_argument(
        "--target_msa",
        type=str,
        default="",
        help="Optional MSA file for target protein chain (B).",
    )
    sequence_group.add_argument(
        "--target_msa_mode",
        type=str,
        default="empty",
        choices=["empty", "precomputed", "mmseqs"],
        help="Target MSA source (protein target only): empty|precomputed|mmseqs.",
    )
    sequence_group.add_argument(
        "--msa_max_seqs",
        type=int,
        default=500,
        help="Maximum number of sequences kept in auto mmseqs A3M.",
    )
    sequence_group.add_argument(
        "--target_pdb",
        type=str,
        default=None,
        help="Optional template PDB file (target_pdb) for design.",
    )
    sequence_group.add_argument(
        "--target_pdb_chain",
        type=str,
        default=None,
        help="Chain ID in template PDB (required if target_pdb provided).",
    )
    sequence_group.add_argument(
        "--binder_template_pdb",
        type=str,
        default=None,
        help="Optional template PDB/mmCIF file for binder chain (A).",
    )
    sequence_group.add_argument(
        "--binder_template_chain",
        type=str,
        default=None,
        help="Chain ID in binder template file (optional if only one chain).",
    )
    sequence_group.add_argument(
        "--cyclic",
        action="store_true",
        default=False,
        help="Enable cyclic topology for the designed chain.",
    )

    # --- Optimization Options ---
    opt_group = parser.add_argument_group("Optimization and Folding Parameters")
    opt_group.add_argument(
        "--n_trials",
        type=int,
        default=2,
        help="Number of independent optimization trials.",
    )
    opt_group.add_argument(
        "--n_cycles",
        type=int,
        default=8,
        help="Number of folding/design optimization cycles (steps).",
    )
    opt_group.add_argument(
        "--n_recycles",
        type=int,
        default=3,
        help="Number of trunk recycles per fold step.",
    )
    opt_group.add_argument(
        "--n_diff_steps",
        type=int,
        default=200,
        help="Diffusion steps for structure sampling.",
    )
    opt_group.add_argument(
        "--hysteresis_mode",
        type=str,
        default="esm",
        choices=["templates", "esm", "partial_diffusion", "none"],
        help="Strategy for template/feature reuse to guide folding (hysteresis).",
    )
    opt_group.add_argument(
        "--repredict",
        action="store_true",
        default=True,
        help="Re-predict final best structure without templates for validation.",
    )

    # --- MPNN Options ---
    mpnn_group = parser.add_argument_group("ProteinMPNN Settings")
    mpnn_group.add_argument(
        "--omit_aa", type=str, default="", help="Amino acid types to omit from design (e.g., 'C')."
    )
    mpnn_group.add_argument(
        "--mpnn_model_type",
        type=str,
        default="auto",
        choices=["auto", "protein_mpnn", "soluble_mpnn", "ligand_mpnn"],
        help="Select MPNN model type or use auto selection.",
    )

    # temp and bias params
    mpnn_group.add_argument("--temperature", default=0.1, type=float)
    mpnn_group.add_argument("--alanine_bias_start", default=-0.5, type=float)
    mpnn_group.add_argument("--alanine_bias_end", default=-0.1, type=float)
    mpnn_group.add_argument("--alanine_bias", action="store_true")

    mpnn_group.add_argument(
        "--scale_temp_by_plddt",
        action="store_true",
        default=False,
        help="Scale MPNN temperature inversely by pLDDT for focused design.",
    )

    # --- Visualization and Hardware Options ---
    vis_group = parser.add_argument_group("Hardware and Visualization")
    vis_group.add_argument(
        "--show_visual",
        action="store_true",
        default=False,
        help="Show interactive py3Dmol visualization during the run.",
    )
    vis_group.add_argument(
        "--render_freq",
        type=int,
        default=100,
        help="Visualization refresh frequency (diffusion steps).",
    )
    vis_group.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Plot cycles figs per run (requires matplotlib).",
    )
    vis_group.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use.")


    af_group = parser.add_argument_group("AlphaFold Settings")
    af_group.add_argument("--use_alphafold3_validation", action="store_true", default=False)
    af_group.add_argument(
        "--alphafold_dir", default=os.path.expanduser("~/alphafold3"), type=str
    )
    af_group.add_argument("--af3_docker_name", default="alphafold3_yc", type=str)
    af_group.add_argument(
        "--af3_database_settings", default="~/alphafold3/alphafold3_data_save", type=str
    )
    af_group.add_argument(
        "--hmmer_path",
        default="~/.conda/envs/alphafold3_venv",
        type=str,
    )
    af_group.add_argument("--use_msa_for_af3", action="store_true")
    af_group.add_argument("--work_dir", default="", type=str)
    af_group.add_argument("--high_iptm_threshold", default=0.8, type=float)
    af_group.add_argument("--high_plddt_threshold", default=0.8, type=float)
    return parser.parse_args()


def main():
    """Main function to run the ProteinHunter pipeline."""
    args = parse_args()
    protein_hunter = ProteinHunter_Chai(args)
    protein_hunter.run_pipeline()


if __name__ == "__main__":
    main()
