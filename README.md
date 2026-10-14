# Protein-Hunter 😈

<p align="center">
  <img src="./protein_hunter.png" alt="Protein Hunter" width="500"/>
</p>

<p align="center" style="font-size:90%">
  <em>DAlphaBall.gcc
    <strong>Note:</strong> Logo is a ChatGPT-modified version of the Netflix animation and is for illustration only.
  </em>
</p>

> 📄 **Paper**: [Protein Hunter: exploiting structure hallucination
within diffusion for protein design](https://www.biorxiv.org/content/10.1101/2025.10.10.681530v2.full.pdf)
---


## 📝 Colab Notebook

[![Open Protein Hunter (Chai)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yehlincho/Protein-Hunter/blob/main/protein_hunter_chai_colab.ipynb)
[![Open Protein Hunter (Boltz)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yehlincho/Protein-Hunter/blob/main/protein_hunter_boltz_colab.ipynb)


## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yehlincho/Protein-Hunter.git
   cd Protein-Hunter
   ```

2. **Run the automated setup**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

> ⚠️ **Note**: AlphaFold3 setup is not included. Please install it separately by following the [official instructions](https://github.com/google-deepmind/alphafold3).

The setup script will automatically:
- ✅ Create a conda environment with Python 3.10
- ✅ Install all required dependencies
- ✅ Set up a Jupyter kernel for notebooks
- ✅ Download Boltz and Chai weights
- ✅ Configure LigandMPNN and ProteinMPNN
- ✅ Optionally install PyRosetta
- ❌ AF3 must be installed separately

---

We have implemented two different AF3-style models in our Protein Hunter pipeline (more models will be added in the future):
- Boltz1/2
- Chai1


## Run Code End-to-End

## 1️⃣ End-to-end structure and sequence generation  
👉 See example usage in `run_protein_hunter.py` for reference. 🐍✨  
This will take you from your initial input all the way to final designed protein structures and sequences, all in one pipeline!
The design chain is "A" and other target chains are "B", "C", etc.



> 💡 **Tips:** The original evaluation in the paper used an all-X sequence for initial design. However, to increase the diversity of generated folds, you can mix random amino acids with X residues by setting the `percent_X` parameter (e.g., `--percent_X 50` for 50% X and 50% random AAs). Adjusting this ratio helps explore a broader design space—but if you see too many floating or disconnected structures, try decreasing `percent_X` to encourage more structured designs.


⚠️ **Warning**: To run the AlphaFold3 cross-validation pipeline, you need to specify your AlphaFold3 directory, Docker name, database settings, and conda environment in the configuration. These can be set using the following arguments:
- `--alphafold_dir`: Path to your AlphaFold3 installation (default: ~/alphafold3)
- `--af3_docker_name`: Name of your AlphaFold3 Docker container
- `--af3_database_settings`: Path to AlphaFold3 database
- `--af3_hmmer_path`: Path to HMMER
- `--use_alphafold3_validation`: Add this flag to enable AlphaFold3-based validation. 

## Protein Hunter (Boltz Edition ⚡) 
To use AlphaFold3 validation, make sure your AlphaFold3 Docker is installed, specify the correct AlphaFold3 directory, and turn on `--use_alphafold3_validation`.


- **Protein-protein design with all X sequence:**  
  To design a protein-protein complex using an all-X sequence (i.e., X for every residue, encouraging de novo exploration), run:  
  ```
  python boltz_ph/design.py --num_designs 3 --num_cycles 7 --protein_seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE --msa_mode "mmseqs" --gpu_id 0 --name PDL1_mix_aa_all_X --percent_X 100 --min_protein_length 90 --max_protein_length 150 --high_iptm_threshold 0.7 --use_msa_for_af3 --plot
  ```
  > 💡 **Tip:** The `--percent_X 100` flag ensures all positions use the X (unknown) amino acid code.

- **Protein-protein design (mixed X and random amino acids):**  
  To design a protein-protein complex using a mix of X and random amino acids for the designable chain, run:  
  ```
  python boltz_ph/design.py --num_designs 3 --num_cycles 7 --protein_seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE --msa_mode "mmseqs" --gpu_id 0 --name PDL1_mix_aa --percent_X 50 --min_protein_length 90 --max_protein_length 150 --high_iptm_threshold 0.7 --use_msa_for_af3 --plot
  ```

  - **Protein-protein design (sample fewer alanine with alanine bias):**  
  To design a protein-protein complex using a mix of X and random amino acids while discouraging alanine sampling during design, run:
  ```
  python boltz_ph/design.py --num_designs 3 --num_cycles 7 --protein_seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE --msa_mode "mmseqs" --gpu_id 0 --name PDL1_mix_aa_alanine_bias --percent_X 50 --min_protein_length 90 --max_protein_length 150 --high_iptm_threshold 0.7 --alanine_bias --use_msa_for_af3 --plot
  ```

- **Protein-protein contact specification design:**  
  By default, contact potentials are disabled. To enable contact-based potentials and specify interface residue positions (e.g., residue positions "2,3,10" in the target chain), add `--no_potentials False` and `--contact_residues 2,3,10` to your command. For example:
  ```
  python boltz_ph/design.py --num_designs 3 --num_cycles 7 --protein_seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE --msa_mode "mmseqs" --gpu_id 0 --name PDL1_mix_aa --min_protein_length 90 --max_protein_length 150 --high_iptm_threshold 0.7 --use_msa_for_af3 --plot --no_potentials False --contact_residues 2,3,10
  ```
  
- **Multimer binder design:**  
  To design a binder for a multimeric protein (e.g., a dimer), separate chain sequences using `:`
  ```
  python boltz_ph/design.py --num_designs 3 --num_cycles 7 --protein_seqs AGIKVFGHPASIATRRVLIALHEKNLDFELVHVELKDGEHKKEPFLSRNPFGQVPAFEDGDLKLFESRAITQYIAHRYENQGTNLLQTDSKNISQYAIMAIGMQVEDHQFDPVASKLAFEQIFKSIYGLTTDEAVVAEEEAKLAKVLDVYEARLKEFKYLAGETFTLTDLHHIPAIQYLLGTPTKKLFTERPRVNEWVAEITKRPASEKVQ:AGIKVFGHPASIATRRVLIALHEKNLDFELVHVELKDGEHKKEPFLSRNPFGQVPAFEDGDLKLFESRAITQYIAHRYENQGTNLLQTDSKNISQYAIMAIGMQVEDHQFDPVASKLAFEQIFKSIYGLTTDEAVVAEEEAKLAKVLDVYEARLKEFKYLAGETFTLTDLHHIPAIQYLLGTPTKKLFTERPRVNEWVAEITKRPASEKVQ --msa_mode "mmseqs" --gpu_id 0 --name 1GNW_mix_aa --min_protein_length 90 --max_protein_length 150 --high_iptm_threshold 0.7 --use_msa_for_af3 --plot
  ```


- **Cyclic peptide binder design**
  ```
  python boltz_ph/design.py --num_designs 3 --num_cycles 7 --protein_seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE --msa_mode "mmseqs" --gpu_id 0 --name PDL1_cyclic_peptide_binder --min_protein_length 10 --max_protein_length 20 --high_iptm_threshold 0.8 --percent_X 100 --use_msa_for_af3 --plot --cyclic
  ```

- **Small molecule binder design:**  
  For designing a protein binder for a small molecule (e.g., SAM), use:  
  ```
  python boltz_ph/design.py --num_designs 5 --num_cycles 7 --ligand_ccd SAM --gpu_id 2 --name SAM_binder --min_protein_length 130 --max_protein_length 150 --high_iptm_threshold 0.7 --use_msa_for_af3 --plot
  ```

- **DNA/RNA PDB design:**  
  To design a protein binder for a nucleic acid (e.g., an RNA sequence), run:
  ```
  python boltz_ph/design.py --num_designs 5 --num_cycles 7 --nucleic_seq AGAGAGAGA --nucleic_type rna --gpu_id 0 --name RNA_bind --min_protein_length 130 --max_protein_length 150 --high_iptm_threshold 0.7 --use_msa_for_af3 --plot
  ```

- **Designs with multiple/heterogeneous target types:**  
  Want to target multiple types of molecules (e.g., a protein with a ligand and a template)? Run:
  ```
  python boltz_ph/design.py --num_designs 5 --num_cycles 7 --protein_seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE --msa_mode "mmseqs" --ligand_ccd SAM --gpu_id 0 --name PDL1_SAM --min_protein_length 90 --max_protein_length 150 --high_iptm_threshold 0.8 --use_msa_for_af3 --plot
  ```

### 本版本的 Boltz 改动说明（仅 Boltz，Chai 未改）

本仓库并非原版 Protein-Hunter，而是在原先版本的 Boltz 管线中对“设计链（binder chain，链 A）”增加了 MSA 与 template 相关能力，并引入了“无 X 时是否允许设计变动”的控制逻辑。以下内容是对代码改动的完整记录，便于后续维护与移植。

**改动摘要**
- 设计链支持 MSA 与 template（仅支持一条设计链 template）
- 设计链 MSA 支持动态更新（每个 cycle 生成新的 MSA）
- 新增参数控制：当输入序列无 X 时，是否固定全序列不参与设计

**新增与相关参数（Boltz CLI）**
- `--binder_msa`: 设计链 A 的 MSA（.npz）。若提供，将启用动态 MSA；同时用于提取模板序列（如果 MSA 内含模板序列）。见 [boltz_ph/design.py](file:///root/Protein-Hunter-server/boltz_ph/design.py#L83-L86)
- `--binder_template_path`: 设计链 A 的模板结构（pdb/cif 或 PDB id），只支持单条模板。见 [boltz_ph/design.py](file:///root/Protein-Hunter-server/boltz_ph/design.py#L55-L56)
- `--binder_template_cif_chain_id`: 设计链模板的链 ID。见 [boltz_ph/design.py](file:///root/Protein-Hunter-server/boltz_ph/design.py#L58-L59)
- `--fix_all_if_no_x`: 当输入序列中没有 X 时，是否将整个序列固定为“不可设计”。默认 False。见 [boltz_ph/design.py](file:///root/Protein-Hunter-server/boltz_ph/design.py#L32-L33)

**设计链 template 的注入逻辑**
- 设计链 A 的模板通过 `binder_template_path` 加入 templates 列表，链 ID 固定为 A；仅支持单条模板路径，不支持多模板列表。实现见 [boltz_ph/pipeline.py](file:///root/Protein-Hunter-server/boltz_ph/pipeline.py#L286-L297)

**设计链 MSA 的动态更新逻辑**
- 当传入 `--binder_msa` 时，会读取 .npz，提取其中模板序列作为“冻结模板序列”。见 [boltz_ph/pipeline.py](file:///root/Protein-Hunter-server/boltz_ph/pipeline.py#L370-L384)
- 每个 cycle 会基于当前设计链序列生成新的 A3M/NPZ，并写入 `binder_cycle_{cycle}.a3m/.npz`。见 [boltz_ph/pipeline.py](file:///root/Protein-Hunter-server/boltz_ph/pipeline.py#L386-L412)
- 生成的新 MSA 会立即注入到 data dict 的 binder chain（链 A），用于下一次预测。见 [boltz_ph/pipeline.py](file:///root/Protein-Hunter-server/boltz_ph/pipeline.py#L362-L368) 与 [boltz_ph/pipeline.py](file:///root/Protein-Hunter-server/boltz_ph/pipeline.py#L477-L625)
- 若未提供 `--binder_msa`，则不会动态更新，默认使用 "empty" MSA

**“无 X 是否固定序列”的控制逻辑**
- 设计链初始序列包含 X：仅固定非 X 的位点；设计时其余位点可被 MPNN 改动
- 设计链初始序列不含 X 且 `--fix_all_if_no_x` 为 True：固定全序列，不允许设计变动
- 实现位于固定残基计算与 MPNN 调用处： [boltz_ph/pipeline.py](file:///root/Protein-Hunter-server/boltz_ph/pipeline.py#L460-L613)

**设计链功能示例**
- 设计链模板（仅 1 条）+ 动态 MSA：
  ```
  python boltz_ph/design.py \
    --protein_seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE \
    --binder_template_path /path/to/binder_template.cif \
    --binder_template_cif_chain_id A \
    --binder_msa /path/to/binder_msa.npz \
    --num_designs 2 --num_cycles 5 --percent_X 50 --gpu_id 0 --name binder_with_msa_template
  ```
- 无 X 时固定全序列：
  ```
  python boltz_ph/design.py \
    --protein_seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE \
    --seq GPDRE...ERYIK \
    --fix_all_if_no_x \
    --num_designs 1 --num_cycles 3 --gpu_id 0 --name fixed_seq
  ```

## Protein Hunter (Chai Edition ☕) 

> ⚠️ **Caution:** The Chai version is under active development
- [x] Support for multiple targets
- [x] Full AlphaFold (AF3) validation

- **Unconditional protein design:**  
  Generate de novo proteins of a desired length:
  ```
  python chai_ph/design.py --jobname unconditional_design --min_protein_length 150 --max_protein_length 150 --percent_X 0 --seq "" --target_seq ACDEFGHIKLMNPQRSTVWY --n_trials 1 --n_cycles 5 --n_recycles 3 --n_diff_steps 200 --hysteresis_mode templates --repredict --omit_aa "" --temperature 0.1 --scale_temp_by_plddt --render_freq 100 --gpu_id 0 --plot

- **Protein binder design:**  
  To design a binder for a specific protein target (e.g., PDL1):
  ```
  python chai_ph/design.py --jobname PDL1_binder --min_protein_length 90 --max_protein_length 150 --percent_X 80 --seq "" --target_seq AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE --n_trials 1 --n_cycles 5 --n_recycles 3 --n_diff_steps 200 --hysteresis_mode templates --repredict --omit_aa "" --temperature 0.1 --scale_temp_by_plddt --render_freq 100 --gpu_id 0 --use_msa_for_af3 --plot
  ```

- **Cyclic peptide binder design:**  
  Design a cyclic peptide binder for a specific protein target:
  ```
  python chai_ph/design.py --jobname PDL1_cyclic_binder --percent_X 80 --seq "" --min_protein_length 10 --max_protein_length 20 --cyclic --target_seq AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE --n_trials 1 --n_cycles 5 --n_recycles 3 --n_diff_steps 200 --hysteresis_mode templates --repredict --omit_aa "" --temperature 0.1 --scale_temp_by_plddt --render_freq 100 --gpu_id 0 --use_msa_for_af3 --plot
  ```

- **Small molecule (ligand) binder design:**  
  To design a protein binder for a small molecule or ligand (SMILES string as target):
  ```
  python chai_ph/design.py --jobname ligand_binder  --percent_X 0 --seq "" --min_protein_length 130 --max_protein_length 150 --target_seq O=C(NCc1cocn1)c1cnn(C)c1C(=O)Nc1ccn2cc(nc2n1)c1ccccc1 --n_trials 1 --n_cycles 5 --n_recycles 3 --n_diff_steps 200 --hysteresis_mode esm --repredict --omit_aa "" --temperature 0.01 --scale_temp_by_plddt --render_freq 100 --gpu_id 2 --plot
  ```

## 2️⃣ Refine your own designs!
🛠️ You can provide your initial designs as input and further improve their structures by iteratively redesigning and predicting them. Repeat as needed for optimal results!

See the code in `refiner.ipynb` for example usage.

For example, you can generate a design using Boltzgen, take the final output, and refine it further using the iterative pipeline. 

---




## 🎥 Trajectory Visualization
We have implemented trajectory visualization using LogMD and py2Dmol (developed by Sergey Ovchinnikov).

## ✅ Structure Validation

### Primary Evaluation: AlphaFold3
Final structures are validated using **AlphaFold3** for:
- Structure quality assessment 
- Confidence scoring
- Cross-validation against design targets

## 🎯 Successful Designs

By default, the `high_iptm_yaml` folder contains only the designs with ipTM above your chosen threshold and alanine percentage below 20%. If you want to browse **all** generated designs, please check the metrics in `summary_all_runs.csv`.

After running the pipeline with `run_protein_hunter.py`, high-confidence designs can be found in:

- `your_output_folder/high_iptm_yaml`
- `your_output_folder/high_iptm_cif`
- Metrics summary: `your_output_folder/summary_high_iptm.csv`

After running AlphaFold3, the validated (successfully predicted) structures are saved in:

`your_output_folder/03_af_pdb_success`


## 📝 To-Do List

- [ ] Add cross-validation between Boltz and Chai (both directions) without using AlphaFold3 as an option
- [ ] Implement multi-timer support for Protein Hunter Chai edition
- [ ] Specify multiple contacts on multiple targets 
- [ ] Explore other cool applications

Collaboration is always welcome! Email me. Let's chat.

## 📄 License & Citation

**License**: MIT License - See LICENSE file for details  
**Citation**: If you use Protein Hunter in your research, please cite:
```
@article{cho2025protein,
  title={Protein Hunter: exploiting structure hallucination within diffusion for protein design},
  author={Cho, Yehlin and Rangel, Griffin and Bhardwaj, Gaurav and Ovchinnikov, Sergey},
  journal={bioRxiv},
  pages={2025--10},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
---

## 📧 Contact & Support

**Questions or Collaboration**: yehlin@mit.edu  
**Issues**: Please report bugs and feature requests via GitHub Issues

---

## ⚠️ Important Disclaimer

> **EXPERIMENTAL SOFTWARE**: This pipeline is under active development and has **NOT been experimentally validated** in laboratory settings. We release this code to enable community contributions and collaborative development. Use at your own discretion and validate results independently.
