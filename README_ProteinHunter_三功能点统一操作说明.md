# Protein-Hunter 三功能点统一操作说明（按固定组织流）

本文将 `Protein-Hunter-server` 中三个核心功能点统一整理为同一写法。  
每个功能点都严格采用以下组织流：

- （1）输入参数说明
- （2）运行命令说明
- （3）内部处理流程说明
- （4）输出结果说明

并重点体现你在管线中新增/增强的能力（模板、MSA、约束、固定残基等），同时补齐“输入文件、命令、输出结构”三要素。

---

## 1. 功能一：Unconditional 设计（使用 Chai 管线）

> 对应脚本：`/data/wenhao/Protein-Hunter-server/chai_ph/design.py`  
> 说明：Chai 管线无显式 `mode` 参数；当不提供有效 target 时，按单链 unconditional 路径运行。

### （1）输入参数说明

输入文件与输入项：

- 输入项：`input-seq`
  - 说明：设计链初始序列（可包含 `X`）
  - 对应参数：`--seq`
- 输入项：`binder-msa`（可选）
  - 说明：设计链外部 MSA 文件（`a3m/fa/fasta/parquet`）
  - 对应参数：`--binder_msa` + `--binder_msa_mode precomputed`
- 输入项：`binder-template`（可选）
  - 说明：设计链模板结构文件（`pdb/mmcif`）
  - 对应参数：`--binder_template_pdb` + `--binder_template_chain`
- 输入项：`work-root`
  - 说明：运行输出根目录（默认写入 `results_chai`）
  - 对应参数：`--jobname`

重点增强参数（建议）：

- `--fix_all_if_no_x`
- `--binder_msa_mode empty|precomputed|mmseqs`
- `--binder_msa`
- `--msa_max_seqs`
- `--binder_template_pdb`
- `--binder_template_chain`
- `--mpnn_model_type`

### （2）运行命令说明

```bash
python /data/wenhao/Protein-Hunter-server/chai_ph/design.py \
  --jobname chai_uncond_demo \
  --seq "MXXKLVXGAXXXLQAA" \
  --fix_all_if_no_x \
  --binder_msa_mode precomputed \
  --binder_msa /data/wenhao/example_data/chai/binder_A.a3m \
  --msa_max_seqs 500 \
  --binder_template_pdb /data/wenhao/example_data/chai/binder_template.pdb \
  --binder_template_chain A \
  --n_trials 2 \
  --n_cycles 6 \
  --n_diff_steps 200 \
  --n_recycles 3 \
  --mpnn_model_type auto \
  --temperature 0.1 \
  --plot
```

### （3）内部处理流程说明

- 输入解析与合法性检查  
  校验 `msa_max_seqs>0`，并根据 `binder_msa_mode` 校验 `binder_msa`。
- MSA 处理（增强点）  
  `precomputed` 读取外部 MSA；`mmseqs` 自动检索生成 A3M；`empty` 不加载 MSA。
- 模板注入（增强点）  
  将 `binder_template_pdb/binder_template_chain` 注入折叠流程；单链模板可自动识别链。
- 循环优化  
  结构预测 -> MPNN 设计 -> 下一轮输入更新，直到达到 `n_cycles`。
- 结果筛选  
  记录每轮指标并输出最优候选。

### （4）输出结果说明

输出目录结构（示例）：

```text
results_chai/
└── {jobname}/
    ├── auto_msa/                          # mmseqs 模式下生成
    │   ├── trial_0_binder_A_mmseqs.a3m
    │   └── ...
    ├── trial_0/                           # 每个 trial 的循环中间产物
    │   ├── cycle_0_*
    │   ├── cycle_1_*
    │   └── ...
    ├── trial_1/
    ├── summary_all_trials.csv             # 若开启汇总导出
    ├── best_result.*                      # 若开启最优导出
    └── cycles_plot.png                    # --plot 时生成
```

关键输出文件说明：

- `auto_msa/*.a3m`：自动 MSA 文件（`binder_msa_mode=mmseqs`）
- `trial_*/cycle_*`：各轮预测与设计中间结果
- `summary_all_trials.csv`：多 trial 汇总
- `cycles_plot.png`：指标可视化图

---

## 2. 功能二：Binder-Target 设计（Target 为蛋白，使用 Boltz 管线）

> 对应脚本：`/data/wenhao/Protein-Hunter-server/boltz_ph/design.py`  
> 说明：本节重点体现你在 Boltz 管线新增的模板/MSA/force/固定残基能力。

### （1）输入参数说明

输入文件与输入项：

- 输入项：`binder-seq`
  - 说明：设计链序列（可含 `X`）
  - 对应参数：`--seq`
- 输入项：`target-seq`
  - 说明：target 蛋白序列（可多链）
  - 对应参数：`--protein_seqs`
- 输入项：`binder-template`（可选）
  - 说明：设计链模板文件
  - 对应参数：`--binder_template_path` + `--binder_template_cif_chain_id`
- 输入项：`target-template`（可选）
  - 说明：target 模板文件
  - 对应参数：`--target_template_path` + `--target_template_cif_chain_id`
- 输入项：`binder-msa`（可选）
  - 说明：预计算 MSA（`npz`）
  - 对应参数：`--binder_msa`
- 输入项：`work-root`
  - 说明：运行输出目录
  - 对应参数：`--save_dir`（或默认 `results_boltz/{name}`）

重点增强参数（建议）：

- `--fix_all_if_no_x`
- `--binder_template_force` / `--binder_template_force_threshold`
- `--target_template_force` / `--target_template_force_threshold`
- `--binder_msa`
- `--msa_max_seqs`
- `--mpnn_model_type`

### （2）运行命令说明

```bash
python /data/wenhao/Protein-Hunter-server/boltz_ph/design.py \
  --name boltz_binder_protein_demo \
  --mode binder \
  --num_designs 3 \
  --num_cycles 5 \
  --seq "XXXAKLXXGXXXXLAA" \
  --fix_all_if_no_x \
  --protein_seqs "MKTIIALSYIFCLVFADYKDDDDK" \
  --binder_template_path /data/wenhao/example_data/boltz/binder_template.pdb \
  --binder_template_cif_chain_id A \
  --binder_template_force \
  --binder_template_force_threshold 0.0 \
  --target_template_path /data/wenhao/example_data/boltz/target_template.pdb \
  --target_template_cif_chain_id B \
  --target_template_force \
  --target_template_force_threshold 0.0 \
  --binder_msa /data/wenhao/example_data/boltz/binder_precomputed.npz \
  --msa_max_seqs 500 \
  --mpnn_model_type auto \
  --temperature 0.1 \
  --omit_AA C \
  --gpu_id 0
```

### （3）内部处理流程说明

- 参数校验与模式解析  
  检查模板文件、MSA 文件、模式组合是否合法。
- 模板装配与链映射（增强点）  
  解析 binder/target 模板路径、链 ID、force 与 threshold。
- 输入数据字典构建  
  将 target、模板、约束合并为 Boltz 推理输入。
- MSA 注入与循环预测（增强点）  
  结合 `binder_msa` 和 `msa_max_seqs` 执行每轮结构预测与设计。
- 固定位点控制（增强点）  
  按 `X` 掩码与 `fix_all_if_no_x` 构造固定残基并传入 MPNN。

### （4）输出结果说明

输出目录结构（示例）：

```text
results_boltz/
└── {name}/
    ├── run_0/
    │   ├── {name}_run_0_predicted_cycle_0.pdb
    │   ├── {name}_run_0_predicted_cycle_1.pdb
    │   ├── ...
    │   └── metrics_run_0.csv
    ├── run_1/
    ├── MSA/
    │   ├── {name}_cycle_0.a3m|npz
    │   └── ...
    ├── summary_all_runs.csv
    ├── best/
    │   ├── best_structure.pdb
    │   ├── best_sequence.txt
    │   └── best_metrics.json
    └── af3_validation/                    # 开启 AF3 验证时生成
```

关键输出文件说明：

- `run_*/...cycle_*.pdb`：每轮预测结构
- `MSA/*`：MSA 注入链路生成或复用产物
- `summary_all_runs.csv`：全 run 汇总
- `best/*`：最优候选结构、序列、评分

---

## 3. 功能三：Binder-Target 设计（Target 为小分子，使用 Protenix 管线）

> 对应脚本：`/data/wenhao/Protein-Hunter-server/Protenix_ph/design.py`  
> 说明：使用 Protenix 管线中与小分子场景强相关、且可闭环执行的增强能力。

### （1）输入参数说明

输入文件与输入项：

- 输入项：`binder-seq`
  - 说明：设计链初始序列
  - 对应参数：`--seq`
- 输入项：`ligand-input`（三选一）
  - `--ligand_ccd`：CCD 名称
  - `--ligand_file`：配体结构文件
  - `--ligand_smiles`：SMILES 字符串
- 输入项：`binder-msa`（可选）
  - 说明：预计算 MSA（`a3m`）
  - 对应参数：`--binder_msa_mode precomputed` + `--binder_msa`
- 输入项：`constraint`（可选）
  - 说明：约束 JSON 文件或命令行约束
  - 对应参数：`--constraint_json` / `--constraint_inline_json` / `--contact_constraints`
- 输入项：`work-root`
  - 说明：输出目录
  - 对应参数：`--save_dir`

重点增强参数（建议）：

- `--target_type ligand`
- `--auto_constraint_model`
- `--contact_constraints`
- `--binder_msa_mode empty|precomputed|mmseqs`
- `--binder_template_path`
- `--mpnn_model_type ligand_mpnn`

### （2）运行命令说明

```bash
python /data/wenhao/Protein-Hunter-server/Protenix_ph/design.py \
  --name protenix_ligand_demo \
  --mode binder \
  --save_dir /data/wenhao/example_output/protenix_ligand_demo \
  --num_designs 2 \
  --num_cycles 5 \
  --seq "XXAKLXXXGXXXLAAXX" \
  --fix_all_if_no_x \
  --target_type ligand \
  --ligand_smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --binder_msa_mode precomputed \
  --binder_msa /data/wenhao/example_data/protenix/binder_precomputed.a3m \
  --msa_max_seqs 500 \
  --binder_template_path /data/wenhao/example_data/protenix/binder_template.pdb \
  --contact_constraints "1:1:120:CA-2:1:1:C1-6:0,1:1:121:CB-2:1:1:O2-6:0" \
  --auto_constraint_model \
  --mpnn_model_type ligand_mpnn \
  --temperature 0.1 \
  --omit_AA C
```

### （3）内部处理流程说明

- 小分子 target 实体构建  
  按 `ligand_ccd > ligand_file > ligand_smiles` 优先级解析 target。
- 约束对象构建（增强点）  
  从 JSON 或命令行构建 pocket/contact 约束，并可触发自动切换约束模型。
- MSA 与模板解析（增强点）  
  解析 binder 的 `empty/precomputed/mmseqs` MSA 策略，并按需注入模板。
- 循环式预测与序列设计  
  每轮执行：构建 input JSON -> Protenix 推理 -> 读取置信度 -> MPNN 设计下一轮。
- 最优候选筛选  
  binder 模式优先按 iPTM 筛选全局最优。

### （4）输出结果说明

输出目录结构（示例）：

```text
{save_dir}/
├── run_0/
│   ├── input_cycle_0.json
│   ├── input_cycle_1.json
│   ├── ...
│   ├── {name}_run_0_cycle_0/seed_42/predictions/*.cif
│   └── {name}_run_0_cycle_0/seed_42/predictions/*_summary_confidence_sample_*.json
├── run_1/
├── summary_all_runs.csv
├── metrics_plot.png
└── best/
    ├── best.cif
    ├── best_confidence.json
    ├── best_input.json
    └── summary_best.csv
```

关键输出文件说明：

- `run_*/input_cycle_*.json`：每轮输入快照
- `run_*/.../predictions/*.cif`：每轮结构结果
- `run_*/.../*_summary_confidence*.json`：每轮评分来源
- `summary_all_runs.csv`：全部 run/cycle 指标总表
- `best/*`：全局最优结果集合

---

## 4. 统一写作模板（后续扩展复用）

后续新增功能点，继续使用以下骨架：

1. 功能名（指定管线）
   - （1）输入参数说明（含输入文件/输入项）
   - （2）运行命令说明（可直接复制）
   - （3）内部处理流程说明（步骤化）
   - （4）输出结果说明（目录树 + 关键文件说明）

