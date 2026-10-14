import copy
import os
import random
import sys
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BOLTZ_SRC = os.path.join(os.path.dirname(__file__), "src")
for path in (BOLTZ_SRC, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
from LigandMPNN.wrapper import LigandMPNNWrapper

from boltz.data import const
from boltz_ph.constants import CHAIN_TO_NUMBER
from boltz.data.parse.a3m import parse_a3m
from utils.metrics import get_CA_and_sequence # Used implicitly in design.py
from utils.convert import calculate_holo_apo_rmsd, convert_cif_files_to_pdb


from model_utils import (
    binder_binds_contacts,
    clean_memory,
    design_sequence,
    get_boltz_model,
    get_cif,
    load_canonicals,
    plot_from_pdb,
    plot_run_metrics,
    process_msa,
    run_prediction,
    sample_seq,
    save_pdb,
    shallow_copy_tensor_dict,
    smart_split,
    unconditional_mmseqs_npz_for_cycle,
)


def effective_unconditional_msa_mode(args) -> str:
    """
    Effective unconditional MSA source. Legacy: --unconditional_msa_mode empty plus a
    non-empty --binder_msa still selects precomputed template MSA.
    """
    u = getattr(args, "unconditional_msa_mode", "empty")
    if args.mode != "unconditional":
        return "empty"
    if u == "empty":
        bm = (getattr(args, "binder_msa", None) or "").strip()
        if bm and bm != "empty":
            return "precomputed"
    return u


def validate_pipeline_args(args) -> None:
    """Fail fast on invalid MSA / mode combinations before loading large models."""
    u = getattr(args, "unconditional_msa_mode", "empty")
    if u != "empty" and args.mode != "unconditional":
        raise ValueError(
            f"--unconditional_msa_mode={u} 只能与 --mode unconditional 一起使用"
        )
    eff = effective_unconditional_msa_mode(args)
    if eff == "precomputed":
        raw = (getattr(args, "binder_msa", None) or "").strip()
        if not raw or raw == "empty":
            raise ValueError(
                "无条件模式 MSA 为 precomputed 时，必须提供 --binder_msa 指向 Boltz .npz"
            )
        bm = os.path.abspath(os.path.expanduser(raw))
        if not os.path.isfile(bm):
            raise FileNotFoundError(f"binder_msa 文件不存在: {bm}")
        if not bm.endswith(".npz"):
            raise ValueError(f"binder_msa 须为 .npz 文件: {bm}")


class InputDataBuilder:
    """Handles parsing command-line arguments and constructing the base Boltz input data dictionary."""

    def __init__(self, args):
        self.args = args
        self.save_dir = (
            args.save_dir if args.save_dir else f"./results_boltz/{args.name}"
        )
        self.protein_hunter_save_dir = f"{self.save_dir}/0_protein_hunter_design"
        os.makedirs(self.protein_hunter_save_dir, exist_ok=True)

    def _smiles_atom_index_to_name(self, smiles: str) -> dict[int, str]:
        """
        Mirror Boltz schema.py naming for SMILES ligands:
        atom_name = element_symbol + (canonical_rank + 1), on AddHs mol.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid ligand_smiles; RDKit failed to parse.")
        mol = AllChem.AddHs(mol)
        canonical_order = AllChem.CanonicalRankAtoms(mol)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        idx_to_name = {}
        for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
            atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
            idx_to_name[int(can_idx + 1)] = atom_name
            atom.SetProp("name", atom_name)
        # Use 1-based atom order index (what users usually mean by token index),
        # then map to Boltz atom-name token expected by schema parser.
        order_map = {}
        for i, atom in enumerate(mol.GetAtoms(), start=1):
            order_map[i] = atom.GetProp("name")
        return order_map

    def _parse_ligand_specs(self, default_ligand_chain_id: str | None):
        """
        Parse multi-ligand CLI specs into Boltz sequence entries.
        Spec format: CHAIN:ccd:CCD_ID or CHAIN:smiles:SMILES; entries separated by ';'.
        Falls back to the legacy single-ligand arguments when --ligand_specs is empty.
        """
        raw = (getattr(self.args, "ligand_specs", "") or "").strip()
        if not raw:
            if self.args.ligand_smiles:
                return [
                    {
                        "chain_id": default_ligand_chain_id,
                        "kind": "smiles",
                        "value": self.args.ligand_smiles,
                    }
                ]
            if self.args.ligand_ccd:
                return [
                    {
                        "chain_id": default_ligand_chain_id,
                        "kind": "ccd",
                        "value": self.args.ligand_ccd,
                    }
                ]
            return []

        if self.args.ligand_smiles or self.args.ligand_ccd:
            raise ValueError(
                "--ligand_specs cannot be combined with --ligand_smiles/--ligand_ccd."
            )

        specs = []
        seen = set()
        for entry in [part.strip() for part in raw.split(";") if part.strip()]:
            parts = entry.split(":", 2)
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid ligand spec '{entry}'. Use CHAIN:ccd:CCD_ID or CHAIN:smiles:SMILES."
                )
            chain_id, kind, value = [part.strip() for part in parts]
            kind = kind.lower()
            if not chain_id or not value or kind not in {"ccd", "smiles"}:
                raise ValueError(
                    f"Invalid ligand spec '{entry}'. Kind must be ccd or smiles."
                )
            if chain_id in seen:
                raise ValueError(f"Duplicate ligand chain in --ligand_specs: {chain_id}")
            seen.add(chain_id)
            specs.append({"chain_id": chain_id, "kind": kind, "value": value})
        return specs

    def _parse_ligand_contact_constraints(self, ligand_chain_ids):
        """
        Parse --ligand_contact_pairs into Boltz 'contact' constraints.
        Pair format: CHAIN:RES_OR_ATOM-CHAIN:RES_OR_ATOM, e.g. A:120-L:5
        where 'L' is an alias for the ligand chain id if exactly one ligand is present.
        """
        raw = (getattr(self.args, "ligand_contact_pairs", "") or "").strip()
        if not raw:
            return []
        ligand_chain_ids = list(ligand_chain_ids or [])
        if not ligand_chain_ids:
            raise ValueError(
                "--ligand_contact_pairs is set but no ligand is provided "
                "(--ligand_specs/--ligand_smiles/--ligand_ccd)."
            )
        ligand_chain_set = set(ligand_chain_ids)
        single_ligand_chain_id = ligand_chain_ids[0] if len(ligand_chain_ids) == 1 else None

        smiles = (getattr(self.args, "ligand_smiles", "") or "").strip()
        atom_idx_to_name = (
            self._smiles_atom_index_to_name(smiles)
            if smiles and single_ligand_chain_id
            else {}
        )

        def parse_token(token: str):
            t = token.strip()
            parts = [part.strip() for part in t.split(":")]
            if len(parts) < 2 or len(parts) > 3:
                raise ValueError(
                    f"Invalid token '{t}' in --ligand_contact_pairs. Use CHAIN:VALUE or CHAIN:RES:ATOM."
                )
            chain = parts[0]
            value_parts = parts[1:]
            if not chain or any(not value for value in value_parts):
                raise ValueError(
                    f"Invalid token '{t}' in --ligand_contact_pairs. Empty chain/value."
                )
            if chain == "L":
                if len(ligand_chain_ids) != 1:
                    raise ValueError(
                        "The L ligand alias is ambiguous with multiple ligands; use "
                        "explicit ligand chain IDs in --ligand_contact_pairs."
                    )
                chain = single_ligand_chain_id
            # If this token is the ligand and user passed numeric atom index, map it for SMILES.
            if len(value_parts) == 1 and chain in ligand_chain_set and value_parts[0].isdigit():
                atom_index = int(value_parts[0])
                if not atom_idx_to_name:
                    raise ValueError(
                        "Numeric ligand token was provided but ligand is not SMILES-based. "
                        "Please provide atom name directly (e.g., B:C12)."
                    )
                if atom_index not in atom_idx_to_name:
                    raise ValueError(
                        f"Ligand atom index {atom_index} not found in SMILES canonical map."
                    )
                value = atom_idx_to_name[atom_index]
            else:
                if len(value_parts) == 2:
                    if chain in ligand_chain_set:
                        raise ValueError(
                            f"Invalid ligand token '{t}'. Ligands only support CHAIN:ATOM."
                        )
                    residue_raw, atom_name = value_parts
                    if not residue_raw.isdigit():
                        raise ValueError(
                            f"Invalid polymer atom token '{t}'. Residue index must be an integer."
                        )
                    value = [int(residue_raw), atom_name]
                else:
                    value = value_parts[0]
                    # Polymer side is usually residue index.
                    if chain not in ligand_chain_set and value.isdigit():
                        value = int(value)
            return [chain, value]

        constraints = []
        for pair in [p.strip() for p in raw.split(",") if p.strip()]:
            if "-" not in pair:
                raise ValueError(
                    f"Invalid pair '{pair}' in --ligand_contact_pairs. Use token1-token2."
                )
            left, right = pair.split("-", 1)
            token1 = parse_token(left)
            token2 = parse_token(right)
            constraints.append(
                {
                    "contact": {
                        "token1": token1,
                        "token2": token2,
                        "max_distance": float(
                            getattr(self.args, "ligand_contact_max_distance", 4.5)
                        ),
                        "weight": float(
                            getattr(self.args, "ligand_contact_weight", 1.0)
                        ),
                        "force": bool(
                            getattr(self.args, "ligand_contact_force", False)
                        ),
                    }
                }
            )
        return constraints

    def _parse_ligand_pocket_constraints(self, ligand_chain_ids):
        """
        Parse ligand-level pocket constraints.
        Format: LIGCHAIN:PROTCHAIN:RES,RES,...;LIGCHAIN:PROTCHAIN:RES,...
        """
        raw = (getattr(self.args, "ligand_pocket_constraints", "") or "").strip()
        if not raw:
            return []
        ligand_chain_set = set(ligand_chain_ids or [])
        if not ligand_chain_set:
            raise ValueError(
                "--ligand_pocket_constraints is set but no ligand is provided."
            )

        constraints = []
        for block in [part.strip() for part in raw.split(";") if part.strip()]:
            parts = block.split(":", 2)
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid ligand pocket block '{block}'. Use LIGCHAIN:PROTCHAIN:RES,RES."
                )
            ligand_chain, protein_chain, residues_raw = [
                part.strip() for part in parts
            ]
            if ligand_chain not in ligand_chain_set:
                raise ValueError(
                    f"Ligand pocket binder '{ligand_chain}' is not a ligand chain."
                )
            residues = [
                int(res.strip())
                for res in residues_raw.split(",")
                if res.strip()
            ]
            if not residues:
                raise ValueError(f"No pocket residues in '{block}'.")
            constraints.append(
                {
                    "pocket": {
                        "binder": ligand_chain,
                        "contacts": [[protein_chain, res] for res in residues],
                        "max_distance": float(
                            getattr(self.args, "ligand_contact_max_distance", 4.5)
                        ),
                        "force": bool(
                            getattr(self.args, "ligand_contact_force", False)
                        ),
                        "weight": float(
                            getattr(self.args, "ligand_pocket_weight", 1.0)
                        ),
                    }
                }
            )
        return constraints


    def _process_sequence_inputs(self):
        """
        Parses and groups protein sequences and MSAs from command line arguments.
        Ensures protein_seqs_list and protein_msas_list are aligned and padded.
        """
        a = self.args
        
        protein_seqs_list = smart_split(a.protein_seqs) if a.protein_seqs else []

        # Handle special "empty" case: --protein_msas "empty" applies to all seqs
        if a.msa_mode == "single":
            protein_msas_list = ["empty"] * len(protein_seqs_list)
        elif a.msa_mode == "mmseqs":
            protein_msas_list = ["mmseqs"] * len(protein_seqs_list)
        else:
            raise ValueError(f"Invalid msa_mode: {a.msa_mode}")

        return protein_seqs_list, protein_msas_list


    def build(self):
        """
        Constructs the base Boltz input data dictionary (sequences, templates, constraints).
        """
        a = self.args

        # 纯无条件（无靶标序列）：仅链 A + 可选 binder 模板
        # 若 --mode unconditional 且提供了 --protein_seqs（如单条 target），
        # 与 binder 模式相同地构建 A+B 异二聚体，便于 target 模板映射到 B 链。
        pure_unconditional = a.mode == "unconditional" and not (a.protein_seqs or "").strip()

        if pure_unconditional:
            data = self._build_unconditional_data()
            pocket_conditioning = False
            templates = self._build_templates(protein_chain_ids=[])
            if templates:
                data["templates"] = templates
        else:
            data, pocket_conditioning = self._build_conditional_data()

        # Sort sequences by chain ID for consistent processing
        data["sequences"] = sorted(
            data["sequences"], key=lambda entry: list(entry.values())[0]["id"][0]
        )
        # Ligand contact/pocket constraints also require Boltz2 conditioning and
        # force-potential features. Do not gate this only on --contact_residues.
        pocket_conditioning = pocket_conditioning or bool(data.get("constraints"))

        print("Data dictionary:\n", data)
        return data, pocket_conditioning


    def _build_unconditional_data(self):
        """Constructs data for unconditional binder design."""
        a = self.args
        binder_msa = getattr(a, "binder_msa", "") or ""
        if binder_msa and binder_msa != "empty":
            binder_msa = os.path.abspath(os.path.expanduser(binder_msa))
            if not os.path.isfile(binder_msa):
                raise FileNotFoundError(f"binder_msa file not found: {binder_msa}")
            if not binder_msa.endswith(".npz"):
                raise ValueError(f"binder_msa must be a .npz file: {binder_msa}")
        else:
            binder_msa = "empty"
        sequences = [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": "X",
                    "msa": binder_msa,
                    "cyclic": a.cyclic,
                }
            }
        ]

        ligand_specs = self._parse_ligand_specs("B")
        ligand_chain_ids = []
        next_chain_ord = ord("B") + len(ligand_specs)

        for spec in ligand_specs:
            chain_id = spec["chain_id"]
            ligand_chain_ids.append(chain_id)
            sequences.append(
                {
                    "ligand": {
                        "id": [chain_id],
                        spec["kind"]: spec["value"],
                    }
                }
            )

        if a.nucleic_seq:
            sequences.append(
                {
                    a.nucleic_type: {
                        "id": [chr(next_chain_ord)],
                        "sequence": a.nucleic_seq,
                    }
                }
            )

        data = {"sequences": sequences}

        constraints = []
        ligand_contact_constraints = self._parse_ligand_contact_constraints(
            ligand_chain_ids
        )
        if ligand_contact_constraints:
            constraints.extend(ligand_contact_constraints)

        ligand_pocket_constraints = self._parse_ligand_pocket_constraints(
            ligand_chain_ids
        )
        if ligand_pocket_constraints:
            constraints.extend(ligand_pocket_constraints)

        if constraints:
            data["constraints"] = constraints
        return data


    def _build_conditional_data(self):
        """Constructs data for conditioned design (binder + target + optional non-protein)."""
        a = self.args
        protein_seqs_list, protein_msas_list = (
            self._process_sequence_inputs()
        )
        sequences = []
        
        # Assign chain IDs to proteins first
        protein_chain_ids = [chr(ord('B') + i) for i in range(len(protein_seqs_list))]
        
        # Find next available chain letter
        next_chain_idx = len(protein_chain_ids)
        
        ligand_chain_id = None
        if a.ligand_smiles or a.ligand_ccd:
            ligand_chain_id = chr(ord('B') + next_chain_idx)
            next_chain_idx += 1
            
        nucleic_chain_id = None
        if a.nucleic_seq:
            nucleic_chain_id = chr(ord('B') + next_chain_idx)
            next_chain_idx += 1
        # --- END NEW ---

        # Step 1: Determine canonical MSA for each unique target sequence
        seq_to_indices = defaultdict(list)
        for idx, seq in enumerate(protein_seqs_list):
            if seq:
                seq_to_indices[seq].append(idx)
        
        seq_to_final_msa = {}
        for seq, idx_list in seq_to_indices.items():
            chosen_msa = next(
                (
                    protein_msas_list[i]
                    for i in idx_list
                ),
                None
            )
            chosen_msa = chosen_msa if chosen_msa is not None else ""

            if chosen_msa == "mmseqs":
                pid = protein_chain_ids[idx_list[0]]
                msa_value = process_msa(pid, seq, Path(self.protein_hunter_save_dir))
                seq_to_final_msa[seq] = str(msa_value)
            elif chosen_msa == "empty":
                seq_to_final_msa[seq] = "empty"
            else:
                raise ValueError(f"Invalid msa_mode: {a.msa_mode}")

        # Step 2: Build sequences list for target proteins
        for i, (seq, msa) in enumerate(zip(protein_seqs_list, protein_msas_list)):
            if not seq:
                continue
            pid = protein_chain_ids[i]
            final_msa = seq_to_final_msa.get(seq, "empty")
            sequences.append(
                {
                    "protein": {
                        "id": [pid],
                        "sequence": seq,
                        "msa": final_msa,
                    }
                }
            )

        # Step 3: Add binder chain
        binder_msa = getattr(a, "binder_msa", "") or ""
        if binder_msa and binder_msa != "empty":
            binder_msa = os.path.abspath(os.path.expanduser(binder_msa))
            if not os.path.isfile(binder_msa):
                raise FileNotFoundError(f"binder_msa file not found: {binder_msa}")
            if not binder_msa.endswith(".npz"):
                raise ValueError(f"binder_msa must be a .npz file: {binder_msa}")
        else:
            binder_msa = "empty"
        sequences.append(
            {
                "protein": {
                    "id": ["A"], # Hardcoded 'A'
                    "sequence": "X",
                    "msa": binder_msa,
                    "cyclic": a.cyclic,
                }
            }
        )

        # Step 4: Add ligands/nucleic acids
        ligand_specs = self._parse_ligand_specs(ligand_chain_id)
        ligand_chain_ids = []
        for spec in ligand_specs:
            chain_id = spec["chain_id"]
            ligand_chain_ids.append(chain_id)
            sequences.append(
                {
                    "ligand": {
                        "id": [chain_id],
                        spec["kind"]: spec["value"],
                    }
                }
            )
        if a.nucleic_seq:
            sequences.append(
                {a.nucleic_type: {"id": [nucleic_chain_id], "sequence": a.nucleic_seq}}
            )

        # Step 5: Add templates
        templates = self._build_templates(protein_chain_ids)

        data = {"sequences": sequences}
        if templates:
            data["templates"] = templates

        # Step 6: Add constraints (pocket conditioning + ligand contacts)
        constraints = []
        pocket_conditioning = bool(a.contact_residues and a.contact_residues.strip())
        if pocket_conditioning:
            contacts = []
            residues_chains = a.contact_residues.split("|")
            for i, residues_chain in enumerate(residues_chains):
                residues = residues_chain.split(",")
                contacts.extend([
                    [protein_chain_ids[i], int(res)]
                    for res in residues
                    if res.strip() != ""
                ])
            constraints.append({"pocket": {"binder": "A", "contacts": contacts}})

        ligand_contact_constraints = self._parse_ligand_contact_constraints(
            ligand_chain_ids
        )
        if ligand_contact_constraints:
            constraints.extend(ligand_contact_constraints)

        ligand_pocket_constraints = self._parse_ligand_pocket_constraints(
            ligand_chain_ids
        )
        if ligand_pocket_constraints:
            constraints.extend(ligand_pocket_constraints)

        if constraints:
            data["constraints"] = constraints

        return data, pocket_conditioning

    def _sniff_first_chain_in_structure(self, file_path):
        """从 PDB/mmCIF 中读取第一条 ATOM/HETATM 所在链（无文件时默认 A）。"""
        if not file_path or not os.path.exists(file_path):
            return "A"
        try:
            if str(file_path).endswith(".pdb"):
                with open(file_path, "r") as f:
                    for line in f:
                        if line.startswith("ATOM  ") or line.startswith("HETATM"):
                            return line[21].strip() or "A"

            from prody import parseMMCIF, parsePDB

            structure = (
                parseMMCIF(str(file_path))
                if str(file_path).endswith(".cif")
                else parsePDB(str(file_path))
            )
            if structure is not None:
                chids = structure.getChids()
                return chids[0] if len(chids) > 0 else "A"
        except Exception as e:
            print(
                f"[Warning] 无法从 {file_path} 自动获取链名，默认使用 'A'。错误: {e}"
            )
        return "A"

    def _resolve_template_source_chain(self, file_path, explicit_chain_id: str) -> str:
        """优先使用 CLI 显式指定的模板源链，否则回退到文件内嗅探。"""
        ex = (explicit_chain_id or "").strip()
        if ex:
            return ex
        return self._sniff_first_chain_in_structure(file_path)

    def _parse_target_template_source_chain_ids(self):
        """
        与 target_template_path / template_path 的路径列表对齐的源链 ID。
        优先 --target_template_cif_chain_id，否则 --template_cif_chain_id；
        支持逗号或冒号分隔；若仅给一条则广播到所有 target 槽位。
        """
        a = self.args
        raw = (getattr(a, "target_template_cif_chain_id", "") or "").strip()
        if not raw:
            raw = (getattr(a, "template_cif_chain_id", "") or "").strip()
        return smart_split(raw) if raw else []

    def _build_templates(self, protein_chain_ids):
        """
        构造模板字典列表。
        无论输入的 PDB 链名是什么，都强制映射到逻辑链 ID（Binder=A, Target=B,C...）。
        """
        a = self.args
        templates = []

        target_src_chain_ids = self._parse_target_template_source_chain_ids()

        # 1. 处理 Target 链模板
        # 检查 target_template_path (新参数) 或原有 template_path
        target_template_input = getattr(a, "target_template_path", "") or getattr(a, "template_path", "")
        
        if target_template_input:
            # 支持逗号分隔的多链模板
            path_list = smart_split(target_template_input)
            num_targets = len(protein_chain_ids)
            
            # 补齐路径列表
            while len(path_list) < num_targets:
                path_list.append("")

            for i in range(num_targets):
                t_path = path_list[i]
                if not t_path: continue
                
                # 获取 Boltz 需要的文件对象
                template_file = get_cif(t_path)

                if i < len(target_src_chain_ids) and target_src_chain_ids[i]:
                    explicit_src = target_src_chain_ids[i]
                elif len(target_src_chain_ids) == 1 and target_src_chain_ids[0]:
                    explicit_src = target_src_chain_ids[0]
                else:
                    explicit_src = ""

                t_block = {
                    "cif" if template_file.endswith(".cif") else "pdb": template_file,
                    # 逻辑链：B, C...
                    "chain_id": protein_chain_ids[i],
                    # PDB/mmCIF 源链（Boltz schema 字段名为 template_id）
                    "template_id": self._resolve_template_source_chain(
                        t_path, explicit_src
                    ),
                }
                
                if getattr(a, "target_template_force", False) or getattr(
                    a, "fix_all_if_no_x", False
                ):
                    t_block["force"] = True
                    t_block["threshold"] = float(
                        getattr(a, "target_template_force_threshold", 0.0)
                    )
                
                templates.append(t_block)

        # 2. 处理 Binder 链模板 (链 A)
        if getattr(a, "binder_template_path", ""):
            b_path = a.binder_template_path
            binder_file = get_cif(b_path)

            b_ex = (getattr(a, "binder_template_cif_chain_id", "") or "").strip()
            binder_src_parts = smart_split(b_ex) if b_ex else []
            if binder_src_parts:
                binder_explicit = binder_src_parts[0]
            else:
                binder_explicit = ""
            
            binder_block = {
                "cif" if binder_file.endswith(".cif") else "pdb": binder_file,
                # 逻辑链 A
                "chain_id": "A",
                # PDB/mmCIF 源链
                "template_id": self._resolve_template_source_chain(
                    b_path, binder_explicit
                ),
            }

            if getattr(a, "binder_template_force", False):
                binder_block["force"] = True
                binder_block["threshold"] = float(
                    getattr(a, "binder_template_force_threshold", 0.0)
                )
            
            templates.append(binder_block)
                
        return templates

class ProteinHunter_Boltz:
    """
    Core class to manage the protein design pipeline, including Boltz structure
    prediction, LigandMPNN design, cycle optimization, and downstream validation.
    """

    def __init__(self, args):
        self.args = args
        validate_pipeline_args(args)
        self.effective_unconditional_msa_mode = effective_unconditional_msa_mode(args)
        self.unconditional_mmseqs_msa = (
            args.mode == "unconditional"
            and self.effective_unconditional_msa_mode == "mmseqs"
        )
        binder_msa_nonempty = bool(
            (args.binder_msa or "").strip() and args.binder_msa != "empty"
        )
        self.dynamic_binder_msa_enabled = (
            (args.mode == "binder" and binder_msa_nonempty)
            or (
                args.mode == "unconditional"
                and self.effective_unconditional_msa_mode == "precomputed"
            )
        )

        self.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        # 1. Initialize shared resources (persists across all runs)
        self.ccd_path = Path(args.ccd_path).expanduser()
        self.ccd_lib = load_canonicals(str(self.ccd_path))
        
        # Initialize the Input Data Builder
        self.data_builder = InputDataBuilder(args)
        
        # 2. Initialize Models
        self.boltz_model = self._load_boltz_model()
        ligand_mpnn_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "LigandMPNN", "run.py")
        self.designer = LigandMPNNWrapper(ligand_mpnn_path, python=sys.executable)

        # 3. Setup Directories
        self.save_dir = self.data_builder.save_dir
        self.protein_hunter_save_dir = self.data_builder.protein_hunter_save_dir
        self.binder_chain = "A"
        self.binder_msa_dir = Path(self.protein_hunter_save_dir) / "MSA"
        self.binder_msa_source_path = (
            os.path.abspath(os.path.expanduser(args.binder_msa))
            if self.dynamic_binder_msa_enabled
            else ""
        )
        self.binder_gap_indices = [i for i, c in enumerate(args.seq) if c == "X"] if args.seq else []
        self.binder_frozen_template_sequences = (
            self._load_binder_msa_templates(self.binder_msa_source_path)
            if self.dynamic_binder_msa_enabled
            else []
        )

        print("✅ ProteinHunter_Boltz initialized.")

    def _load_boltz_model(self):
        """Loads and configures the Boltz model."""
        predict_args = {
            "recycling_steps": self.args.recycling_steps,
            "sampling_steps": self.args.diffuse_steps,
            "diffusion_samples": 1,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
            "max_parallel_samples": 1,
        }
        return get_boltz_model(
            checkpoint=self.args.boltz_model_path,
            predict_args=predict_args,
            device=self.device,
            model_version=self.args.boltz_model_version,
            no_potentials=not bool(
                (getattr(self.args, "contact_residues", "") or "").strip()
                or (getattr(self.args, "ligand_contact_pairs", "") or "").strip()
                or (getattr(self.args, "ligand_pocket_constraints", "") or "").strip()
            ),
            grad_enabled=self.args.grad_enabled,
            msa_max_seqs=self.args.msa_max_seqs,
        )

    def _set_binder_msa_path(self, data_cp, msa_path):
        for seq_entry in data_cp["sequences"]:
            if "protein" in seq_entry and self.binder_chain in seq_entry["protein"]["id"]:
                seq_entry["protein"]["msa"] = str(msa_path)
                print(f"[MSA→Boltz] chain={self.binder_chain} npz={msa_path}")
                return
        raise ValueError("Binder chain not found in data dictionary.")

    def _load_binder_msa_templates(self, msa_npz_path: str):
        data = np.load(msa_npz_path, allow_pickle=True)
        sequences = data["sequences"]
        residues = data["residues"]
        templates = []
        for seq_meta in sequences[1:]:
            start = int(seq_meta["res_start"])
            end = int(seq_meta["res_end"])
            token_ids = residues[start:end]["res_type"]
            chars = []
            for token_id in token_ids:
                token_name = const.tokens[int(token_id)]
                chars.append(const.prot_token_to_letter.get(token_name, "X"))
            templates.append("".join(chars))
        return templates

    def _build_dynamic_binder_msa(self, binder_seq: str, cycle_idx: int) -> str:
        if not self.dynamic_binder_msa_enabled:
            return "empty"
        self.binder_msa_dir.mkdir(parents=True, exist_ok=True)
        name = getattr(self.args, 'name', 'design')
        a3m_path = self.binder_msa_dir / f"{name}_cycle_{cycle_idx}.a3m"
        npz_path = self.binder_msa_dir / f"{name}_cycle_{cycle_idx}.npz"
        template_lines = []
        if self.binder_frozen_template_sequences:
            for tpl in self.binder_frozen_template_sequences:
                if len(tpl) == len(binder_seq):
                    template_lines.append(tpl)
                elif len(tpl) > len(binder_seq):
                    template_lines.append(tpl[:len(binder_seq)])
                else:
                    template_lines.append(tpl + "-" * (len(binder_seq) - len(tpl)))
        else:
            template_lines.append("".join("-" if i in self.binder_gap_indices else "X" for i in range(len(binder_seq))))
        with a3m_path.open("w") as f:
            f.write(f">query\n{binder_seq}\n")
            for idx, tpl in enumerate(template_lines, start=1):
                f.write(f">template_{idx}\n{tpl}\n")
        msa = parse_a3m(a3m_path, taxonomy=None, max_seqs=None)
        msa.dump(npz_path)
        return str(npz_path)

    def _apply_binder_msa_for_prediction(
        self, data_cp, binder_seq: str, cycle_idx: int, mmseqs_state: dict | None = None
    ):
        """Refresh chain-A MSA before Boltz: mmseqs (unconditional only) or dynamic npz templates."""
        if self.unconditional_mmseqs_msa:
            st = mmseqs_state if mmseqs_state is not None else {"base_npz": None, "n_res": None}
            msa_path = unconditional_mmseqs_npz_for_cycle(
                self.binder_chain,
                binder_seq,
                Path(self.protein_hunter_save_dir),
                cycle_idx,
                st,
                max_seqs=self.args.msa_max_seqs,
            )
            self._set_binder_msa_path(data_cp, msa_path)
            return
        if self.dynamic_binder_msa_enabled:
            path = self._build_dynamic_binder_msa(binder_seq, cycle_idx)
            self._set_binder_msa_path(data_cp, path)

    def _run_design_cycle(self, data_cp, run_id, pocket_conditioning):
        """
        Executes a single design run, including Cycle 0 contact filtering and
        the subsequent design/prediction cycles.
        
        Note: The cycle logic remains highly coupled due to the nature of the
        sequential design process, but relies on modular imports.
        """
        a = self.args
        run_save_dir = os.path.join(self.protein_hunter_save_dir, f"run_{run_id}")
        os.makedirs(run_save_dir, exist_ok=True)

        # Initialize run metrics and tracking variables
        best_iptm = float("-inf")
        best_seq = None
        best_structure = None
        best_output = None
        best_pdb_filename = None
        best_cycle_idx = -1
        best_alanine_percentage = None
        run_metrics = {"run_id": run_id}
        mmseqs_state = {"base_npz": None, "n_res": None}

        if a.seq =="":
            binder_length = random.randint(
                a.min_protein_length, a.max_protein_length
            )
        else:
            binder_length = len(a.seq)

        protein_chain_ids = []
        if (a.contact_residues or "").strip():
            pocket_contacts = []
            for c in data_cp.get("constraints", []):
                if "pocket" in c:
                    pocket_contacts.extend(c["pocket"].get("contacts", []))
            protein_chain_ids = sorted({chain for chain, _ in pocket_contacts})

        # Helper function to update sequence in the data dictionary
        def update_binder_sequence(new_seq):
            for seq_entry in data_cp["sequences"]:
                if (
                    "protein" in seq_entry
                    and self.binder_chain in seq_entry["protein"]["id"]
                ):
                    seq_entry["protein"]["sequence"] = new_seq
                    return
            # Should not happen if data_cp is built correctly
            raise ValueError("Binder chain not found in data dictionary.")

        # Set initial binder sequence
        if a.seq =="":
            initial_seq = sample_seq(
                binder_length, exclude_P=a.exclude_P, frac_X=a.percent_X/100
            )
        else:
            initial_seq = a.seq
        seq_upper = initial_seq.upper()
        fixed_residues = ""
        if "X" in seq_upper:
            fixed_residues = " ".join(
                f"{self.binder_chain}{idx + 1}"
                for idx, aa in enumerate(seq_upper)
                if aa != "X"
            )
        elif getattr(a, "fix_all_if_no_x", False):
            fixed_residues = " ".join(f"{self.binder_chain}{i+1}" for i in range(len(seq_upper)))
        update_binder_sequence(initial_seq)
        if self.unconditional_mmseqs_msa or self.dynamic_binder_msa_enabled:
            self._apply_binder_msa_for_prediction(
                data_cp, initial_seq, 0, mmseqs_state
            )
        print(f"Binder initial sequence length: {binder_length}")

        # --- Cycle 0 structure prediction, with contact filtering check ---
        contact_filter_attempt = 0
        pdb_filename = ""
        structure = None
        output = None

        while True:
            output, structure = run_prediction(
                data_cp,
                self.binder_chain,
                randomly_kill_helix_feature=a.randomly_kill_helix_feature,
                negative_helix_constant=a.negative_helix_constant,
                boltz_model=self.boltz_model,
                ccd_lib=self.ccd_lib,
                ccd_path=self.ccd_path,
                logmd=a.logmd,
                device=self.device,
                boltz_model_version=a.boltz_model_version,
                pocket_conditioning=pocket_conditioning,
                msa_max_seqs=a.msa_max_seqs,
            )
            pdb_filename = f"{run_save_dir}/{a.name}_run_{run_id}_predicted_cycle_0.pdb"
            plddts = output["plddt"].detach().cpu().numpy()[0]
            save_pdb(structure, output["coords"], plddts, pdb_filename)

            contact_check_okay = True
            if a.contact_residues.strip() and not a.no_contact_filter:
                try:
                    binds = all([binder_binds_contacts(
                        pdb_filename,
                        self.binder_chain,
                        protein_chain_ids[i],
                        contact_res,
                        cutoff=a.contact_cutoff,
                    )
                    for i, contact_res in enumerate(a.contact_residues.split("|"))
                ])
                    if not binds:
                        print(
                            "❌ Binder does NOT contact required residues after cycle 0. Retrying..."
                        )
                        contact_check_okay = False
                except Exception as e:
                    print(f"WARNING: Could not perform binder-contact check: {e}")
                    contact_check_okay = True  # Fail open

            if contact_check_okay:
                break
            contact_filter_attempt += 1
            if contact_filter_attempt >= a.max_contact_filter_retries:
                print("WARNING: Max retries for contact filtering reached. Proceeding.")
                break

            # Resample initial sequence
            new_seq = sample_seq(binder_length, exclude_P=a.exclude_P, frac_X=a.percent_X/100)
            seq_upper = new_seq.upper()
            fixed_residues = ""
            if "X" in seq_upper:
                fixed_residues = " ".join(
                    f"{self.binder_chain}{idx + 1}"
                    for idx, aa in enumerate(seq_upper)
                    if aa != "X"
                )
            elif getattr(a, "fix_all_if_no_x", False):
                fixed_residues = " ".join(f"{self.binder_chain}{i+1}" for i in range(len(seq_upper)))
            if self.unconditional_mmseqs_msa:
                mmseqs_state["base_npz"] = None
                mmseqs_state["n_res"] = None
            update_binder_sequence(new_seq)
            if self.unconditional_mmseqs_msa or self.dynamic_binder_msa_enabled:
                self._apply_binder_msa_for_prediction(
                    data_cp, new_seq, 0, mmseqs_state
                )
            clean_memory()

        clean_memory() # <-- ADD THIS CALL HERE
        # Capture Cycle 0 metrics
        binder_chain_idx = CHAIN_TO_NUMBER[self.binder_chain]
        pair_chains = output["pair_chains_iptm"]
        
        # Calculate i-pTM
        if len(pair_chains) > 1:
            values = [
                (
                    pair_chains[binder_chain_idx][i].detach().cpu().numpy()
                    + pair_chains[i][binder_chain_idx].detach().cpu().numpy()
                )
                / 2.0
                for i in range(len(pair_chains))
                if i != binder_chain_idx
            ]
            cycle_0_iptm = float(np.mean(values) if values else 0.0)
        else:
            cycle_0_iptm = 0.0

        run_metrics["cycle_0_iptm"] = cycle_0_iptm
        run_metrics["cycle_0_plddt"] = float(
            output.get("complex_plddt", torch.tensor([0.0])).detach().cpu().numpy()[0]
        )
        run_metrics["cycle_0_iplddt"] = float(
            output.get("complex_iplddt", torch.tensor([0.0])).detach().cpu().numpy()[0]
        )
        run_metrics["cycle_0_alanine"] = 0

        # --- Optimization Cycles ---
        for cycle in range(a.num_cycles):
            print(f"\n--- Run {run_id}, Cycle {cycle + 1} ---")

            # Calculate temperature and bias for the cycle
            cycle_norm = (cycle / (a.num_cycles - 1)) if a.num_cycles > 1 else 0.0
            alpha = a.alanine_bias_start - cycle_norm * (
                a.alanine_bias_start - a.alanine_bias_end
            )

            # 1. Design Sequence
            default_model_type = (
                "ligand_mpnn"
                if (
                    a.ligand_smiles
                    or a.ligand_ccd
                    or (getattr(a, "ligand_specs", "") or "").strip()
                    or a.nucleic_seq
                )
                else "soluble_mpnn"
            )
            model_type = (
                a.mpnn_model_type
                if getattr(a, "mpnn_model_type", "auto") != "auto"
                else default_model_type
            )
            design_kwargs = {
                "pdb_file": pdb_filename,
                "temperature": a.temperature,
                "chains_to_design": self.binder_chain,
                "omit_AA": f"{a.omit_AA},P" if cycle == 0 else a.omit_AA,
                "bias_AA": f"A:{alpha}" if a.alanine_bias else "",
                "fixed_residues": fixed_residues,
            }

            seq_str, logits = design_sequence(
                self.designer, model_type, **design_kwargs
            )
            # The output seq_str is a dictionary-like string, we extract the binder chain sequence
            seq = seq_str.split(":")[CHAIN_TO_NUMBER[self.binder_chain]] 

            # Update data_cp with new sequence
            alanine_count = seq.count("A")
            alanine_percentage = (
                alanine_count / binder_length if binder_length != 0 else 0.0
            )
            update_binder_sequence(seq) # Use the helper function
            if self.unconditional_mmseqs_msa or self.dynamic_binder_msa_enabled:
                self._apply_binder_msa_for_prediction(
                    data_cp, seq, cycle + 1, mmseqs_state
                )

            # 2. Structure Prediction
            output, structure = run_prediction(
                data_cp,
                self.binder_chain,
                seq=seq,
                randomly_kill_helix_feature=False,
                negative_helix_constant=0.0,
                boltz_model=self.boltz_model,
                ccd_lib=self.ccd_lib,
                ccd_path=self.ccd_path,
                logmd=False,
                device=self.device,
                msa_max_seqs=a.msa_max_seqs,
                pocket_conditioning=pocket_conditioning,
            )

            # Calculate ipTM
            current_chain_idx = CHAIN_TO_NUMBER[self.binder_chain]
            pair_chains = output["pair_chains_iptm"]
            if len(pair_chains) > 1:
                values = [
                    (
                        pair_chains[current_chain_idx][i].detach().cpu().numpy()
                        + pair_chains[i][current_chain_idx].detach().cpu().numpy()
                    )
                    / 2.0
                    for i in range(len(pair_chains))
                    if i != current_chain_idx
                ]
                current_iptm = float(np.mean(values) if values else 0.0)
            else:
                current_iptm = 0.0

            # Update best structure (only if alanine content is acceptable)
            if alanine_percentage <= 0.20 and current_iptm > best_iptm:
                best_iptm = current_iptm
                best_seq = seq
                best_structure = copy.deepcopy(structure)
                best_output = shallow_copy_tensor_dict(output)
                best_pdb_filename = (
                    f"{run_save_dir}/{a.name}_run_{run_id}_best_structure.pdb"
                )
                best_plddts = best_output["plddt"].detach().cpu().numpy()[0]
                save_pdb(
                    best_structure,
                    best_output["coords"],
                    best_plddts,
                    best_pdb_filename,
                )
                best_cycle_idx = cycle + 1
                best_alanine_percentage = alanine_percentage

            # 3. Log Metrics and Save PDB
            curr_plddt = float(
                output.get("complex_plddt", torch.tensor([0.0]))
                .detach()
                .cpu()
                .numpy()[0]
            )
            curr_iplddt = float(
                output.get("complex_iplddt", torch.tensor([0.0]))
                .detach()
                .cpu()
                .numpy()[0]
            )

            run_metrics[f"cycle_{cycle + 1}_iptm"] = current_iptm
            run_metrics[f"cycle_{cycle + 1}_plddt"] = curr_plddt
            run_metrics[f"cycle_{cycle + 1}_iplddt"] = curr_iplddt
            run_metrics[f"cycle_{cycle + 1}_alanine"] = alanine_count
            run_metrics[f"cycle_{cycle + 1}_seq"] = seq

            pdb_filename = (
                f"{run_save_dir}/{a.name}_run_{run_id}_predicted_cycle_{cycle + 1}.pdb"
            )
            plddts = output["plddt"].detach().cpu().numpy()[0]
            save_pdb(structure, output["coords"], plddts, pdb_filename)
            clean_memory()

            print(
                f"ipTM: {current_iptm:.2f} pLDDT: {curr_plddt:.2f} iPLDDT: {curr_iplddt:.2f} Alanine count: {alanine_count}"
            )

            # 4. Save YAML for High ipTM
            save_yaml_this_design = (alanine_percentage <= 0.20) and (
                current_iptm > a.high_iptm_threshold
                and curr_plddt > a.high_plddt_threshold
            )

            if save_yaml_this_design and a.contact_residues.strip():
                this_cycle_pdb_filename = f"{run_save_dir}/{a.name}_run_{run_id}_predicted_cycle_{cycle + 1}.pdb"
                try:
                    contact_binds = all([binder_binds_contacts(
                        this_cycle_pdb_filename,
                        self.binder_chain,
                        protein_chain_ids[i],
                        contact_res,
                        cutoff=a.contact_cutoff,
                    )
                    for i, contact_res in enumerate(a.contact_residues.split("|"))
                ])
                    if not contact_binds:
                        save_yaml_this_design = False
                        print(
                            "⛔️ Not saving YAML: binder failed contact check for high ipTM save."
                        )
                except Exception as e:
                    print(
                        f"WARNING: Exception during contact check: {e}. Saving YAML anyway."
                    )
            if save_yaml_this_design:
                # --- Define base names and directories ---
                base_filename = f"{a.name}_run_{run_id}_cycle_{cycle + 1}"
                
                # 1. Save YAML
                high_iptm_yaml_dir = os.path.join(self.save_dir, "high_iptm_yaml")
                os.makedirs(high_iptm_yaml_dir, exist_ok=True)
                yaml_base_name = f"{base_filename}_output.yaml"
                yaml_filename = os.path.join(high_iptm_yaml_dir, yaml_base_name)
                
                with open(yaml_filename, "w") as f:
                    yaml.dump(data_cp, f, default_flow_style=False)
                print(f"✅ Saved run {run_id} cycle {cycle + 1} YAML.")

                # 2. Copy PDB
                high_iptm_pdb_dir = os.path.join(self.save_dir, "high_iptm_pdb")
                os.makedirs(high_iptm_pdb_dir, exist_ok=True)
                
                # Use the 'base_filename' for a consistent destination name
                pdb_base_name = f"{base_filename}_structure.pdb" 
                dest_pdb_path = os.path.join(high_iptm_pdb_dir, pdb_base_name)
                
                # Use 'pdb_filename', which is *always* defined just above this block
                # This was the source of the UnboundLocalError
                shutil.copy(pdb_filename, dest_pdb_path)
                print(f"✅ Copied PDB to {dest_pdb_path}")

                # 3. Append to high-ipTM summary CSV
                summary_csv_path = os.path.join(self.save_dir, "summary_high_iptm.csv")
                
                metrics_dict = {
                    "run_id": run_id,
                    "cycle": cycle + 1,
                    "iptm": current_iptm,
                    "plddt": curr_plddt,
                    "iplddt": curr_iplddt,
                    "alanine_count": alanine_count,
                    "sequence": seq,
                    "pdb_filename": pdb_base_name, # Log the *new* filename
                    "yaml_filename": yaml_base_name
                }
                
                # Check if file exists to write header only once
                file_exists = os.path.isfile(summary_csv_path)
                
                df_row = pd.DataFrame([metrics_dict])
                df_row.to_csv(summary_csv_path, mode='a', header=not file_exists, index=False)
                print(f"✅ Appended high-ipTM metrics to {summary_csv_path}")
        # End of cycle visualization
        if best_structure is not None and a.plot:
            plot_from_pdb(best_pdb_filename)
        elif best_structure is None:
            print(
                f"\nNo structure was generated for run {run_id} (no eligible best design with <= 20% alanine)."
            )

        # Finalize best metrics for CSV
        if best_alanine_percentage is not None and best_alanine_percentage <= 0.20:
            run_metrics["best_iptm"] = float(best_iptm)
            run_metrics["best_cycle"] = best_cycle_idx
            run_metrics["best_plddt"] = float(
                best_output.get("complex_plddt", torch.tensor([0.0]))
                .detach()
                .cpu()
                .numpy()[0]
            )
            run_metrics["best_seq"] = best_seq
        else:
            run_metrics["best_iptm"] = float("nan")
            run_metrics["best_cycle"] = None
            run_metrics["best_plddt"] = float("nan")
            run_metrics["best_seq"] = None

        if a.plot:
            plot_run_metrics(run_save_dir, a.name, run_id, a.num_cycles, run_metrics)

        return run_metrics

    def _save_summary_metrics(self, all_run_metrics):
        """Saves all run metrics to a single CSV file."""
        a = self.args
        columns = ["run_id"]
        # Columns for cycle 0 through num_cycles
        for i in range(a.num_cycles + 1):
            columns.extend(
                [
                    f"cycle_{i}_iptm",
                    f"cycle_{i}_plddt",
                    f"cycle_{i}_iplddt",
                    f"cycle_{i}_alanine",
                    f"cycle_{i}_seq",
                ]
            )
        # Best metric columns
        columns.extend(["best_iptm", "best_cycle", "best_plddt", "best_seq"])

        summary_csv = os.path.join(self.save_dir, "summary_all_runs.csv")
        df = pd.DataFrame(all_run_metrics)
        
        # Ensure all expected columns are present (filling missing with NaN)
        for col in columns:
            if col not in df.columns:
                df[col] = float("nan")
        
        # Filter to columns in the correct order (and existing ones)
        df = df[[c for c in columns if c in df.columns]]
        df.to_csv(summary_csv, index=False)
        print(f"\n✅ All run/cycle metrics saved to {summary_csv}")

    def _run_downstream_validation(self):
        # This ensures they are in scope when called later in this function
        sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
        from utils.alphafold_utils import run_alphafold_step
        from utils.pyrosetta_utils import run_rosetta_step
        
        """Executes AlphaFold and Rosetta validation steps."""
        a = self.args
        
        # Determine target type for Rosetta validation
        any_ligand_or_nucleic = a.ligand_smiles or a.ligand_ccd or a.nucleic_seq
        if a.nucleic_type.strip() and a.nucleic_seq.strip():
            target_type = "nucleic"
        elif any_ligand_or_nucleic:
            target_type = "small_molecule"
        else:
            target_type = "protein" # Default for unconditional mode

        success_dir = f"{self.save_dir}/1_af3_rosetta_validation"
        high_iptm_yaml_dir = os.path.join(self.save_dir, "high_iptm_yaml")

        if os.path.exists(high_iptm_yaml_dir):
            print("Starting downstream validation (AlphaFold3 and Rosetta)...")

            # --- AlphaFold Step ---
            af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo = (
                run_alphafold_step(
                    high_iptm_yaml_dir,
                    os.path.expanduser(a.alphafold_dir),
                    a.af3_docker_name,
                    os.path.expanduser(a.af3_database_settings),
                    os.path.expanduser(a.hmmer_path),
                    success_dir,
                    os.path.expanduser(a.work_dir) or os.getcwd(),
                    binder_id=self.binder_chain,
                    gpu_id=a.gpu_id,
                    high_iptm=True,
                    use_msa_for_af3=a.use_msa_for_af3,
                )
            )
            if target_type == "protein":
                # --- Rosetta Step ---
                run_rosetta_step(
                    success_dir,
                    af_pdb_dir,
                    af_pdb_dir_apo,
                    binder_id=self.binder_chain,
                    target_type=target_type,
                )


    def run_pipeline(self):
        """Orchestrates the entire protein design and validation pipeline."""
        # 1. Prepare Base Data (using the new InputDataBuilder)
        base_data, pocket_conditioning = self.data_builder.build()

        # 2. Run Design Cycles
        all_run_metrics = []
        for design_id in range(self.args.num_designs):
            run_id = str(design_id)
            print("\n=======================================================")
            print(f"=== Starting Design Run {run_id}/{self.args.num_designs - 1} ===")
            print("=======================================================")

        
            data_cp = copy.deepcopy(base_data)

            run_metrics = self._run_design_cycle(data_cp, run_id, pocket_conditioning)
            all_run_metrics.append(run_metrics)

        # 3. Save Summary
        self._save_summary_metrics(all_run_metrics)

        # 4. Run Downstream Validation
        if self.args.use_alphafold3_validation:
            self._run_downstream_validation()
