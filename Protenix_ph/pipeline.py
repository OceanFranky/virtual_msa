import glob
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from runner.batch_inference import get_default_runner, preprocess_input
from runner.inference import infer_predict
from runner.msa_search import msa_search

def _split_chains(value: str) -> List[str]:
    if not value:
        return []
    if ":" in value:
        parts = value.split(":")
    else:
        parts = value.split(",")
    return [part.strip() for part in parts if part.strip()]


def _parse_entity_copy(spec: str) -> dict:
    parts = [x.strip() for x in spec.split(":") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid entity:copy spec: {spec}")
    return {"entity": int(parts[0]), "copy": int(parts[1])}


def _parse_entity_copy_pos(spec: str) -> dict:
    parts = [x.strip() for x in spec.split(":") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"Invalid entity:copy:position spec: {spec}")
    return {"entity": int(parts[0]), "copy": int(parts[1]), "position": int(parts[2])}


def _parse_contact_atom_spec(spec: str) -> dict:
    parts = [x.strip() for x in spec.split(":") if x.strip()]
    if len(parts) < 3:
        raise ValueError(f"Invalid contact site spec: {spec}")
    out = {"entity": int(parts[0]), "copy": int(parts[1]), "position": int(parts[2])}
    if len(parts) > 3:
        out["atom"] = parts[3]
    return out

def _maybe_abs(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return os.path.abspath(path)


def _split_designed_output(seq: str) -> List[str]:
    if "/" in seq:
        return [x.strip() for x in seq.split("/") if x.strip()]
    if ":" in seq:
        return [x.strip() for x in seq.split(":") if x.strip()]
    return [seq.strip()]


def _build_fixed_residues(chain_id: str, template_seq: str, fix_all_if_no_x: bool) -> str:
    seq = template_seq.upper()
    if "X" in seq:
        return " ".join(f"{chain_id}{idx + 1}" for idx, aa in enumerate(seq) if aa != "X")
    if fix_all_if_no_x:
        return " ".join(f"{chain_id}{idx + 1}" for idx in range(len(seq)))
    return ""


def _apply_x_mask(seed_seq: str, designed_seq: str) -> str:
    seed = seed_seq.upper()
    if len(seed_seq) != len(designed_seq):
        raise ValueError(
            f"Designed sequence length mismatch: expected {len(seed_seq)}, got {len(designed_seq)}"
        )
    if "X" not in seed:
        return seed_seq
    out = []
    for i, aa in enumerate(seed):
        if aa == "X":
            out.append(designed_seq[i])
        else:
            out.append(seed_seq[i])
    return "".join(out)


def _truncate_a3m_file(a3m_path: str, max_seqs: int) -> None:
    if max_seqs <= 0 or not os.path.isfile(a3m_path):
        return
    out: List[str] = []
    kept = 0
    with open(a3m_path, "r", encoding="utf-8") as f:
        lines = [x.rstrip("\n") for x in f]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(">"):
            if kept >= max_seqs:
                break
            out.append(line)
            i += 1
            while i < len(lines) and not lines[i].startswith(">"):
                out.append(lines[i])
                i += 1
            kept += 1
        else:
            i += 1
    with open(a3m_path, "w", encoding="utf-8") as f:
        if out:
            f.write("\n".join(out) + "\n")

@dataclass
class InputDataBuilder:
    name: str
    mode: str
    seq: str
    target_items: List[dict]
    binder_msa: str
    target_msa: str
    binder_template_path: str
    template_path: str
    target_template_paths: List[str]
    constraint: Optional[dict]

    def build(self) -> List[dict]:
        if self.mode == "unconditional":
            sequences = self._build_unconditional_data()
        elif self.mode == "binder":
            sequences = self._build_binder_data()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        payload = {"name": self.name, "sequences": sequences}
        if self.constraint:
            payload["constraint"] = self.constraint
        return [payload]

    def write_json(self, output_path: str) -> str:
        payload = self.build()
        output_path = os.path.abspath(output_path)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return output_path

    def _build_unconditional_data(self) -> List[dict]:
        if not self.seq:
            raise ValueError("seq is required for unconditional mode")
        binder_chain = self._build_binder_chain(self.seq)
        return [binder_chain]

    def _build_binder_data(self) -> List[dict]:
        if not self.target_items:
            raise ValueError("target inputs are required for binder mode")
        sequences: List[dict] = []
        for i, target_item in enumerate(self.target_items):
            sequences.append(self._build_target_chain(target_item, i))
        if not self.seq:
            raise ValueError("seq is required for binder mode")
        sequences.append(self._build_binder_chain(self.seq))
        return sequences

    def _build_target_chain(self, item: dict, index: int) -> dict:
        target_type = item.get("type", "protein")
        if target_type == "protein":
            return self._build_target_protein(item, index)
        if target_type == "ligand":
            return self._build_target_ligand(item)
        if target_type == "dna":
            return self._build_target_dna(item)
        if target_type == "rna":
            return self._build_target_rna(item)
        if target_type == "ion":
            return self._build_target_ion(item)
        raise ValueError(f"Unsupported target type: {target_type}")

    def _build_target_protein(self, item: dict, index: int) -> dict:
        sequence = item["sequence"]
        count = int(item.get("count", 1))
        chain = {"sequence": sequence, "count": 1}
        if count != 1:
            chain["count"] = count
        if item.get("id"):
            chain["id"] = item["id"]
        template_path = None
        if index < len(self.target_template_paths):
            template_path = _maybe_abs(self.target_template_paths[index])
        if not template_path:
            template_path = _maybe_abs(self.template_path)
        if template_path:
            chain["templatesPath"] = template_path
        msa_path = _maybe_abs(item.get("msa_path") or self.target_msa)
        if msa_path:
            chain["unpairedMsaPath"] = msa_path
        return {"proteinChain": chain}

    def _build_target_ligand(self, item: dict) -> dict:
        chain = {"ligand": item["ligand"], "count": int(item.get("count", 1))}
        if item.get("id"):
            chain["id"] = item["id"]
        return {"ligand": chain}

    def _build_target_dna(self, item: dict) -> dict:
        chain = {"sequence": item["sequence"], "count": int(item.get("count", 1))}
        if item.get("id"):
            chain["id"] = item["id"]
        return {"dnaSequence": chain}

    def _build_target_rna(self, item: dict) -> dict:
        chain = {"sequence": item["sequence"], "count": int(item.get("count", 1))}
        if item.get("id"):
            chain["id"] = item["id"]
        msa_path = _maybe_abs(item.get("msa_path") or self.target_msa)
        if msa_path:
            chain["unpairedMsaPath"] = msa_path
        return {"rnaSequence": chain}

    def _build_target_ion(self, item: dict) -> dict:
        chain = {"ion": item["ion"], "count": int(item.get("count", 1))}
        if item.get("id"):
            chain["id"] = item["id"]
        return {"ion": chain}

    def _build_binder_chain(self, sequence: str) -> dict:
        chain = {"sequence": sequence, "count": 1}
        msa_path = _maybe_abs(self.binder_msa)
        if msa_path:
            chain["unpairedMsaPath"] = msa_path
        template_path = _maybe_abs(self.binder_template_path)
        if template_path:
            chain["templatesPath"] = template_path
        return {"proteinChain": chain}

class ProteinHunter_Protenix:
    def __init__(self, args):
        self.args = args
        self.name = args.name
        self.mode = args.mode
        self.save_dir = os.path.abspath(args.save_dir)
        self.num_designs = args.num_designs
        self.num_cycles = args.num_cycles

        self.protein_hunter_root = self._resolve_protein_hunter_root(
            getattr(args, "protein_hunter_root", "")
        )
        self.ligand_mpnn_run_py = self._resolve_ligand_mpnn_run_py(
            getattr(args, "ligand_mpnn_run_py", "")
        )
        self.LigandMPNNWrapper = self._import_ligand_mpnn_wrapper(self.protein_hunter_root)
        self.mpnn = self.LigandMPNNWrapper(run_py=self.ligand_mpnn_run_py)
        self.mpnn_opts = {
            "model_type": args.mpnn_model_type,
            "temperature": args.temperature,
            "omit_AA": args.omit_AA,
        }
        if hasattr(args, "alanine_bias") and args.alanine_bias:
            self.mpnn_opts["bias_AA"] = "A:1.5"

        self.target_type = args.target_type
        self.target_count = int(args.target_count)
        self.target_ids = _split_chains(args.target_id)
        self.target_template_paths = _split_chains(getattr(args, "target_template_path", ""))
        self.target_seq = args.target_seq.strip()
        self.target_seqs_raw = args.target_seqs.strip()
        self.model_name = getattr(args, "model_name", "protenix_base_default_v1.0.0")
        self.auto_constraint_model = bool(getattr(args, "auto_constraint_model", False))
        self.binder_msa_mode = args.binder_msa_mode
        self.target_msa_mode = args.target_msa_mode
        self.msa_max_seqs = int(args.msa_max_seqs)
        self.binder_msa_precomputed = (
            os.path.abspath(os.path.expanduser(args.binder_msa)) if args.binder_msa else ""
        )
        self.target_msa_precomputed = (
            os.path.abspath(os.path.expanduser(args.target_msa)) if args.target_msa else ""
        )
        self.target_items = self._build_target_items()
        self.constraint_obj = self._build_constraint_object()
        self._validate_inputs()

        # Keep feature toggles aligned with Protenix runner args.
        self.use_msa_feature = (
            self.binder_msa_mode != "empty"
            or self.target_msa_mode != "empty"
            or bool(self.binder_msa_precomputed)
            or bool(self.target_msa_precomputed)
        )
        self.use_template_feature = bool(args.binder_template_path or args.template_path)
        if self.target_template_paths:
            self.use_template_feature = True
        self.dynamic_binder_msa_enabled = self.binder_msa_mode == "precomputed"
        self.binder_msa_source_path = (
            self.binder_msa_precomputed
            if self.dynamic_binder_msa_enabled
            else ""
        )
        self.binder_frozen_template_sequences = (
            self._load_binder_msa_templates(self.binder_msa_source_path)
            if self.dynamic_binder_msa_enabled
            else []
        )
        if self.auto_constraint_model and self.constraint_obj:
            self.model_name = "protenix_base_constraint_v0.5.0"

    def _build_target_items(self) -> List[dict]:
        if self.mode != "binder":
            return []

        if self.target_type == "protein":
            seqs = _split_chains(self.target_seqs_raw or self.args.protein_seqs)
            if not seqs and self.target_seq:
                seqs = [self.target_seq]
            items = []
            for i, s in enumerate(seqs):
                item = {"type": "protein", "sequence": s, "count": self.target_count}
                if i < len(self.target_ids):
                    item["id"] = [self.target_ids[i]]
                items.append(item)
            return items
        if self.target_type == "ligand":
            ligand = ""
            if self.args.ligand_ccd:
                ligand = f"CCD_{self.args.ligand_ccd}"
            elif self.args.ligand_file:
                ligand = f"FILE_{os.path.abspath(os.path.expanduser(self.args.ligand_file))}"
            elif self.args.ligand_smiles:
                ligand = self.args.ligand_smiles
            if not ligand:
                return []
            item = {"type": "ligand", "ligand": ligand, "count": self.target_count}
            if self.target_ids:
                item["id"] = [self.target_ids[0]]
            return [item]
        if self.target_type == "dna":
            if not self.target_seq:
                return []
            item = {"type": "dna", "sequence": self.target_seq, "count": self.target_count}
            if self.target_ids:
                item["id"] = [self.target_ids[0]]
            return [item]
        if self.target_type == "rna":
            if not self.target_seq:
                return []
            item = {"type": "rna", "sequence": self.target_seq, "count": self.target_count}
            if self.target_ids:
                item["id"] = [self.target_ids[0]]
            return [item]
        if self.target_type == "ion":
            if not self.target_seq:
                return []
            item = {"type": "ion", "ion": self.target_seq.upper(), "count": self.target_count}
            if self.target_ids:
                item["id"] = [self.target_ids[0]]
            return [item]
        return []

    def _validate_inputs(self) -> None:
        if self.mode == "binder" and not self.target_items:
            raise ValueError("binder mode requires valid target input (target_seq/target_seqs/ligand_*).")
        if self.mode == "unconditional" and not self.args.seq:
            raise ValueError("unconditional mode requires --seq.")
        if self.msa_max_seqs <= 0:
            raise ValueError("--msa_max_seqs must be > 0.")

        if self.binder_msa_mode == "precomputed":
            if not self.binder_msa_precomputed:
                raise ValueError("binder_msa_mode=precomputed requires --binder_msa.")
            if not os.path.isfile(self.binder_msa_precomputed):
                raise FileNotFoundError(f"binder_msa file not found: {self.binder_msa_precomputed}")
        if self.target_msa_mode == "precomputed":
            if self.target_type not in {"protein", "rna"}:
                raise ValueError("target_msa_mode=precomputed only supports protein/rna target.")
            if not self.target_msa_precomputed:
                raise ValueError("target_msa_mode=precomputed requires --target_msa.")
            if not os.path.isfile(self.target_msa_precomputed):
                raise FileNotFoundError(f"target_msa file not found: {self.target_msa_precomputed}")
        if self.target_type in {"ligand", "dna", "ion"} and self.target_msa_mode != "empty":
            raise ValueError("target_msa_mode must be empty for ligand/dna/ion target.")
        if self.target_type != "protein" and self.target_msa_mode == "mmseqs":
            raise ValueError("target_msa_mode=mmseqs currently supports protein target only.")
        for p in self.target_template_paths:
            ap = os.path.abspath(os.path.expanduser(p))
            if not os.path.isfile(ap):
                raise FileNotFoundError(f"target_template_path file not found: {ap}")
        if self.constraint_obj:
            if not isinstance(self.constraint_obj, dict):
                raise ValueError("constraint must be a JSON object.")

    def _build_constraint_object(self) -> Optional[dict]:
        cfile = (getattr(self.args, "constraint_json", "") or "").strip()
        cinline = (getattr(self.args, "constraint_inline_json", "") or "").strip()
        pocket_binder = (getattr(self.args, "pocket_binder_entity", "") or "").strip()
        pocket_res = (getattr(self.args, "pocket_contact_residues", "") or "").strip()
        contact_raw = (getattr(self.args, "contact_constraints", "") or "").strip()

        if cfile:
            p = os.path.abspath(os.path.expanduser(cfile))
            if not os.path.isfile(p):
                raise FileNotFoundError(f"constraint_json not found: {p}")
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return obj
        if cinline:
            return json.loads(cinline)

        obj: dict = {}
        if pocket_binder and pocket_res:
            binder = _parse_entity_copy(pocket_binder)
            residues = [
                _parse_entity_copy_pos(x)
                for x in [t.strip() for t in pocket_res.split(",") if t.strip()]
            ]
            obj["pocket"] = {
                "binder_chain": binder,
                "contact_residues": residues,
                "max_distance": float(getattr(self.args, "pocket_max_distance", 6.0)),
            }

        if contact_raw:
            contacts = []
            for tok in [t.strip() for t in contact_raw.split(",") if t.strip()]:
                seg = [x.strip() for x in tok.split("-") if x.strip()]
                if len(seg) < 3:
                    raise ValueError(f"Invalid contact constraint: {tok}")
                s1 = _parse_contact_atom_spec(seg[0])
                s2 = _parse_contact_atom_spec(seg[1])
                dist = [x.strip() for x in seg[2].split(":") if x.strip()]
                max_d = float(dist[0])
                min_d = float(dist[1]) if len(dist) > 1 else 0.0
                c = {
                    "entity1": s1["entity"],
                    "copy1": s1["copy"],
                    "position1": s1["position"],
                    "entity2": s2["entity"],
                    "copy2": s2["copy"],
                    "position2": s2["position"],
                    "max_distance": max_d,
                    "min_distance": min_d,
                }
                if "atom" in s1:
                    c["atom1"] = s1["atom"]
                if "atom" in s2:
                    c["atom2"] = s2["atom"]
                contacts.append(c)
            obj["contact"] = contacts

        return obj or None

    def _resolve_protein_hunter_root(self, user_root: str) -> str:
        candidates: List[str] = []
        if user_root:
            candidates.append(os.path.abspath(os.path.expanduser(user_root)))
        # Common locations before/after migration
        candidates.extend(
            [
                "/root/Protein-Hunter-server",
                "/root/Protein-Hunter",
                os.path.abspath(os.path.join(Path(__file__).resolve().parent, "..", "Protein-Hunter")),
                os.path.abspath(os.path.join(Path(__file__).resolve().parent, "..", "Protein-Hunter-server")),
            ]
        )
        for root in candidates:
            if os.path.isfile(os.path.join(root, "LigandMPNN", "wrapper.py")):
                return root
        raise FileNotFoundError(
            "Cannot locate Protein-Hunter root. Please set --protein_hunter_root to the project path "
            "that contains LigandMPNN/wrapper.py."
        )

    def _resolve_ligand_mpnn_run_py(self, user_run_py: str) -> str:
        if user_run_py:
            p = os.path.abspath(os.path.expanduser(user_run_py))
            if not os.path.isfile(p):
                raise FileNotFoundError(f"ligand_mpnn_run_py not found: {p}")
            return p
        p = os.path.join(self.protein_hunter_root, "LigandMPNN", "run.py")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"LigandMPNN run.py not found: {p}")
        return p

    def _import_ligand_mpnn_wrapper(self, protein_hunter_root: str):
        if protein_hunter_root not in sys.path:
            sys.path.insert(0, protein_hunter_root)
        try:
            module = importlib.import_module("LigandMPNN.wrapper")
            return module.LigandMPNNWrapper
        except Exception as exc:
            raise ImportError(
                f"Failed to import LigandMPNNWrapper from {protein_hunter_root}: {exc}"
            ) from exc

    def _load_binder_msa_templates(self, msa_path: str) -> List[str]:
        if not os.path.isfile(msa_path):
            raise FileNotFoundError(f"binder_msa file not found: {msa_path}")
        records: List[str] = []
        seq_buf: List[str] = []
        with open(msa_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith(">"):
                    if seq_buf:
                        records.append("".join(seq_buf))
                        seq_buf = []
                else:
                    seq_buf.append(line)
            if seq_buf:
                records.append("".join(seq_buf))
        if not records:
            raise ValueError(f"binder_msa has no sequences: {msa_path}")
        # Exclude first query row, keep alignment templates fixed.
        return records[1:]

    def _build_dynamic_binder_msa(self, binder_seq: str, cycle_idx: int, run_dir: str) -> str:
        if not self.dynamic_binder_msa_enabled:
            return self.args.binder_msa
        msa_dir = Path(run_dir) / "dynamic_msa"
        msa_dir.mkdir(parents=True, exist_ok=True)
        a3m_path = msa_dir / f"binder_cycle_{cycle_idx}.a3m"

        template_lines: List[str] = []
        if self.binder_frozen_template_sequences:
            for tpl in self.binder_frozen_template_sequences:
                if len(tpl) == len(binder_seq):
                    template_lines.append(tpl)
                elif len(tpl) > len(binder_seq):
                    template_lines.append(tpl[: len(binder_seq)])
                else:
                    template_lines.append(tpl + ("-" * (len(binder_seq) - len(tpl))))
        else:
            # Fallback keeps masked positions unconstrained in MSA.
            template_lines.append("".join("-" if aa.upper() == "X" else "X" for aa in binder_seq))

        with open(a3m_path, "w", encoding="utf-8") as f:
            f.write(f">query\n{binder_seq}\n")
            for idx, tpl in enumerate(template_lines, start=1):
                f.write(f">template_{idx}\n{tpl}\n")
        return str(a3m_path)

    def _search_msa_for_sequence(self, seq: str, out_dir: str, label: str) -> str:
        msa_root = os.path.join(out_dir, "msa_search", label)
        os.makedirs(msa_root, exist_ok=True)
        result_dirs = msa_search([seq], msa_root, mode="protenix")
        if not result_dirs:
            raise RuntimeError(f"MSA search returned empty result for {label}")
        msa_dir = result_dirs[0]
        non_pair = os.path.join(msa_dir, "non_pairing.a3m")
        if os.path.exists(non_pair):
            _truncate_a3m_file(non_pair, self.msa_max_seqs)
            return non_pair
        pairing = os.path.join(msa_dir, "pairing.a3m")
        if os.path.exists(pairing):
            _truncate_a3m_file(pairing, self.msa_max_seqs)
            return pairing
        raise RuntimeError(f"MSA search did not produce pairing/non_pairing for {label}")

    def _resolve_target_items_with_msa(self, run_dir: str) -> List[dict]:
        items = [dict(x) for x in self.target_items]
        if self.mode != "binder":
            return items
        if self.target_msa_mode == "empty":
            return items
        if self.target_msa_mode == "precomputed":
            for item in items:
                if item["type"] in {"protein", "rna"}:
                    item["msa_path"] = self.target_msa_precomputed
            return items
        # mmseqs mode
        for idx, item in enumerate(items):
            if item["type"] == "protein":
                item["msa_path"] = self._search_msa_for_sequence(
                    item["sequence"], run_dir, f"target_{idx}"
                )
        return items

    def _resolve_binder_msa_for_cycle(self, current_seq: str, cycle: int, run_dir: str) -> str:
        if self.binder_msa_mode == "empty":
            return ""
        if self.binder_msa_mode == "precomputed":
            return self._build_dynamic_binder_msa(
                binder_seq=current_seq, cycle_idx=cycle, run_dir=run_dir
            )
        return self._search_msa_for_sequence(current_seq, run_dir, f"binder_cycle_{cycle}")

    def _build_runner(self, run_dir: str):
        return get_default_runner(
            seeds=[42],
            n_cycle=1,
            n_step=2,
            n_sample=1,
            dtype="bf16",
            model_name=self.model_name,
            use_msa=self.use_msa_feature,
            use_template=self.use_template_feature,
            use_rna_msa=False,
            use_seeds_in_json=False,
            need_atom_confidence=False,
        )

    def _predict_one_json(self, runner, json_path: str, run_dir: str):
        updated_json = preprocess_input(
            json_path,
            out_dir=run_dir,
            use_msa=False,
            use_template=False,
            use_rna_msa=False,
            msa_server_mode="protenix",
        )
        configs = runner.configs
        configs["dump_dir"] = run_dir
        configs["input_json_path"] = updated_json
        infer_predict(runner, configs)

    def run(self):
        os.makedirs(self.save_dir, exist_ok=True)
        all_metrics = []
        best_row = None
        best_score = float("-inf")
        best_paths = None

        def _score_row(row: dict) -> float:
            # Binder design: prioritize iptm; Unconditional: prioritize plddt.
            if self.mode == "binder":
                return float(row.get("iptm", 0.0))
            return float(row.get("plddt", 0.0))
        
        for design_idx in range(self.num_designs):
            run_name = f"run_{design_idx}"
            run_dir = os.path.join(self.save_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)
            runner = self._build_runner(run_dir)
            resolved_target_items = self._resolve_target_items_with_msa(run_dir)
            
            seed_seq = self.args.seq
            current_seq = seed_seq
            
            for cycle in range(self.num_cycles):
                print(f"--- Design {design_idx} | Cycle {cycle} ---")
                cycle_binder_msa = self._resolve_binder_msa_for_cycle(
                    current_seq=current_seq, cycle=cycle, run_dir=run_dir
                )
                
                # 1. Build JSON input
                builder = InputDataBuilder(
                    name=f"{self.name}_{run_name}_cycle_{cycle}",
                    mode=self.mode,
                    seq=current_seq,
                    target_items=resolved_target_items,
                    binder_msa=cycle_binder_msa,
                    target_msa=self.target_msa_precomputed,
                    binder_template_path=self.args.binder_template_path,
                    template_path=self.args.template_path,
                    target_template_paths=self.target_template_paths,
                    constraint=self.constraint_obj,
                )
                json_path = os.path.join(run_dir, f"input_cycle_{cycle}.json")
                builder.write_json(json_path)
                
                # 2. Run Protenix (no auto-search, use user-provided paths)
                self._predict_one_json(runner, json_path, run_dir)
                
                # 3. Find Protenix output
                job_name = f"{self.name}_{run_name}_cycle_{cycle}"
                pred_dir = os.path.join(run_dir, job_name, "seed_42", "predictions")
                cif_files = glob.glob(os.path.join(pred_dir, "*.cif"))
                json_files = glob.glob(os.path.join(pred_dir, "*_summary_confidence_sample_*.json"))
                
                if not cif_files or not json_files:
                    print(f"Error: Protenix prediction failed at cycle {cycle}.")
                    break
                
                cif_out = cif_files[0]
                conf_json = json_files[0]
                
                with open(conf_json, 'r') as f:
                    conf_data = json.load(f)
                
                plddt = conf_data.get("plddt", 0)
                iptm = conf_data.get("iptm", 0)
                ptm = conf_data.get("ptm", 0)
                
                metrics = {
                    "design": design_idx,
                    "cycle": cycle,
                    "plddt": plddt,
                    "iptm": iptm,
                    "ptm": ptm,
                    "sequence": current_seq,
                    "cif_path": os.path.abspath(cif_out),
                    "confidence_json_path": os.path.abspath(conf_json),
                    "input_json_path": os.path.abspath(json_path),
                }
                all_metrics.append(metrics)
                
                print(f"Cycle {cycle} metrics: pLDDT={plddt:.2f}, iPTM={iptm:.2f}")

                s = _score_row(metrics)
                if s > best_score:
                    best_score = s
                    best_row = dict(metrics)
                    best_paths = (cif_out, conf_json, json_path)
                
                # 4. MPNN Sequence Design
                if cycle < self.num_cycles - 1:
                    if self.mode == "binder":
                        binder_chain = chr(ord("A") + len(resolved_target_items))
                        chains_to_design = binder_chain
                    else:
                        chains_to_design = "A"

                    fixed_residues = _build_fixed_residues(
                        chain_id=chains_to_design,
                        template_seq=seed_seq,
                        fix_all_if_no_x=getattr(self.args, "fix_all_if_no_x", False),
                    )
                    extra_args = {"--batch_size": 1}
                    if fixed_residues:
                        extra_args["--fixed_residues"] = fixed_residues

                    mpnn_seqs, _ = self.mpnn.run(
                        pdb_path=cif_out,
                        chains_to_design=chains_to_design,
                        extra_args=extra_args,
                        **self.mpnn_opts
                    )
                    
                    if mpnn_seqs:
                        seq_str = mpnn_seqs[0]
                        if self.mode == "binder":
                            chains = _split_designed_output(seq_str)
                            binder_index = len(resolved_target_items)
                            if binder_index < len(chains):
                                designed_binder = chains[binder_index]
                            else:
                                designed_binder = chains[-1]
                            current_seq = _apply_x_mask(seed_seq, designed_binder)
                        else:
                            designed_seq = _split_designed_output(seq_str)[0]
                            current_seq = _apply_x_mask(seed_seq, designed_seq)
                    else:
                        print("Warning: MPNN failed to generate sequence.")
                        
        # Save metrics once (all runs)
        df = pd.DataFrame(all_metrics)
        df.to_csv(os.path.join(self.save_dir, "summary_all_runs.csv"), index=False)
        self._plot_metrics(df, self.save_dir)

        # Save best artifacts
        if best_row and best_paths:
            best_dir = os.path.join(self.save_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            src_cif, src_conf, src_input = best_paths
            try:
                import shutil

                shutil.copy2(src_cif, os.path.join(best_dir, "best.cif"))
                shutil.copy2(src_conf, os.path.join(best_dir, "best_confidence.json"))
                shutil.copy2(src_input, os.path.join(best_dir, "best_input.json"))
            except Exception:
                # Best-effort copying; metrics CSV still records original paths.
                pass
            pd.DataFrame([best_row]).to_csv(
                os.path.join(best_dir, "summary_best.csv"), index=False
            )

    def _plot_metrics(self, df, save_dir):
        if df.empty:
            return
        plt.figure(figsize=(10, 6))
        for metric in ["plddt", "iptm", "ptm"]:
            if metric in df.columns:
                plt.plot(df["cycle"], df[metric], marker='o', label=metric.upper())
        plt.title("Protenix Pipeline Metrics")
        plt.xlabel("Cycle")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "metrics_plot.png"))
        plt.close()
