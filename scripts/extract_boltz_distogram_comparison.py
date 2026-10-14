#!/usr/bin/env python3
"""
Boltz2 distogram logits → P(contact < 8 Å) comparison: virtual MSA vs single-sequence (no MSA).

Dependencies:
  pip install torch numpy matplotlib biopython

Example:
  /data/wenhao/miniconda3/envs/proteinhunter/bin/python \\
    scripts/extract_boltz_distogram_comparison.py \\
    --virtual-msa test_scaffolds/8BAQ/8BAQ_mpnn1000_t0.7.npz \\
    --pdb test_scaffolds/8BAQ/8BAQ.pdb \\
    --out /data/wenhao/Protein-Hunter-server/boltz_distogram_comparison_8BAQ.png
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BOLTZ_PH = REPO_ROOT / "boltz_ph"
BOLTZ_SRC = BOLTZ_PH / "src"
for p in (str(REPO_ROOT), str(BOLTZ_PH), str(BOLTZ_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from boltz.data.mol import load_canonicals  # noqa: E402
from boltz.data.parse.a3m import parse_a3m  # noqa: E402
from boltz.data.parse.schema import parse_boltz_schema  # noqa: E402
from model_utils import get_batch, get_boltz_model, patch_msa_first_row_to_boltz_sequence  # noqa: E402
from boltz.data.types import MSA  # noqa: E402

LOGGER = logging.getLogger("boltz_distogram")

DEFAULT_CHECKPOINT = Path("/data/wenhao/.boltz/boltz2_conf.ckpt")
DEFAULT_CCD = Path("/data/wenhao/.boltz/mols_fixed_v2")
DEFAULT_VIRTUAL_NPZ = REPO_ROOT / "test_scaffolds/8BAQ/8BAQ_mpnn1000_t0.7.npz"
DEFAULT_A3M = REPO_ROOT / "test_scaffolds/8BAQ/8BAQ_mpnn1000_t0.7.a3m"
DEFAULT_PDB = REPO_ROOT / "test_scaffolds/8BAQ/8BAQ.pdb"
DEFAULT_NATIVE_A3M = (
    REPO_ROOT
    / "full_nativateMSAnew_design_boltz/8BAQ/0_protein_hunter_design/A_env/msa.a3m"
)
DEFAULT_OUT = REPO_ROOT / "boltz_distogram_comparison_8BAQ.png"
DEFAULT_OUT_NATIVE = REPO_ROOT / "boltz_distogram_comparison_8BAQ_virtual_vs_native.png"

CONTACT_CUTOFF_A = 8.0


# ---------------------------------------------------------------------------
# Distogram capture (hook fallback)
# ---------------------------------------------------------------------------
class DistogramCapture:
    def __init__(self) -> None:
        self.logits: Optional[torch.Tensor] = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook(self, _m: torch.nn.Module, _inp: Any, out: torch.Tensor) -> None:
        self.logits = out.detach().cpu()

    def register(self, model: torch.nn.Module) -> None:
        self.remove()
        mod = model.distogram_module
        if hasattr(mod, "_orig_mod"):
            mod = mod._orig_mod
        self._handles.append(mod.register_forward_hook(self.hook))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Trunk forward → pdistogram (before diffusion)
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_trunk_to_distogram(
    model: torch.nn.Module,
    feats: dict[str, torch.Tensor],
    *,
    recycling_steps: int = 0,
    use_templates: bool = False,
) -> torch.Tensor:
    """
    Run Boltz2 trunk: embedder → [template] → MSA → pairformer → distogram_module.

    Returns pdistogram logits (B, L, L, num_distograms, num_bins).
    """
    model.eval()

    s_inputs = model.input_embedder(feats)
    s_init = model.s_init(s_inputs)

    z_init = model.z_init_1(s_inputs)[:, :, None] + model.z_init_2(s_inputs)[:, None, :]
    z_init = z_init + model.rel_pos(feats)
    z_init = z_init + model.token_bonds(feats["token_bonds"].float())
    if getattr(model, "bond_type_feature", False):
        z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())
    z_init = z_init + model.contact_conditioning(feats)

    s = torch.zeros_like(s_init)
    z = torch.zeros_like(z_init)
    mask = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    msa_module = model.msa_module
    if getattr(model, "is_msa_compiled", False) and not model.training:
        msa_module = msa_module._orig_mod

    pairformer_module = model.pairformer_module
    if getattr(model, "is_pairformer_compiled", False) and not model.training:
        pairformer_module = pairformer_module._orig_mod

    template_module = None
    if use_templates and getattr(model, "use_templates", False):
        template_module = model.template_module
        if getattr(model, "is_template_compiled", False) and not model.training:
            template_module = template_module._orig_mod

    for i in range(recycling_steps + 1):
        s = s_init + model.s_recycle(model.s_norm(s))
        z = z_init + model.z_recycle(model.z_norm(z))

        if template_module is not None:
            z = z + template_module(z, feats, pair_mask, use_kernels=model.use_kernels)

        z = z + msa_module(z, s_inputs, feats, use_kernels=model.use_kernels)
        s, z = pairformer_module(
            s, z, mask=mask, pair_mask=pair_mask, use_kernels=model.use_kernels
        )

    distogram_module = model.distogram_module
    if hasattr(distogram_module, "_orig_mod"):
        distogram_module = distogram_module._orig_mod
    return distogram_module(z)


# ---------------------------------------------------------------------------
# Logits → P(d < 8 Å)
# ---------------------------------------------------------------------------
def contact_bin_indices(
    min_dist: float, max_dist: float, num_bins: int, cutoff: float
) -> list[int]:
    """
    Boltz assigns bin k via (d > boundary).sum(); bin k spans (boundary[k-1], boundary[k]].

    Sum bins whose lower edge is strictly below `cutoff` → mass for d < cutoff Å.
    """
    boundaries = np.linspace(min_dist, max_dist, num_bins - 1)
    keep: list[int] = []
    for k in range(num_bins):
        low = float(boundaries[k - 1]) if k > 0 else min_dist
        if low < cutoff - 1e-6:
            keep.append(k)
    return keep


def logits_to_contact_probability(
    logits: torch.Tensor,
    min_dist: float,
    max_dist: float,
    num_bins: int,
    cutoff: float = CONTACT_CUTOFF_A,
) -> np.ndarray:
    """(L, L, bins) or (L, L, D, bins) → (L, L) contact probability in [0, 1]."""
    x = logits.float()
    if x.dim() == 4:
        x = x.squeeze(2)  # num_distograms
    if x.dim() != 3:
        raise ValueError(f"Expected 3D logits after squeeze, got {tuple(x.shape)}")

    probs = torch.softmax(x, dim=-1)
    keep = contact_bin_indices(min_dist, max_dist, num_bins, cutoff)
    LOGGER.info(
        "Contact bins: %d/%d (%.2f–%.2f Å, cutoff %.1f Å, width %.4f Å)",
        len(keep),
        num_bins,
        min_dist,
        max_dist,
        cutoff,
        (max_dist - min_dist) / num_bins,
    )
    contact = probs[..., keep].sum(dim=-1).clamp(0.0, 1.0)
    return contact.cpu().numpy()


def slice_to_protein_tokens(
    matrix: np.ndarray, token_indices: np.ndarray
) -> np.ndarray:
    idx = token_indices
    return matrix[np.ix_(idx, idx)]


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------
def read_first_sequence_a3m(a3m_path: Path) -> str:
    lines = a3m_path.read_text().splitlines()
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"Invalid A3M: {a3m_path}")
    i = 1
    chunks: list[str] = []
    while i < len(lines) and not lines[i].startswith(">"):
        chunks.append(lines[i].strip())
        i += 1
    return re.sub(r"[a-z]", "", "".join(chunks))


def build_monomer_data(
    sequence: str,
    msa: str | Path,
) -> dict:
    """
    Build Boltz YAML-equivalent dict.

    msa:
      - path to .npz  → deep MSA
      - "empty"       → single-sequence mode (msa_id = -1, no homologs)
    """
    protein: dict = {"id": ["A"], "sequence": sequence}
    if str(msa) != "empty":
        protein["msa"] = str(Path(msa).resolve())
    else:
        protein["msa"] = "empty"
    return {"sequences": [{"protein": protein}]}


def ensure_msa_npz(
    msa_path: Path,
    sequence: str,
    max_seqs: int,
) -> Path:
    """Resolve .npz or convert .a3m → .npz (Boltz featurizer input)."""
    msa_path = msa_path.resolve()
    if msa_path.suffix == ".npz":
        msa_obj = MSA.load(msa_path)
        msa_obj = patch_msa_first_row_to_boltz_sequence(msa_obj, sequence)
        cache = REPO_ROOT / "test_scaffolds/.msa_npz_cache"
        cache.mkdir(parents=True, exist_ok=True)
        out = cache / f"{msa_path.stem}_row0_patched.npz"
        msa_obj.dump(out)
        return out

    if msa_path.suffix != ".a3m":
        raise ValueError(f"MSA must be .npz or .a3m: {msa_path}")

    cache = REPO_ROOT / "test_scaffolds/.msa_npz_cache"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{msa_path.stem}_max{max_seqs}.npz"
    if not out.is_file():
        LOGGER.info("Converting A3M → NPZ (max_seqs=%d): %s", max_seqs, msa_path)
        msa_obj = parse_a3m(msa_path, taxonomy=None, max_seqs=max_seqs)
        msa_obj = patch_msa_first_row_to_boltz_sequence(msa_obj, sequence)
        msa_obj.dump(out)
    else:
        LOGGER.info("Using cached NPZ: %s", out)
    return out


def make_batch(
    sequence: str,
    msa: str | Path,
    *,
    name: str,
    ccd_path: Path,
    ccd_lib: dict,
    msa_max_seqs: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if str(msa) != "empty":
        msa = ensure_msa_npz(Path(msa), sequence, msa_max_seqs)

    data = build_monomer_data(sequence, msa)
    target = parse_boltz_schema(name, data, ccd_lib, ccd_path, boltz_2=True)
    batch, _ = get_batch(
        target,
        ccd_path,
        ccd_lib,
        max_seqs=msa_max_seqs,
        boltz_model_version="boltz2",
    )
    return {k: v.unsqueeze(0).to(device) for k, v in batch.items()}


def protein_token_indices(feats: dict[str, torch.Tensor]) -> np.ndarray:
    return np.where(feats["token_pad_mask"].bool()[0].cpu().numpy())[0]


# ---------------------------------------------------------------------------
# PDB contacts (baseline, lower-right triangle)
# ---------------------------------------------------------------------------
def pdb_contact_map(pdb_path: Path, cutoff: float = CONTACT_CUTOFF_A) -> np.ndarray:
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import is_aa
    except ImportError as exc:
        raise ImportError("pip install biopython") from exc

    chain = list(PDBParser(QUIET=True).get_structure("x", str(pdb_path))[0].get_chains())[0]
    residues = [r for r in chain if is_aa(r, standard=True)]
    L = len(residues)
    coords = np.zeros((L, 3))
    for i, res in enumerate(residues):
        if res.get_resname() == "GLY":
            coords[i] = res["CA"].get_coord()
        else:
            coords[i] = res["CB" if "CB" in res else "CA"].get_coord()
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    c = (dist < cutoff).astype(np.float64)
    np.fill_diagonal(c, 0.0)
    return c


# ---------------------------------------------------------------------------
# Plotting (upper-left = prediction, lower-right = PDB)
# ---------------------------------------------------------------------------
def plot_split_panel(
    ax: plt.Axes,
    prob: np.ndarray,
    contacts: np.ndarray,
    title: str,
) -> None:
    L = prob.shape[0]
    if contacts.shape[0] != L:
        n = min(L, contacts.shape[0])
        prob, contacts = prob[:n, :n], contacts[:n, :n]
        L = n

    # Upper-left visual: i > j  |  Lower-right visual: i < j
    prob_m = np.ma.masked_array(prob, mask=np.triu(np.ones((L, L)), k=0))
    cont_m = np.ma.masked_array(contacts, mask=np.tril(np.ones((L, L)), k=0))

    im = ax.imshow(
        prob_m,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    ax.imshow(
        cont_m,
        origin="lower",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    ax.plot([0, L - 1], [0, L - 1], color="white", lw=0.6, alpha=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")
    step = max(1, L // 8)
    ticks = np.arange(0, L, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(int(t + 1)) for t in ticks], fontsize=7)
    ax.set_yticklabels([str(int(t + 1)) for t in ticks], fontsize=7)
    return im


@torch.no_grad()
def run_condition(
    model: torch.nn.Module,
    sequence: str,
    msa: str | Path,
    *,
    name: str,
    ccd_path: Path,
    ccd_lib: dict,
    msa_max_seqs: int,
    device: torch.device,
    recycling_steps: int,
    use_templates: bool,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch = make_batch(
        sequence,
        msa,
        name=name,
        ccd_path=ccd_path,
        ccd_lib=ccd_lib,
        msa_max_seqs=msa_max_seqs,
        device=device,
    )
    idx = protein_token_indices(batch)

    capture = DistogramCapture()
    capture.register(model)
    try:
        logits = forward_trunk_to_distogram(
            model,
            batch,
            recycling_steps=recycling_steps,
            use_templates=use_templates,
        )
    finally:
        capture.remove()

    if logits is None and capture.logits is not None:
        logits = capture.logits.to(device)

    LOGGER.info("Distogram logits shape: %s (condition=%s)", tuple(logits.shape), name)

    logits_np = logits[0].cpu()
    prob = logits_to_contact_probability(
        logits_np[idx][:, idx, :],
        min_dist=float(model.min_dist),
        max_dist=float(model.max_dist),
        num_bins=int(model.num_bins),
    )
    return prob


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Boltz distogram: virtual MSA vs no-MSA or native MSA."
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--ccd-path", type=Path, default=DEFAULT_CCD)
    p.add_argument("--virtual-msa", type=Path, default=DEFAULT_VIRTUAL_NPZ)
    p.add_argument(
        "--native-msa",
        type=Path,
        default=None,
        help="If set, panel B uses native MSA (.a3m/.npz) instead of empty/single-seq.",
    )
    p.add_argument(
        "--compare",
        choices=("no-msa", "native"),
        default=None,
        help="B panel: no-msa (default) or native (requires --native-msa).",
    )
    p.add_argument("--a3m", type=Path, default=DEFAULT_A3M, help="Query sequence (1st row)")
    p.add_argument("--pdb", type=Path, default=DEFAULT_PDB)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--msa-max-seqs", type=int, default=500)
    p.add_argument("--recycling-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    compare_native = args.compare == "native" or (
        args.compare is None and args.native_msa is not None
    )
    if compare_native and args.native_msa is None:
        args.native_msa = DEFAULT_NATIVE_A3M

    if args.out is None:
        args.out = (
            DEFAULT_OUT_NATIVE if compare_native else DEFAULT_OUT
        )

    for path, label in [
        (args.checkpoint, "checkpoint"),
        (args.virtual_msa, "virtual MSA"),
        (args.pdb, "PDB"),
    ]:
        if not path.is_file():
            LOGGER.error("%s not found: %s", label, path)
            return 1
    if compare_native:
        if not args.native_msa.is_file():
            LOGGER.error("native MSA not found: %s", args.native_msa)
            return 1

    device = torch.device(args.device)
    sequence = read_first_sequence_a3m(args.a3m)
    LOGGER.info("Query sequence length: %d (from %s)", len(sequence), args.a3m.name)

    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": 1,
        "diffusion_samples": 1,
        "max_parallel_samples": 1,
    }

    LOGGER.info("Loading Boltz2: %s", args.checkpoint)
    model = get_boltz_model(
        checkpoint=str(args.checkpoint),
        predict_args=predict_args,
        device=str(device),
        model_version="boltz2",
        grad_enabled=False,
        no_potentials=True,
        msa_max_seqs=args.msa_max_seqs,
    )
    model.eval().to(device)

    ccd_lib = load_canonicals(str(args.ccd_path))
    contacts = pdb_contact_map(args.pdb)

    LOGGER.info("=== Virtual MSA (precomputed .npz) ===")
    prob_virtual = run_condition(
        model,
        sequence,
        args.virtual_msa,
        name="virtual_msa",
        ccd_path=args.ccd_path,
        ccd_lib=ccd_lib,
        msa_max_seqs=args.msa_max_seqs,
        device=device,
        recycling_steps=args.recycling_steps,
        use_templates=False,
        seed=args.seed,
    )

    if compare_native:
        LOGGER.info("=== Native MSA (%s) ===", args.native_msa)
        prob_b = run_condition(
            model,
            sequence,
            args.native_msa,
            name="native_msa",
            ccd_path=args.ccd_path,
            ccd_lib=ccd_lib,
            msa_max_seqs=args.msa_max_seqs,
            device=device,
            recycling_steps=args.recycling_steps,
            use_templates=False,
            seed=args.seed + 1,
        )
        title_b = (
            "B  Native MSA (UniRef)\n"
            "Upper-left: P(d < 8 Å) | Lower-right: PDB contact"
        )
    else:
        LOGGER.info("=== No MSA (msa='empty' single-sequence mode) ===")
        prob_b = run_condition(
            model,
            sequence,
            "empty",
            name="no_msa",
            ccd_path=args.ccd_path,
            ccd_lib=ccd_lib,
            msa_max_seqs=args.msa_max_seqs,
            device=device,
            recycling_steps=args.recycling_steps,
            use_templates=False,
            seed=args.seed + 1,
        )
        title_b = (
            "B  No MSA (single sequence)\n"
            "Upper-left: P(d < 8 Å) | Lower-right: PDB contact"
        )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
        }
    )
    # 三列布局：两幅热图 + 独立 colorbar 列，避免竖条压住右侧 No MSA 子图
    fig = plt.figure(figsize=(12.0, 5.2))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 0.045],
        wspace=0.38,
        left=0.07,
        right=0.98,
        top=0.88,
        bottom=0.12,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    im0 = plot_split_panel(
        ax_a,
        prob_virtual,
        contacts,
        "A  Virtual MSA\nUpper-left: P(d < 8 Å) | Lower-right: PDB contact",
    )
    plot_split_panel(ax_b, prob_b, contacts, title_b)

    cbar = fig.colorbar(
        im0,
        cax=cax,
        label="Predicted contact probability (< 8 Å)",
    )
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Boltz2 internal geometry belief (distogram logits, pre-diffusion)",
        fontsize=11,
        y=0.98,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    LOGGER.info("Saved %s", args.out)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
