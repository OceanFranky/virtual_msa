#!/usr/bin/env python3
"""
Extract Boltz MSA → Pair signal via OuterProductMean hooks and plot vs PDB contacts.

Dependencies (if missing):
  pip install torch numpy matplotlib biopython

Example:
  /data/wenhao/miniconda3/envs/proteinhunter/bin/python \\
    scripts/extract_boltz_opm_internal_perception.py \\
    --virtual-msa test_scaffolds/8BAQ/8BAQ_mpnn1000_t0.7.npz \\
    --native-msa full_nativateMSAnew_design_boltz/8BAQ/0_protein_hunter_design/A_env/msa.a3m \\
    --pdb test_scaffolds/8BAQ/8BAQ.pdb \\
    --out test_scaffolds/8BAQ/figures/boltz_internal_perception_8BAQ.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths: boltz_ph + repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
BOLTZ_PH = REPO_ROOT / "boltz_ph"
BOLTZ_SRC = BOLTZ_PH / "src"
for p in (str(REPO_ROOT), str(BOLTZ_PH), str(BOLTZ_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from boltz.data.mol import load_canonicals  # noqa: E402
from boltz.data.parse.a3m import parse_a3m  # noqa: E402
from boltz.data.parse.schema import parse_boltz_schema  # noqa: E402
from boltz.data.types import MSA  # noqa: E402
from boltz.model.layers.outer_product_mean import OuterProductMean  # noqa: E402

from model_utils import (  # noqa: E402
    get_batch,
    get_boltz_model,
    patch_msa_first_row_to_boltz_sequence,
)

LOGGER = logging.getLogger("boltz_opm_hook")

DEFAULT_CHECKPOINT = Path("/data/wenhao/.boltz/boltz2_conf.ckpt")
DEFAULT_CCD = Path("/data/wenhao/.boltz/mols_fixed_v2")
DEFAULT_VIRTUAL_NPZ = REPO_ROOT / "test_scaffolds/8BAQ/8BAQ_mpnn1000_t0.7.npz"
DEFAULT_NATIVE_A3M = (
    REPO_ROOT
    / "full_nativateMSAnew_design_boltz/8BAQ/0_protein_hunter_design/A_env/msa.a3m"
)
DEFAULT_PDB = REPO_ROOT / "test_scaffolds/8BAQ/8BAQ.pdb"
DEFAULT_OUT = REPO_ROOT / "test_scaffolds/8BAQ/figures/boltz_internal_perception_8BAQ.png"


# ---------------------------------------------------------------------------
# Hook utilities
# ---------------------------------------------------------------------------
class OuterProductMeanCapture:
    """Collects every OuterProductMean forward output (B, L, L, C)."""

    def __init__(self) -> None:
        self.outputs: list[torch.Tensor] = []
        self.hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    def clear(self) -> None:
        self.outputs.clear()

    def hook_fn(self, _module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        self.outputs.append(output.detach().cpu())

    def register(self, model: torch.nn.Module) -> int:
        self.remove()
        n = 0
        for name, module in model.named_modules():
            if isinstance(module, OuterProductMean):
                h = module.register_forward_hook(self.hook_fn)
                self.hook_handles.append(h)
                LOGGER.debug("Hooked OuterProductMean: %s", name)
                n += 1
        return n

    def remove(self) -> None:
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def last_output(self) -> Optional[torch.Tensor]:
        """Last OPM output = final MSA block's pair injection tensor."""
        return self.outputs[-1] if self.outputs else None


def find_msa_module(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "msa_module"):
        msa = model.msa_module
        if hasattr(msa, "_orig_mod"):
            return msa._orig_mod
        return msa
    raise RuntimeError("Could not locate model.msa_module on Boltz checkpoint.")


# ---------------------------------------------------------------------------
# Trunk forward (MSA module only, eval, no grad)
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_trunk_through_msa(
    model: torch.nn.Module,
    feats: dict[str, torch.Tensor],
    *,
    use_templates: bool = True,
) -> None:
    """
    Mirror Boltz2.forward trunk through msa_module (recycling_steps=0).

    OuterProductMean hooks fire inside each MSALayer; we read the last capture.
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

    s = s_init + model.s_recycle(model.s_norm(s))
    z = z_init + model.z_recycle(model.z_norm(z))

    if use_templates and getattr(model, "use_templates", False):
        template_module = model.template_module
        if getattr(model, "is_template_compiled", False) and not model.training:
            template_module = template_module._orig_mod
        z = z + template_module(z, feats, pair_mask, use_kernels=model.use_kernels)

    msa_module = find_msa_module(model)
    z = z + msa_module(z, s_inputs, feats, use_kernels=model.use_kernels)


# ---------------------------------------------------------------------------
# Feature processing & plotting
# ---------------------------------------------------------------------------
def tensor_to_residue_map(
    opm: torch.Tensor,
    token_indices: np.ndarray,
) -> np.ndarray:
    """
    OPM shape (B, N, N, C) → (L, L) via L2 on C and min-max to [0, 1].
    """
    if opm.dim() != 4:
        raise ValueError(f"Expected 4D OPM tensor, got shape {tuple(opm.shape)}")
    t = opm[0, token_indices][:, token_indices, :]  # (L, L, C)
    mat = torch.norm(t, p=2, dim=-1).numpy()
    lo, hi = float(mat.min()), float(mat.max())
    if hi - lo < 1e-12:
        return np.zeros_like(mat)
    return (mat - lo) / (hi - lo)


def compute_contact_map(pdb_path: Path, distance_cutoff: float = 8.0) -> np.ndarray:
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import is_aa
    except ImportError as exc:
        raise ImportError("biopython required: pip install biopython") from exc

    structure = PDBParser(QUIET=True).get_structure("x", str(pdb_path))
    chain = list(structure[0].get_chains())[0]
    residues = [r for r in chain if is_aa(r, standard=True)]
    L = len(residues)
    coords = np.zeros((L, 3), dtype=np.float64)
    for i, res in enumerate(residues):
        if res.get_resname() == "GLY":
            coords[i] = res["CA"].get_coord()
        else:
            atom = "CB" if "CB" in res else "CA"
            coords[i] = res[atom].get_coord()
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    contacts = (dist < distance_cutoff).astype(np.float64)
    np.fill_diagonal(contacts, 0.0)
    return contacts


def plot_split_panel(
    ax: plt.Axes,
    perception: np.ndarray,
    contacts: np.ndarray,
    title: str,
    cbar_label: str = "Boltz internal score",
) -> None:
    """
    Upper-left triangle: model perception (i > j in matrix / top-left visually).
    Lower-right triangle: PDB contact map (i < j).
    """
    L = perception.shape[0]
    if contacts.shape[0] != L:
        n = min(L, contacts.shape[0])
        LOGGER.warning(
            "Trimming maps to min(L): perception=%d, contacts=%d → %d",
            L,
            contacts.shape[0],
            n,
        )
        perception = perception[:n, :n]
        contacts = contacts[:n, :n]
        L = n

    upper_mask = np.tril(np.ones((L, L), dtype=bool), k=0)  # hide i >= j
    lower_mask = np.triu(np.ones((L, L), dtype=bool), k=0)  # hide i <= j

    perc_m = np.ma.masked_array(perception, mask=upper_mask)
    cont_m = np.ma.masked_array(contacts.astype(float), mask=lower_mask)

    im_p = ax.imshow(
        perc_m,
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
    return im_p


# ---------------------------------------------------------------------------
# Data / MSA helpers
# ---------------------------------------------------------------------------
def sequence_from_pdb(pdb_path: Path, chain_id: str = "A") -> str:
    from utils.metrics import get_CA_and_sequence

    _, seq = get_CA_and_sequence(str(pdb_path), chain_id)
    return seq


def ensure_npz_msa(
    msa_path: Path,
    query_sequence: str,
    cache_dir: Path,
    max_seqs: int,
) -> Path:
    """Accept .npz or convert .a3m → .npz for Boltz featurizer."""
    msa_path = msa_path.resolve()
    if msa_path.suffix == ".npz":
        return msa_path
    if msa_path.suffix != ".a3m":
        raise ValueError(f"MSA must be .npz or .a3m: {msa_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{msa_path.stem}_max{max_seqs}.npz"
    if out.is_file():
        LOGGER.info("Using cached NPZ: %s", out)
        return out

    LOGGER.info("Converting A3M → NPZ (max_seqs=%d): %s", max_seqs, msa_path)
    msa = parse_a3m(msa_path, taxonomy=None, max_seqs=max_seqs)
    msa = patch_msa_first_row_to_boltz_sequence(msa, query_sequence)
    msa.dump(out)
    return out


def build_batch(
    sequence: str,
    msa_npz: Path,
    *,
    name: str,
    ccd_path: Path,
    ccd_lib: dict,
    msa_max_seqs: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    data = {
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": sequence,
                    "msa": str(msa_npz.resolve()),
                }
            }
        ]
    }
    target = parse_boltz_schema(name, data, ccd_lib, ccd_path, boltz_2=True)
    batch, _structure = get_batch(
        target,
        ccd_path,
        ccd_lib,
        max_seqs=msa_max_seqs,
        boltz_model_version="boltz2",
    )
    return {k: v.unsqueeze(0).to(device) for k, v in batch.items()}


def protein_token_indices(feats: dict[str, torch.Tensor]) -> np.ndarray:
    """Indices of padded protein tokens (for slicing OPM maps)."""
    mask = feats["token_pad_mask"].bool()[0].cpu().numpy()
    return np.where(mask)[0]


@torch.no_grad()
def run_one_msa_condition(
    model: torch.nn.Module,
    capture: OuterProductMeanCapture,
    sequence: str,
    msa_path: Path,
    *,
    name: str,
    ccd_path: Path,
    ccd_lib: dict,
    msa_max_seqs: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)

    cache_dir = REPO_ROOT / "test_scaffolds" / ".msa_npz_cache"
    msa_npz = ensure_npz_msa(msa_path, sequence, cache_dir, msa_max_seqs)

    batch = build_batch(
        sequence,
        msa_npz,
        name=name,
        ccd_path=ccd_path,
        ccd_lib=ccd_lib,
        msa_max_seqs=msa_max_seqs,
        device=device,
    )

    capture.clear()
    n_hooks = capture.register(model)
    if n_hooks == 0:
        raise RuntimeError("No OuterProductMean modules found — check Boltz version.")

    try:
        forward_trunk_through_msa(model, batch, use_templates=True)
    finally:
        capture.remove()

    opm = capture.last_output()
    if opm is None:
        raise RuntimeError("Hook did not capture OuterProductMean output.")

    LOGGER.info(
        "Captured OPM %s: shape=%s (hooks=%d)",
        msa_path.name,
        tuple(opm.shape),
        n_hooks,
    )
    idx = protein_token_indices(batch)
    return tensor_to_residue_map(opm, idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Boltz OuterProductMean internal perception vs PDB contacts."
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--ccd-path", type=Path, default=DEFAULT_CCD)
    p.add_argument("--virtual-msa", type=Path, default=DEFAULT_VIRTUAL_NPZ)
    p.add_argument("--native-msa", type=Path, default=DEFAULT_NATIVE_A3M)
    p.add_argument("--pdb", type=Path, default=DEFAULT_PDB)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--msa-max-seqs", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.checkpoint.is_file():
        LOGGER.error("Checkpoint not found: %s", args.checkpoint)
        return 1
    if not args.pdb.is_file():
        LOGGER.error("PDB not found: %s", args.pdb)
        return 1

    device = torch.device(args.device)
    LOGGER.info("Device: %s", device)

    predict_args = {
        "recycling_steps": 0,
        "sampling_steps": 1,
        "diffusion_samples": 1,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    LOGGER.info("Loading Boltz2 from %s", args.checkpoint)
    model = get_boltz_model(
        checkpoint=str(args.checkpoint),
        predict_args=predict_args,
        device=str(device),
        model_version="boltz2",
        grad_enabled=False,
        no_potentials=True,
        msa_max_seqs=args.msa_max_seqs,
    )
    model.eval()
    model.to(device)

    ccd_lib = load_canonicals(str(args.ccd_path))
    sequence = sequence_from_pdb(args.pdb)
    LOGGER.info("Query sequence length (PDB chain A): %d", len(sequence))

    contacts = compute_contact_map(args.pdb)
    capture = OuterProductMeanCapture()

    LOGGER.info("=== Virtual MSA forward ===")
    perc_virtual = run_one_msa_condition(
        model,
        capture,
        sequence,
        args.virtual_msa,
        name="8BAQ_virtual",
        ccd_path=args.ccd_path,
        ccd_lib=ccd_lib,
        msa_max_seqs=args.msa_max_seqs,
        device=device,
        seed=args.seed,
    )

    LOGGER.info("=== Native MSA forward ===")
    perc_native = run_one_msa_condition(
        model,
        capture,
        sequence,
        args.native_msa,
        name="8BAQ_native",
        ccd_path=args.ccd_path,
        ccd_lib=ccd_lib,
        msa_max_seqs=args.msa_max_seqs,
        device=device,
        seed=args.seed + 1,
    )

    # Plot
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    im0 = plot_split_panel(
        axes[0],
        perc_virtual,
        contacts,
        "A  Virtual MSA (MPNN)\nUpper-left: OPM L2 norm | Lower-right: PDB <8Å",
    )
    plot_split_panel(
        axes[1],
        perc_native,
        contacts,
        "B  Native MSA (UniRef)\nUpper-left: OPM L2 norm | Lower-right: PDB <8Å",
    )

    fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Min–max norm. score")
    fig.suptitle(
        "Boltz2 internal coevolution signal (OuterProductMean, pre-pair injection)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    LOGGER.info("Saved %s", args.out)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
