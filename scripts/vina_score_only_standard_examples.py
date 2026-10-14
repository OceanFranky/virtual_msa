#!/usr/bin/env python3
"""Direct Vina --score_only (autobox) for standard_examples designs and native controls."""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

VINA = Path("/data/wenhao/miniconda3/envs/proteinhunter/bin/vina")
OBABEL = Path("/data/wenhao/miniconda3/envs/proteinhunter/bin/obabel")

NATIVE_ROOT = Path("/data/wenhao/Protein-Hunter-server/standard_examples_5.30")
DESIGN_ROOT = Path("/data/wenhao/Protein-Hunter-server/standard_examples_5.30_design")
OUT_CSV = DESIGN_ROOT / "vina_score_only_direct.csv"

BINDING_RE = re.compile(
    r"Estimated Free Energy of Binding\s*:\s*([-\d.]+)\s*\(kcal/mol\)",
    re.I,
)
LIGAND_CCD_RE = re.compile(r"--ligand_ccd\s+(\S+)")
SKIP_RESNAMES = {"HOH", "WAT", "DOD", "SO4", "PO4", "CL", "NA", "MG", "ZN", "CA"}
DESIGN_LIG_RESNAME = "LIG"


def parse_ligand_ccd(case_dir: Path) -> str:
    readme = case_dir / "README.md"
    text = readme.read_text(errors="replace")
    m = LIGAND_CCD_RE.search(text)
    if not m:
        raise ValueError(f"No --ligand_ccd in {readme}")
    return m.group(1).strip()


def split_complex_pdb(
    pdb_path: Path,
    protein_pdb: Path,
    ligand_pdb: Path,
    ligand_resnames: set[str],
) -> tuple[int, int]:
    protein_lines: list[str] = []
    ligand_lines: list[str] = []
    with pdb_path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith("ATOM"):
                protein_lines.append(line.rstrip("\n"))
            elif line.startswith("HETATM"):
                res = line[17:20].strip()
                if res in SKIP_RESNAMES:
                    continue
                if res in ligand_resnames:
                    ligand_lines.append(line.rstrip("\n"))
                else:
                    protein_lines.append(line.rstrip("\n"))
    protein_pdb.write_text("\n".join(protein_lines) + "\nEND\n")
    ligand_pdb.write_text("\n".join(ligand_lines) + "\nEND\n")
    return len(protein_lines), len(ligand_lines)


def pdb_to_pdbqt(pdb: Path, pdbqt: Path, rigid_receptor: bool = False) -> None:
    cmd = [str(OBABEL), "-ipdb", str(pdb), "-opdbqt", "-O", str(pdbqt), "-h", "--partialcharge", "gasteiger"]
    if rigid_receptor:
        cmd.insert(-1, "-xr")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not pdbqt.is_file():
        raise RuntimeError(f"obabel failed: {(result.stderr or result.stdout)[-500:]}")


def vina_score_only(rec_pdbqt: Path, lig_pdbqt: Path, work: Path) -> float:
    conf = work / "vina.conf"
    conf.write_text(
        "\n".join(
            [
                f"receptor = {rec_pdbqt}",
                f"ligand = {lig_pdbqt}",
                "scoring = vina",
                "score_only = true",
                "autobox = true",
                "verbosity = 1",
                "",
            ]
        )
    )
    result = subprocess.run(
        [str(VINA), "--config", str(conf)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(work),
    )
    text = (result.stdout or "") + (result.stderr or "")
    match = BINDING_RE.search(text)
    if not match:
        raise RuntimeError(f"vina parse failed (rc={result.returncode}): {text[-800:]}")
    return float(match.group(1))


def score_complex(
    pdb_path: Path,
    ligand_resnames: set[str],
    sample_type: str,
    group: str,
    case_id: str,
    run_id: str,
) -> dict[str, object]:
    work = Path(tempfile.mkdtemp(prefix="vina_std_"))
    try:
        protein_pdb = work / "protein.pdb"
        ligand_pdb = work / "ligand.pdb"
        n_prot, n_lig = split_complex_pdb(pdb_path, protein_pdb, ligand_pdb, ligand_resnames)
        if n_lig == 0:
            raise ValueError(f"No ligand atoms for resnames={sorted(ligand_resnames)} in {pdb_path}")
        if n_prot == 0:
            raise ValueError(f"No protein atoms in {pdb_path}")

        rec_pdbqt = work / "rec.pdbqt"
        lig_pdbqt = work / "lig.pdbqt"
        pdb_to_pdbqt(protein_pdb, rec_pdbqt, rigid_receptor=True)
        pdb_to_pdbqt(ligand_pdb, lig_pdbqt, rigid_receptor=False)
        affinity = vina_score_only(rec_pdbqt, lig_pdbqt, work)
        return {
            "sample_type": sample_type,
            "group": group,
            "case": case_id,
            "run_id": run_id,
            "pdb_path": str(pdb_path),
            "ligand_resnames": ",".join(sorted(ligand_resnames)),
            "n_protein_atoms": n_prot,
            "n_ligand_atoms": n_lig,
            "vina_score_kcal_mol": affinity,
            "status": "ok",
            "error": "",
        }
    except Exception as exc:
        return {
            "sample_type": sample_type,
            "group": group,
            "case": case_id,
            "run_id": run_id,
            "pdb_path": str(pdb_path),
            "ligand_resnames": ",".join(sorted(ligand_resnames)),
            "n_protein_atoms": "",
            "n_ligand_atoms": "",
            "vina_score_kcal_mol": "",
            "status": "fail",
            "error": str(exc)[:500],
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def collect_jobs() -> list[tuple]:
    jobs: list[tuple] = []
    case_dirs = sorted([p for p in NATIVE_ROOT.iterdir() if p.is_dir() and (p / "README.md").exists()])
    ligand_map = {d.name: parse_ligand_ccd(d) for d in case_dirs}

    for case_id, ligand_ccd in ligand_map.items():
        native_pdb = NATIVE_ROOT / case_id / f"{case_id}_clean.pdb"
        if native_pdb.is_file():
            jobs.append(
                (
                    native_pdb,
                    {ligand_ccd},
                    "native_wt",
                    "wt",
                    case_id,
                    "native",
                )
            )

    for group in ("no_msa", "with_msa"):
        group_dir = DESIGN_ROOT / group
        if not group_dir.is_dir():
            continue
        for case_id in ligand_map:
            case_dir = group_dir / case_id
            if not case_dir.is_dir():
                continue
            for pdb_path in sorted(case_dir.glob("0_protein_hunter_design/run_*/*_best_structure.pdb")):
                run_match = re.search(r"/run_(\d+)/", str(pdb_path))
                run_id = run_match.group(1) if run_match else ""
                jobs.append(
                    (
                        pdb_path,
                        {DESIGN_LIG_RESNAME},
                        "design",
                        group,
                        case_id,
                        run_id,
                    )
                )
    return jobs


def main() -> None:
    if not VINA.is_file() or not OBABEL.is_file():
        raise SystemExit(f"Missing vina/obabel: {VINA} {OBABEL}")

    jobs = collect_jobs()
    print(f"Collected {len(jobs)} scoring jobs")

    rows: list[dict[str, object]] = []
    workers = min(16, max(1, len(jobs)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(score_complex, *job): job for job in jobs
        }
        done = 0
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"Finished {done}/{len(jobs)}")

    fieldnames = [
        "sample_type",
        "group",
        "case",
        "run_id",
        "pdb_path",
        "ligand_resnames",
        "n_protein_atoms",
        "n_ligand_atoms",
        "vina_score_kcal_mol",
        "status",
        "error",
    ]
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = DESIGN_ROOT / "vina_score_only_direct_summary.csv"
    native_rows = [r for r in rows if r["sample_type"] == "native_wt" and r["status"] == "ok"]
    design_rows = [r for r in rows if r["sample_type"] == "design" and r["status"] == "ok"]
    print(f"Wrote {OUT_CSV}")
    print(f"ok native: {len(native_rows)}, ok design: {len(design_rows)}, fail: {len(rows) - len(native_rows) - len(design_rows)}")

    # per-case summary: wt vs design means
    by_key: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (row["group"], row["case"])
        by_key.setdefault(key, []).append(float(row["vina_score_kcal_mol"]))

    wt_scores = {r["case"]: float(r["vina_score_kcal_mol"]) for r in native_rows}
    summary_rows = []
    for group in ("wt", "no_msa", "with_msa"):
        cases = sorted({r["case"] for r in rows if r["group"] == group})
        for case_id in cases:
            scores = by_key.get((group, case_id), [])
            if not scores:
                continue
            wt = wt_scores.get(case_id)
            mean_score = sum(scores) / len(scores)
            summary_rows.append(
                {
                    "group": group,
                    "case": case_id,
                    "n": len(scores),
                    "mean_vina": round(mean_score, 4),
                    "min_vina": round(min(scores), 4),
                    "max_vina": round(max(scores), 4),
                    "wt_vina": wt if wt is not None else "",
                    "delta_mean_vs_wt": round(mean_score - wt, 4) if wt is not None else "",
                }
            )

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group", "case", "n", "mean_vina", "min_vina", "max_vina", "wt_vina", "delta_mean_vs_wt"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
