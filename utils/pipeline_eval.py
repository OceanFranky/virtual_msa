import argparse
import json
import os
import sys
from pathlib import Path

import eval_masked_backbone as emb

try:
    import pyrosetta as pr
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.core.scoring import ScoreType
    from pyrosetta.rosetta.protocols.relax import FastRelax
    pr.init("-mute all -ignore_unrecognized_res -relax:default_repeats 1")
except ImportError:
    print("错误: 无法导入 PyRosetta。请确保已安装并激活。")
    sys.exit(1)


def collect_design_files(design_file: str, design_dir: str):
    files = []
    if design_file:
        p = Path(design_file)
        if not p.exists():
            raise FileNotFoundError(f"未找到设计PDB: {design_file}")
        files.append(str(p.resolve()))
    if design_dir:
        d = Path(design_dir)
        if not d.exists():
            raise FileNotFoundError(f"未找到目录: {design_dir}")
        files.extend(str(p.resolve()) for p in sorted(d.glob("*.pdb")))
    if not files:
        raise ValueError("请提供 --design 或 --design_dir")
    return files


def run_constrained_relax(input_pdb: str, output_pdb: str):
    pose = pr.pose_from_pdb(input_pdb)
    mm = MoveMap()
    mm.set_bb(True)
    mm.set_chi(True)
    scorefxn = pr.get_fa_scorefxn()
    fastrelax = FastRelax()
    fastrelax.set_scorefxn(scorefxn)
    fastrelax.set_movemap(mm)
    fastrelax.constrain_relax_to_start_coords(True)
    fastrelax.apply(pose)
    pose.dump_pdb(output_pdb)
    return pose


def evaluate_loop_energy_from_pose(pose, x_indices, chain_id="A"):
    scorefxn = pr.get_fa_scorefxn()
    total_score = scorefxn(pose)
    total_loop_energy = 0.0
    total_loop_clash = 0.0
    valid_x_count = 0
    details = []
    for seq_pos in x_indices:
        pdb_resseq = seq_pos + 1
        pose_res_num = pose.pdb_info().pdb2pose(chain_id, pdb_resseq)
        if pose_res_num == 0:
            continue
        valid_x_count += 1
        res_energy = pose.energies().residue_total_energy(pose_res_num)
        fa_rep = pose.energies().residue_total_energies(pose_res_num)[ScoreType.fa_rep]
        details.append(
            {
                "seq_pos": int(pdb_resseq),
                "res_name": pose.residue(pose_res_num).name3(),
                "energy": float(res_energy),
                "clash_fa_rep": float(fa_rep),
            }
        )
        total_loop_energy += res_energy
        total_loop_clash += fa_rep
    avg_loop_energy = total_loop_energy / valid_x_count if valid_x_count > 0 else 0.0
    return {
        "total_protein_score": float(total_score),
        "loop_residue_count": int(valid_x_count),
        "loop_total_energy": float(total_loop_energy),
        "loop_average_energy": float(avg_loop_energy),
        "loop_total_clash_fa_rep": float(total_loop_clash),
        "details": details,
    }


def evaluate_initial_backbone(native_pdb, design_pdb, chain, non_x_indices):
    ref_coords = emb.extract_backbone_coords(native_pdb, chain, non_x_indices)
    pred_coords = emb.extract_backbone_coords(design_pdb, chain, non_x_indices)
    if len(ref_coords) != len(pred_coords) or len(ref_coords) == 0:
        raise ValueError("骨架原子配对失败：坐标为空或数量不一致")
    rmsd = emb.calculate_rmsd(ref_coords, pred_coords)
    lddt = emb.calculate_lddt(ref_coords, pred_coords)
    return float(rmsd), float(lddt), ref_coords


def evaluate_relaxed_backbone(ref_coords, relaxed_pdb, chain, non_x_indices):
    relaxed_coords = emb.extract_backbone_coords(relaxed_pdb, chain, non_x_indices)
    if len(ref_coords) != len(relaxed_coords) or len(relaxed_coords) == 0:
        raise ValueError("松弛后骨架原子配对失败")
    rmsd = emb.calculate_rmsd(ref_coords, relaxed_coords)
    lddt = emb.calculate_lddt(ref_coords, relaxed_coords)
    return float(rmsd), float(lddt)


def main():
    parser = argparse.ArgumentParser(description="批量评估: 先按骨架RMSD/lDDT筛选，再松弛并评估Loop")
    parser.add_argument("--a3m", required=True, type=str)
    parser.add_argument("--native", required=True, type=str)
    parser.add_argument("--chain", default="A", type=str)
    parser.add_argument("--design", default="", type=str)
    parser.add_argument("--design_dir", default="", type=str)
    parser.add_argument("--rmsd_max", default=2.0, type=float)
    parser.add_argument("--lddt_min", default=0.7, type=float)
    parser.add_argument("--out_dir", default="./eval_results", type=str)
    args = parser.parse_args()

    design_files = collect_design_files(args.design, args.design_dir)
    out_dir = Path(args.out_dir)
    reports_dir = out_dir / "reports"
    relaxed_dir = out_dir / "relaxed_pdbs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    relaxed_dir.mkdir(parents=True, exist_ok=True)

    non_x_indices, full_seq = emb.get_non_x_indices(args.a3m)
    x_indices = [i for i, c in enumerate(full_seq) if c.upper() == "X"]

    summary = {
        "config": {
            "a3m": str(Path(args.a3m).resolve()),
            "native": str(Path(args.native).resolve()),
            "chain": args.chain,
            "rmsd_max": float(args.rmsd_max),
            "lddt_min": float(args.lddt_min),
            "total_designs": len(design_files),
            "scaffold_length": len(non_x_indices),
            "loop_length": len(x_indices),
        },
        "passed_prefilter": [],
        "failed_prefilter": [],
        "final_results": [],
    }

    for i, design_pdb in enumerate(design_files, start=1):
        base = Path(design_pdb).stem
        print(f"\n[{i}/{len(design_files)}] 处理 {base}")
        report = {
            "design_pdb": design_pdb,
            "sequence_info": {
                "total_length": len(full_seq),
                "scaffold_non_x_count": len(non_x_indices),
                "loop_x_count": len(x_indices),
            },
        }
        try:
            initial_rmsd, initial_lddt, ref_coords = evaluate_initial_backbone(
                args.native, design_pdb, args.chain, non_x_indices
            )
            report["initial_backbone"] = {
                "rmsd": initial_rmsd,
                "lddt": initial_lddt,
            }
            prefilter_pass = (initial_rmsd <= args.rmsd_max) and (initial_lddt >= args.lddt_min)
            if not prefilter_pass:
                report["prefilter"] = {
                    "status": "FAIL",
                    "reason": f"不满足阈值: rmsd={initial_rmsd:.4f}, lddt={initial_lddt:.4f}",
                }
                summary["failed_prefilter"].append(report)
                with open(reports_dir / f"{base}_report.json", "w") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                print(f"  预筛选失败: RMSD={initial_rmsd:.3f}, lDDT={initial_lddt:.3f}")
                continue

            report["prefilter"] = {"status": "PASS"}
            summary["passed_prefilter"].append(
                {
                    "design_pdb": design_pdb,
                    "initial_rmsd": initial_rmsd,
                    "initial_lddt": initial_lddt,
                }
            )
            print(f"  预筛选通过: RMSD={initial_rmsd:.3f}, lDDT={initial_lddt:.3f}")

            relaxed_pdb = str((relaxed_dir / f"{base}_relaxed.pdb").resolve())
            pose = run_constrained_relax(design_pdb, relaxed_pdb)
            final_rmsd, final_lddt = evaluate_relaxed_backbone(
                ref_coords, relaxed_pdb, args.chain, non_x_indices
            )
            loop_scores = evaluate_loop_energy_from_pose(pose, x_indices, args.chain)
            report["relax"] = {"relaxed_pdb": relaxed_pdb}
            report["relaxed_backbone"] = {"rmsd": final_rmsd, "lddt": final_lddt}
            report["relaxed_loop_energy"] = loop_scores
            summary["final_results"].append(report)

            with open(reports_dir / f"{base}_report.json", "w") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(
                f"  完成: 松弛后RMSD={final_rmsd:.3f}, lDDT={final_lddt:.3f}, "
                f"Loop平均能量={loop_scores['loop_average_energy']:.3f}"
            )
        except Exception as e:
            report["error"] = str(e)
            summary["failed_prefilter"].append(report)
            with open(reports_dir / f"{base}_report.json", "w") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"  失败: {e}")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n完成。汇总文件: {summary_path}")


if __name__ == "__main__":
    main()
