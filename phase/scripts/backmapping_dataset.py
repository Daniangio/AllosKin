from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime

from phase.services.project_store import ProjectStore
from phase.workflows.backmapping import build_sample_backmapping_dataset


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Build a per-sample backmapping dataset for one md_eval sample by combining an uploaded "
            "trajectory with the sample's assigned cluster labels."
        )
    )
    ap.add_argument("--root", default="", help="PHASE data root (defaults to $PHASE_DATA_ROOT or ./data).")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--cluster-id", required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--trajectory", required=True, help="Trajectory file matching the sample's source state.")
    ap.add_argument("--progress", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)

    root = (args.root or os.getenv("PHASE_DATA_ROOT") or "").strip()
    if not root:
        root = str((Path(__file__).resolve().parents[2] / "data").resolve())
    os.environ["PHASE_DATA_ROOT"] = root

    trajectory_path = Path(str(args.trajectory)).expanduser().resolve()
    if not trajectory_path.exists():
        raise SystemExit(f"Trajectory file not found: {trajectory_path}")

    store = ProjectStore()
    sample_meta = store.get_sample_entry(args.project_id, args.system_id, args.cluster_id, args.sample_id)
    sample_dir = store._sample_dir(args.project_id, args.system_id, args.cluster_id, args.sample_id)
    system_dir = store.ensure_directories(args.project_id, args.system_id)["system_dir"]
    out_path = sample_dir / "backmapping_dataset.npz"

    def progress_cb(current: int, total: int) -> None:
        if not args.progress:
            return
        print(f"[backmapping_dataset] frames: {current}/{total}")

    summary = build_sample_backmapping_dataset(
        project_id=args.project_id,
        system_id=args.system_id,
        cluster_id=args.cluster_id,
        sample_id=args.sample_id,
        trajectory_path=trajectory_path,
        output_path=out_path,
        progress_callback=progress_cb,
    )

    dataset_meta = dict(sample_meta.get("backmapping_dataset") or {})
    dataset_meta.update(
        {
            "status": "finished",
            "path": str(out_path.relative_to(system_dir)),
            "updated_at": datetime.utcnow().isoformat(),
            "source_trajectory_name": trajectory_path.name,
            "n_frames": int(summary.get("n_frames") or 0),
            "n_atoms": int(summary.get("n_atoms") or 0),
            "n_residues": int(summary.get("n_residues") or 0),
            "dihedral_keys": list(summary.get("dihedral_keys") or []),
            "error": None,
        }
    )
    sample_meta["backmapping_dataset"] = dataset_meta
    store.save_sample_entry(args.project_id, args.system_id, args.cluster_id, args.sample_id, sample_meta)

    print(f"[backmapping_dataset] sample_id={args.sample_id}")
    print(f"[backmapping_dataset] output={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
