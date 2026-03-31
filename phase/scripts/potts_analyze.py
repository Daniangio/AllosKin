from __future__ import annotations

import argparse
import os
from pathlib import Path

from phase.potts.analysis_run import analyze_cluster_samples, append_state_pose_energies


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analyze saved samples for one cluster (MD-vs-sample metrics + optional energies).")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--cluster-id", required=True)
    ap.add_argument(
        "--model",
        default="",
        help="Optional: model_id (from potts_models metadata) or a model NPZ path. If set, energies are computed for all samples.",
    )
    ap.add_argument("--md-label-mode", default="assigned", choices=["assigned", "halo"])
    ap.add_argument("--keep-invalid", action="store_true", help="Do not drop invalid SA samples (invalid_mask rows).")
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers for local analysis (0=auto).")
    ap.add_argument("--progress", action="store_true", help="Show progress while running local analysis.")
    ap.add_argument("--pose-only", action="store_true", help="Append single-PDB state energies under an existing model-energy analysis context.")
    ap.add_argument("--state-pose-id", action="append", default=[], help="State id to evaluate as a single PDB pose. Repeatable.")
    args = ap.parse_args(argv)

    model_ref = args.model.strip() or None
    if model_ref is None:
        data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
        store = ProjectStore(base_dir=data_root / "projects")
        models = store.list_potts_models(args.project_id, args.system_id, args.cluster_id)
        if models:
            print("Available Potts models:")
            for idx, model in enumerate(models, start=1):
                label = model.get("name") or model.get("model_id") or "model"
                print(f"  [{idx}] {label} ({model.get('model_id')})")
            choice = input("Select model number for energies (blank to skip): ").strip()
            if choice:
                try:
                    selected = int(choice)
                    if 1 <= selected <= len(models):
                        model_ref = str(models[selected - 1].get("model_id") or "").strip() or None
                except Exception:
                    model_ref = None

    if args.pose_only:
        if not model_ref:
            raise SystemExit("--pose-only requires --model.")
        if not args.state_pose_id:
            raise SystemExit("--pose-only requires at least one --state-pose-id.")
        summary = append_state_pose_energies(
            project_id=args.project_id,
            system_id=args.system_id,
            cluster_id=args.cluster_id,
            model_ref=model_ref,
            state_ids=args.state_pose_id,
        )
    else:
        progress_cb = None
        if args.progress:
            def progress_cb(message: str, current: int, total: int):
                total_i = max(1, int(total))
                print(f"[{int(current)}/{total_i}] {message}")

        summary = analyze_cluster_samples(
            project_id=args.project_id,
            system_id=args.system_id,
            cluster_id=args.cluster_id,
            model_ref=model_ref,
            md_label_mode=args.md_label_mode,
            drop_invalid=not bool(args.keep_invalid),
            n_workers=(int(args.workers) if int(args.workers) > 0 else None),
            progress_callback=progress_cb,
        )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
