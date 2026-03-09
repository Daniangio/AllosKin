from __future__ import annotations

import argparse
import os
from pathlib import Path

from phase.potts.orchestration import run_ligand_completion_local


def _parse_list(value: str) -> list[str]:
    return [v.strip() for v in str(value or "").split(",") if v.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(v) for v in _parse_list(value)]


def _parse_mixed_list(value: str) -> list[str | int]:
    out: list[str | int] = []
    for token in _parse_list(value):
        try:
            out.append(int(token))
        except Exception:
            out.append(token)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run ligand-guided conditional completion analysis (A/B endpoint models)."
    )
    ap.add_argument("--root", default="", help="PHASE data root (defaults to $PHASE_DATA_ROOT or ./data).")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--cluster-id", required=True)

    ap.add_argument("--model-a-id", required=True, help="Endpoint A model id (typically inactive).")
    ap.add_argument("--model-b-id", required=True, help="Endpoint B model id (typically active).")
    ap.add_argument("--md-sample-id", required=True, help="Starting MD sample id.")
    ap.add_argument(
        "--constrained-residues",
        default="",
        help="Comma-separated constrained residues (indices, res keys, or residue numbers).",
    )
    ap.add_argument("--reference-sample-id-a", default="", help="Optional explicit MD reference sample id for A.")
    ap.add_argument("--reference-sample-id-b", default="", help="Optional explicit MD reference sample id for B.")

    ap.add_argument("--sampler", default="sa", choices=["sa", "gibbs"])
    ap.add_argument("--lambda-values", default="0,0.25,0.5,1,2,4,8")
    ap.add_argument("--n-start-frames", type=int, default=100)
    ap.add_argument("--n-samples-per-frame", type=int, default=100)
    ap.add_argument("--n-steps", type=int, default=1000)
    ap.add_argument("--tail-steps", type=int, default=200)

    ap.add_argument("--target-window-size", type=int, default=11)
    ap.add_argument("--target-pseudocount", type=float, default=1e-3)
    ap.add_argument("--epsilon-logpenalty", type=float, default=1e-8)

    ap.add_argument("--constraint-weight-mode", default="uniform", choices=["uniform", "js_abs", "custom"])
    ap.add_argument("--constraint-weights", default="", help="Comma-separated weights (required for custom mode).")
    ap.add_argument("--constraint-weight-min", type=float, default=0.0)
    ap.add_argument("--constraint-weight-max", type=float, default=1.0)
    ap.add_argument("--constraint-source-mode", default="manual", choices=["manual", "delta_js_auto"])
    ap.add_argument("--constraint-delta-js-analysis-id", default="")
    ap.add_argument("--constraint-delta-js-sample-id", default="")
    ap.add_argument("--constraint-auto-top-k", type=int, default=12)
    ap.add_argument("--constraint-auto-edge-alpha", type=float, default=0.3)
    ap.add_argument("--constraint-auto-exclude-success", dest="constraint_auto_exclude_success", action="store_true")
    ap.add_argument(
        "--no-constraint-auto-exclude-success",
        dest="constraint_auto_exclude_success",
        action="store_false",
    )
    ap.set_defaults(constraint_auto_exclude_success=True)

    ap.add_argument("--gibbs-beta", type=float, default=1.0)
    ap.add_argument("--sa-beta-hot", type=float, default=0.8)
    ap.add_argument("--sa-beta-cold", type=float, default=50.0)
    ap.add_argument("--sa-schedule", default="geom", choices=["geom", "lin"])

    ap.add_argument("--md-label-mode", default="assigned", choices=["assigned", "halo"])
    ap.add_argument("--keep-invalid", action="store_true")
    ap.add_argument("--success-metric-mode", default="deltae", choices=["deltae", "delta_js_edge"])
    ap.add_argument("--delta-js-experiment-id", default="")
    ap.add_argument("--delta-js-analysis-id", default="")
    ap.add_argument("--delta-js-filter-setup-id", default="")
    ap.add_argument("--delta-js-filter-edge-alpha", type=float, default=0.75)
    ap.add_argument("--delta-js-d-residue-min", type=float, default=0.0)
    ap.add_argument("--delta-js-d-residue-max", type=float, default=None)
    ap.add_argument("--delta-js-d-edge-min", type=float, default=0.0)
    ap.add_argument("--delta-js-d-edge-max", type=float, default=None)
    ap.add_argument("--delta-js-node-edge-alpha", type=float, default=None)
    ap.add_argument("--js-success-threshold", type=float, default=0.15)
    ap.add_argument("--js-success-margin", type=float, default=0.02)
    ap.add_argument("--deltae-margin", type=float, default=0.0)
    ap.add_argument("--completion-target-success", type=float, default=0.7)
    ap.add_argument("--completion-cost-if-unreached", type=float, default=None)
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers across starting frames.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", action="store_true")

    args = ap.parse_args(argv)

    root = (args.root or os.getenv("PHASE_DATA_ROOT") or "").strip()
    if not root:
        root = str((Path(__file__).resolve().parents[2] / "data").resolve())
    os.environ["PHASE_DATA_ROOT"] = root

    constrained_residues = _parse_mixed_list(args.constrained_residues)

    constraint_weights = _parse_float_list(args.constraint_weights) if str(args.constraint_weights or "").strip() else None
    if args.constraint_weight_mode == "custom" and not constraint_weights:
        raise SystemExit("--constraint-weight-mode custom requires --constraint-weights.")
    if str(args.constraint_source_mode) == "manual" and not constrained_residues:
        raise SystemExit("--constrained-residues cannot be empty in manual constraint mode.")
    shared_djs = str(args.delta_js_experiment_id or "").strip()
    if str(args.constraint_source_mode) == "delta_js_auto" and not (
        str(args.constraint_delta_js_analysis_id or "").strip() or shared_djs
    ):
        raise SystemExit(
            "--constraint-delta-js-analysis-id (or --delta-js-experiment-id) is required when --constraint-source-mode delta_js_auto."
        )

    lambda_values = _parse_float_list(args.lambda_values)
    if len(lambda_values) < 2:
        raise SystemExit("--lambda-values must contain at least 2 values.")
    if str(args.success_metric_mode) == "delta_js_edge" and not (
        str(args.delta_js_analysis_id or "").strip() or shared_djs
    ):
        raise SystemExit("--delta-js-analysis-id (or --delta-js-experiment-id) is required when --success-metric-mode delta_js_edge.")

    last_pct = {"value": -1}

    def progress_cb(message: str, current: int, total: int):
        if not args.progress or total <= 0:
            return
        pct = int(100.0 * float(current) / float(total))
        if pct == last_pct["value"]:
            return
        last_pct["value"] = pct
        print(f"[ligand_completion] {message} {current}/{total} ({pct}%)")

    out = run_ligand_completion_local(
        project_id=str(args.project_id),
        system_id=str(args.system_id),
        cluster_id=str(args.cluster_id),
        model_a_ref=str(args.model_a_id),
        model_b_ref=str(args.model_b_id),
        md_sample_id=str(args.md_sample_id),
        constrained_residues=constrained_residues,
        reference_sample_id_a=(str(args.reference_sample_id_a).strip() or None),
        reference_sample_id_b=(str(args.reference_sample_id_b).strip() or None),
        sampler=str(args.sampler),
        lambda_values=lambda_values,
        n_start_frames=int(args.n_start_frames),
        n_samples_per_frame=int(args.n_samples_per_frame),
        n_steps=int(args.n_steps),
        tail_steps=int(args.tail_steps),
        target_window_size=int(args.target_window_size),
        target_pseudocount=float(args.target_pseudocount),
        epsilon_logpenalty=float(args.epsilon_logpenalty),
        constraint_weight_mode=str(args.constraint_weight_mode),
        constraint_weights=constraint_weights,
        constraint_weight_min=float(args.constraint_weight_min),
        constraint_weight_max=float(args.constraint_weight_max),
        constraint_source_mode=str(args.constraint_source_mode),
        constraint_delta_js_analysis_id=(str(args.constraint_delta_js_analysis_id).strip() or None),
        constraint_delta_js_sample_id=(str(args.constraint_delta_js_sample_id).strip() or None),
        constraint_auto_top_k=int(args.constraint_auto_top_k),
        constraint_auto_edge_alpha=float(args.constraint_auto_edge_alpha),
        constraint_auto_exclude_success=bool(args.constraint_auto_exclude_success),
        gibbs_beta=float(args.gibbs_beta),
        sa_beta_hot=float(args.sa_beta_hot),
        sa_beta_cold=float(args.sa_beta_cold),
        sa_schedule=str(args.sa_schedule),
        md_label_mode=str(args.md_label_mode),
        drop_invalid=not bool(args.keep_invalid),
        success_metric_mode=str(args.success_metric_mode),
        delta_js_experiment_id=(str(args.delta_js_experiment_id).strip() or None),
        delta_js_analysis_id=(str(args.delta_js_analysis_id).strip() or None),
        delta_js_filter_setup_id=(str(args.delta_js_filter_setup_id).strip() or None),
        delta_js_filter_edge_alpha=float(args.delta_js_filter_edge_alpha),
        delta_js_d_residue_min=float(args.delta_js_d_residue_min),
        delta_js_d_residue_max=args.delta_js_d_residue_max,
        delta_js_d_edge_min=float(args.delta_js_d_edge_min),
        delta_js_d_edge_max=args.delta_js_d_edge_max,
        delta_js_node_edge_alpha=args.delta_js_node_edge_alpha,
        js_success_threshold=float(args.js_success_threshold),
        js_success_margin=float(args.js_success_margin),
        deltae_margin=float(args.deltae_margin),
        completion_target_success=float(args.completion_target_success),
        completion_cost_if_unreached=args.completion_cost_if_unreached,
        n_workers=int(args.workers),
        seed=int(args.seed),
        progress_callback=progress_cb if bool(args.progress) else None,
    )

    meta = out.get("metadata") or {}
    print(f"[ligand_completion] analysis_id: {meta.get('analysis_id')}")
    print(f"[ligand_completion] analysis_npz: {out.get('analysis_npz')}")
    print(f"[ligand_completion] analysis_dir: {out.get('analysis_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
