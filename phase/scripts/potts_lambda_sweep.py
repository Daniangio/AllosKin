from __future__ import annotations

import argparse
import os
from pathlib import Path

from phase.potts.orchestration import run_lambda_sweep_local


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a lambda interpolation sweep between two Potts models (Gibbs sampling at each lambda), "
            "persisting each lambda sample as a correlated sample series and writing a dedicated analysis."
        )
    )
    parser.add_argument("--root", default="", help="PHASE data root (defaults to $PHASE_DATA_ROOT or ./data).")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--cluster-id", required=True)

    parser.add_argument("--model-a-id", required=True, help="Endpoint model A (lambda=1).")
    parser.add_argument("--model-b-id", required=True, help="Endpoint model B (lambda=0).")

    parser.add_argument("--md-sample-id-1", required=True)
    parser.add_argument("--md-sample-id-2", required=True)
    parser.add_argument("--md-sample-id-3", required=True)
    parser.add_argument("--md-label-mode", default="assigned", choices=["assigned", "halo"])

    parser.add_argument("--lambda-count", type=int, default=21)
    parser.add_argument("--series-id", default="", help="Optional series UUID (otherwise generated).")
    parser.add_argument("--series-label", default="", help="Display label for this sweep.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Node/edge mixing weight for match curve.")
    parser.add_argument("--keep-invalid", action="store_true", help="Keep frames with invalid labels (-1) in analysis.")

    parser.add_argument("--gibbs-method", default="rex", choices=["single", "rex"])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--gibbs-samples", type=int, default=500)
    parser.add_argument("--gibbs-burnin", type=int, default=50)
    parser.add_argument("--gibbs-thin", type=int, default=2)
    parser.add_argument("--gibbs-chains", type=int, default=1)

    parser.add_argument("--rex-betas", type=str, default="")
    parser.add_argument("--rex-n-replicas", type=int, default=8)
    parser.add_argument("--rex-beta-min", type=float, default=0.2)
    parser.add_argument("--rex-beta-max", type=float, default=1.0)
    parser.add_argument("--rex-spacing", type=str, default="geom", choices=["geom", "lin"])
    parser.add_argument("--rex-rounds", type=int, default=2000)
    parser.add_argument("--rex-burnin-rounds", type=int, default=50)
    parser.add_argument("--rex-sweeps-per-round", type=int, default=2)
    parser.add_argument("--rex-thin-rounds", type=int, default=1)
    parser.add_argument("--rex-chains", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Optional cap for local lambda-sweep worker fan-out (0 uses all payloads).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.model_a_id == args.model_b_id:
        raise SystemExit("Select two different endpoint models (model-a-id != model-b-id).")
    if args.lambda_count < 2:
        raise SystemExit("--lambda-count must be >= 2.")
    if not (0.0 <= float(args.alpha) <= 1.0):
        raise SystemExit("--alpha must be in [0,1].")

    md_ids = [str(args.md_sample_id_1), str(args.md_sample_id_2), str(args.md_sample_id_3)]
    if len(set(md_ids)) != 3:
        raise SystemExit("MD reference samples must be 3 distinct sample IDs.")

    root = (args.root or os.getenv("PHASE_DATA_ROOT") or "").strip()
    if not root:
        root = str((Path(__file__).resolve().parents[2] / "data").resolve())
    os.environ["PHASE_DATA_ROOT"] = root

    def progress_cb(message: str, current: int, total: int) -> None:
        if not bool(args.progress):
            return
        print(f"[lambda_sweep] {message}: {current}/{total}")

    out = run_lambda_sweep_local(
        project_id=str(args.project_id),
        system_id=str(args.system_id),
        cluster_id=str(args.cluster_id),
        model_a_id=str(args.model_a_id),
        model_b_id=str(args.model_b_id),
        md_sample_id_1=str(args.md_sample_id_1),
        md_sample_id_2=str(args.md_sample_id_2),
        md_sample_id_3=str(args.md_sample_id_3),
        series_id=str(args.series_id or "").strip() or None,
        series_label=str(args.series_label or "").strip() or None,
        lambda_count=int(args.lambda_count),
        alpha=float(args.alpha),
        md_label_mode=str(args.md_label_mode),
        keep_invalid=bool(args.keep_invalid),
        gibbs_method=str(args.gibbs_method),
        beta=float(args.beta),
        seed=int(args.seed),
        gibbs_samples=int(args.gibbs_samples),
        gibbs_burnin=int(args.gibbs_burnin),
        gibbs_thin=int(args.gibbs_thin),
        gibbs_chains=int(args.gibbs_chains),
        rex_betas=str(args.rex_betas),
        rex_n_replicas=int(args.rex_n_replicas),
        rex_beta_min=float(args.rex_beta_min),
        rex_beta_max=float(args.rex_beta_max),
        rex_spacing=str(args.rex_spacing),
        rex_rounds=int(args.rex_rounds),
        rex_burnin_rounds=int(args.rex_burnin_rounds),
        rex_sweeps_per_round=int(args.rex_sweeps_per_round),
        rex_thin_rounds=int(args.rex_thin_rounds),
        rex_chains=int(args.rex_chains),
        n_workers=(int(args.workers) if int(args.workers) > 0 else None),
        progress_callback=progress_cb,
        max_workers_override=(int(args.workers) if int(args.workers) > 0 else None),
    )

    print(f"[lambda_sweep] series_id={out.get('series_id')}")
    print(f"[lambda_sweep] wrote {len(out.get('sample_ids') or [])} samples under clusters/{args.cluster_id}/samples/")
    print(f"[lambda_sweep] wrote analysis under clusters/{args.cluster_id}/analyses/lambda_sweep/{out.get('analysis_id')}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
