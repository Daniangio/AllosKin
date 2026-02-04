from __future__ import annotations

from pathlib import Path

from phase.potts import pipeline as sim_main
from phase.scripts.potts_utils import persist_sample


def _differs_from_default(parser, key: str, value: object) -> bool:
    try:
        default = parser.get_default(key)
    except Exception:
        return True
    if value == default:
        return False
    if isinstance(value, list) and isinstance(default, list):
        return value != default
    return True


def _filter_sampling_params(args: object, parser) -> dict:
    if not hasattr(args, "__dict__"):
        return {}
    raw = vars(args)
    sampling_method = raw.get("sampling_method") or "gibbs"
    gibbs_method = raw.get("gibbs_method") or "single"

    allow = {"sampling_method", "beta", "seed", "estimate_beta_eff"}
    if sampling_method == "gibbs":
        allow |= {"gibbs_method"}
        if gibbs_method == "single":
            allow |= {"gibbs_samples", "gibbs_burnin", "gibbs_thin", "gibbs_chains"}
        else:
            allow |= {
                "rex_betas",
                "rex_n_replicas",
                "rex_beta_min",
                "rex_beta_max",
                "rex_spacing",
                "rex_rounds",
                "rex_burnin_rounds",
                "rex_sweeps_per_round",
                "rex_thin_rounds",
                "rex_chains",
            }
    else:
        allow |= {
            "sa_reads",
            "sa_sweeps",
            "sa_beta_hot",
            "sa_beta_cold",
            "sa_beta_schedule",
            "sa_init",
            "sa_init_md_frame",
            "sa_restart",
            "sa_restart_topk",
            "penalty_safety",
            "repair",
        }

    if raw.get("estimate_beta_eff"):
        allow |= {"beta_eff_grid", "beta_eff_w_marg", "beta_eff_w_pair"}

    out = {"sampling_method": sampling_method}
    if sampling_method == "gibbs":
        out["beta"] = raw.get("beta")
        out["gibbs_method"] = gibbs_method
    for key in allow:
        if key not in raw:
            continue
        val = raw.get(key)
        if val in (None, "", [], {}):
            continue
        if key in out:
            continue
        if not _differs_from_default(parser, key, val):
            continue
        out[key] = val
    return out


def main(argv: list[str] | None = None) -> int:
    parser = sim_main._build_arg_parser()
    parser.add_argument("--project-id", default="")
    parser.add_argument("--system-id", default="")
    parser.add_argument("--cluster-id", default="")
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--sample-name", default="")
    args = parser.parse_args(argv)
    args.fit_only = False
    if not getattr(args, "plot_only", False):
        args.no_plots = True
    args.no_save_model = True
    try:
        results = sim_main.run_pipeline(args, parser=parser)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1
    project_id = (args.project_id or "").strip()
    system_id = (args.system_id or "").strip()
    cluster_id = (args.cluster_id or "").strip()
    if project_id and system_id and cluster_id and results and not getattr(args, "plot_only", False):
        summary_path = results.get("summary_path")
        summary = Path(summary_path) if summary_path is not None else None
        if summary is not None:
            model_paths = []
            for raw in getattr(args, "model_npz", []) or []:
                model_paths.append(Path(str(raw)))
            persist_sample(
                project_id=project_id,
                system_id=system_id,
                cluster_id=cluster_id,
                summary_path=summary,
                metadata_path=None,
                sample_name=args.sample_name or None,
                sample_type="potts_sampling",
                method=str(args.sampling_method) if getattr(args, "sampling_method", None) else None,
                params=_filter_sampling_params(args, parser),
                model_paths=model_paths,
                sample_id=args.sample_id or None,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
