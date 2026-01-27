from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _parse_paths(raw: str) -> list[str]:
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    if not parts:
        raise ValueError("Provide at least one descriptor NPZ path.")
    return parts


def _parse_density_z(raw: str | None) -> float | str:
    if raw is None:
        return "auto"
    value = str(raw).strip()
    if not value:
        return "auto"
    if value.lower() == "auto":
        return "auto"
    return float(value)




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster descriptor NPZs using the same pipeline as the webserver.",
    )
    parser.add_argument(
        "--descriptors",
        required=True,
        help="Comma-separated descriptor NPZ paths.",
    )
    parser.add_argument(
        "--labels",
        help="Comma-separated labels for each descriptor NPZ (defaults to file stem).",
    )
    parser.add_argument(
        "--eval-descriptors",
        help="Comma-separated descriptor NPZ paths for evaluation only (not used for clustering).",
    )
    parser.add_argument("--max-cluster-frames", type=int)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--density-maxk", type=int, default=100)
    parser.add_argument("--density-z", default="auto")
    parser.add_argument("--n-jobs", type=int, default=1, help="Worker processes for residue clustering (default=1; 0=all cpus).")
    parser.add_argument(
        "--output",
        help="Output cluster NPZ path (default: ./cluster_<timestamp>.npz).",
    )
    parser.add_argument(
        "--print-meta",
        action="store_true",
        help="Print the metadata JSON to stdout after clustering.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        descriptor_paths = _parse_paths(args.descriptors)
        density_z = _parse_density_z(args.density_z)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    labels = _parse_paths(args.labels) if args.labels else None
    eval_descriptor_paths = _parse_paths(args.eval_descriptors) if args.eval_descriptors else []
    from backend.services.metastable_clusters import generate_cluster_npz_from_descriptors
    if args.n_jobs is None or int(args.n_jobs) <= 0:
        n_jobs = os.cpu_count() or 1
    else:
        n_jobs = int(args.n_jobs)

    progress_bar = None
    progress_callback = None
    try:
        from tqdm import tqdm  # type: ignore

        progress_bar = tqdm(total=1, desc="Clustering residues", unit="res")

        def progress_callback(message: str, current: int, total: int) -> None:
            nonlocal progress_bar
            if total <= 0:
                return
            if progress_bar is None or progress_bar.total != total:
                if progress_bar:
                    progress_bar.close()
                progress_bar = tqdm(total=total, desc="Clustering residues", unit="res")
            progress_bar.n = min(current, total)
            progress_bar.refresh()
    except Exception:
        progress_callback = None

    try:
        npz_path, metadata = generate_cluster_npz_from_descriptors(
            [Path(p) for p in descriptor_paths],
            labels=labels,
            eval_descriptor_paths=[Path(p) for p in eval_descriptor_paths],
            output_path=Path(args.output) if args.output else None,
            max_cluster_frames=args.max_cluster_frames,
            random_state=args.random_state,
            density_maxk=args.density_maxk,
            density_z=density_z,
            n_jobs=n_jobs,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        print(f"[cluster] Failed to generate cluster NPZ: {exc}", file=sys.stderr)
        return 1
    finally:
        if progress_bar:
            progress_bar.close()

    print(f"[cluster] NPZ saved at: {npz_path}")
    if args.print_meta:
        print(json.dumps(metadata, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
