from __future__ import annotations

import argparse
import json
import os
import sys


def _parse_metastable_ids(raw: str) -> list[str]:
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    if not parts:
        raise ValueError("Provide at least one metastable ID.")
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
    parser.add_argument("--project-id", required=True, help="Project ID in the PHASE data store.")
    parser.add_argument("--system-id", required=True, help="System ID in the PHASE data store.")
    parser.add_argument(
        "--metastable-ids",
        required=True,
        help="Comma-separated metastable IDs (or macro state IDs when in macro-only mode).",
    )
    parser.add_argument("--max-clusters-per-residue", type=int, default=6)
    parser.add_argument("--max-cluster-frames", type=int)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--contact-cutoff", type=float, default=10.0)
    parser.add_argument("--contact-atom-mode", choices=["CA", "CM"], default="CA")
    parser.add_argument("--density-maxk", type=int, default=100)
    parser.add_argument("--density-z", default="auto")
    parser.add_argument(
        "--data-root",
        help="Override PHASE_DATA_ROOT for locating project data.",
    )
    parser.add_argument(
        "--no-assign",
        action="store_true",
        help="Skip generating assigned labels for all states/metastables.",
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

    if args.data_root:
        os.environ["PHASE_DATA_ROOT"] = args.data_root

    try:
        metastable_ids = _parse_metastable_ids(args.metastable_ids)
        density_z = _parse_density_z(args.density_z)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    from backend.services.metastable_clusters import (
        assign_cluster_labels_to_states,
        generate_metastable_cluster_npz,
        update_cluster_metadata_with_assignments,
    )

    try:
        npz_path, metadata = generate_metastable_cluster_npz(
            args.project_id,
            args.system_id,
            metastable_ids,
            max_clusters_per_residue=args.max_clusters_per_residue,
            max_cluster_frames=args.max_cluster_frames,
            random_state=args.random_state,
            contact_cutoff=args.contact_cutoff,
            contact_atom_mode=args.contact_atom_mode,
            cluster_algorithm="density_peaks",
            density_maxk=args.density_maxk,
            density_z=density_z,
        )
    except Exception as exc:
        print(f"[cluster] Failed to generate cluster NPZ: {exc}", file=sys.stderr)
        return 1

    if not args.no_assign:
        try:
            assignments = assign_cluster_labels_to_states(npz_path, args.project_id, args.system_id)
            update_cluster_metadata_with_assignments(npz_path, assignments)
        except Exception as exc:
            print(f"[cluster] Warning: failed to assign labels: {exc}", file=sys.stderr)

    print(f"[cluster] NPZ saved at: {npz_path}")
    if args.print_meta:
        print(json.dumps(metadata, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
