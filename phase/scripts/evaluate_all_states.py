from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from phase.services.project_store import ProjectStore
from phase.workflows.clustering import evaluate_state_with_models


def _pick_latest_md_eval(samples: list[dict], state_id: str) -> tuple[str | None, list[str]]:
    """Return (keep_sample_id, duplicate_sample_ids)."""
    matches = [
        s
        for s in samples
        if isinstance(s, dict)
        and (s.get("type") or "") == "md_eval"
        and (s.get("state_id") or "") == state_id
        and s.get("sample_id")
    ]
    if not matches:
        return None, []
    matches.sort(key=lambda s: str(s.get("created_at") or ""))
    keep = matches[-1]
    keep_id = str(keep.get("sample_id"))
    dups = [str(s.get("sample_id")) for s in matches[:-1] if s.get("sample_id")]
    return keep_id, dups


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recompute MD evaluation samples (md_eval) for all descriptor-ready states.")
    ap.add_argument("--root", required=True, help="Offline data root (contains projects/)")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--cluster-id", required=True)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing md_eval sample folders (default).")
    ap.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Always create new md_eval samples.")
    ap.set_defaults(overwrite=True)
    ap.add_argument("--cleanup", action="store_true", help="Delete duplicate md_eval sample folders (default).")
    ap.add_argument("--no-cleanup", dest="cleanup", action="store_false", help="Keep duplicate md_eval sample folders.")
    ap.set_defaults(cleanup=True)
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve() / "projects"
    store = ProjectStore(base_dir=root)
    system_meta = store.get_system(args.project_id, args.system_id)

    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == args.cluster_id), None)
    if not isinstance(entry, dict):
        raise SystemExit(f"Cluster '{args.cluster_id}' not found.")

    cluster_dirs = store.ensure_cluster_directories(args.project_id, args.system_id, args.cluster_id)
    samples_dir = cluster_dirs["samples_dir"]

    samples = entry.get("samples")
    if not isinstance(samples, list):
        samples = []

    descriptor_states = [s for s in system_meta.states.values() if getattr(s, "descriptor_file", None)]
    if not descriptor_states:
        print("[evaluate-all] no states with descriptors found.")
        return 0

    total = len(descriptor_states)
    refreshed = 0

    for idx, state in enumerate(descriptor_states, start=1):
        state_id = state.state_id
        keep_id, dup_ids = _pick_latest_md_eval(samples, state_id)

        reuse_id = keep_id if args.overwrite else None
        if args.cleanup and dup_ids:
            for sid in dup_ids:
                try:
                    shutil.rmtree(samples_dir / sid, ignore_errors=True)
                except Exception:
                    pass
            # Drop duplicates from metadata list.
            samples = [
                s
                for s in samples
                if not (
                    isinstance(s, dict)
                    and (s.get("type") or "") == "md_eval"
                    and (s.get("state_id") or "") == state_id
                    and s.get("sample_id") in set(dup_ids)
                )
            ]

        print(f"[evaluate-all] {idx}/{total} evaluating {state_id}...")
        sample_entry = evaluate_state_with_models(
            args.project_id,
            args.system_id,
            args.cluster_id,
            state_id,
            store=store,
            sample_id=reuse_id,
        )

        # Replace-or-append the sample entry.
        out_id = sample_entry.get("sample_id")
        replaced = False
        for j, existing in enumerate(samples):
            if not isinstance(existing, dict):
                continue
            if existing.get("sample_id") == out_id:
                samples[j] = sample_entry
                replaced = True
                break
        if not replaced:
            # If we're overwriting but the old entry was malformed, remove any md_eval entries for this state.
            if args.overwrite:
                samples = [
                    s
                    for s in samples
                    if not (
                        isinstance(s, dict)
                        and (s.get("type") or "") == "md_eval"
                        and (s.get("state_id") or "") == state_id
                    )
                ]
            samples.append(sample_entry)

        entry["samples"] = samples
        system_meta.metastable_clusters = clusters
        store.save_system(system_meta)
        refreshed += 1
        print(f"[evaluate-all] saved: {sample_entry.get('path')}")

    print(f"[evaluate-all] refreshed md_eval samples: {refreshed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

