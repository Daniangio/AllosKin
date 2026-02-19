from __future__ import annotations

import argparse
from pathlib import Path

from phase.workflows.clustering import evaluate_state_with_models
from phase.services.project_store import ProjectStore


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Offline data root (contains projects/)")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--cluster-id", required=True)
    ap.add_argument("--state-id", required=True)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Residue prediction workers (1=sequential, 0=auto/all CPUs).",
    )
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve() / "projects"
    store = ProjectStore(base_dir=root)
    system_meta = store.get_system(args.project_id, args.system_id)
    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == args.cluster_id), None)
    if not entry:
        raise SystemExit("Cluster not found.")
    samples = entry.get("samples")
    if not isinstance(samples, list):
        samples = []

    existing = [
        s
        for s in samples
        if isinstance(s, dict)
        and (s.get("type") or "") == "md_eval"
        and (s.get("state_id") or "") == args.state_id
        and s.get("sample_id")
    ]
    reuse_id = None
    if existing:
        existing.sort(key=lambda s: str(s.get("created_at") or ""))
        reuse_id = str(existing[-1].get("sample_id"))
        # Drop duplicates from the metadata list (keeps the latest sample_id stable).
        dup_ids = {str(s.get("sample_id")) for s in existing[:-1] if s.get("sample_id")}
        if dup_ids:
            samples = [s for s in samples if not (isinstance(s, dict) and s.get("sample_id") in dup_ids)]

    sample_entry = evaluate_state_with_models(
        args.project_id,
        args.system_id,
        args.cluster_id,
        args.state_id,
        store=store,
        sample_id=reuse_id,
        workers=int(args.workers),
    )

    out_id = sample_entry.get("sample_id")
    replaced = False
    for idx, s in enumerate(samples):
        if isinstance(s, dict) and s.get("sample_id") == out_id:
            samples[idx] = sample_entry
            replaced = True
            break
    if not replaced:
        samples.append(sample_entry)

    entry["samples"] = samples
    store.save_system(system_meta)
    print(f"[evaluate] sample saved: {sample_entry.get('path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
