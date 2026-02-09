from __future__ import annotations

import argparse
import os
from pathlib import Path
import uuid

from phase.services.project_store import ProjectStore
from phase.potts.potts_model import interpolate_potts_models, load_potts_model, save_potts_model, zero_sum_gauge_model
from phase.scripts.potts_utils import persist_model, sanitize_model_filename


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Create and persist a derived Potts model by interpolating two existing endpoint models:\n"
            "  E_λ = (1-λ) * E_B + λ * E_A\n"
            "where B corresponds to λ=0 and A corresponds to λ=1."
        )
    )
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--cluster-id", required=True)
    ap.add_argument("--model-a-id", required=True, help="Endpoint model A (λ=1).")
    ap.add_argument("--model-b-id", required=True, help="Endpoint model B (λ=0).")
    ap.add_argument("--lam", type=float, required=True, help="Interpolation λ in [0,1].")
    ap.add_argument("--name", default="", help="Optional display name for the saved model.")
    ap.add_argument("--source", default="offline_derived", help="Model source string.")
    ap.add_argument("--no-gauge", action="store_true", help="Do not apply zero-sum gauge before/after interpolation.")
    args = ap.parse_args(argv)

    lam = float(args.lam)
    if not (0.0 <= lam <= 1.0):
        raise SystemExit("--lam must be in [0,1].")
    if str(args.model_a_id) == str(args.model_b_id):
        raise SystemExit("Endpoint models must be different.")

    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")

    system_meta = store.get_system(args.project_id, args.system_id)
    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == args.cluster_id), None)
    if not isinstance(entry, dict):
        raise SystemExit(f"Cluster '{args.cluster_id}' not found in system metadata.")

    models = entry.get("potts_models") or []
    a_meta = next((m for m in models if m.get("model_id") == args.model_a_id), None)
    b_meta = next((m for m in models if m.get("model_id") == args.model_b_id), None)
    if not isinstance(a_meta, dict) or not isinstance(b_meta, dict):
        raise SystemExit("Could not locate both endpoint models in this cluster.")

    for mid, meta in [(args.model_a_id, a_meta), (args.model_b_id, b_meta)]:
        params = meta.get("params") or {}
        if isinstance(params, dict):
            dk = str(params.get("delta_kind") or "").strip().lower()
            if dk.startswith("delta"):
                raise SystemExit(
                    f"Endpoint model {mid} appears delta-only (params.delta_kind={dk!r}). "
                    "Please select sampleable endpoint models (standard or combined)."
                )

    rel_a = a_meta.get("path")
    rel_b = b_meta.get("path")
    if not rel_a or not rel_b:
        raise SystemExit("Endpoint model path missing in metadata.")

    a_path = store.resolve_path(args.project_id, args.system_id, str(rel_a))
    b_path = store.resolve_path(args.project_id, args.system_id, str(rel_b))
    if not a_path.exists() or not b_path.exists():
        raise SystemExit("Endpoint model NPZ missing on disk.")

    model_a = load_potts_model(str(a_path))
    model_b = load_potts_model(str(b_path))
    do_gauge = not bool(args.no_gauge)
    if do_gauge:
        model_a = zero_sum_gauge_model(model_a)
        model_b = zero_sum_gauge_model(model_b)
    derived = interpolate_potts_models(model_b, model_a, lam)
    if do_gauge:
        derived = zero_sum_gauge_model(derived)

    a_name = a_meta.get("name") or str(args.model_a_id)
    b_name = b_meta.get("name") or str(args.model_b_id)
    display_name = (str(args.name or "").strip() or f"Lambda {lam:.3f} {b_name} -> {a_name}")

    # Save into the final potts_models bucket up-front so the model is usable even if interrupted later.
    model_id = str(uuid.uuid4())
    dirs = store.ensure_cluster_directories(args.project_id, args.system_id, args.cluster_id)
    system_dir = dirs["system_dir"]
    model_dir = dirs["potts_models_dir"] / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sanitize_model_filename(display_name)}.npz"
    dest_path = model_dir / filename

    params = {
        "fit_mode": "derived",
        "derived_kind": "lambda_interpolation",
        "lambda": float(lam),
        "endpoint_model_a_id": str(args.model_a_id),
        "endpoint_model_b_id": str(args.model_b_id),
        "endpoint_model_a_name": str(a_name),
        "endpoint_model_b_name": str(b_name),
        "zero_sum_gauge": bool(do_gauge),
    }
    save_potts_model(derived, dest_path, metadata=params)

    # Persist system metadata entry and model_metadata.json (no-op copy because we already wrote to dest_path).
    persist_model(
        project_id=str(args.project_id),
        system_id=str(args.system_id),
        cluster_id=str(args.cluster_id),
        model_path=dest_path,
        model_name=display_name,
        params=params,
        source=str(args.source or "offline_derived"),
        model_id=model_id,
    )

    try:
        rel = str(dest_path.relative_to(system_dir))
    except Exception:
        rel = str(dest_path)
    print(f"[lambda_model] model_id={model_id}")
    print(f"[lambda_model] path={rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

