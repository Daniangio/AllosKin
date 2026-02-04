from __future__ import annotations

import os
from pathlib import Path
import json
import uuid
from datetime import datetime

from phase.services.project_store import ProjectStore
from phase.potts import pipeline as sim_main
from phase.scripts.potts_utils import persist_model, sanitize_model_filename, _write_model_metadata


def _filter_fit_params(args: object) -> dict:
    if not hasattr(args, "__dict__"):
        return {}
    raw = vars(args)
    allow = {
        "npz",
        "fit",
        "fit_only",
        "seed",
        "unassigned_policy",
        "contact_all_vs_all",
        "pdbs",
        "contact_cutoff",
        "contact_atom_mode",
        "plm_init",
        "plm_init_model",
        "plm_resume_model",
        "plm_val_frac",
        "plm_device",
        "plm_epochs",
        "plm_lr",
        "plm_lr_min",
        "plm_lr_schedule",
        "plm_l2",
        "plm_batch_size",
        "plm_progress_every",
        "plm_lambda",
    }
    filtered = {k: raw.get(k) for k in allow if k in raw}
    return {k: v for k, v in filtered.items() if v not in (None, "", [], {})}


def _relativize_param_paths(params: dict, system_dir: Path) -> dict:
    if not isinstance(params, dict):
        return {}
    path_keys = {"npz", "data_npz", "plm_init_model", "plm_resume_model", "pdbs"}
    updated = dict(params)
    for key in path_keys:
        value = updated.get(key)
        if not value:
            continue
        if isinstance(value, str) and "," in value:
            parts = [p.strip() for p in value.split(",") if p.strip()]
            rel_parts = []
            for part in parts:
                try:
                    p = Path(part)
                except Exception:
                    rel_parts.append(part)
                    continue
                if p.is_absolute():
                    try:
                        rel_parts.append(str(p.relative_to(system_dir)))
                    except Exception:
                        rel_parts.append(part)
                else:
                    rel_parts.append(part)
            updated[key] = ",".join(rel_parts)
            continue
        try:
            p = Path(str(value))
        except Exception:
            continue
        if p.is_absolute():
            try:
                updated[key] = str(p.relative_to(system_dir))
            except Exception:
                pass
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = sim_main._build_arg_parser()
    parser.add_argument("--project-id", default="")
    parser.add_argument("--system-id", default="")
    parser.add_argument("--cluster-id", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--model-source", default="offline")
    args = parser.parse_args(argv)
    args.fit_only = True
    project_id = (args.project_id or "").strip()
    system_id = (args.system_id or "").strip()
    cluster_id = (args.cluster_id or "").strip()
    if project_id and system_id and cluster_id and not args.model_out:
        if args.plm_resume_model:
            resume_path = Path(args.plm_resume_model)
            if not resume_path.is_absolute():
                store = ProjectStore(base_dir=Path(os.getenv("PHASE_DATA_ROOT", "/app/data")) / "projects")
                resume_path = store.resolve_path(project_id, system_id, str(args.plm_resume_model))
            args.model_out = str(resume_path)
        else:
            data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
            store = ProjectStore(base_dir=data_root / "projects")
            try:
                system_meta = store.get_system(project_id, system_id)
            except FileNotFoundError:
                system_meta = None
            entry = None
            if system_meta:
                entry = next(
                    (c for c in (system_meta.metastable_clusters or []) if c.get("cluster_id") == cluster_id),
                    None,
                )
            display_name = args.model_name or None
            if not display_name and isinstance(entry, dict):
                cluster_name = entry.get("name")
                if not display_name and isinstance(cluster_name, str) and cluster_name.strip():
                    display_name = f"{cluster_name} Potts Model"
            if not display_name:
                display_name = f"{cluster_id} Potts Model"
            model_id = str(uuid.uuid4())
            model_dir = store.ensure_cluster_directories(project_id, system_id, cluster_id)["potts_models_dir"] / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            args.model_out = str(model_dir / f"{sanitize_model_filename(display_name)}.npz")
    pre_model_meta_path = None
    if project_id and system_id and cluster_id and args.model_out:
        try:
            store = ProjectStore(base_dir=Path(os.getenv("PHASE_DATA_ROOT", "/app/data")) / "projects")
            system_dir = store.ensure_directories(project_id, system_id)["system_dir"]
            model_path = Path(args.model_out)
            model_dir = model_path.parent
            model_id = model_dir.name
            rel_path = str(model_path.relative_to(system_dir)) if model_path.is_absolute() else str(model_path)
            params = _filter_fit_params(args)
            params.setdefault("fit_mode", "standard")
            params = _relativize_param_paths(params, system_dir)
            pre_model_meta_path = model_dir / "model_metadata.json"
            _write_model_metadata(
                model_dir=model_dir,
                model_meta={
                    "model_id": model_id,
                    "name": args.model_name or model_id,
                    "path": rel_path,
                    "created_at": datetime.utcnow().isoformat(),
                    "source": args.model_source or "offline",
                    "params": params,
                },
            )
        except Exception:
            pre_model_meta_path = None
    try:
        results = sim_main.run_pipeline(args, parser=parser)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1
    if project_id and system_id and cluster_id:
        model_path = Path(results.get("model_path")) if results else None
        if model_path:
            params = _filter_fit_params(args)
            params.setdefault("fit_mode", "standard")
            store = ProjectStore(base_dir=Path(os.getenv("PHASE_DATA_ROOT", "/app/data")) / "projects")
            system_dir = store.ensure_directories(project_id, system_id)["system_dir"]
            params = _relativize_param_paths(params, system_dir)
            resume_path = None
            if args.plm_resume_model:
                resume_path = Path(args.plm_resume_model)
                if not resume_path.is_absolute():
                    resume_path = store.resolve_path(project_id, system_id, str(args.plm_resume_model))
            if resume_path and resume_path.resolve() == model_path.resolve():
                model_dir = model_path.parent
                model_id = model_dir.name
                try:
                    rel_path = str(model_path.relative_to(system_dir))
                except Exception:
                    rel_path = str(model_path)
                model_name = args.model_name or model_id
                try:
                    system_meta = store.get_system(project_id, system_id)
                except FileNotFoundError:
                    system_meta = None
                if system_meta:
                    entry = next(
                        (c for c in (system_meta.metastable_clusters or []) if c.get("cluster_id") == cluster_id),
                        None,
                    )
                    if isinstance(entry, dict):
                        models = entry.get("potts_models") or []
                        updated = False
                        for model in models:
                            rel_path = model.get("path")
                            if not rel_path:
                                continue
                            abs_path = store.resolve_path(project_id, system_id, rel_path)
                            if abs_path.resolve() == resume_path.resolve():
                                model["params"] = params
                                updated = True
                                model_name = model.get("name") or model_name
                                break
                        if updated:
                            entry["potts_models"] = models
                            store.save_system(system_meta)
                            _write_model_metadata(
                                model_dir=model_dir,
                                model_meta={
                                    "model_id": model_id,
                                    "name": model_name,
                                    "path": rel_path,
                                    "created_at": datetime.utcnow().isoformat(),
                                    "source": args.model_source or "offline",
                                    "params": params,
                                },
                            )
                            return 0
                _write_model_metadata(
                    model_dir=model_dir,
                    model_meta={
                        "model_id": model_id,
                        "name": model_name,
                        "path": rel_path,
                        "created_at": datetime.utcnow().isoformat(),
                        "source": args.model_source or "offline",
                        "params": params,
                    },
                )
                return 0
            model_id_override = None
            try:
                dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
                potts_models_dir = dirs["potts_models_dir"]
                if model_path.is_absolute():
                    resolved = model_path.resolve()
                else:
                    resolved = (system_dir / model_path).resolve()
                if resolved.is_relative_to(potts_models_dir.resolve()):
                    model_id_override = resolved.parent.name
            except Exception:
                model_id_override = None
            persist_model(
                project_id=project_id,
                system_id=system_id,
                cluster_id=cluster_id,
                model_path=model_path,
                model_name=args.model_name or None,
                params=params,
                source=args.model_source or "offline",
                model_id=model_id_override,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
