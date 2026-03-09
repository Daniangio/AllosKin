from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from backend.api.v1.common import project_store


ANALYSIS_METADATA_FILENAME = "analysis_metadata.json"

_DIRECT_SAMPLE_ANALYSIS_TYPES = {
    "md_vs_sample",
    "model_energy",
    "delta_eval",
    "delta_transition",
    "gibbs_relaxation",
    "ligand_completion",
    "lambda_sweep",
}

_MULTI_SAMPLE_ANALYSIS_TYPES = {
    "delta_commitment",
    "delta_js",
}


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if "," in raw:
            return [part.strip() for part in raw.split(",") if part.strip()]
        return [raw]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    return [s] if s else []


def _extract_sample_refs(meta: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    direct: set[str] = set()
    essential: set[str] = set()
    optional_many: set[str] = set()

    for key in (
        "sample_id",
        "md_sample_id",
        "active_md_sample_id",
        "inactive_md_sample_id",
        "pas_md_sample_id",
        "start_sample_id",
        "reference_sample_id_a",
        "reference_sample_id_b",
        "constraint_delta_js_sample_id",
    ):
        values = _coerce_str_list(meta.get(key))
        direct.update(values)
        essential.update(values)

    for key in ("md_sample_ids", "reference_sample_ids_a", "reference_sample_ids_b"):
        values = _coerce_str_list(meta.get(key))
        direct.update(values)
        essential.update(values)

    summary = meta.get("summary") if isinstance(meta.get("summary"), dict) else {}
    optional_many.update(_coerce_str_list(summary.get("sample_ids")))

    potts_overlay = meta.get("potts_overlay") if isinstance(meta.get("potts_overlay"), dict) else {}
    optional_many.update(_coerce_str_list(potts_overlay.get("sample_ids_a")))
    optional_many.update(_coerce_str_list(potts_overlay.get("sample_ids_b")))

    return direct | optional_many, essential, optional_many


def _extract_model_refs(meta: dict[str, Any]) -> tuple[set[str], set[str]]:
    refs: set[str] = set()
    essential: set[str] = set()
    for key in ("model_id", "model_a_id", "model_b_id"):
        values = _coerce_str_list(meta.get(key))
        refs.update(values)
        essential.update(values)
    return refs, essential


def _analysis_is_orphan(meta: dict[str, Any], sample_ids: set[str], model_ids: set[str]) -> bool:
    analysis_type = str(meta.get("analysis_type") or "").strip().lower()
    sample_refs, essential_sample_refs, optional_sample_refs = _extract_sample_refs(meta)
    _all_model_refs, essential_model_refs = _extract_model_refs(meta)

    if essential_model_refs and any(ref not in model_ids for ref in essential_model_refs):
        return True
    if essential_sample_refs and any(ref not in sample_ids for ref in essential_sample_refs):
        return True

    if analysis_type in _DIRECT_SAMPLE_ANALYSIS_TYPES:
        if sample_refs and any(ref not in sample_ids for ref in sample_refs):
            return True
        return False

    if analysis_type in _MULTI_SAMPLE_ANALYSIS_TYPES:
        if optional_sample_refs and not any(ref in sample_ids for ref in optional_sample_refs):
            return True
        return False

    return False


def cleanup_orphan_cluster_analyses(project_id: str, system_id: str, cluster_id: str) -> int:
    cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
    analyses_root = cluster_dirs["cluster_dir"] / "analyses"
    if not analyses_root.exists():
        return 0

    sample_ids = {
        str(entry.get("sample_id"))
        for entry in project_store.list_samples(project_id, system_id, cluster_id)
        if isinstance(entry, dict) and entry.get("sample_id")
    }
    model_ids = {
        str(entry.get("model_id"))
        for entry in project_store.list_potts_models(project_id, system_id, cluster_id)
        if isinstance(entry, dict) and entry.get("model_id")
    }

    removed = 0
    for kind_dir in sorted((p for p in analyses_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        for analysis_dir in sorted((p for p in kind_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME
            npz_path = analysis_dir / "analysis.npz"
            orphan = False
            if not meta_path.exists() or not npz_path.exists():
                orphan = True
            else:
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    orphan = True
                else:
                    orphan = _analysis_is_orphan(meta, sample_ids=sample_ids, model_ids=model_ids)
            if not orphan:
                continue
            shutil.rmtree(analysis_dir, ignore_errors=True)
            removed += 1

        try:
            if not any(kind_dir.iterdir()):
                kind_dir.rmdir()
        except Exception:
            pass
    return removed


def cleanup_orphan_analyses_all_systems() -> int:
    removed = 0
    for project in project_store.list_projects():
        project_id = str(project.project_id)
        for system in project_store.list_systems(project_id):
            system_id = str(system.system_id)
            for entry in project_store.list_cluster_entries(project_id, system_id):
                cluster_id = str(entry.get("cluster_id") or "").strip()
                if not cluster_id:
                    continue
                removed += cleanup_orphan_cluster_analyses(project_id, system_id, cluster_id)
    return removed
