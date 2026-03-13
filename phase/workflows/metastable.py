"""
Metastable recomputation service.

Metastable discovery is scoped to a single uploaded state at a time. Outputs are
stored under that state's folder and then re-exposed as derived analysis states.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from phase.analysis.vamp_pipeline import run_metastable_pipeline_for_system
from phase.services.project_store import DescriptorState, ProjectStore, SystemMetadata
from phase.services.state_utils import build_analysis_states


def _rel_or_none(path: Optional[Path], root: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _build_state_spec(project_id: str, system: SystemMetadata, state: DescriptorState, store: ProjectStore) -> Dict[str, Any]:
    desc_path = store.resolve_path(project_id, system.system_id, state.descriptor_file)
    traj_path = store.resolve_path(project_id, system.system_id, state.trajectory_file) if state.trajectory_file else None
    pdb_path = store.resolve_path(project_id, system.system_id, state.pdb_file) if state.pdb_file else None
    return {
        "trajectory_id": state.state_id,
        "macro_state": state.name,
        "macro_state_id": state.state_id,
        "descriptor_path": str(desc_path),
        "trajectory_path": str(traj_path) if traj_path and traj_path.exists() else None,
        "topology_path": str(pdb_path) if pdb_path and pdb_path.exists() else None,
    }


def recompute_metastable_states(
    project_id: str,
    system_id: str,
    *,
    state_id: str,
    n_microstates: int = 20,
    k_meta_min: int = 1,
    k_meta_max: int = 4,
    tica_lag_frames: int = 5,
    tica_dim: int = 5,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Run VAMP/TICA metastable discovery for one descriptor-ready state.
    """
    store = ProjectStore()
    system = store.get_system(project_id, system_id)
    state = system.states.get(state_id)
    if not state:
        raise ValueError(f"State '{state_id}' not found.")
    if not state.descriptor_file:
        raise ValueError("Selected state has no descriptors.")

    state_dirs = store.ensure_state_directories(
        project_id,
        system_id,
        state.state_id,
        storage_key=state.storage_key or state.state_id,
    )
    system_dir = state_dirs["system_dir"]
    output_dir = state_dirs["metastable_dir"]
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_metastable_pipeline_for_system(
        [_build_state_spec(project_id, system, state, store)],
        output_dir=output_dir,
        n_microstates=n_microstates,
        k_meta_min=k_meta_min,
        k_meta_max=k_meta_max,
        tica_lag_frames=tica_lag_frames,
        tica_dim=tica_dim,
        random_state=random_state,
    )

    new_metastables: List[Dict[str, Any]] = []
    for macro_res in results.get("macro_results", []):
        label_map = macro_res.get("labels_per_trajectory") or {}
        label_path = label_map.get(state.state_id)
        if label_path:
            state.metastable_labels_file = _rel_or_none(Path(label_path), system_dir)

        for meta in macro_res.get("metastable_states", []):
            meta_copy = dict(meta)
            rep = meta_copy.get("representative_pdb")
            if rep:
                meta_copy["representative_pdb"] = _rel_or_none(Path(rep), system_dir)
            meta_copy["macro_state_id"] = state.state_id
            meta_copy["macro_state"] = state.name
            meta_copy.setdefault("name", meta_copy.get("default_name") or meta_copy.get("metastable_id"))
            new_metastables.append(meta_copy)

    retained = [
        meta
        for meta in (system.metastable_states or [])
        if str(meta.get("macro_state_id") or "") != str(state.state_id)
    ]
    system.metastable_states = retained + new_metastables
    system.metastable_model_dir = None
    system.analysis_states = build_analysis_states(system)

    meta_payload = {
        "state_id": state.state_id,
        "state_name": state.name,
        "storage_key": state.storage_key or state.state_id,
        "model_dir": _rel_or_none(output_dir, system_dir),
        "metastable_labels_file": state.metastable_labels_file,
        "params": {
            "n_microstates": int(n_microstates),
            "k_meta_min": int(k_meta_min),
            "k_meta_max": int(k_meta_max),
            "tica_lag_frames": int(tica_lag_frames),
            "tica_dim": int(tica_dim),
            "random_state": int(random_state),
        },
        "metastable_states": new_metastables,
    }
    meta_path = store._state_metastable_metadata_path(project_id, system_id, state)  # local storage API
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    state.metastable_metadata_file = str(meta_path.relative_to(system_dir))
    store.save_system(system)

    return {
        "state_id": state.state_id,
        "state_name": state.name,
        "metastable_states": new_metastables,
        "model_dir": meta_payload["model_dir"],
    }
