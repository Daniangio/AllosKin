import shutil
from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from backend.api.v1.analysis_cleanup import cleanup_state_linked_artifacts
from backend.api.v1.common import (
    project_store,
    serialize_system,
)
from phase.workflows.metastable import recompute_metastable_states
from phase.services.state_utils import build_analysis_states


router = APIRouter()


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/recompute",
    summary="Recompute metastable states for one descriptor-ready state",
)
async def recompute_metastable(
    project_id: str,
    system_id: str,
    state_id: str = Query(..., description="Source macro state to analyze with VAMP/TICA."),
    n_microstates: int = Query(20, ge=2, le=500),
    k_meta_min: int = Query(1, ge=1, le=10),
    k_meta_max: int = Query(4, ge=1, le=10),
    tica_lag_frames: int = Query(5, ge=1),
    tica_dim: int = Query(5, ge=1),
    random_state: int = Query(0),
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    try:
        result = await run_in_threadpool(
            recompute_metastable_states,
            project_id,
            system_id,
            state_id=state_id,
            n_microstates=n_microstates,
            k_meta_min=k_meta_min,
            k_meta_max=max(k_meta_min, k_meta_max),
            tica_lag_frames=tica_lag_frames,
            tica_dim=tica_dim,
            random_state=random_state,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Metastable recompute failed: {exc}") from exc

    return result


@router.get(
    "/projects/{project_id}/systems/{system_id}/metastable",
    summary="List metastable states for a system",
)
async def list_metastable_states(project_id: str, system_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    return {
        "metastable_states": system_meta.metastable_states or [],
        "model_dir": system_meta.metastable_model_dir,
        "by_state": {
            str(state_id): [
                meta for meta in (system_meta.metastable_states or []) if str(meta.get("macro_state_id") or "") == str(state_id)
            ]
            for state_id in (system_meta.states or {}).keys()
        },
    }


@router.post(
    "/projects/{project_id}/systems/{system_id}/metastable/clear",
    summary="Clear metastable states, labels, and clusters",
)
async def clear_metastable_states(
    project_id: str,
    system_id: str,
    state_id: str | None = Query(None, description="Optional source state to clear only one metastable run."),
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    target_states = []
    if state_id:
        state = system_meta.states.get(state_id)
        if not state:
            raise HTTPException(status_code=404, detail=f"State '{state_id}' not found.")
        target_states = [state]
    else:
        target_states = list(system_meta.states.values())

    for state in target_states:
        if state.metastable_labels_file:
            label_path = project_store.resolve_path(project_id, system_id, state.metastable_labels_file)
            try:
                label_path.unlink(missing_ok=True)
            except Exception:
                pass
            state.metastable_labels_file = None
        if state.metastable_metadata_file:
            meta_path = project_store.resolve_path(project_id, system_id, state.metastable_metadata_file)
            try:
                shutil.rmtree(meta_path.parent, ignore_errors=True)
            except Exception:
                pass
            state.metastable_metadata_file = None

        if state.descriptor_file:
            descriptor_path = project_store.resolve_path(project_id, system_id, state.descriptor_file)
            if descriptor_path.exists():
                try:
                    npz = np.load(descriptor_path, allow_pickle=True)
                    if "metastable_labels" in npz.files:
                        data = {k: npz[k] for k in npz.files if k != "metastable_labels"}
                        tmp_path = descriptor_path.with_suffix(".tmp.npz")
                        np.savez_compressed(tmp_path, **data)
                        tmp_path.replace(descriptor_path)
                except Exception:
                    pass

    if state_id:
        system_meta.metastable_states = [
            meta
            for meta in (system_meta.metastable_states or [])
            if str(meta.get("macro_state_id") or "") != str(state_id)
        ]
    else:
        for cluster in system_meta.metastable_clusters or []:
            rel_path = cluster.get("path")
            if not rel_path:
                continue
            abs_path = project_store.resolve_path(project_id, system_id, rel_path)
            try:
                abs_path.unlink(missing_ok=True)
            except Exception:
                pass
        system_meta.metastable_states = []
        system_meta.metastable_clusters = []
    system_meta.metastable_model_dir = None
    if not system_meta.metastable_states:
        system_meta.metastable_locked = False
        system_meta.analysis_mode = "macro"
    system_meta.analysis_states = build_analysis_states(system_meta)
    project_store.save_system(system_meta)
    cleanup = cleanup_state_linked_artifacts(project_id, system_id)
    return {
        **serialize_system(system_meta),
        "cleanup_summary": cleanup,
    }


def _unlink_system_path(project_id: str, system_id: str, rel_path: str | None) -> None:
    if not rel_path:
        return
    try:
        abs_path = project_store.resolve_path(project_id, system_id, rel_path)
    except Exception:
        return
    try:
        abs_path.unlink(missing_ok=True)
    except Exception:
        pass


@router.delete(
    "/projects/{project_id}/systems/{system_id}/metastable/{metastable_id}",
    summary="Delete a metastable state and cleanup linked artifacts",
)
async def delete_metastable_state(project_id: str, system_id: str, metastable_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    metas = list(system_meta.metastable_states or [])
    target = next((meta for meta in metas if str(meta.get("metastable_id") or meta.get("id") or "") == metastable_id), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"Metastable state '{metastable_id}' not found.")

    parent_state_id = str(target.get("macro_state_id") or "")
    _unlink_system_path(project_id, system_id, target.get("representative_pdb") or target.get("pdb_file"))

    remaining = [
        meta
        for meta in metas
        if str(meta.get("metastable_id") or meta.get("id") or "") != metastable_id
    ]
    system_meta.metastable_states = remaining

    state_obj = (system_meta.states or {}).get(parent_state_id)
    state_has_other_metas = any(str(meta.get("macro_state_id") or "") == parent_state_id for meta in remaining)
    if state_obj and not state_has_other_metas:
        _unlink_system_path(project_id, system_id, getattr(state_obj, "metastable_labels_file", None))
        state_obj.metastable_labels_file = None
        try:
            state_dirs = project_store.ensure_state_directories(
                project_id,
                system_id,
                state_obj.state_id,
                storage_key=state_obj.storage_key or state_obj.state_id,
            )
            shutil.rmtree(state_dirs["metastable_dir"], ignore_errors=True)
        except Exception:
            pass
        state_obj.metastable_metadata_file = None
        if state_obj.descriptor_file:
            descriptor_path = project_store.resolve_path(project_id, system_id, state_obj.descriptor_file)
            if descriptor_path.exists():
                try:
                    npz = np.load(descriptor_path, allow_pickle=True)
                    if "metastable_labels" in npz.files:
                        data = {k: npz[k] for k in npz.files if k != "metastable_labels"}
                        tmp_path = descriptor_path.with_suffix(".tmp.npz")
                        np.savez_compressed(tmp_path, **data)
                        tmp_path.replace(descriptor_path)
                except Exception:
                    pass

    if not system_meta.metastable_states:
        system_meta.metastable_locked = False
        if getattr(system_meta, "analysis_mode", None) == "metastable":
            system_meta.analysis_mode = None
        system_meta.metastable_model_dir = None

    system_meta.analysis_states = build_analysis_states(system_meta)
    project_store.save_system(system_meta)
    cleanup = cleanup_state_linked_artifacts(project_id, system_id)
    return {
        **serialize_system(system_meta),
        "cleanup_summary": cleanup,
    }

@router.get(
    "/projects/{project_id}/systems/{system_id}/metastable/{metastable_id}/pdb",
    summary="Download representative PDB for a metastable state",
)
async def fetch_metastable_pdb(project_id: str, system_id: str, metastable_id: str):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    metas = system_meta.metastable_states or []
    target = next((m for m in metas if m.get("metastable_id") == metastable_id), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"Metastable state '{metastable_id}' not found.")
    pdb_rel = target.get("representative_pdb")
    if not pdb_rel:
        raise HTTPException(status_code=404, detail="No representative PDB stored for this metastable state.")
    pdb_path = project_store.resolve_path(project_id, system_id, pdb_rel)
    if not pdb_path.exists():
        raise HTTPException(status_code=404, detail="Representative PDB file is missing on disk.")
    return FileResponse(pdb_path, filename=pdb_path.name, media_type="chemical/x-pdb")


@router.patch(
    "/projects/{project_id}/systems/{system_id}/metastable/{metastable_id}",
    summary="Rename a metastable state",
)
async def rename_metastable_state(
    project_id: str,
    system_id: str,
    metastable_id: str,
    payload: Dict[str, Any],
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found.")

    new_name = (payload or {}).get("name")
    if not new_name or not str(new_name).strip():
        raise HTTPException(status_code=400, detail="Name is required.")
    new_name = str(new_name).strip()

    updated = False
    metas = system_meta.metastable_states or []
    for meta in metas:
        if meta.get("metastable_id") == metastable_id:
            meta["name"] = new_name
            updated = True
            break

    if not updated:
        raise HTTPException(status_code=404, detail=f"Metastable state '{metastable_id}' not found.")

    system_meta.metastable_states = metas
    system_meta.analysis_states = build_analysis_states(system_meta)
    project_store.save_system(system_meta)
    return {"metastable_states": metas}
