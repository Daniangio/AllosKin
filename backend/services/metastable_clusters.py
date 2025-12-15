"""Cluster per-residue angles inside selected metastable states and persist as NPZ."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from backend.services.descriptors import load_descriptor_npz
from backend.services.project_store import DescriptorState, ProjectStore, SystemMetadata


def _slug(value: str) -> str:
    """Create a filesystem/NPZ-safe slug."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_") or "metastable"


def _cluster_residue_samples(
    samples: np.ndarray, max_k: int, random_state: int
) -> Tuple[np.ndarray, int]:
    """Pick k via silhouette (k>=1) and return labels + cluster count."""
    if samples.size == 0:
        return np.array([], dtype=np.int32), 0

    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.ndim != 2 or samples.shape[1] < 3:
        raise ValueError("Residue samples must be (n_frames, >=3) shaped.")

    # Use sin/cos embedding to respect angular periodicity.
    angles = samples[:, :3]
    emb = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

    upper_k = max(1, min(int(max_k), emb.shape[0]))
    best_k = 1
    best_score = -np.inf

    for k in range(1, upper_k + 1):
        if k == 1:
            score = 0.0
        else:
            km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
            labels = km.fit_predict(emb)
            if len(np.unique(labels)) < 2:
                score = -np.inf
            else:
                score = silhouette_score(emb, labels)
        if score > best_score:
            best_score = score
            best_k = k

    if best_k == 1:
        final_labels = np.zeros(emb.shape[0], dtype=np.int32)
    else:
        km = KMeans(n_clusters=best_k, n_init="auto", random_state=random_state)
        final_labels = km.fit_predict(emb).astype(np.int32)

    return final_labels, int(best_k)


def _resolve_states_for_meta(meta: Dict[str, Any], system: SystemMetadata) -> List[DescriptorState]:
    """Return all states contributing to a metastable macro-state."""
    macro_state_id = meta.get("macro_state_id")
    macro_state_name = meta.get("macro_state")
    states: List[DescriptorState] = []
    for st in system.states.values():
        if macro_state_id and st.state_id == macro_state_id:
            states.append(st)
        elif macro_state_name and st.name == macro_state_name:
            states.append(st)
    return states


def _extract_labels_for_state(
    store: ProjectStore,
    project_id: str,
    system_id: str,
    state: DescriptorState,
    features: Dict[str, Any],
) -> np.ndarray:
    """Extract metastable labels array, preferring embedded NPZ key."""
    labels = features.pop("metastable_labels", None)
    if labels is not None:
        labels = np.asarray(labels)
    elif state.metastable_labels_file:
        label_path = store.resolve_path(project_id, system_id, state.metastable_labels_file)
        if label_path.exists():
            labels = np.load(label_path)
    if labels is None:
        raise ValueError(f"No metastable labels found for state '{state.state_id}'.")
    return np.asarray(labels).astype(np.int32)


def _coerce_residue_keys(
    residue_keys: List[str], features: Dict[str, Any], state: DescriptorState
) -> List[str]:
    """Prefer stored residue_keys but fall back to feature keys."""
    if residue_keys:
        return residue_keys
    if state.residue_keys:
        return list(state.residue_keys)
    return [k for k in features.keys() if k != "metastable_labels"]


def generate_metastable_cluster_npz(
    project_id: str,
    system_id: str,
    metastable_ids: List[str],
    *,
    max_clusters_per_residue: int = 6,
    random_state: int = 0,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Build per-residue cluster labels for selected metastable states and save NPZ.

    Returns the path to the NPZ and a metadata dictionary.
    """
    if not metastable_ids:
        raise ValueError("At least one metastable_id is required.")
    if max_clusters_per_residue < 1:
        raise ValueError("max_clusters_per_residue must be >= 1.")

    unique_meta_ids = list(dict.fromkeys([str(mid) for mid in metastable_ids]))

    store = ProjectStore()
    system = store.get_system(project_id, system_id)
    metastable_lookup = {m.get("metastable_id"): m for m in system.metastable_states or []}

    residue_keys: List[str] = []
    residue_mapping: Dict[str, str] = {}
    merged_angles_per_residue: List[List[np.ndarray]] = []
    merged_frame_state_ids: List[str] = []
    merged_frame_meta_ids: List[str] = []
    per_meta_results: Dict[str, Dict[str, Any]] = {}

    for meta_id in unique_meta_ids:
        meta = metastable_lookup.get(meta_id)
        if not meta:
            raise ValueError(f"Metastable state '{meta_id}' not found on this system.")
        meta_index = meta.get("metastable_index")
        if meta_index is None:
            raise ValueError(f"Metastable state '{meta_id}' is missing its index.")

        candidate_states = _resolve_states_for_meta(meta, system)
        if not candidate_states:
            raise ValueError(f"No descriptor-ready states found for metastable '{meta_id}'.")

        per_meta_angles: List[List[np.ndarray]] = []
        frame_state_ids: List[str] = []

        for state in candidate_states:
            if not state.descriptor_file:
                continue
            desc_path = store.resolve_path(project_id, system_id, state.descriptor_file)
            features = load_descriptor_npz(desc_path)
            labels = _extract_labels_for_state(store, project_id, system_id, state, features)

            # Lock residue ordering on first valid state.
            residue_keys = _coerce_residue_keys(residue_keys, features, state)
            if not residue_keys:
                raise ValueError("Could not determine residue keys for clustering.")
            if not residue_mapping:
                residue_mapping = dict(state.residue_mapping or system.residue_selections_mapping or {})

            if not merged_angles_per_residue:
                merged_angles_per_residue = [[] for _ in residue_keys]
            if not per_meta_angles:
                per_meta_angles = [[] for _ in residue_keys]

            if labels.shape[0] == 0:
                continue
            mask = labels == int(meta_index)
            if not np.any(mask):
                continue

            matched_indices = np.where(mask)[0]
            for idx in matched_indices:
                frame_state_ids.append(state.state_id)
                merged_frame_state_ids.append(state.state_id)
                merged_frame_meta_ids.append(meta_id)
                for col, key in enumerate(residue_keys):
                    arr = np.asarray(features.get(key))
                    if arr is None or arr.shape[0] != labels.shape[0]:
                        raise ValueError(f"Descriptor array for '{key}' is missing or misaligned in state '{state.state_id}'.")
                    # Expected (n_frames, 1, 3) radians; fall back gracefully.
                    if arr.ndim >= 3:
                        vec = arr[idx, 0, :3]
                    elif arr.ndim == 2:
                        vec = arr[idx, :3]
                    else:
                        vec = arr[idx : idx + 1]
                    vec = np.asarray(vec, dtype=float).reshape(-1)
                    if vec.size < 3:
                        padded = np.zeros(3, dtype=float)
                        padded[: vec.size] = vec
                        vec = padded
                    else:
                        vec = vec[:3]
                    per_meta_angles[col].append(vec)
                    merged_angles_per_residue[col].append(vec)

        n_frames = len(frame_state_ids)
        if n_frames == 0:
            raise ValueError(f"No frames matched metastable '{meta_id}'.")

        labels_matrix = np.zeros((n_frames, len(residue_keys)), dtype=np.int32)
        cluster_counts = np.zeros(len(residue_keys), dtype=np.int32)

        for col, samples in enumerate(per_meta_angles):
            sample_arr = np.asarray(samples, dtype=float)
            if sample_arr.shape[0] != n_frames:
                raise ValueError(
                    f"Residue '{residue_keys[col]}' for metastable '{meta_id}' has inconsistent frame count."
                )
            labels_arr, k = _cluster_residue_samples(sample_arr, max_clusters_per_residue, random_state)
            if labels_arr.size == 0:
                labels_matrix[:, col] = -1
                cluster_counts[col] = 0
            else:
                labels_matrix[:, col] = labels_arr
                cluster_counts[col] = k

        per_meta_results[meta_id] = {
            "labels": labels_matrix,
            "cluster_counts": cluster_counts,
            "frame_state_ids": np.array(frame_state_ids),
            "macro_state": meta.get("macro_state"),
            "metastable_index": int(meta_index),
        }

    # Build merged clusters
    if not merged_angles_per_residue:
        raise ValueError("No frames gathered across the selected metastable states.")

    merged_frame_count = len(merged_frame_state_ids)
    merged_labels = np.zeros((merged_frame_count, len(residue_keys)), dtype=np.int32)
    merged_counts = np.zeros(len(residue_keys), dtype=np.int32)
    for col, samples in enumerate(merged_angles_per_residue):
        sample_arr = np.asarray(samples, dtype=float)
        if sample_arr.shape[0] != merged_frame_count:
            raise ValueError("Merged residue samples have inconsistent frame counts.")
        labels_arr, k = _cluster_residue_samples(sample_arr, max_clusters_per_residue, random_state)
        if labels_arr.size == 0:
            merged_labels[:, col] = -1
            merged_counts[col] = 0
        else:
            merged_labels[:, col] = labels_arr
            merged_counts[col] = k

    # Persist NPZ
    dirs = store.ensure_directories(project_id, system_id)
    cluster_dir = dirs["system_dir"] / "metastable" / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    suffix = "-".join(_slug(mid)[:24] for mid in unique_meta_ids) or "metastable"
    out_path = cluster_dir / f"{suffix}_clusters_{timestamp}.npz"

    metadata = {
        "project_id": project_id,
        "system_id": system_id,
        "generated_at": datetime.utcnow().isoformat(),
        "selected_metastable_ids": unique_meta_ids,
        "residue_keys": residue_keys,
        "residue_mapping": residue_mapping,
        "max_clusters_per_residue": max_clusters_per_residue,
        "random_state": random_state,
        "per_metastable": {
            mid: {
                "n_frames": res["labels"].shape[0],
                "metastable_index": res["metastable_index"],
                "macro_state": res["macro_state"],
                "state_ids": sorted(set(res["frame_state_ids"].tolist())),
                "npz_keys": {
                    "labels": f"{_slug(mid)}__labels",
                    "cluster_counts": f"{_slug(mid)}__cluster_counts",
                    "frame_state_ids": f"{_slug(mid)}__frame_state_ids",
                },
            }
            for mid, res in per_meta_results.items()
        },
        "merged": {
            "n_frames": merged_frame_count,
            "npz_keys": {
                "labels": "merged__labels",
                "cluster_counts": "merged__cluster_counts",
                "frame_state_ids": "merged__frame_state_ids",
                "frame_metastable_ids": "merged__frame_metastable_ids",
            },
        },
    }

    payload: Dict[str, Any] = {
        "residue_keys": np.array(residue_keys),
        "metadata_json": np.array(json.dumps(metadata)),
        "merged__labels": merged_labels,
        "merged__cluster_counts": merged_counts,
        "merged__frame_state_ids": np.array(merged_frame_state_ids),
        "merged__frame_metastable_ids": np.array(merged_frame_meta_ids),
    }
    for mid, res in per_meta_results.items():
        key = _slug(mid)
        payload[f"{key}__labels"] = res["labels"]
        payload[f"{key}__cluster_counts"] = res["cluster_counts"]
        payload[f"{key}__frame_state_ids"] = res["frame_state_ids"]

    np.savez_compressed(out_path, **payload)
    return out_path, metadata
