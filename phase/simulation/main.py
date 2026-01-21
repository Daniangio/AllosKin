from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from phase.io.data import load_npz
from phase.simulation.potts_model import (
    compute_pseudolikelihood_loss_torch,
    fit_potts_pmi,
    fit_potts_pseudolikelihood_torch,
    load_potts_model,
    save_potts_model,
)
from phase.simulation.qubo import potts_to_qubo_onehot, decode_onehot
from phase.simulation.sampling import (
    gibbs_sample_potts,
    make_beta_ladder,
    replica_exchange_gibbs_potts,
    sa_sample_qubo_neal,
)
from phase.simulation.metrics import (
    marginals,
    pairwise_joints_on_edges,
    per_residue_js,
    combined_distance,
    pairwise_joints_padded,
    per_residue_js_from_padded,
    per_edge_js_from_padded,
)
from phase.simulation.plotting import plot_marginal_summary_from_npz, plot_sampling_report_from_npz


def _parse_float_list(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]

def _parse_beta_schedule(raw: str) -> tuple[float, float]:
    value = raw.strip()
    if not value:
        raise ValueError("Empty beta schedule.")
    if ":" in value:
        parts = [p.strip() for p in value.split(":", maxsplit=1)]
    elif "," in value:
        parts = [p.strip() for p in value.split(",", maxsplit=1)]
    else:
        parts = value.split()
    if len(parts) != 2:
        raise ValueError(f"Invalid beta schedule '{raw}'. Use 'hot,cold' or 'hot:cold'.")
    hot = float(parts[0])
    cold = float(parts[1])
    return hot, cold


def _ensure_results_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _pad_marginals_for_save(margs: Sequence[np.ndarray]) -> np.ndarray:
    """
    Pad variable-length marginal arrays into a dense matrix with NaN fill.
    This avoids object dtypes so summaries can be loaded with allow_pickle=False.
    """
    if len(margs) == 0:
        return np.zeros((0, 0), dtype=float)
    max_k = max(len(p) for p in margs)
    out = np.full((len(margs), max_k), np.nan, dtype=float)
    for i, p in enumerate(margs):
        out[i, : len(p)] = p
    return out


def _load_assigned_labels(path: Path) -> np.ndarray | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "assigned__labels" in data:
                return np.asarray(data["assigned__labels"], dtype=int)
    except Exception:
        return None
    return None


def _build_md_sources(
    labels: np.ndarray,
    K: Sequence[int],
    edges: Sequence[tuple[int, int]],
    frame_state_ids: np.ndarray | None,
    frame_metastable_ids: np.ndarray | None,
    metadata: dict | None,
    npz_path: str | Path,
) -> list[dict]:
    sources: list[dict] = []
    meta = metadata or {}
    state_labels = meta.get("state_labels") or {}
    metastable_labels = meta.get("metastable_labels") or {}
    assigned_state_paths = meta.get("assigned_state_paths") or {}
    assigned_meta_paths = meta.get("assigned_metastable_paths") or {}
    metastable_kinds = meta.get("metastable_kinds") or {}
    analysis_mode = meta.get("analysis_mode")

    base_dir = Path(npz_path).resolve().parent

    def add_source(source_id: str, label: str, source_type: str, subset: np.ndarray) -> None:
        if subset.size == 0:
            return
        sources.append(
            {
                "id": source_id,
                "label": label,
                "type": source_type,
                "count": int(subset.shape[0]),
                "labels": subset,
                "p": _pad_marginals_for_save(marginals(subset, K)),
                "p2": pairwise_joints_padded(subset, K, edges),
            }
        )

    for state_id, rel_path in assigned_state_paths.items():
        abs_path = Path(rel_path)
        if not abs_path.is_absolute():
            abs_path = base_dir / rel_path
        assigned = _load_assigned_labels(abs_path)
        if assigned is None:
            continue
        label = state_labels.get(str(state_id), str(state_id))
        add_source(f"state:{state_id}", f"Macro: {label}", "macro", assigned)

    if analysis_mode != "macro":
        for meta_id, rel_path in assigned_meta_paths.items():
            if metastable_kinds.get(str(meta_id)) == "macro":
                continue
            abs_path = Path(rel_path)
            if not abs_path.is_absolute():
                abs_path = base_dir / rel_path
            assigned = _load_assigned_labels(abs_path)
            if assigned is None:
                continue
            label = metastable_labels.get(str(meta_id), str(meta_id))
            add_source(f"meta:{meta_id}", f"Metastable: {label}", "metastable", assigned)

    if not sources:
        fallback = labels
        add_source("md_clustered", "MD (clustered frames)", "merged", fallback)
        state_id_set = set()
        if frame_state_ids is not None and frame_state_ids.shape[0] == labels.shape[0]:
            state_id_set = set(map(str, np.unique(frame_state_ids)))
            for state_id in np.unique(frame_state_ids):
                label = state_labels.get(str(state_id), str(state_id))
                mask = frame_state_ids == state_id
                add_source(f"state:{state_id}", f"Macro: {label}", "macro", labels[mask])

        if analysis_mode != "macro" and frame_metastable_ids is not None and frame_metastable_ids.shape[0] == labels.shape[0]:
            for meta_id in np.unique(frame_metastable_ids):
                if metastable_kinds.get(str(meta_id)) == "macro":
                    continue
                if str(meta_id) in state_id_set:
                    continue
                label = metastable_labels.get(str(meta_id), str(meta_id))
                mask = frame_metastable_ids == meta_id
                add_source(f"meta:{meta_id}", f"Metastable: {label}", "metastable", labels[mask])

    return sources


def _build_sample_sources(
    X_gibbs: np.ndarray,
    sa_samples: Sequence[np.ndarray],
    sa_schedule_labels: Sequence[str] | None,
    K: Sequence[int],
    edges: Sequence[tuple[int, int]],
    *,
    gibbs_label: str,
) -> list[dict]:
    sources: list[dict] = []

    def add_source(source_id: str, label: str, source_type: str, X: np.ndarray) -> None:
        if X.size == 0:
            return
        sources.append(
            {
                "id": source_id,
                "label": label,
                "type": source_type,
                "count": int(X.shape[0]),
                "X": X,
                "p": _pad_marginals_for_save(marginals(X, K)),
                "p2": pairwise_joints_padded(X, K, edges),
            }
        )

    add_source("gibbs", gibbs_label, "gibbs", X_gibbs)

    for idx, X_sa in enumerate(sa_samples):
        label = (
            str(sa_schedule_labels[idx])
            if sa_schedule_labels is not None and idx < len(sa_schedule_labels)
            else f"SA {idx + 1}"
        )
        add_source(f"sa_{idx}", label, "sa", X_sa)

    return sources


def _compute_energy_histograms(
    *,
    model,
    md_sources: Sequence[dict],
    sample_sources: Sequence[dict],
    n_bins: int = 40,
) -> dict:
    if model is None:
        return {
            "bins": np.array([], dtype=float),
            "hist_md": np.zeros((0, 0), dtype=float),
            "cdf_md": np.zeros((0, 0), dtype=float),
            "hist_sample": np.zeros((0, 0), dtype=float),
            "cdf_sample": np.zeros((0, 0), dtype=float),
        }

    md_energies = [model.energy_batch(src["labels"]) for src in md_sources]
    sample_energies = [model.energy_batch(src["X"]) for src in sample_sources]
    if not sample_energies:
        return {
            "bins": np.array([], dtype=float),
            "hist_md": np.zeros((0, 0), dtype=float),
            "cdf_md": np.zeros((0, 0), dtype=float),
            "hist_sample": np.zeros((0, 0), dtype=float),
            "cdf_sample": np.zeros((0, 0), dtype=float),
        }

    min_e = float(min([e.min() for e in md_energies] + [e.min() for e in sample_energies]))
    max_e = float(max([e.max() for e in md_energies] + [e.max() for e in sample_energies]))
    if not np.isfinite(min_e) or not np.isfinite(max_e) or min_e == max_e:
        bins = np.linspace(min_e - 1.0, max_e + 1.0, n_bins + 1)
    else:
        pad = 0.05 * (max_e - min_e)
        bins = np.linspace(min_e - pad, max_e + pad, n_bins + 1)

    hist_md = []
    cdf_md = []
    for energies in md_energies:
        counts, _ = np.histogram(energies, bins=bins, density=True)
        hist_md.append(counts)
        cdf = np.cumsum(counts)
        cdf = cdf / cdf[-1] if cdf.size and cdf[-1] > 0 else cdf
        cdf_md.append(cdf)

    hist_sample = []
    cdf_sample = []
    for energies in sample_energies:
        counts, _ = np.histogram(energies, bins=bins, density=True)
        hist_sample.append(counts)
        cdf = np.cumsum(counts)
        cdf = cdf / cdf[-1] if cdf.size and cdf[-1] > 0 else cdf
        cdf_sample.append(cdf)

    return {
        "bins": bins,
        "hist_md": np.asarray(hist_md, dtype=float),
        "cdf_md": np.asarray(cdf_md, dtype=float),
        "hist_sample": np.asarray(hist_sample, dtype=float),
        "cdf_sample": np.asarray(cdf_sample, dtype=float),
    }


def _subsample_rows(X: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if X.shape[0] <= max_rows:
        return X
    idx = rng.choice(X.shape[0], size=max_rows, replace=False)
    return X[idx]


def _min_hamming_distances(A: np.ndarray, B: np.ndarray, *, block_size: int = 256) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.array([], dtype=int)
    mins = np.full(A.shape[0], np.iinfo(np.int32).max, dtype=int)
    for start in range(0, B.shape[0], block_size):
        chunk = B[start : start + block_size]
        dists = np.sum(A[:, None, :] != chunk[None, :, :], axis=2)
        mins = np.minimum(mins, dists.min(axis=1))
    return mins


def _compute_nn_cdfs(
    *,
    md_sources: Sequence[dict],
    sample_sources: Sequence[dict],
    max_md: int = 2000,
    max_sample: int = 1000,
    block_size: int = 256,
) -> dict:
    if not md_sources or not sample_sources:
        return {
            "bins": np.array([], dtype=int),
            "cdf_sample_to_md": np.zeros((0, 0, 0), dtype=float),
            "cdf_md_to_sample": np.zeros((0, 0, 0), dtype=float),
        }

    rng = np.random.default_rng(0)
    n_res = md_sources[0]["labels"].shape[1]
    bins = np.arange(n_res + 1, dtype=int)

    cdf_sample_to_md = np.zeros((len(md_sources), len(sample_sources), len(bins)), dtype=float)
    cdf_md_to_sample = np.zeros((len(md_sources), len(sample_sources), len(bins)), dtype=float)

    for i, md in enumerate(md_sources):
        X_md = md["labels"]
        X_md = _subsample_rows(X_md, max_md, rng)
        for j, sample in enumerate(sample_sources):
            X_samp = _subsample_rows(sample["X"], max_sample, rng)
            if X_md.size == 0 or X_samp.size == 0:
                continue
            d_sm = _min_hamming_distances(X_samp, X_md, block_size=block_size)
            d_ms = _min_hamming_distances(X_md, X_samp, block_size=block_size)

            hist_sm = np.bincount(d_sm, minlength=len(bins))
            hist_ms = np.bincount(d_ms, minlength=len(bins))
            cdf_sample_to_md[i, j] = np.cumsum(hist_sm) / max(1, hist_sm.sum())
            cdf_md_to_sample[i, j] = np.cumsum(hist_ms) / max(1, hist_ms.sum())

    return {
        "bins": bins,
        "cdf_sample_to_md": cdf_sample_to_md,
        "cdf_md_to_sample": cdf_md_to_sample,
    }


def _save_run_summary(
    results_dir: Path,
    args: argparse.Namespace,
    *,
    K: np.ndarray,
    edges: np.ndarray,
    residue_labels: np.ndarray,
    betas: list[float],
    X_gibbs: np.ndarray,
    X_sa: np.ndarray,
    p_md: np.ndarray,
    p_gibbs: np.ndarray,
    p_sa: np.ndarray,
    js_gibbs: np.ndarray,
    js_sa: np.ndarray,
    swap_accept_rate: np.ndarray | None,
    sa_valid_counts: np.ndarray,
    sa_invalid_mask: np.ndarray,
    sa_schedule_labels: list[str] | None = None,
    p_sa_by_schedule: np.ndarray | None = None,
    js_sa_by_schedule: np.ndarray | None = None,
    sa_valid_counts_by_schedule: np.ndarray | None = None,
    sa_invalid_mask_by_schedule: np.ndarray | None = None,
    p_gibbs_by_beta: np.ndarray | None = None,
    js_gibbs_by_beta: np.ndarray | None = None,
    js_pair_gibbs_by_beta: np.ndarray | None = None,
    beta_eff_grid: list[float] | None = None,
    beta_eff_distances: list[float] | None = None,
    beta_eff_value: float | None = None,
    beta_eff_by_schedule: list[float] | None = None,
    beta_eff_distances_by_schedule: np.ndarray | None = None,
    model_path: Path | None = None,
    p2_md: np.ndarray | None = None,
    p2_gibbs: np.ndarray | None = None,
    p2_sa: np.ndarray | None = None,
    js2_gibbs: np.ndarray | None = None,
    js2_sa: np.ndarray | None = None,
    js2_sa_vs_gibbs: np.ndarray | None = None,
    md_source_ids: list[str] | None = None,
    md_source_labels: list[str] | None = None,
    md_source_types: list[str] | None = None,
    md_source_counts: np.ndarray | None = None,
    p_md_by_source: np.ndarray | None = None,
    p2_md_by_source: np.ndarray | None = None,
    sample_source_ids: list[str] | None = None,
    sample_source_labels: list[str] | None = None,
    sample_source_types: list[str] | None = None,
    sample_source_counts: np.ndarray | None = None,
    p_sample_by_source: np.ndarray | None = None,
    p2_sample_by_source: np.ndarray | None = None,
    js_md_sample: np.ndarray | None = None,
    js2_md_sample: np.ndarray | None = None,
    js_gibbs_sample: np.ndarray | None = None,
    js2_gibbs_sample: np.ndarray | None = None,
    energy_bins: np.ndarray | None = None,
    energy_hist_md: np.ndarray | None = None,
    energy_cdf_md: np.ndarray | None = None,
    energy_hist_sample: np.ndarray | None = None,
    energy_cdf_sample: np.ndarray | None = None,
    nn_bins: np.ndarray | None = None,
    nn_cdf_sample_to_md: np.ndarray | None = None,
    nn_cdf_md_to_sample: np.ndarray | None = None,
    edge_strength: np.ndarray | None = None,
) -> Path:
    """
    Save a compact summary bundle of inputs + sampling results into results_dir/run_summary.npz,
    plus metadata JSON for quick inspection.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "run_summary.npz"
    meta_path = results_dir / "run_metadata.json"

    np.savez_compressed(
        summary_path,
        K=np.asarray(K, dtype=int),
        edges=np.asarray(edges, dtype=int),
        residue_labels=np.asarray(residue_labels, dtype=str),
        betas=np.asarray(betas, dtype=float),
        target_beta=np.array([float(args.beta)], dtype=float),
        gibbs_method=np.array([args.gibbs_method], dtype=str),
        X_gibbs=X_gibbs,
        X_sa=X_sa,
        p_md=p_md,
        p_gibbs=p_gibbs,
        p_sa=p_sa,
        js_gibbs=js_gibbs,
        js_sa=js_sa,
        swap_accept_rate=swap_accept_rate if swap_accept_rate is not None else np.array([], dtype=float),
        sa_valid_counts=np.asarray(sa_valid_counts, dtype=int),
        sa_invalid_mask=np.asarray(sa_invalid_mask, dtype=bool),
        sa_schedule_labels=np.asarray(sa_schedule_labels, dtype=str) if sa_schedule_labels is not None else np.array([], dtype=str),
        p_sa_by_schedule=p_sa_by_schedule if p_sa_by_schedule is not None else np.array([], dtype=float),
        js_sa_by_schedule=js_sa_by_schedule if js_sa_by_schedule is not None else np.array([], dtype=float),
        sa_valid_counts_by_schedule=sa_valid_counts_by_schedule if sa_valid_counts_by_schedule is not None else np.array([], dtype=int),
        sa_invalid_mask_by_schedule=sa_invalid_mask_by_schedule if sa_invalid_mask_by_schedule is not None else np.array([], dtype=bool),
        p_gibbs_by_beta=p_gibbs_by_beta if p_gibbs_by_beta is not None else np.array([], dtype=float),
        js_gibbs_by_beta=js_gibbs_by_beta if js_gibbs_by_beta is not None else np.array([], dtype=float),
        js_pair_gibbs_by_beta=js_pair_gibbs_by_beta if js_pair_gibbs_by_beta is not None else np.array([], dtype=float),
        beta_eff_grid=np.asarray(beta_eff_grid, dtype=float) if beta_eff_grid is not None else np.array([], dtype=float),
        beta_eff_distances=np.asarray(beta_eff_distances, dtype=float) if beta_eff_distances is not None else np.array([], dtype=float),
        beta_eff=np.array([beta_eff_value], dtype=float) if beta_eff_value is not None else np.array([], dtype=float),
        beta_eff_by_schedule=np.asarray(beta_eff_by_schedule, dtype=float) if beta_eff_by_schedule is not None else np.array([], dtype=float),
        beta_eff_distances_by_schedule=(
            np.asarray(beta_eff_distances_by_schedule, dtype=float) if beta_eff_distances_by_schedule is not None else np.array([], dtype=float)
        ),
        data_npz=np.array([args.npz], dtype=str),
        p2_md=p2_md if p2_md is not None else np.zeros((0, 0, 0), dtype=float),
        p2_gibbs=p2_gibbs if p2_gibbs is not None else np.zeros((0, 0, 0), dtype=float),
        p2_sa=p2_sa if p2_sa is not None else np.zeros((0, 0, 0), dtype=float),
        js2_gibbs=js2_gibbs if js2_gibbs is not None else np.array([], dtype=float),
        js2_sa=js2_sa if js2_sa is not None else np.array([], dtype=float),
        js2_sa_vs_gibbs=js2_sa_vs_gibbs if js2_sa_vs_gibbs is not None else np.array([], dtype=float),
        md_source_ids=np.asarray(md_source_ids, dtype=str) if md_source_ids is not None else np.array([], dtype=str),
        md_source_labels=np.asarray(md_source_labels, dtype=str) if md_source_labels is not None else np.array([], dtype=str),
        md_source_types=np.asarray(md_source_types, dtype=str) if md_source_types is not None else np.array([], dtype=str),
        md_source_counts=np.asarray(md_source_counts, dtype=int) if md_source_counts is not None else np.array([], dtype=int),
        p_md_by_source=p_md_by_source if p_md_by_source is not None else np.zeros((0, 0, 0), dtype=float),
        p2_md_by_source=p2_md_by_source if p2_md_by_source is not None else np.zeros((0, 0, 0, 0), dtype=float),
        sample_source_ids=np.asarray(sample_source_ids, dtype=str) if sample_source_ids is not None else np.array([], dtype=str),
        sample_source_labels=np.asarray(sample_source_labels, dtype=str) if sample_source_labels is not None else np.array([], dtype=str),
        sample_source_types=np.asarray(sample_source_types, dtype=str) if sample_source_types is not None else np.array([], dtype=str),
        sample_source_counts=np.asarray(sample_source_counts, dtype=int) if sample_source_counts is not None else np.array([], dtype=int),
        p_sample_by_source=p_sample_by_source if p_sample_by_source is not None else np.zeros((0, 0, 0), dtype=float),
        p2_sample_by_source=p2_sample_by_source if p2_sample_by_source is not None else np.zeros((0, 0, 0, 0), dtype=float),
        js_md_sample=js_md_sample if js_md_sample is not None else np.zeros((0, 0, 0), dtype=float),
        js2_md_sample=js2_md_sample if js2_md_sample is not None else np.zeros((0, 0, 0), dtype=float),
        js_gibbs_sample=js_gibbs_sample if js_gibbs_sample is not None else np.zeros((0, 0), dtype=float),
        js2_gibbs_sample=js2_gibbs_sample if js2_gibbs_sample is not None else np.zeros((0, 0), dtype=float),
        energy_bins=energy_bins if energy_bins is not None else np.array([], dtype=float),
        energy_hist_md=energy_hist_md if energy_hist_md is not None else np.zeros((0, 0), dtype=float),
        energy_cdf_md=energy_cdf_md if energy_cdf_md is not None else np.zeros((0, 0), dtype=float),
        energy_hist_sample=energy_hist_sample if energy_hist_sample is not None else np.zeros((0, 0), dtype=float),
        energy_cdf_sample=energy_cdf_sample if energy_cdf_sample is not None else np.zeros((0, 0), dtype=float),
        nn_bins=nn_bins if nn_bins is not None else np.array([], dtype=int),
        nn_cdf_sample_to_md=nn_cdf_sample_to_md if nn_cdf_sample_to_md is not None else np.zeros((0, 0, 0), dtype=float),
        nn_cdf_md_to_sample=nn_cdf_md_to_sample if nn_cdf_md_to_sample is not None else np.zeros((0, 0, 0), dtype=float),
        edge_strength=edge_strength if edge_strength is not None else np.array([], dtype=float),
    )

    metadata = {
        "args": vars(args),
        "summary_file": summary_path.name,
        "data_npz": args.npz,
    }
    if model_path is not None:
        metadata["potts_model_file"] = model_path.name
    if swap_accept_rate is not None and len(swap_accept_rate):
        metadata["swap_accept_rate_mean"] = float(np.mean(swap_accept_rate))
    if beta_eff_value is not None:
        metadata["beta_eff"] = float(beta_eff_value)
    if beta_eff_by_schedule is not None:
        metadata["beta_eff_by_schedule"] = [float(v) for v in beta_eff_by_schedule]
    if sa_schedule_labels is not None:
        metadata["sa_schedule_labels"] = [str(v) for v in sa_schedule_labels]
    meta_path.write_text(json.dumps(metadata, indent=2))
    return summary_path


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz", default="", help="Input dataset (.npz). Required unless --plot-only is set.")
    ap.add_argument("--results-dir", required=True, help="Directory where summaries/plots are stored.")
    ap.add_argument("--plot-only", action="store_true", help="Skip sampling; load run_summary.npz from results-dir (or --summary-file) and emit plots.")
    ap.add_argument("--summary-file", default="", help="Optional explicit path to a summary npz (defaults to results-dir/run_summary.npz).")
    ap.add_argument("--unassigned-policy", default="drop_frames", choices=["drop_frames", "treat_as_state", "error"])
    ap.add_argument("--fit-only", action="store_true", help="Fit and save the Potts model, then exit.")
    ap.add_argument("--model-npz", default="", help="Optional pre-fit Potts model npz to skip fitting.")
    ap.add_argument("--model-out", default="", help="Where to save potts_model.npz (defaults to results-dir/potts_model.npz).")

    ap.add_argument("--fit", default="pmi+plm", choices=["pmi", "plm", "pmi+plm"])
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--plm-epochs", type=int, default=200)
    ap.add_argument("--plm-lr", type=float, default=1e-2)
    ap.add_argument("--plm-lr-min", type=float, default=1e-3)
    ap.add_argument("--plm-lr-schedule", type=str, default="cosine", choices=["cosine", "none"])
    ap.add_argument("--plm-l2", type=float, default=1e-5)
    ap.add_argument("--plm-batch-size", type=int, default=512)
    ap.add_argument("--plm-progress-every", type=int, default=10)
    ap.add_argument(
        "--plm-device",
        type=str,
        default="auto",
        help="Device for PLM training (auto/cpu/cuda or torch device string).",
    )

    # Gibbs / REX-Gibbs
    ap.add_argument("--gibbs-method", default="single", choices=["single", "rex"])
    ap.add_argument("--gibbs-samples", type=int, default=500, help="How many Potts samples to collect from Gibbs (the returned sample count).")
    ap.add_argument("--gibbs-burnin", type=int, default=50, help="How many initial Gibbs sweeps to discard (let the chain forget initialization).")
    ap.add_argument("--gibbs-thin", type=int, default=2, help="Keep one sample every thin sweeps after burn-in (helps reduce correlation).")

    # Replica exchange controls (only used if --gibbs-method rex OR for beta_eff scan)
    ap.add_argument("--rex-betas", type=str, default="", help="Comma-separated betas (ascending), e.g. 0.2,0.3,0.5,0.8,1.0")
    ap.add_argument("--rex-n-replicas", type=int, default=8, help="Number of betas (replicas) when auto-constructing the ladder.")
    ap.add_argument("--rex-beta-min", type=float, default=0.2, help="Minimum beta in the ladder (hottest replica).")
    ap.add_argument("--rex-beta-max", type=float, default=1.0, help="Maximum beta in the ladder (coldest replica).")
    ap.add_argument("--rex-spacing", type=str, default="geom", choices=["geom", "lin"], help="How betas are spaced: geom (geometric): usually better for tempering; lin (linear): sometimes fine for narrow ranges.")
    
    ap.add_argument("--rex-rounds", type=int, default=2000, help="Number of replica-exchange rounds. Each round does: 1) local Gibbs sweeps in each replica; 2) swap attempts between adjacent replicas.")
    ap.add_argument("--rex-burnin-rounds", type=int, default=50, help="Number of initial rounds discarded before saving samples.")
    ap.add_argument("--rex-sweeps-per-round", type=int, default=2, help="How many Gibbs sweeps each replica does per round before swap attempts.")
    ap.add_argument("--rex-thin-rounds", type=int, default=1, help="Save samples every this many rounds after burn-in.")
    ap.add_argument("--rex-max-workers", type=int, default=0, help="Max worker threads for replica exchange (0 = auto, 1 = disable parallelism).")

    # SA/QUBO
    ap.add_argument("--sa-reads", type=int, default=2000, help="Number of independent SA runs (“reads”). Each read outputs one bitstring sample.")
    ap.add_argument("--sa-sweeps", type=int, default=2000, help="Number of sweeps per read. More sweeps = more annealing time.")
    ap.add_argument("--sa-beta-hot", type=float, default=0.0, help="SA hot beta (beta_hot). 0 uses neal default.")
    ap.add_argument("--sa-beta-cold", type=float, default=0.0, help="SA cold beta (beta_cold). 0 uses neal default.")
    ap.add_argument(
        "--sa-beta-schedule",
        action="append",
        default=[],
        help="Additional SA beta schedules as 'hot,cold' or 'hot:cold'. Repeat to add multiple schedules.",
    )
    ap.add_argument("--penalty-safety", type=float, default=3.0, help="Controls how strong the one-hot constraint penalties are in the QUBO. Higher = fewer invalid assignments, but can make the QUBO landscape harder.")
    ap.add_argument("--repair", type=str, default="none", choices=["none", "argmax"], help="What to do when a QUBO bitstring violates one-hot constraints: none: decode invalid slices as “invalid” (still assigns label 0, but validity is tracked; best for honesty). argmax: forcibly repair each residue by picking the largest bit (hides violations but produces a valid label vector).")

    # beta_eff estimation
    ap.add_argument("--estimate-beta-eff", action="store_true", help="If set, the script estimates 𝛽_eff such that Gibbs samples at 𝛽_eff are closest to SA samples.")
    ap.add_argument("--beta-eff-grid", type=str, default="", help="Comma-separated betas to scan. Default: use rex-betas/ladder.")
    ap.add_argument("--beta-eff-w-marg", type=float, default=1.0, help="Weight of marginal-distribution mismatch (per-residue JS divergence) in the distance function.")
    ap.add_argument("--beta-eff-w-pair", type=float, default=1.0, help="Weight of pairwise-on-edges mismatch in the distance function.")

    ap.add_argument("--annotate-plots", action="store_true", help="If set, adds extra annotations to plots (depends on your plotting helper).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", action="store_true")

    return ap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_arg_parser().parse_args(argv)


def _normalize_callbacks(progress_callback):
    if progress_callback is None:
        return []
    if isinstance(progress_callback, (list, tuple)):
        return [cb for cb in progress_callback if cb]
    return [progress_callback]


def run_pipeline(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser | None = None,
    progress_callback=None,
) -> dict[str, object]:
    callbacks = _normalize_callbacks(progress_callback)
    def report(message: str, progress: float) -> None:
        if not callbacks:
            return
        pct = int(max(0, min(100, round(progress))))
        for cb in callbacks:
            try:
                cb(message, pct)
            except Exception:
                pass

    results_dir = _ensure_results_dir(args.results_dir)
    report("Initializing", 0)
    default_summary_path = Path(args.summary_file) if args.summary_file else results_dir / "run_summary.npz"
    default_plot_path = results_dir / "marginals.html"
    beta_scan_plot_path = None
    model_out_path = Path(args.model_out) if args.model_out else results_dir / "potts_model.npz"

    if args.plot_only:
        if not default_summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {default_summary_path}")
        out_path = plot_marginal_summary_from_npz(
            summary_path=default_summary_path,
            out_path=default_plot_path,
            annotate=args.annotate_plots,
        )
        report_path = plot_sampling_report_from_npz(
            summary_path=default_summary_path,
            out_path=results_dir / "sampling_report.html",
        )
        print(f"[plot] loaded summary from {default_summary_path} -> {out_path}")
        return {
            "summary_path": default_summary_path,
            "metadata_path": results_dir / "run_metadata.json",
            "plot_path": out_path,
            "report_path": report_path,
            "beta_scan_path": None,
            "beta_eff": None,
            "model_path": None,
        }

    if not args.npz:
        msg = "--npz is required unless --plot-only is set."
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)

    if args.fit_only and args.model_npz:
        raise ValueError("Use --fit-only without --model-npz.")

    ds = load_npz(args.npz, unassigned_policy=args.unassigned_policy)
    report("Loaded dataset", 8)
    labels = ds.labels
    K = ds.cluster_counts
    edges = ds.edges
    residue_labels = getattr(ds, "residue_keys", np.arange(labels.shape[1]))

    train_points = int(labels.shape[0] * labels.shape[1])
    print(f"[data] T={labels.shape[0]}  N={labels.shape[1]}  edges={len(edges)}  train_points={train_points}")
    plm_device = (args.plm_device or "").strip()
    if plm_device.lower() == "auto":
        plm_device = ""
    plm_device = plm_device or None

    # Fit model(s)
    model = None
    model_pmi = None
    pmi_loss = None
    if args.model_npz:
        report("Loading Potts model", 15)
        model = load_potts_model(args.model_npz)
    else:
        if args.fit in ("pmi", "pmi+plm", "plm"):
            report("Fitting Potts model (PMI)", 15)
            model_pmi = fit_potts_pmi(labels, K, edges)
            model = model_pmi if args.fit == "pmi" else None
            try:
                pmi_loss = compute_pseudolikelihood_loss_torch(
                    model_pmi,
                    labels,
                    batch_size=args.plm_batch_size,
                    device=plm_device,
                )
                print(f"[pmi] avg_pseudolikelihood_loss={pmi_loss:.6f}")
            except RuntimeError:
                print("[pmi] pseudolikelihood loss unavailable (torch missing).")
        if args.fit in ("plm", "pmi+plm"):
            try:
                show_batch_progress = not callbacks and (args.progress or sys.stdout.isatty())
                plm_batch_progress = None
                report_init_loss = pmi_loss is None
                if show_batch_progress:
                    last_epoch = {"val": None}

                    def _plm_batch_progress(ep: int, total_ep: int, batch: int, total_batches: int) -> None:
                        if last_epoch["val"] is None:
                            last_epoch["val"] = ep
                        if ep != last_epoch["val"]:
                            print()
                            last_epoch["val"] = ep
                        bar_len = 28
                        filled = int(bar_len * batch / max(1, total_batches))
                        bar = "#" * filled + "-" * (bar_len - filled)
                        end = "\n" if batch >= total_batches else ""
                        print(
                            f"\r[plm] epoch {ep}/{total_ep} batch {batch}/{total_batches} [{bar}]",
                            end=end,
                            flush=True,
                        )

                    plm_batch_progress = _plm_batch_progress

                def _plm_progress(ep: int, total: int, avg_loss: float) -> None:
                    base = 20.0
                    span = 30.0
                    report(f"Fitting Potts model (PLM) {ep}/{total} loss={avg_loss:.4f}", base + span * ep / max(1, total))
                    if show_batch_progress:
                        print(f"[plm] epoch {ep}/{total} avg_loss={avg_loss:.6f}")

                model_plm = fit_potts_pseudolikelihood_torch(
                    labels,
                    K,
                    edges,
                    l2=args.plm_l2,
                    lr=args.plm_lr,
                    lr_min=args.plm_lr_min,
                    lr_schedule=args.plm_lr_schedule,
                    epochs=args.plm_epochs,
                    batch_size=args.plm_batch_size,
                    seed=args.seed,
                    verbose=not show_batch_progress,
                    init_model=model_pmi,
                    report_init_loss=report_init_loss,
                    device=plm_device,
                    progress_callback=_plm_progress,
                    progress_every=args.plm_progress_every,
                    batch_progress_callback=plm_batch_progress,
                )
                model = model_plm
            except RuntimeError as exc:
                if "PyTorch is required" not in str(exc):
                    raise
                if model_pmi is None:
                    model_pmi = fit_potts_pmi(labels, K, edges)
                model = model_pmi
                args.fit = "pmi"
                print("[fit] warning: PyTorch missing; falling back to PMI fit.")
                report("PyTorch missing; falling back to PMI fit", 35)

    assert model is not None
    save_potts_model(
        model,
        model_out_path,
        metadata={
            "data_npz": args.npz,
            "fit_method": args.fit,
            "source_model": args.model_npz or None,
        },
    )
    report("Potts model fit complete", 40)

    if args.fit_only:
        meta_path = results_dir / "potts_model_metadata.json"
        meta_path.write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "model_file": model_out_path.name,
                    "data_npz": args.npz,
                },
                indent=2,
            )
        )
        return {
            "model_path": model_out_path,
            "metadata_path": meta_path,
        }

    betas: list[float] = []
    beta_eff_grid_result: list[float] | None = None
    beta_eff_distances_result: list[float] | None = None
    beta_eff_value: float | None = None
    swap_accept_rate: np.ndarray | None = None
    samples_by_beta: dict[float, np.ndarray] | None = None

    # --- Sampling baseline: Gibbs or REX-Gibbs at beta=args.beta ---
    if args.gibbs_method == "single":
        report("Sampling Gibbs (single chain)", 50)
        total_steps = args.gibbs_burnin + args.gibbs_samples * args.gibbs_thin
        def _gibbs_progress(step: int, total: int) -> None:
            report(f"Gibbs sweeps {step}/{total}", 50 + 10 * step / max(1, total))
        X_gibbs = gibbs_sample_potts(
            model,
            beta=args.beta,
            n_samples=args.gibbs_samples,
            burn_in=args.gibbs_burnin,
            thinning=args.gibbs_thin,
            seed=args.seed,
            progress=args.progress,
            progress_callback=_gibbs_progress if callbacks else None,
            progress_every=max(1, total_steps // 20) if total_steps else 1,
        )
        rex_info = None
        betas = [float(args.beta)]
    else:
        report("Sampling Gibbs (replica exchange)", 50)
        rex_max_workers = None if args.rex_max_workers <= 0 else args.rex_max_workers
        def _rex_progress(rnd: int, total: int) -> None:
            report(f"Replica exchange {rnd}/{total}", 50 + 10 * rnd / max(1, total))
        if args.rex_betas.strip():
            betas = _parse_float_list(args.rex_betas)
        else:
            betas = make_beta_ladder(
                beta_min=args.rex_beta_min,
                beta_max=args.rex_beta_max,
                n_replicas=args.rex_n_replicas,
                spacing=args.rex_spacing,
            )

        # ensure target beta is in ladder (append + sort if needed)
        if all(abs(b - args.beta) > 1e-12 for b in betas):
            betas = sorted(set(betas + [float(args.beta)]))

        rex_info = replica_exchange_gibbs_potts(
            model,
            betas=betas,
            sweeps_per_round=args.rex_sweeps_per_round,
            n_rounds=args.rex_rounds,
            burn_in_rounds=args.rex_burnin_rounds,
            thinning_rounds=args.rex_thin_rounds,
            seed=args.seed,
            progress=args.progress,
            progress_callback=_rex_progress if callbacks else None,
            progress_every=max(1, args.rex_rounds // 20) if args.rex_rounds else 1,
            max_workers=rex_max_workers,
        )
        samples_by_beta = rex_info["samples_by_beta"]  # type: ignore
        X_gibbs = samples_by_beta[float(args.beta)]

        acc = rex_info["swap_accept_rate"]  # type: ignore
        swap_accept_rate = acc
        print(f"[rex] betas={betas}")
        print(f"[rex] swap_accept_rate (adjacent): mean={float(np.mean(acc)):.3f}, min={float(np.min(acc)):.3f}, max={float(np.max(acc)):.3f}")

    # --- SA/QUBO sampling ---
    report("Sampling SA/QUBO", 70)
    qubo = potts_to_qubo_onehot(model, beta=args.beta, penalty_safety=args.penalty_safety)

    sa_schedule_specs: list[dict[str, object]] = [{"label": "SA auto", "beta_range": None}]
    seen_schedules: set[tuple[float, float]] = set()

    def _add_sa_schedule(hot: float, cold: float) -> None:
        if hot <= 0 or cold <= 0:
            raise ValueError("SA betas must be > 0.")
        if hot > cold:
            raise ValueError("SA beta hot must be <= beta cold.")
        key = (float(hot), float(cold))
        if key in seen_schedules:
            return
        seen_schedules.add(key)
        sa_schedule_specs.append({
            "label": f"SA β={key[0]:g}→{key[1]:g}",
            "beta_range": key,
        })

    if (args.sa_beta_hot and not args.sa_beta_cold) or (args.sa_beta_cold and not args.sa_beta_hot):
        raise ValueError("Provide both sa_beta_hot and sa_beta_cold, or neither.")
    if args.sa_beta_hot and args.sa_beta_cold:
        _add_sa_schedule(float(args.sa_beta_hot), float(args.sa_beta_cold))

    for raw in args.sa_beta_schedule or []:
        hot, cold = _parse_beta_schedule(raw)
        _add_sa_schedule(hot, cold)

    sa_samples: list[np.ndarray] = []
    sa_valid_counts_list: list[np.ndarray] = []
    sa_invalid_mask_list: list[np.ndarray] = []

    total_sa = len(sa_schedule_specs)
    repair = None if args.repair == "none" else args.repair
    for idx, spec in enumerate(sa_schedule_specs):
        label = str(spec["label"])
        beta_range = spec["beta_range"]
        report(f"Sampling SA/QUBO ({label})", 70 + 8 * idx / max(1, total_sa))

        Z_sa = sa_sample_qubo_neal(
            qubo,
            n_reads=args.sa_reads,
            sweeps=args.sa_sweeps,
            seed=args.seed + idx,
            progress=args.progress,
            beta_range=beta_range if isinstance(beta_range, tuple) else None,
        )

        X_sa_local = np.zeros((Z_sa.shape[0], len(qubo.var_slices)), dtype=int)
        valid_counts = np.zeros(Z_sa.shape[0], dtype=int)
        for i in range(Z_sa.shape[0]):
            x, valid = decode_onehot(Z_sa[i], qubo, repair=repair)
            X_sa_local[i] = x
            valid_counts[i] = int(valid.sum())

        viol = np.array([np.any(qubo.constraint_violations(z) != 0) for z in Z_sa], dtype=bool)
        print(
            f"[qubo] {label}: invalid_samples={viol.mean()*100:.2f}%  "
            f"avg_valid_residues={valid_counts.mean():.1f}/{len(qubo.var_slices)}  repair={args.repair}"
        )

        sa_samples.append(X_sa_local)
        sa_valid_counts_list.append(valid_counts)
        sa_invalid_mask_list.append(viol)

    # --- Compare to MD ---
    report("Computing summary metrics", 80)
    p_md = marginals(labels, K)
    p_g = marginals(X_gibbs, K)
    p_sa_list = [marginals(X_sa_local, K) for X_sa_local in sa_samples]

    js_g = per_residue_js(p_md, p_g)
    js_sa_list = [per_residue_js(p_md, p_sa_local) for p_sa_local in p_sa_list]

    sa_schedule_labels = [str(spec["label"]) for spec in sa_schedule_specs]
    p_sa_by_schedule = np.stack([_pad_marginals_for_save(p) for p in p_sa_list], axis=0)
    js_sa_by_schedule = np.stack([np.asarray(js_vec, dtype=float) for js_vec in js_sa_list], axis=0)
    sa_valid_counts_by_schedule = np.stack(sa_valid_counts_list, axis=0)
    sa_invalid_mask_by_schedule = np.stack(sa_invalid_mask_list, axis=0)

    X_sa = sa_samples[0]
    valid_counts = sa_valid_counts_list[0]
    viol = sa_invalid_mask_list[0]

    p_sa = p_sa_list[0]
    js_sa = js_sa_list[0]
    print(
        f"[marginals] JS(MD, {sa_schedule_specs[0]['label']}): "
        f"mean={js_sa.mean():.4f}  median={np.median(js_sa):.4f}  max={js_sa.max():.4f}"
    )
    for idx, js_vec in enumerate(js_sa_list[1:], start=1):
        label = sa_schedule_specs[idx]["label"]
        print(
            f"[marginals] JS(MD, {label}): "
            f"mean={js_vec.mean():.4f}  median={np.median(js_vec):.4f}  max={js_vec.max():.4f}"
        )

    p_gibbs_by_beta: np.ndarray | None = None
    js_gibbs_by_beta: np.ndarray | None = None
    js_pair_gibbs_by_beta: np.ndarray | None = None

    if args.gibbs_method == "single":
        print(
            f"[marginals] JS(MD, Gibbs@beta={args.beta}): "
            f"mean={js_g.mean():.4f}  median={np.median(js_g):.4f}  max={js_g.max():.4f}"
        )
    elif samples_by_beta is not None:
        p_gibbs_by_beta_list = []
        js_gibbs_by_beta_list = []
        js_pair_gibbs_by_beta_list: list[float] = []

        for b in betas:
            X_b = samples_by_beta[float(b)]
            p_b = marginals(X_b, K)
            js_b = per_residue_js(p_md, p_b)
            p_gibbs_by_beta_list.append(_pad_marginals_for_save(p_b))
            js_gibbs_by_beta_list.append(js_b)
            print(
                f"[marginals] JS(MD, Gibbs@beta={b}): "
                f"mean={js_b.mean():.4f}  median={np.median(js_b):.4f}  max={js_b.max():.4f}"
            )

            if len(edges) > 0:
                js_pair_gibbs_by_beta_list.append(
                    combined_distance(labels, X_b, K=K, edges=edges, w_marg=0.0, w_pair=1.0)
                )

        if p_gibbs_by_beta_list:
            p_gibbs_by_beta = np.stack(p_gibbs_by_beta_list, axis=0)
            js_gibbs_by_beta = np.stack(js_gibbs_by_beta_list, axis=0)
        if js_pair_gibbs_by_beta_list:
            js_pair_gibbs_by_beta = np.asarray(js_pair_gibbs_by_beta_list, dtype=float)

    # Optional: pairwise summary
    if len(edges) > 0:
        P_md = pairwise_joints_on_edges(labels, K, edges)
        P_g = pairwise_joints_on_edges(X_gibbs, K, edges)
        P_sa = pairwise_joints_on_edges(X_sa, K, edges)

        # mean over edges via combined_distance pair term
        # (reuse combined_distance with only pair term by passing w_marg=0)
        js_pair_g = combined_distance(labels, X_gibbs, K=K, edges=edges, w_marg=0.0, w_pair=1.0)
        js_pair_sa = combined_distance(labels, X_sa, K=K, edges=edges, w_marg=0.0, w_pair=1.0)
        if js_pair_gibbs_by_beta is None:
            print(f"[pairs]   JS(MD, Gibbs) over edges: {js_pair_g:.4f}")
        print(f"[pairs]   JS(MD, SA-QUBO) over edges: {js_pair_sa:.4f}")
        if js_pair_gibbs_by_beta is not None:
            for b, js_pair in zip(betas, js_pair_gibbs_by_beta):
                print(f"[pairs]   JS(MD, Gibbs@beta={b}) over edges: {js_pair:.4f}")

    # Build per-source stats for dynamic comparisons
    md_sources = _build_md_sources(
        labels,
        K,
        edges,
        getattr(ds, "frame_state_ids", None),
        getattr(ds, "frame_metastable_ids", None),
        getattr(ds, "metadata", None),
        args.npz,
    )
    gibbs_label = f"Gibbs β={float(args.beta):g}"
    sample_sources = _build_sample_sources(
        X_gibbs,
        sa_samples,
        sa_schedule_labels,
        K,
        edges,
        gibbs_label=gibbs_label,
    )

    md_source_ids = [src["id"] for src in md_sources]
    md_source_labels = [src["label"] for src in md_sources]
    md_source_types = [src["type"] for src in md_sources]
    md_source_counts = np.asarray([src["count"] for src in md_sources], dtype=int)
    p_md_by_source = np.stack([src["p"] for src in md_sources], axis=0) if md_sources else np.zeros((0, 0, 0), dtype=float)
    p2_md_by_source = np.stack([src["p2"] for src in md_sources], axis=0) if md_sources else np.zeros((0, 0, 0, 0), dtype=float)

    sample_source_ids = [src["id"] for src in sample_sources]
    sample_source_labels = [src["label"] for src in sample_sources]
    sample_source_types = [src["type"] for src in sample_sources]
    sample_source_counts = np.asarray([src["count"] for src in sample_sources], dtype=int)
    p_sample_by_source = np.stack([src["p"] for src in sample_sources], axis=0) if sample_sources else np.zeros((0, 0, 0), dtype=float)
    p2_sample_by_source = np.stack([src["p2"] for src in sample_sources], axis=0) if sample_sources else np.zeros((0, 0, 0, 0), dtype=float)

    js_md_sample = np.zeros((len(md_sources), len(sample_sources), len(K)), dtype=float)
    js2_md_sample = np.zeros((len(md_sources), len(sample_sources), len(edges)), dtype=float)
    for i, md in enumerate(md_sources):
        for j, sample in enumerate(sample_sources):
            js_md_sample[i, j] = per_residue_js_from_padded(md["p"], sample["p"], K)
            if len(edges) > 0:
                js2_md_sample[i, j] = per_edge_js_from_padded(md["p2"], sample["p2"], edges, K)

    gibbs_idx = next((i for i, src in enumerate(sample_sources) if src["id"] == "gibbs"), None)
    js_gibbs_sample = np.zeros((len(sample_sources), len(K)), dtype=float)
    js2_gibbs_sample = np.zeros((len(sample_sources), len(edges)), dtype=float)
    if gibbs_idx is not None:
        gibbs_p = sample_sources[gibbs_idx]["p"]
        gibbs_p2 = sample_sources[gibbs_idx]["p2"]
        for j, sample in enumerate(sample_sources):
            js_gibbs_sample[j] = per_residue_js_from_padded(gibbs_p, sample["p"], K)
            if len(edges) > 0:
                js2_gibbs_sample[j] = per_edge_js_from_padded(gibbs_p2, sample["p2"], edges, K)

    p2_md = md_sources[0]["p2"] if md_sources else np.zeros((0, 0, 0), dtype=float)
    p2_gibbs = sample_sources[gibbs_idx]["p2"] if gibbs_idx is not None else np.zeros((0, 0, 0), dtype=float)
    sa_idx = next((i for i, src in enumerate(sample_sources) if src["type"] == "sa"), None)
    p2_sa = sample_sources[sa_idx]["p2"] if sa_idx is not None else np.zeros((0, 0, 0), dtype=float)
    js2_gibbs = js2_md_sample[0, gibbs_idx] if md_sources and gibbs_idx is not None else np.array([], dtype=float)
    js2_sa = js2_md_sample[0, sa_idx] if md_sources and sa_idx is not None else np.array([], dtype=float)
    js2_sa_vs_gibbs = js2_gibbs_sample[sa_idx] if sa_idx is not None else np.array([], dtype=float)

    edge_strength = np.array([], dtype=float)
    if model is not None and len(edges) > 0:
        strengths = []
        for r, s in edges:
            strengths.append(float(np.linalg.norm(model.coupling(int(r), int(s)))))
        edge_strength = np.asarray(strengths, dtype=float)

    energy_payload = _compute_energy_histograms(
        model=model,
        md_sources=md_sources,
        sample_sources=sample_sources,
        n_bins=40,
    )
    nn_payload = _compute_nn_cdfs(
        md_sources=md_sources,
        sample_sources=sample_sources,
        max_md=2000,
        max_sample=1000,
        block_size=256,
    )

    beta_eff_grid_result = None
    beta_eff_distances_result = None
    beta_eff_value = None
    beta_eff_by_schedule = None
    distances_by_schedule = None

    # --- Estimate beta_eff for SA (optional) ---
    if args.estimate_beta_eff:
        report("Estimating beta_eff", 88)
        if rex_info is None:
            print("[beta_eff] warning: beta_eff scan requires --gibbs-method rex (reusing REX samples). Skipping beta_eff.")
            summary_path = _save_run_summary(
                results_dir,
                args,
                K=np.asarray(K),
                edges=np.asarray(edges),
                residue_labels=np.asarray(residue_labels),
                betas=betas,
                X_gibbs=X_gibbs,
                X_sa=X_sa,
                p_md=_pad_marginals_for_save(p_md),
                p_gibbs=_pad_marginals_for_save(p_g),
                p_sa=_pad_marginals_for_save(p_sa),
                js_gibbs=js_g,
                js_sa=js_sa,
                swap_accept_rate=swap_accept_rate,
                sa_valid_counts=valid_counts,
                sa_invalid_mask=viol,
                sa_schedule_labels=sa_schedule_labels,
                p_sa_by_schedule=p_sa_by_schedule,
                js_sa_by_schedule=js_sa_by_schedule,
                sa_valid_counts_by_schedule=sa_valid_counts_by_schedule,
                sa_invalid_mask_by_schedule=sa_invalid_mask_by_schedule,
                p_gibbs_by_beta=p_gibbs_by_beta,
                js_gibbs_by_beta=js_gibbs_by_beta,
                js_pair_gibbs_by_beta=js_pair_gibbs_by_beta,
                beta_eff_grid=beta_eff_grid_result,
                beta_eff_distances=beta_eff_distances_result,
                beta_eff_value=beta_eff_value,
                beta_eff_by_schedule=beta_eff_by_schedule,
                beta_eff_distances_by_schedule=None,
                model_path=model_out_path,
                p2_md=p2_md,
                p2_gibbs=p2_gibbs,
                p2_sa=p2_sa,
                js2_gibbs=js2_gibbs,
                js2_sa=js2_sa,
                js2_sa_vs_gibbs=js2_sa_vs_gibbs,
                md_source_ids=md_source_ids,
                md_source_labels=md_source_labels,
                md_source_types=md_source_types,
                md_source_counts=md_source_counts,
                p_md_by_source=p_md_by_source,
                p2_md_by_source=p2_md_by_source,
                sample_source_ids=sample_source_ids,
                sample_source_labels=sample_source_labels,
                sample_source_types=sample_source_types,
                sample_source_counts=sample_source_counts,
                p_sample_by_source=p_sample_by_source,
                p2_sample_by_source=p2_sample_by_source,
                js_md_sample=js_md_sample,
                js2_md_sample=js2_md_sample,
                js_gibbs_sample=js_gibbs_sample,
                js2_gibbs_sample=js2_gibbs_sample,
                energy_bins=energy_payload["bins"],
                energy_hist_md=energy_payload["hist_md"],
                energy_cdf_md=energy_payload["cdf_md"],
                energy_hist_sample=energy_payload["hist_sample"],
                energy_cdf_sample=energy_payload["cdf_sample"],
                nn_bins=nn_payload["bins"],
                nn_cdf_sample_to_md=nn_payload["cdf_sample_to_md"],
                nn_cdf_md_to_sample=nn_payload["cdf_md_to_sample"],
                edge_strength=edge_strength,
            )
            print(f"[results] summary saved to {summary_path}")
            out_path = plot_marginal_summary_from_npz(
                summary_path=summary_path,
                out_path=default_plot_path,
                annotate=args.annotate_plots,
            )
            report_path = plot_sampling_report_from_npz(
                summary_path=summary_path,
                out_path=results_dir / "sampling_report.html",
            )
            print(f"[plot] saved marginal comparison to {out_path}")
            print("[done]")
            return {
                "summary_path": summary_path,
                "metadata_path": results_dir / "run_metadata.json",
                "plot_path": out_path,
                "report_path": report_path,
                "beta_scan_path": None,
                "beta_eff": beta_eff_value,
                "beta_eff_by_schedule": beta_eff_by_schedule,
                "model_path": model_out_path,
            }

        beta_eff_plot_path = results_dir / "beta_scan.html"
        if args.beta_eff_grid.strip():
            grid = _parse_float_list(args.beta_eff_grid)
            grid = sorted(set(grid))
        else:
            # default: use the same ladder as rex if available, else construct one around args.beta
            if args.rex_betas.strip():
                grid = sorted(set(_parse_float_list(args.rex_betas)))
            else:
                grid = make_beta_ladder(
                    beta_min=min(args.rex_beta_min, args.beta / 5.0),
                    beta_max=max(args.rex_beta_max, args.beta),
                    n_replicas=max(args.rex_n_replicas, 8),
                    spacing=args.rex_spacing,
                )
                if all(abs(b - args.beta) > 1e-12 for b in grid):
                    grid = sorted(set(grid + [float(args.beta)]))

        print(f"[beta_eff] scanning betas={grid}")

        # Try to reuse the baseline REX run; only rerun if grid has unseen betas.
        rex_betas_available = set(map(float, rex_info["betas"]))  # type: ignore
        missing = sorted(set(grid) - rex_betas_available)

        if not missing:
            print("[beta_eff] reusing baseline REX samples for beta_eff scan.")
            ref = rex_info["samples_by_beta"]  # type: ignore
        else:
            print(f"[beta_eff] baseline REX missing betas {missing}; running additional REX for beta_eff scan.")
            rex_scan = replica_exchange_gibbs_potts(
                model,
                betas=grid,
                sweeps_per_round=args.rex_sweeps_per_round,
                n_rounds=args.rex_rounds,
                burn_in_rounds=args.rex_burnin_rounds,
                thinning_rounds=args.rex_thin_rounds,
                seed=args.seed + 123,
                progress=args.progress,
            )
            ref = rex_scan["samples_by_beta"]  # type: ignore

        distances_by_schedule: list[list[float]] = []
        beta_eff_by_schedule: list[float] = []

        for idx, X_sa_schedule in enumerate(sa_samples):
            distances = []
            for b in grid:
                X_ref = ref[float(b)]
                d = combined_distance(
                    X_sa_schedule,
                    X_ref,
                    K=K,
                    edges=edges,
                    w_marg=args.beta_eff_w_marg,
                    w_pair=args.beta_eff_w_pair,
                )
                distances.append(d)
            distances_by_schedule.append(distances)
            b_eff = grid[int(np.argmin(distances))]
            beta_eff_by_schedule.append(float(b_eff))
            label = sa_schedule_labels[idx] if idx < len(sa_schedule_labels) else f"SA {idx + 1}"
            print(f"[beta_eff] {label}: beta_eff={b_eff:.6g}  (min distance={min(distances):.6g})")
            for b, d in zip(grid, distances):
                mark = "*" if abs(b - b_eff) < 1e-12 else " "
                print(f"[beta_eff] {mark} {label} beta={b:10.6g}  D={d:.6g}")

        beta_eff_grid_result = grid
        beta_eff_distances_result = distances_by_schedule[0] if distances_by_schedule else []
        beta_eff_value = beta_eff_by_schedule[0] if beta_eff_by_schedule else None

        from phase.simulation.plotting import plot_beta_scan_curve
        outp = plot_beta_scan_curve(
            betas=grid,
            distances=distances_by_schedule,
            labels=sa_schedule_labels,
            out_path=beta_eff_plot_path,
        )
        print(f"[beta_eff] saved D(beta) plot to {outp}")
        beta_scan_plot_path = outp

    summary_path = _save_run_summary(
        results_dir,
        args,
        K=np.asarray(K),
        edges=np.asarray(edges),
        residue_labels=np.asarray(residue_labels),
        betas=betas,
        X_gibbs=X_gibbs,
        X_sa=X_sa,
        p_md=_pad_marginals_for_save(p_md),
        p_gibbs=_pad_marginals_for_save(p_g),
        p_sa=_pad_marginals_for_save(p_sa),
        js_gibbs=js_g,
        js_sa=js_sa,
        swap_accept_rate=swap_accept_rate,
        sa_valid_counts=valid_counts,
        sa_invalid_mask=viol,
        sa_schedule_labels=sa_schedule_labels,
        p_sa_by_schedule=p_sa_by_schedule,
        js_sa_by_schedule=js_sa_by_schedule,
        sa_valid_counts_by_schedule=sa_valid_counts_by_schedule,
        sa_invalid_mask_by_schedule=sa_invalid_mask_by_schedule,
        p_gibbs_by_beta=p_gibbs_by_beta,
        js_gibbs_by_beta=js_gibbs_by_beta,
        js_pair_gibbs_by_beta=js_pair_gibbs_by_beta,
        beta_eff_grid=beta_eff_grid_result,
        beta_eff_distances=beta_eff_distances_result,
        beta_eff_value=beta_eff_value,
        beta_eff_by_schedule=beta_eff_by_schedule,
        beta_eff_distances_by_schedule=(
            np.asarray(distances_by_schedule, dtype=float) if distances_by_schedule is not None else None
        ),
        model_path=model_out_path,
        p2_md=p2_md,
        p2_gibbs=p2_gibbs,
        p2_sa=p2_sa,
        js2_gibbs=js2_gibbs,
        js2_sa=js2_sa,
        js2_sa_vs_gibbs=js2_sa_vs_gibbs,
        md_source_ids=md_source_ids,
        md_source_labels=md_source_labels,
        md_source_types=md_source_types,
        md_source_counts=md_source_counts,
        p_md_by_source=p_md_by_source,
        p2_md_by_source=p2_md_by_source,
        sample_source_ids=sample_source_ids,
        sample_source_labels=sample_source_labels,
        sample_source_types=sample_source_types,
        sample_source_counts=sample_source_counts,
        p_sample_by_source=p_sample_by_source,
        p2_sample_by_source=p2_sample_by_source,
        js_md_sample=js_md_sample,
        js2_md_sample=js2_md_sample,
        js_gibbs_sample=js_gibbs_sample,
        js2_gibbs_sample=js2_gibbs_sample,
        energy_bins=energy_payload["bins"],
        energy_hist_md=energy_payload["hist_md"],
        energy_cdf_md=energy_payload["cdf_md"],
        energy_hist_sample=energy_payload["hist_sample"],
        energy_cdf_sample=energy_payload["cdf_sample"],
        nn_bins=nn_payload["bins"],
        nn_cdf_sample_to_md=nn_payload["cdf_sample_to_md"],
        nn_cdf_md_to_sample=nn_payload["cdf_md_to_sample"],
        edge_strength=edge_strength,
    )
    print(f"[results] summary saved to {summary_path}")

    # --- Plot marginals dashboard ---
    report("Rendering plots", 95)
    out_path = plot_marginal_summary_from_npz(
        summary_path=summary_path,
        out_path=default_plot_path,
        annotate=args.annotate_plots,
    )
    report_path = plot_sampling_report_from_npz(
        summary_path=summary_path,
        out_path=results_dir / "sampling_report.html",
    )
    print(f"[plot] saved marginal comparison to {out_path}")

    print("[done]")
    report("Done", 100)
    return {
        "summary_path": summary_path,
        "metadata_path": results_dir / "run_metadata.json",
        "plot_path": out_path,
        "report_path": report_path,
        "beta_scan_path": beta_scan_plot_path,
        "beta_eff": beta_eff_value,
        "beta_eff_by_schedule": beta_eff_by_schedule,
        "model_path": model_out_path,
    }


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    run_pipeline(args, parser=parser)


if __name__ == "__main__":
    main()
