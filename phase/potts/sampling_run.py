from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from phase.io.data import load_npz
from phase.potts.potts_model import PottsModel, add_potts_models, load_potts_model
from phase.potts.qubo import decode_onehot, encode_onehot, potts_to_qubo_onehot
from phase.potts.sample_io import SAMPLE_NPZ_FILENAME, save_sample_npz
from phase.potts.sampling import (
    gibbs_sample_potts,
    make_beta_ladder,
    replica_exchange_gibbs_potts,
    sa_sample_qubo_neal,
)


@dataclass(frozen=True)
class SamplingResult:
    sample_path: Path
    n_samples: int
    n_residues: int


def _normalize_model_paths(model_npz: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in model_npz or []:
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        if "," in s:
            out.extend([p.strip() for p in s.split(",") if p.strip()])
        else:
            out.append(s)
    return out


def _load_combined_model(model_paths: Sequence[str]) -> PottsModel:
    paths = _normalize_model_paths(model_paths)
    if not paths:
        raise ValueError("No --model-npz provided (sampling requires an existing Potts model).")
    model = load_potts_model(paths[0])
    for p in paths[1:]:
        model = add_potts_models(model, load_potts_model(p))
    return model


def _sample_labels_uniform(K_list: Sequence[int], n_samples: int, rng: np.random.Generator) -> np.ndarray:
    n_res = len(K_list)
    out = np.zeros((n_samples, n_res), dtype=int)
    for r, k in enumerate(K_list):
        out[:, r] = rng.integers(0, int(k), size=n_samples)
    return out


def _sample_labels_from_fields(
    model: PottsModel,
    *,
    beta: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_res = len(model.h)
    out = np.zeros((n_samples, n_res), dtype=int)
    for r, hr in enumerate(model.h):
        hr = np.asarray(hr, dtype=float)
        if hr.size == 0 or not np.all(np.isfinite(hr)):
            out[:, r] = rng.integers(0, max(1, hr.size), size=n_samples)
            continue
        logits = -float(beta) * hr
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        total = float(np.sum(probs))
        if total <= 0 or not np.isfinite(total):
            out[:, r] = rng.integers(0, hr.shape[0], size=n_samples)
            continue
        probs = probs / total
        out[:, r] = rng.choice(hr.shape[0], size=n_samples, p=probs)
    return out


def _build_sa_initial_labels(
    *,
    mode: str,
    md_labels: np.ndarray,
    model: PottsModel,
    beta: float,
    n_reads: int,
    md_frame: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mode = (mode or "md").lower()
    if mode in {"md", "md-frame"}:
        if md_labels is None or md_labels.size == 0:
            if mode == "md-frame":
                raise ValueError("SA init set to md-frame, but MD labels are unavailable.")
            return _sample_labels_from_fields(model, beta=beta, n_samples=n_reads, rng=rng)
        if mode == "md-frame":
            if md_frame < 0:
                raise ValueError("--sa-init md-frame requires --sa-init-md-frame >= 0.")
            if md_frame >= md_labels.shape[0]:
                raise ValueError(f"--sa-init-md-frame {md_frame} out of range (0..{md_labels.shape[0]-1}).")
            return np.repeat(md_labels[md_frame : md_frame + 1], n_reads, axis=0)
        idx = rng.integers(0, md_labels.shape[0], size=n_reads)
        return md_labels[idx]
    if mode in {"random-h", "h"}:
        return _sample_labels_from_fields(model, beta=beta, n_samples=n_reads, rng=rng)
    if mode in {"random-uniform", "uniform"}:
        return _sample_labels_uniform(model.K_list(), n_reads, rng)
    raise ValueError(f"Unknown sa-init mode: {mode}")


def _parse_float_list(raw: str) -> List[float]:
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    return [float(p) for p in parts]


def _run_rex_chain_worker(payload: dict[str, object]) -> dict[str, object]:
    return replica_exchange_gibbs_potts(
        payload["model"],  # type: ignore[arg-type]
        betas=payload["betas"],  # type: ignore[arg-type]
        sweeps_per_round=int(payload["sweeps_per_round"]),
        n_rounds=int(payload["n_rounds"]),
        burn_in_rounds=int(payload["burn_in_rounds"]),
        thinning_rounds=int(payload["thinning_rounds"]),
        seed=int(payload["seed"]),
        progress=bool(payload.get("progress", False)),
        progress_callback=None,
        progress_every=max(1, int(payload.get("progress_every", 1))),
        max_workers=payload.get("max_workers"),
        progress_desc=payload.get("progress_desc"),  # type: ignore[arg-type]
        progress_position=payload.get("progress_position"),  # type: ignore[arg-type]
        progress_mode=str(payload.get("progress_mode", "samples")),
    )


def run_sampling(
    *,
    cluster_npz: str,
    results_dir: str | Path,
    model_npz: Sequence[str],
    sampling_method: str,
    beta: float,
    seed: int,
    progress: bool = False,
    # gibbs
    gibbs_method: str = "single",
    gibbs_samples: int = 500,
    gibbs_burnin: int = 50,
    gibbs_thin: int = 2,
    gibbs_chains: int = 1,
    # rex
    rex_betas: str = "",
    rex_n_replicas: int = 8,
    rex_beta_min: float = 0.2,
    rex_beta_max: float = 1.0,
    rex_spacing: str = "geom",
    rex_rounds: int = 2000,
    rex_burnin_rounds: int = 50,
    rex_sweeps_per_round: int = 2,
    rex_thin_rounds: int = 1,
    rex_chains: int = 1,
    # sa
    sa_reads: int = 2000,
    sa_sweeps: int = 2000,
    sa_beta_hot: float = 0.0,
    sa_beta_cold: float = 0.0,
    sa_init: str = "md",
    sa_init_md_frame: int = -1,
    sa_restart: str = "independent",
    sa_restart_topk: int = 200,
    penalty_safety: float = 3.0,
    repair: str = "none",
    progress_callback: Callable[[str, int], None] | None = None,
) -> SamplingResult:
    """
    Run a sampler and write results_dir/sample.npz (minimal sample schema).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    sample_path = results_dir / SAMPLE_NPZ_FILENAME

    def report(msg: str, pct: int) -> None:
        if progress_callback:
            progress_callback(msg, int(pct))

    model = _load_combined_model(model_npz)
    md_ds = load_npz(cluster_npz, unassigned_policy="drop_frames", allow_missing_edges=True)
    md_labels = md_ds.labels

    method = (sampling_method or "gibbs").strip().lower()
    if method not in {"gibbs", "sa"}:
        raise ValueError("--sampling-method must be gibbs or sa.")

    if method == "gibbs":
        gm = (gibbs_method or "single").strip().lower()
        if gm not in {"single", "rex"}:
            raise ValueError("--gibbs-method must be single or rex.")

        if gm == "single":
            # Optionally split across independent chains and concatenate.
            n_chains = max(1, int(gibbs_chains))
            total_samples = max(0, int(gibbs_samples))
            if n_chains > max(1, total_samples):
                n_chains = max(1, total_samples)
            base = total_samples // n_chains if n_chains else total_samples
            extra = total_samples % n_chains if n_chains else 0
            chain_samples = [base + (1 if i < extra else 0) for i in range(n_chains)]
            parts: List[np.ndarray] = []
            for idx, n_samp in enumerate(chain_samples):
                if n_samp <= 0:
                    continue
                report(f"Gibbs sampling chain {idx + 1}/{n_chains}", 10 + int(80 * idx / max(1, n_chains)))
                part = gibbs_sample_potts(
                    model,
                    beta=float(beta),
                    n_samples=int(n_samp),
                    burn_in=int(gibbs_burnin),
                    thinning=int(gibbs_thin),
                    seed=int(seed) + idx,
                    progress=bool(progress),
                    progress_mode="samples",
                    progress_desc=f"Gibbs chain {idx + 1}/{n_chains} samples",
                    progress_position=idx if progress and n_chains > 1 else None,
                )
                if part.size:
                    parts.append(part)
            labels = np.concatenate(parts, axis=0) if parts else np.zeros((0, len(model.h)), dtype=np.int32)
            save_sample_npz(sample_path, labels=labels)
            return SamplingResult(sample_path=sample_path, n_samples=int(labels.shape[0]), n_residues=int(labels.shape[1]))

        # Replica exchange
        if rex_betas.strip():
            betas = _parse_float_list(rex_betas)
        else:
            betas = make_beta_ladder(
                beta_min=float(rex_beta_min),
                beta_max=float(rex_beta_max),
                n_replicas=int(rex_n_replicas),
                spacing=str(rex_spacing),
            )
        if all(abs(b - float(beta)) > 1e-12 for b in betas):
            betas = sorted(set(betas + [float(beta)]))
        betas = [float(b) for b in betas]

        total_rounds = max(1, int(rex_rounds))
        n_chains = max(1, int(rex_chains))
        if n_chains > total_rounds:
            n_chains = total_rounds
        if n_chains > 1:
            base_rounds = total_rounds // n_chains
            extra = total_rounds % n_chains
            chain_rounds = [base_rounds + (1 if i < extra else 0) for i in range(n_chains)]
        else:
            chain_rounds = [total_rounds]

        chain_runs: List[dict[str, object] | None] = [None] * n_chains
        burnin_clipped = False

        if n_chains == 1:
            burn_in = min(int(rex_burnin_rounds), max(0, total_rounds - 1))
            burnin_clipped = burn_in != int(rex_burnin_rounds)
            chain_runs[0] = replica_exchange_gibbs_potts(
                model,
                betas=betas,
                sweeps_per_round=int(rex_sweeps_per_round),
                n_rounds=int(total_rounds),
                burn_in_rounds=int(burn_in),
                thinning_rounds=int(rex_thin_rounds),
                seed=int(seed),
                progress=bool(progress),
                progress_mode="samples",
                progress_desc="REX samples",
            )
        else:
            with ProcessPoolExecutor(max_workers=n_chains) as executor:
                futures = {}
                for idx in range(n_chains):
                    rounds = int(chain_rounds[idx])
                    burn_in = min(int(rex_burnin_rounds), max(0, rounds - 1))
                    futures[executor.submit(
                        _run_rex_chain_worker,
                        {
                            "model": model,
                            "betas": betas,
                            "sweeps_per_round": int(rex_sweeps_per_round),
                            "n_rounds": rounds,
                            "burn_in_rounds": burn_in,
                            "thinning_rounds": int(rex_thin_rounds),
                            "seed": int(seed) + idx,
                            "progress": bool(progress),
                            "progress_every": max(1, rounds // 20) if rounds else 1,
                            "progress_mode": "samples",
                            "progress_desc": f"REX chain {idx + 1}/{n_chains} samples",
                            "progress_position": idx,
                        },
                    )] = (idx, burn_in != int(rex_burnin_rounds))

                completed = 0
                for future in as_completed(futures):
                    idx, clipped = futures[future]
                    chain_runs[idx] = future.result()
                    burnin_clipped = burnin_clipped or clipped
                    completed += 1
                    report(f"Replica exchange chains {completed}/{n_chains}", 10 + int(80 * completed / max(1, n_chains)))

        if burnin_clipped:
            print("[rex] note: burn-in rounds truncated for short chains.")

        parts = []
        for run in chain_runs:
            if not isinstance(run, dict):
                continue
            samples_by_beta = run.get("samples_by_beta")
            if not isinstance(samples_by_beta, dict):
                continue
            arr = samples_by_beta.get(float(beta))
            if isinstance(arr, np.ndarray) and arr.size:
                parts.append(arr)
        labels = np.concatenate(parts, axis=0) if parts else np.zeros((0, len(model.h)), dtype=np.int32)
        save_sample_npz(sample_path, labels=labels)
        return SamplingResult(sample_path=sample_path, n_samples=int(labels.shape[0]), n_residues=int(labels.shape[1]))

    # SA/QUBO
    if (sa_beta_hot and not sa_beta_cold) or (sa_beta_cold and not sa_beta_hot):
        raise ValueError("Provide both --sa-beta-hot and --sa-beta-cold, or neither.")
    beta_range = None
    if sa_beta_hot and sa_beta_cold:
        beta_range = (float(sa_beta_hot), float(sa_beta_cold))

    qubo = potts_to_qubo_onehot(model, beta=float(beta), penalty_safety=float(penalty_safety))
    init_rng = np.random.default_rng(int(seed) + 1000)
    init_labels = _build_sa_initial_labels(
        mode=str(sa_init),
        md_labels=md_labels,
        model=model,
        beta=float(beta),
        n_reads=int(sa_reads),
        md_frame=int(sa_init_md_frame),
        rng=init_rng,
    )
    init_states = encode_onehot(init_labels, qubo) if init_labels is not None and init_labels.size else None

    Z = sa_sample_qubo_neal(
        qubo,
        n_reads=int(sa_reads),
        sweeps=int(sa_sweeps),
        seed=int(seed),
        progress=bool(progress),
        beta_range=beta_range,
        initial_states=init_states,
    )

    repair_mode = None if str(repair) == "none" else str(repair)
    labels = np.zeros((Z.shape[0], len(qubo.var_slices)), dtype=np.int32)
    valid_counts = np.zeros(Z.shape[0], dtype=np.int32)
    for i in range(Z.shape[0]):
        x, valid = decode_onehot(Z[i], qubo, repair=repair_mode)
        labels[i] = x
        valid_counts[i] = int(valid.sum())
    invalid_mask = np.array([np.any(qubo.constraint_violations(z) != 0) for z in Z], dtype=bool)

    save_sample_npz(
        sample_path,
        labels=labels,
        invalid_mask=invalid_mask,
        valid_counts=valid_counts,
    )
    return SamplingResult(sample_path=sample_path, n_samples=int(labels.shape[0]), n_residues=int(labels.shape[1]))

