"""
Goal 2: The Static Atlas (Set Cover / Dominating Set).

QUBO implementation with:

- Optional per-state Information Imbalance (active / inactive).
- Soft weighting of coverage instead of hard Δ threshold.
- Explicit hub scores based on soft coverage weights.
- Regularization of state scores via a threshold.
- Per-state hub / coverage exported for downstream classification.

Expected input from runner:
    analyzer = QUBOMaxCoverage()
    res = analyzer.run((features_act, features_inact),
                       candidate_indices=...,
                       candidate_state_scores=...,
                       **params)

Where:
    features_act, features_inact : FeatureDict
        Mapping residue index -> np.ndarray of shape (n_frames, d).
    candidate_indices : List[int] or List[str]
        Indices of residues considered in the QUBO.
    candidate_state_scores : Dict[index, float]
        State sensitivity score (e.g. JSD) from Goal 1, per residue.
"""

from __future__ import annotations

import json
import numpy as np
import multiprocessing as mp
from typing import Tuple, List, Dict, Any, Sequence, Hashable

from alloskin.common.types import FeatureDict

# Optional QUBO stack
try:
    import pyqubo
    from neal import SimulatedAnnealingSampler
    QUBO_AVAILABLE = True
except ImportError:
    QUBO_AVAILABLE = False

# Optional dadapy dependency for Information Imbalance
try:
    from dadapy.metric_comparisons import MetricComparisons
    DADAPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    DADAPY_AVAILABLE = False


_IMBALANCE_SHARED = {
    "keys_list": None,
    "coords": None,
    "ranks": None,
}


def _init_imbalance_worker(keys_list, coords, ranks):
    """Store shared data in globals for pool workers to avoid large pickling."""
    _IMBALANCE_SHARED["keys_list"] = keys_list
    _IMBALANCE_SHARED["coords"] = coords
    _IMBALANCE_SHARED["ranks"] = ranks


def _compute_imbalance_row(task_args):
    """Worker to compute imbalance entries for a single source residue."""
    a, k_i, mc_i = task_args
    keys_list = _IMBALANCE_SHARED["keys_list"]
    coords = _IMBALANCE_SHARED["coords"]
    ranks = _IMBALANCE_SHARED["ranks"]

    dim_indices = list(range(coords[k_i].shape[1]))

    row_results = []
    for b in range(a + 1, len(keys_list)):
        k_j = keys_list[b]
        ranks_j = ranks[k_j]

        try:
            imb_ji, imb_ij = mc_i.return_inf_imb_target_selected_coords(
                target_ranks=ranks_j,
                coord_list=[dim_indices],
            )
        except Exception:
            # treat as fully imbalanced (no coverage)
            imb_ij = 1.0
            imb_ji = 1.0

        row_results.append((b, float(imb_ij), float(imb_ji)))

    return a, row_results


class QUBOMaxCoverage:
    """
    Static Atlas via QUBO:
    ----------------------

    Let x_i be a binary variable indicating whether residue i is in the Basis Set.

    We define:

        w_ij  : soft coverage weight of child j by parent i,
                derived from Information Imbalance Δ(i→j).
        hub_i = sum_j w_ij  (hub score)

    Objective (to minimize):

        H =  sum_i [ + alpha * x_i
                     - beta_switch * s_i * x_i
                     - beta_hub    * hub_i * x_i ]
             + sum_{i<j} [ gamma_overlap * overlap_ij * x_i * x_j
                           + gamma_direct * direct_ij * x_i * x_j ]

    where:
        s_i         : regularized state sensitivity score of residue i
                      (after applying ii_threshold).
        overlap_ij  : weighted overlap of domains of i and j:
                      sum_k min(w_ik, w_jk).
        direct_ij   : redundancy from mutual direct coverage:
                      1 if w_ij>0 or w_ji>0, else 0.

    Soft coverage weights:

        Δ_eff(i→j) = combination of per-state Δ_act, Δ_inact (here average).
        w_ij = max(0, 1 - (Δ_eff(i→j)/ii_scale)) ** p

    With ii_scale ≈ 0.6, any Δ ≥ 0.6 yields w_ij = 0 → no coverage, no hub.
    """

    def __init__(self) -> None:
        # Used to store per-state imbalance matrices on last run
        self._IIM_list: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(
        self,
        features: FeatureDict,
        *,
        candidate_indices: Sequence[Hashable] | None = None,
        candidate_state_scores: Dict[Hashable, float] | None = None,
        static_results: Dict[str, Any] | None = None,
        static_results_path: str | None = None,
        filter_min_id: float = 1.5,
        filter_top_jsd: int | None = 20,
        filter_top_total: int | None = 120,
        # Info imbalance / coverage hyperparameters
        ii_scale: float = 0.6,
        soft_threshold_power: float = 2.0,
        # State-score regularization
        ii_threshold: float | None = 0.9,
        maxk: int | None = None,
        # QUBO hyperparameters
        alpha: float = 1.0,
        beta_switch: float = 5.0,
        beta_hub: float = 1.0,
        gamma_redundancy: float = 2.0,
        # Solver
        num_solutions: int = 5,
        num_reads: int = 2000,
        seed: int | None = None,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        features
            FeatureDict mapping residue index -> array of shape (n_frames, d).
            Represents the combined feature set to optimize over.
        candidate_indices
            Residues to be considered in the QUBO.
        candidate_state_scores
            Mapping residue index -> state sensitivity score (e.g. JSD).
            If None, all scores are treated as 0.
        static_results
            Optional static analysis results used to pre-filter candidates and
            populate state scores when candidate_indices is not provided.
        static_results_path
            Path to a JSON file with static analysis results; used only when
            static_results is not provided.
        filter_min_id
            Minimum intrinsic-dimension threshold for inclusion when filtering
            from static_results/static_results_path.
        filter_top_jsd
            Guaranteed number of top state-score residues to keep when
            filtering from static results (None keeps all passing filter_min_id).
        filter_top_total
            Cap on total residues kept after filling by intrinsic dimension
            (None keeps all passing filter_min_id).
        ii_scale
            Scale / threshold parameter for normalizing Δ before soft
            coverage. If Δ >= ii_scale → w_ij = 0 (no coverage, no hub).
        soft_threshold_power
            Exponent p in w_ij = max(0, 1 - Δ/ii_scale)^p. Higher p sharpens
            the distinction between strong and weak coverage.
        ii_threshold
            State-score regularization threshold. If provided:
              - scores > ii_threshold  → mapped to 1.0
              - scores <= ii_threshold → mapped to score / ii_threshold
            so that all s_i lie in [0, 1].
        maxk
            Maximum neighborhood size for dadapy MetricComparisons
            (defaults to n_samples - 1).
        alpha
            Baseline linear cost per selected residue.
        beta_switch
            Linear reward for state sensitivity (switch-like behavior).
        beta_hub
            Linear reward for hub score (total coverage).
        gamma_redundancy
            Quadratic penalty for direct redundancy (mutual coverage).
        num_solutions
            Number of unique solutions to report from the annealer.
        num_reads
            Number of SA reads.
        seed
            Optional RNG seed for deterministic sampling.

        Returns
        -------
        Dict with fields:
            - "solutions": list of solution dicts
            - "matrix_indices": ordered list of residue indices used in QUBO
            - "imbalance_matrix": Δ_avg(i→j) as nested list
            - "coverage_weights": w_ij (avg) as nested list
            - "hub_scores": hub_i (avg) per residue index
            - "regularized_state_scores": s_i after applying ii_threshold
            - "parameters": hyperparameters used
            - "error": only present if QUBO stack not available or failure
        """
        if not QUBO_AVAILABLE:
            return {"error": "pyqubo / neal not available; cannot run QUBO."}

        # Determine candidate pool and accompanying state scores.
        keys, candidate_state_scores = self._prepare_candidates(
            features=features,
            candidate_indices=candidate_indices,
            candidate_state_scores=candidate_state_scores,
            static_results=static_results,
            static_results_path=static_results_path,
            filter_min_id=filter_min_id,
            filter_top_jsd=filter_top_jsd,
            filter_top_total=filter_top_total,
        )

        if len(keys) == 0:
            return {"error": "No candidate indices provided to QUBOMaxCoverage."}

        raw_state_scores = {k: float(candidate_state_scores.get(k, 0.0)) for k in keys}

        # --------------------------------------------------------------
        # 0. Regularize state scores with ii_threshold (if provided)
        # --------------------------------------------------------------
        if ii_threshold is not None and ii_threshold > 0.0:
            reg_scores: Dict[Hashable, float] = {}
            for k in keys:
                s_raw = raw_state_scores.get(k, 0.0)
                if s_raw > ii_threshold:
                    s_reg = 1.0
                else:
                    s_reg = s_raw / ii_threshold
                reg_scores[k] = s_reg
            candidate_state_scores = reg_scores
        else:
            candidate_state_scores = raw_state_scores

        # --------------------------------------------------------------
        # 1. Build feature arrays for each candidate
        # --------------------------------------------------------------
        features_act = features
        feat_act: Dict[Hashable, np.ndarray] = {}

        for k in keys:
            if k not in features_act:
                raise ValueError(f"Residue key {k} is missing in provided features.")
            arr_act = np.asarray(features_act[k])

            # Coerce features to 2D (n_frames, d)
            if arr_act.ndim == 1:
                arr_act = arr_act.reshape(-1, 1)
            elif arr_act.ndim > 2:
                arr_act = arr_act.reshape(arr_act.shape[0], -1)

            if arr_act.ndim != 2:
                raise ValueError(f"Feature arrays for key {k} must be 2D (n_frames, d).")

            feat_act[k] = arr_act

        # --------------------------------------------------------------
        # 2. Compute Information Imbalance Δ(i→j)
        # --------------------------------------------------------------
        print(f"[QUBO] Computing Information Imbalance for {len(keys)} candidates...")

        imbalance_matrix = self._compute_imbalance_matrix(
            keys,
            feat_act,
            maxk=maxk,
        )
        self._IIM_list.append(imbalance_matrix)

        # --------------------------------------------------------------
        # 3. Convert Δ to soft coverage weights w_ij (avg)
        # --------------------------------------------------------------
        coverage_weights, hub_scores = self._compute_soft_coverage(
            keys,
            imbalance_matrix,
            ii_scale=ii_scale,
            power=soft_threshold_power,
        )

        # --------------------------------------------------------------
        # 4. Build QUBO (h_i, J_ij) using average hub / coverage
        # --------------------------------------------------------------
        h_linear, J_quadratic = self._build_hamiltonian(
            keys,
            coverage_weights,
            hub_scores,
            candidate_state_scores,
            alpha=alpha,
            beta_switch=beta_switch,
            beta_hub=beta_hub,
            gamma_redundancy=gamma_redundancy,
        )

        # --------------------------------------------------------------
        # 5. Solve QUBO via Simulated Annealing
        # --------------------------------------------------------------
        try:
            x_vars = {str(k): pyqubo.Binary(str(k)) for k in keys}
            H_expr = 0.0

            for k, val in h_linear.items():
                H_expr += val * x_vars[str(k)]

            for (ki, kj), val in J_quadratic.items():
                if val == 0.0:
                    continue
                H_expr += val * x_vars[str(ki)] * x_vars[str(kj)]

            model = H_expr.compile()
            qubo, offset = model.to_qubo()

            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample_qubo(
                qubo,
                num_reads=num_reads,
                seed=seed,
            )

            # Collect unique solutions (by binary pattern)
            solutions: List[Dict[str, Any]] = []
            seen_patterns = set()

            for sample, energy in zip(sampleset.record.sample, sampleset.record.energy):
                pattern = tuple(int(v) for v in sample)
                if pattern in seen_patterns:
                    continue
                seen_patterns.add(pattern)

                selected_indices = [
                    keys[i] for i, bit in enumerate(pattern) if bit == 1
                ]

                # Compute union coverage and per-residue metrics for this solution
                union_coverage, per_parent_coverage = \
                    self._compute_union_coverage(keys, coverage_weights, pattern)

                solutions.append(
                    {
                        "selected": selected_indices,
                        "energy": float(energy + offset),
                        "raw_energy": float(energy),
                        "union_coverage": float(union_coverage),
                        "per_parent_coverage": per_parent_coverage,
                        "pattern": pattern,
                    }
                )

                if len(solutions) >= num_solutions:
                    break

            result: Dict[str, Any] = {
                "solutions": solutions,
                "matrix_indices": keys,
                "imbalance_matrix": imbalance_matrix.tolist(),
                "coverage_weights": coverage_weights.tolist(),
                "hub_scores": {k: float(hub_scores[i]) for i, k in enumerate(keys)},
                "raw_state_scores": {k: float(raw_state_scores.get(k, 0.0)) for k in keys},
                "regularized_state_scores": {
                    k: float(candidate_state_scores[k]) for k in keys
                },
                "parameters": {
                    "alpha": alpha,
                    "beta_switch": beta_switch,
                    "beta_hub": beta_hub,
                    "gamma_redundancy": gamma_redundancy,
                    "ii_scale": ii_scale,
                    "ii_threshold": ii_threshold,
                    "soft_threshold_power": soft_threshold_power,
                    "num_reads": num_reads,
                    "num_solutions": num_solutions,
                },
            }

            return result

        except Exception as e:  # pragma: no cover
            print(f"[QUBO] Error solving QUBO: {e}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_static_results(
        self, data: Dict[str, Any] | None
    ) -> Dict[str, Dict[str, Any]] | None:
        """
        Extract a residue->stats mapping from a user-provided object.

        Accepts either the raw static results dict or a wrapper dict that
        contains such a mapping.
        """
        if not isinstance(data, dict):
            return None

        dict_values = {k: v for k, v in data.items() if isinstance(v, dict)}
        if dict_values and any("state_score" in v for v in dict_values.values()):
            return dict_values

        # Try to find a nested mapping
        for val in data.values():
            if isinstance(val, dict):
                nested = self._normalize_static_results(val)
                if nested:
                    return nested
        return None

    def _prepare_candidates(
        self,
        *,
        features: FeatureDict,
        candidate_indices: Sequence[Hashable] | None,
        candidate_state_scores: Dict[Hashable, float] | None,
        static_results: Dict[str, Any] | None,
        static_results_path: str | None,
        filter_min_id: float,
        filter_top_jsd: int | None,
        filter_top_total: int | None,
    ) -> Tuple[List[Hashable], Dict[Hashable, float]]:
        """
        Determine candidate residues and their state scores.

        Priority:
          1) static_results/static_results_path → apply filtering
          2) explicit candidate_indices
          3) fallback to all residues in `features`
        """
        static_data = static_results
        if static_data is None and static_results_path:
            try:
                with open(static_results_path, "r") as fh:
                    static_data = json.load(fh)
            except Exception as exc:
                raise ValueError(
                    f"Failed to load static_results_path '{static_results_path}': {exc}"
                )

        parsed_static = self._normalize_static_results(static_data) if static_data else None

        if parsed_static:
            movers = []
            for key, stats in parsed_static.items():
                if not isinstance(stats, dict):
                    continue
                id_val = float(stats.get("id", 0.0) or 0.0)
                state_score = float(stats.get("state_score", 0.0) or 0.0)
                if id_val >= filter_min_id:
                    movers.append((key, id_val, state_score))

            if filter_top_jsd is None:
                filter_top_jsd = len(movers)
            if filter_top_total is None:
                filter_top_total = len(movers)

            by_jsd = sorted(movers, key=lambda x: x[2], reverse=True)
            selected: List[Hashable] = []
            for key, _, _ in by_jsd:
                if len(selected) >= filter_top_jsd:
                    break
                selected.append(key)

            by_entropy = sorted(movers, key=lambda x: x[1], reverse=True)
            for key, _, _ in by_entropy:
                if len(selected) >= filter_top_total:
                    break
                if key not in selected:
                    selected.append(key)

            keys = [k for k in selected if k in features]
            state_scores = {
                k: float(parsed_static.get(k, {}).get("state_score", 0.0) or 0.0)
                for k in keys
            }
            return keys, state_scores

        # No static filtering → use user-provided candidates or everything
        if candidate_indices is not None:
            keys = [k for k in candidate_indices if k in features]
        else:
            keys = list(features.keys())

        base_scores = candidate_state_scores or {}
        state_scores = {k: float(base_scores.get(k, 0.0)) for k in keys}
        return keys, state_scores

    def _compute_imbalance_matrix(
        self,
        keys: Sequence[Hashable],
        feat_act: Dict[Hashable, np.ndarray],
        *,
        maxk: int | None,
    ) -> np.ndarray:
        """
        Compute Information Imbalance Δ(i→j) using dadapy, averaging
        both directions to form a symmetric matrix.
        """
        if not DADAPY_AVAILABLE:
            raise ImportError("dadapy is required for imbalance computation.")

        n = len(keys)
        keys_list = list(keys)

        # Prepare MetricComparisons and ranks for each residue
        n_samples = next(iter(feat_act.values())).shape[0]
        maxk = n_samples - 1 if maxk is None else min(maxk, n_samples - 1)

        coords: Dict[Hashable, np.ndarray] = {}
        comps: Dict[Hashable, MetricComparisons] = {}
        ranks: Dict[Hashable, np.ndarray] = {}

        half_period = np.pi
        for key in keys_list:
            arr = np.clip(feat_act[key].reshape(n_samples, -1) + half_period, 0, 2*half_period)
            coords[key] = arr
            mc = MetricComparisons(coordinates=arr, maxk=maxk, n_jobs=1)
            mc.compute_distances(period=2*half_period)
            comps[key] = mc
            ranks[key] = mc.dist_indices

        IIM = np.zeros((n, n), dtype=float)

        # Parallelize imbalance computation row-wise; each worker receives
        # one MetricComparisons instance (mc_i) and computes all k_js.
        tasks = [(a, keys_list[a], comps[keys_list[a]]) for a in range(n)]
        cpu_total = mp.cpu_count() or 1
        num_workers = min(len(tasks), max(1, cpu_total))

        try:
            ctx = mp.get_context("fork")
        except (AttributeError, ValueError):
            ctx = mp.get_context() if hasattr(mp, "get_context") else mp

        try:
            if num_workers == 1:
                # Avoid Pool startup cost when only one worker is available.
                _init_imbalance_worker(keys_list, coords, ranks)
                results = [_compute_imbalance_row(task) for task in tasks]
            else:
                with ctx.Pool(
                    processes=num_workers,
                    initializer=_init_imbalance_worker,
                    initargs=(keys_list, coords, ranks),
                ) as pool:
                    results = pool.map(_compute_imbalance_row, tasks)
        except Exception:
            # Fallback to sequential computation if multiprocessing fails.
            _init_imbalance_worker(keys_list, coords, ranks)
            results = [_compute_imbalance_row(task) for task in tasks]

        for a, row_results in results:
            for b, imb_ij, imb_ji in row_results:
                IIM[a, b] = imb_ij
                IIM[b, a] = imb_ji

        return IIM

    def _compute_soft_coverage(
        self,
        keys: Sequence[Hashable],
        imbalance_matrix: np.ndarray,
        *,
        ii_scale: float,
        power: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert imbalance Δ(i→j) into soft coverage weights w_ij:

            w_ij = max(0, 1 - Δ(i→j)/ii_scale) ** power

        With this choice, if Δ >= ii_scale → w_ij = 0
        (no coverage / no contribution to hub score).

        Returns
        -------
        coverage_weights : ndarray, shape (N, N)
        hub_scores       : ndarray, shape (N,)
        """
        N = len(keys)
        D = imbalance_matrix.copy()

        D_scaled = D / float(ii_scale)
        D_scaled = np.clip(D_scaled, 0.0, 1.0)

        W = np.power(np.maximum(0.0, 1.0 - D_scaled), power)
        np.fill_diagonal(W, 0.0)

        hub_scores = W.sum(axis=1)
        return W, hub_scores

    def _build_hamiltonian(
        self,
        keys,
        coverage_weights,
        hub_scores,
        state_scores,
        *,
        alpha,
        beta_switch,
        beta_hub,
        gamma_redundancy,   # NEW unified redundancy term
    ):
        """
        Build QUBO with a single redundancy penalty γ * overlap.
        """

        N = len(keys)
        keys_list = list(keys)

        h_linear = {}
        J_quadratic = {}

        W = coverage_weights

        # ------------------------------
        # Linear coefficients
        # ------------------------------
        for idx_i, key_i in enumerate(keys_list):
            s_i = float(state_scores.get(key_i, 0.0))
            hub_i = float(hub_scores[idx_i])

            h = alpha
            h -= beta_switch * s_i
            h -= beta_hub * hub_i

            h_linear[key_i] = float(h)

        # ------------------------------
        # Quadratic overlap penalty
        # ------------------------------
        for i in range(N):
            for j in range(i + 1, N):

                # domain overlap (= redundancy)
                R_ij = float(np.minimum(W[i], W[j]).sum())

                if R_ij > 0:
                    J_quadratic[(keys_list[i], keys_list[j])] = gamma_redundancy * R_ij

        return h_linear, J_quadratic


    def _compute_union_coverage(
        self,
        keys: Sequence[Hashable],
        coverage_weights: np.ndarray,
        pattern: Sequence[int],
    ) -> Tuple[float, Dict[Hashable, float]]:
        """
        Given a binary selection pattern, compute:

        - union_coverage: sum over children j of max_i w_ij
        - per_parent_coverage: hub score restricted to selected parents
        """
        keys_list = list(keys)
        W = coverage_weights

        selected_indices = [i for i, bit in enumerate(pattern) if bit == 1]
        per_parent: Dict[Hashable, float] = {}

        if not selected_indices:
            return 0.0, {}

        for i in selected_indices:
            k = keys_list[i]
            per_parent[k] = float(W[i].sum())

        union_weights = np.max(W[selected_indices, :], axis=0)
        union_coverage = float(union_weights.sum())

        return union_coverage, per_parent
