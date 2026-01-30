# SA/QUBO sampling

This page describes how Potts sampling is mapped to a QUBO and sampled with simulated annealing (SA).

## QUBO mapping
Each residue uses one-hot binary variables z_{r,k} with a constraint sum_k z_{r,k} = 1.
The Potts energy is embedded into a QUBO with a quadratic penalty:

- Penalty term: lambda_r (sum_k z_{r,k} - 1)^2
- lambda_r is chosen from Potts energy bounds and scaled by `penalty-safety`.

The QUBO energy is scaled by beta before sampling.

## Simulated annealing
The implementation uses neal's `SimulatedAnnealingSampler` as a classical baseline.
Controls:
- `sa-reads`: number of SA reads (independent runs).
- `sa-sweeps`: sweeps per read.
- Optional beta schedule: `sa-beta-hot`, `sa-beta-cold`, or multiple schedules via `sa-beta-schedule`.

An "SA auto" schedule is always run using neal's defaults.

### SA initialization (warm-start)
Each SA read can start from a user-chosen initial label assignment. This is useful when you want SA to explore near MD basins instead of fully random initial states.

Options (`sa-init`):
- `md` (default): warm-start each read from a random MD frame in the cluster NPZ.
- `md-frame`: warm-start all reads from a fixed MD frame index (`sa-init-md-frame`).
- `random-h`: sample residues independently from `p(x_r) ∝ exp(-beta * h_r)` (uses only local fields).
- `random-uniform`: sample residues uniformly, independently per residue.

If MD labels are unavailable and `sa-init=md`, the sampler falls back to `random-h` and logs a warning.
If `sa-init=md-frame` but the frame index is invalid or labels are missing, the run fails early.

Note: initial states are passed to neal when supported; older neal versions may ignore them and fall back to random initialization (a warning is logged).

### Restarting across SA schedules
If you add multiple SA schedules, you can choose how to initialize later schedules from earlier ones (`sa-restart`):
- `independent` (default): each schedule starts from its own init (no carryover).
- `prev-topk`: initialize from the lowest-energy top-k samples of the previous schedule (`sa-restart-topk`).
- `prev-uniform`: initialize from a uniform random subset of previous schedule samples.

If you only run the default "SA auto" schedule, restart has no effect.

## Validity and repair
Samples can violate one-hot constraints. The pipeline reports invalid rates.
If `repair=argmax`, invalid samples are coerced to a valid label per residue.

## Related docs
- [Potts analysis overview](doc:potts_overview)
- [Potts model fitting](doc:potts_model)
- [Gibbs and replica exchange](doc:potts_gibbs)
- [beta_eff calibration](doc:potts_beta_eff)
