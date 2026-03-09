# SA/QUBO sampling

This page documents the **current** SA implementation used by Potts sampling.

Important: this SA path is a **heuristic annealer on the one-hot QUBO**, not an exact sampler of the Potts Boltzmann distribution. If you want equilibrium/Boltzmann sampling, use Gibbs or REX-Gibbs.

## QUBO mapping

Each residue `r` with `K_r` discrete states is encoded with one-hot binary variables `z_{r,k}` and the constraint:

`sum_k z_{r,k} = 1`

The Potts energy is mapped to a QUBO:

- linear / quadratic Potts terms are scaled by the Potts sampling inverse temperature `beta`
- one-hot constraints are enforced with a quadratic penalty

Penalty term per residue:

`lambda_r (sum_k z_{r,k} - 1)^2`

where `lambda_r` is chosen from local Potts energy bounds and scaled by `penalty-safety`.

Two different inverse-temperature concepts appear here:

- Potts `beta`: used when converting the Potts Hamiltonian to the QUBO
- SA `beta_hot -> beta_cold`: the annealing schedule used by `neal` while optimizing that QUBO

So the SA schedule is **not** the same thing as the Potts model `beta`.

## What One SA Sweep Means

The backend uses `neal.SimulatedAnnealingSampler`.

One SA **sweep** is one full Metropolis update pass over **all QUBO bits**, not over Potts residues directly.

If you run `sa_sweeps = 2000`, those 2000 sweeps are distributed across the full annealing path from hot to cold. This does **not** mean “2000 extra cold steps at the end”.

Practical consequence:

- increasing sweeps gives the annealer more time to relax
- but if the schedule starts too hot, many sweeps can still spend substantial time far from the warm-start basin

## SA Init: What The First Sample Starts From

`sa_init` defines the starting labels of:

- every read in `independent` mode
- the **first** sample of each correlated SA chain in `previous` / `md` restart modes

Options:

- `md`:
  - pick a random MD frame from the cluster labels
  - if `sa_md_state_ids` is set, only frames from those states are used
- `md-frame`:
  - use the specified fixed MD frame index
- `random-h`:
  - initialize each residue independently from the field-only distribution
  - `p(x_i) ∝ exp(-beta * h_i)`
- `random-uniform`:
  - initialize each residue uniformly at random

If MD labels are unavailable:

- `sa_init = md` falls back to `random-h`
- `sa_init = md-frame` fails

## SA Restart: What Later Samples Start From

Current supported restart modes:

- `previous`
- `md`
- `independent` (default)

### `previous`

This creates a **correlated SA chain**:

1. sample 1 starts from `sa_init`
2. SA runs one full hot→cold anneal
3. the decoded result of sample 1 becomes the start of sample 2
4. sample 2 is then **reheated again** and annealed again
5. repeat

This is the key point: `previous` does **not** continue a cold chain. It restarts each sample from the previous labels, but then runs the full annealing schedule again from hot to cold.

So `previous` means:

- correlated sequence of local searches
- not equilibrium sampling
- not monotone energy descent across samples

### `md`

Each sample restarts from a **fresh random MD frame**.

This ignores the previous sample entirely. It is useful when you want many local descents starting from the empirical MD basin instead of one correlated chain.

Note: if `sa_init = md-frame`, that fixed frame is used only for the **first** sample; later samples still restart from fresh random MD frames.

### `independent`

Each read is independent:

- each read starts from `sa_init`
- no carry-over between reads

This is the cleanest mode if your goal is “many unrelated SA attempts”.

## Why MD Warm-Start + Continue Chain Can Still Give High Energies

If you use:

- `sa_init = md`
- `sa_restart = independent`
- schedule `0.6 -> 10`

it is still normal to see relatively high energies.

Main reasons:

1. **SA is not Boltzmann sampling**
   - unlike Gibbs / REX, it is not designed to reproduce the target equilibrium distribution

2. **Each sample reheats**
   - `previous` does not stay at the cold end
   - every new sample goes through the hot part of the schedule again

3. **A hot start can erase the warm-start**
   - if `beta_hot` is too small, the chain quickly forgets the MD frame it started from

4. **QUBO optimization is not identical to Potts energy minimization after decoding**
   - SA optimizes the penalized one-hot QUBO
   - diagnostics/plots usually show energies of the decoded Potts labels
   - invalid one-hot outputs can therefore look worse after decoding

5. **Invalid samples matter**
   - if many reads violate one-hot constraints and you keep them, decoded labels may have poor Potts energies

## Practical Guidance

If your goal is **Boltzmann sampling**:

- prefer Gibbs or REX-Gibbs

If your goal is **low-energy search near MD basins**:

- use `sa_init = md`
- prefer `sa_restart = md` or `independent` for stable low-invalidity runs
- use more sweeps
- use a colder / narrower schedule instead of reheating too aggressively

Examples:

- more exploratory:
  - `beta_hot = 0.3`, `beta_cold = 10`
- more local descent from MD:
  - `beta_hot = 1.5`, `beta_cold = 10` or `15`

The right values still depend strongly on the scale of the QUBO coefficients and the penalty strength.

## Validity And Repair

SA runs on the binary QUBO and can produce samples that violate one-hot constraints.

Reported outputs include:

- decoded labels
- invalid mask
- valid-count per sample

`repair = argmax` projects each residue slice to one label by argmax.

`repair = none` keeps the invalid mask and reports decoded labels without user-facing repair. Internally, correlated-chain restarts still project the previous QUBO bitstring to a valid label assignment for the next start, to avoid pathological resets of invalid residues.

## Exposed User Knobs

The UI and phase console expose the `neal` options that materially affect the annealing dynamics:

- `sa_reads`
- `sa_chains`
- `sa_sweeps`
- `sa_schedule_type = geometric | linear`
- `sa_custom_beta_schedule`
- `sa_num_sweeps_per_beta`
- `sa_randomize_order`
- `sa_acceptance_criteria = Metropolis | Gibbs`
- `sa_init`
- `sa_init_md_frame`
- `sa_restart`
- `sa_md_state_ids`
- `penalty_safety`
- `repair`

Schedule configuration is grouped into three modes:

- `auto`:
  - let `neal` choose the beta range automatically
  - you still choose `sa_schedule_type`, `sa_sweeps`, and `sa_num_sweeps_per_beta`
- `range`:
  - you provide `sa_beta_hot` and `sa_beta_cold`
  - `neal` interpolates between them using `sa_schedule_type`
- `custom`:
  - you provide the exact beta list in `sa_custom_beta_schedule`
  - `sa_num_sweeps_per_beta` controls how long each listed beta is held
  - in this mode, `sa_sweeps` is not the main control; total work is driven by:
    - `len(sa_custom_beta_schedule) * sa_num_sweeps_per_beta`

We intentionally do **not** expose two `neal` arguments:

- `initial_states_generator`
- `interrupt_function`

Reason:

- this workflow already constructs explicit initial states for all reads/chains
- interruption/cancellation is handled at the job level, not inside the sampler

## Related docs

- [Potts analysis overview](doc:potts_overview)
- [Potts model fitting](doc:potts_model)
- [Gibbs and replica exchange](doc:potts_gibbs)
- [beta_eff calibration](doc:potts_beta_eff)
