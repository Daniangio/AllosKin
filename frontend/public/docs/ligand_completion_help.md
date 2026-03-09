# Ligand Completion Analysis

This analysis tests how a local ligand-imprinted pattern completes globally under two endpoint Potts models.

## Inputs

- Model A and Model B (endpoint Potts models).
- One MD sample used as starting ensemble.
- Constraint source:
  - `manual`: provide constrained residues (indices or residue keys).
  - `delta_js_auto`: pick a Delta-JS analysis + sample; top impactful ligand-specific residues are auto-selected.

In `delta_js_auto`, residue impact is computed from the selected sample using Delta-JS partials
(node term + optional edge term), then top-K residues are chosen. You can also exclude residues
already used by `delta_js_edge` success filtering to avoid circularity.

Current scoring used for auto constraints:

- Node impact per residue `i`:
  - `node_i = D_i * |JS_i(A) - JS_i(B)|`
- Edge impact is accumulated from top Delta-JS edges touching residue `i`:
  - `edge_i = mean_{(i,j)} [ D_ij * |JS_ij(A) - JS_ij(B)| ]`
- Combined impact:
  - `impact_i = (1-alpha) * node_i + alpha * edge_i`

## Conditional objective

For each selected MD frame and each `lambda`:

- Build a target distribution on constrained residues from a local MD window.
- Add a penalty to fields:

`E_cond(s) = E_X(s) + lambda * sum_i w_i * phi_i(s_i)`

where:

- `X` is endpoint A or B.
- `phi_i(a) = -log(pi_i(a) + eps)`.
- `w_i` are constraint weights (`uniform`, `js_abs`, or `custom`).

## Sampling

- `SA` (default): geometric schedule (`beta_hot=0.8` to `beta_cold=50`).
- `Gibbs` (optional): fixed `beta`.
- Per frame/lambda/endpoint, a trajectory is run for `n_steps`.
- Metrics use only the last `tail_steps` states (subsampled to `n_samples_per_frame`).

So for each selected start frame and each `lambda`, the analysis evaluates only the trajectory tail
(for example the last 200 states if `tail_steps=200`).

## Success metric

For sampled states, compute:

`DeltaE_BA = E_B - E_A`

- In `deltae` mode, success is evaluated **state by state on the sampled tail**:
  - success under A for one sampled state means `DeltaE_BA > +margin`
  - success under B for one sampled state means `DeltaE_BA < -margin`
- For one start frame and one `lambda`, the displayed success value is then:
  - the **fraction of tail states** that satisfy the A condition
  - or the **fraction of tail states** that satisfy the B condition

So the success curves are **not binary pass/fail per trajectory**.
They are tail-state fractions in `[0,1]`, averaged later across selected start frames.

Alternative mode (`delta_js_edge`):

- Pick a previous Delta-JS analysis.
- Use its discriminative residue/edge sets (top-ranked, then filtered by `D` ranges).
- For each sampled tail, compute Delta-JS weighted scores to A and B:
  - node-weighted JS on selected residues
  - edge-weighted JS on selected edges
  - blended with `alpha` (`mixed = (1-alpha)*node + alpha*edge`)
- For one start frame, one `lambda`, and one endpoint target:
  - compute the tail distribution
  - compare that tail to reference A and reference B
  - obtain two scores: `JS_A_mixed` and `JS_B_mixed`
- A tail is counted as a **success under A** only if both are true:
  - it is absolutely close enough to A:
    - `JS_A_mixed <= js_success_threshold`
  - and it is also clearly closer to A than to B:
    - `JS_A_mixed + js_success_margin <= JS_B_mixed`
- Success under B is symmetric (swap A/B).

This means a tail can legitimately be:

- successful for A only
- successful for B only
- successful for neither

The last case is common if the trajectory tail is still intermediate, too noisy, or simply not close enough to either reference under the chosen threshold.

This mode avoids selecting extra manual residue sets and reuses the discriminative filter from Delta-JS.

### Recommended defaults

If you often see `0%` success for both A and B with `delta_js_edge`, the most common cause is that the success threshold is too strict for the tail length you are using.

Recommended defaults:

- `js_success_threshold = 0.15`
- `js_success_margin = 0.02`

Practical interpretation:

- threshold controls **absolute closeness** to one reference
- margin controls **how much better** one side must be than the other

If both A and B remain near `0%` even after relaxing the threshold:

- first increase `n_steps` from `1000` to `2000`
- then increase `tail_steps` from `200` to `400`

This usually helps more than lowering the threshold too aggressively, because it gives the trajectory more time to settle and makes the tail estimate less noisy.

## How to read the plots

### Success vs λ (MD→A vs MD→B)

- X axis: `lambda` (constraint strength).
- Y axis: success rate.
- Each point on the displayed mean curve is:
  - computed per start frame on the tail states only (last `tail_steps`),
  - then averaged across all selected start frames.
- Shaded bands are `±1 std` across start frames.

Interpretation:
- Higher red curve (`under A`) means constrained starts are easier to complete into A-like states.
- Higher blue curve (`under B`) means easier completion into B-like states.

### AUC distributions

- Boxplots over start frames.
- `AUC_A` and `AUC_B`: area under success-vs-`lambda` per frame.
- `AUC_B - AUC_A`: directional completion score (positive = B-favored).

### Mean ΔE_BA on sampled tails

- Uses only sampled tail states.
- `ΔE_BA = E_B - E_A`.
- Negative values: sampled states are more B-like.
- Positive values: sampled states are more A-like.

### Arrival JS (lower is closer)

- Also computed on sampled tails.
- `JS to A (under A)` and `JS to B (under B)` track how close generated distributions are to endpoint references.
- Lower means better match to that endpoint.

### LACS histogram

- This histogram is **not** one value per ligand.
- It shows the distribution of recombined LACS scores **across start frames**.
- Each bar counts how many start frames fall in a LACS range.
- LACS per frame is computed with current UI weights:

`LACS_frame = w_completion * completion + w_raw * raw - w_novelty * novelty`

- Changing weights updates the same per-frame values in real time (no recomputation job needed).

## Stored outputs

- Per-frame curves and summaries:
  - `success_a`, `success_b`
  - `auc_a`, `auc_b`, `auc_dir = auc_b - auc_a`
  - completion costs (`cost_a`, `cost_b`)
  - JS diagnostics (`js_a_under_a`, `js_b_under_b`, etc.)
  - novelty proxy (`min(JS_to_A, JS_to_B)`)
- LACS partial components (for visualization-time weighting):
  - `lacs_component_completion = auc_dir`
  - `lacs_component_raw = -raw_deltae`
  - `lacs_component_novelty = novelty_frame`

## LACS recombination in UI

UI combines stored partials as:

`LACS = w_completion * completion + w_raw * raw - w_novelty * novelty`

You can tune `w_*` in visualization without recomputing analysis.
