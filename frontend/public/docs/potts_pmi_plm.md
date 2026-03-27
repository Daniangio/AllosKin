# PMI and PLM in Potts models

This note explains the two fitting strategies used in PHASE: PMI (a fast heuristic)
and PLM (pseudolikelihood maximization).

## PMI (pointwise mutual information)
PMI uses empirical single-site and pairwise frequencies to build a quick initializer:

- Single-site fields:
  h_r(k) = -log p_r(k)
- Pairwise couplings:
  J_rs(k,l) = -log( p_rs(k,l) / (p_r(k) p_s(l)) )

In practice we add a small epsilon to probabilities and optionally center h/J
to remove constant offsets. PMI is fast and useful for quick baselines or as an
initializer, but it is not a maximum-likelihood fit.

## PLM (pseudolikelihood maximization)
PLM maximizes the sum of conditional log-likelihoods:

sum_r log P(x_r | x_-r; h, J)

For each residue r, the conditional distribution is a softmax over its states:

logits_r(k) = h_r(k) + sum_{s in neighbors(r)} J_rs(k, x_s)

This avoids the full partition function and is robust for larger systems.

## Implementation sketch (PHASE)
The PLM fit in `phase/potts/potts_model.py` is a symmetric global fit:

1) Build neighbor lists from the contact edges.
2) Initialize h and J from PMI (optional).
3) Use PyTorch to optimize the negative pseudolikelihood:
   - Sample mini-batches of frames.
   - For each residue r, compute logits_r for the batch.
   - Apply log-softmax and accumulate cross-entropy against the true labels.
   - Add regularization on the effective parameters.
4) If zero-sum gauge is enabled, the forward pass uses differentiable projected views of `h` and `J` rather than mutating parameters in place after each optimizer step.
5) Use a cosine or fixed learning-rate schedule, with optional progress logging and optional gradient accumulation.

This approach avoids the "fit each residue separately then symmetrize" pattern,
and directly optimizes a single set of symmetric couplings.

## Gauge-consistent fitting

PHASE now keeps two concepts separate:

- raw optimizer-owned tensors
- projected zero-sum-gauge views used for forward, regularization, and export

This matters because in-place gauge projection after `optimizer.step()` can corrupt Adam state. The current implementation avoids that.

For delta fits, this is even more important:
- sparsity penalties are applied to the projected `Δh` and `ΔJ`
- otherwise the magnitude of the learned rewiring would depend on arbitrary gauge choice

## Gradient accumulation

PLM and delta-PLM both support gradient accumulation.

If:
- `batch_size = B`
- `grad_accum_steps = G`

then each optimizer step sees an effective batch size of `B * G`, while still loading only `B` frames at a time in memory.
