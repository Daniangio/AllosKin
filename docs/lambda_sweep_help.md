# Lambda Sweep (Validation Ladder 4)

This page runs and visualizes a **model-space interpolation** experiment between two endpoint Potts models:

- **Endpoint B** (λ = 0)
- **Endpoint A** (λ = 1)

For each λ on a uniform grid in `[0, 1]`, we build an interpolated model:

`E_λ = (1 - λ) E_B + λ E_A`

and draw Gibbs samples. The sweep is then compared against:

- **Reference sample A**: anchor for the λ = 1 endpoint
- **Reference sample B**: anchor for the λ = 0 endpoint
- **One or more comparison samples**: extra MD or Potts samples used for the match curves

## What To Look For

### 1) Endpoint Sanity (must pass)

- At **λ = 0**, samples should look close to **Reference sample B** (low node/edge JS).
- At **λ = 1**, samples should look close to **Reference sample A** (low node/edge JS).

If endpoints do not match, the interpolation story is not reliable.

### 2) Order Parameter (ΔE mean + IQR)

We compute an order parameter on sampled sequences:

`ΔE(s) = E_A(s) - E_B(s)`

The plot shows **mean ΔE** and its **IQR** (25–75% quantiles) vs λ.

Expected behavior:

- A smooth trend from B-like to A-like as λ increases.
- Large noise / non-smoothness usually indicates inconsistent endpoints or insufficient sampling.

### 3) JS Distance Curves vs Reference Samples

For each λ-sample and each selected reference/comparison sample, we compute:

- **Node JS**: divergence of per-residue marginals.
- **Edge JS**: divergence of joint distributions on Potts edges.

These curves tell you *which stored ensemble each λ-sample resembles*.

### 4) Match Curve D(λ) to Comparison Samples

For each comparison sample, the “match curve” uses a weighted distance:

`D(λ) = α · mean(node JS) + (1 - α) · mean(edge JS)`

Interpretation:

- A minimum at **λ\*** for a given comparison sample suggests that sample is best explained as an **intermediate deformation** between endpoints in the learned discrete landscape.

## Important Caveat

λ is a **control parameter in model space**, not a physical reaction coordinate. Use it for conditional generation and interpretability, not kinetics.
