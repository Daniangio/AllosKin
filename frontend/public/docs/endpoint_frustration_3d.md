# Endpoint Analysis 3D Viewer

This viewer colors a loaded structure using the endpoint analysis currently selected on the cluster.

## What is colored

Choose a sample row from the selected endpoint analysis, then choose:

- `Commitment`
- `Frustration`

For frustration you can choose:

- `Symmetric`: residues strained under both endpoint models
- `Polarity`: sign of the local strain difference between the two endpoint models

## Commitment modes

- `Base`: direct probability `q`
- `Centered`: `q` recentered using the chosen reference ensemble(s)
- `Mean field`: smooth sigmoid transform of the mean local delta field

Use centered mode when you want neutral residues closer to white.

## Frustration display modes

For frustration, PHASE now supports:

- `Raw normalized`
- `Centered vs reference MD`

Centered frustration subtracts the residue-wise or edge-wise mean of the selected reference MD trajectories already present in the analysis. This makes it easier to highlight deviations from a chosen baseline instead of only absolute normalized strain.

## Frame browser

The page can also browse framewise frustration for the selected sample:

- request only a frame window using `start`, `stop`, `step`
- move inside that window with a frame slider
- jump to backend-ranked hotspot frames

Ranking modes:

- top symmetric hotspot score
- top absolute polarity score

Only the requested slice is sent to the browser.

## Edge-weighted residue coloring

When enabled, residue values are blended with the average value of incident top edges.

- edge weight = `|ΔJ|`
- blend strength default = `0.75`
- this is off by default on webserver
- color contrast is re-normalized after blending so enabling edge-weighted mode does not wash out the palette

## Residue mapping

- `Sequential (label_seq_id)`: most robust default
- `PDB numbering (auth_seq_id)`: use only when the residue numbers in the Potts cluster labels match the loaded PDB

## Interpreting colors

The viewer stores a normalized color value in `[0, 1]`.

- commitment:
  - near `0`: B-like (blue)
  - near `0.5`: neutral
  - near `1`: A-like (red)
- symmetric frustration:
  - raw mode: green = low strain, red = high strain
  - centered mode: green = lower than reference MD, red = higher than reference MD
- polarity frustration:
  - near `0`: more A-like (red; less frustrated under model A)
  - near `0.5`: balanced
  - near `1`: more B-like (blue)

In centered frustration mode, values are shown relative to the selected reference MD baseline, so white means “close to baseline” rather than “low absolute frustration”.

The hover label reports the displayed color value for the current sample and metric.
