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

## Edge-weighted residue coloring

When enabled, residue values are blended with the average value of incident top edges.

- edge weight = `|ΔJ|`
- blend strength default = `0.75`
- this is off by default on webserver

## Residue mapping

- `Sequential (label_seq_id)`: most robust default
- `PDB numbering (auth_seq_id)`: use only when the residue numbers in the Potts cluster labels match the loaded PDB

## Interpreting colors

The viewer stores a normalized color value in `[0, 1]`.

- commitment:
  - near `0`: inactive-leaning
  - near `0.5`: neutral
  - near `1`: active-leaning
- symmetric frustration:
  - near `0.5`: low strain
  - near `1`: high strain
- polarity frustration:
  - near `0`: inactive-like local strain
  - near `0.5`: balanced
  - near `1`: active-like local strain

The hover label reports the displayed color value for the current sample and metric.
