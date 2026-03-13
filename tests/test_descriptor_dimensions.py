from __future__ import annotations

import numpy as np

from phase.workflows.clustering import _effective_angle_columns, _extract_angles_array


def test_effective_angle_columns_drops_missing_zero_only_dihedrals():
    samples = np.array(
        [
            [0.1, -1.2, 3.05, 0.0, 0.0],
            [0.3, -1.0, 3.10, 0.0, 0.0],
            [0.2, -1.1, 3.00, 0.0, 0.0],
        ],
        dtype=float,
    )
    cols = _effective_angle_columns(samples)
    assert cols.tolist() == [0, 1, 2]


def test_extract_angles_array_keeps_all_descriptor_dimensions():
    arr = np.arange(20, dtype=float).reshape(4, 1, 5)
    out = _extract_angles_array({"res_1": arr}, "res_1")
    assert out is not None
    assert out.shape == (4, 5)
    np.testing.assert_allclose(out, arr[:, 0, :])
