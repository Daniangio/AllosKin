import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phase.potts.potts_model import (
    PottsModel,
    fit_potts_delta_pseudolikelihood_torch,
    fit_potts_pseudolikelihood_torch,
)


def _assert_zero_sum_model(model: PottsModel, atol: float = 1e-5) -> None:
    for h in model.h:
        assert abs(float(np.mean(h))) <= atol
    for edge in model.edges:
        mat = np.asarray(model.J[edge], dtype=float)
        np.testing.assert_allclose(mat.sum(axis=0), 0.0, atol=atol)
        np.testing.assert_allclose(mat.sum(axis=1), 0.0, atol=atol)


def test_plm_zero_sum_export_is_centered():
    labels = np.asarray(
        [
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ],
        dtype=int,
    )
    model = fit_potts_pseudolikelihood_torch(
        labels,
        K=[2, 2, 2],
        edges=[(0, 1), (1, 2)],
        zero_sum_gauge=True,
        epochs=2,
        batch_size=3,
        lr=1e-2,
        l2=0.0,
        lambda_J_block=0.0,
        verbose=False,
        device="cpu",
    )
    _assert_zero_sum_model(model)


def test_plm_raw_export_preserves_ungauged_init_when_zero_sum_disabled():
    labels = np.asarray([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=int)
    init_model = PottsModel(
        h=[
            np.asarray([2.0, -0.5], dtype=float),
            np.asarray([1.5, -0.2], dtype=float),
        ],
        J={
            (0, 1): np.asarray(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                dtype=float,
            )
        },
        edges=[(0, 1)],
    )
    model = fit_potts_pseudolikelihood_torch(
        labels,
        K=[2, 2],
        edges=[(0, 1)],
        zero_sum_gauge=False,
        epochs=1,
        batch_size=4,
        lr=0.0,
        l2=0.0,
        lambda_J_block=0.0,
        verbose=False,
        init_from_pmi=False,
        init_model=init_model,
        device="cpu",
    )
    np.testing.assert_allclose(model.h[0], init_model.h[0], atol=1e-7)
    np.testing.assert_allclose(model.h[1], init_model.h[1], atol=1e-7)
    np.testing.assert_allclose(model.J[(0, 1)], init_model.J[(0, 1)], atol=1e-7)
    assert abs(float(np.mean(model.h[0]))) > 1e-3
    assert abs(float(np.mean(model.J[(0, 1)].sum(axis=0)))) > 1e-3


def test_plm_grad_accum_matches_large_batch():
    labels = np.asarray(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        dtype=int,
    )
    common = dict(
        K=[2, 2],
        edges=[(0, 1)],
        zero_sum_gauge=False,
        epochs=1,
        lr=1e-2,
        l2=0.0,
        lambda_J_block=0.0,
        verbose=False,
        init_from_pmi=False,
        device="cpu",
        seed=13,
    )
    full = fit_potts_pseudolikelihood_torch(labels, batch_size=4, grad_accum_steps=1, **common)
    accum = fit_potts_pseudolikelihood_torch(labels, batch_size=2, grad_accum_steps=2, **common)
    for r in range(2):
        np.testing.assert_allclose(full.h[r], accum.h[r], atol=1e-6)
    np.testing.assert_allclose(full.J[(0, 1)], accum.J[(0, 1)], atol=1e-6)


def test_delta_zero_sum_export_is_centered():
    base_model = PottsModel(
        h=[np.zeros(2, dtype=float), np.zeros(2, dtype=float)],
        J={(0, 1): np.zeros((2, 2), dtype=float)},
        edges=[(0, 1)],
    )
    init_delta = PottsModel(
        h=[
            np.asarray([2.0, -1.0], dtype=float),
            np.asarray([3.0, 0.5], dtype=float),
        ],
        J={
            (0, 1): np.asarray(
                [
                    [1.0, 2.0],
                    [4.0, 8.0],
                ],
                dtype=float,
            )
        },
        edges=[(0, 1)],
    )
    labels = np.asarray([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=int)
    delta_model = fit_potts_delta_pseudolikelihood_torch(
        base_model,
        labels,
        zero_sum_gauge=True,
        epochs=1,
        batch_size=4,
        grad_accum_steps=1,
        lr=0.0,
        l2=0.0,
        lambda_h=0.1,
        lambda_J=0.1,
        verbose=False,
        init_model=init_delta,
        device="cpu",
    )
    _assert_zero_sum_model(delta_model)
