import os
from types import SimpleNamespace

import numpy as np

os.environ["PHASE_DATA_ROOT"] = "/tmp/phase-test-data"

from phase.potts import orchestration
from phase.potts.qubo import QUBO
from phase.potts.sampling_run import _sa_project_labels_for_restart


def test_sa_restart_projection_uses_argmax_per_residue_slice():
    qubo = QUBO(
        a=np.zeros(5, dtype=float),
        Q={},
        const=0.0,
        var_slices=[slice(0, 2), slice(2, 5)],
        K_list=[2, 3],
    )
    z = np.array([0, 1, 0, 1, 1], dtype=int)

    x = _sa_project_labels_for_restart(z, qubo)

    assert x.tolist() == [1, 1]


def test_prepare_sampling_batch_accepts_custom_sa_schedule(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "phase.potts.sampling_run._load_combined_model",
        lambda _: SimpleNamespace(h=[np.zeros(2), np.zeros(3)]),
    )
    monkeypatch.setattr(
        "phase.potts.sampling_run._normalize_model_paths",
        lambda model_npz: [str(v) for v in model_npz],
    )

    out = orchestration.prepare_sampling_batch(
        cluster_npz=str(tmp_path / "cluster.npz"),
        results_dir=tmp_path / "sample-out",
        model_npz=[str(tmp_path / "model.npz")],
        sampling_method="sa",
        beta=1.0,
        seed=7,
        sa_reads=5,
        sa_chains=2,
        sa_sweeps=100,
        sa_schedule_type="custom",
        sa_custom_beta_schedule=[0.5, 1.0, 2.0, 4.0],
        sa_num_sweeps_per_beta=3,
        sa_randomize_order=True,
        sa_acceptance_criteria="Gibbs",
    )

    payloads = out["payloads"]
    assert len(payloads) == 2
    assert payloads[0]["sa_schedule_type"] == "custom"
    assert payloads[0]["sa_custom_beta_schedule"] == [0.5, 1.0, 2.0, 4.0]
    assert payloads[0]["sa_num_sweeps_per_beta"] == 3
    assert payloads[0]["sa_randomize_order"] is True
    assert payloads[0]["sa_acceptance_criteria"] == "Gibbs"


def test_prepare_sampling_batch_accepts_linear_range_schedule(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "phase.potts.sampling_run._load_combined_model",
        lambda _: SimpleNamespace(h=[np.zeros(2)]),
    )
    monkeypatch.setattr(
        "phase.potts.sampling_run._normalize_model_paths",
        lambda model_npz: [str(v) for v in model_npz],
    )

    out = orchestration.prepare_sampling_batch(
        cluster_npz=str(tmp_path / "cluster.npz"),
        results_dir=tmp_path / "sample-out",
        model_npz=[str(tmp_path / "model.npz")],
        sampling_method="sa",
        beta=1.0,
        seed=3,
        sa_reads=4,
        sa_chains=1,
        sa_sweeps=200,
        sa_beta_hot=0.8,
        sa_beta_cold=10.0,
        sa_schedule_type="linear",
        sa_num_sweeps_per_beta=2,
    )

    payload = out["payloads"][0]
    assert payload["beta_range"] == (0.8, 10.0)
    assert payload["sa_schedule_type"] == "linear"
    assert payload["sa_num_sweeps_per_beta"] == 2


def test_prepare_sampling_batch_uses_stronger_default_penalty_safety(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "phase.potts.sampling_run._load_combined_model",
        lambda _: SimpleNamespace(h=[np.zeros(2)]),
    )
    monkeypatch.setattr(
        "phase.potts.sampling_run._normalize_model_paths",
        lambda model_npz: [str(v) for v in model_npz],
    )

    out = orchestration.prepare_sampling_batch(
        cluster_npz=str(tmp_path / "cluster.npz"),
        results_dir=tmp_path / "sample-out",
        model_npz=[str(tmp_path / "model.npz")],
        sampling_method="sa",
        beta=1.0,
        seed=11,
    )

    assert out["payloads"][0]["penalty_safety"] == 8.0


def test_prepare_sampling_batch_defaults_to_independent_sa_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "phase.potts.sampling_run._load_combined_model",
        lambda _: SimpleNamespace(h=[np.zeros(2)]),
    )
    monkeypatch.setattr(
        "phase.potts.sampling_run._normalize_model_paths",
        lambda model_npz: [str(v) for v in model_npz],
    )

    out = orchestration.prepare_sampling_batch(
        cluster_npz=str(tmp_path / "cluster.npz"),
        results_dir=tmp_path / "sample-out",
        model_npz=[str(tmp_path / "model.npz")],
        sampling_method="sa",
        beta=1.0,
        seed=12,
        sa_reads=5,
        sa_chains=2,
    )

    assert out["payloads"][0]["worker_kind"] == "sa_independent"
