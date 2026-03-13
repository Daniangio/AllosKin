import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PHASE_DATA_ROOT", "/tmp/phase-test-data")

from phase.potts.analysis_run import compute_lambda_sweep_analysis
from phase.potts.potts_model import PottsModel, save_potts_model
from phase.potts.sample_io import save_sample_npz


def _write_sample(system_dir: Path, cluster_id: str, sample_id: str, meta: dict, *, labels: np.ndarray, labels_halo=None):
    sample_dir = system_dir / "clusters" / cluster_id / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "sample.npz"
    save_sample_npz(sample_path, labels=labels, labels_halo=labels_halo)
    payload = dict(meta)
    payload.setdefault("sample_id", sample_id)
    payload.setdefault("path", str(sample_path.relative_to(system_dir)))
    payload.setdefault("paths", {"summary_npz": str(sample_path.relative_to(system_dir))})
    (sample_dir / "sample_metadata.json").write_text(json.dumps(payload), encoding="utf-8")


def test_compute_lambda_sweep_analysis_supports_multiple_comparison_samples(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    project_id = "proj"
    system_id = "sys"
    cluster_id = "cluster1"
    system_dir = data_root / "projects" / project_id / "systems" / system_id
    cluster_dir = system_dir / "clusters" / cluster_id
    cluster_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cluster_dir / "cluster.npz",
        residue_keys=np.asarray(["res_1", "res_2"], dtype=str),
        merged__labels_assigned=np.asarray([[0, 0], [1, 1]], dtype=np.int32),
        cluster_counts=np.asarray([2, 2], dtype=np.int32),
    )

    _write_sample(
        system_dir,
        cluster_id,
        "ref-a",
        {"name": "Ref A", "type": "md_eval", "method": "md_eval"},
        labels=np.asarray([[0, 0], [0, 1], [0, 0]], dtype=np.int32),
        labels_halo=np.asarray([[0, 0], [1, 1], [0, 0]], dtype=np.int32),
    )
    _write_sample(
        system_dir,
        cluster_id,
        "ref-b",
        {"name": "Ref B", "type": "md_eval", "method": "md_eval"},
        labels=np.asarray([[1, 1], [1, 0], [1, 1]], dtype=np.int32),
        labels_halo=np.asarray([[1, 1], [1, 0], [1, 1]], dtype=np.int32),
    )
    _write_sample(
        system_dir,
        cluster_id,
        "cmp-md",
        {"name": "Cmp MD", "type": "md_eval", "method": "md_eval"},
        labels=np.asarray([[0, 1], [1, 1], [0, 1]], dtype=np.int32),
    )
    _write_sample(
        system_dir,
        cluster_id,
        "cmp-potts",
        {"name": "Cmp Potts", "type": "potts_sampling", "method": "gibbs"},
        labels=np.asarray([[1, 0], [1, 0], [0, 0]], dtype=np.int32),
    )
    _write_sample(
        system_dir,
        cluster_id,
        "lambda-0",
        {"name": "Lambda 0", "type": "potts_lambda_sweep", "method": "gibbs"},
        labels=np.asarray([[1, 1], [1, 0], [1, 1]], dtype=np.int32),
    )
    _write_sample(
        system_dir,
        cluster_id,
        "lambda-1",
        {"name": "Lambda 1", "type": "potts_lambda_sweep", "method": "gibbs"},
        labels=np.asarray([[0, 0], [0, 1], [0, 0]], dtype=np.int32),
    )

    model = PottsModel(
        h=[np.asarray([0.0, 1.0], dtype=float), np.asarray([0.0, 1.0], dtype=float)],
        J={(0, 1): np.asarray([[0.0, 0.2], [0.2, 0.0]], dtype=float)},
        edges=[(0, 1)],
    )
    model_a_path = cluster_dir / "model_a.npz"
    model_b_path = cluster_dir / "model_b.npz"
    save_potts_model(model, model_a_path)
    save_potts_model(model, model_b_path)

    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))

    out = compute_lambda_sweep_analysis(
        project_id=project_id,
        system_id=system_id,
        cluster_id=cluster_id,
        model_a_ref=str(model_a_path),
        model_b_ref=str(model_b_path),
        lambda_sample_ids=["lambda-0", "lambda-1"],
        lambdas=[0.0, 1.0],
        reference_sample_ids=["ref-a", "ref-b", "cmp-md", "cmp-potts"],
        md_label_mode="assigned",
        drop_invalid=True,
        alpha=0.5,
    )

    assert out["reference_sample_ids"] == ["ref-a", "ref-b", "cmp-md", "cmp-potts"]
    assert out["comparison_sample_ids"] == ["cmp-md", "cmp-potts"]
    assert out["reference_sample_names"] == ["Ref A", "Ref B", "Cmp MD", "Cmp Potts"]
    assert out["comparison_sample_names"] == ["Cmp MD", "Cmp Potts"]
    assert out["node_js_mean"].shape == (4, 2)
    assert out["edge_js_mean"].shape == (4, 2)
    assert out["combined_distance"].shape == (4, 2)
    assert out["comparison_ref_indices"].tolist() == [2, 3]
    assert out["lambda_star_by_reference"].shape == (2,)
    assert out["lambda_star_index_by_reference"].shape == (2,)
    assert out["match_min_by_reference"].shape == (2,)
