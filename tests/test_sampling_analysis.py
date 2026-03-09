import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PHASE_DATA_ROOT", "/tmp/phase-test-data")

from phase.potts.analysis_run import analyze_cluster_samples
from phase.potts.sample_io import save_sample_npz


def _write_sample(system_dir: Path, cluster_id: str, sample_id: str, meta: dict, *, labels: np.ndarray, invalid_mask=None):
    sample_dir = system_dir / "clusters" / cluster_id / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "sample.npz"
    save_sample_npz(sample_path, labels=labels, invalid_mask=invalid_mask)
    payload = dict(meta)
    payload.setdefault("sample_id", sample_id)
    payload.setdefault("path", str(sample_path.relative_to(system_dir)))
    payload.setdefault("paths", {"summary_npz": str(sample_path.relative_to(system_dir))})
    (sample_dir / "sample_metadata.json").write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_cluster_samples_reports_all_invalid_sa_sample(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    system_dir = data_root / "projects" / "proj" / "systems" / "sys"
    cluster_id = "cluster1"
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
        "md1",
        {"name": "MD 1", "type": "md_eval", "method": "md_eval"},
        labels=np.asarray([[0, 0], [1, 1]], dtype=np.int32),
    )
    _write_sample(
        system_dir,
        cluster_id,
        "sa1",
        {"name": "SA 1", "type": "potts_sampling", "method": "sa"},
        labels=np.asarray([[0, 0], [1, 1], [0, 1]], dtype=np.int32),
        invalid_mask=np.asarray([True, True, True], dtype=bool),
    )

    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))

    out = analyze_cluster_samples(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        model_ref=None,
        md_label_mode="assigned",
        drop_invalid=True,
    )

    assert out["comparisons_written"] == 0
    assert out["energies_written"] == 0
    assert out["skipped_samples"] == [
        {
            "sample_id": "sa1",
            "sample_name": "SA 1",
            "sample_type": "potts_sampling",
            "sample_method": "sa",
            "stage": "md_vs_sample",
            "reason": "all_frames_invalid",
            "n_frames_total": 3,
            "n_frames_used": 0,
            "invalid_count": 3,
        }
    ]
