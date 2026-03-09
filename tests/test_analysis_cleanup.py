import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ["PHASE_DATA_ROOT"] = "/tmp/phase-test-data"

from backend.api.v1 import analysis_cleanup


class _FakeStore:
    def __init__(self, cluster_dir: Path, sample_ids: list[str], model_ids: list[str] | None = None):
        self._cluster_dir = cluster_dir
        self._sample_ids = sample_ids
        self._model_ids = model_ids or []

    def ensure_cluster_directories(self, project_id: str, system_id: str, cluster_id: str):
        return {"cluster_dir": self._cluster_dir}

    def list_samples(self, project_id: str, system_id: str, cluster_id: str):
        return [{"sample_id": sid} for sid in self._sample_ids]

    def list_potts_models(self, project_id: str, system_id: str, cluster_id: str):
        return [{"model_id": mid} for mid in self._model_ids]


def _write_analysis(root: Path, analysis_type: str, analysis_id: str, meta: dict):
    analysis_dir = root / "analyses" / analysis_type / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "analysis_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    np.savez_compressed(analysis_dir / "analysis.npz", dummy=np.asarray([1], dtype=int))
    return analysis_dir


def test_cleanup_orphan_cluster_analyses_removes_md_vs_sample_with_missing_sample(monkeypatch, tmp_path):
    cluster_dir = tmp_path / "cluster"
    _write_analysis(
        cluster_dir,
        "md_vs_sample",
        "a1",
        {
            "analysis_type": "md_vs_sample",
            "analysis_id": "a1",
            "md_sample_id": "md_1",
            "sample_id": "potts_missing",
        },
    )
    monkeypatch.setattr(
        analysis_cleanup,
        "project_store",
        _FakeStore(cluster_dir=cluster_dir, sample_ids=["md_1"], model_ids=[]),
    )

    removed = analysis_cleanup.cleanup_orphan_cluster_analyses("p", "s", "c")

    assert removed == 1
    assert not (cluster_dir / "analyses" / "md_vs_sample" / "a1").exists()


def test_cleanup_orphan_cluster_analyses_keeps_delta_js_if_some_samples_still_exist(monkeypatch, tmp_path):
    cluster_dir = tmp_path / "cluster"
    _write_analysis(
        cluster_dir,
        "delta_js",
        "djs1",
        {
            "analysis_type": "delta_js",
            "analysis_id": "djs1",
            "model_a_id": "ma",
            "model_b_id": "mb",
            "reference_sample_ids_a": ["ref_a"],
            "reference_sample_ids_b": ["ref_b"],
            "summary": {"sample_ids": ["keep_me", "gone_me"]},
        },
    )
    monkeypatch.setattr(
        analysis_cleanup,
        "project_store",
        _FakeStore(cluster_dir=cluster_dir, sample_ids=["ref_a", "ref_b", "keep_me"], model_ids=["ma", "mb"]),
    )

    removed = analysis_cleanup.cleanup_orphan_cluster_analyses("p", "s", "c")

    assert removed == 0
    assert (cluster_dir / "analyses" / "delta_js" / "djs1").exists()


def test_cleanup_orphan_cluster_analyses_removes_delta_js_if_all_tracked_samples_are_gone(monkeypatch, tmp_path):
    cluster_dir = tmp_path / "cluster"
    _write_analysis(
        cluster_dir,
        "delta_js",
        "djs2",
        {
            "analysis_type": "delta_js",
            "analysis_id": "djs2",
            "model_a_id": "ma",
            "model_b_id": "mb",
            "reference_sample_ids_a": ["ref_a"],
            "reference_sample_ids_b": ["ref_b"],
            "summary": {"sample_ids": ["gone_1", "gone_2"]},
        },
    )
    monkeypatch.setattr(
        analysis_cleanup,
        "project_store",
        _FakeStore(cluster_dir=cluster_dir, sample_ids=["ref_a", "ref_b"], model_ids=["ma", "mb"]),
    )

    removed = analysis_cleanup.cleanup_orphan_cluster_analyses("p", "s", "c")

    assert removed == 1
    assert not (cluster_dir / "analyses" / "delta_js" / "djs2").exists()
