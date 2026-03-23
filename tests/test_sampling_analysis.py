import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PHASE_DATA_ROOT", "/tmp/phase-test-data")

from backend.tasks import run_md_samples_refresh_job
from phase.potts.analysis_run import analyze_cluster_samples, append_state_pose_energies
from phase.potts.potts_model import PottsModel, save_potts_model
from phase.potts.sample_io import save_sample_npz
from phase.services.project_store import DescriptorState, ProjectStore
from phase.workflows.clustering import _predict_cluster_adp


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


def test_append_state_pose_energies_reuses_existing_state_md_eval_sample_and_overwrites_pose(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))

    store = ProjectStore(base_dir=data_root / "projects")
    store.create_project("Project", project_id="proj")
    system = store.create_system("proj", name="System", system_id="sys")
    system.states["state1"] = DescriptorState(
        state_id="state1",
        name="Crystal pose",
        pdb_file="states/crystal/structure.pdb",
        storage_key="crystal",
    )
    store.save_system(system)

    state_dir = data_root / "projects" / "proj" / "systems" / "sys" / "states" / "crystal"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "structure.pdb").write_text("MODEL\nENDMDL\n", encoding="utf-8")

    cluster_id = "cluster1"
    cluster_dirs = store.ensure_cluster_directories("proj", "sys", cluster_id)
    np.savez_compressed(
        cluster_dirs["cluster_dir"] / "cluster.npz",
        residue_keys=np.asarray(["res_1", "res_2"], dtype=str),
        merged__labels_assigned=np.asarray([[0, 0], [1, 1]], dtype=np.int32),
        cluster_counts=np.asarray([2, 2], dtype=np.int32),
    )
    _write_sample(
        data_root / "projects" / "proj" / "systems" / "sys",
        cluster_id,
        "state-sample",
        {"name": "MD Crystal pose", "type": "md_eval", "method": "md_eval", "state_id": "state1"},
        labels=np.asarray([[1, 0]], dtype=np.int32),
    )

    model_dir = cluster_dirs["potts_models_dir"] / "model1"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.npz"
    save_potts_model(
        PottsModel(
            h=[np.asarray([0.0, 1.0], dtype=float), np.asarray([0.0, -1.0], dtype=float)],
            J={},
            edges=[],
        ),
        model_path,
    )

    energies_root = cluster_dirs["cluster_dir"] / "analyses" / "model_energy"
    base_dir = energies_root / "base-analysis"
    base_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(base_dir / "analysis.npz", energies=np.asarray([0.0, 1.0], dtype=float))
    (base_dir / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "analysis_id": "base-analysis",
                "analysis_type": "model_energy",
                "model_id": None,
                "model_name": "model",
                "sample_id": "md1",
                "sample_name": "MD 1",
                "sample_type": "md_eval",
            }
        ),
        encoding="utf-8",
    )

    out = append_state_pose_energies(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        model_ref=str(model_path),
        state_ids=["state1"],
    )
    assert out["pose_energies_written"] == 1

    pose_dirs = [p for p in energies_root.iterdir() if p.is_dir() and p.name != "base-analysis"]
    assert len(pose_dirs) == 1
    pose_meta = json.loads((pose_dirs[0] / "analysis_metadata.json").read_text(encoding="utf-8"))
    assert pose_meta["sample_type"] == "state_pose"
    assert pose_meta["sample_id"] == "state:state1"
    assert pose_meta["state_id"] == "state1"
    with np.load(pose_dirs[0] / "analysis.npz", allow_pickle=False) as data:
        assert np.allclose(np.asarray(data["energies"], dtype=float), np.asarray([1.0], dtype=float))

    out2 = append_state_pose_energies(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        model_ref=str(model_path),
        state_ids=["state1"],
    )
    assert out2["pose_energies_written"] == 1
    pose_dirs_after = [p for p in energies_root.iterdir() if p.is_dir() and p.name != "base-analysis"]
    assert len(pose_dirs_after) == 1


def test_append_state_pose_energies_falls_back_to_state_evaluation(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))

    store = ProjectStore(base_dir=data_root / "projects")
    store.create_project("Project", project_id="proj")
    system = store.create_system("proj", name="System", system_id="sys")
    system.states["state1"] = DescriptorState(
        state_id="state1",
        name="Crystal pose",
        pdb_file="states/crystal/structure.pdb",
        descriptor_file="states/crystal/descriptors.npz",
        storage_key="crystal",
    )
    store.save_system(system)

    state_dir = data_root / "projects" / "proj" / "systems" / "sys" / "states" / "crystal"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "structure.pdb").write_text("MODEL\nENDMDL\n", encoding="utf-8")

    cluster_id = "cluster1"
    cluster_dirs = store.ensure_cluster_directories("proj", "sys", cluster_id)
    np.savez_compressed(
        cluster_dirs["cluster_dir"] / "cluster.npz",
        residue_keys=np.asarray(["res_1", "res_2"], dtype=str),
        merged__labels_assigned=np.asarray([[0, 0], [1, 1]], dtype=np.int32),
        cluster_counts=np.asarray([2, 2], dtype=np.int32),
    )

    model_dir = cluster_dirs["potts_models_dir"] / "model1"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.npz"
    save_potts_model(
        PottsModel(
            h=[np.asarray([0.0, 1.0], dtype=float), np.asarray([0.0, -1.0], dtype=float)],
            J={},
            edges=[],
        ),
        model_path,
    )

    energies_root = cluster_dirs["cluster_dir"] / "analyses" / "model_energy"
    base_dir = energies_root / "base-analysis"
    base_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(base_dir / "analysis.npz", energies=np.asarray([0.0, 1.0], dtype=float))
    (base_dir / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "analysis_id": "base-analysis",
                "analysis_type": "model_energy",
                "model_id": None,
                "model_name": "model",
                "sample_id": "md1",
                "sample_name": "MD 1",
                "sample_type": "md_eval",
            }
        ),
        encoding="utf-8",
    )

    def _fake_evaluate_state_with_models(
        project_id,
        system_id,
        cluster_id,
        state_id,
        *,
        store=None,
        sample_id=None,
        workers=1,
        progress_callback=None,
    ):
        sample_dir = cluster_dirs["samples_dir"] / str(sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = sample_dir / "sample.npz"
        save_sample_npz(
            sample_path,
            labels=np.asarray([[1, 0]], dtype=np.int32),
            labels_halo=np.asarray([[1, 0]], dtype=np.int32),
            frame_indices=np.asarray([0], dtype=np.int64),
            frame_state_ids=np.asarray([state_id], dtype=str),
        )
        rel = str(sample_path.relative_to(cluster_dirs["system_dir"]))
        meta = {
            "sample_id": sample_id,
            "name": "MD Crystal pose",
            "type": "md_eval",
            "method": "md_eval",
            "state_id": state_id,
            "path": rel,
            "paths": {"summary_npz": rel},
        }
        (sample_dir / "sample_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        return meta

    monkeypatch.setattr("phase.potts.analysis_run.evaluate_state_with_models", _fake_evaluate_state_with_models)

    out = append_state_pose_energies(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        model_ref=str(model_path),
        state_ids=["state1"],
    )
    assert out["pose_energies_written"] == 1


def test_predict_cluster_adp_single_frame_duplicates_query():
    class DummyDP:
        X = np.zeros((5, 2), dtype=float)

        def predict_cluster_ADP(self, emb, *, maxk, density_est, n_jobs):
            assert emb.shape == (2, 2)
            assert maxk == 1
            return np.asarray([[3], [3]], dtype=np.int32), np.asarray([[4], [4]], dtype=np.int32)

    assigned, halo = _predict_cluster_adp(DummyDP(), np.asarray([[0.1, 0.2]], dtype=float), density_maxk=100)
    assert assigned.tolist() == [3]
    assert halo.tolist() == [4]


def test_predict_cluster_adp_trims_query_to_model_dimensions():
    class DummyDP:
        X = np.zeros((8, 4), dtype=float)

        def predict_cluster_ADP(self, emb, *, maxk, density_est, n_jobs):
            assert emb.shape == (2, 4)
            return np.asarray([[7], [7]], dtype=np.int32), np.asarray([[8], [8]], dtype=np.int32)

    assigned, halo = _predict_cluster_adp(
        DummyDP(),
        np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=float),
        density_maxk=100,
    )
    assert assigned.tolist() == [7]
    assert halo.tolist() == [8]


def test_predict_cluster_adp_falls_back_to_nearest_training_label_on_dadapy_buffer_error():
    class DummyDP:
        X = np.asarray([[0.1, 0.2], [2.0, 2.1], [4.0, 4.1]], dtype=float)
        cluster_assignment = np.asarray([3, 4, 5], dtype=np.int32)
        cluster_assignment_halo = np.asarray([13, 14, 15], dtype=np.int32)

        def predict_cluster_ADP(self, emb, *, maxk, density_est, n_jobs):
            raise ValueError("Buffer has wrong number of dimensions (expected 2, got 1)")

    assigned, halo = _predict_cluster_adp(DummyDP(), np.asarray([[2.05, 2.0]], dtype=float), density_maxk=100)
    assert assigned.tolist() == [4]
    assert halo.tolist() == [14]


def test_run_md_samples_refresh_job_filters_selected_states(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))

    store = ProjectStore(base_dir=data_root / "projects")
    store.create_project("Project", project_id="proj")
    system = store.create_system("proj", name="System", system_id="sys")
    system.states["state1"] = DescriptorState(
        state_id="state1",
        name="State 1",
        descriptor_file="states/state1/descriptors.npz",
        storage_key="state1",
    )
    system.states["state2"] = DescriptorState(
        state_id="state2",
        name="State 2",
        descriptor_file="states/state2/descriptors.npz",
        storage_key="state2",
    )
    system.metastable_clusters = [{"cluster_id": "cluster1", "samples": []}]
    store.save_system(system)
    store.ensure_cluster_directories("proj", "sys", "cluster1")
    monkeypatch.setattr("backend.tasks.project_store", store)

    seen = []

    def _fake_evaluate_state_with_models(
        project_id,
        system_id,
        cluster_id,
        state_id,
        store=None,
        sample_id=None,
        workers=1,
        progress_callback=None,
    ):
        seen.append(state_id)
        return {
            "sample_id": sample_id or f"sample-{state_id}",
            "name": state_id,
            "type": "md_eval",
            "method": "md_eval",
            "state_id": state_id,
            "paths": {"summary_npz": f"clusters/{cluster_id}/samples/{sample_id or f'sample-{state_id}'}/sample.npz"},
        }

    monkeypatch.setattr("backend.tasks.evaluate_state_with_models", _fake_evaluate_state_with_models)

    out = run_md_samples_refresh_job(
        "job-1",
        {"project_id": "proj", "system_id": "sys", "cluster_id": "cluster1"},
        {"state_ids": ["state2"], "overwrite": True, "cleanup": True},
    )

    assert seen == ["state2"]
    assert out["status"] == "finished"
    assert out["results"]["states"] == 1
    assert out["results"]["requested_state_ids"] == ["state2"]
