import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("PHASE_DATA_ROOT", "/tmp/phase-test-data")

import backend.tasks as tasks
from phase.services.project_store import ProjectStore
import phase.scripts.potts_fit as potts_fit_script


def test_run_potts_fit_job_uses_selected_md_samples_as_training_dataset(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "results" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    potts_models_dir = tmp_path / "clusters" / "cluster-1" / "potts_models"
    potts_models_dir.mkdir(parents=True, exist_ok=True)
    cluster_path = tmp_path / "clusters" / "cluster-1" / "cluster.npz"
    cluster_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cluster_path,
        residue_keys=np.asarray(["res_1", "res_2"], dtype=str),
        cluster_counts=np.asarray([3, 4], dtype=np.int32),
        merged__labels_assigned=np.asarray([[0, 1]], dtype=np.int32),
        metadata_json=np.asarray(json.dumps({"cluster_name": "cluster-1"}), dtype=str),
    )

    sample_a_path = tmp_path / "clusters" / "cluster-1" / "samples" / "sample-a" / "sample.npz"
    sample_a_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sample_a_path,
        labels=np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        frame_state_ids=np.asarray(["state-a", "state-a"], dtype=str),
        frame_indices=np.asarray([10, 11], dtype=np.int64),
    )

    sample_b_path = tmp_path / "clusters" / "cluster-1" / "samples" / "sample-b" / "sample.npz"
    sample_b_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sample_b_path,
        labels=np.asarray([[2, 3]], dtype=np.int32),
        frame_indices=np.asarray([20], dtype=np.int64),
    )

    pdb_old = tmp_path / "old.pdb"
    pdb_a = tmp_path / "a.pdb"
    pdb_b = tmp_path / "b.pdb"
    pdb_old.write_text("MODEL\nENDMDL\n", encoding="utf-8")
    pdb_a.write_text("MODEL\nENDMDL\n", encoding="utf-8")
    pdb_b.write_text("MODEL\nENDMDL\n", encoding="utf-8")

    cluster_entry = {
        "cluster_id": "cluster-1",
        "name": "cluster-1",
        "path": str(cluster_path),
        "state_ids": ["old-state"],
        "samples": [
            {
                "sample_id": "sample-a",
                "name": "MD A",
                "type": "md_eval",
                "state_id": "state-a",
                "path": str(sample_a_path),
            },
            {
                "sample_id": "sample-b",
                "name": "MD B",
                "type": "md_eval",
                "state_id": "state-b",
                "path": str(sample_b_path),
            },
        ],
        "potts_models": [],
    }
    system_meta = SimpleNamespace(
        name="A2A",
        analysis_mode=None,
        metastable_clusters=[cluster_entry],
        metastable_states=[],
        states={
            "old-state": SimpleNamespace(pdb_file=str(pdb_old)),
            "state-a": SimpleNamespace(pdb_file=str(pdb_a)),
            "state-b": SimpleNamespace(pdb_file=str(pdb_b)),
        },
    )

    captured_args = {}

    def fake_parse_simulation_args(args_list):
        captured_args["args"] = list(args_list)
        return SimpleNamespace()

    def fake_run_pipeline(*_args, **_kwargs):
        args_list = captured_args["args"]
        model_out = Path(args_list[args_list.index("--model-out") + 1])
        result_dir = Path(args_list[args_list.index("--results-dir") + 1])
        result_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = result_dir / "model_metadata.json"
        metadata_path.write_text(json.dumps({"data_npz": str(cluster_path)}), encoding="utf-8")
        return {"model_path": str(model_out), "metadata_path": None}

    monkeypatch.setattr(tasks, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(tasks, "get_current_job", lambda: None)
    monkeypatch.setattr(tasks.project_store, "ensure_results_directories", lambda *args, **kwargs: {"jobs_dir": jobs_dir})
    monkeypatch.setattr(tasks.project_store, "ensure_cluster_directories", lambda *args, **kwargs: {
        "system_dir": tmp_path,
        "cluster_dir": cluster_path.parent,
        "potts_models_dir": potts_models_dir,
    })
    monkeypatch.setattr(tasks.project_store, "get_system", lambda *args, **kwargs: system_meta)
    monkeypatch.setattr(tasks.project_store, "resolve_path", lambda *args, **kwargs: Path(args[-1]))
    monkeypatch.setattr(tasks, "parse_simulation_args", fake_parse_simulation_args)
    monkeypatch.setattr(tasks, "run_simulation_pipeline", fake_run_pipeline)
    monkeypatch.setattr(tasks, "_persist_potts_model", lambda *args, **kwargs: "clusters/cluster-1/potts_models/new-model/model.npz")
    monkeypatch.setattr(tasks, "_supports_sim_arg", lambda option: option == "--pdbs")

    tasks.run_potts_fit_job(
        "job-1",
        {
            "project_id": "proj",
            "system_id": "sys",
            "system_name": "A2A",
            "project_name": "Phase",
            "cluster_id": "cluster-1",
        },
        {
            "fit_mode": "standard",
            "fit_method": "plm",
            "model_name": "cluster-1 Potts Model",
            "sample_ids": ["sample-a", "sample-b"],
            "contact_atom_mode": "CA",
            "contact_cutoff": 9.5,
        },
    )

    args_list = captured_args["args"]
    fit_npz = Path(args_list[args_list.index("--npz") + 1])
    assert fit_npz.name == "fit_dataset.npz"
    assert fit_npz.exists()

    fit_data = np.load(fit_npz, allow_pickle=True)
    np.testing.assert_array_equal(
        fit_data["merged__labels_assigned"],
        np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        fit_data["merged__frame_state_ids"],
        np.asarray(["state-a", "state-a", "state-b"], dtype=str),
    )
    np.testing.assert_array_equal(
        fit_data["merged__frame_indices"],
        np.asarray([10, 11, 20], dtype=np.int64),
    )

    pdb_arg = args_list[args_list.index("--pdbs") + 1]
    assert str(pdb_a) in pdb_arg
    assert str(pdb_b) in pdb_arg
    assert str(pdb_old) not in pdb_arg

    results_dir = Path(args_list[args_list.index("--results-dir") + 1])
    saved_meta = json.loads((results_dir / "model_metadata.json").read_text(encoding="utf-8"))
    assert saved_meta["data_npz"].endswith("fit_dataset.npz")
    assert saved_meta["fit_dataset_type"] == "selected_md_samples"
    assert saved_meta["fit_sample_ids"] == ["sample-a", "sample-b"]


def test_potts_fit_resume_keeps_existing_model_name(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))

    store = ProjectStore(base_dir=data_root / "projects")
    store.create_project("Project", project_id="proj")
    system = store.create_system("proj", name="System", system_id="sys")
    system.metastable_clusters = [
        {
            "cluster_id": "cluster1",
            "potts_models": [
                {
                    "model_id": "model-1",
                    "name": "Original model name",
                    "path": "clusters/cluster1/potts_models/model-1/model.npz",
                }
            ],
        }
    ]
    store.save_system(system)

    cluster_dirs = store.ensure_cluster_directories("proj", "sys", "cluster1")
    cluster_npz = cluster_dirs["cluster_dir"] / "cluster.npz"
    np.savez_compressed(cluster_npz, merged__labels_assigned=np.asarray([[0]], dtype=np.int32))

    resume_dir = cluster_dirs["potts_models_dir"] / "model-1"
    resume_dir.mkdir(parents=True, exist_ok=True)
    resume_path = resume_dir / "model.npz"
    resume_path.write_bytes(b"fake")

    def fake_run_pipeline(args, parser=None):
        return {"model_path": str(resume_path)}

    monkeypatch.setattr(potts_fit_script.sim_main, "run_pipeline", fake_run_pipeline)

    rc = potts_fit_script.main(
        [
            "--project-id",
            "proj",
            "--system-id",
            "sys",
            "--cluster-id",
            "cluster1",
            "--npz",
            str(cluster_npz),
            "--results-dir",
            str(tmp_path / "results"),
            "--fit",
            "plm",
            "--plm-resume-model",
            "clusters/cluster1/potts_models/model-1/model.npz",
        ]
    )

    assert rc == 0
    meta = json.loads((resume_dir / "model_metadata.json").read_text(encoding="utf-8"))
    assert meta["name"] == "Original model name"
