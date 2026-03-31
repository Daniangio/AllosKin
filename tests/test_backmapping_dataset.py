import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PHASE_DATA_ROOT", "/tmp/phase-test-data")

import backend.tasks as tasks
import phase.workflows.backmapping as backmapping
from phase.services.project_store import DescriptorState, ProjectStore
from phase.workflows.backmapping import build_sample_backmapping_dataset
from phase.potts.sample_io import save_sample_npz


def _write_multimodel_pdb(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "MODEL        1",
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N",
                "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C",
                "ATOM      3  C   ALA A   1       1.958   1.410   0.000  1.00 20.00           C",
                "ATOM      4  CB  ALA A   1       1.958  -0.540  -1.240  1.00 20.00           C",
                "ATOM      5  N   GLY A   2       3.358   1.610   0.000  1.00 20.00           N",
                "ATOM      6  CA  GLY A   2       3.958   2.960   0.000  1.00 20.00           C",
                "ATOM      7  C   GLY A   2       3.258   3.960   0.950  1.00 20.00           C",
                "ATOM      8  O   GLY A   2       3.758   5.080   1.050  1.00 20.00           O",
                "TER",
                "ENDMDL",
                "MODEL        2",
                "ATOM      1  N   ALA A   1       0.200   0.100   0.000  1.00 20.00           N",
                "ATOM      2  CA  ALA A   1       1.658   0.050   0.100  1.00 20.00           C",
                "ATOM      3  C   ALA A   1       2.158   1.460   0.100  1.00 20.00           C",
                "ATOM      4  CB  ALA A   1       2.158  -0.490  -1.140  1.00 20.00           C",
                "ATOM      5  N   GLY A   2       3.558   1.760   0.050  1.00 20.00           N",
                "ATOM      6  CA  GLY A   2       4.158   3.060   0.100  1.00 20.00           C",
                "ATOM      7  C   GLY A   2       3.458   4.060   1.050  1.00 20.00           C",
                "ATOM      8  O   GLY A   2       3.958   5.180   1.150  1.00 20.00           O",
                "TER",
                "ENDMDL",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _prepare_sample_dataset(tmp_path: Path):
    data_root = tmp_path / "data"
    store = ProjectStore(base_dir=data_root / "projects")
    store.create_project("Project", project_id="proj")
    system = store.create_system("proj", name="System", system_id="sys")
    system.states["state1"] = DescriptorState(
        state_id="state1",
        name="State 1",
        pdb_file="states/state1/structure.pdb",
        storage_key="state1",
    )
    store.save_system(system)

    state_dir = data_root / "projects" / "proj" / "systems" / "sys" / "states" / "state1"
    state_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = state_dir / "structure.pdb"
    _write_multimodel_pdb(pdb_path)

    cluster_id = "cluster1"
    cluster_dir = data_root / "projects" / "proj" / "systems" / "sys" / "clusters" / cluster_id
    sample_dir = cluster_dir / "samples" / "sample1"
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cluster_dir / "cluster.npz",
        residue_keys=np.asarray(["res_1", "res_2"], dtype=str),
        cluster_counts=np.asarray([2, 3], dtype=np.int32),
    )
    sample_npz = sample_dir / "sample.npz"
    save_sample_npz(
        sample_npz,
        labels=np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        frame_state_ids=np.asarray(["state1", "state1"], dtype=str),
    )
    (sample_dir / "sample_metadata.json").write_text(
        json.dumps(
            {
                "sample_id": "sample1",
                "name": "MD state1",
                "type": "md_eval",
                "method": "md_eval",
                "state_id": "state1",
                "path": "clusters/cluster1/samples/sample1/sample.npz",
                "paths": {"summary_npz": "clusters/cluster1/samples/sample1/sample.npz"},
            }
        ),
        encoding="utf-8",
    )
    return data_root, store, pdb_path, cluster_id, sample_dir


def test_build_sample_backmapping_dataset_writes_expected_arrays(monkeypatch, tmp_path):
    data_root, _store, pdb_path, cluster_id, sample_dir = _prepare_sample_dataset(tmp_path)
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))
    monkeypatch.setattr(backmapping, "ProjectStore", lambda: ProjectStore(base_dir=data_root / "projects"))

    out_path = sample_dir / "backmapping_dataset.npz"
    summary = build_sample_backmapping_dataset(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        sample_id="sample1",
        trajectory_path=pdb_path,
        output_path=out_path,
    )

    assert summary["n_frames"] == 2
    assert summary["n_residues"] == 2
    with np.load(out_path, allow_pickle=False) as data:
        assert data["trajectory"].shape == (2, 8, 3)
        assert np.array_equal(data["atom_resids"], np.asarray([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32))
        assert data["residue_cluster_ids"].shape == (2, 2)
        assert np.array_equal(data["residue_cluster_counts"], np.asarray([2, 3], dtype=np.int32))
        assert data["dihedrals"].shape == (2, 2, 5)
        assert np.array_equal(data["dihedral_keys"].astype(str), np.asarray(["phi", "psi", "omega", "chi1", "chi2"]))


def test_build_sample_backmapping_dataset_accepts_legacy_truncated_frame_state_ids(monkeypatch, tmp_path):
    data_root, _store, pdb_path, cluster_id, sample_dir = _prepare_sample_dataset(tmp_path)
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))
    monkeypatch.setattr(backmapping, "ProjectStore", lambda: ProjectStore(base_dir=data_root / "projects"))

    with np.load(sample_dir / "sample.npz", allow_pickle=True) as src:
            np.savez_compressed(
                sample_dir / "sample.npz",
                labels=src["labels"],
                frame_indices=src["frame_indices"],
                frame_state_ids=np.asarray(["s", "s"], dtype="U1"),
            )

    out_path = sample_dir / "backmapping_dataset.npz"
    summary = build_sample_backmapping_dataset(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        sample_id="sample1",
        trajectory_path=pdb_path,
        output_path=out_path,
    )
    assert summary["n_frames"] == 2
    with np.load(out_path, allow_pickle=False) as data:
        assert np.all(data["frame_state_ids"].astype(str) == "state1")


def test_run_sample_backmapping_job_updates_sample_metadata(monkeypatch, tmp_path):
    data_root, store, pdb_path, cluster_id, sample_dir = _prepare_sample_dataset(tmp_path)
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))
    monkeypatch.setattr(tasks, "project_store", ProjectStore(base_dir=data_root / "projects"))
    upload_path = sample_dir / "upload.pdb"
    upload_path.write_text(pdb_path.read_text(encoding="utf-8"), encoding="utf-8")

    def fake_builder(**kwargs):
        out_path = kwargs["output_path"]
        np.savez_compressed(out_path, trajectory=np.zeros((1, 1, 3), dtype=np.float32))
        return {
            "path": str(out_path),
            "n_frames": 1,
            "n_atoms": 1,
            "n_residues": 2,
            "dihedral_keys": ["phi", "psi", "omega", "chi1", "chi2"],
            "sample_id": "sample1",
            "state_id": "state1",
        }

    monkeypatch.setattr(tasks, "build_sample_backmapping_dataset", fake_builder)

    out = tasks.run_sample_backmapping_job(
        "job-1",
        "proj",
        "sys",
        cluster_id,
        "sample1",
        str(upload_path),
    )

    assert out["status"] == "finished"
    sample_meta = store.get_sample_entry("proj", "sys", cluster_id, "sample1")
    dataset_meta = sample_meta["backmapping_dataset"]
    assert dataset_meta["status"] == "finished"
    assert dataset_meta["n_frames"] == 1
    assert dataset_meta["n_atoms"] == 1
    assert Path(data_root / "projects" / "proj" / "systems" / "sys" / dataset_meta["path"]).exists()
    assert not upload_path.exists()
