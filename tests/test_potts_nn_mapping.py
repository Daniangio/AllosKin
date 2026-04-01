import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PHASE_DATA_ROOT", "/tmp/phase-test-data")

from phase.potts.orchestration import run_potts_nn_mapping_local
from phase.potts.potts_model import PottsModel, save_potts_model
from phase.potts.sample_io import save_sample_npz
from phase.services.project_store import DescriptorState, ProjectStore


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


def test_run_potts_nn_mapping_local_identity(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))
    store = ProjectStore(base_dir=data_root / "projects")
    store.create_project("Project", project_id="proj")
    store.create_system("proj", name="System", system_id="sys")

    system_dir = data_root / "projects" / "proj" / "systems" / "sys"
    cluster_id = "cluster1"
    cluster_dir = system_dir / "clusters" / cluster_id
    cluster_dir.mkdir(parents=True, exist_ok=True)

    labels = np.asarray([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=np.int32)
    _write_sample(system_dir, cluster_id, "md1", {"name": "MD 1", "type": "md_eval", "method": "md_eval"}, labels=labels)
    _write_sample(system_dir, cluster_id, "sa1", {"name": "SA 1", "type": "potts_sampling", "method": "gibbs"}, labels=labels)

    model_dir = cluster_dir / "potts_models" / "model1"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.npz"
    save_potts_model(
        PottsModel(
            h=[np.asarray([0.0, 1.0], dtype=float), np.asarray([0.0, 2.0], dtype=float)],
            J={(0, 1): np.asarray([[0.0, 0.3], [0.3, 0.0]], dtype=float)},
            edges=[(0, 1)],
        ),
        model_path,
        metadata={"model_name": "Model 1"},
    )

    out = run_potts_nn_mapping_local(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        model_path=str(model_path),
        sample_id="sa1",
        md_sample_id="md1",
        use_unique=True,
        normalize=True,
        compute_per_residue=True,
        beta_node=1.0,
        beta_edge=1.0,
        n_workers=2,
    )

    assert out["analysis_id"]
    with np.load(out["analysis_npz"], allow_pickle=False) as data:
        nn = np.asarray(data["nn_dist_global"], dtype=float)
        assert nn.shape == (2,)
        assert np.allclose(nn, 0.0)
        assert np.allclose(np.asarray(data["nn_dist_node"], dtype=float), 0.0)
        assert np.allclose(np.asarray(data["nn_dist_edge"], dtype=float), 0.0)
        assert np.asarray(data["sample_unique_counts"], dtype=np.int64).tolist() == [2, 2]
        assert np.asarray(data["analysis_format_version"], dtype=np.int32).tolist() == [2]
        assert np.asarray(data["residue_keys"], dtype=str).tolist() == ["res_1", "res_2"]
        assert np.allclose(np.asarray(data["per_edge_mean"], dtype=float), 0.0)
        assert np.allclose(np.asarray(data["per_edge_std"], dtype=float), 0.0)


def test_run_potts_nn_mapping_local_normalized_range_and_thresholds(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))
    store = ProjectStore(base_dir=data_root / "projects")
    store.create_project("Project", project_id="proj")
    store.create_system("proj", name="System", system_id="sys")

    system_dir = data_root / "projects" / "proj" / "systems" / "sys"
    cluster_id = "cluster1"
    cluster_dir = system_dir / "clusters" / cluster_id
    cluster_dir.mkdir(parents=True, exist_ok=True)

    md_labels = np.asarray([[0, 0], [1, 1], [0, 1]], dtype=np.int32)
    sample_labels = np.asarray([[1, 0], [1, 1], [0, 1], [1, 0]], dtype=np.int32)
    _write_sample(system_dir, cluster_id, "md1", {"name": "MD 1", "type": "md_eval", "method": "md_eval"}, labels=md_labels)
    _write_sample(system_dir, cluster_id, "sa1", {"name": "SA 1", "type": "potts_sampling", "method": "sa"}, labels=sample_labels)

    model_dir = cluster_dir / "potts_models" / "model1"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.npz"
    save_potts_model(
        PottsModel(
            h=[np.asarray([0.0, 2.0], dtype=float), np.asarray([0.0, 1.0], dtype=float)],
            J={(0, 1): np.asarray([[0.0, 0.5], [0.2, 1.0]], dtype=float)},
            edges=[(0, 1)],
        ),
        model_path,
    )

    out = run_potts_nn_mapping_local(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        model_path=str(model_path),
        sample_id="sa1",
        md_sample_id="md1",
        use_unique=True,
        normalize=True,
        compute_per_residue=True,
        beta_node=1.0,
        beta_edge=1.0,
        distance_thresholds=[0.1, 0.5, 1.0],
        n_workers=1,
    )

    with np.load(out["analysis_npz"], allow_pickle=False) as data:
        global_dist = np.asarray(data["nn_dist_global"], dtype=float)
        node = np.asarray(data["nn_dist_node"], dtype=float)
        edge = np.asarray(data["nn_dist_edge"], dtype=float)
        assert np.all(global_dist >= -1e-8)
        assert np.all(global_dist <= 1.0 + 1e-8)
        assert np.all(node >= -1e-8)
        assert np.all(node <= 1.0 + 1e-8)
        assert np.all(edge >= -1e-8)
        assert np.all(edge <= 1.0 + 1e-8)
        thresholds = np.asarray(data["threshold_values"], dtype=float)
        coverage = np.asarray(data["threshold_coverage"], dtype=float)
        assert thresholds.tolist() == [0.1, 0.5, 1.0]
        assert np.all(coverage >= 0.0)
        assert np.all(coverage <= 1.0)
        assert np.allclose(np.asarray(data["nn_dist_global"], dtype=float)[np.asarray(data["sample_unique_sequences"], dtype=np.int32).tolist().index([1, 1])], 0.0)
        for key in [
            "per_residue_std",
            "per_residue_median",
            "per_residue_q25",
            "per_residue_q75",
            "per_edge_mean",
            "per_edge_std",
            "per_edge_median",
            "per_edge_q25",
            "per_edge_q75",
        ]:
            assert key in data.files


def test_run_potts_nn_mapping_local_saves_residue_display_labels(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("PHASE_DATA_ROOT", str(data_root))
    store = ProjectStore(base_dir=data_root / "projects")
    store.create_project("Project", project_id="proj")
    system_meta = store.create_system("proj", name="System", system_id="sys")

    system_dir = data_root / "projects" / "proj" / "systems" / "sys"
    cluster_id = "cluster1"
    cluster_dir = system_dir / "clusters" / cluster_id
    cluster_dir.mkdir(parents=True, exist_ok=True)

    state_dir = system_dir / "states" / "state1"
    state_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = state_dir / "structure.pdb"
    pdb_path.write_text(
        "\n".join([
            "ATOM      1  N   ALA A  10      11.104  13.207   8.210  1.00 20.00           N",
            "ATOM      2  CA  ALA A  10      12.560  13.291   8.390  1.00 20.00           C",
            "ATOM      3  C   ALA A  10      13.061  14.727   8.749  1.00 20.00           C",
            "ATOM      4  N   GLY A  20      14.104  15.207   9.210  1.00 20.00           N",
            "ATOM      5  CA  GLY A  20      15.560  15.291   9.390  1.00 20.00           C",
            "ATOM      6  C   GLY A  20      16.061  16.727   9.749  1.00 20.00           C",
            "TER",
            "END",
        ]),
        encoding="utf-8",
    )
    system_meta.states = {
        "state1": DescriptorState(
            state_id="state1",
            name="State 1",
            pdb_file=str(pdb_path.relative_to(system_dir)),
            residue_mapping={"res_1": "protein and resid 10", "res_2": "protein and resid 20"},
        )
    }
    store.save_system(system_meta)

    np.savez_compressed(
        cluster_dir / "cluster.npz",
        residue_keys=np.asarray(["res_1", "res_2"], dtype=str),
        metadata_json=np.asarray(json.dumps({
            "selected_state_ids": ["state1"],
            "residue_mapping": {"res_1": "protein and resid 10", "res_2": "protein and resid 20"},
        })),
    )

    labels = np.asarray([[0, 0], [1, 1]], dtype=np.int32)
    _write_sample(system_dir, cluster_id, "md1", {"name": "MD 1", "type": "md_eval", "method": "md_eval"}, labels=labels)
    _write_sample(system_dir, cluster_id, "sa1", {"name": "SA 1", "type": "potts_sampling", "method": "gibbs"}, labels=labels)

    model_dir = cluster_dir / "potts_models" / "model1"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.npz"
    save_potts_model(
        PottsModel(
            h=[np.asarray([0.0, 1.0], dtype=float), np.asarray([0.0, 2.0], dtype=float)],
            J={(0, 1): np.asarray([[0.0, 0.3], [0.3, 0.0]], dtype=float)},
            edges=[(0, 1)],
        ),
        model_path,
    )

    out = run_potts_nn_mapping_local(
        project_id="proj",
        system_id="sys",
        cluster_id=cluster_id,
        model_path=str(model_path),
        sample_id="sa1",
        md_sample_id="md1",
        use_unique=True,
        normalize=True,
        compute_per_residue=True,
        beta_node=1.0,
        beta_edge=1.0,
        n_workers=1,
    )

    with np.load(out["analysis_npz"], allow_pickle=False) as data:
        assert np.asarray(data["residue_display_labels"], dtype=str).tolist() == ["ALA10", "GLY20"]
