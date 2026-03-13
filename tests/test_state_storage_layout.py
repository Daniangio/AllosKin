import json
import sys
import types
from pathlib import Path

import numpy as np

from phase.services.project_store import DescriptorState, ProjectStore

fake_vamp_pipeline = types.ModuleType("phase.analysis.vamp_pipeline")
fake_vamp_pipeline.run_metastable_pipeline_for_system = lambda *args, **kwargs: {}
sys.modules.setdefault("phase.analysis.vamp_pipeline", fake_vamp_pipeline)

import phase.workflows.metastable as metastable_workflow


def test_project_store_uses_per_state_metadata_files(tmp_path: Path):
    store = ProjectStore(tmp_path / 'projects')
    store.create_project('Proj', project_id='proj')
    system = store.create_system('proj', name='System', system_id='sys')

    system.states['state-1'] = DescriptorState(
        state_id='state-1',
        name='Active',
        storage_key='active',
        pdb_file='states/active/structure.pdb',
        trajectory_file='states/active/trajectory.xtc',
        descriptor_file='states/active/descriptors.npz',
        descriptor_metadata_file='states/active/descriptor_metadata.json',
    )
    store.save_system(system)

    system_dir = tmp_path / 'projects' / 'proj' / 'systems' / 'sys'
    assert (system_dir / 'states' / 'active' / 'state_metadata.json').exists()
    assert not (system_dir / 'states_metadata.json').exists()
    assert not (system_dir / 'metastable_metadata.json').exists()

    restored = store.get_system('proj', 'sys')
    assert restored.states['state-1'].name == 'Active'
    assert restored.states['state-1'].storage_key == 'active'
    assert restored.states['state-1'].pdb_file == 'states/active/structure.pdb'


def test_metastable_recompute_is_scoped_to_one_state_and_saved_under_state_folder(tmp_path: Path, monkeypatch):
    store = ProjectStore(tmp_path / 'projects')
    store.create_project('Proj', project_id='proj')
    system = store.create_system('proj', name='System', system_id='sys')

    state_dir = tmp_path / 'projects' / 'proj' / 'systems' / 'sys' / 'states' / 'inactive'
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / 'structure.pdb').write_text('MODEL\nENDMDL\n', encoding='utf-8')
    np.savez_compressed(state_dir / 'descriptors.npz', res_1=np.zeros((2, 1, 3), dtype=float))

    system.states['state-1'] = DescriptorState(
        state_id='state-1',
        name='Inactive',
        storage_key='inactive',
        pdb_file='states/inactive/structure.pdb',
        descriptor_file='states/inactive/descriptors.npz',
    )
    store.save_system(system)

    def fake_run(specs, *, output_dir, **_kwargs):
        spec = specs[0]
        label_path = Path(spec['descriptor_path']).with_suffix('.meta_labels.npy')
        np.save(label_path, np.asarray([0, 1], dtype=np.int32))
        rep_dir = Path(output_dir) / spec['macro_state']
        rep_dir.mkdir(parents=True, exist_ok=True)
        rep_path = rep_dir / 'rep.pdb'
        rep_path.write_text('MODEL\nENDMDL\n', encoding='utf-8')
        return {
            'macro_results': [
                {
                    'labels_per_trajectory': {spec['trajectory_id']: str(label_path)},
                    'metastable_states': [
                        {
                            'metastable_id': f"{spec['macro_state']}__m0",
                            'metastable_index': 0,
                            'default_name': f"{spec['macro_state']} m1",
                            'representative_pdb': str(rep_path),
                            'n_frames': 2,
                        }
                    ],
                }
            ]
        }

    monkeypatch.setattr(metastable_workflow, 'ProjectStore', lambda: store)
    monkeypatch.setattr(metastable_workflow, 'run_metastable_pipeline_for_system', fake_run)

    result = metastable_workflow.recompute_metastable_states(
        'proj',
        'sys',
        state_id='state-1',
        n_microstates=8,
        k_meta_min=1,
        k_meta_max=2,
        tica_lag_frames=4,
        tica_dim=3,
        random_state=7,
    )

    assert result['state_id'] == 'state-1'
    meta_path = state_dir / 'metastable' / 'metastable_metadata.json'
    assert meta_path.exists()
    saved = json.loads(meta_path.read_text(encoding='utf-8'))
    assert saved['state_id'] == 'state-1'
    assert saved['params']['n_microstates'] == 8
    assert saved['metastable_states'][0]['macro_state_id'] == 'state-1'

    restored = store.get_system('proj', 'sys')
    assert len(restored.metastable_states) == 1
    assert restored.metastable_states[0]['macro_state_id'] == 'state-1'
    assert restored.states['state-1'].metastable_metadata_file == 'states/inactive/metastable/metastable_metadata.json'
    assert restored.states['state-1'].metastable_labels_file == 'states/inactive/descriptors.meta_labels.npy'
