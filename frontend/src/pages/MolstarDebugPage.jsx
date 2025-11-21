import { useEffect, useRef, useState } from 'react';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui/index';
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18';
import { Asset } from 'molstar/lib/mol-util/assets';
import 'molstar/build/viewer/molstar.css';

export default function MolstarDebugPage() {
  const containerRef = useRef(null);
  const pluginRef = useRef(null);
  const [status, setStatus] = useState('initializing');
  const [error, setError] = useState(null);
  const [activeFile, setActiveFile] = useState(null);
  const [inactiveFile, setInactiveFile] = useState(null);
  const [trajectoryFile, setTrajectoryFile] = useState(null);

  useEffect(() => {
    let disposed = false;
    const init = async () => {
      if (!containerRef.current) return;
      try {
        const plugin = await createPluginUI({
          target: containerRef.current,
          render: renderReact18,
        });
        if (disposed) {
          plugin.dispose?.();
          return;
        }
        pluginRef.current = plugin;
        setStatus('ready');
      } catch (err) {
        console.error('Mol* init failed', err);
        setError(err.message || 'Failed to initialize Mol* plugin.');
        setStatus('error');
      }
    };
    init();
    return () => {
      disposed = true;
      if (pluginRef.current) {
        try {
          pluginRef.current.dispose?.();
        } catch (disposeErr) {
          console.warn('Mol* dispose failed', disposeErr);
        }
        pluginRef.current = null;
      }
    };
  }, []);

  const inferFormatFromName = (name = '') => {
    const lower = name.toLowerCase();
    if (lower.endsWith('.cif') || lower.endsWith('.mmcif')) return 'mmcif';
    if (lower.endsWith('.bcif')) return 'bcif';
    if (lower.endsWith('.ent') || lower.endsWith('.pdb')) return 'pdb';
    if (lower.endsWith('.gro')) return 'gro';
    if (lower.endsWith('.mol')) return 'mol';
    if (lower.endsWith('.sdf')) return 'sdf';
    if (lower.endsWith('.xyz')) return 'xyz';
    if (lower.endsWith('.dcd')) return 'dcd';
    if (lower.endsWith('.xtc')) return 'xtc';
    return 'pdb';
  };

  const loadTrajectoryBytes = async (byteArray, label, format) => {
    if (!pluginRef.current) return;
    const plugin = pluginRef.current;
    await plugin.dataTransaction(async () => {
      const data = await plugin.builders.data.rawData({ data: byteArray, label });
      const trajectory = await plugin.builders.structure.parseTrajectory(data, format);
      await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
    });
  };

  const loadStructureFromFile = async (file) => {
    if (!pluginRef.current || !file) return;
    const label = file.name || 'uploaded';
    const format = inferFormatFromName(label);
    setStatus('loading-structure');
    setError(null);
    try {
      const buffer = await file.arrayBuffer();
      await loadTrajectoryBytes(buffer, label, format);
      setStatus('ready');
    } catch (err) {
      console.error('Mol* file load failed', err);
      setError(err.message || 'Failed to load uploaded structure.');
      setStatus('ready');
    }
  };

  const loadExampleStructure = async (type) => {
    if (!pluginRef.current) return;
    const isActive = type === 'active';
    const url = isActive
      ? 'https://files.rcsb.org/download/1CRN.pdb'
      : 'https://files.rcsb.org/download/4HHB.pdb';
    setStatus('loading-structure');
    setError(null);
    try {
      await pluginRef.current.dataTransaction(async () => {
        const data = await pluginRef.current.builders.data.download(
          { url: Asset.Url(url), isBinary: false },
          { state: { isGhost: true } }
        );
        const trajectory = await pluginRef.current.builders.structure.parseTrajectory(data, 'pdb');
        await pluginRef.current.builders.structure.hierarchy.applyPreset(trajectory, 'default');
      });
      setStatus('ready');
    } catch (err) {
      console.error('Mol* structure load failed', err);
      setError(err.message || 'Failed to load example structure.');
      setStatus('ready');
    }
  };

  const loadUploadedWithTrajectory = async (structureFile) => {
    if (!structureFile) return;
    await loadStructureFromFile(structureFile);
    if (!trajectoryFile) return;
    // Try to load the trajectory as a follow-up step; failures here are non-fatal.
    try {
      const buffer = await trajectoryFile.arrayBuffer();
      await loadTrajectoryBytes(buffer, trajectoryFile.name || 'trajectory', inferFormatFromName(trajectoryFile.name));
    } catch (err) {
      console.warn('Trajectory load failed', err);
      setError((prev) => prev || 'Trajectory could not be loaded (check format).');
      setStatus('ready');
    }
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-white">Mol* Debug Viewer</h1>
      <p className="text-sm text-gray-400">
        Use this page to ensure Mol* loads correctly in your browser. Click &ldquo;Load Example Structure&rdquo; to fetch
        the 1CRN PDB from RCSB and render it directly.
      </p>
      {error && <ErrorMessage message={error} />}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-2">
        {status === 'initializing' && <Loader message="Initializing Mol* plugin..." />}
        <div
          ref={containerRef}
          className="w-full h-[500px] rounded-lg bg-black overflow-hidden border border-gray-700 relative"
        />
      </div>
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
        <h2 className="text-lg font-semibold text-white">Quick example loads</h2>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => loadExampleStructure('active')}
            disabled={status !== 'ready'}
            className="px-4 py-2 rounded-md bg-cyan-600 text-white text-sm disabled:opacity-50"
          >
            {status === 'loading-structure' ? 'Loading...' : 'Load Active (1CRN)'}
          </button>
          <button
            type="button"
            onClick={() => loadExampleStructure('inactive')}
            disabled={status !== 'ready'}
            className="px-4 py-2 rounded-md bg-purple-600 text-white text-sm disabled:opacity-50"
          >
            {status === 'loading-structure' ? 'Loading...' : 'Load Inactive (4HHB)'}
          </button>
        </div>
      </div>

      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
        <h2 className="text-lg font-semibold text-white">Load your own files</h2>
        <p className="text-sm text-gray-400">
          Upload an active or inactive structure (PDB/mmCIF and similar). Optionally add a trajectory file (e.g. DCD or
          XTC) to load right after the structure.
        </p>
        <div className="grid gap-3 md:grid-cols-2">
          <div className="space-y-2">
            <label className="block text-sm text-gray-300">Active structure</label>
            <input
              type="file"
              accept=".pdb,.ent,.cif,.mmcif,.bcif,.gro,.mol,.sdf,.xyz,.dcd,.xtc"
              onChange={(e) => setActiveFile(e.target.files?.[0] || null)}
              className="block w-full text-sm text-gray-300 bg-gray-900 border border-gray-700 rounded-md p-2"
            />
            <button
              type="button"
              onClick={() => loadUploadedWithTrajectory(activeFile)}
              disabled={status !== 'ready' || !activeFile}
              className="px-3 py-2 rounded-md bg-cyan-600 text-white text-sm disabled:opacity-50"
            >
              Load Active
            </button>
          </div>
          <div className="space-y-2">
            <label className="block text-sm text-gray-300">Inactive structure</label>
            <input
              type="file"
              accept=".pdb,.ent,.cif,.mmcif,.bcif,.gro,.mol,.sdf,.xyz,.dcd,.xtc"
              onChange={(e) => setInactiveFile(e.target.files?.[0] || null)}
              className="block w-full text-sm text-gray-300 bg-gray-900 border border-gray-700 rounded-md p-2"
            />
            <button
              type="button"
              onClick={() => loadUploadedWithTrajectory(inactiveFile)}
              disabled={status !== 'ready' || !inactiveFile}
              className="px-3 py-2 rounded-md bg-purple-600 text-white text-sm disabled:opacity-50"
            >
              Load Inactive
            </button>
          </div>
        </div>
        <div className="space-y-2">
          <label className="block text-sm text-gray-300">Trajectory (optional, loaded after structure)</label>
          <input
            type="file"
            accept=".dcd,.xtc,.pdb,.cif,.mmcif,.bcif"
            onChange={(e) => setTrajectoryFile(e.target.files?.[0] || null)}
            className="block w-full text-sm text-gray-300 bg-gray-900 border border-gray-700 rounded-md p-2"
          />
          <p className="text-xs text-gray-500">
            Loaded immediately after the selected structure. If it fails, the structure will stay visible.
          </p>
        </div>
      </div>
    </div>
  );
}
