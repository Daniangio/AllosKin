import { useCallback, useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import Plot from 'react-plotly.js';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { fetchSystem, fetchStateDescriptors } from '../api/projects';

const colors = [
  '#22d3ee',
  '#a855f7',
  '#f97316',
  '#10b981',
  '#f43f5e',
  '#8b5cf6',
  '#06b6d4',
  '#fde047',
  '#60a5fa',
  '#f59e0b',
];

export default function DescriptorVizPage() {
  const { projectId, systemId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [error, setError] = useState(null);

  const [selectedStates, setSelectedStates] = useState([]);
  const [residueFilter, setResidueFilter] = useState('');
  const [selectedResidue, setSelectedResidue] = useState('');
  const [residueOptions, setResidueOptions] = useState([]);
  const [residueLabelCache, setResidueLabelCache] = useState({});

  const sortResidues = useCallback((keys) => {
    const unique = Array.from(new Set(keys || [])).filter((k) => k.startsWith('res_'));
    return unique.sort((a, b) => {
      const pa = parseInt((a.split('_')[1] || '').replace(/\D+/g, ''), 10);
      const pb = parseInt((b.split('_')[1] || '').replace(/\D+/g, ''), 10);
      if (Number.isFinite(pa) && Number.isFinite(pb) && pa !== pb) {
        return pa - pb;
      }
      return a.localeCompare(b);
    });
  }, []);
  const [maxPoints, setMaxPoints] = useState(2000);
  const [selectedMetastables, setSelectedMetastables] = useState([]);
  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [clusterMode, setClusterMode] = useState('merged');
  const [clusterLegend, setClusterLegend] = useState([]);

  const [anglesByState, setAnglesByState] = useState({});
  const [metaByState, setMetaByState] = useState({});
  const [loadingAngles, setLoadingAngles] = useState(false);
  const [anglesError, setAnglesError] = useState(null);

  useEffect(() => {
    const loadSystem = async () => {
      setLoadingSystem(true);
      setError(null);
      try {
        const data = await fetchSystem(projectId, systemId);
        setSystem(data);
        const descriptorStates = Object.values(data.states || {}).filter((s) => s.descriptor_file);
        if (descriptorStates.length) {
          setSelectedStates((prev) => (prev.length ? prev : descriptorStates.map((s) => s.state_id)));
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoadingSystem(false);
      }
    };
    loadSystem();
  }, [projectId, systemId]);

  const descriptorStates = useMemo(
    () => Object.values(system?.states || {}).filter((s) => s.descriptor_file),
    [system]
  );
  const metastableStates = useMemo(() => system?.metastable_states || [], [system]);
  const clusterOptions = useMemo(() => system?.metastable_clusters || [], [system]);

  // Hydrate from query params (cluster selection) whenever search changes.
  useEffect(() => {
    const params = new URLSearchParams(location.search || '');
    const clusterId = params.get('cluster_id');
    const mode = params.get('cluster_mode');
    if (clusterId) {
      setSelectedClusterId(clusterId);
    }
    if (mode && (mode === 'merged' || mode === 'per_meta')) {
      setClusterMode(mode);
    }
  }, [location.search]);

  // Default-select metastable states for initially selected macros (if user hasn't chosen yet)
  useEffect(() => {
    if (!selectedStates.length || selectedMetastables.length || !metastableStates.length) return;
    const metas = metastableStates
      .filter((m) => selectedStates.includes(m.macro_state_id))
      .map((m) => m.metastable_id);
    if (metas.length) {
      setSelectedMetastables(Array.from(new Set(metas)));
    }
  }, [metastableStates, selectedMetastables.length, selectedStates]);

  const residueKeys = useMemo(() => sortResidues(residueOptions), [residueOptions, sortResidues]);

  const residueLabel = useCallback(
    (key) => {
      if (residueLabelCache[key]) return residueLabelCache[key];
      const metasInOrder = [
        ...selectedStates.map((stateId) => metaByState[stateId]).filter(Boolean),
        ...Object.values(metaByState),
      ];

      for (const meta of metasInOrder) {
        const labels = meta?.residue_labels || {};
        if (labels[key]) return labels[key];
        const mapping = meta?.residue_mapping || {};
        if (mapping[key]) {
          const raw = mapping[key] || '';
          const match = raw.match(/\b([A-Z]{3})\b/);
          const resname = match ? match[1].toUpperCase() : null;
          if (resname) return `${key}_${resname}`;
        }
      }

      return key;
    },
    [metaByState, residueLabelCache, selectedStates]
  );

  const filteredResidues = useMemo(() => {
    if (!residueFilter.trim()) return residueKeys;
    const needle = residueFilter.toLowerCase();
    return residueKeys.filter((key) => {
      const label = residueLabel(key).toLowerCase();
      return label.includes(needle);
    });
  }, [residueFilter, residueKeys, residueLabel]);

  const stateName = useCallback(
    (stateId) => descriptorStates.find((s) => s.state_id === stateId)?.name || stateId,
    [descriptorStates]
  );

  const stateColors = useMemo(() => {
    const mapping = {};
    selectedStates.forEach((stateId, idx) => {
      mapping[stateId] = colors[idx % colors.length];
    });
    return mapping;
  }, [selectedStates]);

  const clusterColorMap = useMemo(() => {
    const mapping = {};
    clusterLegend.forEach((c, idx) => {
      mapping[c.id] = colors[idx % colors.length];
    });
    return mapping;
  }, [clusterLegend]);

  const clusterLabelLookup = useMemo(() => {
    const mapping = {};
    clusterLegend.forEach((c) => {
      mapping[c.id] = c.label;
    });
    return mapping;
  }, [clusterLegend]);

  const residueSymbols = useMemo(() => {
    const symbols = [
      'circle',
      'square',
      'diamond',
      'cross',
      'triangle-up',
      'triangle-down',
      'triangle-left',
      'triangle-right',
      'x',
      'star',
      'hexagram',
    ];
    const mapping = {};
    residueKeys.forEach((key, idx) => {
      mapping[key] = symbols[idx % symbols.length];
    });
    return mapping;
  }, [residueKeys]);

  const toggleState = (stateId) => {
    const isSelected = selectedStates.includes(stateId);
    const nextStates = isSelected
      ? selectedStates.filter((id) => id !== stateId)
      : [...selectedStates, stateId];
    setSelectedStates(nextStates);

    // Macro-level metastable toggle: select/deselect all metastables belonging to this macro
    const metasInMacro = metastableStates
      .filter((m) => m.macro_state_id === stateId)
      .map((m) => m.metastable_id);
    if (metasInMacro.length) {
      setSelectedMetastables((prev) => {
        if (isSelected) {
          // deselecting macro: drop its metastables
          return prev.filter((id) => !metasInMacro.includes(id));
        }
        const merged = new Set([...prev, ...metasInMacro]);
        return Array.from(merged);
      });
    }
  };

  const toggleMetastable = (metaId) => {
    setSelectedMetastables((prev) =>
      prev.includes(metaId) ? prev.filter((id) => id !== metaId) : [...prev, metaId]
    );
  };

  const selectResidue = (key) => {
    setSelectedResidue(key);
  };

  // Preload residue labels/resnames so the list keeps informative names even before a residue is loaded
  useEffect(() => {
    const bootstrapLabels = async () => {
      if (!selectedStates.length) return;
      try {
        const stateId = selectedStates[0];
        const data = await fetchStateDescriptors(projectId, systemId, stateId, { max_points: 1 });
        const labels = data.residue_labels || {};
        const mapping = data.residue_mapping || {};
        const combined = { ...labels };
        Object.entries(mapping).forEach(([k, raw]) => {
          if (combined[k]) return;
          const match = (raw || '').match(/\b([A-Z]{3})\b/);
          const resname = match ? match[1].toUpperCase() : null;
          if (resname) combined[k] = `${k}_${resname}`;
        });
        if (Object.keys(combined).length) {
          setResidueLabelCache((prev) => ({ ...prev, ...combined }));
        }
        if (Array.isArray(data.residue_keys)) {
          setResidueOptions((prev) => {
            const merged = new Set([...(prev || []), ...data.residue_keys]);
            return sortResidues(Array.from(merged));
          });
          if (!selectedResidue && data.residue_keys.length) {
            setSelectedResidue(sortResidues(data.residue_keys)[0]);
          }
        }
      } catch (err) {
        // keep silent; fallback labels will be used
      }
    };
    bootstrapLabels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, systemId, selectedStates, sortResidues]);

  const loadAngles = useCallback(async () => {
    if (!selectedStates.length) {
      setAnglesByState({});
      setMetaByState({});
      setClusterLegend([]);
      setSelectedResidue('');
      return;
    }
    setLoadingAngles(true);
    setAnglesError(null);
    try {
      const bootstrapOnly = !selectedResidue;
      const qs = { max_points: bootstrapOnly ? Math.min(maxPoints, 500) : maxPoints };
      if (selectedMetastables.length) {
        qs.metastable_ids = selectedMetastables;
      }
      if (selectedClusterId) {
        qs.cluster_id = selectedClusterId;
        qs.cluster_mode = clusterMode;
      }
      if (selectedResidue) {
        qs.residue_keys = selectedResidue;
      }

      const responses = await Promise.all(
        selectedStates.map(async (stateId) => {
          const data = await fetchStateDescriptors(projectId, systemId, stateId, qs);
          return { stateId, data };
        })
      );

      const newAngles = {};
      const newMeta = {};
      const unionResidues = new Set();

      responses.forEach(({ stateId, data }) => {
        newAngles[stateId] = data.angles || {};
        newMeta[stateId] = {
          residue_keys: data.residue_keys || [],
          residue_mapping: data.residue_mapping || {},
          residue_labels: data.residue_labels || {},
          n_frames: data.n_frames,
          sample_stride: data.sample_stride,
          cluster_legend: data.cluster_legend || [],
          cluster_mode: data.cluster_mode,
        };
        (data.residue_keys || []).forEach((key) => unionResidues.add(key));
        // Cache labels from this response to keep names informative in the list
        const labels = data.residue_labels || {};
        const mapping = data.residue_mapping || {};
        const combined = {};
        Object.entries(labels).forEach(([k, v]) => {
          if (v) combined[k] = v;
        });
        Object.entries(mapping).forEach(([k, raw]) => {
          if (combined[k]) return;
          const match = (raw || '').match(/\b([A-Z]{3})\b/);
          const resname = match ? match[1].toUpperCase() : null;
          if (resname) combined[k] = `${k}_${resname}`;
        });
        if (Object.keys(combined).length) {
          setResidueLabelCache((prev) => ({ ...prev, ...combined }));
        }
      });

      // Cluster legend: use from first response if present
      const firstLegend = responses.find((r) => (r.data.cluster_legend || []).length);
      setClusterLegend(firstLegend ? firstLegend.data.cluster_legend || [] : []);
      setMetaByState(newMeta);

      const sortedResidues = sortResidues(Array.from(unionResidues));
      setResidueOptions((prev) => {
        const merged = [...(prev || []), ...sortedResidues];
        return sortResidues(merged);
      });
      if (!selectedResidue && sortedResidues.length) {
        setSelectedResidue(sortedResidues[0]);
      } else if (selectedResidue && sortedResidues.length && !unionResidues.has(selectedResidue)) {
        setSelectedResidue(sortedResidues[0]);
      } else if (selectedResidue && !sortedResidues.length) {
        setSelectedResidue('');
      }

      if (!bootstrapOnly && selectedResidue) {
        setAnglesByState(newAngles);
      } else {
        // During bootstrap, only populate residue options; a follow-up call will load the selected residue
        setAnglesByState({});
      }
    } catch (err) {
      setAnglesError(err.message);
    } finally {
      setLoadingAngles(false);
    }
  }, [
    clusterMode,
    maxPoints,
    projectId,
    selectedClusterId,
    selectedMetastables,
    selectedResidue,
    selectedStates,
    systemId,
  ]);

  useEffect(() => {
    if (selectedStates.length) {
      loadAngles();
    } else {
      setAnglesByState({});
      setMetaByState({});
      setSelectedResidue('');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedStates, selectedMetastables, selectedClusterId, clusterMode, selectedResidue]);

  const traces3d = useMemo(() => {
    const traces = [];
    const residuesToPlot = selectedResidue ? [selectedResidue] : [];
    selectedStates.forEach((stateId) => {
      const perState = anglesByState[stateId] || {};
      residuesToPlot.forEach((key) => {
        const data = perState[key];
        if (!data) return;
        const clusterLabels = data.cluster_labels;
        const clusterColors =
          clusterLabels && clusterLegend.length
            ? clusterLabels.map((c) => clusterColorMap[c] || '#9ca3af')
            : null;
        const clusterHover =
          clusterLabels && clusterLabels.length
            ? clusterLabels.map((c) =>
                c >= 0 ? clusterLabelLookup[c] || `Cluster ${c}` : 'No cluster'
              )
            : null;
        traces.push({
          type: 'scatter3d',
          mode: 'markers',
          x: data.phi,
          y: data.psi,
          z: data.chi1,
          name: `${residueLabel(key)} — ${stateName(stateId)}`,
          legendgroup: residueLabel(key),
          marker: {
            size: 3,
            opacity: 0.75,
            color: clusterColors || stateColors[stateId],
            symbol: residueSymbols[key] || 'circle',
          },
          customdata: clusterHover,
          hovertemplate:
            `Residue: ${residueLabel(key)}<br>State: ${stateName(stateId)}` +
            '<br>Phi: %{x:.2f}°<br>Psi: %{y:.2f}°<br>Chi1: %{z:.2f}°' +
            (clusterHover ? '<br>Cluster: %{customdata}' : '') +
            '<extra></extra>',
        });
      });
    });
    // Add legend for clusters if present
    if (clusterLegend.length) {
      clusterLegend.forEach((c) => {
        traces.push({
          type: 'scatter3d',
          mode: 'markers',
          x: [null],
          y: [null],
          z: [null],
          name: c.label,
          showlegend: true,
          marker: { color: clusterColorMap[c.id] || '#9ca3af' },
          hoverinfo: 'none',
        });
      });
    }
    return traces;
  }, [
    anglesByState,
    clusterColorMap,
    clusterLabelLookup,
    clusterLegend,
    residueLabel,
    residueSymbols,
    selectedResidue,
    selectedStates,
    stateColors,
    stateName,
  ]);

  const make2DTraces = useCallback(
    (axisX, axisY) =>
      selectedStates
        .map((stateId) => {
          const perState = anglesByState[stateId] || {};
          const residuesToPlot = selectedResidue ? [selectedResidue] : [];
          return residuesToPlot.map((key) => {
            const data = perState[key];
            if (!data) return null;
            const clusterLabels = data.cluster_labels;
            const clusterColors =
              clusterLabels && clusterLegend.length
                ? clusterLabels.map((c) => clusterColorMap[c] || '#9ca3af')
                : null;
            const clusterHover =
              clusterLabels && clusterLabels.length
                ? clusterLabels.map((c) =>
                    c >= 0 ? clusterLabelLookup[c] || `Cluster ${c}` : 'No cluster'
                  )
                : null;
            return {
              type: 'scattergl',
              mode: 'markers',
              x: data[axisX],
              y: data[axisY],
              name: `${residueLabel(key)} — ${stateName(stateId)}`,
              legendgroup: residueLabel(key),
              marker: {
                size: 4,
                opacity: 0.7,
                color: clusterColors || stateColors[stateId],
                symbol: residueSymbols[key] || 'circle',
              },
              customdata: clusterHover,
              hovertemplate:
                `Residue: ${residueLabel(key)}<br>State: ${stateName(stateId)}` +
                `<br>${axisX.toUpperCase()}: %{x:.2f}°<br>${axisY.toUpperCase()}: %{y:.2f}°` +
                (clusterHover ? '<br>Cluster: %{customdata}' : '') +
                '<extra></extra>',
            };
          });
        })
        .flat()
        .filter(Boolean),
    [
      anglesByState,
      residueLabel,
      residueSymbols,
      selectedResidue,
      selectedStates,
      stateColors,
      stateName,
      clusterLegend,
      clusterColorMap,
      clusterLabelLookup,
    ]
  );

  const hasAngles = useMemo(
    () =>
      !!selectedResidue &&
      Object.values(anglesByState).some((residues) => Boolean((residues || {})[selectedResidue])),
    [anglesByState, selectedResidue]
  );

  const stateSummaries = useMemo(
    () =>
      selectedStates.map((stateId) => ({
        stateId,
        name: stateName(stateId),
        frames: metaByState[stateId]?.n_frames,
        stride: metaByState[stateId]?.sample_stride,
      })),
    [metaByState, selectedStates, stateName]
  );

  if (loadingSystem) return <Loader message="Loading system..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!system) return null;

  return (
    <div className="space-y-4">
      <button
        onClick={() => navigate(`/projects/${projectId}/systems/${systemId}`)}
        className="text-cyan-400 hover:text-cyan-300 text-sm"
      >
        ← Back to system
      </button>
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Descriptor Explorer</h1>
          <p className="text-sm text-gray-400">
            Visualize per-residue phi/psi/chi1 angles. Data are down-sampled for plotting.
          </p>
        </div>
        <div className="text-xs text-gray-400 text-right space-y-0.5">
          <div>
            States: {selectedStates.length ? stateSummaries.map((s) => s.name).join(', ') : '—'}
          </div>
          <div>
            Frames:{' '}
            {stateSummaries.length
              ? stateSummaries.map((s) => `${s.name}: ${s.frames ?? '—'}`).join(' • ')
              : '—'}
          </div>
          <div>
            Sample stride:{' '}
            {stateSummaries.length
              ? stateSummaries.map((s) => `${s.name}: ${s.stride ?? '—'}`).join(' • ')
              : '—'}
          </div>
        </div>
      </div>

      {descriptorStates.length === 0 ? (
        <ErrorMessage message="No descriptor-ready states. Upload trajectories and build descriptors first." />
      ) : (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
          <div className="grid md:grid-cols-4 gap-3">
            <div className="md:col-span-2">
              <label className="block text-xs text-gray-400 mb-1">States</label>
              <div className="grid sm:grid-cols-2 gap-2 max-h-28 overflow-y-auto border border-gray-700 rounded-md p-2 bg-gray-900">
                {descriptorStates.map((state) => (
                  <label key={state.state_id} className="flex items-center space-x-2 text-sm text-gray-200">
                    <input
                      type="checkbox"
                      checked={selectedStates.includes(state.state_id)}
                      onChange={() => toggleState(state.state_id)}
                      className="accent-cyan-500"
                    />
                    <span>{state.name}</span>
                  </label>
                ))}
              </div>
              <p className="text-[11px] text-gray-500 mt-1">
                Select one or more states to compare residue distributions.
              </p>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Max points per residue</label>
              <input
                type="number"
                min={10}
                max={50000}
                value={maxPoints}
                onChange={(e) => setMaxPoints(Number(e.target.value) || 2000)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              />
            </div>
            <div className="flex items-end">
              <button
                onClick={loadAngles}
                disabled={loadingAngles || !selectedStates.length}
                className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md disabled:opacity-50"
              >
                {loadingAngles ? 'Loading…' : selectedStates.length ? 'Refresh data' : 'Select states'}
              </button>
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-3">
            <div className="md:col-span-2 space-y-2">
              <label className="block text-xs text-gray-400 mb-1">Metastable states (filter)</label>
              {metastableStates.length === 0 ? (
                <p className="text-[11px] text-gray-500">No metastable states stored.</p>
              ) : (
                <div className="grid sm:grid-cols-2 gap-2 max-h-24 overflow-y-auto border border-gray-700 rounded-md p-2 bg-gray-900">
                  {metastableStates.map((m) => (
                    <label key={m.metastable_id} className="flex items-center space-x-2 text-sm text-gray-200">
                      <input
                        type="checkbox"
                        checked={selectedMetastables.includes(m.metastable_id)}
                        onChange={() => toggleMetastable(m.metastable_id)}
                        className="accent-emerald-400"
                      />
                      <span>{m.name || m.default_name || m.metastable_id}</span>
                    </label>
                  ))}
                </div>
              )}
              <p className="text-[11px] text-gray-500 mt-1">
                Filters are applied only on metastable states. Toggle states above to include their metastables, or pick individual ones here.
              </p>
            </div>
            <div className="space-y-2">
              <label className="block text-xs text-gray-400 mb-1">Cluster NPZ (optional coloring)</label>
              <select
                value={selectedClusterId}
                onChange={(e) => setSelectedClusterId(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              >
                <option value="">None</option>
                {clusterOptions.map((c) => (
                  <option key={c.cluster_id} value={c.cluster_id}>
                    {c.path?.split('/').pop() || c.cluster_id}
                  </option>
                ))}
              </select>
              {selectedClusterId && (
                <div className="flex items-center space-x-3 text-xs text-gray-300">
                  <label className="flex items-center space-x-1">
                    <input
                      type="radio"
                      name="cluster-mode"
                      value="merged"
                      checked={clusterMode === 'merged'}
                      onChange={() => setClusterMode('merged')}
                      className="accent-emerald-400"
                    />
                    <span>Merged</span>
                  </label>
                  <label className="flex items-center space-x-1">
                    <input
                      type="radio"
                      name="cluster-mode"
                      value="per_meta"
                      checked={clusterMode === 'per_meta'}
                      onChange={() => setClusterMode('per_meta')}
                      className="accent-emerald-400"
                    />
                    <span>Per metastable</span>
                  </label>
                </div>
              )}
              {clusterLegend.length > 0 && (
                <p className="text-[11px] text-gray-500">
                  Clusters loaded: {clusterLegend.map((c) => c.label).join(' • ')}
                </p>
              )}
            </div>
          </div>

          <div className="mt-2">
            <label className="block text-xs text-gray-400 mb-1">Choose residue (single selection)</label>
            <input
              type="text"
              value={residueFilter}
              onChange={(e) => setResidueFilter(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              placeholder="Search residue keys"
            />
          </div>

          <div className="grid sm:grid-cols-2 md:grid-cols-10 gap-2 max-h-48 overflow-y-auto border border-gray-700 rounded-md p-2 bg-gray-900">
            {filteredResidues.length === 0 && (
              <p className="text-sm text-gray-500 col-span-full">No residues match this filter.</p>
            )}
            {filteredResidues.map((key) => (
              <label key={key} className="flex items-center space-x-2 text-sm text-gray-200">
                <input
                  type="radio"
                  name="residue-select"
                  checked={selectedResidue === key}
                  onChange={() => selectResidue(key)}
                  className="accent-cyan-500"
                />
                <span>{residueLabel(key)}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      {anglesError && <ErrorMessage message={anglesError} />}
      {loadingAngles && <Loader message="Loading angles..." />}

      {!loadingAngles && !hasAngles && (
        <p className="text-sm text-gray-400">
          {selectedStates.length
            ? 'Pick a residue to load and color its angles.'
            : 'Select at least one state to load descriptor data.'}
        </p>
      )}

      {hasAngles && (
        <div className="space-y-4">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-3">
            <Plot
              data={traces3d}
              layout={{
                height: 500,
                paper_bgcolor: '#111827',
                plot_bgcolor: '#111827',
                font: { color: '#e5e7eb' },
                scene: {
                  xaxis: { title: 'Phi (°)', range: [-180, 180] },
                  yaxis: { title: 'Psi (°)', range: [-180, 180] },
                  zaxis: { title: 'Chi1 (°)', range: [-180, 180] },
                  aspectmode: 'cube',
                },
                margin: { l: 0, r: 0, t: 10, b: 0 },
                legend: { bgcolor: 'rgba(0,0,0,0)' },
              }}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
              config={{ displaylogo: false, responsive: true }}
            />
          </div>

          <div className="grid md:grid-cols-3 gap-3">
            {[
              { x: 'phi', y: 'psi', title: 'Phi vs Psi' },
              { x: 'phi', y: 'chi1', title: 'Phi vs Chi1' },
              { x: 'psi', y: 'chi1', title: 'Psi vs Chi1' },
            ].map((axes) => (
              <div key={axes.title} className="bg-gray-800 border border-gray-700 rounded-lg p-3">
                <Plot
                  data={make2DTraces(axes.x, axes.y)}
                  layout={{
                    height: 350,
                    paper_bgcolor: '#111827',
                    plot_bgcolor: '#111827',
                    font: { color: '#e5e7eb' },
                    margin: { l: 40, r: 10, t: 30, b: 40 },
                  xaxis: { title: `${axes.x.toUpperCase()} (°)`, range: [-180, 180] },
                  yaxis: { title: `${axes.y.toUpperCase()} (°)`, range: [-180, 180] },
                    legend: { bgcolor: 'rgba(0,0,0,0)' },
                  }}
                  useResizeHandler
                  style={{ width: '100%', height: '100%' }}
                  config={{ displaylogo: false, responsive: true }}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
