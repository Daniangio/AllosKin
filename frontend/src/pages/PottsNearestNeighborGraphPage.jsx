import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Plot from 'react-plotly.js';
import { CircleHelp, RefreshCw } from 'lucide-react';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import { fetchClusterAnalyses, fetchClusterAnalysisData, fetchSystem } from '../api/projects';

function clamp01(value) {
  const v = Number(value);
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

function mixHex(a, b, t) {
  const tt = clamp01(t);
  const parse = (hex) => {
    const clean = String(hex || '').replace('#', '');
    return [0, 2, 4].map((start) => parseInt(clean.slice(start, start + 2) || '00', 16));
  };
  const [ar, ag, ab] = parse(a);
  const [br, bg, bb] = parse(b);
  const out = [
    Math.round(ar + (br - ar) * tt),
    Math.round(ag + (bg - ag) * tt),
    Math.round(ab + (bb - ab) * tt),
  ];
  return `rgb(${out[0]}, ${out[1]}, ${out[2]})`;
}

function buildGroupKey(meta) {
  return JSON.stringify({
    model_id: meta?.model_id || null,
    model_name: meta?.model_name || null,
    normalize: !!meta?.normalize,
    beta_node: Number(meta?.beta_node ?? 1),
    beta_edge: Number(meta?.beta_edge ?? 1),
    md_label_mode: String(meta?.md_label_mode || 'assigned'),
    drop_invalid: !!meta?.drop_invalid,
    format: Number(meta?.analysis_format_version ?? meta?.summary?.analysis_format_version ?? 0),
  });
}

function parseGroupLabel(group) {
  if (!group?.meta) return 'Group';
  const meta = group.meta;
  return `${meta.model_name || meta.model_id || 'model'} · norm=${meta.normalize ? 'on' : 'off'} · βnode=${Number(meta.beta_node ?? 1).toFixed(2)} · βedge=${Number(meta.beta_edge ?? 1).toFixed(2)} · md=${meta.md_label_mode || 'assigned'}`;
}

function getArray(data, key) {
  return Array.isArray(data?.[key]) ? data[key].map(Number) : [];
}

function weightedMean(arrays, weights) {
  if (!arrays.length) return [];
  const width = arrays[0]?.length || 0;
  const out = new Array(width).fill(0);
  let total = 0;
  arrays.forEach((arr, idx) => {
    const w = Math.max(0, Number(weights[idx] ?? 0));
    if (!w) return;
    total += w;
    for (let i = 0; i < width; i += 1) out[i] += Number(arr[i] ?? 0) * w;
  });
  if (total <= 0) return out;
  return out.map((v) => v / total);
}

function aggregateStats(bundles, prefix) {
  if (!bundles.length) {
    return { mean: [], std: [], median: [], q25: [], q75: [] };
  }
  const weights = bundles.map((bundle) => Math.max(1, Number(bundle.weight || 1)));
  const meanArrays = bundles.map((bundle) => getArray(bundle.data, `${prefix}_mean`));
  const stdArrays = bundles.map((bundle) => getArray(bundle.data, `${prefix}_std`));
  const medianArrays = bundles.map((bundle) => getArray(bundle.data, `${prefix}_median`));
  const q25Arrays = bundles.map((bundle) => getArray(bundle.data, `${prefix}_q25`));
  const q75Arrays = bundles.map((bundle) => getArray(bundle.data, `${prefix}_q75`));
  const mean = weightedMean(meanArrays, weights);
  const secondMoment = weightedMean(
    meanArrays.map((arr, idx) => arr.map((value, i) => {
      const std = Number(stdArrays[idx]?.[i] ?? 0);
      return Number(value ?? 0) ** 2 + std ** 2;
    })),
    weights
  );
  const std = secondMoment.map((value, idx) => Math.sqrt(Math.max(0, Number(value) - Number(mean[idx] ?? 0) ** 2)));
  const median = weightedMean(medianArrays, weights);
  const q25 = weightedMean(q25Arrays, weights);
  const q75 = weightedMean(q75Arrays, weights);
  return { mean, std, median, q25, q75 };
}

function metricVector(stats, metric) {
  if (!stats) return [];
  if (metric === 'median') return stats.median || [];
  if (metric === 'std') return stats.std || [];
  if (metric === 'iqr') {
    const q25 = stats.q25 || [];
    const q75 = stats.q75 || [];
    return q75.map((value, idx) => Number(value) - Number(q25[idx] ?? 0));
  }
  return stats.mean || [];
}

function buildEdgeLabel(edge, residueKeys) {
  if (!Array.isArray(edge) || edge.length < 2) return 'edge';
  const r = Number(edge[0]);
  const s = Number(edge[1]);
  return `${residueKeys[r] || `res_${r + 1}`}–${residueKeys[s] || `res_${s + 1}`}`;
}

function combineStats(primary, secondary, alpha) {
  const a = Math.max(0, Math.min(1, Number(alpha ?? 0.75)));
  const blend = (left = [], right = []) => {
    const n = Math.max(left.length, right.length);
    return Array.from({ length: n }, (_, idx) => ((1 - a) * Number(left[idx] ?? 0)) + (a * Number(right[idx] ?? 0)));
  };
  const leftStd = primary?.std || [];
  const rightStd = secondary?.std || [];
  const std = Array.from({ length: Math.max(leftStd.length, rightStd.length) }, (_, idx) => {
    const ls = Number(leftStd[idx] ?? 0);
    const rs = Number(rightStd[idx] ?? 0);
    return Math.sqrt(Math.max(0, (((1 - a) ** 2) * (ls ** 2)) + ((a ** 2) * (rs ** 2))));
  });
  return {
    mean: blend(primary?.mean, secondary?.mean),
    median: blend(primary?.median, secondary?.median),
    q25: blend(primary?.q25, secondary?.q25),
    q75: blend(primary?.q75, secondary?.q75),
    std,
  };
}

function arrayFromSelect(event) {
  return Array.from(event.target.selectedOptions || []).map((opt) => opt.value);
}

export default function PottsNearestNeighborGraphPage() {
  const { projectId, systemId } = useParams();
  const navigate = useNavigate();
  const searchRef = useRef(new URLSearchParams(window.location.search || ''));

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [selectedClusterId, setSelectedClusterId] = useState(searchRef.current.get('cluster_id') || '');
  const [bundles, setBundles] = useState([]);
  const [loadingAnalyses, setLoadingAnalyses] = useState(false);
  const [analysesError, setAnalysesError] = useState(null);

  const [selectedGroupKey, setSelectedGroupKey] = useState('');
  const [selectedSourceIds, setSelectedSourceIds] = useState([]);
  const [selectedTargetIds, setSelectedTargetIds] = useState([]);
  const [viewMode, setViewMode] = useState('combined');
  const [metricMode, setMetricMode] = useState('mean');
  const [threshold, setThreshold] = useState(0.3);
  const [alpha, setAlpha] = useState(0.75);
  const [topEdges, setTopEdges] = useState(250);
  const [helpOpen, setHelpOpen] = useState(false);

  useEffect(() => {
    const loadSystem = async () => {
      setLoadingSystem(true);
      setSystemError(null);
      try {
        const data = await fetchSystem(projectId, systemId);
        setSystem(data);
      } catch (err) {
        setSystemError(err.message || 'Failed to load system.');
      } finally {
        setLoadingSystem(false);
      }
    };
    loadSystem();
  }, [projectId, systemId]);

  const clusters = useMemo(() => (system?.metastable_clusters || []).filter((entry) => entry?.cluster_id), [system]);

  useEffect(() => {
    if (!clusters.length) return;
    if (!selectedClusterId || !clusters.some((entry) => entry.cluster_id === selectedClusterId)) {
      setSelectedClusterId(clusters[0].cluster_id);
    }
  }, [clusters, selectedClusterId]);

  const loadBundles = useCallback(async () => {
    if (!selectedClusterId) return;
    setLoadingAnalyses(true);
    setAnalysesError(null);
    try {
      const payload = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'potts_nn_mapping' });
      const analyses = (Array.isArray(payload?.analyses) ? payload.analyses : []).filter((meta) => Number(meta?.analysis_format_version ?? meta?.summary?.analysis_format_version ?? 0) >= 2);
      const loaded = await Promise.all(
        analyses.map(async (meta) => {
          const detail = await fetchClusterAnalysisData(projectId, systemId, selectedClusterId, 'potts_nn_mapping', meta.analysis_id, { summaryOnly: true });
          const weight = Array.isArray(detail?.data?.sample_unique_counts)
            ? detail.data.sample_unique_counts.reduce((sum, value) => sum + Math.max(0, Number(value || 0)), 0)
            : Number(meta?.summary?.n_sample_frames || 1);
          return {
            analysisId: meta.analysis_id,
            meta: { ...meta, ...(detail?.metadata || {}) },
            data: detail?.data || {},
            weight,
          };
        })
      );
      setBundles(loaded);
    } catch (err) {
      setAnalysesError(err.message || 'Failed to load NN analyses.');
      setBundles([]);
    } finally {
      setLoadingAnalyses(false);
    }
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    loadBundles();
  }, [loadBundles]);

  const groups = useMemo(() => {
    const map = new Map();
    bundles.forEach((bundle) => {
      const key = buildGroupKey(bundle.meta);
      if (!map.has(key)) map.set(key, { key, meta: bundle.meta, bundles: [] });
      map.get(key).bundles.push(bundle);
    });
    return Array.from(map.values()).sort((a, b) => parseGroupLabel(a).localeCompare(parseGroupLabel(b)));
  }, [bundles]);

  useEffect(() => {
    if (!groups.length) {
      setSelectedGroupKey('');
      return;
    }
    if (!selectedGroupKey || !groups.some((group) => group.key === selectedGroupKey)) {
      setSelectedGroupKey(groups[0].key);
    }
  }, [groups, selectedGroupKey]);

  const selectedGroup = useMemo(() => groups.find((group) => group.key === selectedGroupKey) || null, [groups, selectedGroupKey]);
  const sourceOptions = useMemo(() => {
    const seen = new Map();
    (selectedGroup?.bundles || []).forEach((bundle) => {
      const id = String(bundle.meta?.sample_id || '');
      if (!id || seen.has(id)) return;
      seen.set(id, { id, name: bundle.meta?.sample_name || id });
    });
    return Array.from(seen.values()).sort((a, b) => String(a.name).localeCompare(String(b.name)));
  }, [selectedGroup]);
  const targetOptions = useMemo(() => {
    const seen = new Map();
    (selectedGroup?.bundles || []).forEach((bundle) => {
      const id = String(bundle.meta?.md_sample_id || '');
      if (!id || seen.has(id)) return;
      seen.set(id, { id, name: bundle.meta?.md_sample_name || id });
    });
    return Array.from(seen.values()).sort((a, b) => String(a.name).localeCompare(String(b.name)));
  }, [selectedGroup]);

  useEffect(() => {
    setSelectedSourceIds((prev) => prev.filter((id) => sourceOptions.some((opt) => opt.id === id)));
  }, [sourceOptions]);
  useEffect(() => {
    setSelectedTargetIds((prev) => prev.filter((id) => targetOptions.some((opt) => opt.id === id)));
  }, [targetOptions]);

  const filteredBundles = useMemo(() => {
    const sourceSet = new Set(selectedSourceIds);
    const targetSet = new Set(selectedTargetIds);
    return (selectedGroup?.bundles || []).filter((bundle) => {
      const sourceOk = !sourceSet.size || sourceSet.has(String(bundle.meta?.sample_id || ''));
      const targetOk = !targetSet.size || targetSet.has(String(bundle.meta?.md_sample_id || ''));
      return sourceOk && targetOk;
    });
  }, [selectedGroup, selectedSourceIds, selectedTargetIds]);

  const residueKeys = useMemo(() => {
    const first = filteredBundles[0]?.data || selectedGroup?.bundles?.[0]?.data || {};
    const keys = Array.isArray(first?.residue_keys) ? first.residue_keys.map(String) : [];
    if (keys.length) return keys;
    const fallbackN = Array.isArray(first?.per_residue_mean) ? first.per_residue_mean.length : 0;
    return Array.from({ length: fallbackN }, (_, idx) => `res_${idx + 1}`);
  }, [filteredBundles, selectedGroup]);

  const residueDisplayLabels = useMemo(() => {
    const first = filteredBundles[0]?.data || selectedGroup?.bundles?.[0]?.data || {};
    const labels = Array.isArray(first?.residue_display_labels) ? first.residue_display_labels.map(String) : [];
    if (labels.length === residueKeys.length) return labels;
    return residueKeys;
  }, [filteredBundles, selectedGroup, residueKeys]);

  const edges = useMemo(() => {
    const first = filteredBundles[0]?.data || selectedGroup?.bundles?.[0]?.data || {};
    return Array.isArray(first?.edges) ? first.edges.map((edge) => [Number(edge?.[0]), Number(edge?.[1])]) : [];
  }, [filteredBundles, selectedGroup]);

  const nodeOnlyStats = useMemo(() => aggregateStats(filteredBundles, 'per_residue_node'), [filteredBundles]);
  const nodeEdgeStats = useMemo(() => aggregateStats(filteredBundles, 'per_residue_edge'), [filteredBundles]);
  const edgeStats = useMemo(() => aggregateStats(filteredBundles, 'per_edge'), [filteredBundles]);
  const nodeCombinedStats = useMemo(() => combineStats(nodeOnlyStats, nodeEdgeStats, alpha), [nodeOnlyStats, nodeEdgeStats, alpha]);

  const nodeStats = useMemo(() => {
    if (viewMode === 'node') return nodeOnlyStats;
    if (viewMode === 'edge') return { mean: [], std: [], median: [], q25: [], q75: [] };
    return nodeCombinedStats;
  }, [viewMode, nodeOnlyStats, nodeCombinedStats]);

  const nodeValues = useMemo(() => metricVector(nodeStats, metricMode), [nodeStats, metricMode]);
  const edgeValues = useMemo(() => metricVector(edgeStats, metricMode), [edgeStats, metricMode]);

  const graphModel = useMemo(() => {
    const n = residueDisplayLabels.length;
    if (!n) return null;
    const center = 500;
    const radius = 390;
    const nodeOnlyMetric = metricVector(nodeOnlyStats, metricMode);
    const edgeResidMetric = metricVector(nodeEdgeStats, metricMode);
    const combinedMetric = metricVector(nodeCombinedStats, metricMode);
    const nodes = residueDisplayLabels.map((key, idx) => {
      const theta = (2 * Math.PI * idx) / Math.max(1, n);
      const x = center + radius * Math.cos(theta - Math.PI / 2);
      const y = center + radius * Math.sin(theta - Math.PI / 2);
      const value = Number(nodeValues[idx] ?? 0);
      const nodeOnly = Number(nodeOnlyMetric[idx] ?? 0);
      const edgeResid = Number(edgeResidMetric[idx] ?? 0);
      const combined = Number(combinedMetric[idx] ?? 0);
      const saturation = clamp01(threshold > 0 ? value / threshold : 1);
      return {
        idx,
        key,
        x,
        y,
        value,
        labelX: x + (x >= center ? 10 : -10),
        labelY: y + 3,
        labelAnchor: x >= center ? 'start' : 'end',
        color: viewMode === 'edge' ? 'rgb(75, 85, 99)' : mixHex('#374151', '#ef4444', saturation),
        radius: viewMode === 'edge' ? 5 : 6 + 6 * saturation,
        title: `${key}\nShown ${metricMode}: ${value.toFixed(4)}\nCombined: ${combined.toFixed(4)}\nNode-only: ${nodeOnly.toFixed(4)}\nEdge-attributed: ${edgeResid.toFixed(4)}`,
      };
    });
    const showEdges = viewMode === 'edge';
    const edgesSorted = edges
      .map((edge, idx) => {
        const value = Number(edgeValues[idx] ?? 0);
        const saturation = clamp01(threshold > 0 ? value / threshold : 1);
        return {
          idx,
          edge,
          value,
          saturation,
        };
      })
      .sort((a, b) => a.value - b.value)
      .slice(-Math.max(1, Number(topEdges) || 1));
    return {
      nodes,
      showEdges,
      edges: showEdges
        ? edgesSorted.map((entry) => {
            const [r, s] = entry.edge;
            const x1 = nodes[r]?.x ?? center;
            const y1 = nodes[r]?.y ?? center;
            const x2 = nodes[s]?.x ?? center;
            const y2 = nodes[s]?.y ?? center;
            return {
              ...entry,
              x1,
              y1,
              x2,
              y2,
              labelX: (x1 + x2) / 2,
              labelY: (y1 + y2) / 2,
              label: buildEdgeLabel(entry.edge, residueDisplayLabels),
              color: mixHex('#374151', '#22c55e', entry.saturation),
              width: 1 + 2 * entry.saturation,
              opacity: 0.2 + 0.75 * entry.saturation,
              title: `${buildEdgeLabel(entry.edge, residueDisplayLabels)}\n${metricMode}: ${entry.value.toFixed(4)}`,
            };
          })
        : [],
    };
  }, [residueDisplayLabels, nodeValues, metricMode, nodeOnlyStats, nodeEdgeStats, nodeCombinedStats, edges, edgeValues, threshold, viewMode, topEdges]);

  const residueRows = useMemo(() => {
    if (!residueDisplayLabels.length || viewMode === 'edge') return [];
    const values = nodeValues;
    const q25 = nodeStats?.q25 || [];
    const q75 = nodeStats?.q75 || [];
    const std = nodeStats?.std || [];
    return residueDisplayLabels
      .map((key, idx) => ({
        key,
        value: Number(values[idx] ?? 0),
        std: Number(std[idx] ?? 0),
        q25: Number(q25[idx] ?? 0),
        q75: Number(q75[idx] ?? 0),
        nodeOnly: Number(metricVector(nodeOnlyStats, metricMode)[idx] ?? 0),
        edgeAttributed: Number(metricVector(nodeEdgeStats, metricMode)[idx] ?? 0),
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 25);
  }, [residueDisplayLabels, viewMode, nodeValues, nodeStats, nodeOnlyStats, nodeEdgeStats, metricMode]);

  const residuePlot = useMemo(() => {
    if (!residueRows.length) return null;
    const rows = residueRows;
    const errorX = metricMode === 'mean'
      ? { type: 'data', array: rows.map((row) => row.std), visible: true }
      : (metricMode === 'median'
        ? { type: 'data', symmetric: false, array: rows.map((row) => Math.max(0, row.q75 - row.value)), arrayminus: rows.map((row) => Math.max(0, row.value - row.q25)), visible: true }
        : undefined);
    return {
      data: [
        {
          type: 'bar',
          x: rows.map((row) => row.value),
          y: rows.map((row) => row.key),
          orientation: 'h',
          marker: { color: '#38bdf8' },
          error_x: errorX,
          customdata: rows.map((row) => [row.nodeOnly, row.edgeAttributed]),
          hovertemplate: '%{y}<br>shown value=%{x:.4f}<br>node-only=%{customdata[0]:.4f}<br>edge-attributed=%{customdata[1]:.4f}<extra></extra>',
        },
      ],
      layout: {
        paper_bgcolor: '#111827',
        plot_bgcolor: '#111827',
        font: { color: '#e5e7eb' },
        margin: { t: 28, r: 16, b: 52, l: 140 },
        title: `Top residues by ${metricMode}` ,
        xaxis: { title: `${metricMode} mismatch` },
        yaxis: { title: 'Residue' },
      },
      config: { responsive: true, displaylogo: false },
    };
  }, [residueRows, metricMode]);

  const edgeRows = useMemo(() => {
    if (!edges.length || viewMode !== 'edge') return [];
    const q25 = edgeStats?.q25 || [];
    const q75 = edgeStats?.q75 || [];
    const std = edgeStats?.std || [];
    return edges
      .map((edge, idx) => ({
        label: buildEdgeLabel(edge, residueDisplayLabels),
        value: Number(edgeValues[idx] ?? 0),
        std: Number(std[idx] ?? 0),
        q25: Number(q25[idx] ?? 0),
        q75: Number(q75[idx] ?? 0),
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 25);
  }, [edges, viewMode, edgeStats, residueDisplayLabels, edgeValues]);

  const edgePlot = useMemo(() => {
    if (!edgeRows.length) return null;
    const rows = edgeRows;
    const errorX = metricMode === 'mean'
      ? { type: 'data', array: rows.map((row) => row.std), visible: true }
      : (metricMode === 'median'
        ? { type: 'data', symmetric: false, array: rows.map((row) => Math.max(0, row.q75 - row.value)), arrayminus: rows.map((row) => Math.max(0, row.value - row.q25)), visible: true }
        : undefined);
    return {
      data: [
        {
          type: 'bar',
          x: rows.map((row) => row.value),
          y: rows.map((row) => row.label),
          orientation: 'h',
          marker: { color: '#34d399' },
          error_x: errorX,
          hovertemplate: '%{y}<br>value=%{x:.4f}<extra></extra>',
        },
      ],
      layout: {
        paper_bgcolor: '#111827',
        plot_bgcolor: '#111827',
        font: { color: '#e5e7eb' },
        margin: { t: 28, r: 16, b: 52, l: 220 },
        title: `Top edges by ${metricMode}`,
        xaxis: { title: `${metricMode} mismatch` },
        yaxis: { title: 'Edge' },
      },
      config: { responsive: true, displaylogo: false },
    };
  }, [edgeRows, metricMode]);

  const summary = useMemo(() => ({
    analyses: filteredBundles.length,
    sources: new Set(filteredBundles.map((bundle) => String(bundle.meta?.sample_id || ''))).size,
    targets: new Set(filteredBundles.map((bundle) => String(bundle.meta?.md_sample_id || ''))).size,
    weight: filteredBundles.reduce((sum, bundle) => sum + Math.max(0, Number(bundle.weight || 0)), 0),
  }), [filteredBundles]);

  if (loadingSystem) return <Loader message="Loading Potts NN mismatch graph..." />;
  if (systemError) return <ErrorMessage message={systemError} />;

  return (
    <div className="space-y-4">
      <HelpDrawer
        open={helpOpen}
        title="Potts NN Mismatch Graph: Help"
        docPath="/docs/potts_nn_mapping_graph_help.md"
        onClose={() => setHelpOpen(false)}
      />

      <div className="flex items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-white">Potts NN Mismatch Graph</h1>
          <p className="text-sm text-gray-400">Aggregate mismatch over multiple NN analyses and visualize node, edge, and combined views with thresholded coloring.</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/potts_nn_mapping${selectedClusterId ? `?cluster_id=${encodeURIComponent(selectedClusterId)}` : ''}`)}
            className="inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:bg-gray-700/40"
          >
            NN analyses
          </button>
          <button
            type="button"
            onClick={loadBundles}
            className="inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:bg-gray-700/40"
          >
            <RefreshCw className="h-4 w-4" /> Refresh
          </button>
          <button
            type="button"
            onClick={() => setHelpOpen(true)}
            className="inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:bg-gray-700/40"
          >
            <CircleHelp className="h-4 w-4" /> Help
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[360px_minmax(0,1fr)] gap-4">
        <div className="space-y-4 rounded-lg border border-gray-800 bg-gray-900/60 p-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Cluster</label>
            <select value={selectedClusterId} onChange={(e) => setSelectedClusterId(e.target.value)} className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100">
              {clusters.map((cluster) => (
                <option key={cluster.cluster_id} value={cluster.cluster_id}>{cluster.name || cluster.cluster_id}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Compatible analysis group</label>
            <select value={selectedGroupKey} onChange={(e) => setSelectedGroupKey(e.target.value)} className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100">
              {groups.map((group) => (
                <option key={group.key} value={group.key}>{parseGroupLabel(group)}</option>
              ))}
            </select>
          </div>
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Source sampled trajectories</label>
              <button type="button" onClick={() => setSelectedSourceIds([])} className="text-[11px] text-cyan-300 hover:underline">All</button>
            </div>
            <select multiple value={selectedSourceIds} onChange={(e) => setSelectedSourceIds(arrayFromSelect(e))} className="w-full min-h-28 rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100">
              {sourceOptions.map((opt) => <option key={opt.id} value={opt.id}>{opt.name}</option>)}
            </select>
          </div>
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Target MD trajectories</label>
              <button type="button" onClick={() => setSelectedTargetIds([])} className="text-[11px] text-cyan-300 hover:underline">All</button>
            </div>
            <select multiple value={selectedTargetIds} onChange={(e) => setSelectedTargetIds(arrayFromSelect(e))} className="w-full min-h-28 rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100">
              {targetOptions.map((opt) => <option key={opt.id} value={opt.id}>{opt.name}</option>)}
            </select>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">View</label>
              <select value={viewMode} onChange={(e) => setViewMode(e.target.value)} className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100">
                <option value="combined">Combined</option>
                <option value="node">Node</option>
                <option value="edge">Edge</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Statistic</label>
              <select value={metricMode} onChange={(e) => setMetricMode(e.target.value)} className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100">
                <option value="mean">Mean</option>
                <option value="median">Median</option>
                <option value="std">Std</option>
                <option value="iqr">IQR</option>
              </select>
            </div>
          </div>
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Node-edge blend</label>
              <span className="text-xs text-gray-300">{Number(alpha).toFixed(2)}</span>
            </div>
            <input type="range" min="0" max="1" step="0.01" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} className="w-full" />
            <p className="mt-1 text-[11px] text-gray-500">Only used in Combined view: displayed node mismatch = (1-α)·node + α·edge-attributed, with edges hidden in the main graph.</p>
          </div>
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Color threshold</label>
              <span className="text-xs text-gray-300">{Number(threshold).toFixed(2)}</span>
            </div>
            <input type="range" min="0" max="1" step="0.01" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="w-full" />
            <p className="mt-1 text-[11px] text-gray-500">Max color is reached when the selected metric is greater than or equal to this threshold.</p>
          </div>
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Max edges shown</label>
              <span className="text-xs text-gray-300">{topEdges}</span>
            </div>
            <input type="range" min="25" max="1000" step="25" value={topEdges} onChange={(e) => setTopEdges(Number(e.target.value))} className="w-full" />
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="rounded-md border border-gray-800 bg-gray-950/70 p-3">
              <div className="text-xs text-gray-400">Analyses</div>
              <div className="mt-1 text-lg font-semibold text-white">{summary.analyses}</div>
            </div>
            <div className="rounded-md border border-gray-800 bg-gray-950/70 p-3">
              <div className="text-xs text-gray-400">Sample mass</div>
              <div className="mt-1 text-lg font-semibold text-white">{summary.weight}</div>
            </div>
            <div className="rounded-md border border-gray-800 bg-gray-950/70 p-3">
              <div className="text-xs text-gray-400">Sources</div>
              <div className="mt-1 text-lg font-semibold text-white">{summary.sources}</div>
            </div>
            <div className="rounded-md border border-gray-800 bg-gray-950/70 p-3">
              <div className="text-xs text-gray-400">Targets</div>
              <div className="mt-1 text-lg font-semibold text-white">{summary.targets}</div>
            </div>
          </div>
          {analysesError ? <ErrorMessage message={analysesError} /> : null}
        </div>

        <div className="space-y-4">
          {loadingAnalyses ? <Loader message="Loading NN analyses..." /> : null}
          {!loadingAnalyses && !filteredBundles.length ? (
            <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-6 text-sm text-gray-400">
              No compatible Potts NN mapping analyses matched the selected filters. Only new analyses with format version 2 are shown here.
            </div>
          ) : null}
          {!loadingAnalyses && filteredBundles.length ? (
            <>
              <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
                <div className="mb-3 flex items-center justify-between gap-2">
                  <div>
                    <h2 className="text-sm font-semibold text-white">Mismatch graph</h2>
                    <p className="text-xs text-gray-400">Red nodes encode residue-level mismatch. Green edges encode edge-level mismatch in Edge view only. Combined view hides edges and folds edge-attributed mismatch into node color using α.</p>
                  </div>
                </div>
                {graphModel ? (
                  <div className="overflow-auto">
                    <svg viewBox="0 0 1000 1000" className="w-full max-w-[980px] h-auto rounded-md border border-gray-800 bg-gray-950/80">
                      {graphModel.edges.map((edge) => (
                        <line
                          key={`edge-${edge.idx}`}
                          x1={edge.x1}
                          y1={edge.y1}
                          x2={edge.x2}
                          y2={edge.y2}
                          stroke={edge.color}
                          strokeWidth={edge.width}
                          opacity={edge.opacity}
                        >
                          <title>{edge.title}</title>
                        </line>
                      ))}
                      {graphModel.showEdges && graphModel.edges.length <= 40 ? graphModel.edges.map((edge) => (
                        <text key={`edge-label-${edge.idx}`} x={edge.labelX} y={edge.labelY} textAnchor="middle" fontSize="10" fill="rgba(226,232,240,0.75)">
                          {edge.label}
                        </text>
                      )) : null}
                      {graphModel.nodes.map((node) => (
                        <g key={`node-${node.idx}`}>
                          <circle cx={node.x} cy={node.y} r={node.radius} fill={node.color} stroke="rgba(17,24,39,0.95)" strokeWidth="1.5">
                            <title>{node.title}</title>
                          </circle>
                          <text x={node.labelX} y={node.labelY} textAnchor={node.labelAnchor} fontSize="11" fill="rgba(229,231,235,0.95)">
                            {node.key}
                          </text>
                        </g>
                      ))}
                    </svg>
                  </div>
                ) : null}
                <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-gray-300">
                  <div className="rounded-md border border-gray-800 bg-gray-950/60 p-3">
                    <div className="font-semibold text-white mb-1">Main graph meaning</div>
                    <div>{viewMode === 'edge' ? 'Green edges show edge mismatch magnitude. Gray nodes are only anchors for residue positions.' : viewMode === 'node' ? 'Red nodes show node-only mismatch.' : `Red nodes show combined mismatch with α=${Number(alpha).toFixed(2)}. Edges are hidden in this mode.`}</div>
                  </div>
                  <div className="rounded-md border border-gray-800 bg-gray-950/60 p-3">
                    <div className="font-semibold text-white mb-1">Labels</div>
                    <div>Node labels are residue keys. Edge labels are shown on the graph when few edges are displayed, and always in the edge ranking plot below.</div>
                  </div>
                  <div className="rounded-md border border-gray-800 bg-gray-950/60 p-3">
                    <div className="font-semibold text-white mb-1">Color scale</div>
                    <div>Color saturates at the selected threshold. Values above threshold keep the max color.</div>
                  </div>
                </div>
              </div>

              {residuePlot ? (
                <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
                  <Plot data={residuePlot.data} layout={residuePlot.layout} config={residuePlot.config} style={{ width: '100%', height: 520 }} />
                </div>
              ) : null}

              {residueRows.length ? (
                <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
                  <h3 className="text-sm font-semibold text-white mb-3">Top residue rankings</h3>
                  <div className="space-y-2 max-h-72 overflow-auto pr-1">
                    {residueRows.slice(0, 20).map((row) => (
                      <div key={row.key} className="grid grid-cols-[minmax(0,1fr),90px,90px,90px] gap-3 text-xs text-gray-300">
                        <span className="text-white">{row.key}</span>
                        <span>shown {row.value.toFixed(3)}</span>
                        <span>node {row.nodeOnly.toFixed(3)}</span>
                        <span>edge {row.edgeAttributed.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {edgePlot ? (
                <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
                  <Plot data={edgePlot.data} layout={edgePlot.layout} config={edgePlot.config} style={{ width: '100%', height: 560 }} />
                </div>
              ) : null}

              {edgeRows.length ? (
                <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
                  <h3 className="text-sm font-semibold text-white mb-3">Top edge rankings</h3>
                  <div className="space-y-2 max-h-72 overflow-auto pr-1">
                    {edgeRows.slice(0, 20).map((row) => (
                      <div key={row.label} className="grid grid-cols-[minmax(0,1fr),90px] gap-3 text-xs text-gray-300">
                        <span className="text-white">{row.label}</span>
                        <span>{row.value.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
