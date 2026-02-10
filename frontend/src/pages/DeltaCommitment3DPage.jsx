import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { CircleHelp, Play, RefreshCw } from 'lucide-react';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui/index';
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18';
import { Asset } from 'molstar/lib/mol-util/assets';
import { MolScriptBuilder as MS } from 'molstar/lib/mol-script/language/builder';
import { Script } from 'molstar/lib/mol-script/script';
import { StructureSelection } from 'molstar/lib/mol-model/structure';
import { clearStructureOverpaint, setStructureOverpaint } from 'molstar/lib/mol-plugin-state/helpers/structure-overpaint';
import 'molstar/build/viewer/molstar.css';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import {
  fetchClusterAnalyses,
  fetchClusterAnalysisData,
  fetchPottsClusterInfo,
  fetchSystem,
} from '../api/projects';
import { fetchJobStatus, submitDeltaTransitionJob } from '../api/jobs';

function clamp01(x) {
  if (!Number.isFinite(x)) return 0;
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function rgbToHex(r, g, b) {
  const to = (v) => {
    const x = Math.max(0, Math.min(255, Math.round(v)));
    return x.toString(16).padStart(2, '0');
  };
  return `#${to(r)}${to(g)}${to(b)}`;
}

function hexToInt(colorHex) {
  if (!colorHex) return 0xffffff;
  const s = String(colorHex).trim();
  const hex = s.startsWith('#') ? s.slice(1) : s;
  const v = parseInt(hex, 16);
  return Number.isFinite(v) ? v : 0xffffff;
}

function commitmentColor(q) {
  // Diverging map: 0 -> blue, 0.5 -> white, 1 -> red
  const t = clamp01(q);
  if (t <= 0.5) {
    const u = t / 0.5;
    return rgbToHex(lerp(59, 255, u), lerp(130, 255, u), lerp(246, 255, u));
  }
  const u = (t - 0.5) / 0.5;
  return rgbToHex(lerp(255, 239, u), lerp(255, 68, u), lerp(255, 68, u));
}

function parseResidueId(label) {
  if (label == null) return null;
  const m = String(label).match(/-?\d+/);
  if (!m) return null;
  const v = Number(m[0]);
  return Number.isFinite(v) ? v : null;
}

export default function DeltaCommitment3DPage() {
  const { projectId, systemId } = useParams();
  const navigate = useNavigate();

  const containerRef = useRef(null);
  const pluginRef = useRef(null);
  // Single base component; we overpaint it instead of stacking duplicate cartoons (which causes z-fighting).
  // Store the component's state-tree ref so we can re-find the corresponding StructureComponentRef wrapper.
  const baseComponentRef = useRef(null); // string | null

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [clusterInfo, setClusterInfo] = useState(null);
  const [clusterInfoError, setClusterInfoError] = useState(null);
  const [loadedStructureStateId, setLoadedStructureStateId] = useState('');

  const [modelAId, setModelAId] = useState('');
  const [modelBId, setModelBId] = useState('');
  const [activeMdSampleId, setActiveMdSampleId] = useState('');
  const [inactiveMdSampleId, setInactiveMdSampleId] = useState('');
  const [pasMdSampleId, setPasMdSampleId] = useState('');

  const [mdLabelMode, setMdLabelMode] = useState('assigned');
  const [keepInvalid, setKeepInvalid] = useState(false);
  const dropInvalid = !keepInvalid;

  const [bandFraction, setBandFraction] = useState(0.1);
  const [topKResidues, setTopKResidues] = useState(20);
  const [topKEdges, setTopKEdges] = useState(30);

  const [analyses, setAnalyses] = useState([]);
  const [analysesError, setAnalysesError] = useState(null);

  const [analysisData, setAnalysisData] = useState(null);
  const [analysisDataError, setAnalysisDataError] = useState(null);
  const [analysisDataLoading, setAnalysisDataLoading] = useState(false);

  const [job, setJob] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [jobError, setJobError] = useState(null);

  const [helpOpen, setHelpOpen] = useState(false);
  const [viewerError, setViewerError] = useState(null);
  const [viewerStatus, setViewerStatus] = useState('initializing'); // initializing | ready | error
  const [structureLoading, setStructureLoading] = useState(false);

  const [commitmentRowIndex, setCommitmentRowIndex] = useState(0);
  // Residue-id mapping between cluster residues and the loaded PDB.
  // In practice, "label" (sequential) is the most robust across PDBs; "auth" depends on PDB numbering.
  const [residueIdMode, setResidueIdMode] = useState('auth'); // label | auth
  const [coloringDebug, setColoringDebug] = useState(null);

  useEffect(() => {
    const load = async () => {
      setLoadingSystem(true);
      setSystemError(null);
      try {
        const sys = await fetchSystem(projectId, systemId);
        setSystem(sys);
      } catch (err) {
        setSystemError(err.message || 'Failed to load system.');
      } finally {
        setLoadingSystem(false);
      }
    };
    load();
  }, [projectId, systemId]);

  const clusterOptions = useMemo(
    () => (system?.metastable_clusters || []).filter((run) => run.path && run.status !== 'failed'),
    [system]
  );
  const selectedCluster = useMemo(
    () => clusterOptions.find((c) => c.cluster_id === selectedClusterId) || null,
    [clusterOptions, selectedClusterId]
  );

  const sampleEntries = useMemo(() => selectedCluster?.samples || [], [selectedCluster]);
  const mdSamples = useMemo(() => sampleEntries.filter((s) => s.type === 'md_eval'), [sampleEntries]);
  const pottsModels = useMemo(() => selectedCluster?.potts_models || [], [selectedCluster]);

  const stateOptions = useMemo(() => {
    const raw = system?.states;
    if (!raw) return [];
    if (Array.isArray(raw)) return raw;
    if (typeof raw === 'object') return Object.values(raw);
    return [];
  }, [system]);

  const ensemble1Name = useMemo(() => mdSamples.find((s) => s.sample_id === activeMdSampleId)?.name || '', [mdSamples, activeMdSampleId]);
  const ensemble2Name = useMemo(() => mdSamples.find((s) => s.sample_id === inactiveMdSampleId)?.name || '', [mdSamples, inactiveMdSampleId]);
  const ensemble3Name = useMemo(() => mdSamples.find((s) => s.sample_id === pasMdSampleId)?.name || '', [mdSamples, pasMdSampleId]);

  const residueLabels = useMemo(() => {
    const keys = clusterInfo?.residue_keys || [];
    if (Array.isArray(keys) && keys.length) return keys;
    const n = clusterInfo?.n_residues || 0;
    return Array.from({ length: n }, (_, i) => `res_${i}`);
  }, [clusterInfo]);

  const selectedTransitionMeta = useMemo(() => {
    if (!activeMdSampleId || !inactiveMdSampleId || !pasMdSampleId || !modelAId || !modelBId) return null;
    const wantBand = Number(bandFraction);
    const wantTopK = Number(topKResidues);
    const wantTopKE = Number(topKEdges);
    return (
      analyses.find((a) => {
        const mode = (a.md_label_mode || 'assigned').toLowerCase();
        const band = typeof a.band_fraction === 'number' ? a.band_fraction : Number(a.band_fraction);
        const topK = typeof a.top_k_residues === 'number' ? a.top_k_residues : Number(a.top_k_residues);
        const topKE = typeof a.top_k_edges === 'number' ? a.top_k_edges : Number(a.top_k_edges);
        return (
          a.active_md_sample_id === activeMdSampleId &&
          a.inactive_md_sample_id === inactiveMdSampleId &&
          a.pas_md_sample_id === pasMdSampleId &&
          a.model_a_id === modelAId &&
          a.model_b_id === modelBId &&
          mode === mdLabelMode &&
          Boolean(a.drop_invalid) === Boolean(dropInvalid) &&
          Math.abs(Number(band) - wantBand) < 1e-6 &&
          Number(topK) === wantTopK &&
          Number(topKE) === wantTopKE
        );
      }) || null
    );
  }, [analyses, activeMdSampleId, inactiveMdSampleId, pasMdSampleId, modelAId, modelBId, mdLabelMode, dropInvalid, bandFraction, topKResidues, topKEdges]);

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    setAnalysesError(null);
    try {
      const data = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'delta_transition' });
      setAnalyses(Array.isArray(data?.analyses) ? data.analyses : []);
    } catch (err) {
      setAnalysesError(err.message || 'Failed to load analyses.');
      setAnalyses([]);
    }
  }, [projectId, systemId, selectedClusterId]);

  const loadClusterInfo = useCallback(async () => {
    if (!selectedClusterId) return;
    setClusterInfoError(null);
    try {
      const data = await fetchPottsClusterInfo(projectId, systemId, selectedClusterId, {
        modelId: modelAId || undefined,
      });
      setClusterInfo(data);
    } catch (err) {
      setClusterInfoError(err.message || 'Failed to load cluster info.');
      setClusterInfo(null);
    }
  }, [projectId, systemId, selectedClusterId, modelAId]);

  useEffect(() => {
    if (!clusterOptions.length) return;
    if (!selectedClusterId || !clusterOptions.some((c) => c.cluster_id === selectedClusterId)) {
      setSelectedClusterId(clusterOptions[clusterOptions.length - 1].cluster_id);
    }
  }, [clusterOptions, selectedClusterId]);

  useEffect(() => {
    if (!selectedClusterId) return;
    setAnalyses([]);
    setAnalysisData(null);
    setJob(null);
    setJobStatus(null);
    setJobError(null);
    loadClusterInfo();
    loadAnalyses();
  }, [selectedClusterId, loadClusterInfo, loadAnalyses]);

  useEffect(() => {
    if (!pottsModels.length) {
      setModelAId('');
      setModelBId('');
      return;
    }
    const ids = new Set(pottsModels.map((m) => m.model_id));
    if (!modelAId || !ids.has(modelAId)) setModelAId(pottsModels[0].model_id);
    if (!modelBId || !ids.has(modelBId)) setModelBId(pottsModels[0].model_id);
  }, [pottsModels, modelAId, modelBId]);

  useEffect(() => {
    if (!mdSamples.length) {
      setActiveMdSampleId('');
      setInactiveMdSampleId('');
      setPasMdSampleId('');
      return;
    }
    const ids = new Set(mdSamples.map((s) => s.sample_id));
    const pickDistinct = (fallbackIdx, avoid) => {
      const found = mdSamples.find((s) => !avoid.includes(s.sample_id));
      return found ? found.sample_id : mdSamples[fallbackIdx % mdSamples.length].sample_id;
    };
    let e1 = activeMdSampleId && ids.has(activeMdSampleId) ? activeMdSampleId : pickDistinct(0, []);
    let e2 = inactiveMdSampleId && ids.has(inactiveMdSampleId) ? inactiveMdSampleId : pickDistinct(1, [e1]);
    if (e2 === e1) e2 = pickDistinct(1, [e1]);
    let e3 = pasMdSampleId && ids.has(pasMdSampleId) ? pasMdSampleId : pickDistinct(2, [e1, e2]);
    if (e3 === e1 || e3 === e2) e3 = pickDistinct(2, [e1, e2]);
    if (e1 !== activeMdSampleId) setActiveMdSampleId(e1);
    if (e2 !== inactiveMdSampleId) setInactiveMdSampleId(e2);
    if (e3 !== pasMdSampleId) setPasMdSampleId(e3);
  }, [mdSamples, activeMdSampleId, inactiveMdSampleId, pasMdSampleId]);

  // Mol* init
  useEffect(() => {
    let disposed = false;
    let rafId;
    const tryInit = async () => {
      if (disposed) return;
      if (!containerRef.current) {
        rafId = requestAnimationFrame(tryInit);
        return;
      }
      if (pluginRef.current) return;
      setViewerStatus('initializing');
      setViewerError(null);
      const timeout = setTimeout(() => {
        if (disposed) return;
        setViewerStatus('error');
        setViewerError('3D viewer initialization timed out.');
      }, 8000);
      try {
        const plugin = await createPluginUI({ target: containerRef.current, render: renderReact18 });
        if (disposed) {
          plugin.dispose?.();
          clearTimeout(timeout);
          return;
        }
        pluginRef.current = plugin;
        setViewerStatus('ready');
      } catch (err) {
        setViewerStatus('error');
        setViewerError('3D viewer initialization failed.');
      } finally {
        clearTimeout(timeout);
      }
    };
    tryInit();
    return () => {
      disposed = true;
      if (rafId) cancelAnimationFrame(rafId);
      if (pluginRef.current) {
        try {
          pluginRef.current.dispose?.();
        } catch (err) {
          // ignore
        }
        pluginRef.current = null;
      }
    };
  }, []);

  const getBaseComponentWrapper = useCallback(() => {
    const plugin = pluginRef.current;
    const baseRef = baseComponentRef.current;
    if (!plugin || !baseRef) return null;
    const root = plugin.managers.structure.hierarchy.current.structures[0];
    const comps = root?.components;
    if (!Array.isArray(comps)) return null;
    const found = comps.find((c) => c?.cell?.transform?.ref === baseRef) || null;
    if (!found) return null;
    // Overpaint helpers expect a StructureComponentRef with iterable `representations`.
    if (!Array.isArray(found.representations)) return null;
    return found;
  }, []);

  const clearOverpaint = useCallback(async () => {
    const plugin = pluginRef.current;
    const base = getBaseComponentWrapper();
    if (!plugin || !base) return;
    try {
      await clearStructureOverpaint(plugin, [base], ['cartoon']);
    } catch (err) {
      // best effort
    }
  }, [getBaseComponentWrapper]);

  const ensureBaseComponent = useCallback(async () => {
    const plugin = pluginRef.current;
    if (!plugin) return null;
    const structureCell = plugin.managers.structure.hierarchy.current.structures[0]?.cell;
    if (!structureCell) return null;

    // Remove any preset components/representations so we own the visuals on this page.
    try {
      const roots = plugin.managers.structure.hierarchy.current.structures;
      if (roots?.length) await plugin.managers.structure.component.clear(roots);
    } catch (err) {
      // best effort
    }

    const allExpr = MS.struct.generator.all();
    const baseComponent = await plugin.builders.structure.tryCreateComponentFromExpression(structureCell, allExpr, 'phase-base');
    if (!baseComponent) return null;

    const baseColor = hexToInt('#9ca3af');
    await plugin.builders.structure.representation.addRepresentation(baseComponent, {
      type: 'cartoon',
      color: 'uniform',
      colorParams: { value: baseColor },
      transparency: { name: 'uniform', params: { value: 0.0 } },
    });

    baseComponentRef.current = baseComponent.ref;
    return getBaseComponentWrapper();
  }, [getBaseComponentWrapper]);

  const loadStructure = useCallback(async (stateIdOverride) => {
    const plugin = pluginRef.current;
    if (!plugin) return;
    setStructureLoading(true);
    setViewerError(null);
    try {
      baseComponentRef.current = null;
      await plugin.clear();
      await plugin.dataTransaction(async () => {
        const sid = String(stateIdOverride || loadedStructureStateId || '').trim();
        if (!sid) throw new Error('Select a structure to load.');
        const url = `/api/v1/projects/${projectId}/systems/${systemId}/structures/${encodeURIComponent(sid)}`;
        const data = await plugin.builders.data.download(
          { url: Asset.Url(url), label: sid },
          { state: { isGhost: true } }
        );
        const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');
        await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
      });
      await ensureBaseComponent();
      await clearOverpaint();
    } catch (err) {
      setViewerError(err.message || 'Failed to load structure.');
    } finally {
      setStructureLoading(false);
    }
  }, [projectId, systemId, loadedStructureStateId, ensureBaseComponent, clearOverpaint]);

  useEffect(() => {
    if (viewerStatus !== 'ready') return;
    if (!stateOptions.length) return;
    if (!loadedStructureStateId) {
      const first = stateOptions[0]?.state_id;
      if (first) setLoadedStructureStateId(first);
    }
  }, [viewerStatus, stateOptions, loadedStructureStateId]);

  const loadedOnceRef = useRef(false);
  useEffect(() => {
    if (viewerStatus !== 'ready') return;
    if (loadedOnceRef.current) return;
    if (!loadedStructureStateId) return;
    loadedOnceRef.current = true;
    loadStructure(loadedStructureStateId);
  }, [viewerStatus, loadedStructureStateId, loadStructure]);

  // Poll job status (if we submit from this page).
  useEffect(() => {
    if (!job?.job_id) return;
    let cancelled = false;
    const terminal = new Set(['finished', 'failed', 'canceled']);
    const poll = async () => {
      try {
        const status = await fetchJobStatus(job.job_id);
        if (cancelled) return;
        setJobStatus(status);
        if (terminal.has(status?.status)) {
          clearInterval(timer);
          if (status?.status === 'finished') {
            await loadAnalyses();
          }
        }
      } catch (err) {
        if (!cancelled) setJobError(err.message || 'Failed to poll job.');
      }
    };
    const timer = setInterval(poll, 2000);
    poll();
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [job, loadAnalyses]);

  // Load selected analysis payload.
  useEffect(() => {
    const run = async () => {
      setAnalysisDataError(null);
      setAnalysisData(null);
      // If the selection doesn't match an existing analysis, clear the overlay so the user
      // doesn't mistake previous coloring for the new selection.
      if (!selectedTransitionMeta?.analysis_id) {
        await clearOverpaint();
        return;
      }
      setAnalysisDataLoading(true);
      try {
        await clearOverpaint();
        const payload = await fetchClusterAnalysisData(projectId, systemId, selectedClusterId, 'delta_transition', selectedTransitionMeta.analysis_id);
        setAnalysisData(payload);
      } catch (err) {
        setAnalysisDataError(err.message || 'Failed to load analysis.');
      } finally {
        setAnalysisDataLoading(false);
      }
    };
    run();
  }, [projectId, systemId, selectedClusterId, selectedTransitionMeta, clearOverpaint]);

  const commitmentLabels = useMemo(() => {
    const raw = analysisData?.data?.ensemble_labels;
    const base = Array.isArray(raw) && raw.length ? raw : ['Ensemble 1', 'Ensemble 2', 'Ensemble 3', 'TS-band'];
    // Replace generic ensemble labels with selected sample names when available.
    return base.map((name) => {
      if (name === 'Ensemble 1' && ensemble1Name) return ensemble1Name;
      if (name === 'Ensemble 2' && ensemble2Name) return ensemble2Name;
      if (name === 'Ensemble 3' && ensemble3Name) return ensemble3Name;
      return name;
    });
  }, [analysisData, ensemble1Name, ensemble2Name, ensemble3Name]);

  const commitmentMatrix = useMemo(() => (Array.isArray(analysisData?.data?.q_residue) ? analysisData.data.q_residue : []), [analysisData]);
  const topResidueIndices = useMemo(
    () => (Array.isArray(analysisData?.data?.top_residue_indices) ? analysisData.data.top_residue_indices : []),
    [analysisData]
  );

  useEffect(() => {
    if (!commitmentLabels.length) return;
    setCommitmentRowIndex((prev) => {
      const idx = Number(prev);
      if (Number.isInteger(idx) && idx >= 0 && idx < commitmentLabels.length) return idx;
      return 0;
    });
  }, [commitmentLabels]);

  const coloringPayload = useMemo(() => {
    if (!Array.isArray(topResidueIndices) || !topResidueIndices.length) return null;
    if (!Array.isArray(commitmentMatrix) || !commitmentMatrix.length) return null;
    const row = commitmentMatrix[commitmentRowIndex];
    if (!Array.isArray(row) || !row.length) return null;

    const residueIdsAuth = [];
    const residueIdsLabel = [];
    const qValues = [];

    for (let col = 0; col < topResidueIndices.length; col += 1) {
      const ridx = Number(topResidueIndices[col]);
      const q = Number(row[col]);
      if (!Number.isFinite(ridx) || ridx < 0) continue;
      const label = residueLabels[ridx];
      const auth = parseResidueId(label);
      if (auth !== null) residueIdsAuth.push(auth);
      residueIdsLabel.push(ridx + 1); // Mol* label_seq_id is 1-based
      qValues.push(Number.isFinite(q) ? q : NaN);
    }
    return { residueIdsAuth, residueIdsLabel, qValues };
  }, [topResidueIndices, commitmentMatrix, commitmentRowIndex, residueLabels]);

  const applyColoring = useCallback(async () => {
    if (viewerStatus !== 'ready') {
      setColoringDebug({ status: 'skip', reason: 'viewer-not-ready' });
      return;
    }
    if (structureLoading) {
      setColoringDebug({ status: 'skip', reason: 'structure-loading' });
      return;
    }
    if (!analysisData) {
      setColoringDebug({ status: 'skip', reason: 'no-analysis-data' });
      return;
    }
    if (!coloringPayload) {
      setColoringDebug({ status: 'skip', reason: 'no-coloring-payload' });
      return;
    }
    const plugin = pluginRef.current;
    const base = getBaseComponentWrapper();
    if (!plugin || !base) {
      setColoringDebug({ status: 'skip', reason: 'no-base-component' });
      return;
    }

    await clearOverpaint();

    const { residueIdsAuth, residueIdsLabel, qValues } = coloringPayload;
    const useAuth = residueIdMode === 'auth';
    const residueIds = useAuth ? residueIdsAuth : residueIdsLabel;
    const prop = useAuth ? 'auth' : 'label';
    if (!residueIds.length) {
      setColoringDebug({ status: 'skip', reason: 'no-residue-ids-for-mode', mode: residueIdMode });
      return;
    }

    const bins = 11;
    const bucket = Array.from({ length: bins }, () => []);
    for (let i = 0; i < residueIds.length; i += 1) {
      const q = qValues[i];
      if (!Number.isFinite(q)) continue;
      const b = Math.max(0, Math.min(bins - 1, Math.floor(clamp01(q) * (bins - 1) + 1e-9)));
      bucket[b].push(residueIds[i]);
    }

    let layers = 0;
    const selectedElementsByBin = Array.from({ length: bins }, () => 0);
    const rootStructure = plugin.managers.structure.hierarchy.current.structures[0]?.cell?.obj?.data;
    for (let b = 0; b < bins; b += 1) {
      const ids = bucket[b];
      if (!ids.length) continue;
      const qCenter = b / (bins - 1);
      const colorHex = commitmentColor(qCenter);
      const colorValue = hexToInt(colorHex);
      const propFn =
        prop === 'auth'
          ? MS.struct.atomProperty.macromolecular.auth_seq_id()
          : MS.struct.atomProperty.macromolecular.label_seq_id();
      const residueTests =
        ids.length === 1
          ? MS.core.rel.eq([propFn, ids[0]])
          : MS.core.set.has([MS.set(...ids), propFn]);
      const expression = MS.struct.generator.atomGroups({ 'residue-test': residueTests });
      if (rootStructure) {
        const sel = Script.getStructureSelection(expression, rootStructure);
        selectedElementsByBin[b] = StructureSelection.unionStructure(sel).elementCount;
        if (selectedElementsByBin[b] === 0) continue;
      }
      const lociGetter = async (structure) => {
        const sel = Script.getStructureSelection(expression, structure);
        return StructureSelection.toLociWithSourceUnits(sel);
      };
      // eslint-disable-next-line no-await-in-loop
      await setStructureOverpaint(plugin, [base], colorValue, lociGetter, ['cartoon']);
      layers += 1;
    }
    setColoringDebug({
      status: 'run',
      prop,
      mode: residueIdMode,
      residues: residueIds.length,
      bins: bucket.map((x) => x.length),
      created: layers,
      note: 'overpaint layers applied',
      selectedElementsByBin,
    });
  }, [viewerStatus, structureLoading, analysisData, coloringPayload, residueIdMode, clearOverpaint, getBaseComponentWrapper]);

  useEffect(() => {
    applyColoring();
  }, [applyColoring]);

  useEffect(() => {
    // Ensure recoloring happens when the selected row changes even if memoization keeps callback identity stable.
    applyColoring();
  }, [commitmentRowIndex, residueIdMode, analysisData, viewerStatus, structureLoading, applyColoring]);

  const handleRun = useCallback(async () => {
    if (!selectedClusterId) return;
    setJobError(null);
    setJob(null);
    setJobStatus(null);
    try {
      const payload = {
        project_id: projectId,
        system_id: systemId,
        cluster_id: selectedClusterId,
        active_md_sample_id: activeMdSampleId,
        inactive_md_sample_id: inactiveMdSampleId,
        pas_md_sample_id: pasMdSampleId,
        model_a_id: modelAId,
        model_b_id: modelBId,
        md_label_mode: mdLabelMode,
        keep_invalid: keepInvalid,
        band_fraction: Number(bandFraction),
        top_k_residues: Number(topKResidues),
        top_k_edges: Number(topKEdges),
      };
      const res = await submitDeltaTransitionJob(payload);
      setJob(res);
    } catch (err) {
      setJobError(err.message || 'Failed to submit job.');
    }
  }, [
    projectId,
    systemId,
    selectedClusterId,
    activeMdSampleId,
    inactiveMdSampleId,
    pasMdSampleId,
    modelAId,
    modelBId,
    mdLabelMode,
    keepInvalid,
    bandFraction,
    topKResidues,
    topKEdges,
  ]);

  if (loadingSystem) return <Loader message="Loading 3D commitment viewer..." />;
  if (systemError) return <ErrorMessage message={systemError} />;

  const missing = Boolean(
    selectedClusterId &&
      modelAId &&
      modelBId &&
      activeMdSampleId &&
      inactiveMdSampleId &&
      pasMdSampleId &&
      !selectedTransitionMeta
  );

  return (
    <div className="space-y-4">
      <HelpDrawer
        open={helpOpen}
        title="Delta Commitment (3D): How To Read It"
        docPath="/docs/delta_commitment_3d_help.md"
        onClose={() => setHelpOpen(false)}
      />

      <div className="flex items-start justify-between gap-3">
        <div>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/delta_eval`)}
            className="text-cyan-400 hover:text-cyan-300 text-sm"
          >
            ← Back to Delta Potts Evaluation
          </button>
          <h1 className="text-2xl font-semibold text-white">Delta Commitment (3D)</h1>
          <p className="text-sm text-gray-400">
            Load a structure and color residues by commitment <code>q_i</code> for a selected ensemble under a fixed model pair (A,B).
            Residues not in the top-K set are left uncolored (base cartoon is gray).
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setHelpOpen(true)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500 inline-flex items-center gap-2"
          >
            <CircleHelp className="h-4 w-4" />
            Help
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-4">
        <aside className="space-y-3">
          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-2">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-gray-200">Selection</h2>
              <button
                type="button"
                onClick={() => {
                  loadClusterInfo();
                  loadAnalyses();
                }}
                className="text-xs px-2 py-1 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500 inline-flex items-center gap-2"
              >
                <RefreshCw className="h-3 w-3" />
                Refresh
              </button>
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Cluster</label>
              <select
                value={selectedClusterId}
                onChange={(e) => setSelectedClusterId(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
              >
                {clusterOptions.map((c) => (
                  <option key={c.cluster_id} value={c.cluster_id}>
                    {c.name || c.cluster_id}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Load structure (PDB)</label>
              {!stateOptions.length && <p className="text-xs text-gray-500">No states with PDBs available.</p>}
              {!!stateOptions.length && (
                <div className="flex flex-wrap gap-2">
                  {stateOptions.map((st) => {
                    const sid = st?.state_id || st?.id || '';
                    if (!sid) return null;
                    const label = st?.name || sid;
                    const active = sid === loadedStructureStateId;
                    return (
                      <button
                        key={sid}
                        type="button"
                        onClick={async () => {
                          setLoadedStructureStateId(sid);
                          // Load immediately so the user gets feedback like in other viz pages.
                          await loadStructure(sid);
                        }}
                        className={`text-xs px-3 py-2 rounded-md border ${
                          active ? 'border-cyan-500 text-cyan-200' : 'border-gray-700 text-gray-200 hover:border-gray-500'
                        }`}
                      >
                        {label}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Model A</label>
                <select
                  value={modelAId}
                  onChange={(e) => setModelAId(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                >
                  {pottsModels.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.name || m.model_id}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Model B</label>
                <select
                  value={modelBId}
                  onChange={(e) => setModelBId(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                >
                  {pottsModels.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.name || m.model_id}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Ensemble 1 (MD)</label>
                <select
                  value={activeMdSampleId}
                  onChange={(e) => setActiveMdSampleId(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                >
                  {mdSamples.map((s) => (
                    <option key={s.sample_id} value={s.sample_id}>
                      {s.name || s.sample_id}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Ensemble 2 (MD)</label>
                <select
                  value={inactiveMdSampleId}
                  onChange={(e) => setInactiveMdSampleId(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                >
                  {mdSamples.map((s) => (
                    <option key={s.sample_id} value={s.sample_id}>
                      {s.name || s.sample_id}
                    </option>
                  ))}
                </select>
              </div>
              <div className="md:col-span-2">
                <label className="block text-xs text-gray-400 mb-1">Ensemble 3 (MD)</label>
                <select
                  value={pasMdSampleId}
                  onChange={(e) => setPasMdSampleId(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                >
                  {mdSamples.map((s) => (
                    <option key={s.sample_id} value={s.sample_id}>
                      {s.name || s.sample_id}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">MD labels</label>
                <select
                  value={mdLabelMode}
                  onChange={(e) => setMdLabelMode(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                >
                  <option value="assigned">assigned</option>
                  <option value="halo">halo</option>
                </select>
              </div>
              <div className="flex items-end">
                <label className="flex items-center gap-2 text-sm text-gray-200">
                  <input
                    type="checkbox"
                    checked={keepInvalid}
                    onChange={(e) => setKeepInvalid(e.target.checked)}
                    className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
                  />
                  Keep invalid
                </label>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Band fraction</label>
                <input
                  type="number"
                  step="0.01"
                  min="0.01"
                  max="0.99"
                  value={bandFraction}
                  onChange={(e) => setBandFraction(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Top residues</label>
                <input
                  type="number"
                  min="1"
                  value={topKResidues}
                  onChange={(e) => setTopKResidues(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Top edges</label>
                <input
                  type="number"
                  min="1"
                  value={topKEdges}
                  onChange={(e) => setTopKEdges(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Color by commitment</label>
              <select
                value={String(commitmentRowIndex)}
                onChange={(e) => setCommitmentRowIndex(Number(e.target.value))}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
              >
                {commitmentLabels.map((name, idx) => (
                  <option key={`${idx}:${name}`} value={String(idx)}>
                    {name}
                  </option>
                ))}
              </select>
              <p className="text-[11px] text-gray-500 mt-1">
                Colors approximate <code>q_i = Pr(δ_i &lt; 0)</code> for top-K residues.
              </p>
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Residue mapping</label>
                  <select
                    value={residueIdMode}
                    onChange={(e) => setResidueIdMode(e.target.value)}
                    className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                  >
                    <option value="label">Sequential (label_seq_id)</option>
                    <option value="auth">PDB numbering (auth_seq_id)</option>
                  </select>
                  <p className="text-[11px] text-gray-500 mt-1">
                    If nothing is colored, use <span className="font-mono">Sequential</span>. <span className="font-mono">PDB numbering</span> requires
                    matching residue numbers in cluster labels (e.g. <span className="font-mono">res_279</span>).
                  </p>
                </div>
              </div>
              {coloringDebug && (
                <div className="mt-2 rounded-md border border-gray-800 bg-gray-950/40 p-2 text-[11px] text-gray-300">
                  <div>
                    overlay: <span className="font-mono">{coloringDebug.status}</span>{' '}
                    {coloringDebug.reason ? <span className="text-gray-500">({coloringDebug.reason})</span> : null}
                  </div>
                  {coloringDebug.status === 'run' && (
                    <>
                      <div>
                        prop: <span className="font-mono">{coloringDebug.prop}</span> · mode:{' '}
                        <span className="font-mono">{coloringDebug.mode}</span> · residues:{' '}
                        <span className="font-mono">{coloringDebug.residues}</span>
                      </div>
                      {coloringDebug.created !== undefined && (
                        <div>
                          created: <span className="font-mono">{coloringDebug.created}</span>{' '}
                          {coloringDebug.note ? <span className="text-gray-500">({coloringDebug.note})</span> : null}
                        </div>
                      )}
                      <div className="font-mono">bins: {JSON.stringify(coloringDebug.bins)}</div>
                      {coloringDebug.selectedElementsByBin && (
                        <div className="font-mono">elements: {JSON.stringify(coloringDebug.selectedElementsByBin)}</div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>

            {clusterInfoError && <ErrorMessage message={clusterInfoError} />}
            {analysesError && <ErrorMessage message={analysesError} />}
            {jobError && <ErrorMessage message={jobError} />}

            {missing && (
              <div className="rounded-md border border-yellow-800 bg-yellow-950/30 p-3 text-sm text-yellow-200 space-y-2">
                <div>No matching TS analysis found for this selection.</div>
                <button
                  type="button"
                  onClick={handleRun}
                  className="inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-yellow-700 text-yellow-200 hover:border-yellow-500"
                >
                  <Play className="h-4 w-4" />
                  Run TS analysis
                </button>
              </div>
            )}

            {jobStatus?.status && (
              <div className="rounded-md border border-gray-800 bg-gray-950/40 p-2 text-xs text-gray-300">
                Job: <span className="font-mono">{jobStatus.status}</span>
              </div>
            )}
          </div>
        </aside>

        <main className="space-y-3">
          {(viewerError || viewerStatus === 'error') && <ErrorMessage message={viewerError || 'Viewer error'} />}
          {analysisDataError && <ErrorMessage message={analysisDataError} />}

          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-gray-200">3D Viewer</h2>
                <p className="text-[11px] text-gray-500">
                  Blue ≈ q→0, white ≈ q≈0.5, red ≈ q→1.
                </p>
              </div>
              <button
                type="button"
                onClick={() => loadStructure()}
                className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500 inline-flex items-center gap-2"
              >
                <RefreshCw className="h-4 w-4" />
                Reload structure
              </button>
            </div>
            <p className="mt-2 text-[11px] text-gray-500">
              Note: some browsers/GPUs may print WebGL warnings for Mol* even when rendering is correct. If the viewer stays blank,
              try a different structure/state button, refresh the page, or switch browsers.
            </p>
            <div className="mt-3 h-[70vh] min-h-[520px] rounded-md border border-gray-800 bg-black/20 overflow-hidden relative">
              {viewerStatus !== 'ready' && (
                <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/60">
                  <Loader message="Initializing viewer..." />
                </div>
              )}
              {viewerStatus === 'ready' && structureLoading && (
                <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/50">
                  <Loader message="Loading structure..." />
                </div>
              )}
              <div ref={containerRef} className="w-full h-full relative" />
            </div>
            {analysisDataLoading && <p className="mt-2 text-sm text-gray-400">Loading analysis…</p>}
            {!analysisDataLoading && selectedTransitionMeta && !analysisData && (
              <p className="mt-2 text-sm text-gray-400">Select an analysis (or run it) to color residues.</p>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
