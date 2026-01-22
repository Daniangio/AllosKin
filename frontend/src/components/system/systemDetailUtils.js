export function getClusterDisplayName(run) {
  if (!run) return 'Cluster';
  return run.name || run.path?.split('/').pop() || run.cluster_id || 'Cluster';
}

export function getArtifactDisplayName(pathValue) {
  if (typeof pathValue !== 'string' || !pathValue) return '—';
  const parts = pathValue.split('/');
  return parts[parts.length - 1] || pathValue;
}

export function formatClusterAlgorithm(run) {
  if (!run) return '';
  const algo = (run.cluster_algorithm || '').toLowerCase();
  const params = run.algorithm_params || {};
  if (algo === 'dbscan') {
    return `dbscan (eps=${params.eps ?? '—'}, min_samples=${params.min_samples ?? '—'})`;
  }
  if (algo === 'hierarchical') {
    return `hierarchical (n_clusters=${params.n_clusters ?? '—'}, linkage=${params.linkage || 'ward'})`;
  }
  if (algo === 'tomato') {
    return `tomato (k=${params.k_neighbors ?? '—'}, tau=${params.tau ?? '—'}, k_max=${params.k_max ?? '—'})`;
  }
  if (algo === 'density_peaks' || algo === 'kmeans') {
    return `${algo} (max_clusters=${run.max_clusters_per_residue ?? '—'})`;
  }
  return algo ? `${algo}` : 'cluster';
}
