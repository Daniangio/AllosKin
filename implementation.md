Documento 1 — Specifica di implementazione (plots + layout + NPZ keys)
0. Obiettivo e principi

Obiettivo: estendere le visualizzazioni attuali (dashboard marginals + curva βeff) con una suite di analisi solida e visivamente immediata per confrontare:

MD vs Gibbs (model fit + representation)

MD vs SA (sampler+encoding+βeff + eventuale mismatch del modello)

Gibbs vs SA (sampler correctness a parità di Hamiltonian)

Principi:

Separare sempre “errore modello” da “errore sampler”: Gibbs è il riferimento per il Potts appreso.

Non affidarsi solo a marginals: includere almeno un controllo su correlazioni pairwise sugli edge E.

Avere almeno 1 “plot a colpo d’occhio” (barcode heatmap / fingerprint) che evidenzi dove stanno gli errori.

1. Estensioni minime all’output run_summary.npz

Attualmente la dashboard usa: p_md, p_gibbs, p_sa, js_gibbs, js_sa, residue_labels, e opzionalmente le griglie su betas, p_gibbs_by_beta, js_gibbs_by_beta, p_sa_by_schedule, js_sa_by_schedule 

plotting

.

1.1 Chiavi nuove (consigliate, minimal)

Aggiungere in run_summary.npz:

Pairwise:

edges: array shape (M,2) con indici residui (già lo avete altrove, ma qui serve per plotting).

p2_md: joint distributions su edges, shape (M, Kmax_r, Kmax_s) oppure formato “ragged” serializzabile.

p2_gibbs: same format.

p2_sa: same format.

Metriche per edge:

js2_gibbs: JS divergence per edge (MD vs Gibbs), shape (M,)

js2_sa: JS divergence per edge (MD vs SA), shape (M,)

js2_sa_vs_gibbs: JS divergence per edge (Gibbs vs SA), shape (M,)

(opzionale ma molto utile):

top_edges_idx: indici dei top edges per mismatch (per plotting rapido) oppure basta computarlo al volo in plotting.

Nota: se le K_r variano per residuo, per p2_* conviene salvare un formato ragged: per ogni edge salvare un blocco 2D Kr x Ks e un array p2_offsets per ricostruire; alternativa: salvare lista di array in .npz non è semplice con allow_pickle=False. Quindi: preferire padding a Kmax + NaN (come fai già per marginals).

2. Nuovi grafici “barcode heatmaps” (fingerprints)
2.1 Barcode: per-residue mismatch (tre righe)

Creare una figura non interattiva (png/pdf) + versione interattiva HTML facoltativa.

Input:

js_gibbs (MD vs Gibbs), js_sa (MD vs SA)

opzionale: tv_gibbs, tv_sa (total variation) se calcolabile

opzionale: dH_gibbs, dH_sa differenza entropia per residuo

Output (barcode heatmap):

Asse x: residui (0..N-1 oppure label)

Asse y: 2 o 3 righe, ad esempio:

JS(MD, Gibbs)

JS(MD, SA)

JS(Gibbs, SA) (se disponibile o calcolabile)

Implementazione:

Aggiungere in plotting.py una funzione plot_barcode_residue_metrics_from_npz(summary_path, out_path_html, out_path_png=None)

Costruire una matrice Y x N e plottare come heatmap Plotly.

Ordinamento residui: fornire toggle/argomento order=:

"index" default

"md_vs_sa" ordina per JS(MD,SA) decrescente

"md_vs_gibbs" idem

"max" usa max su righe

Regola importante:

Se vuoi “visivamente immediato”, non mettere 2000 residui con label lunghi: usa tick sparsi e hover.

2.2 Barcode: per-edge mismatch (due righe)

Stesso concetto per edges.

Input:

js2_gibbs, js2_sa, opzionale js2_sa_vs_gibbs

Output:

Asse x: edges (ordinati per mismatch SA o Gibbs)

Asse y: 2–3 righe:

JS2(MD, Gibbs)

JS2(MD, SA)

JS2(Gibbs, SA) (opzionale)

Aggiunta essenziale: hover deve mostrare l’edge come (r,s) e magari le label residue.

3. “Top offenders” panels (subreport automatico)
3.1 Tabella Top residues

Generare HTML (o JSON) con:

top K residui per JS(MD, SA)

e per ciascuno: JS(MD,Gibbs), JS(MD,SA), max|p_sa-p_md|, max|p_gibbs-p_md|

Puoi aggiungerlo nella dashboard esistente come sezione testuale a lato (semplice).

3.2 Top edges

Analogo per edges:

top K edges per JS2(MD, SA)

stampare (r,s), JS2 per Gibbs e SA.

4. Pairwise plots (immediati e difendibili)
4.1 Network plot (edge mismatch)

Plotly scatter con segmenti:

nodi = residui (posizionati in una riga o cerchio, non serve layout fisico)

edges = subset top M (es 200) per mismatch SA

spessore = mismatch

hover = (r,s) + mismatch Gibbs/SA

Perché utile: fa vedere subito se l’errore è localizzato (poche regioni) o diffuso.

4.2 Heatmap “edge mismatch vs edge strength”

Se hai |J| o una stima di “strength” per edge:

scatter: x = strength, y = js2_sa_vs_gibbs (o js2_sa_vs_md)

color = contact distance o categoria (opzionale)
Serve a diagnosticare: SA fallisce soprattutto sugli accoppiamenti forti?

5. Energetics & βeff sanity checks (oltre alla curva)
5.1 Energy histogram overlay (Gibbs vs SA)

Per ogni campione hai E(x) (dal Potts). Salvare:

E_gibbs_samples, E_sa_samples (array 1D)

(opzionale) E_md_frames se valuti E(x_md)

Plot:

istogramma o KDE (istogramma basta)

curva CDF sovrapposta (molto immediata)

Interpretazione: SA “più caldo” tende a spostarsi verso energie più alte e appiattire.

5.2 βeff per schedule (se avete più schedule SA)

Già fatto in plot_beta_scan_curve; aggiungere:

marker del minimo

stampa del valore numerico in legenda

6. Nearest-neighbor retrieval (solo in ridotto, non 3D)

Implementare solo sulla rappresentazione discreta x.

Input richiesto:

x_md (T x N) labels per frame MD

x_gibbs (S x N) samples Gibbs

x_sa (S x N) samples SA

Metriche:

Hamming distance NN: per ogni sample, distanza al frame MD più vicino

Coverage: per ogni frame MD, distanza al sample più vicino

Plot immediato:

CDF di “NN distance” (precision proxy): Gibbs→MD vs SA→MD

CDF di “coverage” (recall proxy): MD→Gibbs vs MD→SA

Nota: evitare O(TSN) brutale; implementare:

compressione bitset per residui (se K piccole) oppure

LSH/MinHash su one-hot per approssimare, o

baseline più semplice: campionare subset MD (es 50k frames) e subset samples.

Questa parte è molto “wow” visivamente ma va fatta bene per non essere costosissima.

7. Integrazione nel tuo plotting.py
7.1 Struttura file

Nel file attuale 

plotting

 aggiungere:

plot_barcode_metrics_from_npz(...)

plot_edge_barcode_from_npz(...)

plot_energy_overlays_from_npz(...)

plot_nn_cdf_from_npz(...)

helper: _plotly_save_html(payload, layout, out_path) per evitare template duplicati

7.2 Riutilizzare il pattern del dashboard

Hai già un template con sidebar per selezione residui e scelta “sampler” 

plotting

.
Suggerimento pragmatico:

Estendi quel dashboard aggiungendo un tab selector (Marginals / Barcodes / Edges / Energies / NN).
Oppure (più semplice):

generare più HTML separati: marginals.html, barcodes.html, edges.html, energy.html, nn.html.

Io farei multi-HTML: meno JS, più robusto.