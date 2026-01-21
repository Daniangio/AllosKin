Documento 2 — Guida interpretativa (semplice + rigorosa)
1) Perché questi grafici?

Stiamo confrontando distribuzioni di microstati torsionali 
𝑥
𝑟
∈
{
1
,
…
,
𝐾
𝑟
}
x
r
	​

∈{1,…,K
r
	​

} ottenute da:

MD: riferimento “reale” nel ridotto (ma dipende dal clustering)

Gibbs: campionamento corretto della distribuzione del Potts appreso 
𝑝
(
𝑥
)
∝
𝑒
−
𝛽
𝐸
(
𝑥
)
p(x)∝e
−βE(x)

SA/QUBO: un sampler che non garantisce Boltzmann a β noto; per questo stimiamo βeff.

Quindi:

Se Gibbs vs MD è brutto → problema di rappresentazione (clustering) o di modello (Potts fit).

Se SA vs Gibbs è brutto → problema del sampler/encoding/penalties o βeff.

Se SA vs MD è brutto ma Gibbs vs MD è buono → colpa del sampler (buona notizia: diagnosi chiara).

Se entrambi sono brutti → prima si sistema modello/rappresentazione; SA non è interpretabile.

2) JS divergence per-residue: cosa significa davvero?

Per ogni residuo r, confrontiamo due distribuzioni sui microstati:

𝑝
MD
(
𝑥
𝑟
)
p
MD
	​

(x
r
	​

)

𝑝
sample
(
𝑥
𝑟
)
p
sample
	​

(x
r
	​

)

La JS divergence (Jensen–Shannon) è:

0 se le distribuzioni coincidono

cresce se il sampler sbaglia le occupazioni (es: manca uno stato, sovrastima un altro)

Come leggerla:

Un istogramma di JS per residuo è un “riassunto”: più massa vicino a 0 → meglio.

Outlier (barre lontane) indicano residui dove:

il cluster è raro/instabile,

oppure il modello non cattura i vincoli,

oppure SA si blocca su uno subset di stati.

Limite: JS per residuo ignora le correlazioni tra residui.

3) Barcode heatmaps: perché sono potenti

Un barcode è una heatmap con pochissime righe e moltissime colonne (residui o edges).
È “immediato” perché mostra:

dove il problema è localizzato

se è un problema diffuso o a macchie

Barcode per residui

Riga “JS(MD,Gibbs)” = bontà del modello (fit + rappresentazione)

Riga “JS(MD,SA)” = bontà complessiva SA rispetto a MD

Riga “JS(Gibbs,SA)” = bontà del sampler SA rispetto al riferimento corretto per quel Potts

Come interpretare pattern tipici:

Riga 1 bassa, Riga 3 alta: modello ok, SA sbaglia (encoding/βeff/mixing)

Riga 1 alta, Riga 3 bassa: SA imita Gibbs bene, ma Potts non spiega MD (problema del modello/cluster)

Tutte alte: non ha senso discutere SA finché non sistemi rappresentazione/modello

4) Pairwise edge metrics: cosa diagnosticano

Le coppie 
𝑝
(
𝑥
𝑟
,
𝑥
𝑠
)
p(x
r
	​

,x
s
	​

) sono il primo punto in cui vedi “fisica” (coupling) nel ridotto.

Perché servono:
Un sampler può matchare i marginals ma fallire le correlazioni (classico failure mode).
Quindi:

JS2(MD,Gibbs) misura se il Potts appreso cattura le coppie

JS2(Gibbs,SA) misura se SA campiona correttamente la stessa energia

Interpretazione rapida:

Se gli errori edge sono concentrati su pochi edges → probabilmente qualche coupling forte che SA non riesce a rispettare (freeze-out, penalties).

Se sono diffusi → mismatch globale (βeff sbagliato o schedule troppo aggressivo).

5) Energy histogram overlay: cosa ti dice in 2 secondi

Se SA è “più caldo”, vedrai:

distribuzione energie spostata verso valori più alti rispetto a Gibbs a β target

anche dopo βeff: potresti vedere code diverse (segno che SA non è una semplice Boltzmann con βeff)

A cosa serve:

capire se l’errore è “solo temperatura” o “forma della distribuzione”.

6) Nearest-neighbor retrieval (in ridotto): come non farsi ingannare

“Il sample è vicino a un frame MD” può succedere anche con un generatore pessimo che collassa su un modo comune.

Per essere serio, devi guardare due curve:

precision proxy: distanza sample→MD (NN)

coverage/recall proxy: distanza MD→sample (NN)

Interpretazione:

SA può avere precision alta ma coverage bassa → sta ripetendo pochi modi.

Gibbs (se ok) dovrebbe dare buon compromesso.

Cosa fare subito (ordine consigliato)

Se vuoi il massimo impatto con poca fatica:

Barcode residues (3 righe: MD–Gibbs, MD–SA, Gibbs–SA)

Barcode edges (MD–Gibbs, MD–SA, Gibbs–SA) su top edges

Energy histogram + CDF (Gibbs vs SA a βeff)

NN precision/coverage CDF (in ridotto)

Sono 4 figure che, insieme, rendono il report “paper-grade”.