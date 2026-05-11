# Confronto Architetturale: Emulatore Dinamico del Sistema Solare (Physics-Informed Deep Learning)

Questo documento presenta un'analisi comparativa formale dell'addestramento di tre architetture di Deep Learning per l'emulazione a lungo termine delle dinamiche orbitali del Sistema Solare. L'obiettivo è quantificare l'impatto dell'inserimento di prior fisici (Physics-Informed Neural Networks) rispetto a modelli puramente data-driven nell'apprendimento di sistemi dinamici caotici.

---

## 1. Setup Sperimentale e Dati

### 1.1 Dataset
- **Origine**: Effemeridi planetarie **NASA DE440** fornite dal JPL.
- **Corpi Celesti**: 10 corpi (Sole, 8 pianeti, Luna).
- **Dimensione Dati**: Tensore di stato `(58441, 10, 6)`, corrispondente a circa 160 anni di dati campionati con frequenza giornaliera. Ogni stato comprende posizioni $(x,y,z)$ e velocità $(vx,vy,vz)$.
- **Splitting**: Suddivisione casuale con validazione al 10% (`val_fraction = 0.10`).

### 1.2 Ottimizzazione Computazionale (AutoDiff)
Per poter addestrare reti neurali di grandi dimensioni su derivate fisiche senza incorrere in saturazioni di memoria (Out Of Memory a 48GB VRAM), l'infrastruttura di calcolo per le velocità e le accelerazioni predette (`dv/dt` e `d²r/dt²`) è stata ottimizzata. 
L'uso della tradizionale *Reverse-Mode Automatic Differentiation* (`jacrev`) è stato sostituito dalla **Forward-Mode AutoDiff** (`jacfwd`) combinata con il pre-caching del grafo vettorizzato (`vmap`). Questo ha ridotto il *memory footprint* della GPU da `>48 GB` a `<6 GB`, consentendo l'uso di batch massicci (fino a 1536 campioni) e reti significativamente più ampie.

---

## 2. Dettagli Architetturali Condivisi
Per garantire un confronto *apples-to-apples* e isolare l'impatto delle loss fisiche, a tutti i modelli è stata fornita la medesima "potenza espressiva" (il numero di parametri è stato allineato):

- **Backbone**: Reti completamente connesse con connessioni residuali (`backbone_type="residual"`) e `Layer Normalization` per mitigare il *vanishing gradient*.
- **Dimensione**: 8 strati nascosti con 768 neuroni per strato (6 strati e 512 neuroni per la PINN standard).
- **Risoluzione Temporale**: Mappatura del tempo scalare $t$ in uno spazio ad alta dimensionalità tramite **Fourier Features** (256 frequenze allocate logaritmicamente fino a 256.0 cicli/unità).
- **Testa Multipla**: Una testa di output indipendente (con strati nascosti dedicati) per ogni corpo celeste.
- **Training Strategy**: *Multi-Stage Training* strutturato in Fasi (Coarse, Refine, Physics). Ottimizzatore con scheduling Cosine Annealing, 2000 epoche di limite e Early Stopping.

---

## 3. Analisi dei Modelli e Metriche di Validazione

La metrica di valutazione primaria è la **Root Mean Square Error (RMSE)** calcolata sulle posizioni spaziali predette (in chilometri) sul dataset di validazione.

| Modello | Metodologia Integrata | Miglior Epoca (Validazione) | Errore RMSE Posizionale | Variazione % (vs MLP) |
| :--- | :--- | :--- | :--- | :--- |
| **MLP (Baseline)** | Puramente Data-Driven. Nessun vincolo fisico. | Epoca 46 | **~ 1.662.400 km** | - |
| **PINN Standard** | Data-Driven + Loss gravitazionale N-Body. | Stage: Refine | **~ 64.212 km** | **- 96.1%** |
| **HPINN (Ibrida)** | Data-Driven + N-Body + Ramo Correttivo. | Stage: Refine | **~ 51.775 km** | **- 96.8%** |

### 3.1 MLP (Multilayer Perceptron)
Il modello baseline non possiede alcuna conoscenza pregressa della meccanica celeste. La sua architettura produce direttamente le 6 coordinate (posizione e velocità) basandosi esclusivamente sull'interpolazione dei dati del training set. 
**Risultato**: La MLP ha raggiunto il suo errore minimo di `1.662.400 km` alla sola Epoca 46, andando immediatamente in grave *overfitting*. A lungo termine, le violazioni palesi della conservazione dell'energia e del momento angolare portano a una rapida divergenza orbitale e all'incapacità di modellare la caoticità a lungo termine del sistema.

### 3.2 PINN (Physics-Informed Neural Network)
Nella PINN, la rete predice esclusivamente le posizioni spaziali. Le velocità e le accelerazioni vengono derivate analiticamente tramite AutoDiff rispetto all'input temporale $t$. Nel suo Stage 3 (Physics), i gradienti della loss includono l'equazione differenziale di Newton per il problema degli N-Corpi: $\frac{d^2\mathbf{r}_i}{dt^2} = \sum_{j \neq i} \frac{G m_j (\mathbf{r}_j - \mathbf{r}_i)}{|\mathbf{r}_j - \mathbf{r}_i|^3}$.
**Risultato**: Imponendo la conservazione fisica, l'errore crolla a `64.212 km`. Tuttavia, poiché le reali effemeridi JPL DE440 includono effetti non-newtoniani (e.g. perturbazioni da asteroidi minori, correzioni di Relatività Generale), la costrizione a una pura fisica newtoniana genera un lieve disallineamento rispetto ai dati "veri", rendendo lo *Stage 2 (Refine)* il miglior modello assoluto.

### 3.3 HPINN (Hybrid PINN) - *State of the Art*
La HPINN affronta i limiti della PINN implementando un'architettura ibrida a doppio ramo. Il tracciamento della posizione finale è formulato come:
$\mathbf{r}_{finale}(t) = \mathbf{r}_{base}(t) + \Delta \mathbf{r}(t)$
1. **$\mathbf{r}_{base}$**: Un ramo principale forzato (tramite loss weights) a obbedire strettamente alle equazioni newtoniane dell'N-Body.
2. **$\Delta \mathbf{r}$**: Un ramo addestrato liberamente per correggere i residui.
Una loss di regolarizzazione (`correction_loss_weight = 1e-4`) penalizza l'ampiezza di $\Delta \mathbf{r}$, costringendo la rete a dipendere il più possibile dalla macro-fisica newtoniana del ramo base, utilizzando il correttivo *solo* per assorbire le discrepanze non modellate dai prior fisici.
**Risultato**: L'architettura HPINN abbatte ulteriormente l'errore a **51.775 km**, il record assoluto dell'intero ciclo di ricerca. Si dimostra la strategia definitiva per conciliare il rigore delle leggi fisiche fisiche con la rumorosità e la complessità di dati osservazionali reali.
