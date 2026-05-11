# Il Problema della Memoria nelle PINN e la sua Risoluzione

Questo documento illustra la natura del gravissimo *memory leak* (saturazione della memoria) che affliggeva inizialmente i modelli **PINN** e **HPINN** di questo progetto, rendendo impossibile l'addestramento di reti neurali profonde, e documenta la soluzione tecnica adottata per abbattere drasticamente i consumi di VRAM.

---

## 1. Il Sintomo
Durante l'addestramento della PINN, utilizzando *batch size* grandi (es. 1024) e architetture di dimensioni modeste (es. `128x4`), la memoria VRAM della GPU (48 GB) veniva completamente saturata in poche epoche, portando inevitabilmente a un errore fatale `CUDA Out Of Memory` (OOM). 
Inoltre, le epoche di addestramento impiegavano quasi un minuto l'una, rendendo il *Multi-Stage Training* estremamente lento.

## 2. La Causa Tecnica: `jacrev` e la ricompilazione
Il collo di bottiglia risiedeva nella funzione `_position_only_to_full_state_norm`, responsabile del calcolo delle derivate fisiche (velocità e accelerazioni) tramite **Automatic Differentiation (AutoDiff)**.

Il codice originale era il seguente:
```python
def _single_time_pos_norm(t_scalar: Tensor) -> Tensor:
    return model(t_scalar.reshape(1))[0]

vel_norm_time = vmap(jacrev(_single_time_pos_norm))(t_norm)
acc_norm_time = vmap(jacrev(jacrev(_single_time_pos_norm)))(t_norm)
```

I problemi di questo approccio erano due:
1. **Reverse-Mode Inefficiente**: La funzione `jacrev` utilizza la Reverse-Mode AutoDiff (Backpropagation). Sebbene sia efficiente per reti con molti input e un solo output (es. gradienti della loss), è **estremamente inefficiente per funzioni con 1 solo input e molti output**. Poiché la nostra rete prende in input un singolo scalare (il tempo `t`) e restituisce lo stato 3D/6D di 10 pianeti, la Reverse-Mode era costretta a ripercorrere l'intero grafo computazionale al contrario *per ogni singolo output*, accumulando enormi tensori in memoria.
2. **Mancanza di Caching del `vmap`**: Richiamando esplicitamente la funzione `vmap(jacrev(...))` a ogni singolo batch (migliaia di volte), PyTorch era costretto a tracciare (JIT compile) e ricreare il grafo differenziale vettorizzato *da zero* a ogni iterazione. Questo continuo accumulo di alberi computazionali senza un garbage collection immediato creava il memory leak che portava al blocco.

## 3. La Soluzione: `jacfwd` e Caching

La soluzione implementata agisce su entrambi i fronti.

### A) Forward-Mode AutoDiff (`jacfwd`)
Abbiamo sostituito `jacrev` con **`jacfwd`** (Forward-Mode AutoDiff).
Poiché l'input della rete è uno scalare (1 dimensione), la Forward-Mode calcola *tutte* le derivate vettoriali per tutti i pianeti in **un singolo passaggio in avanti**, simultaneamente al calcolo della traiettoria base, senza dover salvare in memoria l'immensa traccia di calcolo richiesta dalla Backpropagation.
Questo ha rimosso quasi il 90% dell'occupazione di memoria strutturale.

### B) VMAP Caching (Pre-compilazione del grafo)
Invece di chiamare `vmap(jacfwd(...))` a ogni batch, abbiamo forzato il modello a compilare queste operazioni differenziali *una sola volta* alla prima esecuzione, salvandole in una cache interna (`_vmap_fns`):

```python
inner_model = _unwrap_model(model)
if not hasattr(inner_model, "_vmap_fns"):
    def _single_time_pos_norm(t_scalar: Tensor) -> Tensor:
        return inner_model(t_scalar.reshape(1))[0]
    
    # Precompiliamo e salviamo le funzioni vettorizzate
    inner_model._vmap_fns = {
        "vel": vmap(jacfwd(_single_time_pos_norm)),
        "acc": vmap(jacfwd(jacfwd(_single_time_pos_norm)))
    }

# Riutilizziamo sempre le stesse funzioni già compilate
vel_norm_time = inner_model._vmap_fns["vel"](t_norm)
acc_norm_time = inner_model._vmap_fns["acc"](t_norm)
```

## 4. Risultati Ottenuti
L'applicazione congiunta di `jacfwd` e del Caching ha generato risultati impressionanti:
- **Crollo dell'uso VRAM**: La memoria utilizzata è scesa da `>48 GB` a meno di `2 GB` per le architetture base.
- **Scaling Architetturale**: Grazie allo spazio liberato, è stato possibile scalare le reti PINN e HPINN a dimensioni "Giganti" (fino a 768 neuroni per 8 strati nascosti) mantenendo un footprint ridicolo (sotto i 6 GB).
- **Abbattimento dei Tempi**: Il calcolo di un'epoca con 1536 elementi per batch è passato da quasi 60 secondi a **meno di 5 secondi**.
