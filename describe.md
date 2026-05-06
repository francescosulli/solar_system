# Solar System Emulator: panoramica dei tre sistemi

Questo repository contiene tre progetti distinti, costruiti per studiare lo stesso problema da tre punti di vista diversi:

- `MLP/`
- `PINN/`
- `HPINN/`

Tutti e tre lavorano sullo stesso dominio applicativo: apprendere una mappa temporale del tipo

- `t -> stato dei corpi del Sistema Solare`

in cui lo stato di ciascun corpo contiene almeno:

- posizione `r = (x, y, z)`
- velocità `v = (vx, vy, vz)`

Il dataset di riferimento è derivato da effemeridi JPL `DE440` in frame barycentrico `ICRS/ICRF`, con unità coerenti in `km` e `s`.

## 1. Obiettivo generale del repository

L'idea di fondo è confrontare tre famiglie di emulatori:

1. un modello puramente data-driven
2. un modello physics-informed con dinamica newtoniana
3. un modello ibrido che combina fisica e correzione appresa

Questa organizzazione permette di rispondere a tre domande diverse:

- quanto bene possiamo interpolare il dataset con un regressore puro?
- quanto aiuta imporre esplicitamente la fisica nel training?
- possiamo ottenere un compromesso migliore usando una fisica di base più una correzione residua appresa?

---

## 2. `MLP/`: baseline puramente data-driven

La cartella `MLP/` contiene il modello più semplice dal punto di vista concettuale.

### Idea

Il modello `MLP` apprende direttamente la relazione:

- `t -> stato 6D di tutti i corpi`

senza imporre vincoli fisici forti durante il training.

### Caratteristiche principali

- usa una rete feed-forward con Fourier features sul tempo
- ottimizza principalmente la loss sui dati del catalogo
- è il sistema più vicino a una regressione pura delle effemeridi
- in genere è il più semplice da addestrare e il più stabile

### Vantaggi

- training relativamente semplice
- ottima baseline per l'interpolazione del dataset
- meno fragile dal punto di vista numerico
- utile come riferimento per misurare quanto le PINN migliorano o peggiorano

### Limiti

- non garantisce coerenza dinamica
- può approssimare molto bene `DE440` senza essere particolarmente “fisico”
- se si esce dal regime di training, non ha vincoli forti che lo guidino

### Quando usarlo

- quando serve una baseline solida
- quando interessa soprattutto il fit del dataset
- quando si vuole un termine di paragone pulito per PINN e HPINN

---

## 3. `PINN/`: PINN Newtoniana pura

La cartella `PINN/` contiene il modello physics-informed “classico” del progetto.

### Idea

La `PINN` non si limita a inseguire i dati, ma aggiunge nel training un vincolo fisico ispirato alla dinamica gravitazionale newtoniana accoppiata tra i corpi.

In altre parole, il modello non deve solo prevedere traiettorie vicine a `DE440`, ma deve anche produrre stati compatibili con una dinamica `N-body` costruita con i parametri gravitazionali `mu = GM`.

### Caratteristiche principali

- usa ancora una rete neurale tempo -> stato
- aggiunge termini di loss fisica nel training
- sfrutta accelerazioni, vincoli gravitazionali e altre quantità dinamiche
- è una PINN nel senso proprio del termine: i dati e la fisica entrano entrambi nell’obiettivo di training

### Vantaggi

- maggiore interpretabilità fisica rispetto alla `MLP`
- la traiettoria appresa è guidata da una struttura dinamica
- è più coerente con il problema orbitale accoppiato

### Limiti

- `DE440` non è un puro sistema newtoniano a 10 corpi idealizzato
- quindi una PINN newtoniana pura può entrare in conflitto con il catalogo
- il training è più costoso, più delicato e più sensibile ai pesi delle loss
- può peggiorare l’RMSE sui dati anche se “migliora” dal punto di vista fisico

### Quando usarla

- quando vogliamo un emulatore con contenuto fisico esplicito
- quando vogliamo studiare il trade-off tra accuratezza sui dati e coerenza dinamica
- quando la domanda scientifica è importante almeno quanto la prestazione numerica

---

## 4. `HPINN/`: PINN ibrida con correction term

La cartella `HPINN/` è la variante più avanzata e più flessibile del repository.

### Idea

La `HPINN` parte dall’idea della `PINN`, ma riconosce un fatto importante:

- la dinamica newtoniana pura è utile come struttura di base
- ma non basta sempre a riprodurre perfettamente `DE440`

Per questo la traiettoria finale viene scritta come:

- `r_final(t) = r_base(t) + delta_r(t)`

in cui:

- `r_base(t)` è la componente physics-informed principale
- `delta_r(t)` è una correzione appresa dalla rete

Questa correzione non dovrebbe dominare il modello: deve restare piccola, residuale e controllata.

### Caratteristiche principali

- mantiene l’impostazione physics-informed
- aggiunge un correction branch dedicato
- include una regolarizzazione esplicita per evitare che la correzione diventi troppo grande
- è pensata per combinare:
  - struttura fisica
  - flessibilità numerica
  - migliore aderenza a `DE440`

### Vantaggi

- è il compromesso più interessante tra `MLP` e `PINN`
- conserva una base fisica
- permette di correggere ciò che la fisica newtoniana pura non cattura bene
- è la candidata naturale quando la `PINN` pura è troppo rigida

### Limiti

- è il modello più complesso dei tre
- richiede più attenzione nel tuning
- se la correzione non è regolarizzata bene, rischia di trasformarsi in una MLP mascherata
- il training può essere molto pesante dal punto di vista computazionale

### Quando usarla

- quando la `PINN` pura non basta
- quando si vuole mantenere una base fisica ma migliorare il fit su `DE440`
- quando si ha disponibilità di GPU importanti e si vuole sperimentare il compromesso migliore

---

## 5. Differenze concettuali in una riga

Possiamo riassumere i tre sistemi così:

- `MLP`: imparo i dati
- `PINN`: imparo i dati, ma rispettando Newton
- `HPINN`: parto da Newton, poi correggo in modo controllato ciò che Newton da solo non spiega abbastanza bene

---

## 6. Dataset e convenzioni condivise

I tre sistemi sono separati come progetto, ma condividono le stesse scelte fisiche fondamentali.

### Frame

- frame interno: barycentric `ICRS/ICRF`
- origine: barycentrica

### Tempo

- tempi gestiti in modo coerente con le effemeridi
- conversioni tramite `astropy.time.Time`
- training costruito su tempi numerici normalizzati, ma con metadati fisici espliciti

### Unità

- unità base: `km`, `s`
- parametri gravitazionali `mu` coerenti con queste unità

### Corpi tipici

I modelli lavorano tipicamente almeno su:

- Sole
- Mercurio
- Venere
- Terra
- Luna
- Marte
- Giove
- Saturno
- Urano
- Nettuno

---

## 7. Come leggere i risultati

Quando confrontiamo i tre sistemi, è importante non guardare una sola metrica.

### Accuratezza sul dataset

La metrica più immediata è l’errore rispetto a `DE440`, per esempio:

- `RMSE posizione [km]`
- `RMSE velocità [km/s]`

Qui spesso la `MLP` parte avvantaggiata, perché ottimizza direttamente il fit.

### Coerenza fisica

Per `PINN` e `HPINN` bisogna anche guardare:

- residuo `N-body`
- stabilità dinamica
- eventuale comportamento in extrapolazione
- dimensione del correction term nel caso `HPINN`

### Interpretazione corretta

- se `MLP` ha RMSE minore, non significa automaticamente che sia “migliore” in senso fisico
- se `PINN` o `HPINN` hanno RMSE leggermente peggiore ma dinamica più coerente, possono essere scientificamente più interessanti
- la `HPINN` serve proprio a cercare il miglior compromesso tra questi due mondi

---

## 8. Flusso di lavoro consigliato

Un flusso di lavoro ragionevole nel repository è:

1. costruire o verificare il dataset `DE440`
2. addestrare la baseline in `MLP/`
3. addestrare la `PINN/`
4. addestrare la `HPINN/`
5. confrontare i risultati su:
   - RMSE per corpo
   - qualità delle orbite
   - comportamento fisico
   - costo di training

---

## 9. Quale usare in pratica?

### Se vuoi la baseline più pulita

Usa `MLP/`.

### Se vuoi il modello più “scientificamente puro”

Usa `PINN/`.

### Se vuoi la strada più promettente per superare il compromesso MLP vs PINN

Usa `HPINN/`.

---

## 10. Stato del repository

Ad oggi il repository è organizzato in modo che:

- `MLP/` rappresenti la baseline data-driven
- `PINN/` rappresenti la PINN Newtoniana pura
- `HPINN/` rappresenti la PINN ibrida con correction term

Questa separazione è intenzionale: permette di confrontare i tre approcci senza mischiare codice, configurazioni e checkpoint.

Se il progetto evolverà ulteriormente, questa struttura rende naturale aggiungere nuove famiglie di modelli senza compromettere quelle esistenti.
