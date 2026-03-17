# Solar System Ephemeris Emulator (PINN Variant)

Progetto Python/Jupyter riproducibile per emulare effemeridi del Sistema Solare con ML.
Questa cartella contiene la variante **PINN** separata dal progetto baseline.

- input: tempo `t`
- output: stato 6D `(x, y, z, vx, vy, vz)` per multi-corpo
- post-processing deterministico: campo gravitazionale totale (e potenziale opzionale)
- visualizzazione 3D interattiva con Plotly

## Convenzioni fisiche

- Frame interno unico: **barycentric ICRS/ICRF** (`FRAME_INTERNAL=icrs`, origin barycentric)
- Scala temporale: **TDB** (gestita con `astropy.time.Time`)
- Unita interne: **km**, **s**, **km/s**, `mu` in **km^3/s^2**
- Modulo gravita basato su **mu = GM** (non usa masse separate + `G`)

## Requisiti

- Python >= 3.10
- vedi `pyproject.toml` o `requirements.txt`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

Alternativa:

```bash
pip install -r requirements.txt
```

## Dati DE440/DE441

Il codice prova a usare un kernel locale in `./data/`:

- `data/de440.bsp`
- `data/de441.bsp`
- `data/de440s.bsp`

Modalita strict: se non trova un kernel valido, la pipeline fallisce con errore esplicito.

Download opzionale (se rete disponibile):

```python
from solsys_emulator.de440_dataset import download_kernel
download_kernel()  # salva in data/de440.bsp
```

## Esecuzione End-to-End (notebook)

Apri Jupyter e lancia in ordine:

1. `notebooks/00_setup_and_check.ipynb`
2. `notebooks/01_build_dataset_de440.ipynb`
3. `notebooks/02_train_emulator.ipynb`
4. `notebooks/03_inference_and_viz.ipynb`
5. `notebooks/04_gravity_field_api.ipynb`
6. `notebooks/05_compare_baseline_vs_pinn.ipynb`


Output attesi:

- dataset `.npz` in `data/`
- checkpoint PINN in `artifacts/emulator_pinn.pt`
- scena 3D Plotly con orbite PINN + overlay orbite DE440
- esempio campo gravitazionale in un punto

### Profilo long-run (consigliato)

I notebook sono configurati in modalita robusta per massimizzare il fit alle orbite:

- dataset: `2010-01-01 -> 2030-01-01`, passo `3h` (strict DE440/DE441)
- modello PINN: `state_mode='position_only'` (predice solo `r(t)`), con
  `v=dr/dt` e `a=d2r/dt2` via autograd (tempo normalizzato)
  Fourier features multi-scala: `fourier_features=40`, `frequency_spacing='log'`, `min_frequency=0.25`, `max_frequency=48`
- training PINN (loss fisica attiva): in **3 fasi** e indipendente dalla MLP
  stage1 coarse: 450 epoche, fit solo sulle posizioni (`velocity_loss_weight=0`) per ridurre molto il costo CPU
  stage2 refine: 220 epoche, riattiva la loss sulle velocita via autograd con peso moderato (`velocity_loss_weight=0.4`) e shuffle attivo
  stage3 physics: 60 epoche, fine-tuning n-body con LR molto bassa (`2e-6`), shuffle attivo e bilanciamento adattivo del residuo (`nbody_loss_weight=1e-6`, `adaptive_nbody_balance=True`, `nbody_target_fraction=2e-3`)
  lo stage physics usa tutti i punti del batch per la loss dati ma solo un sottoinsieme di collocation points (`nbody_batch_size=48`) per il residuo n-body, cosi il costo CPU cala molto senza togliere il vincolo fisico
  loss fisica n-body su residuo accelerazione: `a_pred(t) - a_grav(r_pred(t))`
  `physics_loss_weight`/`smoothness_loss_weight` non sono usati in questa architettura
  derivate `dr/dt` e `d2r/dt2` calcolate in modo vettorizzato con `torch.func` quando disponibile
  nessun ordinamento forzato dei batch (`sort_train_for_derivatives=False`) per evitare drift iniziale
  in stage3 la supervisione velocita viene spenta (`velocity_loss_weight=0`) per lasciare che il fine-tuning fisico corregga soprattutto la traiettoria, mentre la 6D supervisionata rimane nello stage2
  in stage2 lo split resta `random` (evita validazione in extrapolazione pura)
  il checkpoint finale viene scelto tra `refine` e `physics` con metrica `val_pos_rmse_km`
- confronto automatico MLP (progetto principale) vs PINN nel notebook `05_compare_baseline_vs_pinn.ipynb`
  (la MLP viene solo caricata per valutazione, non usata in training)
- metriche: RMSE fisiche in km e km/s + confronto orbitale su un anno in `03_inference_and_viz.ipynb` (tutti i pianeti)
- `03_inference_and_viz.ipynb` include benchmark multi-finestra (interpolazione fino al 2029 ed extrapolazione dal 2030)

## Esecuzione da codice (senza notebook)

```python
from solsys_emulator.config import DEFAULT_BODIES
from solsys_emulator.de440_dataset import build_dataset
from solsys_emulator.train import train_emulator

dataset = build_dataset(
    start_time="2025-01-01T00:00:00",
    end_time="2025-03-01T00:00:00",
    step=86400.0,
    bodies=DEFAULT_BODIES,
    kernel_path="data/de440.bsp",  # obbligatorio in modalita strict
)
artifacts = train_emulator(dataset)
```

## Test

```bash
pytest -q
```

Nota: i test dataset/inferenza richiedono un kernel DE440/DE441 presente in `data/`.

Copertura test minima richiesta:

- parsing tempo e scala TDB
- coerenza unita `mu`
- shape dataset e metadata
- sanity checks campo gravitazionale
- API inferenza (`predict_state`, `predict_trajectory`)

## Struttura repository

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── data/
├── artifacts/
├── notebooks/
│   ├── 00_setup_and_check.ipynb
│   ├── 01_build_dataset_de440.ipynb
│   ├── 02_train_emulator.ipynb
│   ├── 03_inference_and_viz.ipynb
│   └── 04_gravity_field_api.ipynb
├── src/solsys_emulator/
│   ├── __init__.py
│   ├── config.py
│   ├── time_frames.py
│   ├── constants.py
│   ├── de440_dataset.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── gravity_field.py
│   ├── viz_3d.py
│   └── utils.py
└── tests/
    ├── test_time_frames.py
    ├── test_constants_units.py
    ├── test_dataset_shapes.py
    ├── test_gravity_field.py
    └── test_inference_api.py
```
