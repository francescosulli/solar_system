# Solar System Ephemeris Emulator

Progetto Python/Jupyter riproducibile per emulare effemeridi del Sistema Solare con ML:

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


Output attesi:

- dataset `.npz` in `data/`
- checkpoint modello in `artifacts/`
- scena 3D Plotly con orbite PINN + overlay orbite DE440
- esempio campo gravitazionale in un punto

### Profilo long-run (consigliato)

I notebook sono configurati in modalita robusta per massimizzare il fit alle orbite:

- dataset: `2010-01-01 -> 2030-01-01`, passo `3h` (strict DE440/DE441)
- modello: MLP condivisa + Fourier (`max_frequency=6`) con head lineari (config piu stabile)
- training Stage 1 (data-fit): split random, `epochs=1000`, `batch_size=384`, early stopping, cosine LR
- training Stage 2 (fine-tuning fisico leggero): opzionale, disattivato di default se l'obiettivo e solo RMSE
- checkpoint finale: di default Stage 1 (RMSE migliore nelle prove pratiche)
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
