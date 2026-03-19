# Solar System Workspace

Workspace organizzato in due progetti distinti:

- `MLP/`: baseline emulator MLP del Sistema Solare
- `PINN/`: variante PINN separata, con dataset e artifact propri

## Struttura

```text
solar_system/
├── MLP/
│   ├── src/
│   ├── notebooks/
│   ├── tests/
│   ├── data/
│   └── artifacts/
├── PINN/
│   ├── src/
│   ├── notebooks/
│   ├── tests/
│   ├── data/
│   └── artifacts/
└── README.md
```

## Esecuzione

### MLP

```bash
cd MLP
PYTHONPATH=src pytest -q tests
```

### PINN

```bash
cd PINN
PYTHONPATH=src pytest -q tests
```

## Nota

Il notebook `PINN/notebooks/05_compare_baseline_vs_pinn.ipynb` ora cerca il checkpoint baseline in `../MLP/artifacts/`.
