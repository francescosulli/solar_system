
# Solar System Workspace

Workspace organizzato in tre progetti distinti e una suite di analisi:

- `MLP/`: baseline emulator MLP puramente data-driven
- `PINN/`: variante physics-informed standard
- `HPINN/`: variante ibrida physics-informed con correzione appresa
- `comparison/`: suite unificata per la generazione dei paper e dei grafici di benchmark

## Struttura

```text
solar_system/
├── MLP/
│   ├── src/
│   ├── tests/
│   └── artifacts/
├── PINN/
│   ├── src/
│   ├── tests/
│   └── artifacts_768x8/
├── HPINN/
│   ├── src/
│   ├── tests/
│   └── artifacts/
├── comparison/
│   ├── paper1_stability/
│   ├── paper2_benchmark/
│   ├── paper3_optimization/
│   ├── paper4_residuals/
│   └── final_values.md
├── data/
│   ├── de440.bsp
│   └── dataset_de440.npz
└── README.md
```

### MLP, PINN, HPINN

Ogni modello può essere testato spostandosi nella rispettiva cartella:

```bash
cd HPINN
PYTHONPATH=src pytest -q tests
```

### Generazione Grafici e Paper

Tutti gli script di comparazione si trovano nella cartella `comparison/`. Per generare i grafici aggiornati (dopo aver addestrato i modelli), eseguire gli script Python all'interno delle rispettive sotto-cartelle (es. `python plot_paper1.py`).
