# Consolidacion de resultados de regresion

`consolidate_results` combina los resultados de regresion de multiples timestamps en un unico CSV, agregando metricas de dispersion y p-values de tests de permutacion. El CSV consolidado es el punto de entrada para todos los notebooks de analisis.

## Flujo general

```
results/regression/{timestamp1}/    ─┐
results/regression/{timestamp2}/    ─┤  consolidate_results
results/regression/{timestampN}/    ─┘         │
                                               ├─ 1. concat_regression_results (por cada timestamp)
                                               │     └─ Carga todos los summary.csv, agrega columnas target y dataset
                                               ├─ 2. add_dispersion_metrics
                                               │     └─ Agrega sd_y_true, iqr_y_true, p_value_mae
                                               └─ 3. Valida que no haya duplicados (target, dataset, model)
                                                          │
                                                          ▼
                                    analysis/results/consolidated/combined_summary_with_dispersion_*.csv
```

## Uso basico

```bash
python -m analysis.scripts.consolidate_results <timestamp1> <timestamp2> [<timestampN> ...]
```

Ejemplo con tres timestamps (cada uno contiene targets/datasets distintos):

```bash
python -m analysis.scripts.consolidate_results 2026-02-03_2051 2026-02-03_2053 2026-02-05_2206
```

Esto genera:

```
analysis/results/consolidated/combined_summary_with_dispersion_2026-02-03_2051_2026-02-03_2053_2026-02-05_2206.csv
```

Opcionalmente se puede especificar un path de salida custom:

```bash
python -m analysis.scripts.consolidate_results 2026-02-03_2051 2026-02-03_2053 --output mi_consolidado.csv
```

## Que hace internamente

### 1. Concatenar summary.csv (`concat_regression_results`)

Para cada timestamp, recorre la estructura de directorios y carga todos los `summary.csv`:

```
results/regression/{timestamp}/{target}/{dataset}/summary.csv
```

Agrega dos columnas extraidas de los nombres de directorio: `target` y `dataset`.

### 2. Agregar metricas de dispersion (`add_dispersion_metrics`)

Parsea las columnas `y_true` e `y_pred` (almacenadas como strings de listas Python) y computa:

| Columna | Descripcion |
|---------|-------------|
| `sd_y_true` | Desvio estandar de y_true (ddof=1) |
| `iqr_y_true` | Rango intercuartil de y_true |
| `p_value_mae` | P-value del test de permutacion con MAE (1000 permutaciones, seed=42) |

### 3. Validar duplicados

Verifica que no existan combinaciones duplicadas de `(target, dataset, model)` entre los timestamps. Si hay duplicados lanza un `ValueError` detallando cuales son y en que timestamps aparecen.

> Esto previene combinar accidentalmente el mismo run de ML dos veces. Cada timestamp debe contener targets/datasets distintos.

## Estructura de entrada esperada

Cada timestamp debe tener esta estructura:

```
results/regression/{timestamp}/
├── {target1}/
│   └── {dataset1}/
│       ├── summary.csv    ← metricas agregadas por modelo
│       └── folds.csv      ← predicciones fold por fold
├── {target2}/
│   └── {dataset2}/
│       ├── summary.csv
│       └── folds.csv
└── ...
```

Para listar los timestamps disponibles:

```bash
ls results/regression/
```

## Estructura de salida

El CSV consolidado se guarda en `analysis/results/consolidated/` con todas estas columnas:

| Columna | Descripcion |
|---------|-------------|
| `model` | Nombre del modelo (ej: `Ridge`, `SVR`, `XGBRegressor`) |
| `r2` | R² del LOOCV |
| `mse` | Mean Squared Error |
| `mae` | Mean Absolute Error |
| `y_true` | Valores reales (string de lista Python) |
| `y_pred` | Predicciones (string de lista Python) |
| `target` | Variable objetivo (ej: `ssrt`, `K_6`, `sensibilidad`, `age`) |
| `dataset` | Dataset usado (ej: `tmt_ssrt`, `tmt_k6`, `tmt_dprime`, `tmt_age`) |
| `sd_y_true` | Desvio estandar de y_true |
| `iqr_y_true` | Rango intercuartil de y_true |
| `p_value_mae` | P-value del test de permutacion (MAE) |

## Scripts auxiliares

Cada paso se puede ejecutar de forma independiente:

| Script | Comando | Que hace |
|--------|---------|----------|
| `concat_regression_results` | `python -m analysis.scripts.concat_regression_results --timestamp <ts>` | Concatena los summary.csv de un timestamp en un solo DataFrame |
| `add_dispersion_metrics` | `python -m analysis.scripts.add_dispersion_metrics [--timestamp <ts>]` | Agrega sd, iqr y p-value a un timestamp (usa el ultimo si no se especifica) |

## Notebooks de ejemplo

Todos los notebooks cargan el CSV consolidado de la misma forma:

```python
from src.config import REGRESSION_ANALYSIS_FOLDER
from pathlib import Path

consolidated_dir = Path(REGRESSION_ANALYSIS_FOLDER) / "consolidated"
consolidated_path = sorted(consolidated_dir.glob("combined_summary_with_dispersion_*.csv"))[-1]
df_all = pd.read_csv(consolidated_path)
```

### Analisis de resultados ML (`analysis/notebooks/ml_results/`)

| Notebook | Que analiza |
|----------|-------------|
| `tmt_ssrt_ml_results_analysis.ipynb` | Prediccion de SSRT desde TMT. Validacion del pipeline, comparacion de modelos, analisis de residuos y permutaciones |
| `tmt_k_ml_results_analysis.ipynb` | Prediccion de K_6 desde TMT. Incluye comparacion con otros targets |
| `tmt_dprime_ml_results_analysis.ipynb` | Prediccion de d' (sensibilidad) desde TMT. Analisis de la relacion TMT-percepcion |
| `tmt_k6_analysis.ipynb` | Analisis de viabilidad de K_6 con filtracion de datos, correlaciones feature-target y comparacion con K_4 |

### Metricas (`analysis/notebooks/metrics/`)

| Notebook | Que analiza |
|----------|-------------|
| `dispersion_analysis.ipynb` | Visualiza las metricas de dispersion (sd, iqr) y lista los mejores modelos por MAE para cada target |
| `ml_error_metrics_comparison.ipynb` | Compara metricas de error (MAE, MAPE, SMAPE) entre targets |

## Errores comunes

| Error | Causa | Solucion |
|-------|-------|----------|
| `ValueError: Duplicate (target, dataset, model) combinations` | Dos timestamps tienen el mismo target+dataset | Usar timestamps que contengan targets/datasets distintos |
| `ValueError: Directory not found` | El timestamp no existe en `results/regression/` | Verificar con `ls results/regression/` |
| `ValueError: No summary.csv files found` | El directorio del timestamp existe pero no tiene resultados | Verificar que el run de ML haya terminado correctamente |

## Referencia de funciones

| Funcion | Ubicacion | Que hace |
|---------|-----------|----------|
| `consolidate_results()` | `analysis/scripts/consolidate_results.py` | Orquesta todo el flujo: concatena, agrega metricas y valida |
| `concat_regression_results()` | `analysis/scripts/concat_regression_results.py` | Carga y concatena todos los summary.csv de un timestamp |
| `add_metrics_to_results()` | `analysis/scripts/add_dispersion_metrics.py` | Agrega sd_y_true, iqr_y_true y p_value_mae al DataFrame |
| `permutation_test()` | `src/model/permutation_tests.py` | Ejecuta el test de permutacion (1000 permutaciones, seed=42) |
