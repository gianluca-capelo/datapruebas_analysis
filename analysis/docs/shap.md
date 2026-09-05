# Analisis SHAP

SHAP (SHapley Additive exPlanations) permite entender que features contribuyen mas a las predicciones de cada modelo. En este proyecto, se re-entrenan los modelos fold por fold usando los hiperparametros del LOOCV original y se calculan los valores SHAP para cada sujeto de test.

## Generar figuras SHAP

El script batch genera las figuras de importancia SHAP para todas las combinaciones configuradas:

```bash
python -m analysis.scripts.figures.shap_main
```

Esto produce:
- **Figuras PNG** en `analysis/figures/` (ej: `shap_summary_tmt_age_SVR.png`)
- **CSVs con valores SHAP crudos** en `results/regression/{timestamp}/{target}/{dataset}/` (ej: `shap_values_SVR.csv`)

El script muestra una barra de progreso y al final un resumen de cuantas figuras se generaron correctamente.

> Cada combinacion puede tardar varios minutos porque recalcula SHAP para todos los folds del LOOCV.

## Agregar o quitar combinaciones

Las combinaciones se definen en `analysis/scripts/figures/shap_common.py` en la lista `COMBINATIONS`:

```python
COMBINATIONS = [
    {"dataset": "tmt_age",    "model": "SVR",   "task": "regression", "timestamp": "2026-02-03_2053"},
    {"dataset": "tmt_dprime", "model": "Ridge", "task": "regression", "timestamp": "2026-02-03_2053"},
    # ...
]
```

Cada entrada tiene 4 claves:

| Clave | Descripcion | Ejemplo |
|-------|-------------|---------|
| `dataset` | Nombre del dataset en `DatasetBuilder` | `tmt_age`, `tmt_dprime`, `tmt_k_mean` |
| `model` | Nombre de la clase del modelo | `Ridge`, `SVR`, `XGBRegressor`, `Lasso` |
| `task` | Tipo de tarea | `regression` o `classification` |
| `timestamp` | Carpeta de resultados del run de ML | `2026-02-03_2053` |

### Para agregar una combinacion

1. **Encontrar el timestamp**: listar las carpetas disponibles:
   ```bash
   ls results/regression/
   ```

2. **Verificar que el modelo existe** en el `folds.csv` de ese run:
   ```bash
   python -c "
   import pandas as pd
   df = pd.read_csv('results/regression/TIMESTAMP/TARGET/DATASET/folds.csv')
   print(df['model'].unique())
   "
   ```

3. **Agregar la entrada** a la lista `COMBINATIONS` en `figures/shap_common.py`.

### Para quitar una combinacion

Eliminar o comentar la linea correspondiente en la lista `COMBINATIONS`.

## Notebook de analisis interactivo

El notebook `analysis/notebooks/shap_analysis/tmt_age_shap_analysis.ipynb` permite explorar los resultados SHAP de una combinacion individual de forma interactiva.

Para analizar otra combinacion, cambiar las constantes en la primera celda de codigo:

```python
DATASET = "tmt_age"      # Cambiar por el dataset deseado
MODEL = "SVR"             # Cambiar por el modelo deseado
TASK = "regression"       # regression o classification
TIMESTAMP = "2026-02-03_2053"  # Cambiar por el timestamp correspondiente
```

El notebook muestra:
- Informacion del dataset (cantidad de sujetos, features, estadisticas del target)
- Tabla con mean |SHAP| y frecuencia de seleccion por feature
- Grafico de barras con las 20 features mas importantes
- Resumen de los top 10 features

## Archivos de salida

| Archivo | Ubicacion | Contenido |
|---------|-----------|-----------|
| Figura PNG | `analysis/figures/shap_summary_{dataset}_{model}.png` | Grafico de barras con las 20 features mas importantes |
| CSV crudo | `results/{task}/{timestamp}/{target}/{dataset}/shap_values_{model}.csv` | Valores SHAP por fold y feature |

### Formato del CSV crudo

| Columna | Descripcion |
|---------|-------------|
| `fold` | Numero de fold (0 a N-1, uno por sujeto en LOOCV) |
| `base_value` | Prediccion base del modelo (valor esperado promedio) |
| `feature_1` ... `feature_n` | Valor SHAP crudo (con signo) de cada feature |

Las features no seleccionadas por `SelectKBest` en un fold aparecen como `NaN`.

## Referencia de funciones

| Funcion | Ubicacion | Que hace |
|---------|-----------|----------|
| `run_shap()` | `src/model/shap/run_shap.py` | Computa SHAP values para todos los folds y guarda el CSV crudo |
| `analyze_shap_results()` | `src/model/shap/analyze_shap_results.py` | Calcula mean \|SHAP\| y frecuencia de seleccion por feature |
| `plot_shap_summary()` | `src/model/shap/analyze_shap_results.py` | Genera el grafico de barras horizontales |
| `run_analysis()` | `src/model/shap/analyze_shap_results.py` | Combina los 3 pasos anteriores en una sola llamada |
