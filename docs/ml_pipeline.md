# Pipeline de Machine Learning

**Fecha:** 19 de diciembre de 2025

Este documento explica cómo ejecutar el pipeline de ML para predecir variables cognitivas usando datos de tareas digitales.

---

## Índice

1. [Requisitos Previos](#requisitos-previos)
2. [Ejecución del Pipeline](#ejecución-del-pipeline)
3. [Datasets Disponibles](#datasets-disponibles)
4. [Estructura de Resultados](#estructura-de-resultados)
5. [Configuración](#configuración)
6. [Modelos Disponibles](#modelos-disponibles)
7. [Agregar Nuevos Datasets](#agregar-nuevos-datasets)

---

## Requisitos Previos

1. **Datos de análisis generados:**
   - TMT: Debe existir un análisis en `data/hand_analysis/` con `analysis.csv` y `configuration.json`
   - SST: Debe existir un análisis en `data/sst_analysis/` con `analysis.csv`

2. **Dependencias instaladas:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Ejecución del Pipeline

### Regresión (predecir variable continua)

```bash
python -m src.model.run_models --task regression
```

### Clasificación (predecir categoría)

```bash
python -m src.model.run_models --task classification
```

> **Nota:** El target se define en cada dataset, no se pasa como parámetro. Ver sección [Datasets Disponibles](#datasets-disponibles).

---

## Datasets Disponibles

Cada dataset define sus propias features y target:

| Dataset | Features | Target | Descripción |
|---------|----------|--------|-------------|
| `tmt_ssrt` | TMT (136 features) | `ssrt` | Features de Trail Making Test para predecir Stop Signal Reaction Time |

### Cómo se construye `tmt_ssrt`

1. **Carga TMT:** Lee `analysis.csv` del último run en `data/hand_analysis/`
2. **Filtra:** Solo trials con `is_valid == True`
3. **Detecta features:** Columnas numéricas automáticamente (excluyendo metadata)
4. **Agrega:** Promedio por sujeto, pivoteado por `trial_type` (PART_A, PART_B)
5. **Carga SST:** Lee SSRT de `data/sst_analysis/`
6. **Merge:** Une por `subject_id`

Resultado: ~170 sujetos × 136 features

---

## Estructura de Resultados

```
results/
├── classification/
│   └── {timestamp}/
│       └── {target}/
│           └── {dataset}/
│               ├── config.json    # Configuración del experimento
│               ├── folds.csv      # Predicciones por fold (LOOCV)
│               └── summary.csv    # Métricas por modelo
│
└── regression/
    └── {timestamp}/
        └── {target}/
            └── {dataset}/
                ├── config.json
                ├── folds.csv
                └── summary.csv
```

### Archivos de salida

#### `config.json`
Configuración del experimento:
```json
{
    "dataset": "tmt_ssrt",
    "feature_selection": true,
    "tune_hyperparameters": false,
    "is_classification": false,
    "timestamp": "2025-12-19_1200",
    "n_folds": 170,
    "feature_names": ["rt_PART_A", "rt_PART_B", ...]
}
```

#### `summary.csv`
Métricas agregadas por modelo:

| Columna | Descripción |
|---------|-------------|
| `model` | Nombre del modelo |
| `r2` | R² score (regresión) |
| `mse` | Mean Squared Error |
| `mae` | Mean Absolute Error |
| `accuracy` | Accuracy (clasificación) |
| `roc_auc` | ROC AUC (clasificación) |
| `selected_features` | Features seleccionados |

#### `folds.csv`
Predicciones detalladas por fold (Leave-One-Out):

| Columna | Descripción |
|---------|-------------|
| `fold` | Índice del fold |
| `model` | Nombre del modelo |
| `y_true` | Valor real |
| `y_pred` | Predicción |
| `best_params` | Hiperparámetros óptimos |

---

## Configuración

Editar `src/config.py`:

```python
# Seeds para reproducibilidad
MODEL_OUTER_SEED = 47
MODEL_INNER_SEED = 66
INNER_CV_SPLITS = 10

# Feature selection
PERFORM_FEATURE_SELECTION = True
MAX_SELECTED_FEATURES = 20

# Hyperparameter tuning
TUNE_HYPERPARAMETERS = True

# Datasets a evaluar (cada uno define su propio target)
DATASETS = ['tmt_ssrt']
```

---

## Modelos Disponibles

### Regresión

| Modelo | Descripción |
|--------|-------------|
| `DummyRegressor` | Baseline (predice la media) |
| `LinearRegression` | Regresión lineal |
| `Ridge` | Regresión con regularización L2 |
| `Lasso` | Regresión con regularización L1 |
| `ElasticNet` | Combinación L1 + L2 |
| `SVR` | Support Vector Regression |
| `RandomForestRegressor` | Random Forest |
| `XGBRegressor` | XGBoost |

### Clasificación

| Modelo | Descripción |
|--------|-------------|
| `RandomForestClassifier` | Random Forest |
| `SVC` | Support Vector Classification |
| `LogisticRegression` | Regresión logística |
| `XGBClassifier` | XGBoost |

---

## Agregar Nuevos Datasets

Cada dataset encapsula sus features y su target. Para agregar uno nuevo:

### 1. Editar `src/model/datasetbuilder/dataset_builder.py`

```python
def get_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray, list, str]:
    """
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        target_name: Name of target column
    """
    match name:
        case 'tmt_ssrt':
            return self._build_tmt_ssrt()
        case 'nuevo_dataset':           # Agregar nuevo case
            return self._build_nuevo()
        case _:
            raise ValueError(f"Unknown dataset: {name}")

def _build_nuevo(self) -> Tuple[np.ndarray, np.ndarray, list, str]:
    target_name = 'mi_target'
    
    # Cargar datos...
    # Procesar...
    
    X = ...  # np.ndarray (n_samples, n_features)
    y = ...  # np.ndarray (n_samples,)
    feature_names = [...]  # list[str]
    
    return X, y, feature_names, target_name
```

### 2. Agregar a `src/config.py`

```python
DATASETS = [
    'tmt_ssrt',
    'nuevo_dataset',  # Agregar aquí
]
```

---

## Metodología

### Cross-Validation
- **Leave-One-Out (LOOCV):** Cada sujeto es usado una vez como test
- **Inner CV:** 10-fold para tuning de hiperparámetros (si está habilitado)

### Feature Selection
- **Método:** SelectKBest con f_regression o f_classif
- **k:** Máximo 20 features (configurable)

### Pipeline
```
Imputer → StandardScaler → [FeatureSelection] → Model
```

---

## Ejemplo Completo

```bash
# 1. Activar entorno
source venv/bin/activate

# 2. Verificar datos
ls data/hand_analysis/  # Debe tener carpetas con analysis.csv
ls data/sst_analysis/   # Debe tener carpetas con analysis.csv

# 3. Ejecutar regresión
python -m src.model.run_models --task regression

# 4. Ver resultados
cat results/regression/*/ssrt/tmt_ssrt/summary.csv
```

---

## Arquitectura

```
src/model/
├── run_models.py              # Pipeline principal
├── datasetbuilder/
│   └── dataset_builder.py     # Construcción de datasets (X, y, features, target)
├── shap/                      # Análisis SHAP (legacy, requiere actualización)
├── classification/
│   └── roc_curves.py          # Curvas ROC
└── permutation_tests.py       # Tests de permutación
```

### Flujo de datos

```
DatasetBuilder.get_dataset('tmt_ssrt')
       │
       ▼
  (X, y, feature_names, target_name)
       │
       ▼
  perform() → LOOCV + GridSearchCV
       │
       ▼
  save_results() → results/{task}/{timestamp}/{target}/{dataset}/
```
