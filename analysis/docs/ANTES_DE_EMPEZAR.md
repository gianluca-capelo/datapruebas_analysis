# Antes de empezar

Los scripts y notebooks de `analysis/` no generan resultados de ML por si solos. Trabajan sobre los archivos que ya existen en `results/`. Este documento explica que tiene que estar disponible antes de usar cualquier herramienta de analisis.

## Prerequisito: resultados de regresion en `results/`

Todos los scripts (`consolidate_results`, `add_dispersion_metrics`, `generate_shap_figures`) y todos los notebooks (`ml_results/`, `metrics/`, `shap_analysis/`) leen datos de:

```
results/regression/{timestamp}/{target}/{dataset}/
├── config.json     # Configuracion del experimento
├── folds.csv       # Predicciones fold por fold (LOOCV)
└── summary.csv     # Metricas agregadas por modelo
```

Si esta carpeta no existe o esta vacia, nada en `analysis/` va a funcionar.

## Como generar los resultados

Desde la raiz del proyecto:

```bash
source venv/bin/activate
python -m src.model.run_models --task regression
```

Esto ejecuta el pipeline de ML completo (LOOCV con feature selection y tuning) y genera una carpeta con timestamp en `results/regression/`. Dependiendo de los datasets configurados en `src/config.py` (`DATASETS`), puede tardar varias horas.

Los datasets disponibles se configuran en `src/config.py`:

```python
DATASETS = ['tmt_ssrt', 'tmt_k6', 'tmt_k4', ...]
```

Cada dataset genera una subcarpeta `{target}/{dataset}/` dentro del timestamp.

## Estructura esperada

Un ejemplo real con tres timestamps, donde cada uno tiene targets distintos:

```
results/regression/
├── 2026-02-03_2051/          # Run 1
│   ├── ssrt/tmt_ssrt/
│   ├── K_6/tmt_k6/
│   ├── K_4/tmt_k4/
│   ├── K_mean/tmt_k_mean/
│   ├── sensibilidad/tmt_dprime/
│   └── age/tmt_age/
├── 2026-02-03_2053/          # Run 2 (mismos u otros targets)
│   └── ...
└── 2026-02-05_2206/          # Run 3
    └── ...
```

Para verificar que tenes resultados:

```bash
ls results/regression/
```

Y para ver el contenido de un timestamp especifico:

```bash
ls results/regression/<timestamp>/
```

## Que se puede hacer con los resultados

Una vez que tenes al menos un timestamp con resultados, podes usar los scripts y notebooks en este orden:

| Paso | Comando / Ubicacion | Que hace | Documentacion |
|------|---------------------|----------|---------------|
| 1 | `python -m analysis.scripts.consolidate_results <ts1> <ts2> ...` | Combina timestamps en un CSV unico con metricas de dispersion y p-values | [consolidate_results.md](consolidate_results.md) |
| 2 | Notebooks en `notebooks/ml_results/` | Analisis detallado por target (comparacion de modelos, residuos, permutaciones) | - |
| 3 | Notebooks en `notebooks/metrics/` | Comparacion de metricas de error y dispersion entre targets | - |
| 4 | `python -m analysis.scripts.generate_shap_figures` | Genera figuras de importancia SHAP | [shap.md](shap.md) |
| 5 | Notebook en `notebooks/shap_analysis/` | Exploracion interactiva de resultados SHAP | [shap.md](shap.md) |

> Los pasos 2-5 requieren haber ejecutado el paso 1 primero (excepto SHAP, que lee directamente de `results/`).

## Resumen rapido

```bash
# 1. Verificar que hay resultados
ls results/regression/

# 2. Consolidar (usar los timestamps que correspondan)
python -m analysis.scripts.consolidate_results 2026-02-03_2051 2026-02-03_2053 2026-02-05_2206

# 3. Abrir notebooks
jupyter lab analysis/notebooks/
```
