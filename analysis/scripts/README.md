# `analysis/scripts`

Scripts que corren **después** del pipeline de análisis y del pipeline de ML.
Se dividen en dos grupos:

| Carpeta | Qué hace |
|---|---|
| `analysis/scripts/*.py` | Post-procesamiento de resultados de ML (concatenación, dispersión, consolidación, tamaños de muestra). |
| `analysis/scripts/figures/` | Generación de todas las figuras y animaciones de la tesis, el paper y la charla. |

Todos se invocan como módulo desde la raíz del repo, con el venv activo:

```bash
source venv/bin/activate
python -m analysis.scripts.<nombre>
python -m analysis.scripts.figures.<nombre>
```

Antes de correr cualquier cosa, leer [`analysis/docs/ANTES_DE_EMPEZAR.md`](../docs/ANTES_DE_EMPEZAR.md):
todos parten de los resultados ya generados en `data/*_analysis/` y `results/`. La mayoría
solo los lee; las excepciones, que recalculan sobre esos resultados, son las figuras de
SHAP (reentrenan el modelo en cada fold), las de permutación (rehacen la distribución nula)
y `compute_model_subset_sizes` (reconstruye los datasets).

---

## Figuras (`analysis/scripts/figures/`)

Todas escriben en **`analysis/figures/`** (se crea sola). Esa carpeta no está versionada.

### Convenciones comunes

- **`--lang {es,en}`** en todas las figuras salvo `shap_k_mean_slides`, que es solo en
  castellano. **El default es `es`** en todo el paquete. Las salidas en castellano llevan
  sufijo `_es`; las inglesas no llevan sufijo.
- **`--big-fonts`** (figuras de trial) agranda todas las fuentes ×1.4 y saca la leyenda
  fuera del plot, para proyectar. Agrega el sufijo `_big` al archivo.
- Formatos: las cuatro figuras estáticas de trial y `permutation_hist` guardan `.png` **y**
  `.pdf` vectorial; `permutation_dotplot` y `permutation_best`, `.png` y `.svg`; el resto
  solo `.png`. Las estáticas de trial van a 600 dpi, las animaciones a 150 y el resto a 300.
- Las animaciones producen `.mp4` (ffmpeg viene con `imageio-ffmpeg`, no hace falta
  instalarlo aparte) y caen automáticamente a `.gif` si no está disponible.

### 1. Figuras de un ensayo de cTMT

Las cinco muestran **el mismo ensayo**: la selección vive en `figures/trial_data.py`
(`select_trial`), así que cambiarla las mueve a todas a la vez.

| Script | Figura | Comando |
|---|---|---|
| `segmentation` | Trail coloreado por estado (duda / búsqueda / viaje). | `python -m analysis.scripts.figures.segmentation` |
| `speed` | Mismo ensayo coloreado por velocidad instantánea + colorbar. | `python -m analysis.scripts.figures.speed --units s` |
| `time` | Mismo ensayo coloreado por tiempo transcurrido. | `python -m analysis.scripts.figures.time` |
| `area` | `area_difference_from_ideal` sombreada contra la recta ideal. | `python -m analysis.scripts.figures.area --rectified` |
| `time_animation` | Animación del ensayo completándose, coloreado por tiempo. | `python -m analysis.scripts.figures.time_animation --seconds 12` |

Flags propios de cada una:

```bash
# Cualquiera de las cinco: forzar un ensayo concreto
python -m analysis.scripts.figures.speed --subject <subject_id> --trial <trial_id>

python -m analysis.scripts.figures.speed --cmap viridis --vmax-percentile 99
python -m analysis.scripts.figures.time  --units ms --cmap plasma

# area: destacar un segmento, y/o esconder uno que tapa al resto (no afecta la métrica)
python -m analysis.scripts.figures.area --segment B
python -m analysis.scripts.figures.area --exclude-segments D

# time_animation: duración, formato y marcador del cursor
python -m analysis.scripts.figures.time_animation --seconds 15 --hold 2 --fps 30 --format gif --cursor dot
```

### 2. Muestra y distribuciones

| Script | Figura | Comando |
|---|---|---|
| `demographics` | Panel 2×2 de la muestra completa (484 sujetos). | `python -m analysis.scripts.figures.demographics` |
| `demographics` `--presentation` | Los 4 paneles como PNG sueltos, sin títulos, fuentes de slide. | `python -m analysis.scripts.figures.demographics --presentation --stats` |
| `demographics_modeling` | Lo mismo pero restringido a los 368 sujetos que entran a los modelos. | `python -m analysis.scripts.figures.demographics_modeling --stats` |
| `violin` | Violines de rendimiento por tarea (6 paneles). | `python -m analysis.scripts.figures.violin` |

`--stats` superpone media y ±1 DE en el panel de edad (solo en las variantes de slide).

### 3. SHAP

| Script | Figura | Comando |
|---|---|---|
| `shap_main` | Panel 2×2 combinado (edad, K_mean, accuracy, c). | `python -m analysis.scripts.figures.shap_main` |
| `shap_panels` | Los mismos 4 paneles como imágenes sueltas y más grandes. | `python -m analysis.scripts.figures.shap_panels` |
| `shap_accuracy_ridge` | Accuracy con Ridge en vez de SVR (figura suplementaria). | `python -m analysis.scripts.figures.shap_accuracy_ridge` |
| `shap_k_mean_slides` | K_mean para proyectar: top 3 con nombres dentro de la barra, y el panel top 15. | `python -m analysis.scripts.figures.shap_k_mean_slides` |

Los pares modelo/dataset y sus timestamps están en `figures/shap_common.py`
(`COMBINATIONS`); ver [`analysis/docs/shap.md`](../docs/shap.md) para agregar uno nuevo.

**Ojo:** calcular SHAP re-entrena el modelo en cada fold LOO y tarda **minutos**.
Para iterar sobre el diseño sin recalcular:

```bash
python -m analysis.scripts.figures.shap_k_mean_slides --from-csv         # reusa shap_values_*.csv
python -m analysis.scripts.figures.shap_k_mean_slides --figure slide --override  # valores fijos del top 3
python -m analysis.scripts.figures.shap_k_mean_slides --figure panel --top-n-panel 10
```

### 4. Tests de permutación

Las distribuciones nulas **no están guardadas** en ningún lado: se recalculan a partir de
las predicciones almacenadas, con el mismo procedimiento que produjo los p-valores
publicados (se permuta `y_true` contra un `y_pred` fijo, seed 42, 1000 permutaciones).
Con los defaults (`--n-permutations 1000 --seed 42`, ambos overrideables), cada figura *es*
el test detrás del p-valor reportado.

**Ojo con `--timestamp`:** el default es `THESIS_RUN` (`2026-03-07_1213`), que contiene
`age`, `accuracy` y `c`. Los targets `K_mean` y `ssrt` viven en la otra corrida,
`2026-03-06_2028` (`THESIS_RUN_K_MEAN`), y hay que pasarla explícitamente o el script
falla con `FileNotFoundError`.

| Script | Figura | Comando |
|---|---|---|
| `permutation_hist` | Histograma de la nula de un modelo vs. su MAE observado. | `python -m analysis.scripts.figures.permutation_hist` |
| `permutation_dotplot` | Una fila por modelo: banda de su nula + punto en el MAE observado. `DummyRegressor` se excluye salvo `--keep-dummy`. | `python -m analysis.scripts.figures.permutation_dotplot` |
| `permutation_best` | Un panel por tarea con su mejor modelo (K_mean, accuracy, SSRT). | `python -m analysis.scripts.figures.permutation_best` |

```bash
# Otro target / modelo / unidad (con el timestamp que contiene ese target)
python -m analysis.scripts.figures.permutation_hist \
    --timestamp 2026-03-06_2028 --target ssrt --dataset tmt_ssrt --model SVR --unit ms
python -m analysis.scripts.figures.permutation_dotplot \
    --timestamp 2026-03-06_2028 --target K_mean --dataset tmt_k_mean --unit none

# Incluir el modelo trivial, cambiar el binning, o usar la última corrida disponible
python -m analysis.scripts.figures.permutation_dotplot --keep-dummy
python -m analysis.scripts.figures.permutation_hist --bins 30
python -m analysis.scripts.figures.permutation_hist --timestamp latest
```

Para la charla, las dos primeras comparten geometría con `--aligned`: el eje x cae en el
mismo píxel, así que al pasar de slide el histograma "se achata" sobre su fila del dot plot.

```bash
python -m analysis.scripts.figures.permutation_hist    --model SVR --aligned
python -m analysis.scripts.figures.permutation_dotplot --models SVR --aligned
```

### 5. Figuras conceptuales y de cierre

| Script | Figura | Comando |
|---|---|---|
| `no_free_lunch` | Esquema del teorema No Free Lunch (no usa datos). | `python -m analysis.scripts.figures.no_free_lunch --models` |
| `final_slide_animation` | Animación de cierre: trail sintético sobre targets con los hitos de la charla. | `python -m analysis.scripts.figures.final_slide_animation` |

```bash
python -m analysis.scripts.figures.final_slide_animation \
    --labels "cTMT,Ingeniería de atributos,ML,SHAP,Fin"      # 5 targets: layout curado
python -m analysis.scripts.figures.final_slide_animation \
    --labels "A,B,C,D,E,F" --layout random --seed 7          # cualquier otra cantidad
python -m analysis.scripts.figures.final_slide_animation \
    --colors gradient --cmap crest --trail node --fps 30 --dpi 150
```

`no_free_lunch` acepta además `--suffix` para no pisar variantes de la misma figura.

`--layout fixed` (default) solo tiene layouts curados para 5 y 7 targets; con cualquier
otra cantidad de etiquetas hay que usar `--layout random`.

### Módulos compartidos

No son ejecutables; concentran lo que antes estaba duplicado entre scripts.

| Módulo | Contenido |
|---|---|
| `_style.py` | `FIGURES_DIR`, DPIs, estilos (`use_science_style` / `use_slide_style`), `FontSizes`, `save_fig`, `--lang`. |
| `trial_data.py` | Selección y carga del ensayo de cTMT, y el fondo común (targets, ejes, colorbar). |
| `demographics_common.py` | Construcción de la tabla de metadatos, carga de las 4 tareas y los paneles demográficos. |
| `shap_common.py` | `COMBINATIONS`, `FEATURE_LABELS`, cálculo/lectura de SHAP y el panel de barras. |
| `permutation_common.py` | Carga de predicciones, test de permutación, formato de p-valores y paleta. |
| `animation_common.py` | Marcador de cursor, ritmo de cuadros y fallback mp4 → gif. |

---

## Post-procesamiento de ML (`analysis/scripts/`)

| Script | Qué hace | Comando |
|---|---|---|
| `concat_regression_results` | Concatena los `summary.csv` de un timestamp. `--timestamp` es obligatorio. | `python -m analysis.scripts.concat_regression_results --timestamp <ts>` |
| `add_dispersion_metrics` | Agrega métricas de dispersión y p-valores. Sin `--timestamp` usa el último. | `python -m analysis.scripts.add_dispersion_metrics --timestamp <ts>` |
| `consolidate_results` | Consolida varios timestamps en `analysis/results/consolidated/`. | `python -m analysis.scripts.consolidate_results <ts1> <ts2> ...` |
| `compute_model_subset_sizes` | Exporta el N de cada dataset a CSV: cTMT solo para `tmt_age`, la intersección cTMT ∩ tarea para el resto. | `python -m analysis.scripts.compute_model_subset_sizes --out <path.csv>` |

Los tres primeros aceptan `--output` y `compute_model_subset_sizes` acepta `--out` para
redirigir el CSV de salida.

`utils.py` guarda `get_latest_regression_timestamp()` y los timestamps de las corridas
reportadas en la tesis (`THESIS_RUN`, `THESIS_RUN_K_MEAN`), que usan las figuras de SHAP
y de permutación.
