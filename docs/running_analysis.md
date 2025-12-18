# Ejecución de Análisis

Este documento explica cómo ejecutar los análisis de todos los experimentos neuropsicológicos y dónde se guardan los resultados.

## Ejecución Rápida

Para ejecutar todos los análisis de una sola vez:

```bash
python -m src.runner.run_all_analysis
```

Este comando ejecutará secuencialmente:
1. TMT (Trail Making Test)
2. SST (Stop Signal Task)
3. CDT (Change Detection Task)
4. Go/No-Go

## Análisis Individuales

También podés ejecutar cada análisis por separado:

```bash
# TMT
python -m src.loader.analysis_loader

# SST
python -m src.loader.sst_analysis_loader

# CDT
python -m src.loader.cdt_analysis_loader

# Go/No-Go
python -m src.loader.gonogo_analysis_loader
```

## Descripción de las Tareas

| Tarea | Descripción | Métricas Principales |
|-------|-------------|---------------------|
| **TMT** | Trail Making Test - Evaluación de velocidad de procesamiento y flexibilidad cognitiva | Tiempo de ejecución, errores |
| **SST** | Stop Signal Task - Control inhibitorio y tiempo de reacción | SSRT, SSD, Go RT |
| **CDT** | Change Detection Task - Memoria de trabajo visual | Cowan's K (K_4, K_6), accuracy |
| **Go/No-Go** | Control inhibitorio simple | Hit Rate, False Alarm, c, sensibilidad |

## Estructura de Resultados

Cada análisis guarda sus resultados en una carpeta con timestamp:

```
data/
├── hand_analysis/<timestamp>/        # TMT
│   ├── hand_analysis.csv             # Métricas por sujeto
│   └── configuration.json            # Configuración del análisis
│
├── sst_analysis/<timestamp>/         # SST
│   ├── sst_analysis.csv
│   └── configuration.json
│
├── cdt_analysis/<timestamp>/         # CDT
│   ├── cdt_analysis.csv
│   └── configuration.json
│
└── gonogo_analysis/<timestamp>/      # Go/No-Go
    ├── gonogo_analysis.csv
    └── configuration.json
```

### Archivos Generados

| Archivo | Contenido |
|---------|-----------|
| `*_analysis.csv` | DataFrame con métricas calculadas por sujeto |
| `configuration.json` | Metadata del análisis: timestamp, git commit, cantidad de sujetos, paths |

## Acceso Programático

Desde Python, podés cargar el análisis más reciente sin re-ejecutar:

```python
# TMT
from src.loader.analysis_loader import load_analysis
df, path = load_analysis(random_state=78, eval_size=1, split=False, old_split_config_date=None)

# SST
from src.loader import get_latest_sst_analysis
df, config = get_latest_sst_analysis()

# CDT
from src.loader import get_latest_cdt_analysis
df, config = get_latest_cdt_analysis()

# Go/No-Go
from src.loader import get_latest_gonogo_analysis
df, config = get_latest_gonogo_analysis()
```

## Notebooks de Exploración

Para análisis exploratorio de cada tarea:

- `notebooks/sst_analysis.ipynb` - SST
- `notebooks/cdt_analysis.ipynb` - CDT
- `notebooks/gonogo_analysis.ipynb` - Go/No-Go

## Datos de Entrada

Los datos crudos se encuentran en:

```
data/raw/
├── tmt/                              # Trail Making Test
│   ├── datapruebas/
│   │   ├── metadata/
│   │   │   └── metadata.csv
│   │   └── subjects/
│   │       └── *.json
│   └── neuropruebas/
│       ├── metadata/
│       │   └── metadata.csv
│       └── subjects/
│           └── *.csv
│
├── sst/                              # Stop Signal Task
│   ├── datapruebas/
│   │   └── *.csv
│   └── neuropruebas/
│       └── *.csv
│
├── cdt/                              # Change Detection Task
│   ├── datapruebas/
│   │   └── *.csv
│   └── neuropruebas/
│       └── *.csv
│
└── gonogo/                           # Go/No-Go
    ├── datapruebas/
    │   └── *.csv
    └── neuropruebas/
        └── *.csv
```

