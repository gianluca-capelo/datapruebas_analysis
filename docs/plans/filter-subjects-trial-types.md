# Plan: Filtro de Sujetos por Cobertura de Trial Types (A/B)

## Resumen

Implementar un filtro en `DatasetBuilder` que excluya sujetos que no tengan **al menos 1 trial válido de tipo PART_A** y **al menos 1 trial válido de tipo PART_B**.

---

## 1. Análisis del Problema

### Estado Actual

En `src/model/datasetbuilder/dataset_builder.py`, el método `_aggregate_tmt()`:

1. Filtra trials con `is_valid == 'True'` (línea 147)
2. Hace pivot por `trial_type` (PART_A, PART_B) con `aggfunc='mean'` (líneas 162-167)
3. **No valida** que cada sujeto tenga trials de ambos tipos

**Consecuencia**: Si un sujeto solo tiene trials válidos de tipo A (o solo B), el pivot genera columnas con NaN para el tipo faltante. Estos NaN pueden:
- Propagarse silenciosamente al modelo
- Ser imputados incorrectamente
- Sesgar la validación cruzada

### Dónde Filtrar

**Recomendación: Filtrar en `_aggregate_tmt()`, después de pivotar pero antes de retornar.**

**Justificación:**
- **Después del pivot**: Ya tenemos la información agregada por sujeto y tipo, podemos detectar fácilmente NaN por columnas de tipo
- **Antes de retornar**: Garantiza que ningún código downstream reciba sujetos incompletos
- **Único punto de cambio**: Centralizado, no requiere modificar múltiples archivos
- **Trazabilidad**: Podemos loggear exactamente qué sujetos se excluyen y por qué

---

## 2. Implementación

### 2.1 Modificar `_aggregate_tmt()` en `dataset_builder.py`

```python
def _aggregate_tmt(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate TMT trial-level data to subject-level.

    Pivots by trial_type (PART_A, PART_B) and computes mean per subject.
    Excludes subjects without at least one valid trial of each type.

    Args:
        df: TMT DataFrame with trial-level data

    Returns:
        DataFrame with one row per subject and columns like 'rt_PART_A', 'rt_PART_B'
    """
    # Filter valid trials only (handle both bool and string 'True')
    df_valid = df[df['is_valid'].astype(str) == 'True'].copy()

    # === NUEVO: Validar cobertura de trial types por sujeto ===
    trial_type_counts = df_valid.groupby('subject_id')['trial_type'].value_counts().unstack(fill_value=0)

    # Verificar que existan ambas columnas (puede que un dataset no tenga ningún PART_B)
    required_types = ['PART_A', 'PART_B']
    for trial_type in required_types:
        if trial_type not in trial_type_counts.columns:
            trial_type_counts[trial_type] = 0

    # Identificar sujetos válidos (>=1 trial de cada tipo)
    valid_subjects_mask = (trial_type_counts['PART_A'] >= 1) & (trial_type_counts['PART_B'] >= 1)
    valid_subjects = trial_type_counts[valid_subjects_mask].index.tolist()
    excluded_subjects = trial_type_counts[~valid_subjects_mask].index.tolist()

    # Loggear exclusiones
    if excluded_subjects:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Excluding {len(excluded_subjects)} subjects without both PART_A and PART_B trials: "
            f"{excluded_subjects[:5]}{'...' if len(excluded_subjects) > 5 else ''}"
        )
        # Detalle de por qué fueron excluidos
        for subj in excluded_subjects[:5]:  # Log primeros 5
            counts = trial_type_counts.loc[subj]
            logger.debug(f"  Subject {subj}: PART_A={counts.get('PART_A', 0)}, PART_B={counts.get('PART_B', 0)}")

    # Filtrar DataFrame
    df_valid = df_valid[df_valid['subject_id'].isin(valid_subjects)]
    # === FIN NUEVO ===

    # Auto-detect numeric feature columns (exclude metadata)
    feature_cols = []
    for col in df_valid.columns:
        if col in self.EXCLUDE_COLS:
            continue
        # Try to convert to numeric
        numeric_col = pd.to_numeric(df_valid[col], errors='coerce')
        # Keep if at least 50% of values are numeric
        if numeric_col.notna().mean() > 0.5:
            df_valid[col] = numeric_col
            feature_cols.append(col)

    # Pivot by trial_type and aggregate with mean
    agg = df_valid.pivot_table(
        index='subject_id',
        columns='trial_type',
        values=feature_cols,
        aggfunc='mean'
    )

    # Flatten column names: (rt, PART_A) → rt_PART_A
    agg.columns = [f"{col}_{trial_type}" for col, trial_type in agg.columns]

    return agg.reset_index()
```

### 2.2 Agregar método auxiliar para reportes (opcional pero recomendado)

```python
def get_exclusion_report(self, df: pd.DataFrame) -> dict:
    """
    Generate a report of subjects that would be excluded due to missing trial types.

    Returns:
        dict with keys:
            - 'total_subjects': int
            - 'valid_subjects': int
            - 'excluded_subjects': list of subject_ids
            - 'exclusion_reasons': dict mapping subject_id to reason
    """
    df_valid = df[df['is_valid'].astype(str) == 'True'].copy()
    trial_type_counts = df_valid.groupby('subject_id')['trial_type'].value_counts().unstack(fill_value=0)

    for trial_type in ['PART_A', 'PART_B']:
        if trial_type not in trial_type_counts.columns:
            trial_type_counts[trial_type] = 0

    exclusion_reasons = {}
    for subj in trial_type_counts.index:
        part_a = trial_type_counts.loc[subj, 'PART_A']
        part_b = trial_type_counts.loc[subj, 'PART_B']
        if part_a == 0 and part_b == 0:
            exclusion_reasons[subj] = "No valid trials (PART_A=0, PART_B=0)"
        elif part_a == 0:
            exclusion_reasons[subj] = f"Missing PART_A (PART_B={part_b})"
        elif part_b == 0:
            exclusion_reasons[subj] = f"Missing PART_B (PART_A={part_a})"

    valid_mask = (trial_type_counts['PART_A'] >= 1) & (trial_type_counts['PART_B'] >= 1)

    return {
        'total_subjects': len(trial_type_counts),
        'valid_subjects': valid_mask.sum(),
        'excluded_subjects': list(exclusion_reasons.keys()),
        'exclusion_reasons': exclusion_reasons
    }
```

---

## 3. Edge Cases

| Caso | Comportamiento |
|------|----------------|
| Sujeto con múltiples A, solo 1 B | ✅ Incluido (cumple requisito mínimo) |
| Sujeto con 1 A, múltiples B | ✅ Incluido |
| Sujeto con trials A y B pero todos inválidos | ❌ Excluido (filtro `is_valid` los elimina primero) |
| Sujeto pierde trial por filtro posterior | No aplica: el filtro de cobertura se hace **después** del filtro de validez |
| Dataset sin ningún PART_B | ⚠️ Todos los sujetos excluidos + warning |

### Orden de Filtros (Correcto)

1. **Filtro de validez** (`is_valid == 'True'`) - ya existente
2. **Filtro de cobertura A/B** (nuevo) - sobre trials ya validados
3. **Pivot y agregación** - solo con sujetos válidos

---

## 4. Validación y Checks

### 4.1 Checks Automáticos (Agregar al código)

```python
# Después del filtro, verificar integridad
assert len(valid_subjects) > 0, "No subjects remain after trial type coverage filter"
assert df_valid['subject_id'].nunique() == len(valid_subjects), "Subject count mismatch"
```

### 4.2 Checks en Tests (Agregar a `test_dataset_builder.py`)

```python
def test_excludes_subjects_without_both_trial_types():
    """Verify that subjects missing PART_A or PART_B are excluded."""
    builder = DatasetBuilder()

    # Crear DataFrame de prueba con sujeto incompleto
    test_df = pd.DataFrame({
        'subject_id': ['S1', 'S1', 'S2', 'S2', 'S3'],
        'trial_type': ['PART_A', 'PART_B', 'PART_A', 'PART_A', 'PART_B'],
        'is_valid': ['True', 'True', 'True', 'True', 'True'],
        'rt': [100, 200, 150, 160, 250]
    })

    result = builder._aggregate_tmt(test_df)

    # S1 tiene ambos tipos → incluido
    # S2 solo tiene PART_A → excluido
    # S3 solo tiene PART_B → excluido
    assert 'S1' in result['subject_id'].values
    assert 'S2' not in result['subject_id'].values
    assert 'S3' not in result['subject_id'].values
    assert len(result) == 1
```

### 4.3 Logging de Métricas

El código propuesto ya incluye logging de:
- Cantidad de sujetos excluidos
- IDs de sujetos excluidos (primeros 5)
- Razón de exclusión por sujeto (en nivel DEBUG)

---

## 5. Impacto Downstream

### 5.1 Validación A/B
- **Positivo**: Garantiza que cada sujeto tenga datos de ambas partes del TMT
- **Consistencia**: Las features `*_PART_A` y `*_PART_B` nunca tendrán NaN por falta de datos

### 5.2 Entrenamiento de Modelos
- **Positivo**: Datasets más limpios, sin imputaciones espurias
- **Posible reducción de N**: Menos sujetos si muchos son excluidos
- **Reproducibilidad**: Mismo filtro siempre excluye los mismos sujetos

### 5.3 Comparabilidad de Métricas
- **Positivo**: Métricas calculadas sobre sujetos completos
- **Trazabilidad**: El reporte de exclusión documenta exactamente quién fue excluido

### 5.4 Riesgos

| Riesgo | Mitigación |
|--------|------------|
| Reducción excesiva de N | Monitorear ratio excluidos/totales; si >20%, revisar calidad de datos upstream |
| Sesgo de selección | Documentar características de excluidos vs incluidos |
| Filtro demasiado tarde | Se aplica inmediatamente después de validez, antes de cualquier agregación |

---

## 6. Archivos a Modificar

| Archivo | Cambio |
|---------|--------|
| `src/model/datasetbuilder/dataset_builder.py` | Agregar filtro en `_aggregate_tmt()` + método `get_exclusion_report()` |
| `src/model/datasetbuilder/test_dataset_builder.py` | Agregar test de exclusión |

---

## 7. Verificación Post-Implementación

1. **Ejecutar tests existentes**:
   ```bash
   python -m pytest src/model/datasetbuilder/test_dataset_builder.py -v
   ```

2. **Ejecutar test de dataset builder**:
   ```bash
   python -m src.model.datasetbuilder.test_dataset_builder
   ```

3. **Verificar logging**:
   - Configurar logging level a WARNING
   - Ejecutar pipeline completo
   - Verificar que se loggean exclusiones si las hay

4. **Validar integridad del dataset**:
   ```python
   builder = DatasetBuilder()
   X, y, features, target = builder.get_dataset('tmt_ssrt')

   # Verificar que no hay NaN en features de tipo
   part_a_cols = [f for f in features if '_PART_A' in f]
   part_b_cols = [f for f in features if '_PART_B' in f]
   assert not np.isnan(X[:, [features.index(c) for c in part_a_cols]]).any()
   assert not np.isnan(X[:, [features.index(c) for c in part_b_cols]]).any()
   ```
