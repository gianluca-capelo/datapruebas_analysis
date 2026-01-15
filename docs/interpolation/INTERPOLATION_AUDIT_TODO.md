# Auditoría de Interpolación - Pasos Pendientes

## Estado Actual

- **Análisis SIN interpolación**: COMPLETADO (`2026-01-14_10-04-55`)
- **Análisis CON interpolación**: PENDIENTE (fue interrumpido)
- **Script de comparación**: MEJORADO (con validación de configs y merge por trial_id)

## Pasos para Continuar

### 1. Generar análisis CON interpolación

```bash
# 1. Cambiar config.py
#    INTERPOLATE_TRAJECTORY = True
#    (ya está en False, hay que cambiarlo)

# 2. Correr análisis
source venv/bin/activate
python -m src.loader.analysis_loader

# 3. Guardar el timestamp del output (será algo como 2026-01-XX_XX-XX-XX)
```

### 2. Ejecutar comparación

```bash
# Reemplazar INTERP_TIMESTAMP con el timestamp del análisis con interpolación
python -m src.data_analysis.compare_interpolation_analysis \
    --analysis1 2026-01-14_10-04-55 \
    --analysis2 <INTERP_TIMESTAMP> \
    --name1 "Sin interpolación" \
    --name2 "Con interpolación" \
    --visualize
```

### 3. Interpretar resultados

**Métricas de velocidad** (mean_speed, peak_speed, etc.):
- Esperado: DISMINUYEN 5-15%
- Si >20%: Investigar trials específicos

**Métricas de distancia** (total_distance):
- Esperado: SIN CAMBIO (<1%)
- Si >1%: Posible bug

**Métricas de tiempo** (rt):
- Esperado: CAMBIO MÍNIMO (<0.5%)
- Si >1%: Revisar cálculo de RT

## Archivos Relevantes

| Archivo | Descripción |
|---------|-------------|
| [src/config.py](src/config.py) | Configuración - cambiar `INTERPOLATE_TRAJECTORY` |
| [src/data_analysis/compare_interpolation_analysis.py](src/data_analysis/compare_interpolation_analysis.py) | Script de comparación mejorado |
| [data/hand_analysis/2026-01-14_10-04-55/](data/hand_analysis/2026-01-14_10-04-55/) | Análisis SIN interpolación |

## Mejoras al Script de Comparación

Se agregaron las siguientes funciones:

1. **`validate_configs()`**: Verifica que ambos análisis tengan la misma configuración excepto `interpolate_trajectory`
2. **`compare_by_trial_id()`**: Hace merge por `(subject_id, trial_id)` para comparar trial por trial
3. **`calculate_trial_differences()`**: Calcula diferencias con percentiles (p5, p95)
4. **`find_problematic_trials()`**: Encuentra los trials con mayor diferencia
5. **`create_bland_altman_plot()`**: Genera Bland-Altman plots (gold standard para comparar métodos)

## Output Esperado

El script ahora muestra:
- Validación de configuraciones (verifica que solo difiera `interpolate_trajectory`)
- Comparación trial por trial con cobertura del match
- Top 20 métricas con mayor diferencia (con percentiles p5-p95)
- Top 5 trials problemáticos para las 3 métricas más afectadas
- Bland-Altman plots para las top 6 métricas

## Plan Completo

Ver [.claude/plans/piped-honking-honey.md](.claude/plans/piped-honking-honey.md) para el análisis conceptual completo de qué métricas deberían cambiar y por qué.
