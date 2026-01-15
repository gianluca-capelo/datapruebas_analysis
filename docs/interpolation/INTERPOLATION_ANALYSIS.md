# Análisis de Interpolación de Trayectorias en TMT

## Resumen Ejecutivo

Este documento analiza el efecto de la interpolación de trayectorias en el cálculo de métricas del Trail Making Test (TMT). Se identifican problemas con el enfoque actual y se proponen alternativas.

**Conclusión principal**: La interpolación global a 60Hz (enfoque actual) **descarta todos los puntos originales** y los reemplaza con datos sintéticos, lo cual afecta significativamente las métricas de velocidad y la estructura por segmentos.

---

## 1. Contexto: ¿Por qué interpolar?

### Problema original

Los datos de cursor capturados en navegadores web tienen **timestamps irregulares**:

```
Ejemplo de datos reales:
t = [0, 15, 32, 48, 95, 112, 128, 145, 190, ...]  ms
     ↑   ↑   ↑   ↑   ↑
    15  17  16  47  17  ... gaps variables
```

Esto causa problemas en:

1. **Cálculo de velocidad**: `v = Δx/Δt` es muy sensible a variaciones en Δt
2. **Timestamps duplicados**: Browsers a veces disparan dos eventos en el mismo milisegundo
3. **Gaps largos**: Cuando el browser se congela, se generan gaps de 100-500ms

### Solución propuesta (actual)

Interpolar a una frecuencia fija (60Hz = 16.67ms entre puntos) para tener una tasa de muestreo uniforme.

---

## 2. Implementación Actual

### Código de interpolación

Ubicación: `/home/gianluca/Research/neurotask/neurotask/neurotask/tmt/preprocessing/interpolation.py`

```python
def interpolate_trajectory(
    x_coords: List[float],
    y_coords: List[float],
    timestamps: List[int],
    target_freq_hz: int = 60
) -> Tuple[List[float], List[float], List[int]]:

    # 1. Convertir a arrays
    x = np.array(x_coords, dtype=float)
    y = np.array(y_coords, dtype=float)
    t = np.array(timestamps, dtype=float)

    # 2. Eliminar timestamps duplicados
    t_unique, unique_indices = np.unique(t, return_index=True)
    x = x[unique_indices]
    y = y[unique_indices]
    t = t_unique

    # 3. Crear NUEVO eje temporal uniforme
    period_ms = 1000.0 / target_freq_hz  # 16.67ms para 60Hz
    t_new = np.arange(t[0], t[-1], period_ms)

    # 4. Interpolar X e Y a los nuevos timestamps
    x_new = np.interp(t_new, t, x)
    y_new = np.interp(t_new, t, y)

    # 5. Retornar SOLO los nuevos puntos (los originales se descartan)
    return x_new.tolist(), y_new.tolist(), t_new.astype(int).tolist()
```

### Problema crítico identificado

**Los puntos originales se descartan completamente.**

```
Antes de interpolar:
  t = [0, 47, 120]  ms  (3 puntos reales)
  x = [0, 100, 200] px

Después de interpolar a 60Hz:
  t_new = [0, 16.67, 33.33, 50, 66.67, 83.33, 100, 116.67, 120]
  x_new = [0, 35.5, 70.9, 106.4, 141.8, 177.3, 212.8, 248.2, 200]
                              ↑
                    El punto original (47, 100) ya NO existe
```

---

## 3. Impacto en las Métricas

### 3.1 Métricas afectadas

| Categoría | Métrica | Impacto | Dirección |
|-----------|---------|---------|-----------|
| **Velocidad** | `mean_speed` | ALTO | Disminuye (suavizado) |
| **Velocidad** | `peak_speed` | ALTO | Disminuye (picos diluidos) |
| **Velocidad** | `std_speed` | MEDIO | Disminuye (menos variabilidad) |
| **Aceleración** | `mean_acceleration` | ALTO | Disminuye (abs) |
| **Aceleración** | `peak_acceleration` | ALTO | Disminuye (abs) |
| **Distancia** | `total_distance` | BAJO | Sin cambio (<1%) |
| **Tiempo** | `rt` | BAJO | Cambio mínimo |
| **Segmentación** | `hesitation_time` | MEDIO-ALTO | Puede cambiar significativamente |
| **Segmentación** | `travel_avg_speed` | MEDIO | Suavizado |

### 3.2 Por qué la velocidad disminuye

Sin interpolación (datos originales):
```
t:  [0,   50,  120,  180]  ms  (irregular)
x:  [0,  100,  300,  350]  px

velocidad[0→1] = 100px / 50ms = 2.0 px/ms
velocidad[1→2] = 200px / 70ms = 2.86 px/ms  ← SALTO
velocidad[2→3] = 50px / 60ms  = 0.83 px/ms
```

Con interpolación a 60Hz:
```
t:  [0, 16.66, 33.33, 50, 66.66, ...] ms
x:  [0,  33.3,  66.6, 100, 140, ...] px (interpolados linealmente)

velocidad[0→1] = 33.3px / 16.66ms = 2.0 px/ms
velocidad[1→2] = 33.3px / 16.66ms = 2.0 px/ms
... (todas similares)
```

La interpolación lineal **suaviza** las velocidades porque:
1. Elimina gaps temporales grandes (donde había velocidades altas artificiales)
2. Genera más puntos intermedios con velocidades más uniformes
3. La derivada (velocidad) se vuelve más constante

### 3.3 Pérdida de estructura por segmentos

El TMT calcula métricas **por segmento** (entre targets consecutivos):

```
Sin interpolación:
  Segmento 1 (Target 1→2): t=[0, 45, 102]     → puntos reales del segmento
  Segmento 2 (Target 2→3): t=[102, 150, 210]  → puntos reales del segmento
  Límite exacto en t=102 (momento del touch)

Con interpolación:
  Todo el trial: t=[0, 16.67, 33.33, 50, ..., 100, 116.67, ...]
  El límite del segmento (t=102) ya NO es un punto exacto
  Se aproxima a t=100 o t=116.67
```

Esto afecta:
- `segment_speed` - calculada sobre puntos aproximados
- `speed_threshold` - basado en el 2do segmento, ahora con límites inexactos
- Clasificación Search/Travel/Hesitation - usa el threshold afectado

---

## 4. Alternativas de Interpolación

### 4.1 Enfoque actual: Grid global uniforme

```python
# Reemplaza TODOS los puntos con un grid uniforme
t_new = np.arange(t[0], t[-1], 16.67)
x_new = np.interp(t_new, t, x)
```

| Ventajas | Desventajas |
|----------|-------------|
| Tasa uniforme garantizada | Pierde TODOS los puntos originales |
| Simple de implementar | Límites de segmentos aproximados |
| Consistente entre trials | Suaviza excesivamente la velocidad |

### 4.2 Propuesta: Interpolación selectiva (solo gaps)

```python
def interpolate_gaps_only(x, y, t, max_gap_ms=50, target_freq_hz=60):
    """Interpola solo donde hay gaps > max_gap_ms."""
    x_new, y_new, t_new = [x[0]], [y[0]], [t[0]]
    period_ms = 1000.0 / target_freq_hz

    for i in range(1, len(t)):
        gap = t[i] - t[i-1]

        if gap > max_gap_ms:
            # Rellenar el gap con puntos interpolados
            n_points = int(gap / period_ms)
            for j in range(1, n_points):
                frac = j / n_points
                t_interp = t[i-1] + frac * gap
                x_interp = x[i-1] + frac * (x[i] - x[i-1])
                y_interp = y[i-1] + frac * (y[i] - y[i-1])
                t_new.append(t_interp)
                x_new.append(x_interp)
                y_new.append(y_interp)

        # SIEMPRE agregar el punto original
        t_new.append(t[i])
        x_new.append(x[i])
        y_new.append(y[i])

    return x_new, y_new, t_new
```

| Ventajas | Desventajas |
|----------|-------------|
| Preserva TODOS los puntos originales | Tasa no perfectamente uniforme |
| Límites de segmentos exactos | Requiere elegir `max_gap_ms` |
| Solo agrega datos donde es necesario | Más complejo de implementar |
| Menor impacto en métricas | |

### 4.3 Propuesta: Sin interpolación + filtrado de outliers

```python
def calculate_speed_with_outlier_filter(cursor_trail, max_speed_threshold=10):
    """Calcula velocidad descartando valores extremos."""
    speeds = []
    for i in range(1, len(cursor_trail)):
        dt = cursor_trail[i].time - cursor_trail[i-1].time
        if dt <= 0:
            continue
        dx = distance(cursor_trail[i].position, cursor_trail[i-1].position)
        speed = dx / dt
        if speed < max_speed_threshold:  # Descartar outliers
            speeds.append(speed)
    return np.mean(speeds), np.std(speeds)
```

| Ventajas | Desventajas |
|----------|-------------|
| Sin datos sintéticos | No resuelve gaps largos |
| Preserva estructura original | Velocidad sigue siendo variable |
| Simple | Requiere elegir threshold |

### 4.4 Comparación de enfoques

| Aspecto | Grid global | Solo gaps | Sin interpolación |
|---------|-------------|-----------|-------------------|
| Puntos originales | Descartados | Preservados | Preservados |
| Gaps cortos (<50ms) | Sobremuestreados | Sin cambio | Sin cambio |
| Gaps largos (>100ms) | Rellenados | Rellenados | Causan outliers |
| Límites de segmento | Aproximados | Exactos | Exactos |
| Velocidad | Muy suavizada | Levemente suavizada | Variable/ruidosa |
| Complejidad | Baja | Media | Baja |

---

## 5. Pregunta: ¿Se puede uniformizar borrando puntos?

**No es viable** para este caso. Borrar puntos solo puede hacer gaps **más grandes**, no más pequeños.

```
Original:   [0, 15, 32, 48, 95, 112]
Gaps:          15  17  16  47  17

Si borramos 32 y 48:
[0, 15, 95, 112] → gaps: 15, 80, 17  ← Peor, ahora hay un gap de 80ms
```

El submuestreo (borrar puntos) solo funciona si:
- La tasa original es **mayor** a la deseada (ej: 120Hz → 60Hz)
- **No hay gaps largos** en los datos

En datos de browser, el problema son los gaps largos ocasionales, no el sobremuestreo.

---

## 6. Recomendaciones

### Para análisis de métricas globales (ML)

Si el objetivo es usar las métricas como features para modelos de ML:

1. **Opción conservadora**: Usar interpolación selectiva (solo gaps)
   - Preserva puntos originales
   - Estabiliza cálculo de velocidad
   - Mantiene estructura de segmentos

2. **Opción alternativa**: Sin interpolación + features robustas
   - Usar medianas en lugar de medias
   - Calcular percentiles (p25, p75) en lugar de std
   - Descartar trials con gaps muy largos (>500ms)

### Para análisis clínico/comportamental

Si el objetivo es detectar patrones clínicos (hesitaciones, errores, etc.):

1. **NO usar interpolación global** - pierde micro-dinámicas importantes
2. **Preservar puntos originales** - los momentos de touch son clínicamente relevantes
3. **Interpolación selectiva** solo si es estrictamente necesario para cálculos

### Para comparación entre estudios

Si se necesita comparar con otros estudios que usan interpolación:

1. **Documentar claramente** el método usado
2. **Reportar ambas versiones** si es posible
3. **Validar** que los cambios están en rangos esperados (5-15% para velocidad)

---

## 7. Parámetros de Configuración

### Configuración actual (`src/config.py`)

```python
INTERPOLATE_TRAJECTORY = True/False  # Toggle global
```

### Configuración propuesta (futura)

```python
# Modo de interpolación
INTERPOLATION_MODE = "none" | "global" | "selective"

# Parámetros para modo "selective"
INTERPOLATION_MAX_GAP_MS = 50      # Solo interpolar gaps > 50ms
INTERPOLATION_TARGET_FREQ_HZ = 60  # Frecuencia objetivo

# Parámetros para validación
MAX_ALLOWED_GAP_MS = 500           # Marcar trial como inválido si gap > 500ms
```

---

## 8. Archivos Relevantes

| Archivo | Descripción |
|---------|-------------|
| `neurotask/.../preprocessing/interpolation.py` | Función de interpolación |
| `neurotask/.../metrics/speed_metrics.py` | Cálculo de velocidad |
| `neurotask/.../segmentation/segmentation.py` | Segmentación y clasificación |
| `datapruebas_analysis/src/config.py` | Configuración del pipeline |
| `datapruebas_analysis/src/data_analysis/compare_interpolation_analysis.py` | Script de comparación |

---

## 9. Próximos Pasos

1. **Completar auditoría empírica**: Comparar análisis con/sin interpolación usando el script mejorado
2. **Evaluar interpolación selectiva**: Implementar y comparar con enfoque actual
3. **Documentar decisión**: Elegir enfoque basado en resultados y documentar justificación
4. **Actualizar neurotask**: Si se decide cambiar el enfoque, actualizar la librería

---

## Apéndice A: Ejemplo Visual

### Trayectoria original vs interpolada

```
Original (5 puntos reales):
    •
     \
      •----•
            \
             •----•

Timestamps: [0, 45, 90, 150, 200] ms
Posiciones reales del cursor

Interpolada a 60Hz (13 puntos sintéticos):
    •
    .•
     .•
      •.
       .•--•
          ..•
            .•
             .•--•

Timestamps: [0, 16, 33, 50, 66, 83, 100, 116, 133, 150, 166, 183, 200] ms
Posiciones calculadas por interpolación lineal
```

### Impacto en velocidad

```
Original:
  Segmento 0→45ms:  v = 2.5 px/ms
  Segmento 45→90ms: v = 1.8 px/ms
  Segmento 90→150ms: v = 3.2 px/ms  ← Pico de velocidad
  Segmento 150→200ms: v = 2.0 px/ms

  mean_speed = 2.38 px/ms
  peak_speed = 3.2 px/ms

Interpolada:
  Todos los segmentos: v ≈ 2.1-2.3 px/ms (uniformes)

  mean_speed = 2.18 px/ms  (-8.4%)
  peak_speed = 2.35 px/ms  (-26.6%)
```

---

*Documento generado: 2026-01-14*
*Autor: Análisis automatizado con Claude*
