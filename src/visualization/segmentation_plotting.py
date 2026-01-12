from typing import Tuple, List

import matplotlib.pyplot as plt
from neurotask.tmt.model.tmt_model import TMTTrial, CursorInfo, Coordinate
from neurotask.tmt.segmentation.segmentation import classify_cursor_positions_with_hesitation
from neurotask.tmt.crosses.crosses import calculate_crosses_for_trial

from src.visualization.trial_plotting_labels import plot_with_labels_scatter


def plot_segmentation(trial: TMTTrial, target_radius: float, speed_threshold: float, cmap_name='tab10'):
    segmentation: List[Tuple[str, CursorInfo]] = classify_cursor_positions_with_hesitation(trial,
                                                                                           target_radius,
                                                                                           speed_threshold)
    segmentation_labels = [label for (label, color) in segmentation]

    labels_title = 'Segmentation Labels'

    fig = plot_with_labels_scatter(trial, target_radius, segmentation_labels, labels_title=labels_title, cmap_name=cmap_name)

    return fig


def _segments_match(
    seg1: Tuple[Coordinate, Coordinate], 
    seg2: Tuple[Coordinate, Coordinate], 
    tolerance: float = 1e-6
) -> bool:
    """
    Verifica si dos segmentos son iguales (mismo par de coordenadas).
    
    Considera que dos segmentos son iguales si tienen las mismas coordenadas,
    independientemente del orden de los puntos.
    
    Args:
        seg1: Primer segmento como tupla de dos Coordinate
        seg2: Segundo segmento como tupla de dos Coordinate
        tolerance: Tolerancia numérica para comparación de floats
    
    Returns:
        True si los segmentos coinciden, False en caso contrario
    """
    # Comparar coordenadas con tolerancia pequeña
    # Verificar ambas direcciones (p1->p2 y p2->p1)
    match_forward = (
        abs(seg1[0].x - seg2[0].x) < tolerance and 
        abs(seg1[0].y - seg2[0].y) < tolerance and
        abs(seg1[1].x - seg2[1].x) < tolerance and 
        abs(seg1[1].y - seg2[1].y) < tolerance
    )
    
    match_reverse = (
        abs(seg1[0].x - seg2[1].x) < tolerance and 
        abs(seg1[0].y - seg2[1].y) < tolerance and
        abs(seg1[1].x - seg2[0].x) < tolerance and 
        abs(seg1[1].y - seg2[0].y) < tolerance
    )
    
    return match_forward or match_reverse


def plot_crosses_segmentation(
    trial: TMTTrial, 
    target_radius: float, 
    time_threshold: float = 500.0,
    cmap_name: str = 'tab10'
):
    """
    Visualiza un trial destacando los segmentos que participan en cruces.
    
    Similar a plot_segmentation, pero colorea los puntos según si pertenecen
    a segmentos que cruzan o no.
    
    Args:
        trial: Objeto TMTTrial
        target_radius: Radio de los targets en píxeles
        time_threshold: Umbral mínimo de tiempo entre segmentos para considerar cruce (ms)
        cmap_name: Nombre del colormap para los cruces
    
    Returns:
        matplotlib figure
    """
    # Calcular cruces
    num_crosses, cross_segments = calculate_crosses_for_trial(trial, time_threshold)
    
    # Obtener cursor trail
    cursor_trail = trial.get_cursor_trail_from_start()
    
    # Construir lista de segmentos del cursor trail
    trail_segments = []
    for i in range(len(cursor_trail) - 1):
        seg = (
            cursor_trail[i].position,
            cursor_trail[i + 1].position
        )
        trail_segments.append(seg)
    
    # Crear etiquetas: cada punto se etiqueta según si pertenece a un segmento que cruza
    labels = []
    
    for i, point in enumerate(cursor_trail):
        # Un punto puede pertenecer a máximo 2 segmentos (si no es el primero o último)
        is_in_cross = False
        cross_id = None
        
        # Verificar si el punto está en algún segmento que cruza
        # Verificar el segmento actual (desde este punto al siguiente)
        if i < len(trail_segments):
            current_seg = trail_segments[i]
            for cross_idx, (seg1, seg2, _) in enumerate(cross_segments):
                if _segments_match(current_seg, seg1) or _segments_match(current_seg, seg2):
                    is_in_cross = True
                    cross_id = cross_idx
                    break
        
        # Verificar el segmento anterior (desde el punto anterior a este)
        if not is_in_cross and i > 0:
            prev_seg = trail_segments[i - 1]
            for cross_idx, (seg1, seg2, _) in enumerate(cross_segments):
                if _segments_match(prev_seg, seg1) or _segments_match(prev_seg, seg2):
                    is_in_cross = True
                    cross_id = cross_idx
                    break
        
        if is_in_cross:
            labels.append(f"cross_{cross_id}")
        else:
            labels.append("no_cross")
    
    # Usar plot_with_labels_scatter para visualizar
    labels_title = f'Cross Segments ({num_crosses} crosses detected)'
    
    fig = plot_with_labels_scatter(
        trial, 
        target_radius, 
        labels, 
        labels_title=labels_title, 
        cmap_name=cmap_name,
        legend_outside=True  # Colocar leyenda fuera para no tapar el trial
    )

    return fig

