import matplotlib.pyplot as plt
from neurotask.tmt.model.tmt_model import TMTTrial


def plot_trial_simple(ax, trial: TMTTrial, target_radius: float, title: str = ""):
    """
    Plot a trial trajectory on an existing axis without segmentation.

    Args:
        ax: Matplotlib axis to plot on.
        trial: TMTTrial object.
        target_radius: Radius of targets in pixels.
        title: Title for the subplot.
    """
    if trial is None:
        ax.set_title(f"{title}\nNo encontrado")
        ax.axis('off')
        return

    x = [c.position.x for c in trial.cursor_trail]
    y = [c.position.y for c in trial.cursor_trail]

    ax.plot(x, y, 'b-', linewidth=0.5, alpha=0.7)
    ax.scatter(x, y, s=1, c='blue', alpha=0.3)
    ax.plot(x[0], y[0], 'go', markersize=5)
    ax.plot(x[-1], y[-1], 'ro', markersize=5)

    for target in trial.stimuli:
        circle = plt.Circle((target.position.x, target.position.y),
                           target_radius, fill=False, color='gray', linestyle='--')
        ax.add_patch(circle)
        ax.text(target.position.x, target.position.y, target.content,
               ha='center', va='center', fontsize=6)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(f"{title}\n({len(x)} puntos)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def is_pixel_coordinates(trial: TMTTrial) -> bool:
    """Detect if the trial data is in pixel coordinates.
    
    If coordinates are > 10, assume they're in pixels.
    Normalized coordinates are typically in range [-1, 1].
    """
    if trial.cursor_trail:
        first_pos = trial.cursor_trail[0].position
        return abs(first_pos.x) > 10 or abs(first_pos.y) > 10
    return False


def plot_with_labels_scatter(trial: TMTTrial, target_radius: float, labels: list[str], labels_title="Labels",
                                   title="Cursor tracking during a TMT trial", cmap_name='tab10', plot_start=False,
                                   legend_outside=False):

    # Validación
    cursor_trail = trial.get_cursor_trail_from_start()
    if len(labels) != len(cursor_trail):
        raise ValueError(
            "La cantidad de etiquetas debe coincidir con la cantidad de puntos en la trayectoria del cursor.")

    # Detectar si los datos ya están en píxeles
    pixel_coords = is_pixel_coordinates(trial)
    
    if pixel_coords:
        # Datos ya en píxeles (datapruebas y neuropruebas)
        cursor_coords = [(p.position.x, p.position.y) for p in cursor_trail]
        radius_px = target_radius
    else:
        # Coordenadas no esperadas - lanzar excepción
        first_pos = cursor_trail[0].position if cursor_trail else None
        raise ValueError(
            f"Unexpected coordinate format detected. First position: "
            f"({first_pos.x if first_pos else 'N/A'}, {first_pos.y if first_pos else 'N/A'}). "
            f"Expected pixel coordinates (values > 10)."
        )
    
    cursor_x, cursor_y = zip(*cursor_coords)

    # Asignar colores únicos a cada etiqueta
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap(cmap_name) if len(unique_labels) <= 10 else plt.get_cmap('tab20')
    label_to_color = {label: cmap(i % cmap.N) for i, label in enumerate(unique_labels)}
    point_colors = [label_to_color[label] for label in labels]

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 6))  # proporción 16:9
    ax.scatter(cursor_x, cursor_y, c=point_colors, s=20, zorder=4)

    # Dibujar los targets
    for target in trial.stimuli:
        tx, ty = target.position.x, target.position.y
        circle = plt.Circle((tx, ty), radius_px, color='steelblue', alpha=0.3, zorder=5)
        ax.add_patch(circle)
        ax.text(tx, ty, target.content, color='black', fontsize=8, ha='center', va='center', zorder=6)

    # Marcar el primer clic
    if plot_start and trial.start:
        sx, sy = trial.start.position.x, trial.start.position.y
        ax.scatter(sx, sy, color='cyan', edgecolor='black', s=100, marker='o', alpha=0.3, label='First Click', zorder=7)

    # Leyenda
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color, markersize=8)
               for label, color in label_to_color.items()]
    
    if legend_outside:
        # Colocar leyenda fuera del área del plot
        ax.legend(handles=handles, title=labels_title, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    else:
        ax.legend(handles=handles, title=labels_title)

    # Estética - ajustar límites basado en los datos
    ax.set_xlabel('X screen coordinate (pixels)')
    ax.set_ylabel('Y screen coordinate (pixels)')
    ax.set_title(title)
    
    # Calcular límites basados en los datos reales
    margin = 50
    x_min, x_max = min(cursor_x) - margin, max(cursor_x) + margin
    y_min, y_max = min(cursor_y) - margin, max(cursor_y) + margin
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # invertir eje Y (como en pantalla)
    ax.set_aspect('equal', adjustable='box')

    return fig
