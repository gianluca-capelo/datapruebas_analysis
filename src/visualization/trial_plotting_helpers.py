"""Helper functions for TMT trial plotting to avoid code duplication."""

import matplotlib.pyplot as plt
from neurotask.tmt.model.tmt_model import TMTTrial


def extract_cursor_coordinates(
    trial: TMTTrial,
) -> tuple[list[float], list[float]]:
    """
    Extract X/Y coordinates from cursor trail.

    Args:
        trial: TMTTrial object.

    Returns:
        Tuple of (x_coords, y_coords) lists.
    """
    cursor_trail = trial.get_cursor_trail_from_start()
    x = [c.position.x for c in cursor_trail]
    y = [c.position.y for c in cursor_trail]
    return x, y


def draw_trial_targets(
    ax,
    trial: TMTTrial,
    target_radius: float,
    circle_color: str = 'steelblue',
    circle_alpha: float = 0.3,
    circle_fill: bool = False,
    circle_linestyle: str = '-',
    text_fontsize: int = 8,
    text_color: str = 'black',
    zorder_circle: int = 5,
    zorder_text: int = 6,
    radius_multiplier: float = None,
    multiplier_color: str = 'orange',
    multiplier_alpha: float = 0.15
) -> None:
    """
    Draw target circles with content labels.

    Args:
        ax: Matplotlib axis to plot on.
        trial: TMTTrial object.
        target_radius: Radius of target circles in pixels.
        circle_color: Color of target circles.
        circle_alpha: Alpha transparency of circles.
        circle_fill: Whether to fill circles.
        circle_linestyle: Line style for circle border.
        text_fontsize: Font size for target labels.
        text_color: Color for target labels.
        zorder_circle: Z-order for circles.
        zorder_text: Z-order for text labels.
        radius_multiplier: If set, draws an additional circle showing the effective radius.
        multiplier_color: Color for the multiplier shadow circle.
        multiplier_alpha: Alpha transparency for the multiplier shadow.
    """
    for target in trial.stimuli:
        tx, ty = target.position.x, target.position.y

        # Draw multiplier shadow first (behind main circle)
        if radius_multiplier is not None and radius_multiplier != 1.0:
            effective_radius = target_radius * radius_multiplier
            shadow_circle = plt.Circle(
                (tx, ty),
                effective_radius,
                fill=True,
                color=multiplier_color,
                alpha=multiplier_alpha,
                zorder=zorder_circle - 1
            )
            ax.add_patch(shadow_circle)

        circle = plt.Circle(
            (tx, ty),
            target_radius,
            fill=circle_fill,
            color=circle_color,
            alpha=circle_alpha,
            linestyle=circle_linestyle,
            zorder=zorder_circle
        )
        ax.add_patch(circle)
        ax.text(
            tx, ty, target.content,
            color=text_color,
            fontsize=text_fontsize,
            ha='center',
            va='center',
            zorder=zorder_text
        )


def draw_trial_trajectory(
    ax,
    x: list[float],
    y: list[float],
    line_color: str = 'blue',
    linewidth: float = 0.5,
    line_alpha: float = 0.7,
    scatter: bool = True,
    scatter_size: int = 1,
    scatter_alpha: float = 0.3,
    zorder: int = 4,
    show_line: bool = True
) -> None:
    """
    Draw cursor trajectory line with optional scatter points.

    Args:
        ax: Matplotlib axis to plot on.
        x: List of x coordinates.
        y: List of y coordinates.
        line_color: Color for the trajectory line.
        linewidth: Width of trajectory line.
        line_alpha: Alpha transparency of line.
        scatter: Whether to draw scatter points.
        scatter_size: Size of scatter points.
        scatter_alpha: Alpha transparency of scatter points.
        zorder: Z-order for trajectory elements.
        show_line: Whether to draw the connecting line between points.
    """
    if show_line:
        ax.plot(x, y, color=line_color, linestyle='-', linewidth=linewidth, alpha=line_alpha, zorder=zorder)
    if scatter:
        ax.scatter(x, y, s=scatter_size, c=line_color, alpha=scatter_alpha, zorder=zorder)


def mark_start_end_points(
    ax,
    x: list[float],
    y: list[float],
    start_color: str = 'green',
    end_color: str = 'red',
    marker_size: int = 5,
    zorder: int = 7
) -> None:
    """
    Mark start and end points of trajectory.

    Args:
        ax: Matplotlib axis to plot on.
        x: List of x coordinates.
        y: List of y coordinates.
        start_color: Color for start point marker.
        end_color: Color for end point marker.
        marker_size: Size of markers.
        zorder: Z-order for markers.
    """
    ax.plot(x[0], y[0], 'o', color=start_color, markersize=marker_size, zorder=zorder)
    ax.plot(x[-1], y[-1], 'o', color=end_color, markersize=marker_size, zorder=zorder)


def configure_trial_axes(
    ax,
    x: list[float] = None,
    y: list[float] = None,
    margin: float = 50.0,
    invert_y: bool = True,
    aspect: str = 'equal',
    hide_ticks: bool = False,
    show_labels: bool = False,
    xlabel: str = 'X screen coordinate (pixels)',
    ylabel: str = 'Y screen coordinate (pixels)'
) -> None:
    """
    Configure axes for TMT trial visualization.

    Args:
        ax: Matplotlib axis to configure.
        x: List of x coordinates (for calculating limits).
        y: List of y coordinates (for calculating limits).
        margin: Margin around data for axis limits.
        invert_y: Whether to invert Y axis (screen coordinates).
        aspect: Aspect ratio mode.
        hide_ticks: Whether to hide axis ticks.
        show_labels: Whether to show axis labels.
        xlabel: Label for X axis.
        ylabel: Label for Y axis.
    """
    if x is not None and y is not None:
        x_min, x_max = min(x) - margin, max(x) + margin
        y_min, y_max = min(y) - margin, max(y) + margin
        ax.set_xlim(x_min, x_max)
        if invert_y:
            ax.set_ylim(y_max, y_min)
        else:
            ax.set_ylim(y_min, y_max)
    elif invert_y:
        ax.invert_yaxis()

    ax.set_aspect(aspect, adjustable='box')

    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    if show_labels:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
