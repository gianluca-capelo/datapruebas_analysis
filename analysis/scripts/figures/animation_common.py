"""Frame pacing, cursor marker and writer fallback shared by the animations."""
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path

# Classic mouse-pointer outline as a marker path. The hotspot (tip) sits at the
# origin so the marker points exactly at the cursor position. Markers are drawn
# in display space, so the arrow stays upright despite the inverted y axis.
MOUSE_CURSOR_MARKER = MarkerStyle(Path([
    (0.00, 0.00),
    (0.00, -1.00),
    (0.25, -0.75),
    (0.44, -1.13),
    (0.56, -1.06),
    (0.38, -0.69),
    (0.69, -0.69),
    (0.00, 0.00),
], closed=True))

CURSOR_SIZES = {"arrow": 22, "dot": 11}

DEFAULT_BITRATE = 2400


def cursor_style(cursor: str, scale: float = 1.0):
    """(marker, markersize) for the current-position highlight.

    The arrow needs a bigger size than the dot because its path spans about
    twice the unit radius of the "o" marker.
    """
    marker = MOUSE_CURSOR_MARKER if cursor == "arrow" else "o"
    return marker, round(CURSOR_SIZES[cursor] * scale)


def frame_point_counts(n_points: int, reveal_frames: int, hold_frames: int) -> list[int]:
    """Number of trail points visible at each frame.

    The first `reveal_frames` frames grow the trail linearly from one point to
    all of them; the trailing `hold_frames` keep the finished trail on screen.
    """
    counts = [max(1, round(n_points * (frame + 1) / reveal_frames))
              for frame in range(reveal_frames)]
    counts.extend([n_points] * hold_frames)
    return counts


def _register_bundled_ffmpeg():
    """Point matplotlib at the ffmpeg binary shipped with imageio-ffmpeg.

    Avoids requiring a system-wide install. No-op when the package is absent,
    in which case `resolve_writer` falls back to a GIF.
    """
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass


def resolve_writer(base: str, fmt: str, fps: int, bitrate: int = DEFAULT_BITRATE):
    """(output path, writer), falling back from mp4 to gif when ffmpeg is missing."""
    _register_bundled_ffmpeg()
    if fmt == "mp4" and animation.FFMpegWriter.isAvailable():
        return f"{base}.mp4", animation.FFMpegWriter(fps=fps, bitrate=bitrate)
    if fmt == "mp4":
        print("ffmpeg not available; falling back to GIF (pillow).")
    return f"{base}.gif", animation.PillowWriter(fps=fps)
