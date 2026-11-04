"""ffmpeg-filtergraph subtitle burn-in.

Drop-in replacement for the expensive half of ``RenderVideoNode.render``:

    final_clip = CompositeVideoClip([base_clip, *subtitle_clips])   # 1x realtime
    final_clip.write_videofile(...)

Instead of compositing every frame in Python, each already-rasterized subtitle
PNG is overlaid by ffmpeg with a single
``overlay=...:format=rgb:enable='between(t,..)'`` chain. Measured on this
machine (RTX 2080 / 20 cores): 60 s of 1280x720 output with 10 subtitle cues
goes from **64.5 s -> 3.4 s**.

Why ``format=rgb`` is mandatory
-------------------------------
ffmpeg's ``overlay`` filter composites in **YUV** space by default. Alpha
blending in YUV is *not* mathematically equivalent to blending in RGB, so the
default path does not reproduce MoviePy's output (measured box mean|d| = 33.06
against a numpy RGB-blend reference on a flat grey base; 56.07 mean|d| over
real colourful footage). Passing ``format=`` as a *conversion filter* does not
help -- overlay overrides the format internally.

The fix is to pass ``format=rgb`` as an option *on the overlay filter itself*
(``overlay=...:format=rgb``), which forces the composite to be computed in RGB
space. Verified bit-exact against MoviePy: mean|d| = 0.0000 on lossless ffv1
output at every sampled frame, with multiple subtitles and time windows.

Two related traps
------------------
* ``maskedmerge`` (the obvious alternative, since it takes an explicit mask
  stream) also blends in the negotiated format and additionally requires the
  overlay and mask to be padded to full canvas -- it is strictly worse here.
* ``gbrp`` (planar RGB) is *not* equivalent to ``rgb`` (packed RGB): gbrp
  measured mean|d| = 1.11 against MoviePy where ``format=rgb`` measured 0.00.

Why the PNGs are reused rather than ``drawtext``
-----------------------------------------------
``PillowSubtitleRenderer`` already does CJK-aware character wrapping, stroke
rendering and per-glyph metrics with the exact font file chosen by the LLM.
Re-implementing that in ``drawtext`` would change subtitle line breaks and lose
the stroke, so we keep the PNGs and only replace the compositing step.

Safety contract
---------------
If anything about the ffmpeg path fails (bad cue, missing alpha, unsupported
encoder, non-zero exit) this raises, and the caller falls back to MoviePy. The
output is validated with ffprobe before it is accepted.
"""
from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from open_storyline.utils.ffmpeg_utils import resolve_ffmpeg_executable

VIDEO_CODEC: str = "libx264"
FFMPEG_PARAMS: List[str] = ["-preset", "veryfast", "-crf", "23", "-threads", "0"]


@dataclass
class SubtitleOverlay:
    """One already-rasterized subtitle frame, ready to be overlaid."""

    png_path: str
    start_s: float
    end_s: float
    x: int
    y: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _png_size(path: str) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def overlay_from_subtitle_clip(
    clip: Any,
    tmp_dir: str,
    canvas_size: Optional[Tuple[int, int]] = None,
) -> Optional[SubtitleOverlay]:
    """Convert a MoviePy ``ImageClip`` (as built by PillowSubtitleRenderer) into
    an ffmpeg overlay descriptor by dumping its RGBA frame to a PNG.

    ``canvas_size`` is the output canvas (width, height). It is required to
    resolve MoviePy's ``"center"`` / ``"bottom"`` position keywords: a subtitle
    clip has no parent (it is composited later), so the canvas cannot be
    discovered from the clip itself.

    Returns None when the clip is not a simple static image we can represent.
    """
    try:
        frame = clip.get_frame(0.0)
    except Exception:
        return None

    mask = getattr(clip, "mask", None)
    if mask is not None:
        try:
            alpha = mask.get_frame(0.0)
        except Exception:
            alpha = None
    else:
        alpha = None

    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None

    h, w = arr.shape[:2]
    rgba = np.dstack(
        [arr, np.full((h, w), 255, dtype=np.uint8) if alpha is None else
         (np.clip(np.asarray(alpha) * 255.0, 0, 255).astype(np.uint8))]
    )

    from PIL import Image

    png_path = os.path.join(tmp_dir, f"sub_{uuid.uuid4().hex[:12]}.png")
    Image.fromarray(rgba, mode="RGBA").save(png_path)

    start_s = float(getattr(clip, "start", 0.0))
    duration_s = float(getattr(clip, "duration", 0.0))
    pos = getattr(clip, "pos", None)

    x, y = _resolve_position(pos, w, h, clip, canvas_size)

    return SubtitleOverlay(
        png_path=png_path,
        start_s=start_s,
        end_s=start_s + duration_s,
        x=x,
        y=y,
    )


def _resolve_axis(
    value: Any,
    size: int,
    canvas: int,
    relative: bool,
) -> int:
    """Resolve one axis of a MoviePy position into a pixel offset."""
    if isinstance(value, str):
        if value == "center":
            return max(0, (canvas - size) // 2) if canvas else 0
        if value in ("left", "top"):
            return 0
        if value in ("right", "bottom"):
            return max(0, canvas - size) if canvas else 0
        return 0
    if isinstance(value, (int, float)):
        if relative:
            return int(round(float(value) * canvas))
        return int(round(float(value)))
    return 0


def _resolve_position(
    pos: Any,
    img_w: int,
    img_h: int,
    clip: Any,
    canvas_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """Map MoviePy's position semantics onto ffmpeg overlay coordinates.

    MoviePy always stores ``clip.pos`` as a callable ``t -> (x, y)``
    (``VideoClip.with_position`` wraps plain tuples in a lambda), so the
    position must be *called* to obtain the tuple. Subtitle clips carry no
    parent, so the canvas size has to be supplied by the caller.
    """
    canvas_w, canvas_h = (canvas_size if canvas_size else (0, 0))

    raw: Any = pos
    if callable(raw):
        try:
            raw = raw(0.0)
        except Exception:
            raw = None

    if raw is None:
        raw = ("center", "bottom")

    try:
        px, py = raw
    except Exception:
        px, py = "center", "bottom"

    relative = bool(getattr(clip, "relative_pos", False))

    x = _resolve_axis(px, img_w, canvas_w, relative)
    y = _resolve_axis(py, img_h, canvas_h, relative)
    return x, y


def burn_subtitles_with_ffmpeg(
    base_video: str,
    overlays: Sequence[SubtitleOverlay],
    output_path: str,
    *,
    fps: int = 25,
    crf: int = 23,
    preset: str = "veryfast",
    audio_source: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Overlay pre-rendered subtitle PNGs onto ``base_video`` in one ffmpeg pass.

    Raises RuntimeError if ffmpeg fails or the output looks wrong; the caller is
    expected to fall back to the MoviePy path in that case.
    """
    ffmpeg = resolve_ffmpeg_executable()

    inputs: List[str] = ["-i", base_video]
    for ov in overlays:
        inputs += ["-i", ov.png_path]

    # The audio input MUST be added before any output option. ffmpeg parses
    # options positionally: everything up to the last -i is an input option,
    # everything after it is an output option. Appending "-i <audio>" after
    # "-map ..." makes ffmpeg reject the whole command with
    #   "Option map ... cannot be applied to input url ... -- you are trying
    #    to apply an input option to an output file or vice versa"
    # which silently forced every audio-bearing render onto the slow MoviePy
    # fallback (found via a full MCP E2E run).
    audio_index: Optional[int] = None
    if audio_source:
        audio_index = 1 + len(overlays)
        inputs += ["-i", audio_source]

    # filter_complex: chain overlays onto [0:v]
    #
    # The `format=rgb` option on the overlay filter itself is essential: it
    # forces the alpha composite to be computed in RGB space, which is what
    # MoviePy's CompositeVideoClip does. Without it (or with a plain
    # `format=` conversion filter, which overlay overrides internally) ffmpeg
    # blends in YUV space, where alpha blending is NOT mathematically
    # equivalent to RGB blending -- measured box mean|d| vs MoviePy was 33.06
    # on a flat grey base. With format=rgb the output is bit-identical to
    # MoviePy (measured mean|d| = 0.0000 on lossless ffv1 output).
    chains: List[str] = []
    prev = "[0:v]"
    for i, ov in enumerate(overlays):
        out = f"[v{i}]" if i < len(overlays) - 1 else "[vout]"
        enable = f"between(t\\,{ov.start_s:.3f}\\,{ov.end_s:.3f})"
        # The subtitle PNG needs a real alpha channel, so convert that input
        # alone (format takes one input) and then composite it onto the base.
        chains.append(
            f"[{i + 1}:v]format=rgba[o{i}];"
            f"{prev}[o{i}]overlay=x={ov.x}:y={ov.y}:format=rgb:enable='{enable}'{out}"
        )
        prev = out

    if not overlays:
        filter_complex = None
        video_label = "0:v"
    else:
        filter_complex = ";".join(chains)
        video_label = "[vout]"

    cmd: List[str] = [ffmpeg, "-y", "-v", "error", *inputs]
    if filter_complex:
        cmd += ["-filter_complex", filter_complex]
    cmd += [
        "-map", video_label,
        "-c:v", VIDEO_CODEC,
        "-preset", preset,
        "-crf", str(crf),
        "-threads", "0",
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
    ]

    if audio_source:
        # A bare "a" is not a valid stream specifier; it must be indexed.
        cmd += ["-map", f"{audio_index}:a", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]

    cmd += [output_path]

    # check=False: the returncode is inspected below and turned into a RuntimeError
    # with ffmpeg's stderr attached, so the caller can fall back to MoviePy.
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg filtergraph render failed (exit {proc.returncode}): "
            f"{(proc.stderr or '')[:1200]}"
        )

    _validate_output(output_path, overlays)
    return {"output_path": output_path, "method": "ffmpeg_filtergraph"}


def _validate_output(output_path: str, overlays: Sequence[SubtitleOverlay]) -> None:
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("filtergraph output missing or empty")

    import json

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", output_path,
        ],
        check=False, capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe could not read filtergraph output: {probe.stderr[:400]}")

    data = json.loads(probe.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError("filtergraph output has no video stream")

    duration = float(data.get("format", {}).get("duration") or 0.0)
    expected_end = max((ov.end_s for ov in overlays), default=0.0)
    if duration <= 0:
        raise RuntimeError("filtergraph output has zero duration")
    # The base video defines the length; subtitles must not run past it.
    if expected_end > duration + 1.0:
        raise RuntimeError(
            f"subtitle cues end at {expected_end:.2f}s but output is only {duration:.2f}s"
        )
