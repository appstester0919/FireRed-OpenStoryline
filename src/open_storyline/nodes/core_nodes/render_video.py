import os
import sys
import tempfile
import time
import uuid
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
from open_storyline.utils.logging import MCPMoviePyLogger
from PIL import Image, ImageDraw, ImageFont, ImageOps
from open_storyline.utils.register import NODE_REGISTRY

# MoviePy import compatibility (v2 preferred)
try:
    from moviepy import (
        VideoFileClip,
        AudioFileClip,
        ImageClip,
        VideoClip,
        ColorClip,
        CompositeVideoClip,
        CompositeAudioClip,
        concatenate_videoclips,
        concatenate_audioclips,
        vfx,
    )
except Exception:  # pragma: no cover
    from moviepy.editor import (  # type: ignore
        VideoFileClip,
        AudioFileClip,
        ImageClip,
        VideoClip,
        ColorClip,
        CompositeVideoClip,
        CompositeAudioClip,
        concatenate_videoclips,
        concatenate_audioclips,
        vfx,
    )

from src.open_storyline.config import Settings
from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_state import NodeState
from open_storyline.nodes.node_schema import RenderVideoInput
from open_storyline.utils.util import get_video_rotation

# =============================================================================
# Constants
# =============================================================================

MILLISECONDS_PER_SECOND: float = 1000.0

MAX_MEDIA_DIMENSION_PX: int = 4096  # requirement: any media <=4096
DEFAULT_OUTPUT_MAX_DIMENSION_PX: int = 1080

DEFAULT_OUTPUT_ASPECT_RATIO: float = 16.0 / 9.0
BACKGROUND_COLOR_RGB: Tuple[int, int, int] = (0, 0, 0)
CENTER_POSITION = ("center", "center")

DEFAULT_OUTPUT_FPS: int = 25

SUBCLIP_END_SAFETY_MARGIN_S = 1e-3 # 

# Encoding
VIDEO_CODEC: str = "libx264"
AUDIO_CODEC: str = "aac"
FFMPEG_PARAMS: list[str] = ["-preset", "veryfast", "-crf", "23", "-threads", "0"]
TEMP_DIRECTORY_PREFIX: str = "render_video_"
TEMP_AUDIO_FILENAME: str = "temp-audio.m4a"

# Subtitle baseline
SUBTITLE_BASE_HEIGHT_PX: float = 1080.0
SUBTITLE_FONT_SIZE_AT_BASE: int = 40
SUBTITLE_FONT_SIZE_MIN: int = 28
SUBTITLE_FONT_SIZE_MAX: int = 120
SUBTITLE_FONT_COLOR: Tuple[int, int, int, int] = (255, 255, 255, 255)
SUBTITLE_MARGIN_BOTTOM_AT_BASE: int = 270
SUBTITLE_MARGIN_BOTTOM_MIN: int = 40
SUBTITLE_MARGIN_BOTTOM_MAX: int = 1040
SUBTITLE_STROKE_WIDTH_AT_BASE: int = 2
SUBTITLE_STROKE_WIDTH_MIN: int = 0
SUBTITLE_STROKE_WIDTH_MAX: int = 8
SUBTITLE_STROKE_COLOR: Tuple[int, int, int, int] = (0, 0, 0, 255)
SUBTITLE_MAX_WIDTH_RATIO: float = 0.90
SUBTITLE_PADDING_X: int = 20
SUBTITLE_PADDING_Y: int = 10

SOURCE_VIDEO_VOLUME_SCALE = 1.0
TTS_VOLUME_SCALE: float = 2.0
BGM_VOLUME_SCALE: float = 0.25
VIDEO_VOLUME_SCALE: float = 1.0
AUDIO_DURATION_TOLERANCE_SECONDS: float = 0.05
DEFAULT_CRF = 23


# =============================================================================
# Small utilities
# =============================================================================

def close_quietly(obj: Any) -> None:
    try:
        if obj is not None:
            obj.close()
    except Exception:
        pass


def milliseconds_to_seconds(value: Any) -> float:
    try:
        return float(value) / MILLISECONDS_PER_SECOND
    except Exception:
        return 0.0


def clamp_int(value: float, minimum: int, maximum: int) -> int:
    return int(max(minimum, min(maximum, round(value))))


def make_even(value: int) -> int:
    v = int(value)
    if v < 2:
        v = 2
    if v % 2 == 1:
        v -= 1
    return max(2, v)


def parse_aspect_ratio(value: Any) -> Optional[float]:
    """
    Accept:
      - "16:9"
      - float/int like 1.777
      - (w, h)
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        r = float(value)
        return r if r > 0 else None

    if isinstance(value, str):
        text = value.strip()
        if ":" in text:
            parts = text.split(":")
            if len(parts) == 2:
                try:
                    w = float(parts[0].strip())
                    h = float(parts[1].strip())
                    if w > 0 and h > 0:
                        return w / h
                except Exception:
                    return None
        else:
            try:
                r = float(text)
                return r if r > 0 else None
            except Exception:
                return None

    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            w = float(value[0])
            h = float(value[1])
            if w > 0 and h > 0:
                return w / h
        except Exception:
            return None

    return None


def resolve_output_canvas_size(inputs: Dict[str, Any]) -> Tuple[int, int]: 
    """
    Requirement:
      1) output aspect ratio decided by inputs
      2) output max dimension px decided by inputs
    """

    # Adaptively select the canvas size based on the proportion of the size of the material.
    def find_dominant_aspect_ratio(ratios):
        if not ratios:
            return None
        standard_ratios = [9/16, 3/4, 1.0, 4/3, 16/9]
        counts = [0] * len(standard_ratios)

        for r in ratios:
            idx = min(range(len(standard_ratios)), key=lambda i: abs(standard_ratios[i] - r))
            counts[idx] += 1

        # apply max count idx
        max_count = max(counts)
        max_count_idx = counts.index(max_count)

        return standard_ratios[max_count_idx]
    
    # Specify the aspect ratio and the longest side
    video_items = inputs.get('plan_timeline', {}).get("tracks", {}).get("video", []) or []
    ratio = (
        parse_aspect_ratio(inputs.get("aspect_ratio"))
        or find_dominant_aspect_ratio([item.get("size")[0] / item.get("size")[1] for item in video_items if item and item.get("size")])
        or DEFAULT_OUTPUT_ASPECT_RATIO
    )

    max_dim = inputs.get("output_max_dimension_px", DEFAULT_OUTPUT_MAX_DIMENSION_PX)
    try:
        max_dim = int(max_dim)
    except Exception:
        max_dim = DEFAULT_OUTPUT_MAX_DIMENSION_PX
    max_dim = max(2, min(MAX_MEDIA_DIMENSION_PX, max_dim))

    if ratio >= 1.0:
        width = max_dim
        height = max(2, int(round(width / ratio)))
    else:
        height = max_dim
        width = max(2, int(round(height * ratio)))

    return (make_even(width), make_even(height))


def build_media_id_to_path_map(load_media: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in (load_media.get("videos") or []) + (load_media.get("images") or []):
        media_id = item.get("media_id")
        path = item.get("path")
        if media_id and path:
            mapping[media_id] = path
    return mapping


def is_image_file(path: str) -> bool:
    try:
        return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    except Exception:
        return False


def make_mask_clip(mask: np.ndarray) -> ImageClip:
    # moviepy versions may accept is_mask / ismask
    try:
        return ImageClip(mask, is_mask=True)
    except TypeError:  # pragma: no cover
        return ImageClip(mask, ismask=True)


# =============================================================================
# Media cache: scale <=1080, drop alpha, and speed up images
# =============================================================================

class MediaCache:
    def __init__(
        self, 
        *, 
        include_video_audio: bool, 
        canvas_size: Tuple[int, int], 
        clip_compose_mode:str = "padding",
        bg_color: Tuple | List | None = None,
    ) -> None:
        self._include_video_audio = include_video_audio
        self._canvas_size = canvas_size
        self._clip_compose_mode = clip_compose_mode
        self._bg_color = tuple(bg_color) if bg_color else (0, 0, 0) # RGB

        self._video_sources: Dict[str, VideoFileClip] = {}
        self._audio_sources: Dict[str, AudioFileClip] = {}
        self._audio_to_close: List[AudioFileClip] = []

        # Key optimization: cache full-canvas RGB frames for images
        self._image_padded_frame_cache: Dict[str, np.ndarray] = {}
        self._video_size_cache: Dict[str, Tuple[int, int]] = {}

    def close(self) -> None:
        for v in self._video_sources.values():
            close_quietly(v)
        for a in self._audio_to_close:
            close_quietly(a)

    def get_audio(self, path: str) -> AudioFileClip:
        cached = self._audio_sources.get(path)
        if cached is not None:
            return cached
        clip = AudioFileClip(path)
        self._audio_sources[path] = clip
        self._audio_to_close.append(clip)
        return clip

    def get_video(self, path: str) -> VideoFileClip:
        cached = self._video_sources.get(path)
        if cached is not None:
            return cached

        src_w, src_h = self._probe_video_size(path)
        canvas_w, canvas_h = self._canvas_size

        # fit into canvas and <=1080
        max_w = min(canvas_w, MAX_MEDIA_DIMENSION_PX)
        max_h = min(canvas_h, MAX_MEDIA_DIMENSION_PX)

        if src_w > 0 and src_h > 0:
            scale = min(max_w / float(src_w), max_h / float(src_h))

            target_w = make_even(int(src_w * scale))
            target_h = make_even(int(src_h * scale))
            src_ratio = src_w / float(src_h)
            canvas_ratio = canvas_w / float(canvas_h)
        
        clip = VideoFileClip(path, audio=self._include_video_audio, target_resolution=(src_w, src_h))

        # maybe crop
        if self._clip_compose_mode == 'crop':
            x0, y0, x1, y1 = self.center_crop_calc((canvas_w, canvas_h), clip.size)
            clip = clip.cropped(x0, y0, x1, y1)
        
        # resize to canvas
        if src_ratio >= canvas_ratio:
            clip = clip.resized(width=target_w)
        else:
            clip = clip.resized(height=target_h)

        self._video_sources[path] = clip
        return clip

    def get_image(self, path: str) -> np.ndarray:
        cached = self._image_padded_frame_cache.get(path)
        if cached is not None:
            return cached

        canvas_w, canvas_h = self._canvas_size

        with Image.open(path) as image:
            try:
                image = ImageOps.exif_transpose(image)
            except Exception:
                pass

            # drop alpha channel: RGBA -> alpha composite on black -> RGB
            image = image.convert("RGBA")
            black_bg = Image.new("RGBA", image.size, (0, 0, 0, 255))
            try:
                image = Image.alpha_composite(black_bg, image).convert("RGB")
            except Exception:
                image = image.convert("RGB")
            
            # maybe crop
            if self._clip_compose_mode == 'crop':
                x0, y0, x1, y1 = self.center_crop_calc(self._canvas_size, image.size)
                image = image.crop(box=(x0, y0, x1, y1))

            # resize to fit (<=1080 and <=canvas)
            try:
                resample = Image.Resampling.LANCZOS
            except Exception:  # pragma: no cover
                resample = Image.LANCZOS  # type: ignore

            scale = min(canvas_w / float(image.width), canvas_h / float(image.height))
            image = image.resize((make_even(scale * image.width), make_even(scale * image.height)), resample=resample)
            resized = np.array(image, dtype=np.uint8)

        # build full-canvas frame (black + centered image)
        canvas = np.full((canvas_h, canvas_w, 3), fill_value=self._bg_color, dtype=np.uint8)
        h, w = resized.shape[0], resized.shape[1]
        x0 = max(0, (canvas_w - w) // 2)
        y0 = max(0, (canvas_h - h) // 2)
        x1 = min(canvas_w, x0 + w)
        y1 = min(canvas_h, y0 + h)
        canvas[y0:y1, x0:x1] = resized[0 : (y1 - y0), 0 : (x1 - x0)]

        self._image_padded_frame_cache[path] = canvas
        return canvas

    def _probe_video_size(self, path: str) -> Tuple[int, int]:
        cached = self._video_size_cache.get(path)
        if cached is not None:
            return cached

        w = h = 0
        # simplest: open once (metadata fetch), then close
        try:
            tmp = VideoFileClip(path, audio=False)
            w = int(getattr(tmp, "w", 0) or 0)
            h = int(getattr(tmp, "h", 0) or 0)
            close_quietly(tmp)
        except Exception:
            w, h = 0, 0

        self._video_size_cache[path] = (w, h)
        return w, h
    
    @staticmethod
    def center_crop_calc(canvas_size, media_size):
        # unpack sizes
        canvas_width, canvas_height = canvas_size
        media_width, media_height = media_size

        canvas_ratio = canvas_width / canvas_height
        media_ratio = media_width / media_height

        if media_ratio > canvas_ratio:
            # crop left and right
            crop_width = int(media_height * canvas_ratio)
            x1 = (media_width - crop_width) // 2
            return x1, 0, x1 + crop_width, media_height

        elif media_ratio < canvas_ratio:
            # crop top and bottom
            crop_height = int(media_width / canvas_ratio)
            y1 = (media_height - crop_height) // 2
            return 0, y1, media_width, y1 + crop_height

        else:
            # same ratio, no crop
            return 0, 0, media_width, media_height


# =============================================================================
# Subtitle renderer (RGB + mask; output frames are RGB, no alpha channel)
# =============================================================================

def _normalize_rgba(value: Any, default: Tuple[int, int, int, int] = (0, 0, 0, 255)) -> Tuple[int, int, int, int]:
    """Coerce a loosely-typed colour into an (r, g, b, a) int tuple.

    The LLM calls this node with colours in whatever shape it likes. Observed
    in a real MCP run: ``font_color=['255','255','255','255']`` (a list of
    *strings*) reached Pillow's ``multiline_text`` and raised
    ``TypeError: 'str' object cannot be interpreted as an integer`` -- which
    burned two tool retries before the agent happened to drop the argument.

    Accepts: (r,g,b[,a]) int tuple/list, list/tuple of numeric strings,
    "255,255,255,255" / "255,255,255" / "#FFFFFF" / "#FFFFFFFF" / "white",
    a single int (grey). Returns ``default`` when nothing sensible is given.
    """

    def _mk(nums: List[int]) -> Tuple[int, int, int, int]:
        if len(nums) == 3:
            nums = list(nums) + [255]
        clamped = [max(0, min(255, int(n))) for n in nums[:4]]
        while len(clamped) < 4:
            clamped.append(255)
        return (clamped[0], clamped[1], clamped[2], clamped[3])

    if value is None:
        return default
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("#"):
            hexs = s[1:]
            if len(hexs) in (6, 8):
                try:
                    nums = [int(hexs[i:i + 2], 16) for i in range(0, len(hexs), 2)]
                except ValueError:
                    return default
                return _mk(nums)
            return default
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
        else:
            parts = [s]
        nums: List[int] = []
        for p in parts:
            try:
                nums.append(int(float(p)))
            except ValueError:
                if p.lower() in _NAMED_COLORS:
                    r, g, b = _NAMED_COLORS[p.lower()]
                    nums.extend([r, g, b])
                else:
                    return default
        if len(nums) not in (3, 4):
            return default
        return _mk(nums)
    if isinstance(value, int):
        v = max(0, min(255, value))
        return (v, v, v, 255)
    if isinstance(value, (list, tuple)):
        nums = []
        for p in value:
            if isinstance(p, (int, float)):
                nums.append(int(p))
            elif isinstance(p, str):
                try:
                    nums.append(int(float(p.strip())))
                except ValueError:
                    return default
            else:
                return default
        if len(nums) not in (3, 4):
            return default
        return _mk(nums)
    return default


def _font_cmap(font_path: str) -> Optional[set]:
    """Return the set of codepoints the font actually has glyphs for.

    Uses fontTools' real ``cmap`` table -- no rasterisation heuristics. The
    rasterised-mask approach was tried first and was WRONG in both
    directions: it falsely flagged CP_Revenge as missing 與/彩/畫 (it has
    them), and falsely cleared 标小智无界黑 (it is missing 與/畫/測/試).
    A wrong fallback is worse than no fallback, so read the cmap.
    """
    if not font_path:
        return None
    cached = _FONT_CMAP_CACHE.get(font_path)
    if cached is not None:
        return cached
    try:
        from fontTools.ttLib import TTFont  # type: ignore[import-not-found]

        tt = TTFont(font_path, fontNumber=0, lazy=True)
        order = tt.getGlyphOrder()
        best = None
        for table in tt["cmap"].tables:
            if best is None or len(table.cmap) > len(best.cmap):
                best = table
        chars: Optional[set] = set()
        if best is not None:
            for codepoint, glyph_name in best.cmap.items():
                # A cmap entry that points at glyph 0 (.notdef) is NOT a real
                # glyph -- the character still renders as a hollow box.
                try:
                    if order.index(glyph_name) == 0:
                        continue
                except ValueError:
                    continue
                chars.add(codepoint)
        tt.close()
    except Exception:
        chars = None
    if chars is None:
        return None
    _FONT_CMAP_CACHE[font_path] = chars
    return chars


_FONT_CMAP_CACHE: dict = {}


def _mask_bytes(font: ImageFont.FreeTypeFont, ch: str) -> bytes:
    """Rasterise one character and return its raw mask bytes.

    Used to detect .notdef ("tofu") glyphs without needing a cmap parser:
    a missing character renders exactly like any other missing character.
    """
    mask = font.getmask(ch)
    return bytes(mask)  # type: ignore[arg-type]


_CJK_PROBE_CHARS = (
    "測試與畫面校正色彩呈現銳利飽和和階宣傳片教會香港靈糧堂主日崇拜"
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

_NAMED_COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
}


class PillowSubtitleRenderer:
    def __init__(self, font_path: str) -> None:
        self._font_path = font_path
        self._fallback_font_paths: List[str] = []

    def render(
        self,
        subtitle_items: List[Dict[str, Any]],
        *,
        video_size: Tuple[int, int],
        font_color: Tuple[int, int, int, int],
        **kwargs,
    ) -> List[ImageClip]:
        if not self._font_path:
            return []
        canvas_w, canvas_h = video_size
        scale = (canvas_h / SUBTITLE_BASE_HEIGHT_PX) if canvas_h > 0 else 1.0

        font_size: int = kwargs.get('font_size') or SUBTITLE_FONT_SIZE_AT_BASE
        margin_bottom: int = kwargs.get('margin_bottom') or SUBTITLE_MARGIN_BOTTOM_AT_BASE
        stroke_width: int = kwargs.get('stroke_width') or SUBTITLE_STROKE_WIDTH_AT_BASE
        stroke_color: Tuple = kwargs.get('stroke_color') or SUBTITLE_STROKE_COLOR
        font_size = clamp_int(font_size * scale, SUBTITLE_FONT_SIZE_MIN, SUBTITLE_FONT_SIZE_MAX)
        margin_bottom = clamp_int(
            margin_bottom * scale, SUBTITLE_MARGIN_BOTTOM_MIN, SUBTITLE_MARGIN_BOTTOM_MAX
        )
        stroke_width = clamp_int(
            stroke_width * scale, SUBTITLE_STROKE_WIDTH_MIN, SUBTITLE_STROKE_WIDTH_MAX
        )

        clips: List[ImageClip] = []
        for item in subtitle_items:
            text = str(item.get("text", "")).strip()
            tw = item.get("timeline_window", {}) or {}
            start_s = milliseconds_to_seconds(tw.get("start", 0.0))
            end_s = milliseconds_to_seconds(tw.get("end", 0.0))
            dur = end_s - start_s
            if not text or dur <= 0:
                continue

            clip = self._make_clip(
                text=text,
                start_s=start_s,
                end_s=end_s,
                video_size=video_size,
                font_size=font_size,
                font_color=font_color,
                margin_bottom=margin_bottom,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
            )
            if clip is not None:
                clips.append(clip)

        return clips

    def _make_clip(
        self,
        *,
        text: str,
        start_s: float,
        end_s: float,
        video_size: Tuple[int, int],
        font_size: int,
        font_color: Tuple[int, int, int, int],
        margin_bottom: int,
        stroke_width: int,
        stroke_color: Tuple[int, int, int, int],
    ) -> Optional[ImageClip]:
        canvas_w, canvas_h = video_size
        dur = end_s - start_s
        if dur <= 0:
            return None

        font = self._select_font(font_size, text)

        max_text_w = max(1, int(canvas_w * SUBTITLE_MAX_WIDTH_RATIO))
        wrapped = self._wrap_text_by_width(text, font, max_text_w)

        measure = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        draw = ImageDraw.Draw(measure)
        bbox = draw.multiline_textbbox(
            (0, 0), wrapped, font=font, align="center", stroke_width=stroke_width
        )
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        img_w = int(text_w + SUBTITLE_PADDING_X * 2)
        img_h = int(text_h + SUBTITLE_PADDING_Y * 2)

        rgba = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rgba)
        draw.multiline_text(
            (SUBTITLE_PADDING_X - bbox[0], SUBTITLE_PADDING_Y - bbox[1]),
            wrapped,
            font=font,
            fill=tuple(font_color),
            align="center",
            stroke_width=stroke_width,
            stroke_fill=tuple(stroke_color),
        )

        rgba_arr = np.array(rgba, dtype=np.uint8)
        rgb_arr = rgba_arr[:, :, :3]
        alpha_mask = (rgba_arr[:, :, 3].astype(np.float32) / 255.0)

        subtitle_clip = ImageClip(rgb_arr).with_mask(make_mask_clip(alpha_mask))
        y = max(0, int(canvas_h - margin_bottom - img_h))
        subtitle_clip = subtitle_clip.with_start(start_s).with_duration(dur).with_position(("center", y))
        return subtitle_clip

    def _load_font(self, font_size: int) -> ImageFont.FreeTypeFont:
        try:
            if self._font_path and os.path.exists(self._font_path):
                return ImageFont.truetype(font=self._font_path, size=font_size)
        except Exception:
            pass
        return ImageFont.load_default()

    @staticmethod
    def _font_covers(font: ImageFont.FreeTypeFont, text: str) -> bool:
        """True when the font has a real glyph for every character in ``text``.

        Primary check reads the font's real ``cmap`` table via fontTools.
        The rasterised-mask fallback below is only used when fontTools is
        unavailable -- and it is known to be unreliable (it can both invent
        missing glyphs and hide real ones), so it is never the deciding
        factor when a cmap is readable.
        """
        path = getattr(font, "path", None)
        cmap = _font_cmap(path) if path else None
        if cmap is not None:
            return all(ord(ch) in cmap for ch in set(text)
                       if ch not in ("\n", "\r", "\t"))
        try:
            reference = _mask_bytes(font, "\U000E0100")
        except Exception:
            return True
        for ch in set(text):
            if ch in ("\n", "\r", "\t"):
                continue
            try:
                if _mask_bytes(font, ch) == reference:
                    return False
            except Exception:
                return False
        return True

    def _select_font(self, font_size: int, text: str) -> ImageFont.FreeTypeFont:
        """Pick a font that can actually render ``text``.

        The recommended "花字" fonts are display faces; several of them only
        carry Simplified-Chinese glyphs, so traditional-Chinese subtitles
        (與 / 畫 / 測 / 試 / 靈 / 糧 ...) silently rendered as hollow boxes in
        a real MCP E2E run. Keep the recommended face whenever it covers the
        text, otherwise fall back to the first chain font that does.
        """
        primary = self._load_font(font_size)
        if self._font_covers(primary, text):
            return primary
        for path in self._fallback_font_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                candidate = ImageFont.truetype(font=path, size=font_size)
            except Exception:
                continue
            if self._font_covers(candidate, text):
                return candidate
        return primary

    @staticmethod
    def _wrap_text_by_width(text: str, font: ImageFont.FreeTypeFont, max_width_px: int) -> str:
        text = text.strip()
        if not text:
            return ""

        dummy = Image.new("RGB", (10, 10))
        draw = ImageDraw.Draw(dummy)

        lines: List[str] = []
        for paragraph in text.splitlines():
            paragraph = paragraph.strip()
            if not paragraph:
                lines.append("")
                continue

            current = ""
            for ch in paragraph:
                candidate = current + ch
                if draw.textlength(candidate, font=font) <= max_width_px or not current:
                    current = candidate
                else:
                    lines.append(current)
                    current = ch
            if current:
                lines.append(current)

        return "\n".join(lines)


# =============================================================================
# Audio composer (same behavior as before)
# =============================================================================

class AudioTrackComposer:
    def __init__(self, *, cache: MediaCache) -> None:
        self._cache = cache

    def compose(
        self,
        *,
        voiceover_items: List[Dict[str, Any]],
        bgm_items: List[Dict[str, Any]],
        video_audio_clip: AudioFileClip,
        final_duration_s: float,
        **kwargs,
    ):
        layers: List[Any] = []

        # video original audio
        if video_audio_clip is not None:
            video_volume_scale = kwargs.get('video_volume_scale') or VIDEO_VOLUME_SCALE
            video_layer = video_audio_clip.with_start(0).with_volume_scaled(video_volume_scale)

            if video_layer.duration is not None:
                src_end = max(0.0, min(video_layer.duration, final_duration_s) - SUBCLIP_END_SAFETY_MARGIN_S)
                video_layer = video_layer.subclipped(0, src_end)

            layers.append(video_layer)

        # voiceover
        for item in voiceover_items:
            path = item.get("path")
            if not path:
                continue
            src = self._cache.get_audio(path)

            sw = item.get("source_window", {}) or {}
            tw = item.get("timeline_window", {}) or {}

            src_start = milliseconds_to_seconds(sw.get("start", 0.0))
            src_end = milliseconds_to_seconds(sw.get("end", 0.0))

            src_end = max(src_start, src_end - SUBCLIP_END_SAFETY_MARGIN_S)
            src_end = self._clamp_end_to_duration(src, src_end)
            if src_end <= src_start:
                continue

            tl_start = milliseconds_to_seconds(tw.get("start", 0.0))
            tl_end = milliseconds_to_seconds(tw.get("end", 0.0))
            expected = max(0.0, tl_end - tl_start)

            max_available = max(0.0, src_end - src_start)
            expected = min(expected, max_available)

            if expected <= 0:
                continue

            sub_end = src_start + expected

            sub_end = self._clamp_end_to_duration(src, sub_end)
            if sub_end <= src_start:
                continue

            clip = src.subclipped(src_start, sub_end).with_start(tl_start)
            clip = clip.with_volume_scaled(kwargs.get('tts_volume_scale') or TTS_VOLUME_SCALE)
            layers.append(clip)

        # bgm (concat then loop/trim)
        if bgm_items:
            segments: List[Any] = []
            for item in bgm_items:
                path = item.get("path")
                if not path:
                    continue
                src = self._cache.get_audio(path)
                sw = item.get("source_window", {}) or {}
                src_start = milliseconds_to_seconds(sw.get("start", 0.0))
                src_end = milliseconds_to_seconds(sw.get("end", 0.0))
                
                src_end = max(src_start, src_end - SUBCLIP_END_SAFETY_MARGIN_S)
                src_end = self._clamp_end_to_duration(src, src_end)
                if src_end <= src_start:
                    continue
                segments.append(src.subclipped(src_start, src_end))

            if segments:
                bgm = concatenate_audioclips(segments).with_volume_scaled(kwargs.get('bgm_volume_scale') or BGM_VOLUME_SCALE)
                if bgm.duration is not None:
                    if bgm.duration < final_duration_s - AUDIO_DURATION_TOLERANCE_SECONDS:
                        bgm = self._loop_audio(bgm, final_duration_s)
                    elif bgm.duration > final_duration_s + AUDIO_DURATION_TOLERANCE_SECONDS:
                        bgm = bgm.subclipped(0, max(0.0, final_duration_s - SUBCLIP_END_SAFETY_MARGIN_S))
                layers.append(bgm)

        if not layers:
            return None
        return CompositeAudioClip(layers).with_duration(final_duration_s)

    @staticmethod
    def _loop_audio(audio_clip: Any, duration_s: float) -> Any:
        try:
            n = int(duration_s // audio_clip.duration) + 2
            looped = concatenate_audioclips([audio_clip] * max(1, n))
            return looped.subclipped(0, duration_s)
        except Exception:
            return audio_clip
    
    @staticmethod
    def _clamp_end_to_duration(clip: Any, end_s: float) -> float:
        duration = getattr(clip, "duration", None)
        if duration is None:
            return end_s
        return min(end_s, max(0.0, duration - SUBCLIP_END_SAFETY_MARGIN_S))


# =============================================================================
# Pipeline (non-overlapping visuals + subtitle overlay)
# =============================================================================

class RenderVideoPipeline:
    def __init__(self, *, server_cache_dir: Path, font_info_path: Path) -> None:
        self._server_cache_dir = server_cache_dir
        self.font_info_path = font_info_path
        with open(font_info_path, encoding='utf-8') as f:
            self.font_info = json.load(f)
        self._fontname2path = {font['font_name']: font['font_path'] for font in self.font_info}
        self._fallback_font_cache: Optional[List[str]] = None

    def _fallback_font_paths(self) -> List[str]:
        """Font chain used when the recommended face cannot render the text.

        Several recommended display faces only carry Simplified-Chinese
        glyphs (CP_Revenge, PingFangXingChenTi, HanChanShouZhuoTi, ...), so
        traditional-Chinese subtitles rendered as hollow boxes in a real MCP
        E2E run. The chain is ordered by CJK coverage: the packaged fonts
        that cover the most, then the system's own CJK fonts as a last
        resort.
        """
        if self._fallback_font_cache is not None:
            return self._fallback_font_cache

        packaged = [f.get("font_path") for f in self.font_info if f.get("font_path")]
        system = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        ]

        def score(path: str) -> int:
            cmap = _font_cmap(path)
            if cmap is None:
                return -1
            return sum(1 for ch in _CJK_PROBE_CHARS if ord(ch) in cmap)

        candidates = [p for p in packaged + system if p and os.path.exists(p)]
        ranked = sorted(candidates, key=score, reverse=True)
        self._fallback_font_cache = ranked
        return ranked

    async def render(self, *, node_state: NodeState, inputs: Dict[str, Any]) -> Dict[str, Any]:
        load_media: Dict[str, Any] = inputs["load_media"]
        tracks: Dict[str, Any] = (inputs.get("plan_timeline") or {}).get("tracks", {}) or {}
        transition_rec = inputs.get('transition_rec', [])
        text_rec = inputs.get('text_rec', [])
        # cut settings
        crf = inputs.get('crf', DEFAULT_CRF)
        clip_compose_mode = inputs.get('clip_compose_mode', 'crop')  # one of `padding` and `crop`
        bg_color = inputs.get('bg_color')
        font_color = _normalize_rgba(inputs.get('font_color'), SUBTITLE_FONT_COLOR)
        font_size = inputs.get('font_size')
        margin_bottom = inputs.get('margin_bottom')
        bgm_volume_scale = inputs.get('bgm_volume_scale')
        tts_volume_scale = inputs.get('tts_volume_scale')
        include_video_audio = inputs.get('include_video_audio')
        video_volume_scale = inputs.get('video_volume_scale')
        stroke_width = inputs.get('stroke_width')
        stroke_color = _normalize_rgba(inputs.get('stroke_color'), SUBTITLE_STROKE_COLOR)

        artifact_id: str = node_state.artifact_id
        session_id: str = node_state.session_id
        outputs_dir: Path = self._server_cache_dir / session_id / artifact_id
        outputs_dir.mkdir(parents=True, exist_ok=True)

        video_items = tracks.get("video", []) or []
        subtitle_items = tracks.get("subtitles", []) or []
        voiceover_items = tracks.get("voiceover", []) or []
        bgm_items = tracks.get("bgm", []) or []

        if not video_items:
            raise ValueError("timeline result has no video track")

        output_canvas_size = resolve_output_canvas_size(inputs)
        media_map = build_media_id_to_path_map(load_media)

        cache = MediaCache(
            include_video_audio=include_video_audio, 
            canvas_size=output_canvas_size, 
            clip_compose_mode=clip_compose_mode,
            bg_color=bg_color,
        )
        
        font_path = self._fontname2path.get(text_rec[0]['font_name']) if len(text_rec) > 0 else None
        subtitle_renderer = PillowSubtitleRenderer(font_path=font_path)
        # Fallback chain: several recommended display faces only carry
        # Simplified-Chinese glyphs, so traditional-Chinese subtitles would
        # render as hollow boxes. Offer every known font (plus the system's
        # CJK fonts) as fallbacks, best-covering first.
        subtitle_renderer._fallback_font_paths = list(self._fallback_font_paths())
        audio_composer = AudioTrackComposer(cache=cache)

        temp_dir = tempfile.mkdtemp(prefix=TEMP_DIRECTORY_PREFIX)
        output_name = f"output_{uuid.uuid4().hex[:8]}_{int(time.time() * 1000)}.mp4"
        output_path = str((outputs_dir / output_name).resolve())

        clips_to_close: List[Any] = []
        subtitle_clips: List[Any] = []
        base_clip = None
        final_clip = None

        try:
            final_duration_s = self._final_duration_seconds(video_items)

            # Build base video: only concat video track
            base_clip, clips_to_close, output_fps = self._build_base_video_concat(
                video_items=video_items,
                media_map=media_map,
                cache=cache,
                canvas_size=output_canvas_size,
                final_duration_s=final_duration_s,
                transition_rec=transition_rec
            )

            # Build subtitle: add subtitle track on base video while `subtitle_clips` is not empty.
            subtitle_clips = subtitle_renderer.render(
                subtitle_items, 
                video_size=output_canvas_size, 
                font_color=font_color,
                font_size=font_size,
                margin_bottom=margin_bottom,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
            )
            if subtitle_clips:
                final_clip = CompositeVideoClip([base_clip, *subtitle_clips]).with_duration(final_duration_s)
            else:
                final_clip: CompositeVideoClip = base_clip

            # Build audios: add music and tts track
            final_audio = audio_composer.compose(
                voiceover_items=voiceover_items,
                bgm_items=bgm_items,
                video_audio_clip=final_clip.audio if include_video_audio else None,
                final_duration_s=final_duration_s,
                tts_volume_scale=tts_volume_scale,
                bgm_volume_scale=bgm_volume_scale,
                video_volume_scale=video_volume_scale,
            )
            if final_audio is not None:
                final_clip = final_clip.with_audio(final_audio)

            loop = asyncio.get_running_loop()

            def report(progress: float, total: float | None, message: str | None):
                asyncio.run_coroutine_threadsafe(
                    node_state.mcp_ctx.report_progress(progress, total, message),
                    loop,
                )

            logger = MCPMoviePyLogger(report)
            
            FFMPEG_PARAMS[3] = f"{crf}" # set crf (video quality setting), default is 23 (medium quality)

            # Fast path: burn the already-rasterized subtitle PNGs with a single
            # ffmpeg filtergraph instead of compositing every frame in Python.
            # The MoviePy composite runs at ~1x realtime regardless of machine
            # speed; the filtergraph runs at ~15-20x. Falls back on any failure.
            rendered_with_filtergraph = False
            if subtitle_clips:
                try:
                    from open_storyline.utils.filtergraph_render import (
                        overlay_from_subtitle_clip,
                        burn_subtitles_with_ffmpeg,
                    )

                    overlays = []
                    for sub_clip in subtitle_clips:
                        ov = overlay_from_subtitle_clip(
                            sub_clip, temp_dir, canvas_size=output_canvas_size
                        )
                        if ov is None:
                            raise RuntimeError("subtitle clip not representable as a static overlay")
                        overlays.append(ov)

                    # The base video is encoded to a LOSSLESS intermediate so the
                    # final subtitle burn-in is the only lossy generation. A
                    # lossy intermediate would compound with the final encode
                    # and degrade the picture relative to the MoviePy path.
                    base_only_path = os.path.join(temp_dir, "base_video.mkv")
                    base_clip.write_videofile(
                        base_only_path,
                        fps=output_fps,
                        codec=VIDEO_CODEC,
                        audio=False,
                        remove_temp=True,
                        logger=None,
                        ffmpeg_params=[
                            "-preset", "ultrafast",
                            "-crf", "0",
                            "-threads", "0",
                        ],
                    )

                    audio_for_render = None
                    if final_clip is not None and getattr(final_clip, "audio", None) is not None:
                        candidate = os.path.join(temp_dir, TEMP_AUDIO_FILENAME)
                        try:
                            # codec must be explicit: MoviePy infers 'libfdk_aac' when left
                            # unset, and most ffmpeg builds (including this host's) do not have it.
                            final_clip.audio.write_audiofile(candidate, codec=AUDIO_CODEC, logger=None)
                            if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                                audio_for_render = candidate
                        except Exception as audio_exc:
                            # Losing the mixed audio track is recoverable; losing the render is not.
                            print(
                                f"[render_video] filtergraph path: audio export failed "
                                f"({audio_exc!r}); rendering video only",
                                file=sys.stderr,
                            )

                    burn_subtitles_with_ffmpeg(
                        base_only_path,
                        overlays,
                        output_path,
                        fps=int(output_fps),
                        crf=int(crf),
                        audio_source=audio_for_render,
                    )
                    rendered_with_filtergraph = True
                    print(
                        "[render_video] RENDERED_WITH_FFMPEG_FILTERGRAPH "
                        f"(base={os.path.basename(base_only_path)} "
                        f"overlays={len(overlays)} "
                        f"out={os.path.basename(output_path)})",
                        file=sys.stderr,
                        flush=True,
                    )
                    report(float(final_duration_s), float(final_duration_s),
                          "rendering 100.0%")
                except Exception as exc:  # pragma: no cover - fallback path
                    print(
                        f"[render_video] ffmpeg filtergraph render unavailable "
                        f"({exc!r}); falling back to MoviePy composite",
                        file=sys.stderr,
                    )

            if not rendered_with_filtergraph:
                await asyncio.to_thread(
                    final_clip.write_videofile,  # type: ignore[union-attr]
                    output_path,
                    codec=VIDEO_CODEC,
                    audio_codec=AUDIO_CODEC,
                    temp_audiofile=os.path.join(temp_dir, TEMP_AUDIO_FILENAME),
                    remove_temp=True,
                    fps=output_fps,
                    ffmpeg_params=FFMPEG_PARAMS,
                    logger=logger,
                )

            node_state.node_summary.info_for_user(
                f"Video generated successfully, duration: {final_duration_s} seconds, path: {output_path}",
                preview_urls=[output_path],
            )

            node_state.node_summary.info_for_llm(f"Video generated successfully, duration: {final_duration_s} seconds, path: {output_path}")
            return {
                "output_path": output_path,
                "output_basename": output_name,
                "duration_s": float(final_duration_s),
                "output_size": {"width": int(output_canvas_size[0]), "height": int(output_canvas_size[1])},
            }

        finally:
            for c in subtitle_clips:
                close_quietly(c)
            for c in clips_to_close:
                close_quietly(c)
            close_quietly(base_clip)
            close_quietly(final_clip)
            cache.close()

    @staticmethod
    def _final_duration_seconds(video_items: List[Dict[str, Any]]) -> float:
        end_ms = max(float((it.get("timeline_window") or {}).get("end", 0.0)) for it in video_items)
        return milliseconds_to_seconds(end_ms)

    def _build_base_video_concat(
        self,
        *,
        video_items: List[Dict[str, Any]],
        media_map: Dict[str, str],
        cache: MediaCache,
        canvas_size: Tuple[int, int],
        final_duration_s: float,
        transition_rec: List[Dict[str,Any]],
    ) -> Tuple[Any, List[Any], float]:
        # Force non-overlapping: concat only
        sorted_items = sorted(video_items, key=lambda x: float((x.get("timeline_window") or {}).get("start", 0.0)))

        clips: List[Any] = []
        clips_to_close: List[Any] = []
        current_time = 0.0

        def black_clip(duration: float) -> Any:
            c = ColorClip(size=canvas_size, color=BACKGROUND_COLOR_RGB).with_duration(max(0.0, duration))
            clips_to_close.append(c)
            return c
        
        for seg_idx, segment in enumerate(sorted_items):
            timeline_window = segment.get("timeline_window", {}) or {}
            start_s = milliseconds_to_seconds(timeline_window.get("start", 0.0))
            end_s = milliseconds_to_seconds(timeline_window.get("end", 0.0))
            expected_dur = max(0.0, end_s - start_s)
            if expected_dur <= 0:
                continue

            # fill gap
            if start_s > current_time:
                clips.append(black_clip(start_s - current_time))
                current_time = start_s

            seg_clip = RenderVideoPipeline._build_full_canvas_segment(
                segment=segment,
                media_map=media_map,
                cache=cache,
                canvas_size=canvas_size,
                expected_duration_s=expected_dur,
            )

            if seg_clip is None:
                clips.append(black_clip(expected_dur))
            else:
                clips.append(seg_clip)
                clips_to_close.append(seg_clip)

            current_time = start_s + expected_dur

        # trailing gap
        if final_duration_s > current_time:
            clips.append(black_clip(final_duration_s - current_time))

        if not clips:
            raise ValueError("no valid video segments")

        base = concatenate_videoclips(clips, method="chain").with_duration(final_duration_s)

        for transition in transition_rec:
            transition_type = transition.get('type', "")
            duration = transition.get('duration', 1000) / 1000 # ms -> s
            if transition.get('position', '') in ('opening', 'ending'):
                base = self._get_transition_clip(base, transition_type, duration)

        fps_values = [float(it.get("fps")) for it in video_items if it.get("fps")]
        output_fps = max(fps_values) if fps_values else float(getattr(base, "fps", None) or DEFAULT_OUTPUT_FPS)

        return base, clips_to_close, output_fps

    @staticmethod
    def _build_full_canvas_segment(
        *,
        segment: Dict[str, Any],
        media_map: Dict[str, str],
        cache: MediaCache,
        canvas_size: Tuple[int, int],
        expected_duration_s: float,
    ) -> Optional[Any]:
        source_path = segment.get("source_path") or media_map.get(segment.get("media_id"))
        if not source_path:
            return None

        if is_image_file(source_path):
            frame = cache.get_image(source_path)
            return ImageClip(frame).with_duration(expected_duration_s)

        # video
        source = cache.get_video(source_path)
        source_window = segment.get("source_window", {}) or {}

        src_start = milliseconds_to_seconds(source_window.get("start", 0.0))

        end_ms = source_window.get("end", None)
        if end_ms is None:
            src_end = float(getattr(source, "duration", 0.0) or 0.0)
        else:
            src_end = milliseconds_to_seconds(end_ms)

        if src_end <= src_start:
            return None
        
        # Clamp to avoid "end_time > duration" (common near EOF due to rounding/encoding)
        source_duration_s = float(getattr(source, "duration", 0.0) or 0.0)
        if source_duration_s > 0.0:
            if src_start >= source_duration_s:
                return None
            if src_end > source_duration_s:
                src_end = source_duration_s  # <= duration

        if src_end <= src_start:
            return None

        clip = source.subclipped(src_start, src_end)

        playback_rate = float(segment.get("playback_rate", 1.0) or 1.0)
        if playback_rate != 1.0:
            clip = clip.with_speed_scaled(playback_rate)

        clip_dur = float(getattr(clip, "duration", 0.0) or 0.0)
        if clip_dur > 0.0 and expected_duration_s > clip_dur:
            last_t = max(0.0, clip_dur - SUBCLIP_END_SAFETY_MARGIN_S)
            # Freeze video/mask at last frame for the remaining duration
            clip = clip.time_transform(
                lambda t, lt=last_t: np.minimum(t, lt),
                apply_to=["mask", "audio", "video"],
                keep_duration=True,
            )

        clip = clip.with_duration(expected_duration_s)

        # pad to full canvas (keep original "center on black" look)
        if hasattr(clip, "on_color"):
            clip = clip.on_color(size=canvas_size, color=cache._bg_color or BACKGROUND_COLOR_RGB, pos=CENTER_POSITION)
        else:  # pragma: no cover
            bg = ColorClip(size=canvas_size, color=cache._bg_color or BACKGROUND_COLOR_RGB).with_duration(expected_duration_s)
            clip = CompositeVideoClip([bg, clip.with_position(CENTER_POSITION)]).with_duration(expected_duration_s)

        return clip.with_duration(expected_duration_s)

    @staticmethod
    def _get_transition_clip(clip: VideoClip, transition_type="fade_in", transition_duration=1.0):
        
        all_transition = {
            "": clip,
            "fade_in": clip.with_effects([vfx.FadeIn(transition_duration)]),
            "fade_out": clip.with_effects([vfx.FadeOut(transition_duration)]),
        }

        return all_transition.get(transition_type, clip)


# =============================================================================
# Node entrypoint
# =============================================================================

@NODE_REGISTRY.register()
class RenderVideoNode(BaseNode):
    meta = NodeMeta(
        name="render_video",
        description="Render final video from the timeline",
        node_id="render_video",
        node_kind="render_video",
        require_prior_kind=["load_media", "plan_timeline", "transition_rec", "text_rec"],
        default_require_prior_kind=["load_media", "plan_timeline", "transition_rec", "text_rec"],
    )

    input_schema = RenderVideoInput

    def __init__(self, server_cfg: Settings) -> None:
        super().__init__(server_cfg)
        self._pipeline = RenderVideoPipeline(server_cache_dir=self.server_cache_dir,font_info_path=Path(server_cfg.recommend_text.font_info_path))

    async def default_process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        return await self.process(node_state, inputs)

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return await self._pipeline.render(node_state=node_state, inputs=inputs)