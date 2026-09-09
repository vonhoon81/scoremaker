#!/usr/bin/env python3
"""Cross-platform bits shared by every pipeline.

The pipelines themselves are pure numpy and PIL and do not care what they run on, but
three things do: where the source videos live, where to keep the scratch cache, and which
font can draw a Korean and Japanese title block. Those are here so there is one place to
fix rather than nine.

Import it from a pipeline with the two lines every `extract.py` already starts with:

    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(os.path.dirname(HERE))
    sys.path.insert(0, ROOT)
    import common
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))


def video(name):
    """A source video, which lives beside this file in the repo root."""
    return os.path.join(ROOT, name)


def out_pdf(name):
    """Where a finished score is written -- also the repo root."""
    return os.path.join(ROOT, name)


def work(name):
    """A scratch directory for cached scans and page composites.

    Under the system temp directory, so `/tmp/sm9` on Linux and something like
    `C:\\Users\\<you>\\AppData\\Local\\Temp\\sm9` on Windows. Deleting it forces a
    re-extract; keeping it makes re-running stitch.py or make_pdf.py free.
    """
    path = os.path.join(tempfile.gettempdir(), name)
    os.makedirs(path, exist_ok=True)
    return path


# Faces that carry both Korean and Japanese, best first. A variable font is preferred
# because one file then covers every weight; otherwise the bold sibling is looked up by
# name and, failing that, the regular face is reused rather than erroring out.
FONTS = [
    ("/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc", None),
    ("/usr/share/fonts/google-noto-sans-cjk-fonts/NotoSansCJK-Regular.ttc", None),
    ("/usr/share/fonts/naver-nanum-gothic-fonts/NanumGothic.ttf",
     "/usr/share/fonts/naver-nanum-gothic-fonts/NanumGothicBold.ttf"),
    ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"),
    ("C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/YuGothB.ttc"),
    ("C:/Windows/Fonts/msgothic.ttc", None),
    ("/System/Library/Fonts/AppleSDGothicNeo.ttc", None),
    ("/Library/Fonts/Arial Unicode.ttf", None),
]


def _faces():
    """(regular, bold) paths of the first installed face, or SCOREMAKER_FONT if set."""
    override = os.environ.get("SCOREMAKER_FONT")
    if override:
        return override, os.environ.get("SCOREMAKER_FONT_BOLD") or None
    for regular, bold in FONTS:
        if os.path.exists(regular):
            return regular, bold if bold and os.path.exists(bold) else None
    raise SystemExit(
        "no CJK font found -- install Noto Sans CJK or Nanum Gothic, or point "
        "SCOREMAKER_FONT at a .ttf/.ttc that covers Korean and Japanese")


def font(size, weight="Regular"):
    """A font for the title block, at `size` px, in "Regular" or "Bold".

    Pillow is imported here rather than at module scope so that download.py, which only
    needs the paths, does not require it -- `pip install yt-dlp curl_cffi` is enough to
    fetch a video on a bare machine.

    The named instance is selected even for "Regular": a variable font's default
    instance is not necessarily its Regular one, and leaving it at the default renders
    text visibly differently from every score built before this was factored out.
    """
    from PIL import ImageFont

    regular, bold = _faces()
    if weight == "Bold" and bold:
        return ImageFont.truetype(bold, size)
    f = ImageFont.truetype(regular, size)
    try:  # a variable font carries every weight in the one file
        f.set_variation_by_name(weight)
    except (OSError, AttributeError):
        pass  # a static face cannot be emboldened; the regular one will do
    return f


def utf8_console():
    """Make print() survive Japanese and Korean filenames on a legacy Windows console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
