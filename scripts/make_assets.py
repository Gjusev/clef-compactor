#!/usr/bin/env python3
"""Render the visual assets: logo, social preview and the brag video.

Editorial style per the repo's design language: warm monochrome canvas,
charcoal ink, muted pastel accents, serif display type, mono for data.
Deterministic and dependency-light: PIL for frames, ffmpeg for the encode.

Usage::

    python scripts/make_assets.py            # logo + social preview
    python scripts/make_assets.py --video    # also render docs/brag.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
DOCS = ROOT / "docs"

CANVAS = "#F7F6F3"
INK = "#1A1A18"
MUTED = "#787774"
BORDER = "#E3E2DE"
PALE_GREEN = "#EDF3EC"
GREEN_TEXT = "#346538"
PALE_RED = "#FDEBEC"
RED_TEXT = "#9F2F2D"
PALE_YELLOW = "#FBF3DB"
YELLOW_TEXT = "#956400"
PALE_BLUE = "#E1F3FE"
BLUE_TEXT = "#1F6C9F"

SERIF = "C:/Windows/Fonts/georgia.ttf"
SERIF_BOLD = "C:/Windows/Fonts/georgiab.ttf"
MONO = "C:/Windows/Fonts/consola.ttf"
MONO_BOLD = "C:/Windows/Fonts/consolab.ttf"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:  # pragma: no cover - non-Windows dev boxes
        return ImageFont.truetype("DejaVuSans.ttf", size)


def ease(t: float) -> float:
    """Cubic ease-out for smooth entrances."""
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def rounded_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    radius: int = 10,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=BORDER, width=1)


def draw_gate_motif(image: Image.Image, origin: tuple[int, int], scale: float = 1.0) -> None:
    """The logo motif: three chunk bars meet a gate; two pass, one is deflected."""
    x, y = origin
    bar_h, gap = int(26 * scale), int(18 * scale)
    bar_w = int(150 * scale)
    gate_x = x + bar_w + int(60 * scale)
    for index, (color, passes) in enumerate(
        [(INK, True), (MUTED, False), (INK, True)]
    ):
        top = y + index * (bar_h + gap)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            (x, top, x + bar_w, top + bar_h), radius=int(6 * scale), fill=color
        )
        if passes:
            draw.rounded_rectangle(
                (gate_x + int(70 * scale), top, gate_x + int(70 * scale) + int(70 * scale), top + bar_h),
                radius=int(6 * scale),
                fill=color,
            )
        else:
            draw.line(
                [
                    (gate_x, top + bar_h / 2),
                    (gate_x + int(56 * scale), top + bar_h / 2 + int(46 * scale)),
                ],
                fill=RED_TEXT,
                width=max(2, int(5 * scale)),
            )
    draw = ImageDraw.Draw(image)
    draw.line(
        [(gate_x, y - int(20 * scale)), (gate_x, y + 3 * bar_h + 2 * gap + int(20 * scale))],
        fill=MUTED,
        width=max(2, int(4 * scale)),
    )


def make_logo(path: Path) -> None:
    image = Image.new("RGB", (512, 512), CANVAS)
    draw_gate_motif(image, (86, 150), scale=1.15)
    draw = ImageDraw.Draw(image)
    draw.text((86, 380), "clef-compactor", font=font(SERIF_BOLD, 40), fill=INK)
    draw.text((88, 434), "query-aware context compaction", font=font(MONO, 17), fill=MUTED)
    image.save(path, "PNG")
    print(f"wrote {path}")


def make_social_preview(path: Path) -> None:
    image = Image.new("RGB", (1280, 640), CANVAS)
    draw = ImageDraw.Draw(image)
    draw.text((72, 64), "clef-compactor", font=font(MONO_BOLD, 24), fill=MUTED)
    draw.text((68, 140), "Keep the evidence.", font=font(SERIF_BOLD, 76), fill=INK)
    draw.text((68, 236), "Cut the noise.", font=font(SERIF_BOLD, 76), fill=INK)
    draw.text(
        (72, 360),
        "Query-aware RAG context compaction with Cloudflare Clef.",
        font=font(SERIF, 30),
        fill="#3F3F3C",
    )
    for label, bg, fg, x in [
        ("verbatim chunks", PALE_GREEN, GREEN_TEXT, 72),
        ("auditable cuts", PALE_YELLOW, YELLOW_TEXT, 268),
        ("64k context", PALE_BLUE, BLUE_TEXT, 464),
    ]:
        width = int(draw.textlength(label, font=font(MONO, 20))) + 40
        rounded_card(draw, (x, 430, x + width, 478), bg, radius=24)
        draw.text((x + 20, 442), label, font=font(MONO, 20), fill=fg)
    draw.text((72, 540), "pip install clef-compactor", font=font(MONO_BOLD, 26), fill=INK)
    draw_gate_motif(image, (820, 190), scale=1.5)
    image.save(path, "PNG")
    print(f"wrote {path}")


# --- brag video -------------------------------------------------------------

WIDTH, HEIGHT, FPS = 1280, 720, 24
CHUNKS = [
    ("doc 1", "Refunds are accepted within 30 days of purchase.", 0.93, "keep"),
    ("doc 2", "The Eiffel Tower is located in Paris, France.", 0.05, "irrelevant"),
    ("doc 3", "Refunds go to the original payment method.", 0.85, "keep"),
    ("doc 4", "Our office hours are 9am to 5pm, Monday to Friday.", 0.42, "budget"),
    ("doc 5", "Email support to start a refund request.", 0.71, "keep"),
]


def frame_base() -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), CANVAS)


def text_w(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> float:
    return draw.textlength(text, font=fnt)


def render_frame(frame_index: int, total: int) -> Image.Image:
    image = frame_base()
    draw = ImageDraw.Draw(image)
    t = frame_index / total

    draw.text((72, 56), "clef-compactor", font=font(MONO_BOLD, 22), fill=MUTED)
    draw.line((72, 96, WIDTH - 72, 96), fill=BORDER, width=1)

    if t < 0.18:  # title scene
        k = ease(t / 0.18)
        draw.text((72, 200 + 24 * (1 - k)), "Keep the evidence.", font=font(SERIF_BOLD, 72), fill=INK)
        if t > 0.09:
            k2 = ease((t - 0.09) / 0.09)
            draw.text((72, 292 + 24 * (1 - k2)), "Cut the noise.", font=font(SERIF_BOLD, 72), fill=MUTED)
        return image

    draw.text((72, 128), "Keep the evidence. Cut the noise.", font=font(SERIF_BOLD, 34), fill=INK)
    draw.text((72, 190), 'query: "What is the refund policy?"', font=font(MONO, 24), fill=MUTED)

    # chunk cards
    card_top = 250
    for index, (name, preview, score, verdict) in enumerate(CHUNKS):
        appear = ease((t - 0.18 - index * 0.03) / 0.08)
        if appear <= 0:
            continue
        y = card_top + index * 66 + 30 * (1 - appear)
        x = int(72 + 200 * (1 - appear))
        bg = CANVAS
        if t > 0.40:
            bg = PALE_GREEN if verdict == "keep" else (PALE_RED if verdict == "irrelevant" else PALE_YELLOW)
        rounded_card(draw, (x, y, x + 720, y + 54), bg, radius=10)
        fg = INK
        if bg == PALE_GREEN:
            fg = GREEN_TEXT
        elif bg == PALE_RED:
            fg = RED_TEXT
        elif bg == PALE_YELLOW:
            fg = YELLOW_TEXT
        draw.text((x + 18, y + 15), name, font=font(MONO_BOLD, 20), fill=fg)
        draw.text((x + 92, y + 15), preview[:52], font=font(MONO, 18), fill=MUTED)
        # probability stamp
        stamp_t = ease((t - 0.38 - index * 0.045) / 0.06)
        if stamp_t > 0:
            stamp = "P=0.93" if False else f"P={score:.2f}"
            sw = text_w(draw, stamp, font(MONO_BOLD, 20))
            bx = x + 620
            rounded_card(draw, (bx, y + 11, bx + sw + 28, y + 43), bg, radius=16)
            alpha_shift = int((1 - stamp_t) * 6)
            draw.text((bx + 14, y + 15), stamp, font=font(MONO_BOLD, 20), fill=fg)
            del alpha_shift

    # cut animation
    if t > 0.60:
        for slot, verdict_label in [(1, "cut: irrelevant"), (3, "cut: budget_exhausted")]:
            slide = ease((t - 0.60 - (0 if slot == 1 else 0.08)) / 0.10)
            if slide <= 0:
                continue
            y = card_top + slot * 66
            x = 72 + int(560 * slide)
            fg = RED_TEXT if slot == 1 else YELLOW_TEXT
            draw.text((x + 740, y + 15), verdict_label, font=font(MONO_BOLD, 20), fill=fg)
    if t > 0.75:
        k = ease((t - 0.75) / 0.10)
        draw.text(
            (int(800 + 60 * (1 - k)), 300 + 24 * (1 - k)),
            "context: 3 chunks",
            font=font(SERIF_BOLD, 34),
            fill=INK,
        )
        draw.text(
            (804, 356),
            "1,024 -> 676 tokens  (-34%)",
            font=font(MONO, 24),
            fill=GREEN_TEXT,
        )
        draw.text((804, 396), "cost $0.0001 / call", font=font(MONO, 20), fill=MUTED)
    if t > 0.90:
        k = ease((t - 0.90) / 0.10)
        draw.line((72, 620, 72 + int(400 * k), 620), fill=BORDER, width=1)
        draw.text((72, 644), "pip install clef-compactor", font=font(MONO_BOLD, 26), fill=INK)
        draw.text(
            (72 + text_w(draw, "pip install clef-compactor", font(MONO_BOLD, 26)) + 28, 650),
            "github.com/Gjusev/clef-compactor",
            font=font(MONO, 18),
            fill=MUTED,
        )
    return image


def make_video(out_path: Path, poster_path: Path, seconds: float = 12.5) -> None:
    total = int(FPS * seconds)
    frames_dir = DOCS / ".brag-frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index in range(total):
        image = render_frame(index, total)
        image.save(frames_dir / f"f{index:04d}.png")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS),
            "-i", str(frames_dir / "f%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            "-movflags", "+faststart",
            str(out_path),
        ],
        check=True,
    )
    poster = render_frame(int(total * 0.82), total)
    poster.save(poster_path, "JPEG", quality=90)
    for stale in frames_dir.glob("*.png"):
        stale.unlink()
    frames_dir.rmdir()
    print(f"wrote {out_path} ({total} frames) and {poster_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", action="store_true", help="also render docs/brag.mp4")
    args = parser.parse_args(argv)

    ASSETS.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    make_logo(ASSETS / "logo.png")
    make_social_preview(ASSETS / "social-preview.png")
    if args.video:
        make_video(DOCS / "brag.mp4", DOCS / "brag.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
