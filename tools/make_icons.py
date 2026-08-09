"""
Generate the launcher icon and splash image.

Kept as a script rather than hand-drawn files so the app's mark and its
packaging cannot drift apart: both are built from the same hexagon and the
same palette constants the interface uses.

    python tools/make_icons.py
"""

from __future__ import annotations

import math
import pathlib
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from catanmind.ui import ACCENT, BG, ON_ACCENT, SEA  # noqa: E402

ASSETS = ROOT / "assets"

#: Android wants 512 at minimum and rescales down; 1024 leaves headroom for
#: the Play Store listing, which asks for exactly that.
ICON_SIZE = 1024
SPLASH_SIZE = (1242, 2208)


def hexagon(centre, radius, rotation=-90):
    cx, cy = centre
    return [
        (
            cx + radius * math.cos(math.radians(60 * i + rotation)),
            cy + radius * math.sin(math.radians(60 * i + rotation)),
        )
        for i in range(6)
    ]


def _font(size: int):
    for name in ("seguibl.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_mark(image: Image.Image, centre, radius: float, letter: str = "C") -> None:
    """The gold tile with a dark initial — the same mark the app shows."""
    draw = ImageDraw.Draw(image)
    # A slightly larger hex behind it reads as the tile's edge at small sizes.
    draw.polygon(hexagon(centre, radius * 1.06), fill=SEA)
    draw.polygon(hexagon(centre, radius), fill=ACCENT)

    font = _font(int(radius * 1.15))
    box = draw.textbbox((0, 0), letter, font=font)
    draw.text(
        (
            centre[0] - (box[2] - box[0]) / 2 - box[0],
            centre[1] - (box[3] - box[1]) / 2 - box[1],
        ),
        letter,
        font=font,
        fill=ON_ACCENT,
    )


def sea_lattice(image: Image.Image, spacing: float, colour: str) -> None:
    """The faint hex field the app draws behind the island."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    dx = spacing * math.sqrt(3)
    dy = spacing * 1.5
    row = 0
    y = -spacing
    while y < height + spacing:
        offset = 0 if row % 2 == 0 else dx / 2
        x = -spacing + offset
        while x < width + spacing:
            draw.polygon(hexagon((x, y), spacing), outline=colour, width=3)
            x += dx
        y += dy
        row += 1


#: Android crops adaptive icons to a circle or squircle and only guarantees
#: the middle ~66%. A point-up hexagon is tallest at its points, so the radius
#: is set from the height: 0.28 puts the points at 56% and keeps them clear.
ICON_MARK_RADIUS = 0.28


def build_icon() -> None:
    image = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), SEA)
    sea_lattice(image, ICON_SIZE * 0.11, "#16405c")
    draw_mark(
        image, (ICON_SIZE / 2, ICON_SIZE / 2), ICON_SIZE * ICON_MARK_RADIUS
    )
    image.save(ASSETS / "icon.png")
    print(f"wrote {ASSETS / 'icon.png'} ({ICON_SIZE}x{ICON_SIZE})")


def build_splash() -> None:
    image = Image.new("RGBA", SPLASH_SIZE, BG)
    sea_lattice(image, SPLASH_SIZE[0] * 0.09, "#102a3e")
    centre = (SPLASH_SIZE[0] / 2, SPLASH_SIZE[1] / 2)
    draw_mark(image, centre, SPLASH_SIZE[0] * 0.16)

    draw = ImageDraw.Draw(image)
    font = _font(int(SPLASH_SIZE[0] * 0.075))
    text = "CatanMind"
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (centre[0] - (box[2] - box[0]) / 2 - box[0],
         centre[1] + SPLASH_SIZE[0] * 0.22),
        text, font=font, fill="#eef5fa",
    )
    image.save(ASSETS / "splash.png")
    print(f"wrote {ASSETS / 'splash.png'} ({SPLASH_SIZE[0]}x{SPLASH_SIZE[1]})")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    build_icon()
    build_splash()
