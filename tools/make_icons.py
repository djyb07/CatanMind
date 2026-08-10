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

from catanmind import art  # noqa: E402
from catanmind.ui import BG, SEA  # noqa: E402

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


def _gradient_polygon(
    image: Image.Image, points, top_colour: str, bottom_colour: str
) -> None:
    """
    Fill a polygon with a vertical gradient.

    PIL has no gradient fill, so this paints one into a scratch layer and uses
    the polygon as a mask. Cheap, and it is what stops the tiles reading as
    flat coloured paper.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    box = (int(min(xs)) - 1, int(min(ys)) - 1, int(max(xs)) + 2, int(max(ys)) + 2)
    w, h = max(1, box[2] - box[0]), max(1, box[3] - box[1])

    ramp = Image.new("RGB", (1, h))
    top = art.shade(top_colour, 0)
    bottom = art.shade(bottom_colour, 0)
    tr, tg, tb = art._hex_to_rgb(top)
    br, bg, bb = art._hex_to_rgb(bottom)
    for y in range(h):
        t = y / max(1, h - 1)
        ramp.putpixel(
            (0, y),
            (
                int(tr + (br - tr) * t),
                int(tg + (bg - tg) * t),
                int(tb + (bb - tb) * t),
            ),
        )
    ramp = ramp.resize((w, h))

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(
        [(x - box[0], y - box[1]) for x, y in points], fill=255
    )
    image.paste(ramp, (box[0], box[1]), mask)


def draw_tile(
    image: Image.Image, centre, radius: float, colour: str, resource
) -> None:
    """One extruded tile with its terrain, the way the board draws it."""
    draw = ImageDraw.Draw(image)
    depth = radius * 0.17
    corners = hexagon(centre, radius)

    # The slab underneath, showing as thickness along the bottom edge.
    draw.polygon(
        [(x, y + depth) for x, y in corners], fill=art.shade(colour, -0.55)
    )
    _gradient_polygon(
        image, corners, art.shade(colour, 0.30), art.shade(colour, -0.27)
    )
    _terrain(draw, centre, radius * 0.82, colour, resource)
    draw.line(corners + [corners[0]], fill="#07131c", width=max(2, int(radius * 0.06)))


def _terrain(draw, centre, radius, colour, resource) -> None:
    """A hint of the tile's landscape. Simplified for icon sizes."""
    cx, cy = centre
    if resource == "wood":
        for dx, dy, s in ((-0.42, 0.16, 0.52), (0.0, -0.12, 0.62), (0.44, 0.20, 0.48)):
            x, y = cx + radius * dx, cy + radius * dy
            size = radius * s
            draw.polygon(
                [(x, y - size), (x + size * 0.62, y + size * 0.5),
                 (x - size * 0.62, y + size * 0.5)],
                fill=art.shade(colour, -0.38),
            )
            draw.polygon(
                [(x, y - size), (x, y + size * 0.5),
                 (x - size * 0.62, y + size * 0.5)],
                fill=art.shade(colour, -0.18),
            )
    elif resource == "wheat":
        for i in range(4):
            x = cx + radius * (-0.5 + i * 0.33)
            draw.line(
                [(x, cy + radius * 0.55), (x, cy - radius * 0.35)],
                fill=art.shade(colour, -0.34), width=max(2, int(radius * 0.07)),
            )
            draw.ellipse(
                [x - radius * 0.11, cy - radius * 0.52,
                 x + radius * 0.11, cy - radius * 0.20],
                fill=art.shade(colour, 0.34),
            )
    elif resource == "ore":
        for dx, height, width in ((-0.34, 0.72, 0.42), (0.30, 0.86, 0.46)):
            apex = (cx + radius * dx, cy - radius * height * 0.55)
            left = (cx + radius * (dx - width), cy + radius * 0.48)
            right = (cx + radius * (dx + width), cy + radius * 0.48)
            draw.polygon([apex, right, left], fill=art.shade(colour, -0.22))
            draw.polygon(
                [apex, left, (cx + radius * dx, cy + radius * 0.48)],
                fill=art.shade(colour, 0.26),
            )
            draw.polygon(
                [apex,
                 (apex[0] + radius * width * 0.28, apex[1] + radius * 0.15),
                 (apex[0] - radius * width * 0.28, apex[1] + radius * 0.15)],
                fill="#e8eef3",
            )


def draw_mark(image: Image.Image, centre, radius: float, letter: str = "C") -> None:
    """
    The app's mark: three tiles in a cluster, the way the island reads.

    A gold hexagon with a letter in it says nothing about the game. Three
    terrain tiles say Catan at a glance, which is the entire job of an icon on
    a crowded home screen.
    """
    from catanmind.ui import RESOURCE_COLOR
    from catanmind.board import Resource

    tile = radius * 0.62
    # Point-up hexes tile on a 30-degree offset; these three share a corner.
    positions = [
        (centre[0], centre[1] - tile * 0.92, Resource.WOOD, "wood"),
        (centre[0] - tile * 0.87, centre[1] + tile * 0.52, Resource.ORE, "ore"),
        (centre[0] + tile * 0.87, centre[1] + tile * 0.52, Resource.WHEAT, "wheat"),
    ]
    for cx, cy, resource, name in positions:
        draw_tile(image, (cx, cy), tile, RESOURCE_COLOR[resource], name)


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
