"""
Drawing the island so it looks like something on a table.

A flat polygon filled with one colour reads as a diagram. A physical Catan tile
has thickness, catches light on one side, and carries a picture of the terrain
— and all three of those are what make it read as a board rather than a chart.

Everything here is vector work on ``flet.canvas``: gradients for the lit faces,
a second hexagon underneath for the extrusion, and a small amount of drawn
terrain per resource. No bitmaps, so it stays sharp at any size and adds
nothing to the APK.

The light is fixed at the top-left throughout — every highlight, shadow and
gradient in this module agrees on that, which is most of what sells the depth.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import flet as ft
import flet.canvas as cv

from catanmind.board import Resource

Point = Tuple[float, float]

#: Where the light comes from. Everything shades against this.
LIGHT = (-0.45, -0.75)

#: How thick a tile is, as a fraction of its radius.
DEPTH = 0.17


# --------------------------------------------------------------------------
# Colour helpers
# --------------------------------------------------------------------------


def _hex_to_rgb(colour: str) -> Tuple[int, int, int]:
    colour = colour.lstrip("#")
    return tuple(int(colour[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: Sequence[float]) -> str:
    return "#" + "".join(f"{max(0, min(255, int(round(c)))):02x}" for c in rgb)


def shade(colour: str, amount: float) -> str:
    """
    Lighten (``amount`` > 0) or darken (< 0) a colour.

    Mixes toward white or black rather than scaling, so dark colours still
    lighten visibly and bright ones do not blow out.
    """
    r, g, b = _hex_to_rgb(colour)
    target = 255.0 if amount > 0 else 0.0
    weight = abs(amount)
    return _rgb_to_hex(
        [c + (target - c) * weight for c in (r, g, b)]
    )


def alpha(colour: str, opacity: float) -> str:
    """An ``#AARRGGBB`` string, which is what Flet wants for translucency."""
    value = max(0, min(255, int(round(opacity * 255))))
    return f"#{value:02x}{colour.lstrip('#')}"


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def polygon(points: Sequence[Point]) -> List[cv.Path.PathElement]:
    out: List[cv.Path.PathElement] = [cv.Path.MoveTo(*points[0])]
    out += [cv.Path.LineTo(*p) for p in points[1:]]
    out.append(cv.Path.Close())
    return out


def hexagon(centre: Point, radius: float, rotation: float = -90) -> List[Point]:
    cx, cy = centre
    return [
        (
            cx + radius * math.cos(math.radians(60 * i + rotation)),
            cy + radius * math.sin(math.radians(60 * i + rotation)),
        )
        for i in range(6)
    ]


def shift(points: Sequence[Point], dx: float, dy: float) -> List[Point]:
    return [(x + dx, y + dy) for x, y in points]


def scaled(points: Sequence[Point], centre: Point, factor: float) -> List[Point]:
    cx, cy = centre
    return [(cx + (x - cx) * factor, cy + (y - cy) * factor) for x, y in points]


def fill(colour: str) -> ft.Paint:
    return ft.Paint(color=colour, style=ft.PaintingStyle.FILL)


def stroke(colour: str, width: float) -> ft.Paint:
    return ft.Paint(
        color=colour, stroke_width=width, style=ft.PaintingStyle.STROKE,
        stroke_join=ft.StrokeJoin.ROUND, stroke_cap=ft.StrokeCap.ROUND,
    )


def lit(colour: str, centre: Point, radius: float, strength: float = 0.22) -> ft.Paint:
    """A face gradient: brighter where the light hits, darker away from it."""
    cx, cy = centre
    return ft.Paint(
        gradient=ft.PaintLinearGradient(
            begin=ft.Offset(cx + LIGHT[0] * radius, cy + LIGHT[1] * radius),
            end=ft.Offset(cx - LIGHT[0] * radius, cy - LIGHT[1] * radius),
            colors=[shade(colour, strength), colour, shade(colour, -strength * 0.9)],
            color_stops=[0.0, 0.55, 1.0],
        ),
        style=ft.PaintingStyle.FILL,
    )


# --------------------------------------------------------------------------
# Terrain
# --------------------------------------------------------------------------


def _conifer(cx: float, cy: float, size: float, colour: str) -> List[cv.Shape]:
    """One fir tree: a trunk and two stacked canopies."""
    trunk = size * 0.16
    out = [
        cv.Path(
            elements=polygon([
                (cx - trunk * 0.4, cy + size * 0.55),
                (cx + trunk * 0.4, cy + size * 0.55),
                (cx + trunk * 0.4, cy + size * 0.9),
                (cx - trunk * 0.4, cy + size * 0.9),
            ]),
            paint=fill(shade(colour, -0.45)),
        ),
    ]
    for i, (top, spread, base) in enumerate(
        ((-0.95, 0.58, 0.08), (-0.32, 0.80, 0.66))
    ):
        apex = (cx, cy + size * top)
        left = (cx - size * spread, cy + size * base)
        right = (cx + size * spread, cy + size * base)
        body = shade(colour, -0.34 if i == 0 else -0.44)
        out.append(cv.Path(elements=polygon([apex, right, left]), paint=fill(body)))
        # The left half catches the light, which is what stops a triangle
        # reading as a flat cut-out.
        out.append(
            cv.Path(
                elements=polygon([apex, (cx, cy + size * base), left]),
                paint=fill(shade(body, 0.26)),
            )
        )
    return out


def _sheep(cx: float, cy: float, size: float) -> List[cv.Shape]:
    """A woolly body and a dark head — legible at thumbnail size."""
    body = shade("#f4f1e8", 0.0)
    return [
        cv.Circle(cx - size * 0.18, cy, size * 0.42, paint=fill(body)),
        cv.Circle(cx + size * 0.12, cy - size * 0.12, size * 0.34, paint=fill(body)),
        cv.Circle(cx + size * 0.16, cy + size * 0.12, size * 0.34, paint=fill(body)),
        cv.Circle(cx + size * 0.42, cy + size * 0.10, size * 0.21,
                  paint=fill("#43423f")),
    ]


def _wheat_stalk(cx: float, cy: float, size: float, colour: str) -> List[cv.Shape]:
    out: List[cv.Shape] = [
        cv.Path(
            elements=[cv.Path.MoveTo(cx, cy + size * 0.9),
                      cv.Path.LineTo(cx, cy - size * 0.5)],
            paint=stroke(shade(colour, -0.4), max(1.0, size * 0.13)),
        )
    ]
    for i in range(3):
        y = cy - size * (0.45 - i * 0.28)
        for direction in (-1, 1):
            out.append(
                cv.Path(
                    elements=[
                        cv.Path.MoveTo(cx, y + size * 0.12),
                        cv.Path.QuadraticTo(
                            cx + direction * size * 0.42, y - size * 0.05,
                            cx + direction * size * 0.20, y - size * 0.30,
                        ),
                    ],
                    paint=stroke(shade(colour, 0.35), max(1.0, size * 0.15)),
                )
            )
    return out


def _brick_courses(
    centre: Point, radius: float, colour: str
) -> List[cv.Shape]:
    """Staggered courses of brick, clipped to a band across the tile."""
    cx, cy = centre
    out: List[cv.Shape] = []
    course = radius * 0.17
    brick_w = radius * 0.38
    for row in range(-2, 3):
        y = cy + row * course * 1.35 + radius * 0.06
        half = radius * 0.66 * math.cos(math.asin(max(-0.99, min(0.99, (y - cy) / (radius * 1.05)))))
        offset = (brick_w / 2) if row % 2 else 0.0
        x = cx - half + offset
        while x + brick_w * 0.5 < cx + half:
            out.append(
                cv.Path(
                    elements=polygon([
                        (x, y), (x + brick_w * 0.86, y),
                        (x + brick_w * 0.86, y + course * 0.78),
                        (x, y + course * 0.78),
                    ]),
                    paint=fill(shade(colour, 0.13 if row % 2 else 0.05)),
                )
            )
            x += brick_w
    return out


def _peaks(centre: Point, radius: float, colour: str) -> List[cv.Shape]:
    """Angular mountains with a lit facet, for the ore tile."""
    cx, cy = centre
    out: List[cv.Shape] = []
    # A range rather than three bumps: outer peaks first, the tall one last so
    # it overlaps them and the ridge reads as having depth.
    for dx, height, width in (
        (-0.46, 0.78, 0.44), (0.44, 0.70, 0.40),
        (-0.16, 0.92, 0.42), (0.18, 1.02, 0.46),
    ):
        apex = (cx + radius * dx, cy - radius * height * 0.55)
        left = (cx + radius * (dx - width), cy + radius * 0.52)
        right = (cx + radius * (dx + width), cy + radius * 0.52)
        out.append(
            cv.Path(elements=polygon([apex, right, left]),
                    paint=fill(shade(colour, -0.18)))
        )
        # The face turned toward the light.
        out.append(
            cv.Path(
                elements=polygon([
                    apex, left,
                    (cx + radius * (dx - width * 0.1), cy + radius * 0.42),
                ]),
                paint=fill(shade(colour, 0.24)),
            )
        )
        # Snow cap.
        out.append(
            cv.Path(
                elements=polygon([
                    apex,
                    (apex[0] + radius * width * 0.30, apex[1] + radius * 0.16),
                    (apex[0] - radius * width * 0.30, apex[1] + radius * 0.16),
                ]),
                paint=fill("#e8eef3"),
            )
        )
    return out


def _dunes(centre: Point, radius: float, colour: str) -> List[cv.Shape]:
    cx, cy = centre
    out: List[cv.Shape] = []
    for i, (dx, dy, w) in enumerate(((-0.28, 0.20, 0.55), (0.26, 0.38, 0.45))):
        out.append(
            cv.Path(
                elements=[
                    cv.Path.MoveTo(cx + radius * (dx - w), cy + radius * dy),
                    cv.Path.QuadraticTo(
                        cx + radius * dx, cy + radius * (dy - 0.34),
                        cx + radius * (dx + w), cy + radius * dy,
                    ),
                    cv.Path.Close(),
                ],
                paint=fill(shade(colour, 0.14 if i == 0 else -0.10)),
            )
        )
    # A cactus, because the desert should be recognisable at a glance.
    out.append(
        cv.Path(
            elements=polygon([
                (cx - radius * 0.06, cy - radius * 0.42),
                (cx + radius * 0.06, cy - radius * 0.42),
                (cx + radius * 0.06, cy + radius * 0.10),
                (cx - radius * 0.06, cy + radius * 0.10),
            ]),
            paint=fill("#4f7a45"),
        )
    )
    out.append(
        cv.Path(
            elements=[
                cv.Path.MoveTo(cx - radius * 0.06, cy - radius * 0.16),
                cv.Path.LineTo(cx - radius * 0.20, cy - radius * 0.16),
                cv.Path.LineTo(cx - radius * 0.20, cy - radius * 0.30),
            ],
            paint=stroke("#4f7a45", max(1.5, radius * 0.07)),
        )
    )
    return out


#: The number token sits dead centre and covers everything under it, so
#: scattered terrain half-disappears behind it. Anything drawn as separate
#: objects goes on a ring outside this radius instead.
TOKEN_CLEARANCE = 0.52


def _ring(centre: Point, radius: float, count: int, start: float = -90):
    """``count`` positions evenly spaced around the token, clockwise."""
    cx, cy = centre
    step = 360 / count
    for i in range(count):
        angle = math.radians(start + i * step)
        yield cx + radius * math.cos(angle), cy + radius * math.sin(angle)


def terrain(
    resource: Optional[Resource], centre: Point, radius: float, colour: str
) -> List[cv.Shape]:
    """
    The picture on a tile.

    Objects are arranged around the number token rather than across the whole
    face; the token is opaque and a tree behind it is just wasted ink.
    """
    out: List[cv.Shape] = []
    ring_radius = radius * TOKEN_CLEARANCE

    if resource is Resource.WOOD:
        for i, (x, y) in enumerate(_ring(centre, ring_radius * 1.06, 6, -75)):
            out += _conifer(x, y, radius * (0.40 if i % 2 else 0.46), colour)
        return out
    if resource is Resource.BRICK:
        return _brick_courses(centre, radius, colour)
    if resource is Resource.SHEEP:
        for i, (x, y) in enumerate(_ring(centre, ring_radius * 1.02, 4, -55)):
            out += _sheep(x, y, radius * (0.24 if i % 2 else 0.27))
        return out
    if resource is Resource.WHEAT:
        for x, y in _ring(centre, ring_radius * 1.04, 7, -80):
            out += _wheat_stalk(x, y, radius * 0.30, colour)
        return out
    if resource is Resource.ORE:
        return _peaks(centre, radius, colour)
    return _dunes(centre, radius, colour)


# --------------------------------------------------------------------------
# The tile itself
# --------------------------------------------------------------------------


def tile_shapes(
    corners: Sequence[Point],
    centre: Point,
    radius: float,
    colour: str,
    resource: Optional[Resource],
    *,
    with_terrain: bool = True,
) -> List[cv.Shape]:
    """
    One tile, drawn as a solid object: extruded side, lit top, bevelled edge.
    """
    depth = radius * DEPTH
    out: List[cv.Shape] = []

    # The slab below, showing as thickness along the bottom edge.
    out.append(
        cv.Path(
            elements=polygon(shift(corners, 0, depth)),
            paint=fill(shade(colour, -0.55)),
        )
    )
    # The lit top face.
    out.append(
        cv.Path(elements=polygon(corners),
                paint=lit(colour, centre, radius, 0.30))
    )

    if with_terrain and resource is not None or resource is None and with_terrain:
        out += terrain(resource, centre, radius * 0.90, colour)

    # A bright inner rim on the lit side, a dark one opposite. Two arcs of the
    # same hexagon rather than a full outline, or it reads as a sticker.
    inner = scaled(corners, centre, 0.995)
    out.append(
        cv.Path(
            elements=[cv.Path.MoveTo(*inner[4]), cv.Path.LineTo(*inner[5]),
                      cv.Path.LineTo(*inner[0]), cv.Path.LineTo(*inner[1])],
            paint=stroke(alpha("ffffff", 0.30), max(1.0, radius * 0.045)),
        )
    )
    out.append(
        cv.Path(
            elements=[cv.Path.MoveTo(*inner[1]), cv.Path.LineTo(*inner[2]),
                      cv.Path.LineTo(*inner[3]), cv.Path.LineTo(*inner[4])],
            paint=stroke(alpha("000000", 0.28), max(1.0, radius * 0.045)),
        )
    )
    # The seam between neighbouring tiles.
    out.append(
        cv.Path(elements=polygon(corners),
                paint=stroke("#07131c", max(1.2, radius * 0.05)))
    )
    return out


def island_shadow(outline: Sequence[Point], radius: float) -> List[cv.Shape]:
    """A soft shadow under the whole island, so it sits above the water."""
    return [
        cv.Shadow(
            path=polygon(shift(outline, 0, radius * 0.10)),
            color=alpha("000000", 0.55),
            elevation=radius * 0.34,
        )
    ]


def number_token(
    centre: Point, radius: float, number: int, dots: int
) -> List[cv.Shape]:
    """A raised disc, the way the printed token looks in the hand."""
    cx, cy = centre
    hot = number in (6, 8)
    ink = "#b3261e" if hot else "#20242a"
    out: List[cv.Shape] = [
        cv.Shadow(
            path=[cv.Path.Oval(cx - radius, cy - radius + radius * 0.14,
                               radius * 2, radius * 2)],
            color=alpha("000000", 0.5),
            elevation=radius * 0.5,
        ),
        cv.Circle(cx, cy, radius, paint=ft.Paint(
            gradient=ft.PaintRadialGradient(
                center=ft.Offset(cx - radius * 0.35, cy - radius * 0.45),
                radius=radius * 1.7,
                colors=["#fffdf6", "#f0e9d8", "#d9cfb6"],
                color_stops=[0.0, 0.6, 1.0],
            ),
            style=ft.PaintingStyle.FILL,
        )),
        cv.Circle(cx, cy, radius, paint=stroke(alpha("000000", 0.30),
                                               max(1.0, radius * 0.09))),
    ]
    out.append(
        cv.Text(
            cx, cy - radius * 0.14, str(number),
            style=ft.TextStyle(size=radius * 1.05, color=ink,
                               weight=ft.FontWeight.BOLD),
            alignment=ft.Alignment(0, 0),
        )
    )
    spacing = radius * 0.25
    start = cx - spacing * (dots - 1) / 2
    for i in range(dots):
        out.append(
            cv.Circle(start + i * spacing, cy + radius * 0.60, radius * 0.075,
                      paint=fill(ink))
        )
    return out


# --------------------------------------------------------------------------
# Pieces
# --------------------------------------------------------------------------


def settlement(centre: Point, size: float, colour: str) -> List[cv.Shape]:
    """A little house, seen from the front."""
    cx, cy = centre
    body = size * 0.82
    roof = size * 0.72
    points = [
        (cx - body, cy + body * 0.85),
        (cx - body, cy - body * 0.15),
        (cx, cy - body * 0.15 - roof),
        (cx + body, cy - body * 0.15),
        (cx + body, cy + body * 0.85),
    ]
    return [
        cv.Shadow(path=polygon(shift(points, 0, size * 0.22)),
                  color=alpha("000000", 0.55), elevation=size * 0.5),
        cv.Path(elements=polygon(points), paint=lit(colour, centre, size, 0.30)),
        cv.Path(elements=polygon(points),
                paint=stroke(shade(colour, -0.55), max(1.0, size * 0.16))),
    ]


def city(centre: Point, size: float, colour: str) -> List[cv.Shape]:
    """A larger house with a tower — the silhouette reads even when tiny."""
    cx, cy = centre
    w = size * 1.12
    points = [
        (cx - w, cy + w * 0.72),
        (cx - w, cy - w * 0.10),
        (cx - w * 0.42, cy - w * 0.62),
        (cx + w * 0.10, cy - w * 0.10),
        (cx + w * 0.10, cy - w * 0.42),
        (cx + w, cy - w * 0.42),
        (cx + w, cy + w * 0.72),
    ]
    return [
        cv.Shadow(path=polygon(shift(points, 0, size * 0.24)),
                  color=alpha("000000", 0.55), elevation=size * 0.55),
        cv.Path(elements=polygon(points), paint=lit(colour, centre, size, 0.30)),
        cv.Path(elements=polygon(points),
                paint=stroke(shade(colour, -0.55), max(1.0, size * 0.15))),
    ]


def road(a: Point, b: Point, width: float, colour: str) -> List[cv.Shape]:
    """A raised plank rather than a line."""
    return [
        cv.Path(elements=[cv.Path.MoveTo(*a), cv.Path.LineTo(*b)],
                paint=stroke(alpha("000000", 0.45), width * 1.7)),
        cv.Path(elements=[cv.Path.MoveTo(*a), cv.Path.LineTo(*b)],
                paint=stroke(shade(colour, -0.42), width * 1.25)),
        cv.Path(elements=[cv.Path.MoveTo(*a), cv.Path.LineTo(*b)],
                paint=stroke(colour, width)),
        cv.Path(
            elements=[
                cv.Path.MoveTo(a[0], a[1] - width * 0.22),
                cv.Path.LineTo(b[0], b[1] - width * 0.22),
            ],
            paint=stroke(alpha("ffffff", 0.28), width * 0.3),
        ),
    ]


def robber(centre: Point, size: float) -> List[cv.Shape]:
    """The robber, as a hooded figure rather than a letter."""
    cx, cy = centre
    return [
        cv.Shadow(
            path=[cv.Path.Oval(cx - size * 0.8, cy - size * 0.5,
                               size * 1.6, size * 1.6)],
            color=alpha("000000", 0.6), elevation=size * 0.7,
        ),
        # Cloak.
        cv.Path(
            elements=[
                cv.Path.MoveTo(cx - size * 0.78, cy + size * 0.92),
                cv.Path.QuadraticTo(cx - size * 0.62, cy - size * 0.35,
                                    cx, cy - size * 0.62),
                cv.Path.QuadraticTo(cx + size * 0.62, cy - size * 0.35,
                                    cx + size * 0.78, cy + size * 0.92),
                cv.Path.Close(),
            ],
            paint=lit("#2b2f36", centre, size, 0.35),
        ),
        cv.Circle(cx, cy - size * 0.62, size * 0.40,
                  paint=lit("#22262c", (cx, cy - size * 0.62), size, 0.4)),
        cv.Path(
            elements=[
                cv.Path.MoveTo(cx - size * 0.78, cy + size * 0.92),
                cv.Path.QuadraticTo(cx - size * 0.62, cy - size * 0.35,
                                    cx, cy - size * 0.62),
                cv.Path.QuadraticTo(cx + size * 0.62, cy - size * 0.35,
                                    cx + size * 0.78, cy + size * 0.92),
                cv.Path.Close(),
            ],
            paint=stroke(alpha("ffffff", 0.22), max(1.0, size * 0.10)),
        ),
    ]
