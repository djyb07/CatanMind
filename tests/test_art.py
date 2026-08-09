"""
The board's drawing layer.

Art is judged by eye, but the parts underneath it are not: a colour helper
that returns something Flet cannot parse, terrain drawn where the number token
will cover it, or a tile that silently renders nothing are all defects a
screenshot hides and a test catches.
"""

import math

import pytest

from catanmind import art
from catanmind.board import Board, Layout, Resource
from catanmind.view import Viewport


HEX_COLOUR = 7        # "#rrggbb"
ARGB_COLOUR = 9       # "#aarrggbb"


@pytest.fixture(scope="module")
def board():
    return Board(Layout.standard())


# -- colour ----------------------------------------------------------------


def test_shading_lightens_and_darkens():
    base = "#2f6b3a"
    assert art.shade(base, 0.4) != base
    assert art.shade(base, -0.4) != base
    lighter = art._hex_to_rgb(art.shade(base, 0.4))
    darker = art._hex_to_rgb(art.shade(base, -0.4))
    original = art._hex_to_rgb(base)
    assert all(l >= o for l, o in zip(lighter, original))
    assert all(d <= o for d, o in zip(darker, original))


def test_shading_stays_inside_the_colour_range():
    """Mixing toward white or black, so extremes cannot overflow."""
    for amount in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for base in ("#000000", "#ffffff", "#2f6b3a"):
            out = art.shade(base, amount)
            assert len(out) == HEX_COLOUR
            assert all(0 <= c <= 255 for c in art._hex_to_rgb(out))


def test_a_dark_colour_still_lightens_visibly():
    """Scaling a near-black colour does nothing; mixing toward white works."""
    assert art._hex_to_rgb(art.shade("#0a0a0a", 0.5))[0] > 100


def test_alpha_produces_the_argb_string_flet_expects():
    out = art.alpha("ffffff", 0.5)
    assert len(out) == ARGB_COLOUR
    assert out.startswith("#")
    assert out.endswith("ffffff")
    assert art.alpha("#ffffff", 1.0) == "#ffffffff"
    assert art.alpha("ffffff", 0.0) == "#00ffffff"


# -- geometry --------------------------------------------------------------


def test_a_hexagon_has_six_evenly_spaced_corners():
    points = art.hexagon((0, 0), 10)
    assert len(points) == 6
    for x, y in points:
        assert math.hypot(x, y) == pytest.approx(10)


def test_a_hexagon_points_up():
    """Pointy-top, matching the board's own geometry."""
    points = art.hexagon((0, 0), 10)
    highest = min(points, key=lambda p: p[1])
    assert highest[0] == pytest.approx(0, abs=1e-6)


def test_shifting_and_scaling_leave_the_shape_alone():
    points = art.hexagon((5, 5), 10)
    moved = art.shift(points, 3, -2)
    for before, after in zip(points, moved):
        assert after[0] - before[0] == pytest.approx(3)
        assert after[1] - before[1] == pytest.approx(-2)
    smaller = art.scaled(points, (5, 5), 0.5)
    for x, y in smaller:
        assert math.hypot(x - 5, y - 5) == pytest.approx(5)


# -- tiles -----------------------------------------------------------------


@pytest.mark.parametrize("resource", list(Resource) + [None])
def test_every_resource_draws_a_tile(resource):
    corners = art.hexagon((100, 100), 50)
    shapes = art.tile_shapes(corners, (100, 100), 50, "#2f6b3a", resource)
    assert len(shapes) > 4, f"{resource} produced almost nothing"


def test_a_tile_is_built_from_a_slab_a_face_and_an_edge():
    corners = art.hexagon((100, 100), 50)
    plain = art.tile_shapes(
        corners, (100, 100), 50, "#2f6b3a", None, with_terrain=False
    )
    # Slab, lit face, two bevel arcs, seam.
    assert len(plain) == 5


def test_terrain_makes_a_tile_busier_than_a_bare_one():
    corners = art.hexagon((100, 100), 50)
    bare = art.tile_shapes(
        corners, (100, 100), 50, "#2f6b3a", Resource.WOOD, with_terrain=False
    )
    dressed = art.tile_shapes(
        corners, (100, 100), 50, "#2f6b3a", Resource.WOOD
    )
    assert len(dressed) > len(bare)


@pytest.mark.parametrize(
    "resource", [Resource.WOOD, Resource.SHEEP, Resource.WHEAT]
)
def test_scattered_terrain_keeps_clear_of_the_number_token(resource):
    """
    The token is opaque and sits dead centre. Anything drawn as separate
    objects belongs on a ring outside it, or it is ink nobody sees.
    """
    radius = 100.0
    shapes = art.terrain(resource, (0.0, 0.0), radius, "#2f6b3a")
    assert shapes

    # Every point the terrain draws, whether it came from a circle or a path.
    points = []
    for shape in shapes:
        if hasattr(shape, "radius") and hasattr(shape, "x"):
            points.append((shape.x, shape.y))
        for element in getattr(shape, "elements", []) or []:
            if hasattr(element, "x") and hasattr(element, "y"):
                points.append((element.x, element.y))
    assert points, f"{resource} drew nothing measurable"

    # Scattered objects ring the token rather than sitting under it. A few
    # stray points may fall inside — a tree trunk reaching inward — so this
    # checks the bulk of the drawing, not every last vertex.
    outside = sum(
        1 for x, y in points
        if math.hypot(x, y) > radius * art.TOKEN_CLEARANCE * 0.6
    )
    assert outside > len(points) * 0.7, (
        f"{resource}: most of the terrain sits under the number token"
    )


def test_the_number_token_shows_its_number_and_its_dots():
    shapes = art.number_token((0, 0), 20, 8, 5)
    texts = [s for s in shapes if type(s).__name__ == "Text"]
    circles = [s for s in shapes if type(s).__name__ == "Circle"]
    assert len(texts) == 1
    assert texts[0].value == "8"
    # The disc, its rim, and one circle per dot.
    assert len(circles) == 2 + 5


def test_a_hot_number_is_printed_in_red():
    hot = art.number_token((0, 0), 20, 8, 5)
    cool = art.number_token((0, 0), 20, 5, 4)
    hot_text = next(s for s in hot if type(s).__name__ == "Text")
    cool_text = next(s for s in cool if type(s).__name__ == "Text")
    assert hot_text.style.color != cool_text.style.color


# -- pieces ----------------------------------------------------------------


def test_pieces_all_draw_something():
    assert art.settlement((0, 0), 10, "#e05260")
    assert art.city((0, 0), 10, "#e05260")
    assert art.road((0, 0), (10, 10), 6, "#e05260")
    assert art.robber((0, 0), 12)


def test_a_city_is_bigger_than_a_settlement():
    """The two must be distinguishable at a glance on a crowded board."""
    def spread(shapes):
        points = [
            (e.x, e.y)
            for s in shapes if hasattr(s, "elements")
            for e in s.elements if hasattr(e, "x")
        ]
        return max(math.hypot(x, y) for x, y in points)

    assert spread(art.city((0, 0), 10, "#e05260")) > spread(
        art.settlement((0, 0), 10, "#e05260")
    )


def test_every_piece_casts_a_shadow():
    """Depth is what separates a piece from the tile it stands on."""
    for shapes in (
        art.settlement((0, 0), 10, "#e05260"),
        art.city((0, 0), 10, "#e05260"),
        art.robber((0, 0), 12),
        art.number_token((0, 0), 20, 6, 5),
    ):
        assert any(type(s).__name__ == "Shadow" for s in shapes)


def test_the_island_casts_one_shadow(board):
    view = Viewport.fit(board, 390, 354)
    outline = [view.node_xy(board, n.id) for n in board.nodes[:8]]
    shapes = art.island_shadow(outline, view.hex_radius(board))
    assert len(shapes) == 1
    assert type(shapes[0]).__name__ == "Shadow"


# -- lighting --------------------------------------------------------------


def test_faces_are_lit_from_a_single_direction():
    """
    Every gradient in the module agrees on where the light is. If one
    disagreed the board would look wrong without any single part being wrong.
    """
    paint = art.lit("#2f6b3a", (0, 0), 10)
    begin = paint.gradient.begin
    assert begin.x == pytest.approx(art.LIGHT[0] * 10)
    assert begin.y == pytest.approx(art.LIGHT[1] * 10)
    assert art.LIGHT[1] < 0, "light comes from above"


def test_a_lit_face_runs_from_light_to_dark():
    paint = art.lit("#808080", (0, 0), 10)
    colours = [art._hex_to_rgb(c) for c in paint.gradient.colors]
    assert colours[0][0] > colours[-1][0]
