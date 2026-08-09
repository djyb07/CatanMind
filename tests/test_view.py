"""Fitting the board to a widget, and turning taps back into game objects."""

import math

import pytest

from catanmind.board import Board
from catanmind.view import EDGE_RADIUS, NODE_RADIUS, Viewport


@pytest.fixture(scope="module")
def board():
    return Board()


@pytest.fixture(scope="module")
def view(board):
    return Viewport.fit(board, 400, 400)


# -- fitting ---------------------------------------------------------------


def test_the_whole_board_lands_inside_the_widget(board, view):
    for node in board.nodes:
        x, y = view.to_screen(node.x, node.y)
        assert 0 <= x <= view.width
        assert 0 <= y <= view.height


def test_the_board_is_centred(board, view):
    xs = [view.to_screen(n.x, n.y)[0] for n in board.nodes]
    ys = [view.to_screen(n.x, n.y)[1] for n in board.nodes]
    assert (min(xs) + max(xs)) / 2 == pytest.approx(view.width / 2, abs=0.5)
    assert (min(ys) + max(ys)) / 2 == pytest.approx(view.height / 2, abs=0.5)


def test_scaling_is_uniform_so_hexes_stay_regular(board, view):
    """Every tile must render the same size and shape."""
    lengths = []
    for coord in board.tiles:
        corners = view.tile_corners(board, coord)
        for i in range(6):
            ax, ay = corners[i]
            bx, by = corners[(i + 1) % 6]
            lengths.append(math.hypot(bx - ax, by - ay))
    assert max(lengths) == pytest.approx(min(lengths), rel=1e-6)


def test_a_non_square_widget_still_fits(board):
    view = Viewport.fit(board, 800, 300)
    for node in board.nodes:
        x, y = view.to_screen(node.x, node.y)
        assert 0 <= x <= 800
        assert 0 <= y <= 300


def test_fitting_uses_the_space_available(board):
    """A bigger widget must actually draw a bigger board."""
    small = Viewport.fit(board, 300, 300)
    large = Viewport.fit(board, 600, 600)
    assert large.scale > small.scale


def test_padding_leaves_room_for_touch_targets(board):
    view = Viewport.fit(board, 400, 400, pad=26.0)
    for node in board.nodes:
        x, y = view.to_screen(node.x, node.y)
        assert 20 <= x <= 380
        assert 20 <= y <= 380


# -- round trips -----------------------------------------------------------


def test_screen_and_board_coordinates_round_trip(view):
    for x, y in ((0.0, 0.0), (100.0, -50.0), (-216.0, 200.0)):
        sx, sy = view.to_screen(x, y)
        bx, by = view.to_board(sx, sy)
        assert bx == pytest.approx(x)
        assert by == pytest.approx(y)


# -- hit testing -----------------------------------------------------------


def test_every_node_is_hittable_at_its_own_centre(board, view):
    for node in board.nodes:
        sx, sy = view.to_screen(node.x, node.y)
        assert view.node_at(board, sx, sy) == node.id


def test_every_edge_is_hittable_at_its_own_midpoint(board, view):
    for edge in board.edges:
        sx, sy = view.edge_xy(board, edge.id)
        assert view.edge_at(board, sx, sy) == edge.id


def test_every_tile_is_hittable_at_its_own_centre(board, view):
    for coord in board.tiles:
        sx, sy = view.tile_xy(board, coord)
        assert view.tile_at(board, sx, sy) == coord


def test_an_edge_is_hittable_along_its_length(board, view):
    """Roads must be tappable near their ends, not only dead centre."""
    for edge in board.edges:
        ax, ay = view.node_xy(board, edge.a)
        bx, by = view.node_xy(board, edge.b)
        for t in (0.35, 0.5, 0.65):
            sx = ax + (bx - ax) * t
            sy = ay + (by - ay) * t
            assert view.edge_at(board, sx, sy) == edge.id


def test_a_tap_far_from_any_node_hits_nothing(board, view):
    assert view.node_at(board, 0, 0) is None


def test_node_hits_beat_edge_hits_in_any_mode(board, view):
    """At an intersection the node wins, since that is what a tap there means."""
    for node in board.nodes:
        sx, sy = view.to_screen(node.x, node.y)
        kind, target = view.hit(board, sx, sy)
        assert kind == "node"
        assert target == node.id


def test_road_mode_ignores_nearby_intersections(board, view):
    """
    While placing a road, a tap near an intersection must still resolve to a
    path — otherwise the node swallows every tap and roads are unplaceable.
    """
    for edge in board.edges[:20]:
        ax, ay = view.node_xy(board, edge.a)
        bx, by = view.node_xy(board, edge.b)
        # 25% along, well inside the node's own radius.
        sx, sy = ax + (bx - ax) * 0.25, ay + (by - ay) * 0.25
        kind, target = view.hit(board, sx, sy, want="edge")
        assert kind == "edge"
        assert target == edge.id


def test_tile_mode_resolves_a_tap_near_a_corner_to_a_tile(board, view):
    """Robber placement must work even if the tap drifts toward a corner."""
    for coord in board.tiles:
        cx, cy = view.tile_xy(board, coord)
        corners = view.tile_corners(board, coord)
        for gx, gy in corners:
            sx, sy = cx + (gx - cx) * 0.5, cy + (gy - cy) * 0.5
            kind, target = view.hit(board, sx, sy, want="tile")
            assert kind == "tile"
            assert target == coord


def test_node_mode_returns_none_rather_than_a_wrong_object(board, view):
    kind, target = view.hit(board, 0, 0, want="node")
    assert kind == "none"
    assert target is None


def test_neighbouring_nodes_do_not_steal_each_others_taps(board, view):
    """
    The gap between adjacent intersections must exceed the touch radius, or
    tapping one spot would select the one next door.
    """
    closest = min(
        math.hypot(
            view.node_xy(board, n.id)[0] - view.node_xy(board, m)[0],
            view.node_xy(board, n.id)[1] - view.node_xy(board, m)[1],
        )
        for n in board.nodes
        for m in n.neighbors
    )
    assert closest > NODE_RADIUS
