"""
Does it fit, and can you hit it?

The app is aimed at a phone held in one hand, so the things that break first
are the ones a desktop never shows you: a board that overflows a narrow
screen, an intersection too small to tap, a button under the minimum comfort
size. Each of those is measurable, so each is checked here rather than left to
a screenshot.
"""

import math

import pytest


from catanmind.board import Layout, Resource
from catanmind.ui import (
    BUTTON_STYLE,
    CatanMind,
    RESOURCE_COLOR,
    RESOURCE_ICON,
    RESOURCE_LABEL,
    TAP_TARGET,
    board_shapes,
    btn,
)
from catanmind.view import NODE_RADIUS


#: Real devices, smallest first. The first is a small Android phone, which is
#: the hardest case and the one most likely to be in someone's pocket.
DEVICES = [
    ("small phone", 360, 640),
    ("iphone", 390, 844),
    ("large phone", 430, 932),
    ("tablet", 768, 1024),
    ("desktop", 1280, 800),
]


class StubPage:
    def __init__(self):
        self.updates = 0
        self.dialogs = []

    def update(self):
        self.updates += 1

    def add(self, control):
        pass

    def show_dialog(self, dialog):
        self.dialogs.append(dialog)

    def pop_dialog(self):
        return self.dialogs.pop() if self.dialogs else None


@pytest.fixture
def app():
    a = CatanMind(StubPage(), num_players=4, me=1)
    a.build()
    layout = Layout.standard()
    a.pending = dict(layout.tiles)
    a.pending_ports = dict(layout.ports)
    a._finish_editing()
    return a


def sized(app, width, height):
    """Put the app on a device of this size and return the board viewport."""
    app.page_width, app.page_height = float(width), float(height)
    app.canvas_w = float(width)
    app.canvas_h = app.board_height
    return app.view


# -- the board fits ---------------------------------------------------------


@pytest.mark.parametrize("name,width,height", DEVICES)
def test_the_whole_board_is_on_screen(app, name, width, height):
    view = sized(app, width, height)
    for node in app.board.nodes:
        x, y = view.node_xy(app.board, node.id)
        assert 0 <= x <= width, f"{name}: intersection off the side"
        assert 0 <= y <= view.height, f"{name}: intersection off the top/bottom"


@pytest.mark.parametrize("name,width,height", DEVICES)
def test_port_jetties_are_not_clipped(app, name, width, height):
    """Ports are drawn beyond the coastline, so they clip before anything else."""
    view = sized(app, width, height)
    radius = view.hex_radius(app.board)
    badge = radius * 0.26
    for slot in range(len(app.board.port_slots)):
        px, py = app.port_badge_xy(slot)
        assert badge <= px <= width - badge, f"{name}: port off the side"
        assert badge <= py <= view.height - badge, f"{name}: port off the top"


@pytest.mark.parametrize("name,width,height", DEVICES)
def test_the_board_takes_a_sensible_share_of_the_screen(app, name, width, height):
    """Big enough to tap, small enough to leave the advice visible."""
    sized(app, width, height)
    assert 280 <= app.board_height <= 520
    assert app.board_height < height * 0.75, f"{name}: board crowds the advice"


# -- you can hit it ---------------------------------------------------------


@pytest.mark.parametrize("name,width,height", DEVICES)
def test_neighbouring_intersections_do_not_overlap(app, name, width, height):
    """
    If two intersections sit closer than the touch radius, tapping one selects
    the other and the board becomes a guessing game.
    """
    view = sized(app, width, height)
    closest = min(
        math.dist(
            view.node_xy(app.board, n.id), view.node_xy(app.board, m)
        )
        for n in app.board.nodes
        for m in n.neighbors
    )
    assert closest > NODE_RADIUS, f"{name}: intersections {closest:.1f}px apart"


@pytest.mark.parametrize("name,width,height", DEVICES)
def test_every_intersection_can_be_tapped(app, name, width, height):
    view = sized(app, width, height)
    for node in app.board.nodes:
        x, y = view.node_xy(app.board, node.id)
        assert view.node_at(app.board, x, y) == node.id


@pytest.mark.parametrize("name,width,height", DEVICES)
def test_every_path_can_be_tapped_along_its_length(app, name, width, height):
    view = sized(app, width, height)
    for edge in app.board.edges:
        ax, ay = view.node_xy(app.board, edge.a)
        bx, by = view.node_xy(app.board, edge.b)
        for t in (0.35, 0.5, 0.65):
            sx, sy = ax + (bx - ax) * t, ay + (by - ay) * t
            assert view.edge_at(app.board, sx, sy) == edge.id


@pytest.mark.parametrize("name,width,height", DEVICES)
def test_every_tile_can_be_tapped(app, name, width, height):
    view = sized(app, width, height)
    for coord in app.board.tiles:
        x, y = view.tile_xy(app.board, coord)
        assert view.tile_at(app.board, x, y) == coord


def test_buttons_clear_the_minimum_tap_size():
    """
    Padding is what makes a button comfortable, and it lives in one place so
    it cannot be forgotten at a call site.
    """
    padding = BUTTON_STYLE.padding
    text_height = 17  # a 13.5–14.5px label, rounded up
    assert padding.top + padding.bottom + text_height >= TAP_TARGET
    assert padding.left >= 12 and padding.right >= 12


def test_every_button_carries_the_shared_style():
    control = btn("Roll", on_click=lambda _e: None)
    assert control.style is BUTTON_STYLE


def test_a_disabled_button_looks_disabled():
    live = btn("Go", on_click=lambda _e: None, primary=True)
    dead = btn("Go", on_click=lambda _e: None, primary=True, disabled=True)
    assert live.bgcolor != dead.bgcolor
    assert live.color != dead.color


# -- the design system is coherent -----------------------------------------


def test_every_resource_has_a_colour_a_label_and_an_icon():
    """
    Icons rather than emoji: an emoji is drawn by whatever font the device
    has, so it changes size and weight between phones.
    """
    for resource in list(Resource) + [None]:
        assert RESOURCE_COLOR[resource].startswith("#")
        assert RESOURCE_LABEL[resource]
        assert RESOURCE_ICON[resource] is not None


def test_the_sea_is_drawn_behind_the_island(app):
    """The hex field continuing past the coast is what makes it read as Catan."""
    view = sized(app, 390, 844)
    with_sea = board_shapes(app.board, view, app.state)
    assert len(with_sea) > len(app.board.tiles) * 3


@pytest.mark.parametrize("name,width,height", DEVICES)
def test_the_screen_builds_at_every_size(app, name, width, height):
    """Every screen, at every device size, without raising."""
    sized(app, width, height)
    app.refresh()
    assert app.root.controls

    app.screen = "config"
    app.refresh()
    assert app.root.controls

    app.screen = "editor"
    app.refresh()
    assert app.root.controls

    app.screen = "game"
    app.refresh()
    for tab in ("advice", "hand", "table"):
        app.tab = tab
        app.refresh()
        assert app.root.controls


def test_the_app_keeps_clear_of_the_system_bars(app):
    """
    Android paints the clock and battery over the top of the window and the
    gesture bar over the bottom. Without a SafeArea the turn banner sits under
    the status bar and is unreadable — something no desktop run reveals.
    """
    root = app.build()
    assert type(root).__name__ == "SafeArea"


@pytest.mark.parametrize("name,width,height", DEVICES)
def test_no_dialog_can_outgrow_the_screen(app, name, width, height):
    """
    An unbounded dialog runs off the bottom of a phone: the rows past the fold
    cannot be reached and the buttons end up painted over them.
    """
    sized(app, width, height)
    body = app._dialog_body(ft_column_of(12))
    assert body.height is not None
    assert body.height <= height * 0.6, f"{name}: dialog taller than the screen"
    assert body.height >= 220
    inner = body.content
    assert inner.scroll is not None, "a bounded dialog has to scroll"


def ft_column_of(rows):
    import flet as ft

    return ft.Column([ft.Text(f"row {i}") for i in range(rows)])


@pytest.mark.parametrize("name,width,height", DEVICES)
def test_every_dialog_the_app_opens_is_bounded(app, name, width, height):
    """Walk the real dialogs rather than trusting the helper is used."""
    sized(app, width, height)
    play_setup(app)
    app.flow.roll(8)
    app.refresh()

    openers = [
        app._roll_dialog,
        app._trade_dialog,
        app._buy_dev_dialog,
        app._monopoly_dialog,
        app._year_of_plenty_dialog,
    ]
    for opener in openers:
        app.page.dialogs.clear()
        opener()
        assert app.page.dialogs, f"{opener.__name__} opened nothing"
        content = app.page.dialogs[-1].content
        assert getattr(content, "height", None) is not None, (
            f"{opener.__name__} is unbounded at {name}"
        )
        assert content.height <= height * 0.6


def play_setup(app):
    from catanmind import rules

    while app.flow.in_setup:
        node = next(
            n for n in rules.legal_settlements(
                app.state, app.flow.current, setup=True
            )
        )
        app.flow.place_setup_settlement(node)
        edge = next(
            e for e in app.board.node_edges[node] if e not in app.state.roads
        )
        app.flow.place_setup_road(edge)


def test_resizing_the_window_resizes_the_board(app):
    small = sized(app, 390, 700).scale
    large = sized(app, 390, 1200).scale
    assert large > small, "a taller window should draw a bigger board"


def test_a_very_short_window_still_keeps_a_usable_board(app):
    """Landscape on a phone is the worst case; the board must not vanish."""
    sized(app, 800, 380)
    assert app.board_height >= 280
    view = app.view
    for node in app.board.nodes:
        x, y = view.node_xy(app.board, node.id)
        assert 0 <= x <= 800
        assert 0 <= y <= view.height
