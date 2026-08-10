"""
The CatanMind screen.

The screen does not decide what is legal — :mod:`catanmind.flow` does. Every
control on it comes from ``flow.available_actions()``, so a step that offers
three moves shows three buttons and a step that offers one shows one. That is
the difference between a companion that walks you through a game and the
free-form recorder this used to be, which let a single player take four free
settlements in a row and offered every action at every moment.

Three screens, in order: choose the table, tap in the board that is actually on
it, then play. Drawing is vector work through ``flet.canvas`` — no matplotlib,
no base64 round-trip — and hit-testing lives in :mod:`catanmind.view`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import flet as ft
import flet.canvas as cv

from catanmind.advisor import (
    Advice,
    SetupAdvisor,
    SetupPlan,
    TurnAdvisor,
    describe_edge,
    describe_edge_end,
)
from catanmind.board import (
    Board,
    Building,
    DevCard,
    Layout,
    NUMBER_TOKENS,
    Port,
    PORT_POOL,
    Resource,
    RESOURCES,
    SPIRAL,
    TILE_POOL,
    pips,
)
from catanmind.flow import Action, Step, TurnFlow
from catanmind import art
from catanmind import splash
from catanmind.scoring import Scorer
from catanmind.state import GameState
from catanmind.tracker import Tracker
from catanmind.view import Viewport
from catanmind import rules

# --------------------------------------------------------------------------
# Palette — "sea and sand"
# --------------------------------------------------------------------------
#
# The colours come from the game's own materials rather than from a generic
# dark theme: the island sits in deep water, the interface furniture is the
# sand and timber of the coast, and the five resources keep their board
# colours so a chip on screen reads as the tile it stands for.

BG = "#08131d"           # open ocean, the page behind everything
SEA = "#0d2233"          # the water the island floats in
SURFACE = "#132c40"      # cards and panels
SURFACE_HI = "#1b3d56"   # raised, selected, pressed
LINE = "#25506e"         # hairline borders

TEXT = "#eef5fa"
MUTED = "#8ba7bf"

ACCENT = "#f0b43a"       # lantern gold: the primary action
ON_ACCENT = "#0a1622"    # text that sits on top of it
SAND = "#e6d3a8"         # parchment, for quieter highlights

GOOD = "#3fb383"         # sea green
WARN = "#e8734a"         # terracotta

#: A move you cannot make yet should read as unavailable at a glance, not as
#: a slightly different shade of available.
DISABLED_BG = "#0f2334"
DISABLED_TEXT = "#4e6c85"

RESOURCE_COLOR: Dict[Optional[Resource], str] = {
    Resource.WOOD: "#2f6b3a",     # forest
    Resource.BRICK: "#b8532c",    # fired clay
    Resource.SHEEP: "#8dc63f",    # pasture
    Resource.WHEAT: "#e8b23a",    # ripe field
    Resource.ORE: "#6b8095",      # slate
    None: "#d9c39a",              # desert sand
}

RESOURCE_LABEL: Dict[Optional[Resource], str] = {
    Resource.WOOD: "Wood",
    Resource.BRICK: "Brick",
    Resource.SHEEP: "Sheep",
    Resource.WHEAT: "Wheat",
    Resource.ORE: "Ore",
    None: "Desert",
}

#: Vector icons, not emoji. Emoji are drawn by whichever font the device
#: happens to have, so they change size, weight and even colour between
#: phones; these are part of the app and look the same everywhere.
RESOURCE_ICON: Dict[Optional[Resource], object] = {
    Resource.WOOD: ft.Icons.FOREST,
    Resource.BRICK: ft.Icons.VIEW_MODULE,
    Resource.SHEEP: ft.Icons.PETS,
    Resource.WHEAT: ft.Icons.GRASS,
    Resource.ORE: ft.Icons.FILTER_HDR,
    None: ft.Icons.LANDSCAPE,
}


def resource_icon(resource: Optional[Resource], size: float = 15) -> ft.Icon:
    return ft.Icon(RESOURCE_ICON[resource], size=size,
                   color=RESOURCE_COLOR[resource])

PLAYER_COLOR: Dict[int, str] = {
    1: "#e05260", 2: "#3f9ae0", 3: "#46b877", 4: "#e8913c",
}

#: Height reserved for the board. Fixed so the canvas cannot resize itself in a
#: loop; everything below it scrolls. Adjusted to the window in `_board_pane`.
BOARD_HEIGHT = 340

#: Anything a finger has to hit is at least this tall.
TAP_TARGET = 44

#: Corner rounding, one step per level of elevation.
RADIUS_CARD = 16
RADIUS_CHIP = 999

#: How the ranked recommendations are titled.
RANK_TITLE: Dict[int, str] = {
    1: "Best opening", 2: "Second choice", 3: "Third choice",
}


def _paint(color: str, *, stroke: float = 0.0) -> ft.Paint:
    if stroke:
        return ft.Paint(
            color=color, stroke_width=stroke, style=ft.PaintingStyle.STROKE
        )
    return ft.Paint(color=color, style=ft.PaintingStyle.FILL)


def _text(
    x: float, y: float, value: str, size: float, color: str,
    bold: bool = True,
) -> cv.Text:
    return cv.Text(
        x, y, value,
        style=ft.TextStyle(
            size=size, color=color,
            weight=ft.FontWeight.BOLD if bold else ft.FontWeight.NORMAL,
        ),
        alignment=ft.Alignment(0, 0),
    )


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def board_shapes(
    board: Board,
    view: Viewport,
    state: Optional[GameState] = None,
    *,
    highlight_nodes: Sequence[int] = (),
    highlight_edges: Sequence[int] = (),
    highlight_tiles: Sequence[Tuple[int, int]] = (),
    legal_nodes: Sequence[int] = (),
    legal_edges: Sequence[int] = (),
    legal_tiles: Sequence[Tuple[int, int]] = (),
    show_node_ids: bool = False,
    pending: Optional[Dict[Tuple[int, int], Tuple[Optional[Resource], int]]] = None,
    ports: Optional[Dict[int, Port]] = None,
) -> List[cv.Shape]:
    """
    Every shape needed to draw the board once.

    ``legal_*`` are the places the armed action could go — drawn faintly so the
    player can see where tapping will work. ``highlight_*`` are the advisor's
    picks, drawn boldly and numbered.

    ``pending`` is the board editor's half-finished layout: tiles it has no
    answer for yet are drawn hollow so the remaining work is obvious.
    """
    shapes: List[cv.Shape] = []
    radius = view.hex_radius(board)

    shapes += _sea_shapes(board, view)
    shapes += art.island_shadow(_island_outline(board, view), radius)

    # -- tiles ------------------------------------------------------------
    # Back to front, so a tile's raised edge overlaps the one behind it. That
    # ordering is what makes the extrusion read as thickness rather than as an
    # outline: sort by screen y and the island stacks like real pieces.
    for coord in sorted(board.tiles, key=lambda c: view.tile_xy(board, c)[1]):
        tile = board.tiles[coord]
        corners = view.tile_corners(board, coord)
        resource, number = tile.resource, tile.number
        known = True
        if pending is not None:
            entry = pending.get(coord)
            known = entry is not None
            resource, number = entry if entry else (None, 0)

        cx, cy = view.tile_xy(board, coord)
        if not known:
            shapes += art.tile_shapes(
                corners, (cx, cy), radius, "#1b3d56", None, with_terrain=False
            )
            shapes.append(_text(cx, cy, "?", radius * 0.46, "#6f93ac"))
            continue

        shapes += art.tile_shapes(
            corners, (cx, cy), radius, RESOURCE_COLOR[resource], resource
        )
        if number:
            shapes += art.number_token(
                (cx, cy), radius * 0.30, number, pips(number)
            )

    # -- tappable tiles ----------------------------------------------------
    for coord in legal_tiles:
        cx, cy = view.tile_xy(board, coord)
        shapes.append(
            cv.Circle(cx, cy, radius * 0.60, paint=_paint("#66ffc85c", stroke=2.0))
        )

    for coord in highlight_tiles:
        cx, cy = view.tile_xy(board, coord)
        shapes.append(
            cv.Circle(cx, cy, radius * 0.72, paint=_paint(ACCENT, stroke=3.5))
        )

    # -- ports -------------------------------------------------------------
    # Drawn the way the physical board shows them: a jetty out at sea with two
    # planks running back to the exact intersections that can use it. Drawn
    # before the early return so the editor shows them too — entering a board
    # blind to where the harbours are is guesswork.
    port_map = board.layout.ports if ports is None else ports
    for slot, edge_id in enumerate(board.port_slots):
        port = port_map.get(slot)
        if port is None:
            continue
        edge = board.edge(edge_id)
        ax, ay = view.node_xy(board, edge.a)
        bx, by = view.node_xy(board, edge.b)
        ex, ey = view.edge_xy(board, edge_id)
        cx, cy = view.to_screen(0, 0)
        dx, dy = ex - cx, ey - cy
        length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        px = ex + dx / length * radius * 0.50
        py = ey + dy / length * radius * 0.50

        colour = RESOURCE_COLOR.get(port.resource, "#5b6b80")
        for nx, ny in ((ax, ay), (bx, by)):
            shapes.append(
                cv.Line(px, py, nx, ny, paint=_paint(colour, stroke=2.5))
            )
        badge = radius * 0.26
        shapes.append(cv.Circle(px, py, badge, paint=_paint("#0d1520")))
        shapes.append(cv.Circle(px, py, badge, paint=_paint(colour, stroke=2.5)))
        if port is Port.GENERIC:
            shapes.append(_text(px, py, "3:1", badge * 0.80, "#e8eaf0"))
        else:
            res = port.resource
            assert res is not None
            shapes.append(
                _text(px, py - badge * 0.28, "2:1", badge * 0.62, "#e8eaf0")
            )
            shapes.append(
                _text(px, py + badge * 0.34, res.value[:4].upper(),
                      badge * 0.50, colour)
            )

    if state is None:
        if show_node_ids:
            shapes += _node_id_shapes(board, view)
        return shapes

    # -- tappable paths ----------------------------------------------------
    for edge_id in legal_edges:
        edge = board.edge(edge_id)
        ax, ay = view.node_xy(board, edge.a)
        bx, by = view.node_xy(board, edge.b)
        shapes.append(
            cv.Line(ax, ay, bx, by, paint=_paint("#59ffc85c", stroke=5.0))
        )

    # -- roads -------------------------------------------------------------
    for edge_id, owner in state.roads.items():
        edge = board.edge(edge_id)
        shapes += art.road(
            view.node_xy(board, edge.a), view.node_xy(board, edge.b),
            max(4.0, radius * 0.13), PLAYER_COLOR[owner],
        )

    for edge_id in highlight_edges:
        edge = board.edge(edge_id)
        ax, ay = view.node_xy(board, edge.a)
        bx, by = view.node_xy(board, edge.b)
        shapes.append(
            cv.Line(ax, ay, bx, by, paint=_paint(ACCENT, stroke=4.5))
        )

    # -- the robber --------------------------------------------------------
    rx, ry = view.tile_xy(board, state.robber)
    shapes += art.robber((rx, ry - radius * 0.10), radius * 0.26)

    # -- tappable intersections -------------------------------------------
    for node_id in legal_nodes:
        x, y = view.node_xy(board, node_id)
        shapes.append(
            cv.Circle(x, y, radius * 0.15, paint=_paint(art.alpha(ACCENT, 0.30)))
        )
        shapes.append(
            cv.Circle(x, y, radius * 0.15,
                      paint=_paint(art.alpha(ACCENT, 0.85), stroke=1.6))
        )

    # -- buildings ---------------------------------------------------------
    # Painted back to front so a piece lower on the board sits in front of one
    # behind it, the same reason the tiles are ordered.
    for node_id, (owner, kind) in sorted(
        state.buildings.items(), key=lambda item: view.node_xy(board, item[0])[1]
    ):
        x, y = view.node_xy(board, node_id)
        colour = PLAYER_COLOR[owner]
        if kind is Building.CITY:
            shapes += art.city((x, y), radius * 0.22, colour)
        else:
            shapes += art.settlement((x, y), radius * 0.19, colour)

    # -- the advisor's picks, numbered ------------------------------------
    for rank, node_id in enumerate(highlight_nodes):
        x, y = view.node_xy(board, node_id)
        badge = radius * 0.26
        shapes.append(
            cv.Shadow(
                path=[cv.Path.Oval(x - badge, y - badge + badge * 0.2,
                                   badge * 2, badge * 2)],
                color=art.alpha("000000", 0.55), elevation=badge * 0.6,
            )
        )
        shapes.append(
            cv.Circle(x, y, badge, paint=ft.Paint(
                gradient=ft.PaintRadialGradient(
                    center=ft.Offset(x - badge * 0.35, y - badge * 0.4),
                    radius=badge * 1.8,
                    colors=[art.shade(ACCENT, 0.35), ACCENT,
                            art.shade(ACCENT, -0.25)],
                    color_stops=[0.0, 0.55, 1.0],
                ),
                style=ft.PaintingStyle.FILL,
            ))
        )
        shapes.append(
            cv.Circle(x, y, badge,
                      paint=_paint(art.alpha("000000", 0.35), stroke=badge * 0.12))
        )
        shapes.append(_text(x, y, str(rank + 1), badge * 1.15, ON_ACCENT))

    if show_node_ids:
        shapes += _node_id_shapes(board, view)
    return shapes


def _island_outline(board: Board, view: Viewport) -> List[Tuple[float, float]]:
    """
    The coastline, as a ring of points.

    Walking the coastal edges gives the true silhouette, which is what the
    island's drop shadow is cast from — a bounding shape would leave shadow
    hanging in the bays.
    """
    ring: List[Tuple[float, float]] = []
    for edge_id in board.coastal_ring:
        edge = board.edge(edge_id)
        point = view.node_xy(board, edge.a)
        if not ring or ring[-1] != point:
            ring.append(point)
    return ring or [view.to_screen(0, 0)]


def _sea_shapes(board: Board, view: Viewport) -> List[cv.Shape]:
    """
    The water the island sits in.

    Drawn as the hex field continuing past the coast, faintly. It costs one
    outline per empty hex and it is the single strongest cue that this is
    Catan and not a generic board of coloured shapes.
    """
    from catanmind.board import hex_distance

    out: List[cv.Shape] = []
    faint = _paint("#14304a", stroke=1.0)
    size = board.SIZE * view.scale
    corner_angles = [math.radians(60 * i - 90) for i in range(6)]
    reach = 4

    for q in range(-reach, reach + 1):
        for r in range(-reach, reach + 1):
            if hex_distance((q, r), (0, 0)) > reach or (q, r) in board.tiles:
                continue
            cx, cy = view.to_screen(*board.tile_center((q, r)))
            # Anything this far out cannot have a single pixel on screen.
            if not -size <= cx <= view.width + size:
                continue
            if not -size <= cy <= view.height + size:
                continue
            corners = [
                (cx + size * math.cos(a), cy + size * math.sin(a))
                for a in corner_angles
            ]
            elements = [cv.Path.MoveTo(*corners[0])]
            elements += [cv.Path.LineTo(*c) for c in corners[1:]]
            elements.append(cv.Path.Close())
            out.append(cv.Path(elements=elements, paint=faint))
    return out


def _node_id_shapes(board: Board, view: Viewport) -> List[cv.Shape]:
    out: List[cv.Shape] = []
    for node in board.nodes:
        x, y = view.node_xy(board, node.id)
        out.append(_text(x, y, str(node.id), 9, MUTED, bold=False))
    return out


# --------------------------------------------------------------------------
# Small building blocks
# --------------------------------------------------------------------------


def chip(label: str, color: str = SURFACE_HI, text_color: str = TEXT) -> ft.Container:
    return ft.Container(
        content=ft.Text(label, size=11.5, color=text_color,
                        weight=ft.FontWeight.W_600),
        bgcolor=color,
        padding=ft.Padding(10, 4, 10, 4),
        border_radius=RADIUS_CHIP,
    )


def tile_chip(resource: Optional[Resource], number: int) -> ft.Container:
    """
    A resource/number badge — how a player reads a spot at a glance.

    Carries the tile's own colour and the token's own look, red for the two
    numbers that come up most, so the chip and the board agree.
    """
    hot = number in (6, 8)
    return ft.Container(
        content=ft.Row(
            [
                resource_icon(resource, 15),
                ft.Text(
                    RESOURCE_LABEL[resource] if resource else "Desert",
                    size=11.5, color=TEXT, weight=ft.FontWeight.W_500,
                ),
                ft.Container(
                    content=ft.Text(
                        str(number) if number else "—", size=11,
                        weight=ft.FontWeight.BOLD,
                        color="#c62828" if hot else "#1a1a1a",
                    ),
                    bgcolor="#f4efe2", width=20, height=20,
                    border_radius=RADIUS_CHIP,
                    alignment=ft.Alignment(0, 0),
                ),
            ],
            spacing=5, tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        bgcolor=SURFACE_HI,
        padding=ft.Padding(8, 4, 6, 4),
        border_radius=RADIUS_CHIP,
        border=ft.Border.all(1, RESOURCE_COLOR[resource]),
    )


def label(text: str) -> ft.Text:
    """The small heading above a group. Quiet, never shouting."""
    return ft.Text(text.upper(), size=10.5, color=MUTED,
                   weight=ft.FontWeight.BOLD)


#: Cards sit above the page rather than being painted on it. A soft, wide
#: shadow with a slight downward offset reads as elevation; a hard one reads
#: as a mistake.
CARD_SHADOW = ft.BoxShadow(
    spread_radius=0, blur_radius=18,
    color="#4d000000", offset=ft.Offset(0, 6),
)

LIFT_SHADOW = ft.BoxShadow(
    spread_radius=0, blur_radius=22,
    color="#59000000", offset=ft.Offset(0, 8),
)


def surface_gradient(top: str, bottom: str) -> ft.LinearGradient:
    """Light from above, matching the board's own lighting."""
    return ft.LinearGradient(
        begin=ft.Alignment(0, -1), end=ft.Alignment(0, 1), colors=[top, bottom]
    )


def section(
    title: str, *controls: ft.Control, accent: Optional[str] = None
) -> ft.Container:
    """A titled card. ``accent`` draws a coloured spine down the left edge."""
    return ft.Container(
        content=ft.Column([label(title)] + list(controls), spacing=8),
        gradient=surface_gradient(art.shade(SURFACE, 0.05), SURFACE),
        border_radius=RADIUS_CARD,
        padding=14,
        shadow=CARD_SHADOW,
        border=(
            ft.Border(left=ft.BorderSide(3, accent)) if accent
            else ft.Border.all(1, LINE)
        ),
    )


def _wordmark(size: float = 30) -> ft.Control:
    """
    The app's name, wearing a hex.

    A small drawn tile rather than an image file: it scales cleanly, needs no
    asset pipeline, and repeats the shape the whole app is built out of.
    """
    badge = size * 1.35
    hexagon = cv.Canvas(
        shapes=[
            cv.Path(
                elements=(
                    [cv.Path.MoveTo(
                        badge / 2 + badge * 0.46 * math.cos(math.radians(-90)),
                        badge / 2 + badge * 0.46 * math.sin(math.radians(-90)),
                    )]
                    + [
                        cv.Path.LineTo(
                            badge / 2 + badge * 0.46 * math.cos(math.radians(60 * i - 90)),
                            badge / 2 + badge * 0.46 * math.sin(math.radians(60 * i - 90)),
                        )
                        for i in range(1, 6)
                    ]
                    + [cv.Path.Close()]
                ),
                paint=_paint(ACCENT),
            ),
            _text(badge / 2, badge / 2, "C", size * 0.62, ON_ACCENT),
        ],
        width=badge, height=badge,
    )
    return ft.Row(
        [
            ft.Container(content=hexagon, width=badge, height=badge),
            ft.Text("CatanMind", size=size, weight=ft.FontWeight.BOLD,
                    color=TEXT),
        ],
        spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


def _resource_strip() -> ft.Control:
    """The five resources, as a quiet footer. Sets the game's colours early."""
    return ft.Row(
        [
            ft.Container(
                content=ft.Column(
                    [
                        resource_icon(r, 20),
                        ft.Text(RESOURCE_LABEL[r], size=9.5, color=MUTED),
                    ],
                    spacing=2,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                bgcolor=SURFACE,
                border=ft.Border(bottom=ft.BorderSide(3, RESOURCE_COLOR[r])),
                border_radius=10,
                padding=ft.Padding(10, 8, 10, 6),
                expand=True,
            )
            for r in RESOURCES
        ],
        spacing=6,
    )


#: Every button in the app shares this shape: a full pill, and enough padding
#: that the tap area clears :data:`TAP_TARGET` without the caller thinking
#: about it.
BUTTON_STYLE = ft.ButtonStyle(
    shape=ft.RoundedRectangleBorder(radius=RADIUS_CHIP),
    # 14 + 14 either side of a ~17px label clears TAP_TARGET with a pixel to
    # spare. At 13 it came to 43, which is close enough to feel fine on a
    # desktop and not quite enough under a thumb.
    padding=ft.Padding(18, 14, 18, 14),
)


def btn(
    content=None,
    *,
    on_click=None,
    bgcolor: Optional[str] = None,
    color: Optional[str] = None,
    disabled: bool = False,
    tooltip: Optional[str] = None,
    primary: bool = False,
    **kwargs,
) -> ft.Button:
    """
    A button in the app's own shape.

    One helper rather than a style argument repeated at thirty call sites:
    the padding here is what guarantees a comfortable tap target on a phone,
    and it should not be possible to forget it.
    """
    if primary:
        bgcolor = bgcolor or ACCENT
        color = color or ON_ACCENT
    if disabled:
        # Explicit, because a bgcolor set here would otherwise override the
        # theme's own disabled shade and leave the button looking live.
        bgcolor, color = DISABLED_BG, DISABLED_TEXT
    return ft.Button(
        content,
        on_click=on_click,
        bgcolor=bgcolor or SURFACE_HI,
        color=color or TEXT,
        disabled=disabled,
        tooltip=tooltip,
        style=BUTTON_STYLE,
        **kwargs,
    )


# --------------------------------------------------------------------------
# The application
# --------------------------------------------------------------------------


class CatanMind:
    """Holds the game, the advisors, and the widget tree."""

    def __init__(self, page: ft.Page, num_players: int = 4, me: int = 1):
        self.page = page
        self.board = Board(Layout.standard())
        self.state = GameState(self.board, num_players=num_players, me=me)
        self._rebuild_engines()

        self.screen = "config"          # config -> editor -> game
        self.pending_action: Optional[Action] = None
        self.show_ids = False
        self.canvas_w = 360.0
        self.canvas_h = float(BOARD_HEIGHT)
        #: Window size, so the board can take a sensible share of it. Set from
        #: the page on resize; the defaults are a common phone.
        self.page_height: float = 844.0
        self.page_width: float = 390.0

        self.pending: Dict[Tuple[int, int], Tuple[Optional[Resource], int]] = {}
        self.pending_ports: Dict[int, Port] = dict(enumerate(PORT_POOL))

        self.status = ""
        self.tab = "advice"

        self.canvas = cv.Canvas(shapes=[], expand=True, on_resize=self._on_resize)
        self.board_holder = ft.GestureDetector(
            content=self.canvas, on_tap_down=self._on_tap
        )
        self.root = ft.Column(spacing=0, expand=True)

    # -- engines -----------------------------------------------------------

    def _rebuild_engines(self) -> None:
        self.scorer = Scorer(self.board)
        self.setup_advisor = SetupAdvisor(self.board, self.scorer)
        self.turn_advisor = TurnAdvisor(self.board, self.scorer)
        self.tracker = Tracker(self.state)
        self.flow = TurnFlow(self.state)

    @property
    def view(self) -> Viewport:
        return Viewport.fit(self.board, self.canvas_w, self.canvas_h)

    @property
    def me(self) -> int:
        return self.state.me

    # -- shell -------------------------------------------------------------

    def build(self) -> ft.Control:
        """
        The app, kept clear of the system furniture.

        Android draws the clock, signal and battery over the top of the window
        and the gesture bar over the bottom. Without this the turn banner sits
        underneath the status bar and is unreadable on a real phone — which no
        amount of desktop testing shows you.
        """
        return ft.SafeArea(content=self.root, expand=True)

    def refresh(self) -> None:
        """Rebuild the whole screen from state. The single update path."""
        if self.screen == "config":
            self.root.controls = [self._screen_config()]
        elif self.screen == "editor":
            self.root.controls = [self._screen_editor()]
        else:
            self.flow = TurnFlow(self.state)
            self.root.controls = [self._screen_game()]
        self.page.update()

    def _on_resize(self, e) -> None:
        width = getattr(e, "width", 0) or 0
        height = getattr(e, "height", 0) or 0
        if width <= 0 or height <= 0:
            return
        if abs(width - self.canvas_w) < 1 and abs(height - self.canvas_h) < 1:
            return
        self.canvas_w, self.canvas_h = float(width), float(height)
        self.refresh()

    def _board_pane(self, **draw) -> ft.Control:
        """
        The island and the water around it.

        Sits on its own darker surface with a hairline coast, so the board
        reads as a place rather than as another panel in the stack.
        """
        self.canvas.shapes = board_shapes(
            self.board, self.view,
            None if self.screen == "editor" else self.state,
            show_node_ids=self.show_ids, **draw,
        )
        return ft.Container(
            content=self.board_holder,
            bgcolor=SEA,
            height=self.board_height,
            padding=0,
            border=ft.Border(
                top=ft.BorderSide(1, LINE), bottom=ft.BorderSide(1, LINE)
            ),
        )

    @property
    def board_height(self) -> float:
        """
        How tall the board should be on this screen.

        A phone wants roughly two fifths of the height — enough to tap
        confidently, with the advice still visible underneath. A desktop
        window has room to spare, so the board is allowed to grow, but never
        so far that it pushes everything else off the fold.
        """
        available = self.page_height or 844
        return float(max(280, min(520, available * 0.42)))

    # ------------------------------------------------------------------
    # Screen 1 — the table
    # ------------------------------------------------------------------

    def _screen_config(self) -> ft.Control:
        def set_players(n: int) -> None:
            me = min(self.state.me, n)
            self.state = GameState(self.board, num_players=n, me=me)
            self._rebuild_engines()
            self.refresh()

        def set_seat(seat: int) -> None:
            self.state.me = seat
            self.refresh()

        n = self.state.num_players
        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(height=8),
                    _wordmark(),
                    ft.Text(
                        "Your seat at the table, one move ahead.",
                        size=14, color=SAND,
                    ),
                    ft.Text(
                        "Tell me about the table, then tap in the board in "
                        "front of you.",
                        size=13, color=MUTED,
                    ),
                    ft.Container(height=10),
                    section(
                        "How many players?",
                        ft.Row(
                            [
                                btn(
                                    str(count),
                                    bgcolor=ACCENT if n == count else SURFACE_HI,
                                    color=ON_ACCENT if n == count else TEXT,
                                    on_click=lambda _e, c=count: set_players(c),
                                )
                                for count in (3, 4)
                            ],
                            spacing=8,
                        ),
                    ),
                    section(
                        "Which seat are you?",
                        ft.Text(
                            "Seat order decides who places first in setup.",
                            size=11, color=MUTED,
                        ),
                        ft.Row(
                            [
                                btn(
                                    f"Seat {seat}",
                                    bgcolor=(
                                        PLAYER_COLOR[seat] if self.me == seat
                                        else SURFACE_HI
                                    ),
                                    color="#ffffff" if self.me == seat else TEXT,
                                    on_click=lambda _e, s=seat: set_seat(s),
                                )
                                for seat in range(1, n + 1)
                            ],
                            wrap=True, spacing=8, run_spacing=8,
                        ),
                    ),
                    ft.Container(height=14),
                    ft.Row(
                        [
                            ft.Container(
                                content=btn(
                                    "Next — enter the board",
                                    primary=True,
                                    on_click=lambda _e: self._go_editor(),
                                ),
                                expand=True,
                            )
                        ],
                    ),
                    ft.Container(height=18),
                    _resource_strip(),
                ],
                spacing=10, scroll=ft.ScrollMode.AUTO,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
            padding=ft.Padding(20, 16, 20, 20),
            expand=True,
        )

    def _go_editor(self) -> None:
        self.screen = "editor"
        self.status = "Tap each tile to enter what is on it."
        self.refresh()

    # ------------------------------------------------------------------
    # Screen 2 — the board editor
    # ------------------------------------------------------------------

    def _screen_editor(self) -> ft.Control:
        done = len(self.pending)
        problems = self._editor_problems()
        remaining_res = self._remaining_resources()
        remaining_num = self._remaining_numbers()

        header = ft.Container(
            content=ft.Row(
                [
                    ft.Text("Enter the board", size=17,
                            weight=ft.FontWeight.BOLD, color=TEXT),
                    ft.Container(expand=True),
                    chip(f"{done}/19 tiles", GOOD if done == 19 else SURFACE_HI),
                ],
            ),
            bgcolor=SURFACE, padding=ft.Padding(12, 10, 12, 10),
        )

        panel = ft.Column(
            [
                section(
                    "Still to place",
                    ft.Row(
                        [
                            chip(
                                f"{RESOURCE_LABEL[r]} ×{remaining_res.get(r, 0)}",
                                RESOURCE_COLOR[r] if remaining_res.get(r, 0)
                                else SURFACE_HI,
                                "#ffffff" if remaining_res.get(r, 0) else MUTED,
                            )
                            for r in list(RESOURCES) + [None]
                        ],
                        wrap=True, spacing=6, run_spacing=6,
                    ),
                    ft.Row(
                        [
                            chip(f"{num}×{count}")
                            for num, count in sorted(remaining_num.items()) if count
                        ],
                        wrap=True, spacing=6, run_spacing=6,
                    ),
                ),
                section(
                    "Checks",
                    ft.Column(
                        [ft.Text("• " + p, size=12, color=WARN) for p in problems],
                        spacing=2,
                    ) if problems
                    else ft.Text("Board is a legal setup.", size=12, color=GOOD),
                ),
                ft.Row(
                    [
                        btn("Standard", on_click=self._use_standard,
                                  bgcolor=SURFACE_HI, color=TEXT),
                        btn("Random", on_click=self._use_random,
                                  bgcolor=SURFACE_HI, color=TEXT),
                        btn("Clear", on_click=self._clear_board,
                                  bgcolor=SURFACE_HI, color=TEXT),
                        btn("Ports", on_click=self._open_port_editor,
                                  bgcolor=SURFACE_HI, color=TEXT),
                    ],
                    wrap=True, spacing=8, run_spacing=8,
                ),
                btn(
                    "Start the game", on_click=self._finish_editing,
                    bgcolor=ACCENT if not problems else SURFACE_HI,
                    color=ON_ACCENT if not problems else MUTED,
                    disabled=bool(problems),
                ),
            ],
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        return ft.Column(
            [
                header,
                ft.Container(
                    content=ft.Text(self.status, size=12, color=MUTED),
                    padding=ft.Padding(12, 6, 12, 6),
                ),
                self._board_pane(pending=self.pending, ports=self.pending_ports),
                ft.Container(content=panel,
                             padding=ft.Padding(12, 8, 12, 24)),
            ],
            spacing=0, expand=True, scroll=ft.ScrollMode.AUTO,
        )

    def _remaining_resources(self) -> Dict[Optional[Resource], int]:
        counts: Dict[Optional[Resource], int] = {
            r: TILE_POOL.count(r) for r in RESOURCES
        }
        counts[None] = 1
        for res, _ in self.pending.values():
            counts[res] = counts.get(res, 0) - 1
        return {r: max(0, c) for r, c in counts.items()}

    def _remaining_numbers(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for n in NUMBER_TOKENS:
            counts[n] = counts.get(n, 0) + 1
        for _, num in self.pending.values():
            if num:
                counts[num] = counts.get(num, 0) - 1
        return {n: max(0, c) for n, c in counts.items()}

    def _editor_problems(self) -> List[str]:
        layout = Layout(tiles=dict(self.pending), ports=dict(self.pending_ports))
        problems = layout.validate()
        if not problems:
            problems = [
                w + " (legal, but unusual)" for w in layout.warnings(self.board)
            ]
        return problems

    def _edit_tile(self, coord: Tuple[int, int]) -> None:
        current = self.pending.get(coord, (None, 0))
        chosen: Dict[str, object] = {"resource": current[0], "number": current[1]}
        index = SPIRAL.index(coord) + 1
        res_row = ft.Row(wrap=True, spacing=6, run_spacing=6)
        num_row = ft.Row(wrap=True, spacing=6, run_spacing=6)

        def redraw() -> None:
            res_row.controls = [
                btn(
                    RESOURCE_LABEL[r],
                    bgcolor=RESOURCE_COLOR[r] if chosen["resource"] is r else SURFACE_HI,
                    color="#ffffff" if chosen["resource"] is r else TEXT,
                    on_click=lambda _e, r=r: pick_resource(r),
                )
                for r in list(RESOURCES) + [None]
            ]
            desert = chosen["resource"] is None
            num_row.controls = [
                btn(
                    str(n),
                    bgcolor=ACCENT if chosen["number"] == n else SURFACE_HI,
                    color=ON_ACCENT if chosen["number"] == n else TEXT,
                    disabled=desert,
                    on_click=lambda _e, n=n: pick_number(n),
                )
                for n in (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
            ]
            self.page.update()

        def pick_resource(r: Optional[Resource]) -> None:
            chosen["resource"] = r
            if r is None:
                chosen["number"] = 0
            redraw()

        def pick_number(n: int) -> None:
            chosen["number"] = n
            redraw()

        def save(_e) -> None:
            if chosen["resource"] is None:
                self.pending[coord] = (None, 0)
            elif not chosen["number"]:
                self.status = "Pick a number token for that tile."
                self.page.pop_dialog()
                self.refresh()
                return
            else:
                self.pending[coord] = (chosen["resource"], int(chosen["number"]))
            self.page.pop_dialog()
            self.refresh()

        def clear(_e) -> None:
            self.pending.pop(coord, None)
            self.page.pop_dialog()
            self.refresh()

        redraw()
        self.page.show_dialog(
            ft.AlertDialog(
                modal=True, bgcolor=SURFACE,
                title=ft.Text(f"Tile {index} of 19", color=TEXT),
                content=self._dialog_body(
                    ft.Column(
                        [
                            ft.Text("Resource", size=12, color=MUTED), res_row,
                            ft.Text("Number", size=12, color=MUTED), num_row,
                        ],
                        tight=True, spacing=8,
                    ),
                    340,
                ),
                actions=[
                    ft.TextButton("Clear", on_click=clear),
                    ft.TextButton("Cancel", on_click=lambda _e: self._close_dialog()),
                    btn("Save", on_click=save, bgcolor=ACCENT, color=ON_ACCENT),
                ],
            )
        )
        self.page.update()

    #: Short labels for the port chips, in the order they are offered.
    PORT_CHOICES: Tuple[Tuple[Port, str], ...] = (
        (Port.GENERIC, "3:1"),
        (Port.WOOD, "Wood"),
        (Port.BRICK, "Brick"),
        (Port.SHEEP, "Sheep"),
        (Port.WHEAT, "Wheat"),
        (Port.ORE, "Ore"),
    )

    def set_port(self, slot: int, port: Port) -> None:
        self.pending_ports[slot] = port

    def port_badge_xy(self, slot: int) -> Tuple[float, float]:
        """Screen position of a port's jetty — the thing the player taps."""
        view = self.view
        radius = view.hex_radius(self.board)
        ex, ey = view.edge_xy(self.board, self.board.port_slots[slot])
        cx, cy = view.to_screen(0, 0)
        dx, dy = ex - cx, ey - cy
        length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        return (ex + dx / length * radius * 0.50, ey + dy / length * radius * 0.50)

    def _port_slot_at(self, sx: float, sy: float) -> Optional[int]:
        radius = self.view.hex_radius(self.board)
        reach = (radius * 0.34) ** 2
        best, best_d = None, reach
        for slot in range(len(self.board.port_slots)):
            px, py = self.port_badge_xy(slot)
            d = (px - sx) ** 2 + (py - sy) ** 2
            if d <= best_d:
                best, best_d = slot, d
        return best

    def _edit_port(self, slot: int) -> None:
        """Pick the type for one port, tapped straight off the board."""
        def pick(port: Port) -> None:
            self.set_port(slot, port)
            self.page.pop_dialog()
            self.status = (
                f"Port {slot + 1} set to "
                + ("3:1 any" if port is Port.GENERIC else f"2:1 {port.value}")
            )
            self.refresh()

        current = self.pending_ports.get(slot, Port.GENERIC)
        self.page.show_dialog(
            ft.AlertDialog(
                modal=True, bgcolor=SURFACE,
                title=ft.Text(f"Port {slot + 1}", color=TEXT),
                content=self._dialog_body(
                    ft.Row(
                        [
                            btn(
                                label,
                                bgcolor=(
                                    RESOURCE_COLOR.get(port.resource, ACCENT)
                                    if current is port else SURFACE_HI
                                ),
                                color="#ffffff" if current is port else TEXT,
                                on_click=lambda _e, p=port: pick(p),
                            )
                            for port, label in self.PORT_CHOICES
                        ],
                        wrap=True, spacing=6, run_spacing=6,
                    ),
                ),
                actions=[
                    ft.TextButton("Cancel", on_click=lambda _e: self._close_dialog())
                ],
            )
        )
        self.page.update()

    def _open_port_editor(self, _e=None) -> None:
        """
        Nine rows of tap chips.

        A dropdown per port would be six options behind two taps each, on a
        control whose keyword arguments moved between Flet versions. Chips are
        one tap, always visible, and use nothing but a button.
        """
        rows = ft.Column(tight=True, spacing=10, scroll=ft.ScrollMode.AUTO,
                         height=340)

        def pick(slot: int, port: Port) -> None:
            self.set_port(slot, port)
            redraw()

        def redraw() -> None:
            rows.controls = []
            for slot in range(len(PORT_POOL)):
                current = self.pending_ports.get(slot, Port.GENERIC)
                rows.controls.append(
                    ft.Column(
                        [
                            ft.Text(f"Port {slot + 1}", size=11, color=MUTED),
                            ft.Row(
                                [
                                    btn(
                                        label,
                                        bgcolor=(
                                            RESOURCE_COLOR.get(port.resource, ACCENT)
                                            if current is port else SURFACE_HI
                                        ),
                                        color=(
                                            "#ffffff" if current is port else TEXT
                                        ),
                                        on_click=(
                                            lambda _e, s=slot, p=port: pick(s, p)
                                        ),
                                    )
                                    for port, label in self.PORT_CHOICES
                                ],
                                wrap=True, spacing=4, run_spacing=4,
                            ),
                        ],
                        spacing=3, tight=True,
                    )
                )
            self.page.update()

        redraw()
        self.page.show_dialog(
            ft.AlertDialog(
                modal=True, bgcolor=SURFACE,
                title=ft.Text("Which port is where?", color=TEXT),
                content=self._dialog_body(rows),
                actions=[
                    btn("Done", on_click=lambda _e: self._close_dialog(),
                              bgcolor=ACCENT, color=ON_ACCENT)
                ],
            )
        )
        self.page.update()

    def _dialog_body(self, content: ft.Control, width: float = 330) -> ft.Control:
        """
        Wrap a dialog's contents so it can never outgrow the screen.

        An unbounded dialog body simply runs off the bottom on a phone: the
        rows past the fold are unreachable and the action buttons end up
        painted on top of them. Bounding the height and letting it scroll is
        the whole fix.
        """
        available = self.page_height or 844
        return ft.Container(
            content=ft.Column(
                [content], scroll=ft.ScrollMode.AUTO, tight=True,
            ),
            width=width,
            height=max(220.0, min(430.0, available * 0.52)),
        )

    def _close_dialog(self) -> None:
        self.page.pop_dialog()
        self.refresh()

    def _use_standard(self, _e=None) -> None:
        layout = Layout.standard()
        self.pending, self.pending_ports = dict(layout.tiles), dict(layout.ports)
        self.status = "Loaded the standard beginner layout."
        self.refresh()

    def _use_random(self, _e=None) -> None:
        layout = Layout.random()
        self.pending, self.pending_ports = dict(layout.tiles), dict(layout.ports)
        self.status = "Generated a random legal layout."
        self.refresh()

    def _clear_board(self, _e=None) -> None:
        self.pending = {}
        self.status = "Cleared. Tap a tile to set what is on it."
        self.refresh()

    def _finish_editing(self, _e=None) -> None:
        layout = Layout(tiles=dict(self.pending), ports=dict(self.pending_ports))
        problems = layout.validate()
        if problems:
            self.status = "Fix the board first: " + problems[0]
            self.refresh()
            return
        self.state.set_layout(layout)
        self.board = self.state.board
        self._rebuild_engines()
        self.screen = "game"
        self.status = ""
        self.refresh()

    # ------------------------------------------------------------------
    # Screen 3 — the game
    # ------------------------------------------------------------------

    def _screen_game(self) -> ft.Control:
        self._arm_default_action()
        draw = self._draw_targets()

        # One scroll for the whole screen. Everything below the board used to
        # live in a short box with its own scrollbar, which on a phone meant
        # dragging a two-inch window to read a five-inch panel.
        return ft.Column(
            [
                self._turn_banner(),
                self._board_pane(**draw),
                self._prompt_bar(),
                ft.Container(
                    content=ft.Column(
                        [self._tab_bar()] + self._tab_body(),
                        spacing=10,
                        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                    ),
                    padding=ft.Padding(12, 8, 12, 28),
                ),
            ],
            spacing=0, expand=True, scroll=ft.ScrollMode.AUTO,
        )

    def _turn_banner(self) -> ft.Control:
        flow = self.flow
        mine = flow.is_my_turn()
        colour = PLAYER_COLOR.get(flow.current, SURFACE_HI)
        scores = rules.scores(self.state, include_hidden=False)

        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Container(width=10, height=10, border_radius=5,
                                         bgcolor=colour),
                            ft.Text(
                                flow.banner(), size=15,
                                weight=ft.FontWeight.BOLD,
                                color=TEXT if mine else MUTED, expand=True,
                            ),
                            ft.IconButton(
                                icon=ft.Icons.UNDO, icon_color=TEXT,
                                tooltip="Undo", on_click=self._undo,
                            ),
                        ],
                        spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Row(
                        [
                            chip(
                                f"P{p}{' (you)' if p == self.me else ''} · "
                                f"{scores[p]} VP",
                                PLAYER_COLOR[p] if p == self.flow.current
                                else SURFACE_HI,
                                "#ffffff" if p == self.flow.current else MUTED,
                            )
                            for p in sorted(self.state.players)
                        ],
                        wrap=True, spacing=6, run_spacing=4,
                    ),
                ],
                spacing=6,
            ),
            bgcolor=SURFACE,
            border=ft.Border(left=ft.BorderSide(4, colour)),
            padding=ft.Padding(12, 10, 8, 10),
        )

    def _prompt_bar(self) -> ft.Control:
        """The armed action, or the buttons that arm one."""
        flow = self.flow
        actions = flow.available_actions()

        if self.pending_action is not None:
            a = self.pending_action
            return ft.Container(
                content=ft.Row(
                    [
                        ft.Icon(ft.Icons.TOUCH_APP, color=ACCENT, size=18),
                        ft.Text(a.hint or a.label, size=13, color=ACCENT,
                                weight=ft.FontWeight.BOLD, expand=True),
                        ft.TextButton("Cancel", on_click=self._cancel_action),
                    ],
                    spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                bgcolor="#2a2412", padding=ft.Padding(12, 6, 6, 6),
            )

        # The expected move goes on its own row at full width; everything else
        # wraps below it. On a phone that puts the button you almost always
        # want under your thumb, at a size you cannot miss.
        primary = [a for a in actions if a.primary and a.enabled]
        rest = [a for a in actions if a not in primary]

        def button(action: Action, wide: bool = False) -> ft.Control:
            control = btn(
                action.label,
                primary=action.primary and action.enabled,
                disabled=not action.enabled,
                tooltip=action.hint or None,
                on_click=lambda _e, a=action: self._choose_action(a),
            )
            return ft.Container(content=control, expand=True) if wide else control

        rows: List[ft.Control] = []
        if primary:
            rows.append(
                ft.Row([button(a, wide=True) for a in primary], spacing=8)
            )
        if rest:
            rows.append(
                ft.Row([button(a) for a in rest], wrap=True, spacing=8,
                       run_spacing=8)
            )

        note = self.status or (
            "" if flow.is_my_turn()
            else f"Record what Player {flow.current} actually did."
        )
        if note:
            rows.append(ft.Text(note, size=11.5, color=MUTED))

        return ft.Container(
            content=ft.Column(rows, spacing=8),
            bgcolor=SURFACE,
            padding=ft.Padding(12, 12, 12, 12),
            border=ft.Border(bottom=ft.BorderSide(1, LINE)),
        )

    def _tab_bar(self) -> ft.Control:
        """
        A plain segmented control, not ``Tabs``/``TabBarView``.

        The Flutter tab widgets need a bounded height, which forced the panel
        into a short box with its own scrollbar inside an already-scrolling
        page. Swapping the body inline lets the whole page scroll as one, so
        the content below the board is reachable with a normal swipe.
        """
        labels = [("advice", "Advice"), ("hand", "Cards"), ("table", "Players")]
        return ft.Container(
            content=ft.Row(
                [
                    ft.Container(
                        content=ft.Text(
                            title, size=13,
                            weight=ft.FontWeight.BOLD,
                            color=ACCENT if self.tab == key else MUTED,
                        ),
                        on_click=lambda _e, k=key: self._set_tab(k),
                        padding=ft.Padding(14, 8, 14, 8),
                        border_radius=10,
                        bgcolor=SURFACE_HI if self.tab == key else None,
                    )
                    for key, title in labels
                ],
                spacing=6,
            ),
            padding=ft.Padding(0, 4, 0, 4),
        )

    def _tab_body(self) -> List[ft.Control]:
        if self.tab == "hand":
            return self._tab_hand()
        if self.tab == "table":
            return self._tab_table()
        return self._tab_advice()

    def _set_tab(self, key: str) -> None:
        self.tab = key
        self.refresh()

    # -- tab: advice -------------------------------------------------------

    def _tab_advice(self) -> List[ft.Control]:
        flow = self.flow
        if not flow.is_my_turn():
            return self._panel(
                [
                    section(
                        f"Player {flow.current}'s turn",
                        ft.Text(
                            "Record their move on the board and their cards on "
                            "the Hand tab. Advice comes back on your turn.",
                            size=12, color=MUTED,
                        ),
                    ),
                    self._watchlist(),
                ]
            )

        step = flow.step
        if step is Step.SETUP_SETTLEMENT:
            return self._panel([self._setup_advice()])
        if step is Step.SETUP_ROAD:
            return self._panel([self._setup_road_advice()])
        if step is Step.MOVE_ROBBER:
            return self._panel([self._robber_advice()])
        if step is Step.DISCARD:
            return self._panel([self._discard_advice()])
        if step is Step.OVER:
            return self._panel(
                [section("Game over", ft.Text(flow.banner(), size=14, color=TEXT))]
            )
        return self._panel(
            [self._strategy_card(), self._turn_advice(), self._alerts()]
        )

    def _strategy_card(self) -> ft.Control:
        """
        The through-line, above the move list.

        A ranked list of moves answers "what now"; this answers "what am I
        playing for", which is the question that makes the list make sense.
        """
        plan = self.turn_advisor.plan(self.state, self.me)
        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.FLAG, color=ACCENT, size=16),
                            ft.Text("Your plan", size=12, color=MUTED,
                                    weight=ft.FontWeight.BOLD),
                        ],
                        spacing=6,
                    ),
                    ft.Text(plan.title, size=16, weight=ft.FontWeight.BOLD,
                            color=TEXT),
                    ft.Text(plan.focus, size=13, color=ACCENT),
                    ft.Text(plan.reason, size=12, color=MUTED),
                ],
                spacing=4,
            ),
            bgcolor=SURFACE, border_radius=12, padding=12,
            border=ft.Border(left=ft.BorderSide(3, ACCENT)),
        )

    def _panel(self, controls: List[ft.Control]) -> List[ft.Control]:
        """Drop the empty slots; the page scrolls these itself."""
        return [c for c in controls if c is not None]

    def _setup_advice(self) -> ft.Control:
        plans = self.setup_advisor.recommend(
            self.state, self.me, seat=self.me, top=3
        )
        if not plans:
            return section("Setup", ft.Text("No legal spot left.", color=MUTED))
        cards = [self._plan_card(i + 1, p) for i, p in enumerate(plans)]
        return section(
            "Best opening spots" if self.flow.setup_round == 1
            else "Best second settlement",
            ft.Column(cards, spacing=8),
        )

    def _plan_card(self, rank: int, plan: SetupPlan) -> ft.Control:
        d = plan.detail
        return ft.Container(
            content=ft.Column(
                [
                    # The rank leads, because the list is ordered by it. The
                    # raw strength of the spot is supporting detail — a
                    # slightly weaker spot can still be the better opening once
                    # the second pick is taken into account, and showing
                    # "Strong" above "Excellent" without that framing just
                    # looks like a mistake.
                    ft.Row(
                        [
                            chip(str(rank), ACCENT, ON_ACCENT),
                            ft.Text(
                                RANK_TITLE.get(rank, f"Option {rank}"), size=15,
                                weight=ft.FontWeight.BOLD, color=TEXT,
                                expand=True,
                            ),
                        ],
                        spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Row(
                        [
                            chip(f"Marked {rank} on the board", SURFACE, ACCENT),
                            chip(
                                f"{d.strength()} · pays on "
                                f"{d.payout_share():.0%} of rolls",
                                SURFACE, MUTED,
                            ),
                        ],
                        wrap=True, spacing=6, run_spacing=6,
                    ),
                    ft.Row(
                        [tile_chip(res, num) for res, num in d.tiles],
                        wrap=True, spacing=6, run_spacing=6,
                    ),
                    ft.Text(plan.reason, size=12, color=MUTED),
                ],
                spacing=8,
            ),
            gradient=surface_gradient(art.shade(SURFACE_HI, 0.06), SURFACE_HI),
            border_radius=12, padding=12, shadow=CARD_SHADOW,
            border=ft.Border.all(1, art.shade(SURFACE_HI, 0.10)),
        )

    def _setup_road_advice(self) -> ft.Control:
        node = self.flow.setup_settlement_node()
        edge = (
            self.setup_advisor.recommend_road(self.state, self.me, node)
            if node is not None else None
        )
        body = (
            "Point it toward "
            + describe_edge_end(
                self.board,
                self.board.edge(edge).other(node),
            )
            + " — that keeps the most room to expand. It is highlighted on the "
              "board."
            if edge is not None and node is not None
            else "Any path touching your new settlement is fine."
        )
        return section("Where to point the road", ft.Text(body, size=12, color=MUTED))

    def _turn_advice(self) -> ft.Control:
        advice = self.turn_advisor.recommend(self.state, self.me, top=5)
        if not advice:
            return section("What to do",
                           ft.Text("Nothing worth doing — end your turn.",
                                   size=12, color=MUTED))
        cards = [self._advice_card(i + 1, a) for i, a in enumerate(advice)]
        return section("What to do", ft.Column(cards, spacing=8))

    def _advice_card(self, rank: int, a: Advice) -> ft.Control:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            chip(str(rank), ACCENT if a.affordable else SURFACE,
                                 ON_ACCENT if a.affordable else MUTED),
                            ft.Text(a.label, size=14, weight=ft.FontWeight.BOLD,
                                    color=TEXT if a.affordable else MUTED,
                                    expand=True),
                        ],
                        spacing=8,
                    ),
                    ft.Text(a.reason, size=12, color=MUTED),
                    ft.Text(
                        f"Costs {a.cost_text()}" if a.affordable and a.cost
                        else (a.missing or ""),
                        size=11, color=GOOD if a.affordable else WARN,
                    ),
                ],
                spacing=4,
            ),
            gradient=(
                surface_gradient(art.shade(SURFACE_HI, 0.06), SURFACE_HI)
                if a.affordable else None
            ),
            bgcolor=None if a.affordable else "#10222f",
            border_radius=12, padding=12,
            shadow=CARD_SHADOW if a.affordable else None,
            border=ft.Border.all(
                1, art.shade(SURFACE_HI, 0.10) if a.affordable else LINE
            ),
        )

    def _robber_advice(self) -> ft.Control:
        advice = self.turn_advisor.robber_advice(self.state, self.me)
        if advice is None:
            return section("Robber",
                           ft.Text("No tile worth blocking.", size=12, color=MUTED))
        return section(
            "Where to put the robber",
            ft.Text(advice.label, size=14, weight=ft.FontWeight.BOLD, color=TEXT),
            ft.Text(advice.reason, size=12, color=MUTED),
        )

    def _discard_advice(self) -> ft.Control:
        advice = self.turn_advisor.discard_advice(self.state, self.me)
        if advice is None:
            return section("Discard",
                           ft.Text("You are under the limit.", size=12, color=MUTED))
        return section("What to throw away",
                       ft.Text(advice.reason, size=12, color=MUTED))

    def _alerts(self) -> Optional[ft.Control]:
        alerts = self.turn_advisor.alerts(self.state, self.me)
        if not alerts:
            return None
        return section(
            "Watch out",
            ft.Column([ft.Text("• " + a, size=12, color=WARN)
                       for a in alerts[:4]], spacing=4),
        )

    def _watchlist(self) -> ft.Control:
        threats = self.tracker.threats(self.me)
        if not threats:
            return section("Opponents",
                           ft.Text("Nothing pressing.", size=12, color=MUTED))
        return section(
            "Opponents",
            ft.Column([ft.Text("• " + t, size=12, color=WARN)
                       for t in threats], spacing=4),
        )

    # -- tab: hand ---------------------------------------------------------

    def _tab_hand(self) -> List[ft.Control]:
        rows: List[ft.Control] = []
        for player in sorted(self.state.players):
            rows.append(self._hand_editor(player))
        return self._panel(rows)

    def _hand_editor(self, player: int) -> ft.Control:
        hand = self.state.players[player].hand
        mine = player == self.me
        total = hand.total()
        # Nobody holds cards until the second settlement pays out, and the app
        # deals that itself. Editing here during setup could only introduce an
        # error, so the controls are locked until play begins.
        locked = self.flow.in_setup
        controls: List[ft.Control] = []
        for r in RESOURCES:
            controls.append(
                ft.Row(
                    [
                        ft.Container(width=12, height=12, border_radius=3,
                                     bgcolor=RESOURCE_COLOR[r]),
                        ft.Text(RESOURCE_LABEL[r], size=12, color=TEXT, width=52),
                        ft.IconButton(
                            icon=ft.Icons.REMOVE, icon_size=15, icon_color=MUTED,
                            disabled=locked,
                            on_click=lambda _e, p=player, r=r: self._adjust(p, r, -1),
                        ),
                        ft.Text(str(hand.cards[r]), size=14, color=TEXT, width=20,
                                text_align=ft.TextAlign.CENTER),
                        ft.IconButton(
                            icon=ft.Icons.ADD, icon_size=15, icon_color=MUTED,
                            disabled=locked,
                            on_click=lambda _e, p=player, r=r: self._adjust(p, r, 1),
                        ),
                    ],
                    spacing=1, vertical_alignment=ft.CrossAxisAlignment.CENTER,
                )
            )
        if locked:
            controls.append(
                ft.Text(
                    "Cards are dealt automatically when setup ends.",
                    size=11, color=MUTED,
                )
            )
        title = (
            f"Your hand — {total} card{'s' if total != 1 else ''}"
            if mine else
            f"Player {player} — {total} card{'s' if total != 1 else ''}"
        )
        if total > 7:
            title += "  ⚠ over the 7-card limit"
        p = self.state.players[player]
        held = [
            f"{count}× {card.label}" for card, count in sorted(
                p.dev_cards.items(), key=lambda kv: kv[0].value
            ) if count
        ]
        if p.unknown_dev:
            held.append(f"{p.unknown_dev} face-down")
        extra = ft.Row(
            [
                ft.Text(
                    "Dev cards: " + (", ".join(held) if held else "none"),
                    size=11, color=MUTED, expand=True,
                ),
                ft.Text(f"Knights played: {p.knights_played}", size=11,
                        color=MUTED),
            ],
            spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        return section(title, ft.Column(controls, spacing=0), extra)

    # -- tab: table --------------------------------------------------------

    def _tab_table(self) -> List[ft.Control]:
        rows: List[ft.Control] = []
        scores = rules.scores(self.state, include_hidden=False)
        road_holder = rules.longest_road_holder(self.state)
        army_holder = rules.largest_army_holder(self.state)

        for p in sorted(self.state.players):
            est = self.tracker.estimate(p)
            badges = []
            if road_holder == p:
                badges.append(chip("Longest road", GOOD, "#ffffff"))
            if army_holder == p:
                badges.append(chip("Largest army", GOOD, "#ffffff"))
            rows.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Row(
                                [
                                    chip(f"P{p}", PLAYER_COLOR[p], "#ffffff"),
                                    ft.Text(
                                        "You" if p == self.me else f"Player {p}",
                                        size=14, weight=ft.FontWeight.BOLD,
                                        color=TEXT, expand=True,
                                    ),
                                    chip(f"{scores[p]} VP", SURFACE, TEXT),
                                ],
                                spacing=8,
                            ),
                            ft.Text(
                                f"{len(self.state.settlements_of(p))} settlements · "
                                f"{len(self.state.cities_of(p))} cities · "
                                f"{len(self.state.edges_of(p))} roads · "
                                f"longest {rules.longest_road(self.state, p)}",
                                size=11, color=MUTED,
                            ),
                            ft.Text(
                                f"Cards: {est.describe()}", size=11, color=MUTED,
                            ),
                            ft.Row(badges, spacing=6) if badges else ft.Container(),
                        ],
                        spacing=4,
                    ),
                    bgcolor=SURFACE, border_radius=10, padding=10,
                )
            )
        rows.append(
            ft.Row(
                [
                    btn(
                        "Node numbers: " + ("on" if self.show_ids else "off"),
                        on_click=self._toggle_ids, bgcolor=SURFACE_HI, color=TEXT,
                    ),
                    btn("Edit board", on_click=self._back_to_editor,
                              bgcolor=SURFACE_HI, color=TEXT),
                ],
                wrap=True, spacing=8, run_spacing=8,
            )
        )
        return self._panel(rows)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _arm_default_action(self) -> None:
        """
        Steps with exactly one board move arm themselves.

        Setup placement, free roads and the robber have no choice to make, so
        making the player press a button first would be pure ceremony.
        """
        actions = self.flow.available_actions()

        # Drop an armed action the step has moved past, or the board keeps
        # asking for a settlement while the game waits on a dice roll.
        if self.pending_action is not None:
            if self.pending_action.id not in {a.id for a in actions}:
                self.pending_action = None
            else:
                return

        if self.flow.step in (
            Step.SETUP_SETTLEMENT, Step.SETUP_ROAD, Step.MOVE_ROBBER,
            Step.ROAD_BUILDING,
        ):
            if len(actions) == 1 and actions[0].target:
                self.pending_action = actions[0]

    def _choose_action(self, action: Action) -> None:
        if action.target in ("node", "edge", "tile"):
            self.pending_action = action
            self.status = ""
            self.refresh()
            return
        if action.target == "player":
            self._player_target_dialog(action)
            return
        self._perform(action, None)

    def _cancel_action(self, _e=None) -> None:
        self.pending_action = None
        self.status = ""
        self.refresh()

    def _draw_targets(self) -> Dict[str, object]:
        """What to light up on the board right now."""
        flow, draw = self.flow, {}
        pending = self.pending_action

        if pending is not None:
            if pending.target == "node":
                setup = flow.in_setup
                draw["legal_nodes"] = rules.legal_settlements(
                    self.state, flow.current, setup=setup
                ) if pending.id != "city" else rules.legal_cities(
                    self.state, flow.current
                )
            elif pending.target == "edge":
                if flow.step is Step.SETUP_ROAD:
                    node = flow.setup_settlement_node()
                    draw["legal_edges"] = [
                        e for e in self.state.board.node_edges[node or 0]
                        if e not in self.state.roads
                    ]
                else:
                    draw["legal_edges"] = rules.legal_roads(
                        self.state, flow.current
                    )
            elif pending.target == "tile":
                draw["legal_tiles"] = [
                    c for c in self.board.tiles if c != self.state.robber
                ]

        if not flow.is_my_turn():
            return draw

        # The advisor's picks, on top of whatever is tappable.
        if flow.step is Step.SETUP_SETTLEMENT:
            plans = self.setup_advisor.recommend(
                self.state, self.me, seat=self.me, top=3
            )
            draw["highlight_nodes"] = [p.first for p in plans]
        elif flow.step is Step.SETUP_ROAD:
            node = flow.setup_settlement_node()
            if node is not None:
                edge = self.setup_advisor.recommend_road(self.state, self.me, node)
                if edge is not None:
                    draw["highlight_edges"] = [edge]
        elif flow.step is Step.MOVE_ROBBER:
            advice = self.turn_advisor.robber_advice(self.state, self.me)
            if advice and advice.coord:
                draw["highlight_tiles"] = [advice.coord]
        elif flow.step is Step.MAIN:
            advice = self.turn_advisor.recommend(self.state, self.me, top=5)
            draw["highlight_nodes"] = [
                a.node for a in advice if a.node is not None and a.affordable
            ][:3]
            draw["highlight_edges"] = [
                a.edge for a in advice if a.edge is not None and a.affordable
            ][:2]
        return draw

    # -- board taps --------------------------------------------------------

    def _on_tap(self, e) -> None:
        pos = getattr(e, "local_position", None)
        if pos is None:
            return
        sx, sy = pos.x, pos.y

        if self.screen == "editor":
            # A tap out at sea near a jetty means "that port", not the tile
            # behind it — otherwise the harbours are unreachable on the board.
            slot = self._port_slot_at(sx, sy)
            if slot is not None:
                self._edit_port(slot)
                return
            kind, target = self.view.hit(self.board, sx, sy, want="tile")
            if kind == "tile":
                self._edit_tile(target)
            return

        action = self.pending_action
        if action is None or not action.target:
            kind, target = self.view.hit(self.board, sx, sy)
            self.status = self._describe(kind, target)
            self.refresh()
            return

        kind, target = self.view.hit(self.board, sx, sy, want=action.target)
        if kind == "none":
            return
        self._perform(action, target)

    def _perform(self, action: Action, target) -> None:
        flow = self.flow
        result = None

        if action.id == "setup_settlement":
            result = flow.place_setup_settlement(target)
        elif action.id == "setup_road":
            result = flow.place_setup_road(target)
        elif action.id == "build_settlement":
            result = flow.build("settlement", target)
        elif action.id == "city":
            result = flow.build("city", target)
        elif action.id == "build_road":
            result = flow.build("road", target)
        elif action.id == "move_robber":
            result = flow.move_robber(target)
        elif action.id == "free_road":
            result = flow.place_free_road(target)
        elif action.id == "roll":
            self._roll_dialog()
            return
        elif action.id == "buy_dev":
            self._buy_dev_dialog()
            return
        elif action.id.startswith("play_dev:"):
            card = DevCard(action.id.split(":", 1)[1])
            if card is DevCard.KNIGHT:
                result = flow.play_dev(card)
            elif card is DevCard.MONOPOLY:
                self._monopoly_dialog()
                return
            elif card is DevCard.YEAR_OF_PLENTY:
                self._year_of_plenty_dialog()
                return
            else:
                result = flow.play_dev(card)
        elif action.id == "reveal_vp":
            result = flow.reveal_vp()
        elif action.id == "steal":
            result = flow.steal(target)
        elif action.id == "skip_steal":
            result = flow.skip_steal()
        elif action.id == "discard":
            self._discard_dialog(target)
            return
        elif action.id == "trade":
            self._trade_dialog()
            return
        elif action.id == "end_turn":
            result = flow.end_turn()
        elif action.id == "new_game":
            self._new_game()
            return

        self.pending_action = None
        if result is not None and not result.ok:
            self.status = result.reason or "That move isn't legal."
            # Keep the action armed so the player can simply try again.
            if action.target in ("node", "edge", "tile"):
                self.pending_action = action
        else:
            self.status = ""
        self.refresh()

    def _describe(self, kind: str, target) -> str:
        if kind == "node":
            node = self.board.node(target)
            bits = []
            for coord in node.tiles:
                tile = self.board.tiles[coord]
                name = tile.resource.value if tile.resource else "desert"
                bits.append(f"{name} {tile.number}" if tile.number else name)
            text = ", ".join(bits)
            if node.port:
                text += f" · {node.port.value} port"
            owner = self.state.owner_of(target)
            if owner:
                text += f" · Player {owner}"
            return text
        if kind == "edge":
            owner = self.state.road_owner(target)
            return describe_edge(self.board, target) + (
                f" · Player {owner}'s road" if owner else " · open path"
            )
        if kind == "tile":
            tile = self.board.tiles[target]
            name = tile.resource.value if tile.resource else "desert"
            return f"{name.capitalize()} {tile.number or '—'}"
        return ""

    # -- dialogs -----------------------------------------------------------

    def _roll_dialog(self) -> None:
        def roll(n: int) -> None:
            self.page.pop_dialog()
            self.pending_action = None
            result = self.flow.roll(n)
            if not result.ok:
                self.status = result.reason or ""
            elif n != 7:
                gained = self.tracker.production_forecast(self.me).get(n, {})
                self.status = (
                    "You collected "
                    + ", ".join(f"{c} {r.value}" for r, c in gained.items())
                    if gained else f"Rolled {n} — nothing for you."
                )
            self.refresh()

        self.page.show_dialog(
            ft.AlertDialog(
                modal=True, bgcolor=SURFACE,
                title=ft.Text("What did the dice show?", color=TEXT),
                content=self._dialog_body(
                    ft.Row(
                        [
                            btn(
                                str(n),
                                bgcolor=WARN if n == 7 else SURFACE_HI,
                                color="#ffffff" if n == 7 else TEXT,
                                on_click=lambda _e, n=n: roll(n),
                            )
                            for n in range(2, 13)
                        ],
                        wrap=True, spacing=6, run_spacing=6,
                    ),
                    320,
                ),
                actions=[
                    ft.TextButton("Cancel", on_click=lambda _e: self._close_dialog())
                ],
            )
        )
        self.page.update()

    def _buy_dev_dialog(self) -> None:
        """
        Record which card actually came off the deck.

        For your own purchase you can read the card; for an opponent you only
        see the back of it, so "unknown" is offered too.
        """
        mine = self.flow.is_my_turn()

        def buy(card: Optional[DevCard]) -> None:
            self.page.pop_dialog()
            self.pending_action = None
            result = self.flow.buy_dev(card)
            self.status = (
                "" if result.ok else (result.reason or "")
            )
            if result.ok and card is not None:
                self.status = (
                    f"Drew {card.label}. "
                    + ("Playable from your next turn." if card.playable else
                       "Keep it hidden — it counts at the end.")
                )
            self.refresh()

        buttons = [
            btn(
                card.label, bgcolor=SURFACE_HI, color=TEXT,
                tooltip=card.blurb,
                on_click=lambda _e, c=card: buy(c),
            )
            for card in DevCard
        ]
        if not mine:
            buttons.append(
                btn("Unknown", bgcolor=SURFACE_HI, color=MUTED,
                          on_click=lambda _e: buy(None))
            )

        self.page.show_dialog(
            ft.AlertDialog(
                modal=True, bgcolor=SURFACE,
                title=ft.Text(
                    "Which card was drawn?" if mine
                    else f"Player {self.flow.current} drew…", color=TEXT,
                ),
                content=self._dialog_body(
                    ft.Column(
                        [
                            ft.Text(
                                "Read it off the card you just took."
                                if mine else
                                "Pick it if you saw it, or leave it unknown.",
                                size=12, color=MUTED,
                            ),
                            ft.Row(buttons, wrap=True, spacing=6, run_spacing=6),
                        ],
                        tight=True, spacing=8,
                    ),
                ),
                actions=[
                    ft.TextButton("Cancel", on_click=lambda _e: self._close_dialog())
                ],
            )
        )
        self.page.update()

    def _monopoly_dialog(self) -> None:
        def pick(resource: Resource) -> None:
            self.page.pop_dialog()
            self.pending_action = None
            before = self.state.players[self.flow.current].hand.cards[resource]
            result = self.flow.play_dev(DevCard.MONOPOLY, resource=resource)
            if result.ok:
                gained = (
                    self.state.players[self.flow.current].hand.cards[resource]
                    - before
                )
                self.status = (
                    f"Monopoly on {resource.value} — collected {gained} card"
                    f"{'s' if gained != 1 else ''}."
                )
            else:
                self.status = result.reason or ""
            self.refresh()

        self.page.show_dialog(
            ft.AlertDialog(
                modal=True, bgcolor=SURFACE,
                title=ft.Text("Monopoly — name a resource", color=TEXT),
                content=self._dialog_body(
                    ft.Row(
                        [
                            btn(
                                RESOURCE_LABEL[r], bgcolor=RESOURCE_COLOR[r],
                                color="#ffffff",
                                on_click=lambda _e, r=r: pick(r),
                            )
                            for r in RESOURCES
                        ],
                        wrap=True, spacing=6, run_spacing=6,
                    ),
                ),
                actions=[
                    ft.TextButton("Cancel", on_click=lambda _e: self._close_dialog())
                ],
            )
        )
        self.page.update()

    def _year_of_plenty_dialog(self) -> None:
        picked: List[Resource] = []
        body = ft.Column(tight=True, spacing=8)

        def toggle(resource: Resource) -> None:
            picked.append(resource)
            if len(picked) > 2:
                picked.pop(0)
            redraw()

        def confirm(_e) -> None:
            self.page.pop_dialog()
            self.pending_action = None
            result = self.flow.play_dev(DevCard.YEAR_OF_PLENTY, cards=picked)
            self.status = (
                f"Took {' and '.join(r.value for r in picked)} from the bank."
                if result.ok else (result.reason or "")
            )
            self.refresh()

        def redraw() -> None:
            body.controls = [
                ft.Text("Take any two cards from the bank.", size=12,
                        color=MUTED),
                ft.Row(
                    [
                        btn(
                            RESOURCE_LABEL[r],
                            bgcolor=(
                                RESOURCE_COLOR[r] if r in picked else SURFACE_HI
                            ),
                            color="#ffffff" if r in picked else TEXT,
                            on_click=lambda _e, r=r: toggle(r),
                        )
                        for r in RESOURCES
                    ],
                    wrap=True, spacing=6, run_spacing=6,
                ),
                ft.Text(
                    "Chosen: "
                    + (", ".join(r.value for r in picked) or "nothing yet"),
                    size=12, color=GOOD if len(picked) == 2 else WARN,
                ),
            ]
            self.page.update()

        redraw()
        self.page.show_dialog(
            ft.AlertDialog(
                modal=True, bgcolor=SURFACE,
                title=ft.Text("Year of plenty", color=TEXT),
                content=self._dialog_body(body),
                actions=[
                    ft.TextButton("Cancel", on_click=lambda _e: self._close_dialog()),
                    btn("Take them", on_click=confirm, bgcolor=ACCENT,
                              color="#1a1a1a"),
                ],
            )
        )
        self.page.update()

    def _player_target_dialog(self, action: Action) -> None:
        """Steal and discard name a player in the label; act on that player."""
        if action.id == "steal":
            victims = self.flow.steal_victims()
        else:
            victims = self.flow.pending_discards()

        def pick(p: int) -> None:
            self.page.pop_dialog()
            self._perform(action, p)

        self.page.show_dialog(
            ft.AlertDialog(
                modal=True, bgcolor=SURFACE,
                title=ft.Text(action.label, color=TEXT),
                content=self._dialog_body(
                    ft.Row(
                        [
                            btn(f"Player {p}", bgcolor=PLAYER_COLOR[p],
                                color="#ffffff",
                                on_click=lambda _e, p=p: pick(p))
                            for p in victims
                        ],
                        wrap=True, spacing=6, run_spacing=6,
                    ),
                ),
                actions=[
                    ft.TextButton("Cancel", on_click=lambda _e: self._close_dialog())
                ],
            )
        )
        self.page.update()

    def _discard_dialog(self, player: int) -> None:
        owed = rules.must_discard(self.state, player)
        hand = self.state.players[player].hand
        picked: Dict[Resource, int] = {r: 0 for r in RESOURCES}
        body = ft.Column(tight=True, spacing=6)

        suggestion = (
            self.turn_advisor.discard_advice(self.state, player)
            if player == self.me else None
        )

        def total() -> int:
            return sum(picked.values())

        def bump(r: Resource, delta: int) -> None:
            new = picked[r] + delta
            if 0 <= new <= hand.cards[r] and total() + delta <= owed:
                picked[r] = new
            redraw()

        def confirm(_e) -> None:
            cards: List[Resource] = []
            for r, n in picked.items():
                cards.extend([r] * n)
            result = self.flow.discard(player, cards)
            self.page.pop_dialog()
            self.status = "" if result.ok else (result.reason or "")
            self.refresh()

        def redraw() -> None:
            rows: List[ft.Control] = [
                ft.Text(
                    f"Player {player} must discard {owed} of "
                    f"{hand.total()} cards.",
                    size=12, color=MUTED,
                )
            ]
            if suggestion:
                rows.append(ft.Text(suggestion.reason, size=11, color=ACCENT))
            for r in RESOURCES:
                if hand.cards[r] == 0:
                    continue
                rows.append(
                    ft.Row(
                        [
                            ft.Container(width=12, height=12, border_radius=3,
                                         bgcolor=RESOURCE_COLOR[r]),
                            ft.Text(f"{RESOURCE_LABEL[r]} ({hand.cards[r]})",
                                    size=12, color=TEXT, width=100),
                            ft.IconButton(
                                icon=ft.Icons.REMOVE, icon_size=15,
                                icon_color=MUTED,
                                on_click=lambda _e, r=r: bump(r, -1),
                            ),
                            ft.Text(str(picked[r]), size=14, color=TEXT, width=20,
                                    text_align=ft.TextAlign.CENTER),
                            ft.IconButton(
                                icon=ft.Icons.ADD, icon_size=15, icon_color=MUTED,
                                on_click=lambda _e, r=r: bump(r, 1),
                            ),
                        ],
                        spacing=2,
                    )
                )
            rows.append(
                ft.Text(f"Chosen {total()} of {owed}", size=12,
                        color=GOOD if total() == owed else WARN)
            )
            body.controls = rows
            if dialog_ref:
                dialog_ref[0].actions[-1].disabled = total() != owed
            self.page.update()

        dialog_ref: List[ft.AlertDialog] = []
        dialog = ft.AlertDialog(
            modal=True, bgcolor=SURFACE,
            title=ft.Text("Discard", color=TEXT),
            content=self._dialog_body(body),
            actions=[
                ft.TextButton("Cancel", on_click=lambda _e: self._close_dialog()),
                btn("Discard", on_click=confirm, bgcolor=ACCENT,
                          color="#1a1a1a", disabled=True),
            ],
        )
        dialog_ref.append(dialog)
        redraw()
        self.page.show_dialog(dialog)
        self.page.update()

    def _trade_dialog(self) -> None:
        """
        One dialog, two kinds of trade.

        Most Catan turns are settled by trading with the person across the
        table, not with the bank, so both live here: the bank tab swaps at your
        port rate, the player tab moves whatever the two of you agreed.
        """
        player = self.flow.current
        rates = self.state.ports_of(player)
        hand = self.state.players[player].hand
        opponents = [p for p in sorted(self.state.players) if p != player]

        mode = {"kind": "bank"}
        bank = {"give": None, "get": None}
        deal: Dict[str, object] = {
            "with": opponents[0] if opponents else None,
            "give": {r: 0 for r in RESOURCES},
            "get": {r: 0 for r in RESOURCES},
        }
        body = ft.Column(tight=True, spacing=8)

        def counter_row(bucket: str, resource: Resource, cap: Optional[int]) -> ft.Row:
            counts = deal[bucket]  # type: ignore[index]
            return ft.Row(
                [
                    ft.Container(width=12, height=12, border_radius=3,
                                 bgcolor=RESOURCE_COLOR[resource]),
                    ft.Text(RESOURCE_LABEL[resource], size=12, color=TEXT,
                            width=52),
                    ft.IconButton(
                        icon=ft.Icons.REMOVE, icon_size=15, icon_color=MUTED,
                        on_click=lambda _e, b=bucket, r=resource: bump(b, r, -1),
                    ),
                    ft.Text(str(counts[resource]), size=14, color=TEXT, width=18,
                            text_align=ft.TextAlign.CENTER),
                    ft.IconButton(
                        icon=ft.Icons.ADD, icon_size=15, icon_color=MUTED,
                        disabled=cap is not None and counts[resource] >= cap,
                        on_click=lambda _e, b=bucket, r=resource: bump(b, r, 1),
                    ),
                ],
                spacing=1, vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )

        def bump(bucket: str, resource: Resource, delta: int) -> None:
            counts = deal[bucket]  # type: ignore[index]
            cap = hand.cards[resource] if bucket == "give" else 20
            counts[resource] = max(0, min(cap, counts[resource] + delta))
            redraw()

        def set_mode(kind: str) -> None:
            mode["kind"] = kind
            redraw()

        def set_partner(p: int) -> None:
            deal["with"] = p
            redraw()

        def pick_bank(slot: str, r: Resource) -> None:
            bank[slot] = r
            redraw()

        def redraw() -> None:
            tabs = ft.Row(
                [
                    btn(
                        label,
                        bgcolor=ACCENT if mode["kind"] == key else SURFACE_HI,
                        color=ON_ACCENT if mode["kind"] == key else TEXT,
                        on_click=lambda _e, k=key: set_mode(k),
                    )
                    for key, label in (("bank", "Bank / port"),
                                       ("player", "Another player"))
                ],
                spacing=6,
            )

            if mode["kind"] == "bank":
                rows = [
                    ft.Text("Give", size=12, color=MUTED),
                    ft.Row(
                        [
                            btn(
                                f"{rates[r]}× {RESOURCE_LABEL[r]}",
                                bgcolor=(RESOURCE_COLOR[r] if bank["give"] is r
                                         else SURFACE_HI),
                                color="#ffffff" if bank["give"] is r else TEXT,
                                disabled=hand.cards[r] < rates[r],
                                on_click=lambda _e, r=r: pick_bank("give", r),
                            )
                            for r in RESOURCES
                        ],
                        wrap=True, spacing=6, run_spacing=6,
                    ),
                    ft.Text("Receive", size=12, color=MUTED),
                    ft.Row(
                        [
                            btn(
                                RESOURCE_LABEL[r],
                                bgcolor=(RESOURCE_COLOR[r] if bank["get"] is r
                                         else SURFACE_HI),
                                color="#ffffff" if bank["get"] is r else TEXT,
                                on_click=lambda _e, r=r: pick_bank("get", r),
                            )
                            for r in RESOURCES
                        ],
                        wrap=True, spacing=6, run_spacing=6,
                    ),
                ]
            else:
                rows = [
                    ft.Text("With", size=12, color=MUTED),
                    ft.Row(
                        [
                            btn(
                                f"Player {p}",
                                bgcolor=(PLAYER_COLOR[p] if deal["with"] == p
                                         else SURFACE_HI),
                                color="#ffffff" if deal["with"] == p else TEXT,
                                on_click=lambda _e, p=p: set_partner(p),
                            )
                            for p in opponents
                        ],
                        wrap=True, spacing=6, run_spacing=6,
                    ),
                    ft.Text("You give", size=12, color=MUTED),
                ]
                rows += [
                    counter_row("give", r, hand.cards[r])
                    for r in RESOURCES if hand.cards[r] > 0
                ]
                rows.append(ft.Text("You receive", size=12, color=MUTED))
                rows += [counter_row("get", r, None) for r in RESOURCES]

            body.controls = [tabs] + rows
            self.page.update()

        def confirm(_e) -> None:
            if mode["kind"] == "bank":
                give, get = bank["give"], bank["get"]
                if give is None or get is None or give is get:
                    self.status = (
                        "Pick one resource to give and a different one to get."
                    )
                else:
                    result = self.flow.trade_bank(give, get)
                    self.status = "" if result.ok else (result.reason or "")
            else:
                partner = deal["with"]
                giving = [
                    r for r in RESOURCES
                    for _ in range(deal["give"][r])  # type: ignore[index]
                ]
                getting = [
                    r for r in RESOURCES
                    for _ in range(deal["get"][r])  # type: ignore[index]
                ]
                if partner is None:
                    self.status = "Pick who you traded with."
                else:
                    result = self.flow.trade_player(partner, giving, getting)
                    self.status = (
                        f"Traded with Player {partner}." if result.ok
                        else (result.reason or "")
                    )
            self.page.pop_dialog()
            self.refresh()

        redraw()
        self.page.show_dialog(
            ft.AlertDialog(
                modal=True, bgcolor=SURFACE,
                title=ft.Text("Trade", color=TEXT),
                content=self._dialog_body(body),
                actions=[
                    ft.TextButton("Cancel", on_click=lambda _e: self._close_dialog()),
                    btn("Confirm", on_click=confirm, bgcolor=ACCENT,
                              color="#1a1a1a"),
                ],
            )
        )
        self.page.update()

    # -- misc --------------------------------------------------------------

    def _adjust(self, player: int, resource: Resource, delta: int) -> None:
        if self.flow.in_setup:
            self.status = "Cards are dealt when setup ends."
            self.refresh()
            return
        if delta < 0 and self.state.players[player].hand.cards[resource] == 0:
            return
        self.state.adjust(player, resource, delta)
        self.refresh()

    def _adjust_dev(self, player: int, delta: int) -> None:
        """
        Nudge a player's face-down card count.

        Only the unknown pile moves: a card whose type we recorded is a fact,
        and losing it to a stray tap would quietly corrupt the count.
        """
        p = self.state.players[player]
        p.unknown_dev = max(0, p.unknown_dev + delta)
        self.refresh()

    def _undo(self, _e=None) -> None:
        self.pending_action = None
        self.status = "" if self.flow.undo() else "Nothing left to undo."
        self.refresh()

    def _toggle_ids(self, _e=None) -> None:
        self.show_ids = not self.show_ids
        self.refresh()

    def _back_to_editor(self, _e=None) -> None:
        self.pending = dict(self.board.layout.tiles)
        self.pending_ports = dict(self.board.layout.ports)
        self.screen = "editor"
        self.status = "Editing the board. Starting the game resets play."
        self.refresh()

    def _new_game(self) -> None:
        self.state = GameState(
            self.board, num_players=self.state.num_players, me=self.me
        )
        self._rebuild_engines()
        self.pending_action = None
        self.status = ""
        self.refresh()


# --------------------------------------------------------------------------


def main(page: ft.Page) -> None:
    page.title = "CatanMind"
    page.bgcolor = BG
    page.padding = 0
    page.theme_mode = ft.ThemeMode.DARK
    page.scroll = None
    page.theme = ft.Theme(
        color_scheme_seed=ACCENT,
        font_family="Roboto",
        visual_density=ft.VisualDensity.COMFORTABLE,
    )

    app = CatanMind(page)

    def on_resized(_e=None) -> None:
        """Keep the board's share of the screen right as the window changes."""
        height = getattr(page, "height", None)
        width = getattr(page, "width", None)
        if not height:
            return
        if abs(height - app.page_height) < 2:
            return
        app.page_height = float(height)
        app.page_width = float(width or app.page_width)
        app.refresh()

    page.on_resized = on_resized

    # The opening sequence owns the window until it hands over. It is
    # decoration, so anything going wrong with it drops straight through to
    # the app rather than leaving the player looking at a blank screen.
    splash.set_palette(RESOURCE_COLOR)
    shell = ft.Container(expand=True)

    def start_app() -> None:
        shell.content = app.build()
        app.refresh()
        try:
            page.update()
        except Exception:
            pass

    try:
        opening = splash.Splash(
            page, start_app,
            width=getattr(page, "width", None) or 390,
            height=getattr(page, "height", None) or 844,
            palette={
                "bg": BG, "accent": ACCENT, "on_accent": ON_ACCENT,
                "text": TEXT, "muted": MUTED,
            },
        )
        shell.content = opening.root
        page.add(shell)
        page.run_task(opening.play)
    except Exception:
        page.add(shell)
        start_app()
