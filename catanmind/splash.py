"""
The opening sequence.

A cold start into a settings form is a flat first impression. This lands the
island first: tiles drop into place one ring at a time, the mark and name fade
up, and the whole thing hands over to the app.

It is built from the same drawing code as the board, so it cannot drift out of
step with the game's own look, and it is skippable — a tap anywhere ends it.
The animation is decoration, so it must never be the reason someone waits.
"""

from __future__ import annotations

import asyncio
from typing import Callable, List, Optional, Tuple

import flet as ft
import flet.canvas as cv

from catanmind import art
from catanmind.board import Board, Layout, hex_distance

#: How long the whole sequence runs if nobody taps through it.
TOTAL_SECONDS = 2.1

#: Delay between one ring of tiles landing and the next.
RING_DELAY = 0.22


def _tile_rings(board: Board) -> List[List[Tuple[int, int]]]:
    """The board's coordinates grouped by distance from the centre."""
    rings: List[List[Tuple[int, int]]] = [[], [], []]
    for coord in board.tiles:
        rings[hex_distance(coord, (0, 0))].append(coord)
    return rings


class Splash:
    """
    The opening animation, and the handover to the app.

    ``on_done`` is called exactly once, whether the sequence finished on its
    own or the player tapped through it.
    """

    def __init__(
        self,
        page: ft.Page,
        on_done: Callable[[], None],
        *,
        width: float = 390,
        height: float = 844,
        palette: Optional[dict] = None,
    ):
        self.page = page
        self.on_done = on_done
        self.width = float(width)
        self.height = float(height)
        self.finished = False

        colours = palette or {}
        self.bg = colours.get("bg", "#08131d")
        self.accent = colours.get("accent", "#f0b43a")
        self.on_accent = colours.get("on_accent", "#0a1622")
        self.text = colours.get("text", "#eef5fa")
        self.muted = colours.get("muted", "#8ba7bf")

        self.board = Board(Layout.standard())
        self.rings = _tile_rings(self.board)

        #: How many rings have landed. Drawn from this, so a redraw is cheap.
        self.revealed = 0

        self.canvas = cv.Canvas(shapes=[], expand=True)
        self.wordmark = ft.Container(
            content=self._wordmark(),
            opacity=0,
            offset=ft.Offset(0, 0.35),
            animate_opacity=ft.Animation(520, ft.AnimationCurve.EASE_OUT),
            animate_offset=ft.Animation(520, ft.AnimationCurve.EASE_OUT),
            alignment=ft.Alignment(0, 0),
        )
        # A column rather than a stack: the island gets its own band and the
        # wordmark sits under it, so the two cannot overlap at any screen
        # shape. Stack alignment put the name across the middle of the board.
        self.root = ft.GestureDetector(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Container(
                            content=self.canvas, height=self.island_height,
                        ),
                        ft.Container(content=self.wordmark, expand=True,
                                     alignment=ft.Alignment(0, -0.2)),
                    ],
                    spacing=0,
                    expand=True,
                    horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
                bgcolor=self.bg,
                expand=True,
            ),
            on_tap=lambda _e: self.finish(),
        )

    @property
    def island_height(self) -> float:
        """The band the board is drawn in, leaving room for the name below."""
        return max(240.0, self.height * 0.56)

    # -- drawing -----------------------------------------------------------

    def _wordmark(self) -> ft.Control:
        return ft.Column(
            [
                ft.Text("CatanMind", size=34, weight=ft.FontWeight.BOLD,
                        color=self.text),
                ft.Text("One move ahead", size=14, color=self.muted),
            ],
            spacing=2,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            tight=True,
        )

    def _draw(self) -> None:
        """
        The island, with the rings that have landed so far.

        Rings appear from the outside in, so the last thing to arrive is the
        centre of the board — which is where the eye ends up for the wordmark.
        """
        from catanmind.view import Viewport

        band = self.island_height
        size = min(self.width * 0.92, band * 0.92)
        view = Viewport.fit(self.board, size, size)
        radius = view.hex_radius(self.board)
        offset_x = (self.width - size) / 2
        offset_y = (band - size) / 2

        shapes: List[cv.Shape] = []
        for distance in range(2, -1, -1):
            step = 2 - distance
            if step >= self.revealed:
                continue
            landed = step == self.revealed - 1
            for coord in self.rings[distance]:
                tile = self.board.tiles[coord]
                corners = [
                    (x + offset_x, y + offset_y)
                    for x, y in view.tile_corners(self.board, coord)
                ]
                cx, cy = view.tile_xy(self.board, coord)
                centre = (cx + offset_x, cy + offset_y)
                shapes += art.tile_shapes(
                    corners, centre, radius,
                    _RESOURCE_COLOUR.get(tile.resource, "#d9c39a"),
                    tile.resource,
                )
                # The newest ring gets a brief highlight, so the eye follows
                # the sequence rather than watching the whole board flicker.
                if landed:
                    shapes.append(
                        cv.Path(
                            elements=art.polygon(corners),
                            paint=art.stroke(
                                art.alpha(self.accent, 0.55),
                                max(1.5, radius * 0.07),
                            ),
                        )
                    )
        self.canvas.shapes = shapes

    # -- sequence ----------------------------------------------------------

    async def play(self) -> None:
        """Run the sequence. Returns as soon as it is done or skipped."""
        for step in range(1, 4):
            if self.finished:
                return
            self.revealed = step
            self._draw()
            self._safe_update()
            await asyncio.sleep(RING_DELAY)

        if self.finished:
            return
        self.wordmark.opacity = 1
        self.wordmark.offset = ft.Offset(0, 0)
        self._safe_update()

        await asyncio.sleep(TOTAL_SECONDS - 3 * RING_DELAY)
        self.finish()

    def finish(self) -> None:
        """Hand over to the app. Safe to call twice; the second is ignored."""
        if self.finished:
            return
        self.finished = True
        self.on_done()

    def _safe_update(self) -> None:
        # The page may already be gone if the player skipped or closed the
        # window mid-sequence; a splash is never worth an exception.
        try:
            self.page.update()
        except Exception:
            pass


#: Kept local rather than imported from the UI, so this module can be built
#: and tested without pulling in the whole interface.
_RESOURCE_COLOUR = {}


def set_palette(colours: dict) -> None:
    """Let the UI hand over its own resource colours at import time."""
    _RESOURCE_COLOUR.update(colours)
