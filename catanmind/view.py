"""
Board geometry for a screen: fitting, and turning a tap back into a game object.

This is deliberately free of any UI framework. The board publishes abstract
coordinates around the origin; a :class:`Viewport` maps them into a widget of a
given pixel size, and maps taps back the other way. Keeping it separate means
hit-testing is unit-testable without starting an app — which matters, because
"the tap landed on the wrong node" is invisible in a screenshot but obvious in
a test.

The old UI had no reverse mapping at all: it laid 54 invisible buttons over a
bitmap, so roads and tiles simply could not be clicked, and every road or robber
action fell back to typing node numbers by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from catanmind.board import Board

#: A tap this many pixels from an intersection counts as hitting it.
NODE_RADIUS = 17.0

#: ...and this far from the midpoint of a path counts as hitting the road.
EDGE_RADIUS = 15.0

#: Port jetties are drawn outside the coastline, so the fit has to leave room
#: for them or they get clipped. Expressed as a fraction of a hex radius:
#: the jetty sits at 0.50 and its badge is another 0.26 across.
PORT_ALLOWANCE = 0.80


@dataclass(frozen=True)
class Viewport:
    """
    Maps board coordinates to pixels and back.

    ``scale`` is uniform so hexes stay regular; ``pad`` keeps the outermost
    nodes and their touch targets inside the widget.
    """

    scale: float
    offset_x: float
    offset_y: float
    width: float
    height: float

    @staticmethod
    def fit(
        board: Board, width: float, height: float, pad: Optional[float] = None
    ) -> "Viewport":
        """
        Largest uniform scale that fits the whole board, centred.

        ``pad`` defaults to enough room for the port jetties, which are drawn
        beyond the outermost intersections. Solving for it exactly would need
        the scale that the padding itself determines, so this takes one pass at
        a nominal scale and then leaves that margin.
        """
        min_x, min_y, max_x, max_y = board.bounds()
        span_x = max(1e-6, max_x - min_x)
        span_y = max(1e-6, max_y - min_y)

        if pad is None:
            nominal = min(width / span_x, height / span_y)
            pad = board.SIZE * nominal * PORT_ALLOWANCE

        usable_w = max(1.0, width - 2 * pad)
        usable_h = max(1.0, height - 2 * pad)
        scale = min(usable_w / span_x, usable_h / span_y)

        # Centre the scaled board inside the widget.
        offset_x = (width - span_x * scale) / 2 - min_x * scale
        offset_y = (height - span_y * scale) / 2 - min_y * scale
        return Viewport(
            scale=scale, offset_x=offset_x, offset_y=offset_y,
            width=width, height=height,
        )

    # -- forward -----------------------------------------------------------

    def to_screen(self, x: float, y: float) -> Tuple[float, float]:
        return (x * self.scale + self.offset_x, y * self.scale + self.offset_y)

    def node_xy(self, board: Board, node_id: int) -> Tuple[float, float]:
        node = board.node(node_id)
        return self.to_screen(node.x, node.y)

    def edge_xy(self, board: Board, edge_id: int) -> Tuple[float, float]:
        edge = board.edge(edge_id)
        return self.to_screen(edge.x, edge.y)

    def tile_xy(self, board: Board, coord: Tuple[int, int]) -> Tuple[float, float]:
        return self.to_screen(*board.tile_center(coord))

    def hex_radius(self, board: Board) -> float:
        """Centre-to-corner distance of one tile, in pixels."""
        return board.SIZE * self.scale

    def tile_corners(
        self, board: Board, coord: Tuple[int, int]
    ) -> List[Tuple[float, float]]:
        """The six corners of a tile in screen order, ready to stroke."""
        return [
            self.node_xy(board, nid) for nid in board.nodes_of_tile(coord)
        ]

    # -- reverse -----------------------------------------------------------

    def to_board(self, sx: float, sy: float) -> Tuple[float, float]:
        return ((sx - self.offset_x) / self.scale, (sy - self.offset_y) / self.scale)

    def node_at(
        self, board: Board, sx: float, sy: float,
        radius: float = NODE_RADIUS,
    ) -> Optional[int]:
        """Nearest intersection within ``radius`` pixels, or ``None``."""
        best, best_d = None, radius * radius
        for node in board.nodes:
            px, py = self.to_screen(node.x, node.y)
            d = (px - sx) ** 2 + (py - sy) ** 2
            if d <= best_d:
                best, best_d = node.id, d
        return best

    def edge_at(
        self, board: Board, sx: float, sy: float,
        radius: float = EDGE_RADIUS,
    ) -> Optional[int]:
        """
        Nearest path within ``radius`` pixels of the tap.

        Measured to the segment rather than only its midpoint, so long roads are
        as easy to hit near their ends as in the middle.
        """
        best, best_d = None, radius * radius
        for edge in board.edges:
            ax, ay = self.node_xy(board, edge.a)
            bx, by = self.node_xy(board, edge.b)
            d = _point_segment_distance_sq(sx, sy, ax, ay, bx, by)
            if d <= best_d:
                best, best_d = edge.id, d
        return best

    def tile_at(
        self, board: Board, sx: float, sy: float
    ) -> Optional[Tuple[int, int]]:
        """Which tile the tap fell inside, by nearest centre."""
        radius = self.hex_radius(board)
        best, best_d = None, radius * radius
        for coord in board.tiles:
            px, py = self.tile_xy(board, coord)
            d = (px - sx) ** 2 + (py - sy) ** 2
            if d <= best_d:
                best, best_d = coord, d
        return best

    def hit(
        self, board: Board, sx: float, sy: float, *, want: str = "any"
    ) -> Tuple[str, object]:
        """
        Resolve a tap to ``(kind, target)`` where kind is
        ``"node"``, ``"edge"``, ``"tile"`` or ``"none"``.

        ``want`` restricts the search, which is what the interaction modes use:
        while placing a road only paths should respond, so a slightly-off tap
        still lands on the road the player meant rather than on a nearby
        intersection.
        """
        if want in ("node", "any"):
            node = self.node_at(board, sx, sy)
            if node is not None:
                return ("node", node)
            if want == "node":
                return ("none", None)

        if want in ("edge", "any"):
            edge = self.edge_at(board, sx, sy)
            if edge is not None:
                return ("edge", edge)
            if want == "edge":
                return ("none", None)

        if want in ("tile", "any"):
            tile = self.tile_at(board, sx, sy)
            if tile is not None:
                return ("tile", tile)

        return ("none", None)


def _point_segment_distance_sq(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Squared distance from a point to the segment ``a``–``b``."""
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    nx, ny = ax + t * dx, ay + t * dy
    return (px - nx) ** 2 + (py - ny) ** 2
