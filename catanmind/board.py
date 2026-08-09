"""
Board geometry and graph.

The board is *topology only* — which tiles, nodes and edges exist and how they
connect. Everything mutable (who owns what, where the robber is) lives in
:mod:`catanmind.state`.

Coordinates
-----------
Tiles use axial coordinates ``(q, r)`` with pointy-top orientation. The 19-tile
standard board is every coordinate within distance 2 of the origin.

Node positions are derived with *exact integer arithmetic* rather than rounded
floats: measured in units of ``(size*sqrt(3)/2, size/2)`` every hex centre and
every corner lands on an integer lattice point, so two hexes sharing a corner
produce a byte-identical key. This is what guarantees exactly 54 nodes and 72
edges, with no dependence on a rounding tolerance.

Screen convention: ``+y`` points **down**, so increasing ``r`` moves down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Enums and constants
# --------------------------------------------------------------------------


class Resource(Enum):
    """The five producible resources."""

    WOOD = "wood"
    BRICK = "brick"
    SHEEP = "sheep"
    WHEAT = "wheat"
    ORE = "ore"


#: All producible resources, in a stable display order.
RESOURCES: Tuple[Resource, ...] = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
)

#: Sentinel used for the desert tile, which produces nothing.
DESERT: None = None


class Building(Enum):
    SETTLEMENT = "settlement"
    CITY = "city"


class Port(Enum):
    """Port types. ``GENERIC`` is the 3:1 any-resource port."""

    GENERIC = "3:1"
    WOOD = "wood"
    BRICK = "brick"
    SHEEP = "sheep"
    WHEAT = "wheat"
    ORE = "ore"

    @property
    def resource(self) -> Optional[Resource]:
        """The resource this port discounts, or ``None`` for a 3:1 port."""
        if self is Port.GENERIC:
            return None
        return Resource(self.value)

    @property
    def rate(self) -> int:
        """Cards required to trade in one card of any other resource."""
        return 3 if self is Port.GENERIC else 2


#: Number of dice combinations (out of 36) that produce each roll.
DICE_WAYS: Dict[int, int] = {
    2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
    8: 5, 9: 4, 10: 3, 11: 2, 12: 1,
}


def pips(number: int) -> int:
    """The 'dots' printed on a number token: ways to roll it, 0 for 7/desert."""
    if number == 7:
        return 0
    return DICE_WAYS.get(number, 0)


def probability(number: int) -> float:
    """Probability that a single roll produces ``number``."""
    return pips(number) / 36.0


#: Cost of each purchasable item.
COSTS: Dict[str, Dict[Resource, int]] = {
    "road": {Resource.WOOD: 1, Resource.BRICK: 1},
    "settlement": {
        Resource.WOOD: 1,
        Resource.BRICK: 1,
        Resource.SHEEP: 1,
        Resource.WHEAT: 1,
    },
    "city": {Resource.WHEAT: 2, Resource.ORE: 3},
    "dev_card": {Resource.SHEEP: 1, Resource.WHEAT: 1, Resource.ORE: 1},
}

#: Buildings each player starts with, i.e. the supply limit.
SUPPLY = {"settlement": 5, "city": 4, "road": 15}


class DevCard(Enum):
    """The five development cards."""

    KNIGHT = "knight"
    VICTORY_POINT = "victory_point"
    ROAD_BUILDING = "road_building"
    YEAR_OF_PLENTY = "year_of_plenty"
    MONOPOLY = "monopoly"

    @property
    def label(self) -> str:
        return {
            DevCard.KNIGHT: "Knight",
            DevCard.VICTORY_POINT: "Victory point",
            DevCard.ROAD_BUILDING: "Road building",
            DevCard.YEAR_OF_PLENTY: "Year of plenty",
            DevCard.MONOPOLY: "Monopoly",
        }[self]

    @property
    def blurb(self) -> str:
        """What the card does, for the player choosing one."""
        return {
            DevCard.KNIGHT: "Move the robber and steal a card.",
            DevCard.VICTORY_POINT: "Worth 1 point. Kept hidden until you win.",
            DevCard.ROAD_BUILDING: "Place two roads for free.",
            DevCard.YEAR_OF_PLENTY: "Take any two resources from the bank.",
            DevCard.MONOPOLY: "Name a resource; every player hands theirs over.",
        }[self]

    @property
    def playable(self) -> bool:
        """Victory points are revealed, never played as an action."""
        return self is not DevCard.VICTORY_POINT


#: Composition of the development-card deck in the base game.
DEV_DECK: Dict[DevCard, int] = {
    DevCard.KNIGHT: 14,
    DevCard.VICTORY_POINT: 5,
    DevCard.ROAD_BUILDING: 2,
    DevCard.YEAR_OF_PLENTY: 2,
    DevCard.MONOPOLY: 2,
}
DEV_DECK_SIZE = sum(DEV_DECK.values())


# --------------------------------------------------------------------------
# Layout: the part of a board that changes between games
# --------------------------------------------------------------------------

#: The 19 standard tile coordinates, ordered as an inward clockwise spiral
#: starting from the top-left tile. This is the order number tokens are dealt
#: in the physical game, and the order the board editor walks tiles in.
SPIRAL: Tuple[Tuple[int, int], ...] = (
    # outer ring, clockwise from top-left
    (0, -2), (1, -2), (2, -2),
    (2, -1), (2, 0),
    (1, 1), (0, 2),
    (-1, 2), (-2, 2),
    (-2, 1), (-2, 0),
    (-1, -1),
    # inner ring, clockwise from top-left
    (0, -1), (1, -1),
    (1, 0),
    (0, 1), (-1, 1),
    (-1, 0),
    # centre
    (0, 0),
)

#: Tile composition of the base game: 4 wood, 3 brick, 4 sheep, 4 wheat,
#: 3 ore, 1 desert.
TILE_POOL: Tuple[Optional[Resource], ...] = (
    (Resource.WOOD,) * 4
    + (Resource.BRICK,) * 3
    + (Resource.SHEEP,) * 4
    + (Resource.WHEAT,) * 4
    + (Resource.ORE,) * 3
    + (None,)
)

#: The 18 number tokens, in the order they are dealt along the spiral in the
#: "beginner/balanced" setup (letters A..R on the physical tokens).
NUMBER_TOKENS: Tuple[int, ...] = (
    5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11,
)

#: The nine ports of the base game.
PORT_POOL: Tuple[Port, ...] = (
    Port.GENERIC, Port.GENERIC, Port.GENERIC, Port.GENERIC,
    Port.WOOD, Port.BRICK, Port.SHEEP, Port.WHEAT, Port.ORE,
)


@dataclass(frozen=True)
class Layout:
    """
    The randomised part of a board: what is on each tile and where the ports are.

    ``tiles`` maps an axial coordinate to ``(resource, number)``. The desert has
    ``resource=None`` and ``number=0``. ``ports`` maps a *port slot index*
    (0..8, see :attr:`Board.port_slots`) to a port type.
    """

    tiles: Dict[Tuple[int, int], Tuple[Optional[Resource], int]]
    ports: Dict[int, Port]

    def validate(self) -> List[str]:
        """
        Return a list of human-readable problems with this layout.

        An empty list means the layout is a legal base-game setup.
        """
        problems: List[str] = []

        missing = set(SPIRAL) - set(self.tiles)
        if missing:
            problems.append(f"{len(missing)} tiles have not been filled in")
            return problems  # everything below assumes a complete board

        resources = [res for res, _ in self.tiles.values()]
        for res in RESOURCES:
            want = TILE_POOL.count(res)
            got = resources.count(res)
            if got != want:
                problems.append(
                    f"{res.value}: {got} tiles, expected {want}"
                )
        deserts = resources.count(None)
        if deserts != 1:
            problems.append(f"{deserts} deserts, expected 1")

        numbers = sorted(n for _, n in self.tiles.values() if n)
        if numbers != sorted(NUMBER_TOKENS):
            problems.append("number tokens do not match the standard set")

        for coord, (res, num) in self.tiles.items():
            if res is None and num:
                problems.append(f"desert at {coord} must not carry a number")
            if res is not None and not num:
                problems.append(f"tile at {coord} is missing its number")

        if len(self.ports) != len(PORT_POOL):
            problems.append(
                f"{len(self.ports)} ports assigned, expected {len(PORT_POOL)}"
            )
        else:
            got_ports = sorted(p.value for p in self.ports.values())
            want_ports = sorted(p.value for p in PORT_POOL)
            if got_ports != want_ports:
                problems.append("ports do not match the standard set")

        return problems

    def warnings(self, board: "Board") -> List[str]:
        """
        Return soft warnings: legal layouts that the official setup rules avoid.

        Separate from :meth:`validate` because a player may genuinely be looking
        at a board where 6 and 8 ended up adjacent.
        """
        warns: List[str] = []
        hot = [c for c, (_, n) in self.tiles.items() if n in (6, 8)]
        for i, a in enumerate(hot):
            for b in hot[i + 1:]:
                if board.tiles_adjacent(a, b):
                    warns.append(
                        f"{self.tiles[a][1]} and {self.tiles[b][1]} are adjacent "
                        f"at {a} / {b}"
                    )
        return warns

    @staticmethod
    def standard() -> "Layout":
        """The fixed 'beginner' layout: spiral tile order with tokens A..R."""
        tiles: Dict[Tuple[int, int], Tuple[Optional[Resource], int]] = {}
        token_iter = iter(NUMBER_TOKENS)
        # Beginner setup places the desert in the centre.
        resources = [r for r in TILE_POOL if r is not None]
        res_iter = iter(resources)
        for coord in SPIRAL:
            if coord == (0, 0):
                tiles[coord] = (None, 0)
            else:
                tiles[coord] = (next(res_iter), next(token_iter))
        return Layout(tiles=tiles, ports=dict(enumerate(PORT_POOL)))

    @staticmethod
    def random(rng=None) -> "Layout":
        """
        A uniformly shuffled legal layout.

        Retries until no two 6/8 tiles are adjacent, matching the official
        setup restriction.
        """
        import random as _random

        rng = rng or _random.Random()
        board = Board(Layout.standard())  # topology only; layout is replaced below

        for _ in range(500):
            resources = list(TILE_POOL)
            rng.shuffle(resources)
            numbers = list(NUMBER_TOKENS)
            rng.shuffle(numbers)

            tiles: Dict[Tuple[int, int], Tuple[Optional[Resource], int]] = {}
            num_iter = iter(numbers)
            for coord, res in zip(SPIRAL, resources):
                tiles[coord] = (None, 0) if res is None else (res, next(num_iter))

            ports = list(PORT_POOL)
            rng.shuffle(ports)
            layout = Layout(tiles=tiles, ports=dict(enumerate(ports)))
            if not layout.warnings(board):
                return layout

        return layout  # pragma: no cover - astronomically unlikely


# --------------------------------------------------------------------------
# Graph elements
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    coord: Tuple[int, int]
    resource: Optional[Resource]
    number: int
    nodes: Tuple[int, ...]  # the six corner node ids, clockwise from top

    @property
    def pips(self) -> int:
        return 0 if self.resource is None else pips(self.number)

    @property
    def is_desert(self) -> bool:
        return self.resource is None


@dataclass(frozen=True)
class Node:
    """An intersection: a place a settlement or city can stand."""

    id: int
    x: float
    y: float
    tiles: Tuple[Tuple[int, int], ...]  # coords of the 1-3 tiles touching it
    neighbors: Tuple[int, ...]
    port: Optional[Port] = None

    @property
    def is_coastal(self) -> bool:
        return len(self.tiles) < 3


@dataclass(frozen=True)
class Edge:
    """A path between two intersections: a place a road can stand."""

    id: int
    a: int
    b: int
    x: float  # midpoint, for rendering and hit-testing
    y: float

    @property
    def nodes(self) -> Tuple[int, int]:
        return (self.a, self.b)

    def other(self, node_id: int) -> int:
        return self.b if node_id == self.a else self.a


# Corner offsets in lattice units, clockwise starting from the top corner.
_CORNER_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1),
)


def hex_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Distance between two axial tile coordinates."""
    dq, dr = a[0] - b[0], a[1] - b[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


class Board:
    """
    Immutable board topology plus the layout painted onto it.

    All lookups are precomputed dictionaries — nothing here scans a list.
    """

    #: Distance in "hex sizes" used for the pixel coordinates exposed by the
    #: board. The UI scales these; the value only fixes the aspect ratio.
    SIZE: float = 50.0

    def __init__(self, layout: Optional[Layout] = None):
        self.layout = layout or Layout.standard()

        self._build_graph()
        self._build_tiles()
        self._find_port_slots()
        self._apply_ports()

    # -- construction ------------------------------------------------------

    def _build_graph(self) -> None:
        sx = self.SIZE * math.sqrt(3) / 2.0
        sy = self.SIZE / 2.0

        # Lattice key -> list of tile coords touching that corner.
        corner_tiles: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        # Tile coord -> its six corner lattice keys, clockwise from top.
        tile_corners: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        for coord in SPIRAL:
            q, r = coord
            cx, cy = 2 * q + r, 3 * r
            keys = []
            for dx, dy in _CORNER_OFFSETS:
                key = (cx + dx, cy + dy)
                keys.append(key)
                corner_tiles.setdefault(key, []).append(coord)
            tile_corners[coord] = keys

        # Deterministic ids: top-to-bottom, then left-to-right.
        ordered = sorted(corner_tiles, key=lambda k: (k[1], k[0]))
        key_to_id = {key: i for i, key in enumerate(ordered)}
        self._tile_corner_ids: Dict[Tuple[int, int], Tuple[int, ...]] = {
            coord: tuple(key_to_id[k] for k in keys)
            for coord, keys in tile_corners.items()
        }

        # Edges come from consecutive corners of each tile.
        edge_pairs: Dict[Tuple[int, int], None] = {}
        for ids in self._tile_corner_ids.values():
            for i in range(6):
                a, b = ids[i], ids[(i + 1) % 6]
                edge_pairs[(min(a, b), max(a, b))] = None

        neighbors: Dict[int, set] = {i: set() for i in key_to_id.values()}
        for a, b in edge_pairs:
            neighbors[a].add(b)
            neighbors[b].add(a)

        self.nodes: Tuple[Node, ...] = tuple(
            Node(
                id=key_to_id[key],
                x=key[0] * sx,
                y=key[1] * sy,
                tiles=tuple(sorted(corner_tiles[key])),
                neighbors=tuple(sorted(neighbors[key_to_id[key]])),
            )
            for key in ordered
        )

        self.edges: Tuple[Edge, ...] = tuple(
            Edge(
                id=i,
                a=a,
                b=b,
                x=(self.nodes[a].x + self.nodes[b].x) / 2,
                y=(self.nodes[a].y + self.nodes[b].y) / 2,
            )
            for i, (a, b) in enumerate(sorted(edge_pairs))
        )

        #: ``(node_a, node_b)`` -> edge id, in both orders. O(1) lookup.
        self.edge_id: Dict[Tuple[int, int], int] = {}
        for e in self.edges:
            self.edge_id[(e.a, e.b)] = e.id
            self.edge_id[(e.b, e.a)] = e.id

        #: node id -> ids of the edges meeting there.
        self.node_edges: Dict[int, Tuple[int, ...]] = {
            n.id: tuple(self.edge_id[(n.id, m)] for m in n.neighbors)
            for n in self.nodes
        }

    def _build_tiles(self) -> None:
        self.tiles: Dict[Tuple[int, int], Tile] = {}
        for coord in SPIRAL:
            res, num = self.layout.tiles.get(coord, (None, 0))
            self.tiles[coord] = Tile(
                coord=coord,
                resource=res,
                number=num,
                nodes=self._tile_corner_ids[coord],
            )

        #: node id -> the tiles touching it (objects, not coords).
        self.node_tiles: Dict[int, Tuple[Tile, ...]] = {
            n.id: tuple(self.tiles[c] for c in n.tiles) for n in self.nodes
        }

        #: dice number -> tiles that produce on it (deserts excluded).
        self.tiles_by_number: Dict[int, Tuple[Tile, ...]] = {}
        for t in self.tiles.values():
            if t.resource is not None:
                self.tiles_by_number.setdefault(t.number, ()) # type: ignore[arg-type]
                self.tiles_by_number[t.number] = self.tiles_by_number[t.number] + (t,)

    def _find_port_slots(self) -> None:
        """
        Locate the nine canonical port positions by walking the coastline.

        Coastal edges are those belonging to exactly one tile. Walking them in
        order gives a 30-edge ring; the physical board places ports on it with
        a repeating 'port, gap, gap / port, gap, gap / port, gap, gap, gap'
        rhythm, which is what the 2-2-3 gap pattern below encodes.
        """
        edge_tile_count: Dict[int, int] = {e.id: 0 for e in self.edges}
        for coord, ids in self._tile_corner_ids.items():
            for i in range(6):
                eid = self.edge_id[(ids[i], ids[(i + 1) % 6])]
                edge_tile_count[eid] += 1

        coastal = {eid for eid, c in edge_tile_count.items() if c == 1}

        # Walk the ring. Every coastal node touches exactly two coastal edges.
        start = min(coastal)
        ring: List[int] = [start]
        cur_node = self.edges[start].b
        while len(ring) < len(coastal):
            nxt = None
            for eid in self.node_edges[cur_node]:
                if eid in coastal and eid != ring[-1]:
                    nxt = eid
                    break
            if nxt is None:  # pragma: no cover - would mean a broken coastline
                break
            ring.append(nxt)
            cur_node = self.edges[nxt].other(cur_node)

        gaps = (2, 2, 3)
        slots: List[int] = []
        pos = 0
        gi = 0
        while len(slots) < len(PORT_POOL) and pos < len(ring):
            slots.append(ring[pos])
            pos += 1 + gaps[gi % len(gaps)]
            gi += 1

        #: port slot index -> the coastal edge id that port sits on.
        self.port_slots: Tuple[int, ...] = tuple(slots)
        self.coastal_ring: Tuple[int, ...] = tuple(ring)

    def _apply_ports(self) -> None:
        node_port: Dict[int, Port] = {}
        for slot, eid in enumerate(self.port_slots):
            port = self.layout.ports.get(slot)
            if port is None:
                continue
            edge = self.edges[eid]
            node_port[edge.a] = port
            node_port[edge.b] = port

        self.nodes = tuple(
            Node(
                id=n.id, x=n.x, y=n.y, tiles=n.tiles,
                neighbors=n.neighbors, port=node_port.get(n.id),
            )
            for n in self.nodes
        )

    # -- queries -----------------------------------------------------------

    def node(self, node_id: int) -> Node:
        return self.nodes[node_id]

    def edge(self, edge_id: int) -> Edge:
        return self.edges[edge_id]

    def edge_between(self, a: int, b: int) -> Optional[Edge]:
        eid = self.edge_id.get((a, b))
        return None if eid is None else self.edges[eid]

    def tiles_adjacent(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return hex_distance(a, b) == 1

    def nodes_of_tile(self, coord: Tuple[int, int]) -> Tuple[int, ...]:
        return self._tile_corner_ids[coord]

    def bounds(self) -> Tuple[float, float, float, float]:
        """``(min_x, min_y, max_x, max_y)`` covering every node."""
        xs = [n.x for n in self.nodes]
        ys = [n.y for n in self.nodes]
        return (min(xs), min(ys), max(xs), max(ys))

    def tile_center(self, coord: Tuple[int, int]) -> Tuple[float, float]:
        q, r = coord
        return ((2 * q + r) * self.SIZE * math.sqrt(3) / 2, 3 * r * self.SIZE / 2)

    def with_layout(self, layout: Layout) -> "Board":
        """A new board with the same topology but a different layout."""
        return Board(layout)

    # -- reporting ---------------------------------------------------------

    def resource_pips(self) -> Dict[Resource, int]:
        """Total pips printed on each resource, across the whole board."""
        totals = {r: 0 for r in RESOURCES}
        for t in self.tiles.values():
            if t.resource is not None:
                totals[t.resource] += t.pips
        return totals

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<Board {len(self.tiles)} tiles, {len(self.nodes)} nodes, "
            f"{len(self.edges)} edges>"
        )
