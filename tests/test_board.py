"""Board geometry and layout."""

import collections
import math

import pytest

from catanmind.board import (
    Board,
    Layout,
    NUMBER_TOKENS,
    Port,
    PORT_POOL,
    Resource,
    SPIRAL,
    TILE_POOL,
    hex_distance,
    pips,
    probability,
)


@pytest.fixture(scope="module")
def board():
    return Board()


def test_standard_board_shape(board):
    assert len(board.tiles) == 19
    assert len(board.nodes) == 54
    assert len(board.edges) == 72


def test_node_degrees(board):
    """54 nodes and 72 edges force exactly 18 corners and 36 junctions."""
    degrees = collections.Counter(len(n.neighbors) for n in board.nodes)
    assert degrees == {2: 18, 3: 36}
    assert sum(len(n.neighbors) for n in board.nodes) == 2 * len(board.edges)


def test_tiles_per_node(board):
    counts = collections.Counter(len(n.tiles) for n in board.nodes)
    assert counts == {3: 24, 2: 12, 1: 18}
    assert sum(len(n.tiles) for n in board.nodes) == 19 * 6


def test_every_tile_has_six_distinct_corners(board):
    for coord in board.tiles:
        corners = board.nodes_of_tile(coord)
        assert len(corners) == 6
        assert len(set(corners)) == 6


def test_neighbors_and_edges_agree(board):
    from_neighbors = {
        frozenset((n.id, m)) for n in board.nodes for m in n.neighbors
    }
    from_edges = {frozenset((e.a, e.b)) for e in board.edges}
    assert from_neighbors == from_edges


def test_edge_lookup_is_symmetric(board):
    for edge in board.edges:
        assert board.edge_id[(edge.a, edge.b)] == edge.id
        assert board.edge_id[(edge.b, edge.a)] == edge.id
        assert board.edge_between(edge.a, edge.b) is edge
    assert board.edge_between(0, 53) is None


def test_node_edges_are_complete(board):
    for node in board.nodes:
        assert len(board.node_edges[node.id]) == len(node.neighbors)


def test_shared_corners_are_the_same_node(board):
    """Adjacent tiles must share exactly two corner nodes."""
    coords = list(board.tiles)
    for i, a in enumerate(coords):
        for b in coords[i + 1:]:
            shared = set(board.nodes_of_tile(a)) & set(board.nodes_of_tile(b))
            expected = 2 if hex_distance(a, b) == 1 else 0
            assert len(shared) == expected, f"{a} vs {b}"


def test_geometry_is_regular(board):
    """Every edge is the same length: proof the lattice keys merged correctly."""
    lengths = {
        round(math.dist(
            (board.node(e.a).x, board.node(e.a).y),
            (board.node(e.b).x, board.node(e.b).y),
        ), 6)
        for e in board.edges
    }
    assert len(lengths) == 1
    assert lengths.pop() == pytest.approx(board.SIZE)


def test_coastline(board):
    assert len(board.coastal_ring) == 30
    coastal_nodes = [n for n in board.nodes if n.is_coastal]
    assert len(coastal_nodes) == 30


def test_ports_are_on_the_coast_and_evenly_spread(board):
    assert len(board.port_slots) == 9
    assert len(set(board.port_slots)) == 9
    assert set(board.port_slots) <= set(board.coastal_ring)

    port_nodes = [n for n in board.nodes if n.port is not None]
    assert len(port_nodes) == 18
    assert all(n.is_coastal for n in port_nodes)

    counts = collections.Counter(n.port for n in port_nodes)
    assert counts[Port.GENERIC] == 8  # four 3:1 ports, two nodes each
    for port in (Port.WOOD, Port.BRICK, Port.SHEEP, Port.WHEAT, Port.ORE):
        assert counts[port] == 2

    # The physical frame is three-fold symmetric; the gaps should repeat.
    angles = sorted(
        math.degrees(math.atan2(board.edge(e).y, board.edge(e).x)) % 360
        for e in board.port_slots
    )
    gaps = [round((angles[(i + 1) % 9] - angles[i]) % 360) for i in range(9)]
    assert gaps[0:3] == gaps[3:6] == gaps[6:9]


def test_port_rates():
    assert Port.GENERIC.rate == 3
    assert Port.GENERIC.resource is None
    assert Port.ORE.rate == 2
    assert Port.ORE.resource is Resource.ORE


def test_pips_and_probability():
    assert pips(7) == 0
    assert pips(6) == 5 and pips(8) == 5
    assert pips(2) == 1 and pips(12) == 1
    assert sum(pips(n) for n in range(2, 13)) == 30  # 36 minus the six 7s
    assert probability(6) == pytest.approx(5 / 36)


def test_tile_pool_and_tokens():
    assert len(TILE_POOL) == 19
    assert TILE_POOL.count(None) == 1
    assert len(NUMBER_TOKENS) == 18
    assert 7 not in NUMBER_TOKENS
    assert len(PORT_POOL) == 9
    assert len(SPIRAL) == 19
    assert len(set(SPIRAL)) == 19


def test_spiral_covers_the_standard_hex_field():
    assert set(SPIRAL) == {
        (q, r) for q in range(-2, 3) for r in range(-2, 3)
        if hex_distance((q, r), (0, 0)) <= 2
    }


def test_standard_layout_is_legal(board):
    layout = Layout.standard()
    assert layout.validate() == []
    assert layout.warnings(board) == []


def test_layout_validation_catches_a_duplicate_resource(board):
    layout = Layout.standard()
    tiles = dict(layout.tiles)
    victim = next(c for c, (r, _) in tiles.items() if r is Resource.ORE)
    tiles[victim] = (Resource.WOOD, tiles[victim][1])
    problems = Layout(tiles=tiles, ports=layout.ports).validate()
    assert any("ore" in p for p in problems)
    assert any("wood" in p for p in problems)


def test_layout_validation_catches_a_missing_number(board):
    layout = Layout.standard()
    tiles = dict(layout.tiles)
    victim = next(c for c, (r, _) in tiles.items() if r is not None)
    tiles[victim] = (tiles[victim][0], 0)
    problems = Layout(tiles=tiles, ports=layout.ports).validate()
    assert problems


def test_layout_warns_about_adjacent_red_numbers(board):
    tiles = dict(Layout.standard().tiles)
    # Force a 6 and an 8 next to each other.
    tiles[(0, -1)] = (Resource.WOOD, 6)
    tiles[(1, -1)] = (Resource.BRICK, 8)
    layout = Layout(tiles=tiles, ports=Layout.standard().ports)
    assert layout.warnings(board)


def test_random_layouts_are_legal_and_avoid_adjacent_reds(board):
    import random

    rng = random.Random(1234)
    for _ in range(15):
        layout = Layout.random(rng)
        assert layout.validate() == []
        assert layout.warnings(board) == []


def test_board_accepts_a_new_layout():
    import random

    layout = Layout.random(random.Random(7))
    board = Board(layout)
    assert len(board.tiles) == 19
    assert len(board.nodes) == 54
    # Topology must not depend on the layout painted onto it.
    assert [n.neighbors for n in board.nodes] == [n.neighbors for n in Board().nodes]


def test_resource_pips_sum_to_the_token_total(board):
    total = sum(board.resource_pips().values())
    assert total == sum(pips(n) for n in NUMBER_TOKENS)
