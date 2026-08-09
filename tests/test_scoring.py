"""The scoring model."""


import pytest

from catanmind.board import Board, Layout, Resource, RESOURCES
from catanmind.scoring import (
    SATURATION,
    Scorer,
    demand_weights,
    resource_values,
    supply_weights,
)
from catanmind.state import GameState


@pytest.fixture(scope="module")
def board():
    return Board()


@pytest.fixture(scope="module")
def scorer(board):
    return Scorer(board)


@pytest.fixture
def state(board):
    return GameState(board)


# -- the value model -------------------------------------------------------


def test_demand_weights_are_normalised():
    weights = demand_weights()
    assert set(weights) == set(RESOURCES)
    assert sum(weights.values()) == pytest.approx(len(RESOURCES))


def test_sheep_is_the_least_demanded_resource():
    """Nothing but the development card leans on sheep."""
    weights = demand_weights()
    assert weights[Resource.SHEEP] == min(weights.values())


def test_supply_weights_track_the_board(board):
    supply = supply_weights(board)
    pips = board.resource_pips()
    ordered_supply = sorted(RESOURCES, key=lambda r: supply[r])
    ordered_pips = sorted(RESOURCES, key=lambda r: pips[r])
    assert ordered_supply == ordered_pips


def test_resource_values_are_normalised_and_bounded(board):
    values = resource_values(board)
    assert sum(values.values()) == pytest.approx(len(RESOURCES))
    assert all(0.2 < v < 3.0 for v in values.values())


def test_a_scarce_resource_is_worth_more(board):
    """Halve the ore on the board and ore should become more valuable."""
    base = resource_values(board)

    tiles = dict(Layout.standard().tiles)
    for coord, (res, num) in list(tiles.items()):
        if res is Resource.ORE:
            tiles[coord] = (Resource.SHEEP, num)
            break
    scarce = resource_values(Board(Layout(tiles=tiles, ports=Layout.standard().ports)))
    assert scarce[Resource.ORE] > base[Resource.ORE]


# -- the utility model -----------------------------------------------------


def test_utility_is_zero_for_nothing(scorer):
    assert scorer.utility({}) == 0.0
    assert scorer.utility({r: 0.0 for r in RESOURCES}) == 0.0


def test_utility_increases_with_production(scorer):
    small = scorer.utility({Resource.WHEAT: 0.05})
    large = scorer.utility({Resource.WHEAT: 0.20})
    assert large > small


def test_utility_is_concave(scorer):
    """Doubling one stream must be worth less than twice as much."""
    single = scorer.utility({Resource.WHEAT: SATURATION})
    double = scorer.utility({Resource.WHEAT: SATURATION * 2})
    assert double < single * 2


def test_spreading_production_beats_concentrating_it(scorer):
    """This is where diversity comes from — no bonus term required."""
    concentrated = scorer.utility({Resource.WHEAT: 0.3})
    spread = scorer.utility({Resource.WHEAT: 0.15, Resource.BRICK: 0.15})
    assert spread > concentrated


def test_marginal_utility_shrinks_as_you_already_hold_a_resource(scorer):
    added = {Resource.WHEAT: 0.14}
    from_nothing = scorer.marginal_utility({}, added)
    from_plenty = scorer.marginal_utility({Resource.WHEAT: 0.3}, added)
    assert from_plenty < from_nothing


def test_marginal_utility_of_a_new_resource_beats_more_of_the_same(scorer):
    portfolio = {Resource.WHEAT: 0.2}
    more_wheat = scorer.marginal_utility(portfolio, {Resource.WHEAT: 0.14})
    new_brick = scorer.marginal_utility(portfolio, {Resource.BRICK: 0.14})
    assert new_brick > more_wheat


# -- spot scoring ----------------------------------------------------------


def test_every_node_scores_without_error(scorer, state):
    for node in state.board.nodes:
        score = scorer.score_spot(state, node.id, 1)
        assert score.total >= 0
        assert score.node_id == node.id


def test_a_three_tile_spot_beats_a_one_tile_spot(scorer, state):
    board = state.board
    rich = max(
        (n for n in board.nodes if len(n.tiles) == 3),
        key=lambda n: sum(board.tiles[c].pips for c in n.tiles),
    )
    poor = min(
        (n for n in board.nodes if len(n.tiles) == 1),
        key=lambda n: sum(board.tiles[c].pips for c in n.tiles),
    )
    assert scorer.score_spot(state, rich.id, 1).production > \
           scorer.score_spot(state, poor.id, 1).production


def test_desert_contributes_nothing(scorer, state):
    board = state.board
    desert = next(c for c, t in board.tiles.items() if t.is_desert)
    for node_id in board.nodes_of_tile(desert):
        yields = scorer._node_yield[node_id]
        contributing = [
            board.tiles[c] for c in board.node(node_id).tiles
            if board.tiles[c].resource
        ]
        assert sum(yields.values()) == pytest.approx(
            sum(t.pips / 36 for t in contributing)
        )


def test_score_is_reported_with_a_breakdown(scorer, state):
    score = scorer.score_spot(state, 25, 1)
    assert score.explain()
    assert isinstance(score.numbers, list)
    assert score.total == pytest.approx(
        score.production * 1.0 + score.expansion * 0.6
        + score.port * 0.5 + score.blocking * 0.3,
        abs=0.02,
    )


def test_rank_spots_is_sorted_and_truncated(scorer, state):
    ranked = scorer.rank_spots(state, 1, setup=True, top=5)
    assert len(ranked) == 5
    assert ranked == sorted(ranked, key=lambda s: s.total, reverse=True)


def test_ranking_excludes_illegal_spots(scorer, state):
    state.build_settlement(1, 0, free=True)
    ranked = scorer.rank_spots(state, 1, setup=True)
    ids = {s.node_id for s in ranked}
    assert 0 not in ids
    assert not (ids & set(state.board.node(0).neighbors))


# -- ports -----------------------------------------------------------------


def test_a_port_you_cannot_feed_is_worth_little(scorer, state):
    node = next(
        n for n in state.board.nodes if n.port and n.port.resource
    )
    value, note = scorer.port_value(node.id, {}, {r: 4 for r in RESOURCES})
    assert 0 < value < 2
    assert "no production" in note


def test_a_port_matching_your_surplus_is_worth_more(scorer, state):
    node = next(n for n in state.board.nodes if n.port and n.port.resource)
    resource = node.port.resource
    lean, _ = scorer.port_value(node.id, {}, {r: 4 for r in RESOURCES})
    fed, _ = scorer.port_value(
        node.id, {resource: 0.4}, {r: 4 for r in RESOURCES}
    )
    assert fed > lean


def test_a_port_you_already_own_adds_nothing(scorer, state):
    node = next(n for n in state.board.nodes if n.port and n.port.resource)
    resource = node.port.resource
    rates = {r: 4 for r in RESOURCES}
    rates[resource] = 2
    value, _ = scorer.port_value(node.id, {resource: 0.4}, rates)
    assert value == 0.0


def test_nodes_without_ports_score_zero(scorer, state):
    node = next(n for n in state.board.nodes if n.port is None)
    assert scorer.port_value(node.id, {}, {r: 4 for r in RESOURCES}) == (0.0, None)


# -- expansion and blocking ------------------------------------------------


def test_expansion_falls_when_neighbours_are_taken(scorer, state):
    node = 25
    before = scorer.expansion_value(state, node, 1)
    for neighbour in state.board.node(node).neighbors:
        for far in state.board.node(neighbour).neighbors:
            if far != node and far not in state.buildings:
                state.build_settlement(2, far, free=True)
                break
    after = scorer.expansion_value(state, node, 1)
    assert after < before


def test_an_enemy_road_closes_a_direction(scorer, state):
    node = 25
    scorer.expansion_value(state, node, 1)
    for eid in state.board.node_edges[node]:
        state.build_road(2, eid, free=True)
    after = scorer.expansion_value(state, node, 1)
    assert after == 0.0


def test_blocking_only_counts_contested_spots(scorer, state):
    node = 25
    assert scorer.blocking_value(state, node, 1) == 0.0
    state.build_road(2, state.board.node_edges[node][0], free=True)
    assert scorer.blocking_value(state, node, 1) > 0.0


def test_your_own_road_is_not_blocking(scorer, state):
    node = 25
    state.build_road(1, state.board.node_edges[node][0], free=True)
    assert scorer.blocking_value(state, node, 1) == 0.0
