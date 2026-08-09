"""Game rules: legality, longest road, largest army, victory points."""

import pytest

from catanmind.board import Board, Resource, SUPPLY
from catanmind.state import GameState
from catanmind import rules


@pytest.fixture
def state():
    return GameState(Board())


def chain(state, player, length, start=0):
    """Lay ``length`` roads in a simple non-repeating path. Returns the nodes."""
    board = state.board
    node, seen, nodes = start, {start}, [start]
    for _ in range(length):
        nxt = [m for m in board.node(node).neighbors if m not in seen]
        if not nxt:
            break
        m = nxt[0]
        seen.add(m)
        nodes.append(m)
        state.build_road(player, board.edge_id[(node, m)], free=True)
        node = m
    return nodes


# -- settlements -----------------------------------------------------------


def test_setup_settlement_needs_no_road(state):
    assert rules.can_place_settlement(state, 1, 0, setup=True)


def test_settlement_outside_setup_needs_a_road(state):
    check = rules.can_place_settlement(state, 1, 0, setup=False)
    assert not check
    assert "road" in check.reason

    state.build_road(1, state.board.node_edges[0][0], free=True)
    assert rules.can_place_settlement(state, 1, 0, setup=False)


def test_distance_rule(state):
    state.build_settlement(1, 0, free=True)
    for neighbour in state.board.node(0).neighbors:
        check = rules.can_place_settlement(state, 2, neighbour, setup=True)
        assert not check
        assert "Distance rule" in check.reason


def test_cannot_build_on_an_occupied_node(state):
    state.build_settlement(1, 0, free=True)
    check = rules.can_place_settlement(state, 2, 0, setup=True)
    assert not check and "taken" in check.reason


def test_unknown_node_is_rejected(state):
    assert not rules.can_place_settlement(state, 1, 999, setup=True)
    assert not rules.can_place_settlement(state, 1, -1, setup=True)


def test_settlement_supply_limit(state):
    board = state.board
    placed = 0
    for node in board.nodes:
        if rules.can_place_settlement(state, 1, node.id, setup=True):
            state.build_settlement(1, node.id, free=True)
            placed += 1
        if placed == SUPPLY["settlement"]:
            break
    assert state.remaining(1, "settlement") == 0
    spare = next(
        n.id for n in board.nodes
        if n.id not in state.buildings
        and not any(m in state.buildings for m in board.node(n.id).neighbors)
    )
    check = rules.can_place_settlement(state, 1, spare, setup=True)
    assert not check and "no settlements left" in check.reason


# -- cities ----------------------------------------------------------------


def test_city_upgrade_rules(state):
    assert not rules.can_upgrade_city(state, 1, 0)  # nothing there

    state.build_settlement(1, 0, free=True)
    assert rules.can_upgrade_city(state, 1, 0)
    assert not rules.can_upgrade_city(state, 2, 0)  # not yours

    state.build_city(1, 0, free=True)
    check = rules.can_upgrade_city(state, 1, 0)
    assert not check and "already a city" in check.reason


# -- roads -----------------------------------------------------------------


def test_road_must_touch_your_network(state):
    edge = state.board.edges[0]
    assert not rules.can_place_road(state, 1, edge.id)

    state.build_settlement(1, edge.a, free=True)
    assert rules.can_place_road(state, 1, edge.id)


def test_road_cannot_be_placed_twice(state):
    edge = state.board.edges[0]
    state.build_settlement(1, edge.a, free=True)
    state.build_road(1, edge.id, free=True)
    check = rules.can_place_road(state, 1, edge.id)
    assert not check and "already has a road" in check.reason


def test_enemy_settlement_blocks_road_continuation(state):
    """You may not extend *through* an opponent's building..."""
    board = state.board
    nodes = chain(state, 1, 2, start=0)
    junction = nodes[2]
    state.build_settlement(2, junction, free=True)

    beyond = [m for m in board.node(junction).neighbors if m not in nodes]
    for m in beyond:
        check = rules.can_place_road(state, 1, board.edge_id[(junction, m)])
        assert not check, f"road past enemy settlement at {junction} should be illegal"


def test_road_from_your_own_settlement_past_an_enemy_is_legal(state):
    """...but a road anchored at your *own* building on the far side is fine.

    The old validator rejected this case, which is a legal and common play.
    """
    board = state.board
    nodes = chain(state, 1, 1, start=0)          # my road 0 -> nodes[1]
    enemy_node = nodes[1]
    state.build_settlement(2, enemy_node, free=True)

    onward = next(m for m in board.node(enemy_node).neighbors if m not in nodes)
    state.build_settlement(1, onward, free=True)  # my building on the far side

    check = rules.can_place_road(state, 1, board.edge_id[(enemy_node, onward)])
    assert check, check.reason


def test_setup_road_must_touch_the_new_settlement(state):
    board = state.board
    node = 0
    good = board.node_edges[node][0]
    assert rules.can_place_setup_road(state, 1, good, node)

    far = next(
        e.id for e in board.edges if node not in (e.a, e.b)
    )
    check = rules.can_place_setup_road(state, 1, far, node)
    assert not check and "touch" in check.reason


# -- legal move enumeration ------------------------------------------------


def test_legal_settlements_on_an_empty_board(state):
    assert len(rules.legal_settlements(state, 1, setup=True)) == 54
    assert rules.legal_settlements(state, 1, setup=False) == []


def test_legal_moves_shrink_as_the_board_fills(state):
    before = len(rules.legal_settlements(state, 1, setup=True))
    state.build_settlement(1, 0, free=True)
    after = len(rules.legal_settlements(state, 1, setup=True))
    assert after == before - 1 - len(state.board.node(0).neighbors)


def test_legal_roads(state):
    assert rules.legal_roads(state, 1) == []
    state.build_settlement(1, 0, free=True)
    assert sorted(rules.legal_roads(state, 1)) == sorted(state.board.node_edges[0])


# -- longest road ----------------------------------------------------------


def test_no_roads_means_zero(state):
    assert rules.longest_road(state, 1) == 0


def test_straight_chain(state):
    chain(state, 1, 4)
    assert rules.longest_road(state, 1) == 4


def test_three_roads_from_one_junction_is_length_two(state):
    """The classic line-graph bug: this used to report 3."""
    board = state.board
    hub = next(n.id for n in board.nodes if len(n.neighbors) == 3)
    for m in board.node(hub).neighbors:
        state.build_road(1, board.edge_id[(hub, m)], free=True)
    assert len(state.edges_of(1)) == 3
    assert rules.longest_road(state, 1) == 2


def test_branch_does_not_add_to_the_trunk(state):
    board = state.board
    nodes = chain(state, 1, 4)
    # Hang a spur off the middle of the trunk.
    middle = nodes[2]
    spur = next(m for m in board.node(middle).neighbors if m not in nodes)
    state.build_road(1, board.edge_id[(middle, spur)], free=True)
    assert len(state.edges_of(1)) == 5
    # Longest single route is trunk-start -> middle -> spur, or the full trunk.
    assert rules.longest_road(state, 1) == 4


def test_enemy_building_splits_a_chain(state):
    nodes = chain(state, 1, 4)
    state.build_settlement(2, nodes[2], free=True)
    assert rules.longest_road(state, 1) == 2


def test_own_building_does_not_split_a_chain(state):
    nodes = chain(state, 1, 4)
    state.build_settlement(1, nodes[2], free=True)
    assert rules.longest_road(state, 1) == 4


def test_roads_of_other_players_are_ignored(state):
    chain(state, 1, 3)
    chain(state, 2, 4, start=30)
    assert rules.longest_road(state, 1) == 3
    assert rules.longest_road(state, 2) == 4


def test_longest_road_award_needs_five_roads(state):
    chain(state, 1, 4)
    assert rules.longest_road_holder(state) is None
    chain(state, 1, 5)
    assert rules.longest_road(state, 1) >= 5
    assert rules.longest_road_holder(state) == 1


def test_the_longest_road_holder_keeps_it_on_a_tie(state):
    """
    The card does not change hands on a tie — it moves only when someone
    strictly beats the holder. Nobody hands over two points for drawing level.
    """
    chain(state, 1, 5, start=0)
    assert rules.longest_road_holder(state) == 1
    chain(state, 2, 5, start=30)
    assert rules.longest_road(state, 1) == rules.longest_road(state, 2) == 5
    assert rules.longest_road_holder(state) == 1, "a tie leaves the card put"


def test_beating_the_holder_takes_the_longest_road(state):
    chain(state, 1, 5, start=0)
    chain(state, 2, 6, start=30)
    assert rules.longest_road(state, 2) > rules.longest_road(state, 1)
    assert rules.longest_road_holder(state) == 2


def test_an_enemy_settlement_can_cost_you_the_longest_road(state):
    """Cutting a chain shortens it, and the card follows the new lengths."""
    chain(state, 1, 6, start=0)
    assert rules.longest_road_holder(state) == 1
    middle = state.board.edge(state.edges_of(1)[2]).b
    state.build_settlement(2, middle, free=True)
    assert rules.longest_road(state, 1) < 5
    assert rules.longest_road_holder(state) is None


def test_undo_restores_the_award_holder(state):
    """
    The holder is history, not a property of the position, so it has to come
    back correctly when the log is rewound.
    """
    chain(state, 1, 5, start=0)
    assert rules.longest_road_holder(state) == 1
    chain(state, 2, 6, start=30)
    assert rules.longest_road_holder(state) == 2
    while rules.longest_road(state, 2) >= rules.longest_road(state, 1):
        assert state.undo()
    assert rules.longest_road_holder(state) == 1


def test_nobody_holds_the_road_before_anyone_reaches_five(state):
    chain(state, 1, 4, start=0)
    chain(state, 2, 4, start=30)
    assert rules.longest_road_holder(state) is None


# -- largest army ----------------------------------------------------------


def test_largest_army_needs_three_knights(state):
    from catanmind.state import Event

    for _ in range(2):
        state.apply(Event.make("play_knight", player=1))
    assert rules.largest_army_holder(state) is None
    state.apply(Event.make("play_knight", player=1))
    assert rules.largest_army_holder(state) == 1


def test_the_largest_army_holder_keeps_it_on_a_tie(state):
    from catanmind.state import Event

    for _ in range(3):
        state.apply(Event.make("play_knight", player=1))
    assert rules.largest_army_holder(state) == 1
    for _ in range(3):
        state.apply(Event.make("play_knight", player=2))
    assert rules.largest_army_holder(state) == 1, "a tie leaves the card put"


def test_a_fourth_knight_takes_the_largest_army(state):
    from catanmind.state import Event

    for _ in range(3):
        state.apply(Event.make("play_knight", player=1))
        state.apply(Event.make("play_knight", player=2))
    state.apply(Event.make("play_knight", player=2))
    assert rules.largest_army_holder(state) == 2


# -- victory points --------------------------------------------------------


def test_victory_points_count_buildings_and_awards(state):
    from catanmind.state import Event

    state.build_settlement(1, 0, free=True)
    assert rules.victory_points(state, 1) == 1

    state.build_city(1, 0, free=True)
    assert rules.victory_points(state, 1) == 2

    chain(state, 1, 5, start=30)
    assert rules.victory_points(state, 1) == 4  # +2 longest road

    for _ in range(3):
        state.apply(Event.make("play_knight", player=1))
    assert rules.victory_points(state, 1) == 6  # +2 largest army

    state.apply(Event.make("reveal_vp", player=1))
    assert rules.victory_points(state, 1) == 7
    assert rules.victory_points(state, 1, include_hidden=False) == 6


def test_winner_is_detected(state):
    from catanmind.state import Event

    assert rules.winner(state) is None
    for i, node in enumerate(rules.legal_settlements(state, 1, setup=True)[:5]):
        if rules.can_place_settlement(state, 1, node, setup=True):
            state.build_settlement(1, node, free=True)
            state.build_city(1, node, free=True)
    for _ in range(5):
        state.apply(Event.make("reveal_vp", player=1))
    assert rules.victory_points(state, 1) >= 10
    assert rules.winner(state) == 1


def test_sync_awards_writes_to_players(state):
    chain(state, 1, 5)
    rules.sync_awards(state)
    assert state.players[1].has_longest_road
    assert not state.players[2].has_longest_road


# -- production ------------------------------------------------------------


def test_expected_yield_counts_cities_double(state):
    state.build_settlement(1, 0, free=True)
    single = rules.expected_yield(state, 1, ignore_robber=True)
    state.build_city(1, 0, free=True)
    double = rules.expected_yield(state, 1, ignore_robber=True)
    for r in single:
        assert double[r] == pytest.approx(single[r] * 2)


def test_robber_suppresses_production(state):
    board = state.board
    node = next(
        n.id for n in board.nodes
        if len(n.tiles) == 3 and all(board.tiles[c].resource for c in n.tiles)
    )
    state.build_settlement(1, node, free=True)
    full = rules.expected_yield(state, 1, ignore_robber=True)

    state.move_robber(board.node(node).tiles[0])
    robbed = rules.expected_yield(state, 1)
    assert sum(robbed.values()) < sum(full.values())


def test_must_discard(state):
    assert rules.must_discard(state, 1) == 0
    for _ in range(7):
        state.adjust(1, Resource.WOOD, 1)
    assert rules.must_discard(state, 1) == 0  # exactly 7 is safe
    state.adjust(1, Resource.WOOD, 1)
    assert rules.must_discard(state, 1) == 4
    state.adjust(1, Resource.WOOD, 1)
    assert rules.must_discard(state, 1) == 4  # 9 cards -> discard 4


def test_can_afford_reports_what_is_missing(state):
    check = rules.can_afford(state, 1, "city")
    assert not check
    assert "wheat" in check.reason and "ore" in check.reason

    state.adjust(1, Resource.WHEAT, 2)
    state.adjust(1, Resource.ORE, 3)
    assert rules.can_afford(state, 1, "city")
