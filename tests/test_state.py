"""Game state, the event log, and undo."""

import pytest

from catanmind.board import Board, COSTS, Layout, Resource, RESOURCES
from catanmind.state import Event, GameState, Hand
from catanmind import rules


@pytest.fixture
def state():
    return GameState(Board())


# -- hands -----------------------------------------------------------------


def test_hand_never_goes_negative():
    hand = Hand()
    hand.add(Resource.WOOD, 2)
    assert hand.take(Resource.WOOD, 5) == 2
    assert hand.cards[Resource.WOOD] == 0
    assert hand.total() == 0


def test_hand_pay_is_all_or_nothing():
    hand = Hand()
    hand.add(Resource.WOOD, 1)
    assert not hand.pay(COSTS["settlement"])
    assert hand.cards[Resource.WOOD] == 1, "a failed payment must not take cards"

    for r in RESOURCES:
        hand.add(r, 1)
    assert hand.pay(COSTS["settlement"])
    assert hand.cards[Resource.WOOD] == 1


# -- production ------------------------------------------------------------


def test_roll_pays_settlements_and_cities(state):
    board = state.board
    tile = next(t for t in board.tiles.values() if t.resource and t.number)
    node = tile.nodes[0]

    state.build_settlement(1, node, free=True)
    state.roll(tile.number)
    assert state.players[1].hand.cards[tile.resource] == 1

    state.build_city(1, node, free=True)
    state.roll(tile.number)
    assert state.players[1].hand.cards[tile.resource] == 3


def test_seven_pays_nobody(state):
    state.board
    for node in range(0, 54, 7):
        if rules.can_place_settlement(state, 1, node, setup=True):
            state.build_settlement(1, node, free=True)
    state.roll(7)
    assert state.players[1].hand.total() == 0
    assert state.last_roll == 7


def test_robber_blocks_its_tile(state):
    board = state.board
    tile = next(t for t in board.tiles.values() if t.resource and t.number)
    node = tile.nodes[0]
    state.build_settlement(1, node, free=True)
    state.move_robber(tile.coord)
    state.roll(tile.number)
    assert state.players[1].hand.cards[tile.resource] == 0


def test_building_pays_for_itself(state):
    for r in RESOURCES:
        state.adjust(1, r, 1)
    state.build_settlement(1, 0)
    assert state.players[1].hand.cards[Resource.WOOD] == 0
    assert state.players[1].hand.cards[Resource.ORE] == 1  # not part of the cost


def test_free_building_costs_nothing(state):
    for r in RESOURCES:
        state.adjust(1, r, 1)
    state.build_settlement(1, 0, free=True)
    assert state.players[1].hand.total() == 5


# -- undo ------------------------------------------------------------------


def test_undo_rewinds_exactly_one_action(state):
    """The old tracker's undo jumped back two steps. This checks it does not."""
    state.adjust(1, Resource.WOOD, 5)
    state.adjust(1, Resource.BRICK, 5)

    state.build_road(1, state.board.node_edges[0][0])
    after_first = state.players[1].hand.cards[Resource.WOOD]
    assert after_first == 4

    edge = state.board.node_edges[0][1]
    state.build_road(1, edge)
    assert state.players[1].hand.cards[Resource.WOOD] == 3

    assert state.undo()
    assert state.players[1].hand.cards[Resource.WOOD] == after_first
    assert state.players[1].hand.cards[Resource.BRICK] == 4


def test_the_first_action_can_be_undone(state):
    """The old implementation required two entries and could never undo one."""
    state.build_settlement(1, 0, free=True)
    assert state.buildings
    assert state.undo()
    assert state.buildings == {}
    assert state.log == []


def test_undo_on_an_empty_log_is_false(state):
    assert not state.undo()


def test_undo_restores_the_board_not_just_hands(state):
    state.build_settlement(1, 0, free=True)
    state.build_road(1, state.board.node_edges[0][0], free=True)
    state.move_robber((1, 1))

    state.undo()
    assert state.robber != (1, 1)
    state.undo()
    assert state.roads == {}
    state.undo()
    assert state.buildings == {}


def test_undo_all_the_way_returns_to_the_start(state):
    start = state.snapshot()
    state.build_settlement(1, 0, free=True)
    state.roll(6)
    state.adjust(1, Resource.ORE, 3)
    state.move_robber((2, 0))
    while state.undo():
        pass
    assert state.snapshot() == start


def test_replay_is_deterministic(state):
    for node in (0, 10, 20):
        if rules.can_place_settlement(state, 1, node, setup=True):
            state.build_settlement(1, node, free=True)
    state.roll(6)
    state.roll(8)
    before = state.snapshot()

    state._replay()
    assert state.snapshot() == before


def test_undo_after_many_actions_matches_a_fresh_replay(state):
    for node in (0, 10, 20, 30):
        if rules.can_place_settlement(state, 1, node, setup=True):
            state.build_settlement(1, node, free=True)
    for n in (4, 5, 6, 8, 9, 10):
        state.roll(n)
    state.adjust(2, Resource.SHEEP, 4)
    state.steal(1, 2)

    expected = GameState(state.board)
    for event in state.log[:-1]:
        expected.apply(event)

    state.undo()
    assert state.snapshot() == expected.snapshot()


# -- steal and discard -----------------------------------------------------


def test_known_steal_moves_a_specific_card(state):
    state.adjust(2, Resource.ORE, 2)
    state.steal(1, 2, Resource.ORE)
    assert state.players[2].hand.cards[Resource.ORE] == 1
    assert state.players[1].hand.cards[Resource.ORE] == 1


def test_unknown_steal_takes_the_likeliest_card(state):
    state.adjust(2, Resource.SHEEP, 1)
    state.adjust(2, Resource.WHEAT, 4)
    state.steal(1, 2)
    assert state.players[2].hand.cards[Resource.WHEAT] == 3
    assert state.players[1].hand.cards[Resource.WHEAT] == 1


def test_stealing_from_an_empty_hand_creates_nothing(state):
    state.steal(1, 2)
    assert state.players[1].hand.total() == 0
    assert state.players[2].hand.total() == 0


def test_discard_actually_removes_cards(state):
    """A 4-wheat/4-ore hand discarding 4 used to lose nothing at all."""
    state.adjust(1, Resource.WHEAT, 4)
    state.adjust(1, Resource.ORE, 4)
    assert rules.must_discard(state, 1) == 4

    state.apply(Event.make("discard", player=1, count=4))
    assert state.players[1].hand.total() == 4


def test_discard_sheds_from_the_biggest_stack(state):
    state.adjust(1, Resource.WHEAT, 6)
    state.adjust(1, Resource.ORE, 2)
    state.apply(Event.make("discard", player=1, count=4))
    assert state.players[1].hand.cards[Resource.WHEAT] == 2
    assert state.players[1].hand.cards[Resource.ORE] == 2


def test_explicit_discard_list_is_honoured(state):
    state.adjust(1, Resource.WHEAT, 4)
    state.adjust(1, Resource.ORE, 4)
    state.apply(Event.make("discard", player=1, count=2, cards="ore,ore"))
    assert state.players[1].hand.cards[Resource.ORE] == 2
    assert state.players[1].hand.cards[Resource.WHEAT] == 4


# -- trading ---------------------------------------------------------------


def test_bank_trade(state):
    state.adjust(1, Resource.SHEEP, 4)
    state.apply(
        Event.make("trade_bank", player=1, give="sheep", get="ore", rate=4)
    )
    assert state.players[1].hand.cards[Resource.SHEEP] == 0
    assert state.players[1].hand.cards[Resource.ORE] == 1


def test_bank_trade_without_the_cards_does_nothing(state):
    state.adjust(1, Resource.SHEEP, 2)
    state.apply(
        Event.make("trade_bank", player=1, give="sheep", get="ore", rate=4)
    )
    assert state.players[1].hand.cards[Resource.ORE] == 0


def test_player_trade_moves_cards_both_ways(state):
    state.adjust(1, Resource.WOOD, 2)
    state.adjust(2, Resource.ORE, 1)
    state.apply(
        Event.make("trade_player", player=1, other=2, give="wood,wood", get="ore")
    )
    assert state.players[1].hand.cards[Resource.WOOD] == 0
    assert state.players[1].hand.cards[Resource.ORE] == 1
    assert state.players[2].hand.cards[Resource.WOOD] == 2
    assert state.players[2].hand.cards[Resource.ORE] == 0


# -- ports and supply ------------------------------------------------------


def test_ports_of_reports_best_rates(state):
    board = state.board
    rates = state.ports_of(1)
    assert set(rates.values()) == {4}

    node = next(n for n in board.nodes if n.port and n.port.resource)
    state.build_settlement(1, node.id, free=True)
    assert state.ports_of(1)[node.port.resource] == 2


def test_generic_port_gives_three_to_one_on_everything(state):
    from catanmind.board import Port

    node = next(n for n in state.board.nodes if n.port is Port.GENERIC)
    state.build_settlement(1, node.id, free=True)
    assert all(rate == 3 for rate in state.ports_of(1).values())


def test_supply_counters(state):
    assert state.remaining(1, "settlement") == 5
    state.build_settlement(1, 0, free=True)
    assert state.remaining(1, "settlement") == 4
    state.build_city(1, 0, free=True)
    assert state.remaining(1, "settlement") == 5
    assert state.remaining(1, "city") == 3


def test_dev_deck_runs_down(state):
    before = state.dev_deck_left()
    for r in (Resource.SHEEP, Resource.WHEAT, Resource.ORE):
        state.adjust(1, r, 1)
    state.apply(Event.make("buy_dev", player=1))
    assert state.dev_deck_left() == before - 1
    assert state.players[1].dev_cards_held == 1


# -- misc ------------------------------------------------------------------


def test_setting_a_layout_resets_the_game(state):
    import random

    state.build_settlement(1, 0, free=True)
    state.set_layout(Layout.random(random.Random(3)))
    assert state.buildings == {}
    assert state.log == []


def test_robber_starts_on_the_desert(state):
    assert state.board.tiles[state.robber].is_desert


def test_unknown_event_kind_is_rejected(state):
    with pytest.raises(ValueError):
        state.apply(Event.make("teleport", player=1))


def test_events_describe_themselves(state):
    state.build_settlement(1, 0, free=True)
    state.roll(8)
    assert "settled" in state.log[0].describe()
    assert "8" in state.log[1].describe()
