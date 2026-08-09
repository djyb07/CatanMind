"""
The turn machine: sequence, legality, and who is allowed to act.

These are the rules the old screen had none of — it let one player take four
free settlements in a row and offered every action at every moment.
"""

import pytest

from catanmind.board import Board, Building, DevCard, Layout, Resource
from catanmind.flow import Step, TurnFlow, snake_order
from catanmind.state import GameState, Phase
from catanmind import rules


@pytest.fixture(scope="module")
def board():
    return Board()


@pytest.fixture
def state(board):
    return GameState(board, num_players=4, me=1)


@pytest.fixture
def flow(state):
    return TurnFlow(state)


def give(state, player, **resources):
    for name, count in resources.items():
        state.adjust(player, Resource(name), count)


def place_one(flow):
    """Place the current seat's settlement and a road touching it."""
    node = next(
        n for n in rules.legal_settlements(flow.state, flow.current, setup=True)
    )
    assert flow.place_setup_settlement(node).ok
    edge = next(
        e for e in flow.state.board.node_edges[node]
        if e not in flow.state.roads
    )
    assert flow.place_setup_road(edge).ok
    return node


def run_setup(flow):
    """Play the whole setup phase out legally."""
    while flow.in_setup:
        place_one(flow)
    return flow


def clear_hands(state):
    """
    Empty every hand.

    Setup legitimately pays out each player's second settlement, so play does
    not begin from zero. Tests about cost and affordability want a known
    starting point, and say so explicitly rather than assuming one.
    """
    for player in state.players.values():
        for resource in Resource:
            player.hand.cards[resource] = 0


def reach_main(flow, roll=8, empty_hands=True):
    """Get to the building step of the first turn."""
    run_setup(flow)
    if roll == 7:
        raise ValueError("use a non-7 roll")
    flow.roll(roll)
    if empty_hands:
        clear_hands(flow.state)
    return flow


# -- setup sequence --------------------------------------------------------


def test_the_game_starts_in_setup_with_the_first_seat(flow):
    assert flow.in_setup is True
    assert flow.step is Step.SETUP_SETTLEMENT
    assert flow.current == 1


def test_setup_follows_snake_order(flow):
    seen = []
    while flow.in_setup:
        seen.append(flow.current)
        place_one(flow)
    assert seen == snake_order(4)


def test_a_settlement_must_be_followed_by_a_road(flow):
    node = next(rules.legal_settlements(flow.state, 1, setup=True).__iter__())
    flow.place_setup_settlement(node)
    assert flow.step is Step.SETUP_ROAD
    assert flow.current == 1, "the same player lays the road"


def test_you_cannot_place_two_settlements_in_a_row(flow):
    """The exact abuse the old screen allowed."""
    first = next(iter(rules.legal_settlements(flow.state, 1, setup=True)))
    flow.place_setup_settlement(first)
    other = next(
        n for n in rules.legal_settlements(flow.state, 1, setup=True)
        if n != first
    )
    result = flow.place_setup_settlement(other)
    assert not result.ok
    assert other not in flow.state.buildings


def test_the_setup_road_must_touch_the_new_settlement(flow):
    spot = next(iter(rules.legal_settlements(flow.state, 1, setup=True)))
    flow.place_setup_settlement(spot)
    far = next(
        e.id for e in flow.state.board.edges
        if spot not in (e.a, e.b) and e.id not in flow.state.roads
    )
    assert not flow.place_setup_road(far).ok
    assert far not in flow.state.roads


def test_setup_placements_are_free(flow):
    run_setup(flow)
    # Only the second settlement pays out; nothing was ever charged.
    for player in flow.state.players:
        assert len(flow.state.settlements_of(player)) == 2
        assert len(flow.state.edges_of(player)) == 2


def test_the_second_settlement_pays_out_its_tiles(state):
    flow = TurnFlow(state)
    # First round: nobody collects.
    for _ in range(state.num_players):
        place_one(flow)
    assert all(p.hand.total() == 0 for p in state.players.values())

    # Second round: each placement collects from its own tiles.
    node = next(rules.legal_settlements(state, flow.current, setup=True).__iter__())
    player = flow.current
    expected = sum(
        1 for t in state.board.node_tiles[node] if t.resource is not None
    )
    flow.place_setup_settlement(node)
    assert state.players[player].hand.total() == expected


def test_setup_ends_in_the_play_phase_with_the_first_seat(flow):
    run_setup(flow)
    assert flow.in_setup is False
    assert flow.step is Step.PRE_ROLL
    assert flow.current == 1
    assert flow.state.phase is Phase.PLAY


# -- turn sequence ---------------------------------------------------------


def test_a_turn_starts_before_the_roll(flow):
    run_setup(flow)
    ids = {a.id for a in flow.available_actions()}
    assert "roll" in ids
    assert "build_settlement" not in ids, "cannot build before rolling"


def test_rolling_moves_to_the_building_step(flow):
    reach_main(flow)
    assert flow.step is Step.MAIN
    assert flow.has_rolled is True


def test_you_cannot_roll_twice(flow):
    reach_main(flow)
    assert not flow.roll(6).ok


def test_a_roll_pays_every_player_who_owns_the_tile(flow):
    run_setup(flow)
    tile = next(
        t for t in flow.state.board.tiles.values()
        if not t.is_desert and any(n in flow.state.buildings for n in t.nodes)
        and t.coord != flow.state.robber
    )
    owners = [
        flow.state.buildings[n][0] for n in tile.nodes
        if n in flow.state.buildings
    ]
    before = {p: flow.state.players[p].hand.total() for p in flow.state.players}
    flow.roll(tile.number)
    for p in owners:
        assert flow.state.players[p].hand.total() > before[p]


def test_ending_a_turn_advances_to_the_next_player(flow):
    reach_main(flow)
    assert flow.current == 1
    flow.end_turn()
    assert flow.current == 2
    assert flow.step is Step.PRE_ROLL
    assert flow.has_rolled is False


def test_turns_wrap_around_the_table(flow):
    reach_main(flow)
    for expected in (2, 3, 4, 1):
        flow.end_turn()
        assert flow.current == expected
        flow.roll(8)


def test_you_cannot_end_a_turn_before_rolling(flow):
    run_setup(flow)
    assert not flow.end_turn().ok


# -- rolling a seven -------------------------------------------------------


def test_a_seven_with_small_hands_goes_straight_to_the_robber(flow):
    run_setup(flow)
    clear_hands(flow.state)
    flow.roll(7)
    assert flow.step is Step.MOVE_ROBBER


def test_a_seven_makes_a_fat_hand_discard_first(flow):
    run_setup(flow)
    clear_hands(flow.state)
    give(flow.state, 2, wood=5, ore=5)
    flow.roll(7)
    assert flow.step is Step.DISCARD
    assert 2 in flow.pending_discards()


def test_the_robber_waits_until_everyone_has_discarded(flow):
    run_setup(flow)
    clear_hands(flow.state)
    give(flow.state, 2, wood=5, ore=5)
    give(flow.state, 3, sheep=9)
    flow.roll(7)
    assert set(flow.pending_discards()) == {2, 3}

    flow.discard(2, [Resource.WOOD] * 5)
    assert flow.step is Step.DISCARD, "player 3 still owes"
    flow.discard(3, [Resource.SHEEP] * 4)
    assert flow.step is Step.MOVE_ROBBER


def test_a_discard_must_be_the_right_size(flow):
    run_setup(flow)
    clear_hands(flow.state)
    give(flow.state, 2, wood=5, ore=5)
    flow.roll(7)
    assert not flow.discard(2, [Resource.WOOD]).ok
    assert flow.discard(2, [Resource.WOOD] * 5).ok


def test_the_robber_has_to_actually_move(flow):
    run_setup(flow)
    flow.roll(7)
    assert not flow.move_robber(flow.state.robber).ok


def test_moving_the_robber_offers_only_real_victims(flow):
    run_setup(flow)
    flow.roll(7)
    tile = next(
        t for t in flow.state.board.tiles.values()
        if t.coord != flow.state.robber
        and any(
            flow.state.buildings.get(n, (0,))[0] not in (0, 1)
            for n in t.nodes
        )
    )
    for node in tile.nodes:
        entry = flow.state.buildings.get(node)
        if entry and entry[0] != 1:
            give(flow.state, entry[0], wood=1)
    flow.move_robber(tile.coord)
    victims = flow.steal_victims()
    assert victims
    assert 1 not in victims, "you never rob yourself"
    for v in victims:
        assert flow.state.players[v].hand.total() > 0


def test_an_empty_tile_skips_the_steal(flow):
    run_setup(flow)
    flow.roll(7)
    empty = next(
        t for t in flow.state.board.tiles.values()
        if t.coord != flow.state.robber
        and not any(n in flow.state.buildings for n in t.nodes)
    )
    flow.move_robber(empty.coord)
    assert flow.step is Step.MAIN, "nobody to rob, so play continues"


def test_stealing_moves_a_card(flow):
    run_setup(flow)
    flow.roll(7)
    tile = next(
        t for t in flow.state.board.tiles.values()
        if t.coord != flow.state.robber
        and any(
            n in flow.state.buildings and flow.state.buildings[n][0] != 1
            for n in t.nodes
        )
    )
    victim = next(
        flow.state.buildings[n][0] for n in tile.nodes
        if n in flow.state.buildings and flow.state.buildings[n][0] != 1
    )
    give(flow.state, victim, ore=3)
    flow.move_robber(tile.coord)
    before = flow.state.players[1].hand.total()
    assert flow.steal(victim, Resource.ORE).ok
    assert flow.state.players[1].hand.total() == before + 1
    assert flow.step is Step.MAIN


# -- building --------------------------------------------------------------


def test_building_costs_resources(flow):
    reach_main(flow)
    give(flow.state, 1, wood=1, brick=1)
    edge = rules.legal_roads(flow.state, 1)[0]
    assert flow.build("road", edge).ok
    assert flow.state.players[1].hand.total() == 0


def test_you_cannot_build_what_you_cannot_pay_for(flow):
    reach_main(flow)
    flow.state.players[1].hand.cards[Resource.WOOD] = 0
    edge = rules.legal_roads(flow.state, 1)[0]
    result = flow.build("road", edge)
    assert not result.ok
    assert edge not in flow.state.roads


def test_building_is_not_offered_before_the_roll(flow):
    run_setup(flow)
    give(flow.state, 1, wood=4, brick=4, sheep=4, wheat=4, ore=4)
    edge = rules.legal_roads(flow.state, 1)[0]
    assert not flow.build("road", edge).ok


def test_unaffordable_actions_are_listed_but_disabled(flow):
    reach_main(flow)
    actions = {a.id: a for a in flow.available_actions()}
    assert "build_road" in actions
    assert actions["build_road"].enabled is False
    assert actions["build_road"].hint


def test_affordable_actions_become_enabled(flow):
    reach_main(flow)
    give(flow.state, 1, wood=1, brick=1)
    actions = {a.id: a for a in flow.available_actions()}
    assert actions["build_road"].enabled is True


# -- development cards -----------------------------------------------------


def test_a_knight_before_the_roll_moves_the_robber(flow):
    run_setup(flow)
    flow.state.players[1].dev_cards[DevCard.KNIGHT] = 1
    assert flow.play_knight().ok
    assert flow.step is Step.MOVE_ROBBER


def test_after_a_pre_roll_knight_you_still_have_to_roll(flow):
    run_setup(flow)
    flow.state.players[1].dev_cards[DevCard.KNIGHT] = 1
    flow.play_knight()
    empty = next(
        t for t in flow.state.board.tiles.values()
        if t.coord != flow.state.robber
        and not any(n in flow.state.buildings for n in t.nodes)
    )
    flow.move_robber(empty.coord)
    assert flow.step is Step.PRE_ROLL
    assert "roll" in {a.id for a in flow.available_actions()}


def test_only_one_development_card_per_turn(flow):
    reach_main(flow)
    flow.state.players[1].dev_cards[DevCard.KNIGHT] = 2
    flow.play_knight()
    empty = next(
        t for t in flow.state.board.tiles.values()
        if t.coord != flow.state.robber
        and not any(n in flow.state.buildings for n in t.nodes)
    )
    flow.move_robber(empty.coord)
    assert not flow.play_knight().ok


def test_a_card_drawn_this_turn_cannot_be_played_yet(flow):
    """The rule everyone forgets, and the one the old screen broke outright."""
    reach_main(flow)
    give(flow.state, 1, sheep=1, wheat=1, ore=1)
    assert flow.buy_dev(DevCard.KNIGHT).ok
    assert flow.playable_dev_cards() == [], "just-drawn cards are not playable"
    result = flow.play_dev(DevCard.KNIGHT)
    assert not result.ok
    assert "this turn" in (result.reason or "")


def test_the_same_card_becomes_playable_next_turn(flow):
    reach_main(flow)
    give(flow.state, 1, sheep=1, wheat=1, ore=1)
    flow.buy_dev(DevCard.KNIGHT)
    flow.end_turn()
    for _ in range(flow.state.num_players - 1):
        flow.roll(8)
        flow.end_turn()
    assert flow.current == 1
    assert DevCard.KNIGHT in flow.playable_dev_cards()


def test_buying_records_which_card_was_drawn(flow):
    reach_main(flow)
    give(flow.state, 1, sheep=1, wheat=1, ore=1)
    flow.buy_dev(DevCard.MONOPOLY)
    assert flow.state.players[1].holds(DevCard.MONOPOLY) == 1


def test_an_opponents_card_is_recorded_as_unknown(flow):
    reach_main(flow)
    flow.end_turn()
    flow.roll(8)
    give(flow.state, 2, sheep=1, wheat=1, ore=1)
    flow.buy_dev(None)
    assert flow.state.players[2].unknown_dev == 1
    assert flow.state.players[2].dev_cards_held == 1


def test_a_victory_point_card_is_revealed_not_played(flow):
    reach_main(flow)
    flow.state.players[1].dev_cards[DevCard.VICTORY_POINT] = 1
    assert DevCard.VICTORY_POINT not in flow.playable_dev_cards()
    assert flow.revealable_vp() is True
    assert flow.reveal_vp().ok
    assert flow.state.players[1].vp_cards == 1


def test_revealing_a_point_does_not_use_up_your_card_for_the_turn(flow):
    """A victory point is not an action, so a knight can still follow it."""
    reach_main(flow)
    flow.state.players[1].dev_cards[DevCard.VICTORY_POINT] = 1
    flow.state.players[1].dev_cards[DevCard.KNIGHT] = 1
    flow.reveal_vp()
    assert flow.dev_played is False
    assert DevCard.KNIGHT in flow.playable_dev_cards()


def test_monopoly_takes_that_resource_from_everyone(flow):
    reach_main(flow)
    flow.state.players[1].dev_cards[DevCard.MONOPOLY] = 1
    give(flow.state, 2, ore=3)
    give(flow.state, 3, ore=2)
    give(flow.state, 4, wood=4)
    assert flow.play_dev(DevCard.MONOPOLY, resource=Resource.ORE).ok
    assert flow.state.players[1].hand.cards[Resource.ORE] == 5
    assert flow.state.players[2].hand.cards[Resource.ORE] == 0
    assert flow.state.players[3].hand.cards[Resource.ORE] == 0
    assert flow.state.players[4].hand.cards[Resource.WOOD] == 4


def test_monopoly_needs_a_resource_named(flow):
    reach_main(flow)
    flow.state.players[1].dev_cards[DevCard.MONOPOLY] = 1
    assert not flow.play_dev(DevCard.MONOPOLY).ok


def test_year_of_plenty_hands_over_exactly_two_cards(flow):
    reach_main(flow)
    flow.state.players[1].dev_cards[DevCard.YEAR_OF_PLENTY] = 1
    before = flow.state.players[1].hand.total()
    assert flow.play_dev(
        DevCard.YEAR_OF_PLENTY, cards=[Resource.ORE, Resource.WHEAT]
    ).ok
    assert flow.state.players[1].hand.total() == before + 2
    assert flow.state.players[1].hand.cards[Resource.ORE] >= 1


def test_year_of_plenty_refuses_the_wrong_number_of_cards(flow):
    reach_main(flow)
    flow.state.players[1].dev_cards[DevCard.YEAR_OF_PLENTY] = 1
    assert not flow.play_dev(DevCard.YEAR_OF_PLENTY, cards=[Resource.ORE]).ok


def test_road_building_grants_two_free_roads(flow):
    reach_main(flow)
    flow.state.players[1].dev_cards[DevCard.ROAD_BUILDING] = 1
    assert flow.play_dev(DevCard.ROAD_BUILDING).ok
    assert flow.step is Step.ROAD_BUILDING
    assert flow.free_roads == 2

    for expected_left in (1, 0):
        edge = rules.legal_roads(flow.state, 1)[0]
        assert flow.place_free_road(edge).ok
        assert flow.free_roads == expected_left
    assert flow.step is Step.MAIN
    assert flow.state.players[1].hand.total() == 0, "free roads cost nothing"


def test_road_building_before_the_roll_returns_to_the_roll(flow):
    run_setup(flow)
    clear_hands(flow.state)
    flow.state.players[1].dev_cards[DevCard.ROAD_BUILDING] = 1
    flow.play_dev(DevCard.ROAD_BUILDING)
    for _ in range(2):
        flow.place_free_road(rules.legal_roads(flow.state, 1)[0])
    assert flow.step is Step.PRE_ROLL


def test_only_one_card_of_any_kind_per_turn(flow):
    reach_main(flow)
    flow.state.players[1].dev_cards[DevCard.MONOPOLY] = 1
    flow.state.players[1].dev_cards[DevCard.YEAR_OF_PLENTY] = 1
    flow.play_dev(DevCard.MONOPOLY, resource=Resource.ORE)
    assert flow.playable_dev_cards() == []
    assert not flow.play_dev(
        DevCard.YEAR_OF_PLENTY, cards=[Resource.ORE, Resource.WOOD]
    ).ok


def test_you_cannot_play_a_card_you_do_not_hold(flow):
    reach_main(flow)
    assert not flow.play_dev(DevCard.MONOPOLY, resource=Resource.ORE).ok


def test_dev_cards_are_not_offered_during_setup(flow):
    flow.state.players[1].dev_cards[DevCard.KNIGHT] = 1
    assert flow.playable_dev_cards() == []


def test_buying_a_development_card_costs_and_draws(flow):
    reach_main(flow)
    give(flow.state, 1, sheep=1, wheat=1, ore=1)
    assert flow.buy_dev().ok
    assert flow.state.players[1].dev_cards_held == 1
    assert flow.state.players[1].hand.total() == 0


# -- trading ---------------------------------------------------------------


def test_a_bank_trade_needs_the_full_rate(flow):
    """The rate is whatever the player's ports give them, not a flat 4:1."""
    reach_main(flow)
    rate = flow.state.ports_of(1)[Resource.WOOD]
    give(flow.state, 1, wood=rate - 1)
    assert not flow.trade_bank(Resource.WOOD, Resource.ORE).ok
    give(flow.state, 1, wood=1)
    assert flow.trade_bank(Resource.WOOD, Resource.ORE).ok
    assert flow.state.players[1].hand.cards[Resource.ORE] == 1
    assert flow.state.players[1].hand.cards[Resource.WOOD] == 0


def test_a_port_beats_the_default_bank_rate(flow):
    """A settlement on a 3:1 port must trade three, not four."""
    reach_main(flow)
    rates = flow.state.ports_of(1)
    assert all(2 <= r <= 4 for r in rates.values())


# -- undo ------------------------------------------------------------------


def test_undo_rewinds_the_turn_machine_too(flow):
    reach_main(flow)
    assert flow.step is Step.MAIN
    flow.undo()          # undo the roll
    assert flow.step is Step.PRE_ROLL
    assert flow.has_rolled is False


def test_undo_steps_back_across_a_turn_boundary(flow):
    reach_main(flow)
    flow.end_turn()
    assert flow.current == 2
    flow.undo()
    assert flow.current == 1
    assert flow.step is Step.MAIN


def test_undo_returns_to_the_setup_phase(flow):
    run_setup(flow)
    assert flow.in_setup is False
    flow.undo()          # the set_phase event
    flow.undo()          # the last setup road
    assert flow.in_setup is True
    assert flow.step is Step.SETUP_ROAD


# -- the banner and available actions --------------------------------------


def test_only_legal_actions_are_ever_offered(flow):
    """Whatever the step, every action returned must actually run."""
    run_setup(flow)
    for _ in range(3):
        for action in flow.available_actions():
            assert action.id
            assert action.label
        flow.roll(8)
        flow.end_turn()


def test_the_banner_names_the_player_and_the_step(flow):
    assert "Setup" in flow.banner()
    run_setup(flow)
    assert "roll" in flow.banner().lower()
    flow.roll(8)
    flow.end_turn()
    assert "Player 2" in flow.banner()


def test_the_banner_reads_as_english(flow):
    """Regression: the setup line used to read 'Your place a settlement'."""
    assert flow.banner() == "Setup round 1 — You: place a settlement"
    node = next(iter(rules.legal_settlements(flow.state, 1, setup=True)))
    flow.place_setup_settlement(node)
    assert flow.banner() == "Setup round 1 — You: place a road"

    edge = next(
        e for e in flow.state.board.node_edges[node] if e not in flow.state.roads
    )
    flow.place_setup_road(edge)
    assert flow.banner() == "Setup round 1 — Player 2: place a settlement"

    run_setup(flow)
    assert flow.banner() == "Your turn — roll the dice"
    flow.roll(8)
    flow.end_turn()
    assert flow.banner() == "Player 2's turn — roll the dice"


def test_the_game_ends_when_someone_reaches_the_target(flow):
    reach_main(flow)
    for node in flow.state.settlements_of(1):
        flow.state.build_city(1, node, free=True)
    flow.state.players[1].vp_cards = 6
    flow._derive()
    assert flow.step is Step.OVER
    assert flow.winner() == 1
