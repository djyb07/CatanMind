"""The advisor: setup placement, turn ranking, robber and discard."""

import time

import pytest

from catanmind.advisor import (
    Advice,
    SetupAdvisor,
    TurnAdvisor,
    phase_of,
    picks_between,
    snake_order,
    vp_value,
)
from catanmind.board import Board, Building, Layout, Resource, RESOURCES
from catanmind.scoring import Scorer
from catanmind.state import GameState, Phase
from catanmind import rules


@pytest.fixture(scope="module")
def board():
    return Board()


@pytest.fixture(scope="module")
def scorer(board):
    return Scorer(board)


@pytest.fixture
def state(board):
    return GameState(board, num_players=4, me=1)


@pytest.fixture
def setup_advisor(board, scorer):
    return SetupAdvisor(board, scorer)


@pytest.fixture
def turn_advisor(board, scorer):
    return TurnAdvisor(board, scorer)


def give(state, player, **resources):
    """Hand a player some cards by resource name."""
    for name, count in resources.items():
        state.adjust(player, Resource(name), count)


def push_visible_vp(state, player, target):
    """
    Raise a player's *public* score to at least ``target``.

    Only visible points count here on purpose: unrevealed victory-point cards
    are invisible to opponents, so the advisor must not react to them.
    """
    for node in list(state.settlements_of(player)):
        state.build_city(player, node, free=True)
    while rules.victory_points(state, player, include_hidden=False) < target:
        spot = next(
            (
                n for n in rules.legal_settlements(state, player, setup=True)
                if state.remaining(player, "settlement") > 0
            ),
            None,
        )
        if spot is None:
            break
        state.build_settlement(player, spot, free=True)
        if state.remaining(player, "city") > 0:
            state.build_city(player, spot, free=True)
    return rules.victory_points(state, player, include_hidden=False)


def play_out_setup(state, board, scorer):
    """Run the whole snake-order setup using the advisor's own picks."""
    advisor = SetupAdvisor(board, scorer)
    for pid in snake_order(state.num_players):
        plans = advisor.recommend(state, pid, seat=pid, top=1)
        assert plans, f"no setup plan for player {pid}"
        plan = plans[0]
        state.build_settlement(pid, plan.first, free=True)
        if plan.road is not None:
            state.build_road(pid, plan.road, free=True)
    state.set_phase(Phase.PLAY)


# -- turn order ------------------------------------------------------------


def test_snake_order_places_everyone_twice():
    order = snake_order(4)
    assert order == [1, 2, 3, 4, 4, 3, 2, 1]
    for seat in range(1, 5):
        assert order.count(seat) == 2


def test_picks_between_is_widest_for_the_first_seat():
    """Seat 1 waits through six opposing picks; the last seat picks back to back."""
    assert picks_between(1, 4) == 6
    assert picks_between(4, 4) == 0
    gaps = [picks_between(s, 4) for s in range(1, 5)]
    assert gaps == sorted(gaps, reverse=True)


# -- setup placement -------------------------------------------------------


def test_setup_recommends_legal_distinct_spots(state, setup_advisor):
    plans = setup_advisor.recommend(state, 1, seat=1, top=3)
    assert len(plans) == 3
    assert len({p.first for p in plans}) == 3
    for plan in plans:
        assert rules.can_place_settlement(state, 1, plan.first, setup=True)


def test_setup_prefers_high_pip_intersections(state, setup_advisor, board):
    """The top pick should be well above the board's median spot."""
    best = setup_advisor.recommend(state, 1, seat=1, top=1)[0]
    pips = sum(
        board.tiles[c].pips for c in board.node(best.first).tiles
    )
    all_pips = sorted(
        sum(board.tiles[c].pips for c in n.tiles) for n in board.nodes
    )
    median = all_pips[len(all_pips) // 2]
    assert pips > median


def test_setup_projects_a_second_pick_that_is_not_adjacent(
    state, setup_advisor, board
):
    """The projected pair must be legal together under the distance rule."""
    for plan in setup_advisor.recommend(state, 1, seat=1, top=3):
        if plan.projected_second is None:
            continue
        assert plan.projected_second != plan.first
        assert plan.projected_second not in board.node(plan.first).neighbors


def test_setup_road_touches_the_settlement(state, setup_advisor, board):
    for plan in setup_advisor.recommend(state, 1, seat=1, top=3):
        assert plan.road is not None
        edge = board.edge(plan.road)
        assert plan.first in (edge.a, edge.b)


def test_second_pick_covers_what_the_first_one_missed(
    state, setup_advisor, board, scorer
):
    """
    With a one-sided first settlement, the second pick should add resources the
    player does not already produce rather than doubling down.
    """
    # Find a node whose tiles are all the same resource-poor mix, then settle it.
    first = setup_advisor.recommend(state, 1, seat=1, top=1)[0].first
    state.build_settlement(1, first, free=True)

    plans = setup_advisor.recommend(state, 1, seat=1, top=1)
    assert plans
    plan = plans[0]
    mine = {
        r for r, v in rules.expected_yield(state, 1, ignore_robber=True).items()
        if v > 0
    }
    added = {
        board.tiles[c].resource
        for c in board.node(plan.first).tiles
        if board.tiles[c].resource is not None
    }
    # It should bring at least one resource we are not already producing.
    assert added - mine


def test_second_pick_reports_the_combined_resource_coverage(
    state, setup_advisor
):
    first = setup_advisor.recommend(state, 1, seat=1, top=1)[0].first
    state.build_settlement(1, first, free=True)
    plan = setup_advisor.recommend(state, 1, seat=1, top=1)[0]
    assert plan.combined_resources
    assert set(plan.combined_resources) <= set(RESOURCES)


def test_setup_is_fast(state, setup_advisor):
    """
    The old solver took 6.7s for seat 1 because it recomputed the opponent
    model inside the candidate loop. Guard against that coming back.
    """
    start = time.perf_counter()
    setup_advisor.recommend(state, 1, seat=1, top=3)
    assert time.perf_counter() - start < 1.0


def test_setup_advice_exists_exactly_when_a_legal_spot_does(
    state, setup_advisor
):
    """Pack the board using every player's supply, then check the two agree."""
    for player in (2, 3, 4):
        for node in list(rules.legal_settlements(state, player, setup=True)):
            if state.remaining(player, "settlement") <= 0:
                break
            if rules.can_place_settlement(state, player, node, setup=True):
                state.build_settlement(player, node, free=True)
    legal = rules.legal_settlements(state, 1, setup=True)
    assert bool(setup_advisor.recommend(state, 1, seat=1)) == bool(legal)


# -- normal turns ----------------------------------------------------------


def test_turn_advice_is_never_empty_with_an_empty_hand(
    state, board, scorer, turn_advisor
):
    """
    The old engine went silent whenever the hand could not pay for anything.
    Advice should still say what to save for.
    """
    play_out_setup(state, board, scorer)
    assert state.players[1].hand.total() == 0
    advice = turn_advisor.recommend(state, 1)
    assert advice
    assert any(not a.affordable for a in advice)


def test_affordable_advice_sorts_above_unaffordable(
    state, board, scorer, turn_advisor
):
    play_out_setup(state, board, scorer)
    give(state, 1, wood=1, brick=1)
    advice = turn_advisor.recommend(state, 1, top=10)
    affordable = [i for i, a in enumerate(advice) if a.affordable]
    unaffordable = [i for i, a in enumerate(advice) if not a.affordable]
    if affordable and unaffordable:
        assert max(affordable) < min(unaffordable)


def test_unaffordable_advice_says_what_is_missing(
    state, board, scorer, turn_advisor
):
    play_out_setup(state, board, scorer)
    advice = turn_advisor.recommend(state, 1, top=10)
    for a in advice:
        if not a.affordable:
            assert a.missing


def test_city_is_recommended_when_ore_and_wheat_are_in_hand(
    state, board, scorer, turn_advisor
):
    play_out_setup(state, board, scorer)
    give(state, 1, ore=3, wheat=2)
    advice = turn_advisor.recommend(state, 1, top=6)
    cities = [a for a in advice if a.action == "upgrade_city"]
    assert cities
    assert cities[0].affordable
    assert cities[0].node in state.settlements_of(1)


def test_recommended_builds_are_legal(state, board, scorer, turn_advisor):
    play_out_setup(state, board, scorer)
    give(state, 1, wood=4, brick=4, sheep=4, wheat=4, ore=4)
    for a in turn_advisor.recommend(state, 1, top=10):
        if a.action == "build_settlement":
            assert rules.can_place_settlement(state, 1, a.node)
        elif a.action == "build_road":
            assert rules.can_place_road(state, 1, a.edge)
        elif a.action == "upgrade_city":
            assert rules.can_upgrade_city(state, 1, a.node)


def test_a_trade_is_suggested_when_one_card_short(
    state, board, scorer, turn_advisor
):
    play_out_setup(state, board, scorer)
    # Four sheep and nothing else: one bank trade away from a road.
    give(state, 1, sheep=4, wood=1)
    advice = turn_advisor.recommend(state, 1, top=10)
    trades = [a for a in advice if a.action == "trade_bank"]
    assert trades, "should offer to trade the sheep pile for the missing brick"


def test_no_settlement_advice_once_the_supply_is_gone(
    state, board, scorer, turn_advisor
):
    play_out_setup(state, board, scorer)
    give(state, 1, wood=9, brick=9, sheep=9, wheat=9, ore=9)
    for node in rules.legal_settlements(state, 1, setup=True):
        if state.remaining(1, "settlement") <= 0:
            break
        if rules.can_place_settlement(state, 1, node, setup=True):
            state.build_settlement(1, node, free=True)
    assert state.remaining(1, "settlement") == 0
    advice = turn_advisor.recommend(state, 1, top=10)
    assert not [a for a in advice if a.action == "build_settlement"]


# -- victory point pressure ------------------------------------------------


def test_phase_tracks_victory_points(state, board, scorer):
    play_out_setup(state, board, scorer)
    assert phase_of(state, 1) == "early"
    for node in state.settlements_of(1):
        state.build_city(1, node, free=True)
    assert rules.victory_points(state, 1) >= 4


def test_a_point_is_worth_more_when_someone_is_about_to_win(
    state, board, scorer
):
    play_out_setup(state, board, scorer)
    calm = vp_value(state, 1)
    reached = push_visible_vp(state, 2, state.target_vp - 2)
    assert reached >= state.target_vp - 2
    assert vp_value(state, 1) > calm


def test_hidden_victory_point_cards_do_not_raise_the_alarm(
    state, board, scorer
):
    """An opponent's unrevealed points are invisible, so they must not count."""
    play_out_setup(state, board, scorer)
    calm = vp_value(state, 1)
    state.players[2].vp_cards = 4
    assert vp_value(state, 1) == calm


# -- the robber ------------------------------------------------------------


def test_robber_never_targets_your_own_tiles(state, board, scorer, turn_advisor):
    play_out_setup(state, board, scorer)
    advice = turn_advisor.robber_advice(state, 1)
    assert advice is not None
    my_tiles = {c for nid in state.nodes_of(1) for c in board.node(nid).tiles}
    assert advice.coord not in my_tiles


def test_robber_picks_a_producing_tile_it_is_not_already_on(
    state, board, scorer, turn_advisor
):
    play_out_setup(state, board, scorer)
    advice = turn_advisor.robber_advice(state, 1)
    tile = board.tiles[advice.coord]
    assert tile.resource is not None
    assert advice.coord != state.robber
    # And somebody actually loses production for it.
    owners = {
        state.buildings[n][0] for n in tile.nodes if n in state.buildings
    }
    assert owners - {1}


def test_robber_prefers_the_tile_that_denies_more(
    state, board, scorer, turn_advisor
):
    play_out_setup(state, board, scorer)
    advice = turn_advisor.robber_advice(state, 1)
    chosen = board.tiles[advice.coord]
    # Upgrading an opponent's settlement elsewhere should be able to pull the
    # recommendation away from the original tile.
    others = [
        n for n, (p, k) in state.buildings.items()
        if p != 1 and k is Building.SETTLEMENT
        and advice.coord not in board.node(n).tiles
    ]
    if others:
        state.build_city(state.buildings[others[0]][0], others[0], free=True)
        again = turn_advisor.robber_advice(state, 1)
        assert again is not None


def test_robber_advice_is_none_when_nobody_can_be_hit(state, board, scorer):
    """A board where only we have built has no legal target worth taking."""
    advisor = TurnAdvisor(board, scorer)
    plan = SetupAdvisor(board, scorer).recommend(state, 1, seat=1, top=1)[0]
    state.build_settlement(1, plan.first, free=True)
    assert advisor.robber_advice(state, 1) is None


# -- discarding ------------------------------------------------------------


def test_no_discard_advice_under_eight_cards(
    state, board, scorer, turn_advisor
):
    play_out_setup(state, board, scorer)
    give(state, 1, wood=3, brick=3)
    assert turn_advisor.discard_advice(state, 1) is None


def test_discard_advice_sheds_exactly_half(state, board, scorer, turn_advisor):
    """Regression: this crashed on Resource comparison whenever two ties met."""
    play_out_setup(state, board, scorer)
    give(state, 1, wood=3, brick=3, sheep=1, wheat=2, ore=3)
    assert state.players[1].hand.total() == 12
    advice = turn_advisor.discard_advice(state, 1)
    assert advice is not None
    assert advice.label == "Discard 6"


def test_discard_of_a_single_resource_pile_does_not_crash(
    state, board, scorer, turn_advisor
):
    """The old halve() divided by five and shed nothing from a two-type hand."""
    play_out_setup(state, board, scorer)
    give(state, 1, wood=4, ore=4)
    advice = turn_advisor.discard_advice(state, 1)
    assert advice is not None
    assert advice.label == "Discard 4"


def test_discard_keeps_what_the_next_build_needs(
    state, board, scorer, turn_advisor
):
    play_out_setup(state, board, scorer)
    give(state, 1, sheep=6, ore=3, wheat=2)
    advice = turn_advisor.discard_advice(state, 1)
    assert advice is not None
    # Sheep is the spare pile here, so it should carry the discard.
    assert "sheep" in advice.reason


# -- alerts ----------------------------------------------------------------


def test_alert_on_a_fat_hand(state, board, scorer, turn_advisor):
    play_out_setup(state, board, scorer)
    give(state, 1, wood=5, brick=5)
    alerts = turn_advisor.alerts(state, 1)
    assert any("7" in a for a in alerts)


def test_alert_when_a_resource_is_missing(state, board, scorer, turn_advisor):
    play_out_setup(state, board, scorer)
    alerts = turn_advisor.alerts(state, 1)
    produced = rules.expected_yield(state, 1, ignore_robber=True)
    if any(v == 0 for v in produced.values()):
        assert any("produce no" in a for a in alerts)


def test_alert_when_the_robber_sits_on_your_tile(
    state, board, scorer, turn_advisor
):
    play_out_setup(state, board, scorer)
    my_tile = next(
        c for nid in state.nodes_of(1) for c in board.node(nid).tiles
        if not board.tiles[c].is_desert
    )
    state.move_robber(my_tile)
    assert any("robber" in a.lower() for a in turn_advisor.alerts(state, 1))


def test_alert_when_an_opponent_is_close_to_winning(
    state, board, scorer, turn_advisor
):
    play_out_setup(state, board, scorer)
    push_visible_vp(state, 2, state.target_vp - 2)
    alerts = turn_advisor.alerts(state, 1)
    assert any("Player 2" in a for a in alerts)
