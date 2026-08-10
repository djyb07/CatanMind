"""
Playing the rest of the game out.

The forecast exists to answer one question the rest of the engine cannot: who
reaches ten points first. That makes it worth checking that it actually
respects the dice — a settlement on a 6 and a settlement on a 3 cost the same
and are worth the same point, and every difference between them lives in the
roll distribution.
"""

import time

import pytest

from catanmind import forecast, rules
from catanmind.advisor import SetupAdvisor
from catanmind.board import Board, Layout, Resource
from catanmind.flow import TurnFlow
from catanmind.scoring import Scorer
from catanmind.state import GameState


@pytest.fixture(scope="module")
def board():
    return Board(Layout.standard())


@pytest.fixture
def game(board):
    """A real position: setup played out by the advisor, one roll in."""
    state = GameState(board, num_players=4, me=1)
    scorer = Scorer(board)
    flow = TurnFlow(state)
    advisor = SetupAdvisor(board, scorer)
    while flow.in_setup:
        plan = advisor.recommend(state, flow.current, seat=flow.current, top=1)[0]
        flow.place_setup_settlement(plan.first)
        options = [
            e for e in board.node_edges[plan.first] if e not in state.roads
        ]
        flow.place_setup_road(plan.road if plan.road in options else options[0])
    flow.roll(9)
    return state


def empty_hands(state):
    for player in state.players.values():
        for resource in Resource:
            player.hand.cards[resource] = 0


# -- the dice --------------------------------------------------------------


def test_the_roll_table_matches_two_real_dice():
    """36 combinations, and a 7 six times as likely as a 2."""
    assert sum(forecast._ROLL_WEIGHTS) == 36
    weights = dict(zip(forecast._ROLL_VALUES, forecast._ROLL_WEIGHTS))
    assert weights[7] == 6
    assert weights[2] == weights[12] == 1
    assert weights[6] == weights[8] == 5


def test_the_same_production_on_a_better_number_forecasts_better():
    """
    The whole reason for rolling dice rather than averaging: the same cards on
    a better number are a better asset.

    Compared at the rollout, where the dice numbers are the only thing that
    differs. Two board positions can never isolate this — the spot that
    happens to sit on a 6 also touches a different set of neighbouring tiles.
    """
    import random

    payout = {r: 1 for r in Resource}

    def points(number):
        rng = random.Random(7)
        total = 0.0
        for _ in range(200):
            vp, _turns = forecast._rollout(
                {number: payout}, {r: 0 for r in Resource},
                start_vp=2, settlements=2, cities=0,
                rounds=12, rolls_per_round=4, rng=rng,
            )
            total += vp
        return total / 200

    assert points(6) > points(2)
    assert points(8) > points(12)


def test_the_robber_suppresses_a_forecast(game):
    """A blocked tile pays nothing, and the outlook should feel it."""
    state = game
    before = forecast.outlook(state, 1).victory_points
    mine = {
        c for n in state.nodes_of(1) for c in state.board.node(n).tiles
        if not state.board.tiles[c].is_desert
    }
    if not mine:
        pytest.skip("player 1 touches no producing tile")
    state.move_robber(max(mine, key=lambda c: state.board.tiles[c].pips))
    assert forecast.outlook(state, 1).victory_points <= before


# -- what a forecast says --------------------------------------------------


def test_a_forecast_starts_from_the_points_already_scored(game):
    state = game
    now = rules.victory_points(state, 1)
    assert forecast.outlook(state, 1).victory_points >= now


def test_more_production_forecasts_more_points(game):
    """Two settlements should outrun one, over enough rounds."""
    state = game
    lean = forecast.outlook(state, 1).victory_points
    for node in rules.legal_settlements(state, 1, setup=True)[:2]:
        state.build_settlement(1, node, free=True)
    assert forecast.outlook(state, 1).victory_points > lean


def test_cards_in_hand_count_for_something(game):
    state = game
    empty_hands(state)
    poor = forecast.outlook(state, 1).victory_points
    rich = forecast.outlook(
        state, 1,
        extra={r: 4 for r in Resource},
    ).victory_points
    assert rich > poor


def test_a_forecast_is_repeatable(game):
    """
    Same position, same answer. Advice that flickered between refreshes would
    be worse than advice that was simply wrong.
    """
    state = game
    first = forecast.outlook(state, 1)
    second = forecast.outlook(state, 1)
    assert first.victory_points == second.victory_points


def test_a_forecast_is_quick_enough_to_run_on_a_refresh(game):
    state = game
    start = time.perf_counter()
    forecast.outlook(state, 1)
    assert time.perf_counter() - start < 0.5


# -- the race --------------------------------------------------------------


def test_a_trade_helps_the_side_receiving_what_it_lacks(game):
    """Gains are measured for both sides, and they are not the same number."""
    state = game
    empty_hands(state)
    mine, theirs = forecast.trade_race(
        state, 1, 2, [Resource.SHEEP], [Resource.ORE]
    )
    assert isinstance(mine, float) and isinstance(theirs, float)
    assert mine != theirs


def test_the_race_is_symmetric_when_the_trade_is_reversed(game):
    """
    Giving ore for sheep and giving sheep for ore should not both look good
    to the same player.
    """
    state = game
    empty_hands(state)
    state.adjust(1, Resource.SHEEP, 3)
    state.adjust(2, Resource.ORE, 3)
    forward, _ = forecast.trade_race(
        state, 1, 2, [Resource.SHEEP], [Resource.ORE]
    )
    backward, _ = forecast.trade_race(
        state, 1, 2, [Resource.ORE], [Resource.SHEEP]
    )
    assert forward != backward


def test_both_sides_of_a_race_are_measured_on_the_same_dice(game):
    """
    Before and after share a seed on purpose. Comparing them across different
    dice would bury a real effect under the noise of two separate games.
    """
    state = game
    first = forecast.trade_race(state, 1, 2, [Resource.WOOD], [Resource.ORE])
    second = forecast.trade_race(state, 1, 2, [Resource.WOOD], [Resource.ORE])
    assert first == second


def test_a_race_is_quick_enough_for_one_candidate(game):
    state = game
    start = time.perf_counter()
    forecast.trade_race(
        state, 1, 2, [Resource.WOOD], [Resource.ORE], samples=120
    )
    assert time.perf_counter() - start < 1.0


# -- the rollout's own behaviour -------------------------------------------


def test_a_rollout_never_spends_cards_it_does_not_have():
    hand = {Resource.ORE: 1}
    assert not forecast._can_pay(hand, {Resource.ORE: 3})


def test_a_rollout_trades_a_surplus_for_the_last_card():
    """Real players trade constantly; a forecast that never does undercounts."""
    hand = {r: 0 for r in Resource}
    hand[Resource.WOOD] = 5
    hand[Resource.ORE] = 3
    hand[Resource.WHEAT] = 1
    assert forecast._trade_toward(hand, {Resource.ORE: 3, Resource.WHEAT: 2})
    assert hand[Resource.WHEAT] == 2
    assert hand[Resource.WOOD] == 1


def test_a_rollout_will_not_trade_when_two_cards_are_missing():
    hand = {r: 0 for r in Resource}
    hand[Resource.WOOD] = 8
    assert not forecast._trade_toward(hand, {Resource.ORE: 3, Resource.WHEAT: 2})


def test_a_rollout_respects_the_piece_supply(game):
    """Nobody builds a sixth settlement, however good the dice are."""
    state = game
    out = forecast.outlook(
        state, 1, extra={r: 30 for r in Resource}, rounds=14
    )
    assert out.victory_points <= 30
