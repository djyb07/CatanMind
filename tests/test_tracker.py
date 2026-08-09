"""Reading the opponents: hand estimates, steal targeting, threats."""

import pytest

from catanmind.board import Board, Resource, RESOURCES
from catanmind.state import GameState
from catanmind.tracker import Tracker
from catanmind import rules


@pytest.fixture(scope="module")
def board():
    return Board()


@pytest.fixture
def state(board):
    return GameState(board, num_players=4, me=1)


@pytest.fixture
def tracker(state):
    return Tracker(state)


def give(state, player, **resources):
    for name, count in resources.items():
        state.adjust(player, Resource(name), count)


# -- the estimate itself ---------------------------------------------------


def test_an_empty_hand_reads_as_empty(state, tracker):
    est = tracker.estimate(2)
    assert est.total == 0
    assert est.likeliest() is None
    assert est.chance_of(Resource.WOOD) == 0.0
    assert est.describe() == "empty"


def test_watched_cards_are_fully_known(state, tracker):
    give(state, 2, wood=3, ore=1)
    est = tracker.estimate(2)
    assert est.total == 4
    assert est.unknown == 0
    assert est.confidence() == 1.0
    assert est.known[Resource.WOOD] == 3
    assert est.likeliest() is Resource.WOOD


def test_chances_across_a_known_hand_sum_to_one(state, tracker):
    give(state, 2, wood=3, ore=1)
    total = sum(tracker.estimate(2).chance_of(r) for r in RESOURCES)
    assert total == pytest.approx(1.0)


def test_a_blind_steal_is_recorded_as_uncertainty(state, tracker):
    """
    A stolen card of unknown type must not be reported as a fact. The old
    tracker claimed exact knowledge and drifted; this one downgrades confidence.
    """
    give(state, 2, wood=4)
    give(state, 3, ore=4)
    state.steal(thief=2, victim=3)  # type not observed
    est = tracker.estimate(2)
    assert est.total == 5
    assert est.unknown == 1
    assert est.confidence() < 1.0
    assert "unknown" in est.describe()


def test_a_seen_steal_stays_certain(state, tracker):
    give(state, 2, wood=4)
    give(state, 3, ore=4)
    state.steal(thief=2, victim=3, resource=Resource.ORE)
    est = tracker.estimate(2)
    assert est.unknown == 0
    assert est.confidence() == 1.0
    assert est.known[Resource.ORE] == 1


def test_unknown_cards_spread_across_the_resources(state, tracker):
    give(state, 2, wood=4)
    give(state, 3, ore=4)
    state.steal(thief=2, victim=3)
    est = tracker.estimate(2)
    # The unknown card gives every resource some chance, not just wood.
    assert est.chance_of(Resource.ORE) > 0
    assert sum(est.chance_of(r) for r in RESOURCES) == pytest.approx(1.0)


def test_uncertainty_never_exceeds_the_hand(state, tracker):
    give(state, 2, wood=1)
    give(state, 3, ore=5)
    for _ in range(4):
        state.steal(thief=2, victim=3)
    est = tracker.estimate(2)
    assert est.unknown <= est.total
    assert 0.0 <= est.confidence() <= 1.0


def test_estimates_follow_undo(state, tracker):
    give(state, 2, wood=3)
    before = tracker.estimate(2).total
    give(state, 2, wood=2)
    assert tracker.estimate(2).total == before + 2
    state.undo()
    assert tracker.estimate(2).total == before


def test_all_estimates_can_skip_a_player(state, tracker):
    ids = [e.player for e in tracker.all_estimates(exclude=1)]
    assert ids == [2, 3, 4]


# -- who to rob ------------------------------------------------------------


def test_never_recommends_robbing_yourself(state, tracker):
    """The old tracker returned player 1 with a wood card when nobody had any."""
    give(state, 1, wood=5)
    assert tracker.steal_target(1) is None


def test_steal_target_is_none_when_every_opponent_is_empty(state, tracker):
    assert tracker.steal_target(1) is None


def test_steal_prefers_the_fuller_hand(state, tracker):
    give(state, 2, wood=1)
    give(state, 3, ore=6)
    target = tracker.steal_target(1)
    assert target is not None
    player, resource, reason = target
    assert player == 3
    assert resource is Resource.ORE
    assert "Player 3" in reason


def test_steal_can_be_aimed_at_a_wanted_resource(state, tracker):
    give(state, 2, brick=3)
    give(state, 3, ore=4)
    player, resource, _ = tracker.steal_target(1, want=Resource.BRICK)
    assert player == 2
    assert resource is Resource.BRICK


def test_steal_skips_empty_opponents(state, tracker):
    give(state, 4, sheep=2)
    player, _, _ = tracker.steal_target(1)
    assert player == 4


def test_steal_reason_flags_low_confidence(state, tracker):
    give(state, 2, wood=1)
    give(state, 3, ore=1)
    for _ in range(3):
        state.steal(thief=2, victim=3)
    target = tracker.steal_target(1)
    assert target is not None
    _, _, reason = target
    assert "confidence" in reason.lower() or "%" in reason


# -- threats ---------------------------------------------------------------


def test_no_threats_from_empty_hands(state, tracker):
    assert tracker.threats(1) == []


def test_threat_when_an_opponent_can_afford_a_city(state, tracker):
    give(state, 2, ore=3, wheat=2)
    threats = tracker.threats(1)
    assert any("city" in t for t in threats)
    assert any("Player 2" in t for t in threats)


def test_threat_when_an_opponent_is_holding_too_many_cards(state, tracker):
    give(state, 3, sheep=9)
    assert any("9 cards" in t for t in tracker.threats(1))


def test_threats_ignore_your_own_hand(state, tracker):
    give(state, 1, ore=3, wheat=2)
    assert tracker.threats(1) == []


# -- production forecast ---------------------------------------------------


def test_forecast_is_empty_before_anything_is_built(state, tracker):
    assert tracker.production_forecast(2) == {}


def test_forecast_reports_what_a_number_pays(state, board, tracker):
    node = next(
        n.id for n in board.nodes
        if any(board.tiles[c].number == 6 for c in n.tiles)
    )
    state.build_settlement(2, node, free=True)
    forecast = tracker.production_forecast(2)
    assert 6 in forecast
    assert sum(forecast[6].values()) >= 1


def test_forecast_counts_cities_double(state, board, tracker):
    node = next(
        n.id for n in board.nodes
        if any(board.tiles[c].number == 6 for c in n.tiles)
    )
    state.build_settlement(2, node, free=True)
    as_settlement = sum(tracker.production_forecast(2)[6].values())
    state.build_city(2, node, free=True)
    as_city = sum(tracker.production_forecast(2)[6].values())
    assert as_city == as_settlement * 2


def test_forecast_never_reports_a_seven(state, board, tracker):
    """The desert carries no token, so no roll should ever pay it out."""
    for node in (0, 10, 20):
        if rules.can_place_settlement(state, 2, node, setup=True):
            state.build_settlement(2, node, free=True)
    assert 7 not in tracker.production_forecast(2)


def test_forecast_respects_the_robber(state, board, tracker):
    node = next(
        n.id for n in board.nodes
        if any(board.tiles[c].number == 6 for c in n.tiles)
    )
    state.build_settlement(2, node, free=True)
    blocked = next(c for c in board.node(node).tiles if board.tiles[c].number == 6)
    before = sum(tracker.production_forecast(2).get(6, {}).values())
    state.move_robber(blocked)
    after = sum(tracker.production_forecast(2).get(6, {}).values())
    assert after < before
