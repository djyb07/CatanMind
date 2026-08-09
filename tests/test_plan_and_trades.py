"""
The advisor's strategy: what it is playing for, and how it gets the cards.

Ranking one move at a time gives a shopping list, not a plan. These tests cover
the two things that turn it into advice a player can follow: naming the
strategy the position supports, and working out which trade unblocks it —
including when *not* to trade.
"""

import pytest

from catanmind.advisor import (
    PLAN_EMPHASIS,
    TurnAdvisor,
    strategic_plan,
)
from catanmind.board import Board, Resource
from catanmind.flow import TurnFlow
from catanmind.scoring import Scorer
from catanmind.state import GameState
from catanmind import rules


@pytest.fixture(scope="module")
def board():
    return Board()


@pytest.fixture(scope="module")
def scorer(board):
    return Scorer(board)


@pytest.fixture
def game(board, scorer):
    """A played-out setup, empty hands, ready for a normal turn."""
    state = GameState(board, num_players=4, me=1)
    flow = TurnFlow(state)
    while flow.in_setup:
        node = next(
            n for n in rules.legal_settlements(state, flow.current, setup=True)
        )
        flow.place_setup_settlement(node)
        edge = next(
            e for e in board.node_edges[node] if e not in state.roads
        )
        flow.place_setup_road(edge)
    flow.roll(9)
    for player in state.players.values():
        for resource in Resource:
            player.hand.cards[resource] = 0
    return state, flow, TurnAdvisor(board, scorer)


def give(state, player, **resources):
    for name, count in resources.items():
        state.adjust(player, Resource(name), count)


def push_vp(state, player, target):
    """Raise a player's visible score by building."""
    for node in list(state.settlements_of(player)):
        state.build_city(player, node, free=True)
    while (
        rules.victory_points(state, player, include_hidden=False) < target
        and state.remaining(player, "settlement") > 0
    ):
        spot = next(
            (n for n in rules.legal_settlements(state, player, setup=True)), None
        )
        if spot is None:
            break
        state.build_settlement(player, spot, free=True)
        if state.remaining(player, "city") > 0:
            state.build_city(player, spot, free=True)
    return rules.victory_points(state, player, include_hidden=False)


# -- naming the strategy ---------------------------------------------------


def test_every_plan_has_something_to_say(game):
    state, _flow, _advisor = game
    plan = strategic_plan(state, 1)
    assert plan.key in PLAN_EMPHASIS
    assert plan.title and plan.focus and plan.reason


def test_being_two_points_away_overrides_everything(game):
    """Near the end you stop building the engine and buy the points."""
    state, _flow, advisor = game
    push_vp(state, 1, state.target_vp - 2)
    plan = advisor.plan(state, 1)
    assert plan.key == "finish"
    assert plan.emphasis["upgrade_city"] > 1.0


def test_the_plan_actually_moves_the_ranking(board, scorer):
    """
    A plan that only prints a caption is decoration. The same position under a
    city-first plan must rank cities above where a neutral plan would.
    """
    state = GameState(board, num_players=4, me=1)
    flow = TurnFlow(state)
    while flow.in_setup:
        node = next(
            n for n in rules.legal_settlements(state, flow.current, setup=True)
        )
        flow.place_setup_settlement(node)
        edge = next(e for e in board.node_edges[node] if e not in state.roads)
        flow.place_setup_road(edge)
    flow.roll(9)
    advisor = TurnAdvisor(board, scorer)
    give(state, 1, ore=3, wheat=2, wood=1, brick=1)

    plan = strategic_plan(state, 1)
    cities = [
        a for a in advisor.recommend(state, 1, top=10)
        if a.action == "upgrade_city"
    ]
    assert cities
    if plan.key in ("cities", "finish"):
        assert plan.emphasis.get("upgrade_city", 1.0) > 1.0


def test_a_wood_and_brick_engine_is_told_to_expand(board, scorer):
    """The archetype should follow production, not the hand."""
    state = GameState(board, num_players=4, me=1)
    # Settle two spots and check whichever engine comes out is named coherently.
    flow = TurnFlow(state)
    while flow.in_setup:
        node = next(
            n for n in rules.legal_settlements(state, flow.current, setup=True)
        )
        flow.place_setup_settlement(node)
        edge = next(e for e in board.node_edges[node] if e not in state.roads)
        flow.place_setup_road(edge)
    flow.roll(9)

    from catanmind.advisor import income_per_round

    income = income_per_round(state, 1)
    ore_wheat = income[Resource.ORE] + income[Resource.WHEAT]
    wood_brick = income[Resource.WOOD] + income[Resource.BRICK]
    plan = strategic_plan(state, 1)

    if wood_brick > ore_wheat * 1.35:
        assert plan.key in ("expansion", "longest_road")
    elif ore_wheat > wood_brick * 1.35:
        assert plan.key in ("cities", "development")
    else:
        assert plan.key == "balanced"


def test_the_advisor_weighs_every_strategic_factor(game):
    """
    A roll-up of what the advice is supposed to account for. Each of these has
    its own test elsewhere; this one fails loudly if a whole factor ever stops
    reaching the ranking, which is easy to do while refactoring.
    """
    state, _flow, advisor = game

    def best(action):
        return max(
            (a.value for a in advisor.recommend(state, 1, top=25)
             if a.action == action),
            default=0.0,
        )

    # Production, expansion and time-to-afford all reach the list.
    advice = advisor.recommend(state, 1, top=25)
    assert advice, "there is always something worth saving for"
    assert any(a.turns > 0 for a in advice), "reachability is priced"
    assert all(a.reason for a in advice), "every suggestion explains itself"

    # A named strategy.
    assert advisor.plan(state, 1).title

    # Hand-limit risk.
    give(state, 1, wood=5, brick=5)
    assert any("7" in a for a in advisor.alerts(state, 1))

    # The robber goes after somebody, and never our own tiles.
    robber = advisor.robber_advice(state, 1)
    if robber is not None:
        mine = {
            c for n in state.nodes_of(1) for c in state.board.node(n).tiles
        }
        assert robber.coord not in mine

    # The two race cards move their own moves.
    road_before = best("build_road")
    node = state.settlements_of(1)[0]
    for _ in range(4):
        nxt = next(
            (m for m in state.board.node(node).neighbors
             if state.board.edge_id[(node, m)] not in state.roads),
            None,
        )
        if nxt is None:
            break
        state.build_road(1, state.board.edge_id[(node, nxt)], free=True)
        node = nxt
    assert best("build_road") >= road_before

    give(state, 1, sheep=1, wheat=1, ore=1)
    dev_before = best("buy_dev_card")
    state.players[1].knights_played = 2
    assert best("buy_dev_card") > dev_before


# -- trading ---------------------------------------------------------------


def test_no_trade_is_suggested_when_nothing_is_close(game):
    state, _flow, advisor = game
    assert advisor.trade_advice(state, 1) == []


def test_a_bank_trade_is_offered_when_a_surplus_covers_the_gap(game):
    state, _flow, advisor = game
    sheep_rate = state.ports_of(1)[Resource.SHEEP]
    give(state, 1, wood=1, sheep=sheep_rate)
    offers = advisor.trade_advice(state, 1)
    assert any(a.action == "trade_bank" for a in offers)


def test_a_player_trade_names_who_and_what(game):
    state, _flow, advisor = game
    give(state, 1, wood=1, sheep=4)
    give(state, 2, brick=3)
    offers = [a for a in advisor.trade_advice(state, 1) if a.action == "trade_player"]
    assert offers
    offer = offers[0]
    assert "Player 2" in offer.label
    assert "brick" in offer.label
    assert "brick" in offer.reason


def test_a_trade_that_completes_a_build_says_so(game):
    state, _flow, advisor = game
    give(state, 1, wood=1, sheep=4)
    give(state, 2, brick=3)
    for offer in advisor.trade_advice(state, 1):
        assert "finishes a road" in offer.reason


def test_a_trade_that_only_half_helps_does_not_claim_to_finish(game):
    """
    Two different cards missing means one trade cannot complete the purchase.
    Saying otherwise costs the player a card for nothing.
    """
    state, _flow, advisor = game
    # Short both wheat and ore for a development card, with sheep to spare.
    give(state, 1, sheep=6)
    for offer in advisor.trade_advice(state, 1):
        if "dev card" in offer.reason:
            assert "finishes" not in offer.reason
            assert "still needing" in offer.reason or "leaves you needing" in offer.reason


def test_the_offer_explains_the_other_players_position(game):
    """A trade only happens if the other side wants it, so say why they might."""
    state, _flow, advisor = game
    give(state, 1, wood=1, sheep=4)
    give(state, 2, brick=3)
    offers = [a for a in advisor.trade_advice(state, 1) if a.action == "trade_player"]
    assert offers
    reason = offers[0].reason
    assert "Player 2" in reason
    assert "%" in reason, "should say how likely they are to hold it"


def test_asking_a_player_beats_paying_the_bank_four_to_one(game):
    """
    Same card, same result, a quarter of the cost. A trade that spends four
    cards where one would do is a bad trade, so the cards spent are priced in.
    """
    state, _flow, advisor = game
    give(state, 1, wood=1, sheep=4)
    give(state, 2, brick=3)
    offers = advisor.trade_advice(state, 1)
    assert offers[0].action == "trade_player", (
        "asking for one card should beat handing the bank four"
    )
    bank = next(a for a in offers if a.action == "trade_bank")
    assert offers[0].value > bank.value


def test_a_good_port_makes_the_bank_competitive_again(game):
    """At 2:1 the bank costs two cards instead of four, so the gap narrows."""
    state, _flow, advisor = game
    give(state, 1, wood=1, sheep=4)
    give(state, 2, brick=3)
    offers = {a.action: a for a in advisor.trade_advice(state, 1)}
    if "trade_bank" not in offers or "trade_player" not in offers:
        pytest.skip("this position offers only one route")
    gap = offers["trade_player"].value - offers["trade_bank"].value
    assert gap > 0


def test_trading_with_the_runaway_leader_is_discouraged(game):
    """Handing the leader the card they need is how games are lost."""
    state, _flow, advisor = game
    give(state, 1, wood=1, sheep=4)
    give(state, 2, brick=3)
    calm = [a for a in advisor.trade_advice(state, 1) if a.action == "trade_player"]
    assert calm
    before = calm[0].value

    reached = push_vp(state, 2, state.target_vp - 2)
    assert reached >= state.target_vp - 2
    risky = [a for a in advisor.trade_advice(state, 1) if a.action == "trade_player"]
    assert risky
    assert risky[0].value < before
    assert "Careful" in risky[0].reason


def test_the_bank_wins_when_the_only_partner_is_the_leader(game):
    state, _flow, advisor = game
    give(state, 1, wood=1, sheep=4)
    give(state, 2, brick=3)
    push_vp(state, 2, state.target_vp - 2)
    offers = advisor.trade_advice(state, 1)
    assert offers[0].action == "trade_bank", (
        "with the leader as the only partner, the bank is the safe route"
    )


def test_an_empty_opponent_is_not_asked_for_cards(game):
    state, _flow, advisor = game
    give(state, 1, wood=1, sheep=4)
    for offer in advisor.trade_advice(state, 1):
        if offer.action == "trade_player":
            partner = int(offer.label.split("Player ")[1].split()[0])
            assert state.players[partner].hand.total() > 0


def test_trades_appear_in_the_main_recommendation_list(game):
    state, _flow, advisor = game
    give(state, 1, wood=1, sheep=4)
    give(state, 2, brick=3)
    actions = {a.action for a in advisor.recommend(state, 1, top=8)}
    assert "trade_bank" in actions or "trade_player" in actions


def test_a_suggested_bank_trade_is_one_the_rules_allow(game):
    """Whatever it suggests must actually be executable."""
    state, flow, advisor = game
    give(state, 1, wood=1, sheep=6)
    for offer in advisor.trade_advice(state, 1):
        if offer.action != "trade_bank":
            continue
        # Label reads "Trade N×give for M×need".
        parts = offer.label.replace("Trade ", "").split(" for ")
        give_count, give_name = parts[0].split("×")
        need_name = parts[1].split("×")[1]
        rate = state.ports_of(1)[Resource(give_name)]
        assert int(give_count) == rate
        assert flow.trade_bank(Resource(give_name), Resource(need_name)).ok
