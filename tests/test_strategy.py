"""
Does the advice agree with how Catan is actually played?

The other test files check that the engine does what it says. This one checks
that what it says is *right*: opening theory, robber targeting, the longest
road and largest army races, and whether a plan is reachable before the game
ends. Each test states the principle it encodes, so a disagreement is a
conversation about strategy rather than a mystery.
"""

import pytest

from catanmind.advisor import (
    SetupAdvisor,
    TurnAdvisor,
    income_per_round,
    phase_of,
    turns_to_afford,
    vp_value,
)
from catanmind.board import (
    Board,
    COSTS,
    Layout,
    NUMBER_TOKENS,
    Resource,
    SPIRAL,
    TILE_POOL,
    pips,
)
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
def state(board):
    return GameState(board, num_players=4, me=1)


def node_pips(board, node_id):
    return sum(board.tiles[c].pips for c in board.node(node_id).tiles)


def node_resources(board, node_id):
    return {
        board.tiles[c].resource for c in board.node(node_id).tiles
        if board.tiles[c].resource is not None
    }


def played_setup(state):
    """Run a legal setup so the board looks like a real game."""
    flow = TurnFlow(state)
    while flow.in_setup:
        node = next(
            n for n in rules.legal_settlements(state, flow.current, setup=True)
        )
        flow.place_setup_settlement(node)
        edge = next(
            e for e in state.board.node_edges[node] if e not in state.roads
        )
        flow.place_setup_road(edge)
    return flow


def empty_hands(state):
    for player in state.players.values():
        for resource in Resource:
            player.hand.cards[resource] = 0


def give(state, player, **resources):
    for name, count in resources.items():
        state.adjust(player, Resource(name), count)


# -- opening theory --------------------------------------------------------


def test_the_opening_pick_is_a_high_pip_spot(state, board, scorer):
    """Openings are won on production. The top pick must be near the best."""
    plan = SetupAdvisor(board, scorer).recommend(state, 1, seat=1, top=1)[0]
    best = max(range(len(board.nodes)), key=lambda n: node_pips(board, n))
    assert node_pips(board, plan.first) >= node_pips(board, best) - 2


def test_at_equal_pips_variety_wins(state, board, scorer):
    """
    Three resources on one spot beats two, all else equal — the standard
    opening heuristic, and the reason the pure-pip leader is not the pick.
    """
    advisor = SetupAdvisor(board, scorer)
    pick = advisor.recommend(state, 1, seat=1, top=1)[0].first
    pip_leader = max(range(len(board.nodes)), key=lambda n: node_pips(board, n))
    if node_pips(board, pick) == node_pips(board, pip_leader):
        assert len(node_resources(board, pick)) >= len(
            node_resources(board, pip_leader)
        )


def test_a_three_tile_spot_beats_a_two_tile_spot_of_equal_pips(
    state, board, scorer
):
    coastal = [n.id for n in board.nodes if len(n.tiles) < 3]
    inland = [n.id for n in board.nodes if len(n.tiles) == 3]
    pairs = [
        (c, i) for c in coastal for i in inland
        if node_pips(board, c) == node_pips(board, i)
    ]
    if not pairs:
        pytest.skip("this board has no equal-pip coastal/inland pair")
    for c, i in pairs[:5]:
        sea = scorer.score_spot(state, c, 1, portfolio={})
        land = scorer.score_spot(state, i, 1, portfolio={})
        assert land.total >= sea.total


def test_the_second_pick_complements_the_first(state, board, scorer):
    """The pair should cover more resources than doubling down would."""
    advisor = SetupAdvisor(board, scorer)
    first = advisor.recommend(state, 1, seat=1, top=1)[0].first
    state.build_settlement(1, first, free=True)
    second = advisor.recommend(state, 1, seat=1, top=1)[0].first
    combined = node_resources(board, first) | node_resources(board, second)
    assert len(combined) > len(node_resources(board, first))


def test_seat_position_changes_the_opening():
    """
    Seat 1 waits through six picks before choosing again; the last seat picks
    twice in a row. Those are different problems, so the answer should usually
    differ — measured across boards, since any single board may have one spot
    that is simply best from every seat.
    """
    import random

    varied = 0
    boards = 0
    for seed in range(20):
        local_board = Board(Layout.random(random.Random(seed)))
        advisor = SetupAdvisor(local_board, Scorer(local_board))
        picks = {
            seat: advisor.recommend(
                GameState(local_board, 4, me=1), 1, seat=seat, top=1
            )[0].first
            for seat in (1, 2, 3, 4)
        }
        boards += 1
        if len(set(picks.values())) > 1:
            varied += 1
    assert varied > boards // 2, (
        f"seat mattered on only {varied}/{boards} boards — the lookahead is "
        "not affecting the pick"
    )


def test_room_to_grow_never_outweighs_what_a_spot_produces():
    """
    Regression: expansion summed every node within three roads, which made it
    as large as production. A mediocre spot in an open corner outranked a much
    richer one because it could see futures it would never build.
    """
    import random

    for seed in range(15):
        local_board = Board(Layout.random(random.Random(seed)))
        local_scorer = Scorer(local_board)
        st = GameState(local_board, 4, me=1)
        for plan in SetupAdvisor(local_board, local_scorer).recommend(
            st, 1, seat=1, top=3
        ):
            detail = plan.detail
            if detail.production <= 0:
                continue
            assert detail.expansion < detail.production, (
                f"seed {seed} node {detail.node_id}: expansion "
                f"{detail.expansion:.1f} rivals production {detail.production:.1f}"
            )


def test_the_top_opening_is_not_a_visibly_weak_spot():
    """The best pick should not be far behind the runner-up on raw output."""
    import random

    offenders = 0
    for seed in range(25):
        local_board = Board(Layout.random(random.Random(seed)))
        local_scorer = Scorer(local_board)
        plans = SetupAdvisor(local_board, local_scorer).recommend(
            GameState(local_board, 4, me=1), 1, seat=1, top=2
        )
        if len(plans) < 2:
            continue
        if plans[0].detail.pips < plans[1].detail.pips - 2:
            offenders += 1
    assert offenders <= 3, f"{offenders}/25 boards led with a much weaker spot"


def test_the_projected_pair_is_actually_placeable(state, board, scorer):
    for plan in SetupAdvisor(board, scorer).recommend(state, 1, seat=1, top=3):
        if plan.projected_second is None:
            continue
        assert plan.projected_second not in board.node(plan.first).neighbors


# -- looking ahead ---------------------------------------------------------


def test_income_counts_every_players_roll(state, board):
    """
    You collect on everyone's roll, not just your own. Costing a plan against
    a single roll per round makes every goal look four times further away.
    """
    played_setup(state)
    per_roll = rules.expected_yield(state, 1)
    per_round = income_per_round(state, 1)
    for resource in Resource:
        assert per_round[resource] == pytest.approx(
            per_roll[resource] * state.num_players
        )


def test_something_you_can_already_pay_for_is_zero_turns_away(state):
    played_setup(state)
    empty_hands(state)
    give(state, 1, wood=1, brick=1)
    assert turns_to_afford(state, 1, COSTS["road"]) == 0.0


def test_a_goal_you_produce_for_is_closer_than_one_you_do_not(state, board):
    played_setup(state)
    empty_hands(state)
    income = income_per_round(state, 1)
    produced = [r for r in Resource if income[r] > 0.2]
    absent = [r for r in Resource if income[r] == 0]
    if not produced or not absent:
        pytest.skip("this setup produces everything or nothing")
    near = turns_to_afford(state, 1, {produced[0]: 2})
    far = turns_to_afford(state, 1, {absent[0]: 2})
    assert far > near


def test_a_cheap_reachable_move_outranks_a_rich_unreachable_one(
    state, board, scorer
):
    """
    The greedy engine chased whatever scored highest even when the player had
    no way to pay for it. Value has to be divided by time.
    """
    played_setup(state)
    empty_hands(state)
    advice = TurnAdvisor(board, scorer).recommend(state, 1, top=6)
    assert advice
    ranked = [a for a in advice if not a.affordable]
    if len(ranked) < 2:
        pytest.skip("nothing to compare on this board")
    # Whatever comes first must not be strictly worse on both counts.
    best = ranked[0]
    for other in ranked[1:]:
        assert best.rate >= other.rate


def test_affordable_moves_still_come_first(state, board, scorer):
    played_setup(state)
    empty_hands(state)
    give(state, 1, wood=1, brick=1)
    advice = TurnAdvisor(board, scorer).recommend(state, 1, top=8)
    affordable = [i for i, a in enumerate(advice) if a.affordable]
    blocked = [i for i, a in enumerate(advice) if not a.affordable]
    if affordable and blocked:
        assert max(affordable) < min(blocked)


def test_unreachable_advice_says_how_far_off_it_is(state, board, scorer):
    played_setup(state)
    empty_hands(state)
    for advice in TurnAdvisor(board, scorer).recommend(state, 1, top=6):
        if not advice.affordable and advice.turns >= 1.5:
            assert "round" in advice.reason or "reach" in advice.reason


# -- building priorities ---------------------------------------------------


def test_a_city_beats_a_road_when_you_are_ore_and_wheat_rich(
    state, board, scorer
):
    """Cities are the densest points in the game once the ore is flowing."""
    played_setup(state)
    empty_hands(state)
    give(state, 1, ore=3, wheat=2, wood=1, brick=1)
    advice = TurnAdvisor(board, scorer).recommend(state, 1, top=6)
    actions = [a.action for a in advice if a.affordable]
    if "upgrade_city" in actions and "build_road" in actions:
        assert actions.index("upgrade_city") < actions.index("build_road")


def test_a_settlement_is_offered_when_one_is_reachable(state, board, scorer):
    played_setup(state)
    empty_hands(state)
    give(state, 1, wood=2, brick=2, sheep=1, wheat=1)
    advice = TurnAdvisor(board, scorer).recommend(state, 1, top=8)
    assert any(a.action in ("build_settlement", "build_road") for a in advice)


def test_cities_are_ranked_by_what_they_actually_add(board, scorer):
    """
    Every city costs the same and is worth the same point, so their order must
    follow the production they add — nothing else may reorder them.
    """
    import random

    checked = 0
    for seed in range(12):
        rng = random.Random(seed)
        local_board = Board(Layout.random(rng))
        st = GameState(local_board, 4, me=1)
        local_scorer = Scorer(local_board)
        flow = TurnFlow(st)
        while flow.in_setup:
            node = next(
                (n for n in rules.legal_settlements(st, flow.current, setup=True)),
                None,
            )
            if node is None:
                break
            flow.place_setup_settlement(node)
            edge = next(
                e for e in local_board.node_edges[node] if e not in st.roads
            )
            flow.place_setup_road(edge)
        if flow.in_setup:
            continue
        flow.roll(9)

        portfolio = rules.expected_yield(st, 1, ignore_robber=True)
        cities = [
            a for a in TurnAdvisor(local_board, local_scorer).recommend(
                st, 1, top=20
            )
            if a.action == "upgrade_city"
        ]
        if len(cities) < 2:
            continue
        checked += 1
        gains = [
            local_scorer.marginal_utility(
                portfolio,
                {
                    r: v for r, v in
                    rules.node_yield(st, a.node, ignore_robber=True).items() if v
                },
            )
            for a in cities
        ]
        assert gains == sorted(gains, reverse=True), (
            f"seed {seed}: city order disagrees with the production they add"
        )
    assert checked, "no board produced two upgradeable settlements"


def test_every_recommended_build_is_legal(state, board, scorer):
    played_setup(state)
    give(state, 1, wood=4, brick=4, sheep=4, wheat=4, ore=4)
    for a in TurnAdvisor(board, scorer).recommend(state, 1, top=10):
        if a.action == "build_settlement":
            assert rules.can_place_settlement(state, 1, a.node)
        elif a.action == "build_road":
            assert rules.can_place_road(state, 1, a.edge)
        elif a.action == "upgrade_city":
            assert rules.can_upgrade_city(state, 1, a.node)


# -- the races -------------------------------------------------------------


def test_roads_get_more_valuable_as_longest_road_comes_into_reach(
    state, board, scorer
):
    played_setup(state)
    empty_hands(state)
    give(state, 1, wood=1, brick=1)
    advisor = TurnAdvisor(board, scorer)

    def best_road_value():
        return max(
            (a.value for a in advisor.recommend(state, 1, top=10)
             if a.action == "build_road"),
            default=0.0,
        )

    before = best_road_value()
    # Walk a chain out from an existing settlement.
    node = state.settlements_of(1)[0]
    for _ in range(4):
        nxt = next(
            (m for m in board.node(node).neighbors
             if board.edge_id[(node, m)] not in state.roads),
            None,
        )
        if nxt is None:
            break
        state.build_road(1, board.edge_id[(node, nxt)], free=True)
        node = nxt
    after = best_road_value()
    assert after > before


def test_development_cards_get_more_valuable_near_largest_army(
    state, board, scorer
):
    played_setup(state)
    empty_hands(state)
    give(state, 1, sheep=1, wheat=1, ore=1)
    advisor = TurnAdvisor(board, scorer)

    def dev_value():
        return max(
            (a.value for a in advisor.recommend(state, 1, top=10)
             if a.action == "buy_dev_card"),
            default=0.0,
        )

    before = dev_value()
    state.players[1].knights_played = 2   # one short of the army
    after = dev_value()
    assert after > before


def test_a_point_is_worth_more_when_the_game_is_nearly_over(state, board):
    played_setup(state)
    early = vp_value(state, 1)
    for node in list(state.settlements_of(2)):
        state.build_city(2, node, free=True)
    while (
        rules.victory_points(state, 2, include_hidden=False) < state.target_vp - 2
        and state.remaining(2, "settlement") > 0
    ):
        spot = next(
            (n for n in rules.legal_settlements(state, 2, setup=True)), None
        )
        if spot is None:
            break
        state.build_settlement(2, spot, free=True)
    assert vp_value(state, 1) > early


# -- the robber ------------------------------------------------------------


def test_the_robber_goes_after_the_leader(state, board, scorer):
    played_setup(state)
    for node in list(state.settlements_of(3)):
        state.build_city(3, node, free=True)
    state.players[3].knights_played = 3
    assert rules.victory_points(state, 3) > rules.victory_points(state, 2)

    advice = TurnAdvisor(board, scorer).robber_advice(state, 1)
    assert advice is not None
    owners = {
        state.buildings[n][0] for n in board.tiles[advice.coord].nodes
        if n in state.buildings
    }
    assert 3 in owners, "the runaway leader should be the one blocked"


def test_the_robber_never_blocks_your_own_production(state, board, scorer):
    played_setup(state)
    advice = TurnAdvisor(board, scorer).robber_advice(state, 1)
    assert advice is not None
    mine = {c for n in state.nodes_of(1) for c in board.node(n).tiles}
    assert advice.coord not in mine


def test_the_robber_prefers_a_productive_tile(state, board, scorer):
    played_setup(state)
    advice = TurnAdvisor(board, scorer).robber_advice(state, 1)
    tile = board.tiles[advice.coord]
    assert tile.resource is not None
    assert tile.pips >= 2, "blocking a 2 or 12 is close to wasting the move"


# -- discarding ------------------------------------------------------------


def test_a_discard_keeps_what_the_plan_needs(state, board, scorer):
    played_setup(state)
    empty_hands(state)
    give(state, 1, sheep=6, ore=3, wheat=2)
    advice = TurnAdvisor(board, scorer).discard_advice(state, 1)
    assert advice is not None
    assert "sheep" in advice.reason, "the spare pile should carry the discard"
