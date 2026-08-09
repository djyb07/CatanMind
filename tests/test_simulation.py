"""
Whole games, played through the public interface.

The other suites test pieces. This one plays: it takes only the moves
:meth:`TurnFlow.available_actions` offers, on randomly generated boards, and
checks the invariants that must hold at every single step of every game.

If the turn machine can ever offer a move that then fails, or let a player
build with an empty hand, or lose track of whose turn it is, this is what
catches it.
"""

import random

import pytest

from catanmind.board import Board, Building, DevCard, Layout, Resource, SUPPLY
from catanmind.flow import Step, TurnFlow
from catanmind.scoring import Scorer
from catanmind.state import GameState
from catanmind.advisor import SetupAdvisor, TurnAdvisor
from catanmind import rules


def check_invariants(state: GameState, flow: TurnFlow) -> None:
    """Things that must be true after every single action, in every game."""
    # Nobody exceeds their supply of pieces.
    for player in state.players:
        assert len(state.settlements_of(player)) <= SUPPLY["settlement"]
        assert len(state.cities_of(player)) <= SUPPLY["city"]
        assert len(state.edges_of(player)) <= SUPPLY["road"]

    # No hand is ever negative.
    for p in state.players.values():
        assert all(n >= 0 for n in p.hand.cards.values())
        assert p.unknown_dev >= 0
        assert all(n >= 0 for n in p.dev_cards.values())

    # The distance rule holds across the whole board.
    for node_id in state.buildings:
        for neighbour in state.board.node(node_id).neighbors:
            assert neighbour not in state.buildings, (
                f"#{node_id} and #{neighbour} are adjacent settlements"
            )

    # Turn bookkeeping stays in range.
    assert 1 <= flow.current <= state.num_players
    assert flow.free_roads >= 0

    # The robber is always on a real tile.
    assert flow.state.robber in state.board.tiles

    # More development cards can never be held than the deck contains.
    drawn = sum(p.dev_cards_held for p in state.players.values())
    assert drawn <= 25


def play_game(seed: int, max_actions: int = 900) -> dict:
    """Play one game with a random-but-legal policy. Returns a small summary."""
    rng = random.Random(seed)
    board = Board(Layout.random(rng))
    state = GameState(board, num_players=4, me=1)
    flow = TurnFlow(state)
    scorer = Scorer(board)
    setup_advisor = SetupAdvisor(board, scorer)
    turn_advisor = TurnAdvisor(board, scorer)

    steps = 0
    for _ in range(max_actions):
        if flow.step is Step.OVER:
            break
        actions = [a for a in flow.available_actions() if a.enabled]
        assert actions, f"no legal action at {flow.step} (seed {seed})"
        action = rng.choice(actions)
        steps += 1

        result = None
        if action.id == "setup_settlement":
            spots = rules.legal_settlements(state, flow.current, setup=True)
            if not spots:
                break
            result = flow.place_setup_settlement(rng.choice(spots))
        elif action.id == "setup_road":
            node = flow.setup_settlement_node()
            options = [
                e for e in board.node_edges[node] if e not in state.roads
            ]
            result = flow.place_setup_road(rng.choice(options))
        elif action.id == "roll":
            result = flow.roll(rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
        elif action.id == "discard":
            player = flow.pending_discards()[0]
            owed = rules.must_discard(state, player)
            hand = state.players[player].hand
            picks = []
            for resource in Resource:
                while hand.cards[resource] > len(
                    [p for p in picks if p is resource]
                ) and len(picks) < owed:
                    picks.append(resource)
            result = flow.discard(player, picks[:owed])
        elif action.id == "move_robber":
            targets = [c for c in board.tiles if c != state.robber]
            result = flow.move_robber(rng.choice(targets))
        elif action.id == "steal":
            result = flow.steal(rng.choice(flow.steal_victims()))
        elif action.id == "skip_steal":
            result = flow.skip_steal()
        elif action.id == "free_road":
            options = rules.legal_roads(state, flow.current)
            if not options:
                break
            result = flow.place_free_road(rng.choice(options))
        elif action.id == "build_road":
            result = flow.build("road", rng.choice(
                rules.legal_roads(state, flow.current)))
        elif action.id == "build_settlement":
            result = flow.build("settlement", rng.choice(
                rules.legal_settlements(state, flow.current)))
        elif action.id == "city":
            result = flow.build("city", rng.choice(
                rules.legal_cities(state, flow.current)))
        elif action.id == "buy_dev":
            result = flow.buy_dev(rng.choice(list(DevCard)))
        elif action.id.startswith("play_dev:"):
            card = DevCard(action.id.split(":", 1)[1])
            if card is DevCard.MONOPOLY:
                result = flow.play_dev(card, resource=rng.choice(list(Resource)))
            elif card is DevCard.YEAR_OF_PLENTY:
                result = flow.play_dev(
                    card, cards=[rng.choice(list(Resource)) for _ in range(2)]
                )
            else:
                result = flow.play_dev(card)
        elif action.id == "reveal_vp":
            result = flow.reveal_vp()
        elif action.id == "trade":
            rates = state.ports_of(flow.current)
            hand = state.players[flow.current].hand
            options = [r for r in Resource if hand.cards[r] >= rates[r]]
            if not options:
                continue
            give = rng.choice(options)
            get = rng.choice([r for r in Resource if r is not give])
            result = flow.trade_bank(give, get)
        elif action.id == "end_turn":
            result = flow.end_turn()
        elif action.id == "new_game":
            break
        else:
            continue

        assert result is None or result.ok, (
            f"seed {seed}: the screen offered {action.id!r} at {flow.step} "
            f"but it failed: {result.reason}"
        )

        check_invariants(state, flow)

        # The advisor must survive every position it can be asked about.
        if flow.is_my_turn() and steps % 11 == 0:
            if flow.in_setup:
                setup_advisor.recommend(state, 1, seat=1, top=3)
            else:
                turn_advisor.recommend(state, 1, top=5)
                turn_advisor.alerts(state, 1)

    return {
        "steps": steps,
        "scores": rules.scores(state),
        "phase": flow.step,
        "log": len(state.log),
    }


@pytest.mark.parametrize("seed", range(8))
def test_a_whole_game_plays_without_an_illegal_move(seed):
    summary = play_game(seed)
    assert summary["steps"] > 40, "the game stalled almost immediately"
    assert summary["log"] > 40


def test_games_reach_a_real_position():
    """Across several games, someone should be building and scoring."""
    best = 0
    for seed in range(6):
        summary = play_game(seed)
        best = max(best, max(summary["scores"].values()))
    assert best >= 3, f"nobody got past {best} points in any game"


def test_undo_from_any_point_in_a_game_is_consistent():
    """Rewind a played game move by move; the flow must stay derivable."""
    rng = random.Random(3)
    board = Board(Layout.random(rng))
    state = GameState(board, num_players=4, me=1)
    flow = TurnFlow(state)

    while flow.in_setup:
        node = next(
            n for n in rules.legal_settlements(state, flow.current, setup=True)
        )
        flow.place_setup_settlement(node)
        edge = next(e for e in board.node_edges[node] if e not in state.roads)
        flow.place_setup_road(edge)
    for _ in range(6):
        flow.roll(8)
        flow.end_turn()

    while flow.undo():
        check_invariants(state, flow)
        assert flow.available_actions(), "a rewound game must still be playable"
    assert state.log == []
    assert flow.step is Step.SETUP_SETTLEMENT
    assert flow.current == 1
