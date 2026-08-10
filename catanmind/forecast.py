"""
How the rest of the game is likely to go.

Everything else in the engine scores a move by what it is worth *now*. That is
enough to rank builds, but it cannot answer the question a trade really poses:
if I hand this card over, do I reach ten points before they do?

The answer needs the dice. A settlement on a 6 and a settlement on a 3 have the
same cost and the same victory point, and are not remotely the same asset — the
difference only shows up when you play the rounds out. So this module plays
them out: it rolls two dice the way two dice actually behave, collects, buys
whatever the position most wants, and reports where each player has got to.

It is a *forecast*, not a solver. It assumes everyone keeps building along the
route their board already supports, which is what people mostly do and what
makes the numbers stable enough to compare. Where it approximates, it says so.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from catanmind.board import (
    Building,
    COSTS,
    DevCard,
    DEV_DECK,
    DEV_DECK_SIZE,
    DICE_WAYS,
    Resource,
    SUPPLY,
)
from catanmind.state import GameState
from catanmind import rules

#: Rounds to look ahead. Far enough that a production advantage compounds into
#: a visible lead, short enough that the board would not have changed shape
#: beyond recognition.
HORIZON = 8

#: Rollouts per forecast. The spread between two candidate trades is small, so
#: this needs to be high enough that the comparison is not noise — but it runs
#: on every refresh, so it cannot be thousands.
SAMPLES = 160

#: A development card is worth this much in points on average: the victory
#: point cards themselves, plus a share for the knights that win Largest Army.
DEV_CARD_VP = DEV_DECK[DevCard.VICTORY_POINT] / DEV_DECK_SIZE + 0.09

#: Rolls, and how many of the 36 combinations produce each. Sampling from this
#: rather than from ``randint(2, 12)`` is the whole point: a 6 really is five
#: times more likely than a 2, and that is what separates a good spot from a
#: bad one.
_ROLLS: Tuple[Tuple[int, int], ...] = tuple(sorted(DICE_WAYS.items()))
_ROLL_VALUES = [roll for roll, _ in _ROLLS]
_ROLL_WEIGHTS = [ways for _, ways in _ROLLS]


@dataclass
class Outlook:
    """Where a player is likely to stand after the horizon."""

    player: int
    victory_points: float
    turns_to_target: float

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<Outlook p{self.player} {self.victory_points:.2f} vp, "
            f"{self.turns_to_target:.1f} turns>"
        )


def _production_table(
    state: GameState, player: int
) -> Dict[int, Dict[Resource, int]]:
    """
    What each dice number pays this player, counting cities double.

    Built once per forecast rather than per simulated round: the board does
    not change during a rollout, only the hand does.
    """
    table: Dict[int, Dict[Resource, int]] = {}
    for number in _ROLL_VALUES:
        if number == 7:
            continue
        gains: Dict[Resource, int] = {}
        for tile in state.board.tiles_by_number.get(number, ()):
            if tile.resource is None or tile.coord == state.robber:
                continue
            for node_id in tile.nodes:
                entry = state.buildings.get(node_id)
                if entry is None or entry[0] != player:
                    continue
                amount = 2 if entry[1] is Building.CITY else 1
                gains[tile.resource] = gains.get(tile.resource, 0) + amount
        if gains:
            table[number] = gains
    return table


def _can_pay(hand: Dict[Resource, int], cost: Dict[Resource, int]) -> bool:
    return all(hand.get(r, 0) >= n for r, n in cost.items())


def _pay(hand: Dict[Resource, int], cost: Dict[Resource, int]) -> None:
    for resource, count in cost.items():
        hand[resource] -= count


def _trade_toward(
    hand: Dict[Resource, int], cost: Dict[Resource, int], rate: int = 4
) -> bool:
    """
    Convert a surplus into a missing card at the bank, once.

    Real players trade constantly; a forecast that never does badly
    underestimates everyone, and underestimates lopsided production most of
    all — which is exactly the case a trade is being judged in.
    """
    missing = [r for r, n in cost.items() if hand.get(r, 0) < n]
    if len(missing) != 1:
        return False
    need = missing[0]
    for resource, held in sorted(hand.items(), key=lambda kv: -kv[1]):
        if resource in cost and held - cost.get(resource, 0) < rate:
            continue
        if held >= rate:
            hand[resource] -= rate
            hand[need] = hand.get(need, 0) + 1
            return True
    return False


def _rollout(
    production: Dict[int, Dict[Resource, int]],
    hand: Dict[Resource, int],
    start_vp: int,
    settlements: int,
    cities: int,
    rounds: int,
    rolls_per_round: int,
    rng: random.Random,
) -> Tuple[float, float]:
    """
    Play the position forward. Returns ``(victory points, rounds to target)``.

    The buying policy is deliberately plain — cities, then settlements, then
    development cards — because that is the order that turns cards into points
    fastest, and because a cleverer policy would make the forecast harder to
    trust than the thing it is forecasting.
    """
    hand = dict(hand)
    vp = float(start_vp)
    remaining_settlements = SUPPLY["settlement"] - settlements
    upgradeable = settlements
    remaining_cities = SUPPLY["city"] - cities
    reached: Optional[float] = None
    target = 10

    for round_index in range(rounds):
        for _ in range(rolls_per_round):
            roll = rng.choices(_ROLL_VALUES, weights=_ROLL_WEIGHTS, k=1)[0]
            if roll == 7:
                total = sum(hand.values())
                if total > 7:                       # the hand limit bites
                    for _ in range(total // 2):
                        richest = max(hand, key=lambda r: hand[r])
                        if hand[richest] <= 0:
                            break
                        hand[richest] -= 1
                continue
            for resource, amount in production.get(roll, {}).items():
                hand[resource] = hand.get(resource, 0) + amount

        # Spend. Several purchases can land in one round when the cards allow.
        for _ in range(3):
            if remaining_cities > 0 and upgradeable > 0 and (
                _can_pay(hand, COSTS["city"])
                or _trade_toward(hand, COSTS["city"])
            ):
                _pay(hand, COSTS["city"])
                vp += 1
                remaining_cities -= 1
                upgradeable -= 1
                continue
            if remaining_settlements > 0 and (
                _can_pay(hand, COSTS["settlement"])
                or _trade_toward(hand, COSTS["settlement"])
            ):
                # A settlement needs a road out to somewhere legal; charging
                # for one keeps the forecast from expanding for free.
                if _can_pay(hand, COSTS["road"]):
                    _pay(hand, COSTS["road"])
                _pay(hand, COSTS["settlement"])
                vp += 1
                remaining_settlements -= 1
                upgradeable += 1
                continue
            if _can_pay(hand, COSTS["dev_card"]):
                _pay(hand, COSTS["dev_card"])
                vp += DEV_CARD_VP
                continue
            break

        if reached is None and vp >= target:
            reached = round_index + 1

    return vp, float(reached if reached is not None else rounds * 2.5)


def outlook(
    state: GameState,
    player: int,
    *,
    extra: Optional[Dict[Resource, int]] = None,
    rounds: int = HORIZON,
    samples: int = SAMPLES,
    seed: Optional[int] = None,
) -> Outlook:
    """
    Forecast one player's position after ``rounds`` more rounds.

    ``extra`` adjusts the starting hand, which is how a trade is evaluated:
    run it once as things stand and once with the cards moved, and the
    difference is what the trade is worth to that player.

    The random seed is fixed by default so two forecasts compared against each
    other see the same dice. Comparing a trade against no trade across
    different dice would drown the effect in noise.
    """
    hand = dict(state.players[player].hand.cards)
    for resource, delta in (extra or {}).items():
        hand[resource] = max(0, hand.get(resource, 0) + delta)

    production = _production_table(state, player)
    start_vp = rules.victory_points(state, player)
    settlements = len(state.settlements_of(player))
    cities = len(state.cities_of(player))

    rng = random.Random(seed if seed is not None else 20260810 + player)
    total_vp = 0.0
    total_turns = 0.0
    for _ in range(samples):
        vp, turns = _rollout(
            production, hand, start_vp, settlements, cities,
            rounds, state.num_players, rng,
        )
        total_vp += vp
        total_turns += turns

    return Outlook(
        player=player,
        victory_points=total_vp / samples,
        turns_to_target=total_turns / samples,
    )


def trade_race(
    state: GameState,
    me: int,
    other: int,
    give: List[Resource],
    get: List[Resource],
    *,
    rounds: int = HORIZON,
    samples: int = SAMPLES,
) -> Tuple[float, float]:
    """
    What a trade is worth to each side, in expected victory points.

    Returns ``(my gain, their gain)``. This is the question the old heuristic
    could not answer: a card that carries an opponent from seven points to
    nine is fine if it carries me to ten first, and ruinous if it does not.
    Scarcity alone cannot tell those apart, because it never looks at the
    scoreboard or at how fast either engine actually runs.
    """
    mine_before = outlook(state, me, rounds=rounds, samples=samples)
    theirs_before = outlook(state, other, rounds=rounds, samples=samples)

    my_delta: Dict[Resource, int] = {}
    their_delta: Dict[Resource, int] = {}
    for resource in give:
        my_delta[resource] = my_delta.get(resource, 0) - 1
        their_delta[resource] = their_delta.get(resource, 0) + 1
    for resource in get:
        my_delta[resource] = my_delta.get(resource, 0) + 1
        their_delta[resource] = their_delta.get(resource, 0) - 1

    mine_after = outlook(
        state, me, extra=my_delta, rounds=rounds, samples=samples
    )
    theirs_after = outlook(
        state, other, extra=their_delta, rounds=rounds, samples=samples
    )
    return (
        mine_after.victory_points - mine_before.victory_points,
        theirs_after.victory_points - theirs_before.victory_points,
    )
