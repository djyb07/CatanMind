"""
How good is a spot?

The old scorer multiplied pips by a hand-tuned per-resource constant *and* by a
scarcity multiplier — counting the same signal twice — then bolted on separate
"diversity" and "synergy" bonuses with magic numbers. This module replaces all
of that with two ideas that between them produce diversity and synergy for
free.

**1. What a card is worth.** Value comes from demand over supply. Demand is how
much of each resource a winning game actually consumes (:data:`BUILD_PLAN`
priced out through the cost table). Supply is how many pips the board prints of
it. Nothing is hand-picked per resource; change the board and the values move.

**2. Diminishing returns.** A second wheat stream is worth less than the first,
and the fifth is worth almost nothing. Utility of a production stream is

    U(portfolio) = sum over r of  value[r] * S * (1 - exp(-EV[r] / S))

which is concave and saturates at ``value[r] * S`` per resource. A spot is
scored by its *marginal* utility: how much it adds to what you already own.
That single formula is why a spot covering three new resources beats a spot
tripling one you already have — no diversity bonus required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from catanmind.board import (
    Board,
    COSTS,
    Port,
    Resource,
    RESOURCES,
    pips as pips_of,
    probability,
)
from catanmind.state import GameState
from catanmind import rules

#: What a typical winning game buys. Used to price demand per resource.
BUILD_PLAN: Dict[str, int] = {
    "settlement": 4,   # on top of the two free setup settlements
    "city": 4,
    "road": 10,
    "dev_card": 4,
}

#: Expected cards-per-roll at which a single resource stream stops adding much.
#: 0.14/roll is roughly one good settlement tile (a 5 or 9 on a settlement).
SATURATION = 0.14

#: Cards per roll below which a stream is treated as "not really covered".
COVERAGE_FLOOR = 0.02

#: Converts "utility of the best spots I can reach" into the same units as
#: production. Room to grow is real but secondary — an opening is won on what
#: the settlement pays out, not on what it might reach in six turns.
EXPANSION_SCALE = 0.55


def _join(items: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c`` — for sentences a player reads."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def demand_weights() -> Dict[Resource, float]:
    """Cards of each resource consumed by :data:`BUILD_PLAN`, normalised to mean 1."""
    totals = {r: 0.0 for r in RESOURCES}
    for item, count in BUILD_PLAN.items():
        for res, n in COSTS[item].items():
            totals[res] += n * count
    mean = sum(totals.values()) / len(totals)
    return {r: v / mean for r, v in totals.items()}


def supply_weights(board: Board) -> Dict[Resource, float]:
    """Pips the board prints for each resource, normalised to mean 1."""
    pips = board.resource_pips()
    mean = sum(pips.values()) / len(pips)
    if mean == 0:  # pragma: no cover - a board with no numbers
        return {r: 1.0 for r in RESOURCES}
    return {r: (pips[r] / mean if pips[r] else 0.25) for r in RESOURCES}


def resource_values(board: Board) -> Dict[Resource, float]:
    """
    Value of one card of each resource on this specific board.

    Demand over supply, clamped to a factor of two either side so a freak board
    cannot make one resource dominate the whole evaluation, then renormalised.
    """
    demand = demand_weights()
    supply = supply_weights(board)
    raw = {r: demand[r] / supply[r] for r in RESOURCES}
    clamped = {r: min(2.0, max(0.5, v)) for r, v in raw.items()}
    mean = sum(clamped.values()) / len(clamped)
    return {r: v / mean for r, v in clamped.items()}


# --------------------------------------------------------------------------


@dataclass
class SpotScore:
    """
    A scored intersection, with the reasoning kept alongside the number.

    The component scores are kept because the solver compares them, but they
    are engine units and mean nothing to a player. What a player reads is
    :meth:`explain` and the :attr:`tiles` chips — the resources and numbers
    actually printed on the board.
    """

    node_id: int
    total: float
    production: float      # marginal utility of the production it adds
    expansion: float       # value of what it opens up to build next
    port: float            # value of the trading rate it unlocks
    blocking: float        # value of denying it to an opponent
    yields: Dict[Resource, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    #: The tiles this spot touches, as ``(resource, number)``, best number
    #: first. ``resource`` is ``None`` for the desert.
    tiles: List[Tuple[Optional[Resource], int]] = field(default_factory=list)
    #: Total dots on those tiles — the number Catan players compare spots by.
    pips: int = 0
    port_type: Optional[Port] = None

    @property
    def numbers(self) -> List[int]:
        return sorted(self._numbers, reverse=True)

    _numbers: List[int] = field(default_factory=list)

    def strength(self) -> str:
        """A plain word for how productive this spot is."""
        if self.pips >= 12:
            return "Excellent"
        if self.pips >= 10:
            return "Strong"
        if self.pips >= 7:
            return "Fair"
        return "Weak"

    def payout_share(self) -> float:
        """
        Share of rolls that pay this spot something.

        Each dot on a number token is one of the 36 dice combinations, so the
        dots add up to a probability directly. This is what "pips" means, said
        in a way a player can act on.
        """
        return min(1.0, self.pips / 36.0)

    def headline(self) -> str:
        """
        A name for the place, built from what is on it.

        Intersections have no names in Catan, so the engine's node number was
        leaking onto the screen — "Spot #34" tells a player nothing. The tiles
        it touches are how people actually point at a spot.
        """
        if not self.tiles:
            return "Empty spot"
        return " · ".join(
            f"{(res.value if res else 'desert').capitalize()} {num}"
            if num else (res.value if res else "desert").capitalize()
            for res, num in self.tiles
        )

    def explain(self) -> str:
        """
        One or two sentences of plain English.

        No engine units: a player wants to know what it produces, what it adds
        that they lack, and what it opens up.
        """
        if not self.notes:
            return f"{self.pips} pips of production."
        return " ".join(self.notes)


class Scorer:
    """
    Evaluates intersections for one board.

    Values that depend only on the board (resource values, per-node standalone
    yield) are computed once at construction. Values that depend on the game
    state are computed per call.
    """

    def __init__(self, board: Board):
        self.board = board
        self.values = resource_values(board)
        # Standalone expected yield of every node, robber ignored.
        self._node_yield: Dict[int, Dict[Resource, float]] = {}
        for node in board.nodes:
            out = {r: 0.0 for r in RESOURCES}
            for coord in node.tiles:
                tile = board.tiles[coord]
                if tile.resource is not None:
                    out[tile.resource] += probability(tile.number)
            self._node_yield[node.id] = out
        self._node_utility = {
            nid: self.utility(y) for nid, y in self._node_yield.items()
        }

    # -- the utility model -------------------------------------------------

    def utility(self, portfolio: Dict[Resource, float]) -> float:
        """Concave, saturating value of a production stream."""
        total = 0.0
        for r in RESOURCES:
            ev = portfolio.get(r, 0.0)
            if ev <= 0:
                continue
            total += self.values[r] * SATURATION * (
                1.0 - math.exp(-ev / SATURATION)
            )
        return total * 100.0  # scale into a comfortable reading range

    def marginal_utility(
        self, existing: Dict[Resource, float], added: Dict[Resource, float]
    ) -> float:
        combined = {r: existing.get(r, 0.0) + added.get(r, 0.0) for r in RESOURCES}
        return self.utility(combined) - self.utility(existing)

    # -- component scores --------------------------------------------------

    def port_value(
        self,
        node_id: int,
        portfolio: Dict[Resource, float],
        current_rates: Optional[Dict[Resource, int]] = None,
    ) -> Tuple[float, Optional[str]]:
        """
        What a port at this node is worth.

        A port only pays when you have a surplus to push through it. The value
        is the share of production above the saturation point, converted at the
        improved rate.
        """
        port = self.board.node(node_id).port
        if port is None:
            return 0.0, None

        rates = current_rates or {r: 4 for r in RESOURCES}

        if port is Port.GENERIC:
            gain = 0.0
            for r in RESOURCES:
                old = rates.get(r, 4)
                if old <= 3:
                    continue
                surplus = max(0.0, portfolio.get(r, 0.0) - SATURATION)
                gain += self.values[r] * surplus * (1 / 3 - 1 / old) * 100.0
            base = 1.2  # option value even with nothing to sell yet
            return gain + base, "3:1 port"

        res = port.resource
        assert res is not None
        old = rates.get(res, 4)
        if old <= 2:
            return 0.0, None
        surplus = max(0.0, portfolio.get(res, 0.0) - SATURATION * 0.5)
        gain = self.values[res] * surplus * (1 / 2 - 1 / old) * 100.0
        note = f"2:1 {res.value} port"
        if portfolio.get(res, 0.0) < COVERAGE_FLOOR:
            return gain + 0.3, note + " (no production to feed it yet)"
        return gain + 0.8, note

    def expansion_value(
        self,
        state: GameState,
        node_id: int,
        player: int,
        *,
        max_roads: int = 3,
        blocked: Optional[set] = None,
    ) -> float:
        """
        Value of the best spots this node can reach.

        Breadth-first over roads out to ``max_roads``, collecting nodes that
        could legally hold a settlement later, discounted by how many roads it
        would take to get there.

        Only the two best destinations count, not every node in range. Summing
        them all made expansion as large as production — a mediocre spot in a
        wide-open corner outscored a far richer one, because it could *see*
        fifteen futures it would never have the roads or turns to build. You
        expand once or twice from an opening settlement, so that is what this
        prices.
        """
        board = self.board
        taken = set(state.buildings)
        if blocked:
            taken = taken | set(blocked)

        # Nodes made illegal by the distance rule.
        forbidden = set(taken)
        for n in taken:
            forbidden.update(board.node(n).neighbors)
        forbidden.update(board.node(node_id).neighbors)
        forbidden.add(node_id)

        seen = {node_id}
        frontier = [(node_id, 0)]
        reachable: List[float] = []
        while frontier:
            current, dist = frontier.pop(0)
            if dist >= max_roads:
                continue
            for nxt in board.node(current).neighbors:
                if nxt in seen:
                    continue
                eid = board.edge_id[(current, nxt)]
                owner = state.roads.get(eid)
                if owner is not None and owner != player:
                    continue  # an opponent's road closes this direction
                seen.add(nxt)
                frontier.append((nxt, dist + 1))
                if nxt in forbidden:
                    continue
                # Discount by the roads needed to reach it.
                reachable.append(self._node_utility[nxt] * (0.55 ** dist))

        reachable.sort(reverse=True)
        best = reachable[:2]
        # The second option is worth less: you will get there later, if at all.
        total = sum(v * w for v, w in zip(best, (1.0, 0.45)))
        return total * EXPANSION_SCALE

    def blocking_value(
        self, state: GameState, node_id: int, player: int
    ) -> float:
        """
        Value of taking this spot away from opponents.

        Only counts when an opponent's road network already reaches the node —
        otherwise nobody was going to get there soon anyway.
        """
        board = self.board
        contested = False
        for eid in board.node_edges[node_id]:
            owner = state.roads.get(eid)
            if owner is not None and owner != player:
                contested = True
                break
        if not contested:
            return 0.0
        return self._node_utility[node_id] * 0.35

    # -- the full score ----------------------------------------------------

    def score_spot(
        self,
        state: GameState,
        node_id: int,
        player: int,
        *,
        portfolio: Optional[Dict[Resource, float]] = None,
        as_city: bool = False,
        weights: Optional[Dict[str, float]] = None,
    ) -> SpotScore:
        """
        Score one intersection for ``player``.

        ``portfolio`` overrides the player's current expected yield, which is
        what the setup solver uses to evaluate "if I had already taken spot A,
        how good is spot B?".
        """
        board = self.board
        w = weights or DEFAULT_WEIGHTS

        if portfolio is None:
            portfolio = rules.expected_yield(state, player, ignore_robber=True)

        added = dict(self._node_yield[node_id])
        if as_city:
            added = {r: v * 2 for r, v in added.items()}

        production = self.marginal_utility(portfolio, added)
        combined = {r: portfolio.get(r, 0.0) + added.get(r, 0.0) for r in RESOURCES}

        expansion = self.expansion_value(state, node_id, player)
        port, port_note = self.port_value(node_id, combined, state.ports_of(player))
        blocking = self.blocking_value(state, node_id, player)

        total = (
            production * w["production"]
            + expansion * w["expansion"]
            + port * w["port"]
            + blocking * w["blocking"]
        )

        node = board.nodes[node_id]
        numbers = [
            board.tiles[c].number for c in node.tiles if board.tiles[c].number
        ]
        tile_list = sorted(
            ((board.tiles[c].resource, board.tiles[c].number) for c in node.tiles),
            key=lambda t: -pips_of(t[1]),
        )
        pip_total = sum(pips_of(n) for n in numbers)

        # Everything below is written to be read by a player, not an engineer.
        notes: List[str] = []
        spread = ", ".join(
            f"{res.value} {num}" for res, num in tile_list if res is not None
        )
        if spread:
            notes.append(f"Produces {spread}.")

        hot = sorted(n for n in numbers if n in (6, 8))
        if hot:
            notes.append(
                f"The {' and '.join(str(n) for n in hot)} "
                f"{'are' if len(hot) > 1 else 'is'} the most frequent roll"
                f"{'s' if len(hot) > 1 else ''} in the game."
            )

        new_resources = [
            r.value for r in RESOURCES
            if added.get(r, 0) > 0 and portfolio.get(r, 0.0) < COVERAGE_FLOOR
        ]
        if new_resources:
            notes.append(
                f"Gives you {_join(new_resources)}, which you don't produce yet."
            )

        if port_note:
            notes.append(f"Sits on a {port_note}.")

        if len(node.tiles) < 3:
            notes.append(
                f"On the coast, so it only touches {len(node.tiles)} "
                f"tile{'s' if len(node.tiles) > 1 else ''}."
            )
        if expansion > 12:
            notes.append("Plenty of room to expand from here.")
        elif expansion < 4:
            notes.append("Little room to expand from here.")

        return SpotScore(
            node_id=node_id,
            total=round(total, 2),
            production=round(production, 2),
            expansion=round(expansion, 2),
            port=round(port, 2),
            blocking=round(blocking, 2),
            yields={r: round(v, 4) for r, v in added.items() if v},
            notes=notes,
            tiles=tile_list,
            pips=pip_total,
            port_type=board.node(node_id).port,
            _numbers=numbers,
        )

    def rank_spots(
        self,
        state: GameState,
        player: int,
        candidates: Optional[Sequence[int]] = None,
        *,
        setup: bool = False,
        portfolio: Optional[Dict[Resource, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        top: Optional[int] = None,
    ) -> List[SpotScore]:
        if candidates is None:
            candidates = rules.legal_settlements(state, player, setup=setup)
        scored = [
            self.score_spot(
                state, nid, player, portfolio=portfolio, weights=weights
            )
            for nid in candidates
        ]
        scored.sort(key=lambda s: s.total, reverse=True)
        return scored[:top] if top else scored


#: Relative importance of each component. Setup play is about production and
#: room to grow; later the balance shifts toward trade and denial.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "production": 1.0,
    "expansion": 0.6,
    "port": 0.5,
    "blocking": 0.3,
}

PHASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "setup": {"production": 1.0, "expansion": 0.7, "port": 0.4, "blocking": 0.1},
    "early": {"production": 1.0, "expansion": 0.6, "port": 0.5, "blocking": 0.3},
    "mid":   {"production": 0.8, "expansion": 0.35, "port": 0.7, "blocking": 0.5},
    "late":  {"production": 0.5, "expansion": 0.1, "port": 0.5, "blocking": 0.4},
}
