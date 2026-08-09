"""
What to do next.

Replaces the old ``solver_initial`` / ``solver_midgame`` / ``strategy_manager``
trio. Two entry points:

``SetupAdvisor``  where to put the two starting settlements and their roads
``TurnAdvisor``   what to do on a normal turn, and what to do about the robber

The setup solver used to take 6.7 seconds because it recomputed every
opponent's view of every spot inside a loop over every candidate. The scores it
was recomputing do not depend on the candidate, so they are computed once here
and reused; the same search now runs in milliseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from catanmind.board import (
    Board,
    Building,
    COSTS,
    DevCard,
    DEV_DECK,
    DEV_DECK_SIZE,
    Port,
    Resource,
    RESOURCES,
    probability,
)
from catanmind.scoring import (
    COVERAGE_FLOOR,
    PHASE_WEIGHTS,
    SATURATION,
    Scorer,
    SpotScore,
    _join,
)
from catanmind.state import GameState, Phase
from catanmind import rules


# --------------------------------------------------------------------------
# Shared vocabulary
# --------------------------------------------------------------------------


@dataclass
class Advice:
    """One recommended action, with everything needed to explain it."""

    action: str                       # "build_settlement", "upgrade_city", ...
    label: str                        # short human title
    value: float                      # comparable score across all actions
    reason: str
    node: Optional[int] = None
    edge: Optional[int] = None
    coord: Optional[Tuple[int, int]] = None
    cost: Optional[Dict[Resource, int]] = None
    affordable: bool = True
    missing: Optional[str] = None     # what you still need, if not affordable
    urgency: str = "normal"           # "normal" | "high"
    #: Rounds of production needed before this is affordable, 0 if it is now.
    turns: float = 0.0
    #: ``value`` discounted by :attr:`turns` — what the ranking actually uses.
    rate: float = 0.0

    def cost_text(self) -> str:
        if not self.cost:
            return "free"
        return " + ".join(
            f"{n}×{r.value}" if n > 1 else r.value
            for r, n in self.cost.items()
        )


#: Pair scores closer together than this are treated as equal, because the
#: opponent model behind them is a guess, not a measurement. Small on purpose:
#: coarser values start discarding the lookahead itself, and with it the
#: difference between picking first and picking last.
PAIR_TOLERANCE = 0.5

#: Cards per round you can realistically raise by trading a surplus away.
#: A round of production rarely converts cleanly, so this discounts it.
TRADE_EFFICIENCY = 0.45


def income_per_round(state: GameState, player: int) -> Dict[Resource, float]:
    """
    Expected cards per *round*, not per roll.

    Every player's roll pays everybody, so a four-player round is four
    chances to collect. Getting this wrong makes every plan look four times
    further away than it is.
    """
    per_roll = rules.expected_yield(state, player)
    rolls = state.num_players
    return {r: v * rolls for r, v in per_roll.items()}


def turns_to_afford(
    state: GameState,
    player: int,
    cost: Optional[Dict[Resource, int]],
    income: Optional[Dict[Resource, float]] = None,
) -> float:
    """
    Rounds until ``player`` can pay ``cost``, given what they already hold.

    A resource the player cannot produce at all is not hopeless — it can be
    traded for — so the shortfall is costed against a discounted surplus
    income rather than being treated as infinite.
    """
    if not cost:
        return 0.0
    hand = state.players[player].hand
    income = income if income is not None else income_per_round(state, player)

    shortfalls: Dict[Resource, int] = {
        r: n - hand.cards[r] for r, n in cost.items() if hand.cards[r] < n
    }
    if not shortfalls:
        return 0.0

    rates = state.ports_of(player)
    surplus = sum(
        income.get(r, 0.0) for r in RESOURCES if r not in cost
    ) * TRADE_EFFICIENCY

    worst = 0.0
    for resource, missing in shortfalls.items():
        direct = income.get(resource, 0.0)
        # Anything we cannot dig up ourselves has to come through the bank.
        traded = surplus / max(2, rates.get(resource, 4))
        rate = direct + traded
        if rate <= 1e-6:
            worst = max(worst, 99.0)
        else:
            worst = max(worst, missing / rate)
    return round(worst, 1)


def describe_node(board: Board, node_id: int, short: bool = False) -> str:
    """
    Name an intersection by the tiles around it.

    Node ids are an implementation detail; a player points at a corner by the
    numbers on it. ``short`` gives just the two best tiles, for a button.
    """
    tiles = sorted(
        (
            (board.tiles[c].resource, board.tiles[c].number)
            for c in board.node(node_id).tiles
        ),
        key=lambda t: -probability(t[1]) if t[1] else 0,
    )
    if not tiles:
        return "the board edge"
    if short:
        tiles = tiles[:2]
    parts = [
        f"{(res.value if res else 'desert')} {num}" if num
        else (res.value if res else "desert")
        for res, num in tiles
    ]
    return "/".join(parts)


def describe_edge(board: Board, edge_id: int) -> str:
    """Name a path by where it leads, not by its id."""
    edge = board.edge(edge_id)
    ends = [describe_edge_end(board, edge.a), describe_edge_end(board, edge.b)]
    return " ↔ ".join(ends)


def describe_edge_end(board: Board, node_id: int) -> str:
    return describe_node(board, node_id, short=True)


def _rounds(turns: float) -> str:
    if turns >= 90:
        return "out of reach without trading for it"
    if turns < 1.5:
        return "1 round away"
    return f"{turns:.0f} rounds away"


def phase_of(state: GameState, player: int) -> str:
    """Which strategic phase the player is in, by public victory points."""
    if state.phase is Phase.SETUP:
        return "setup"
    vp = rules.victory_points(state, player)
    if vp <= 4:
        return "early"
    if vp <= 7:
        return "mid"
    return "late"


#: What one victory point is worth, in the same units as production utility.
#: Points are nearly worthless early — you cannot win on them — and are the
#: only thing that matters at the end.
VP_VALUE = {"setup": 6.0, "early": 9.0, "mid": 22.0, "late": 60.0}


def vp_value(state: GameState, player: int) -> float:
    """Value of one victory point, raised when anyone is close to winning."""
    base = VP_VALUE[phase_of(state, player)]
    leader = max(
        (rules.victory_points(state, p, include_hidden=(p == player))
         for p in state.players),
        default=0,
    )
    if leader >= state.target_vp - 2:
        base *= 2.0
    elif leader >= state.target_vp - 3:
        base *= 1.4
    return base


# --------------------------------------------------------------------------
# Setup phase
# --------------------------------------------------------------------------


def snake_order(num_players: int) -> List[int]:
    """Placement order for the setup phase: 1..N then N..1."""
    forward = list(range(1, num_players + 1))
    return forward + forward[::-1]


def picks_between(seat: int, num_players: int) -> int:
    """How many opponents place between this seat's first and second pick."""
    order = snake_order(num_players)
    first = order.index(seat)
    second = len(order) - 1 - order[::-1].index(seat)
    return second - first - 1


@dataclass
class SetupPlan:
    """A first pick, the second pick it is aiming for, and why."""

    first: int
    score: float
    projected_second: Optional[int]
    combined_resources: List[Resource]
    reason: str
    detail: SpotScore
    road: Optional[int] = None


class SetupAdvisor:
    """Picks the opening settlements, accounting for what opponents will take."""

    def __init__(self, board: Board, scorer: Optional[Scorer] = None):
        self.board = board
        self.scorer = scorer or Scorer(board)

    def recommend(
        self, state: GameState, player: int, seat: Optional[int] = None,
        top: int = 3,
    ) -> List[SetupPlan]:
        """
        Recommend setup placements.

        On the first placement this looks ahead: it simulates the opponents
        taking the best spots still open, then asks which of *our* opening picks
        leaves us with the best pair once they are done.

        On the second placement there is nothing left to look ahead to, so it
        simply maximises the marginal value given what we already own.
        """
        seat = seat if seat is not None else player
        placed = rules.setup_placements_done(state, player)
        legal = rules.legal_settlements(state, player, setup=True)
        if not legal:
            return []

        weights = PHASE_WEIGHTS["setup"]
        mine = rules.expected_yield(state, player, ignore_robber=True)

        # One pass over the board, reused everywhere below.
        base = {
            nid: self.scorer.score_spot(
                state, nid, player, portfolio={}, weights=weights
            )
            for nid in legal
        }

        if placed >= 1:
            return self._second_pick(state, player, legal, mine, weights, top)

        return self._first_pick(
            state, player, seat, legal, base, weights, top
        )

    # -- second (and final) setup pick ------------------------------------

    def _second_pick(
        self, state: GameState, player: int, legal: Sequence[int],
        mine: Dict[Resource, float], weights: Dict[str, float], top: int,
    ) -> List[SetupPlan]:
        scored = [
            self.scorer.score_spot(
                state, nid, player, portfolio=mine, weights=weights
            )
            for nid in legal
        ]
        scored.sort(key=lambda s: s.total, reverse=True)

        plans = []
        for spot in scored[:top]:
            combined = self._combined_resources(state, player, spot.node_id)
            missing = [r for r in RESOURCES if r not in combined]
            reason = spot.explain()
            if missing:
                reason += (
                    f" Together with your first settlement you still have no "
                    f"{_join([r.value for r in missing])} — you'll need to trade "
                    "or expand for it."
                )
            else:
                reason += " That completes all five resources."
            plans.append(
                SetupPlan(
                    first=spot.node_id,
                    score=spot.total,
                    projected_second=None,
                    combined_resources=sorted(combined, key=lambda r: r.value),
                    reason=reason,
                    detail=spot,
                    road=self.recommend_road(state, player, spot.node_id),
                )
            )
        return plans

    # -- first setup pick, with opponent lookahead ------------------------

    def _first_pick(
        self, state: GameState, player: int, seat: int,
        legal: Sequence[int], base: Dict[int, SpotScore],
        weights: Dict[str, float], top: int,
    ) -> List[SetupPlan]:
        board = self.board
        gap = picks_between(seat, state.num_players)

        # Opponents are modelled as taking the best spot on the board by the
        # same standalone measure. Precomputed once — this is the loop that
        # used to be rebuilt for every candidate.
        ranked = sorted(legal, key=lambda n: base[n].total, reverse=True)

        candidates: List[Tuple[float, int, Optional[int], SpotScore]] = []
        # Only the strongest openers are worth a full second-pick search.
        shortlist = ranked[: min(len(ranked), 18)]

        for first in shortlist:
            # ``blocked`` already carries the distance rule for both our own
            # pick and every simulated opponent pick, so a projected second
            # settlement can never sit next door to one of them.
            blocked = self._simulate_opponents(ranked, base, first, gap)

            first_yield = dict(self.scorer._node_yield[first])
            best_second, best_second_score = None, 0.0
            for second in ranked:
                if second in blocked:
                    continue
                spot = self.scorer.score_spot(
                    state, second, player, portfolio=first_yield, weights=weights
                )
                if spot.total > best_second_score:
                    best_second_score = spot.total
                    best_second = second

            combined = base[first].total + best_second_score
            candidates.append((combined, first, best_second, base[first]))

        # The pair score rests on a guess about what opponents will take, so
        # fractions of a point in it are noise. Quantise before comparing and
        # let the spot's own strength settle anything close — otherwise a
        # speculative second pick can promote a visibly weaker opening.
        candidates.sort(
            key=lambda c: (-round(c[0] / PAIR_TOLERANCE), -c[3].total, c[1])
        )

        plans = []
        for combined, first, second, detail in candidates[:top]:
            resources = self._combined_resources(state, player, first, second)
            reason = detail.explain()
            if second is not None:
                missing = [r for r in RESOURCES if r not in resources]
                reason += (
                    f" Then aim for {describe_node(self.board, second)} "
                    f"on your second pick — "
                    f"{gap} other player{'s' if gap != 1 else ''} "
                    f"pick{'' if gap == 1 else ''} before you go again. "
                )
                reason += (
                    "That pair covers all five resources."
                    if not missing
                    else f"That pair would still have no {_join([r.value for r in missing])}."
                )
            plans.append(
                SetupPlan(
                    first=first,
                    score=round(combined, 2),
                    projected_second=second,
                    combined_resources=sorted(resources, key=lambda r: r.value),
                    reason=reason,
                    detail=detail,
                    road=self.recommend_road(state, player, first, toward=second),
                )
            )
        return plans

    def _simulate_opponents(
        self, ranked: Sequence[int], base: Dict[int, SpotScore],
        our_pick: int, count: int,
    ) -> Set[int]:
        """
        Greedy model: each opponent takes the best spot still open.

        Returns everything we may no longer settle on afterwards — the picks
        themselves, our own pick, and all of their neighbours under the
        distance rule.
        """
        board = self.board
        gone = {our_pick} | set(board.node(our_pick).neighbors)
        for _ in range(count):
            for nid in ranked:
                if nid in gone:
                    continue
                gone.add(nid)
                gone.update(board.node(nid).neighbors)
                break
            else:
                break
        return gone

    def _combined_resources(
        self, state: GameState, player: int, *nodes: Optional[int]
    ) -> Set[Resource]:
        out: Set[Resource] = {
            r for r, v in
            rules.expected_yield(state, player, ignore_robber=True).items()
            if v > 0
        }
        for node_id in nodes:
            if node_id is None:
                continue
            for coord in self.board.node(node_id).tiles:
                tile = self.board.tiles[coord]
                if tile.resource is not None:
                    out.add(tile.resource)
        return out

    # -- setup road --------------------------------------------------------

    def recommend_road(
        self, state: GameState, player: int, settlement: int,
        toward: Optional[int] = None,
    ) -> Optional[int]:
        """
        Which road to lay from a fresh settlement.

        Points at the second settlement if we have one in mind, otherwise at
        whichever neighbour opens the most future value.
        """
        board = self.board
        options = [
            (nbr, board.edge_id[(settlement, nbr)])
            for nbr in board.node(settlement).neighbors
            if board.edge_id[(settlement, nbr)] not in state.roads
        ]
        if not options:
            return None

        if toward is not None:
            dist = self._road_distances(state, player, toward)
            options.sort(key=lambda o: (dist.get(o[0], 99), -self._reach(state, player, o[0])))
            return options[0][1]

        options.sort(key=lambda o: -self._reach(state, player, o[0]))
        return options[0][1]

    def _reach(self, state: GameState, player: int, node_id: int) -> float:
        return self.scorer.expansion_value(state, node_id, player, max_roads=2)

    def _road_distances(
        self, state: GameState, player: int, target: int
    ) -> Dict[int, int]:
        """Road-steps from every node to ``target``."""
        board = self.board
        dist = {target: 0}
        frontier = [target]
        while frontier:
            nxt = []
            for node in frontier:
                for nbr in board.node(node).neighbors:
                    if nbr in dist:
                        continue
                    dist[nbr] = dist[node] + 1
                    nxt.append(nbr)
            frontier = nxt
        return dist


# --------------------------------------------------------------------------
# Normal turns
# --------------------------------------------------------------------------


class TurnAdvisor:
    """Ranks every action available on a normal turn."""

    def __init__(self, board: Board, scorer: Optional[Scorer] = None):
        self.board = board
        self.scorer = scorer or Scorer(board)

    def recommend(
        self, state: GameState, player: Optional[int] = None, top: int = 6,
    ) -> List[Advice]:
        """
        Every action worth considering, best first.

        Unaffordable actions are included and marked, so the advice answers
        "what should I be saving for?" rather than going silent when your hand
        is empty — which is what the old engine did on every single turn.
        """
        player = state.me if player is None else player
        phase = phase_of(state, player)
        weights = PHASE_WEIGHTS[phase]
        vp = vp_value(state, player)

        out: List[Advice] = []
        out += self._settlements(state, player, weights, vp)
        out += self._cities(state, player, weights, vp)
        out += self._roads(state, player, weights, vp)
        out += self._dev_cards(state, player, vp)
        out += self._trades(state, player)

        # Look ahead: a move is worth what it gains divided by how long it
        # takes to get there. Without this the engine is purely greedy and
        # will chase an expensive city it cannot fund for six rounds over a
        # settlement it can pay for next turn.
        income = income_per_round(state, player)
        for advice in out:
            advice.turns = turns_to_afford(state, player, advice.cost, income)
            advice.rate = advice.value / (1.0 + advice.turns)
            if advice.turns >= 1.0 and not advice.affordable:
                advice.reason += (
                    f" About {_rounds(advice.turns)} at your current production."
                )

        out.sort(key=lambda a: (a.affordable, a.rate), reverse=True)
        return out[:top]

    # -- individual action families ---------------------------------------

    def _settlements(
        self, state: GameState, player: int,
        weights: Dict[str, float], vp: float,
    ) -> List[Advice]:
        if state.remaining(player, "settlement") <= 0:
            return []
        legal = rules.legal_settlements(state, player)
        if not legal:
            return []
        afford = rules.can_afford(state, player, "settlement")
        spots = self.scorer.rank_spots(
            state, player, legal, weights=weights, top=3
        )
        out = []
        for spot in spots:
            gain = spot.total + vp
            out.append(
                Advice(
                    action="build_settlement",
                    label="Settle on " + describe_node(self.board, spot.node_id),
                    value=round(gain, 2),
                    reason=spot.explain() + f" Worth 1 VP.",
                    node=spot.node_id,
                    cost=COSTS["settlement"],
                    affordable=bool(afford),
                    missing=afford.reason,
                )
            )
        return out

    def _cities(
        self, state: GameState, player: int,
        weights: Dict[str, float], vp: float,
    ) -> List[Advice]:
        if state.remaining(player, "city") <= 0:
            return []
        afford = rules.can_afford(state, player, "city")
        portfolio = rules.expected_yield(state, player, ignore_robber=True)
        out = []
        for node_id in rules.legal_cities(state, player):
            # Upgrading adds a second copy of what this node already makes.
            added = {
                r: v for r, v in
                rules.node_yield(state, node_id, ignore_robber=True).items() if v
            }
            gain_production = self.scorer.marginal_utility(portfolio, added)
            gain = gain_production * weights["production"] + vp
            numbers = sorted(
                (self.board.tiles[c].number
                 for c in self.board.node(node_id).tiles
                 if self.board.tiles[c].number),
                reverse=True,
            )
            reason = (
                f"Doubles what this spot already makes "
                f"({describe_node(self.board, node_id)}). Worth 1 point."
            )
            if state.robber in self.board.node(node_id).tiles:
                reason += " The robber is sitting on one of its tiles right now."
            out.append(
                Advice(
                    action="upgrade_city",
                    label="City on " + describe_node(self.board, node_id),
                    value=round(gain, 2),
                    reason=reason,
                    node=node_id,
                    cost=COSTS["city"],
                    affordable=bool(afford),
                    missing=afford.reason,
                )
            )
        out.sort(key=lambda a: a.value, reverse=True)
        return out[:3]

    def _roads(
        self, state: GameState, player: int,
        weights: Dict[str, float], vp: float,
    ) -> List[Advice]:
        if state.remaining(player, "road") <= 0:
            return []
        legal = rules.legal_roads(state, player)
        if not legal:
            return []
        afford = rules.can_afford(state, player, "road")

        current = rules.longest_road(state, player)
        holder = rules.longest_road_holder(state)
        rival = max(
            (rules.longest_road(state, p) for p in state.opponents(player)),
            default=0,
        )

        out = []
        for edge_id in legal:
            edge = self.board.edge(edge_id)
            reach = 0.0
            targets = []
            for endpoint in (edge.a, edge.b):
                if endpoint in state.buildings:
                    continue
                spot = self.scorer.score_spot(
                    state, endpoint, player, weights=weights
                )
                if spot.production > 0:
                    reach = max(reach, spot.total)
                    targets.append(endpoint)

            # Longest Road: worth 2 VP, but only once you are actually near it.
            road_vp = 0.0
            road_note = ""
            if holder != player:
                needed = max(rules.LONGEST_ROAD_MINIMUM, rival + 1)
                gap = needed - (current + 1)
                if gap <= 0:
                    road_vp = 2 * vp * 0.9
                    road_note = " Takes Longest Road."
                elif gap <= 2:
                    road_vp = 2 * vp * (0.45 if gap == 1 else 0.2)
                    road_note = f" {gap} more road{'s' if gap > 1 else ''} to Longest Road."

            value = reach * 0.5 + road_vp
            if value <= 0:
                continue
            reason = (
                "Opens up "
                + _join([describe_node(self.board, t) for t in targets])
                if targets else "Extends your road network"
            )
            out.append(
                Advice(
                    action="build_road",
                    label="Road toward " + describe_edge_end(
                        self.board, targets[0] if targets else edge.b
                    ),
                    value=round(value, 2),
                    reason=reason + "." + road_note,
                    edge=edge_id,
                    cost=COSTS["road"],
                    affordable=bool(afford),
                    missing=afford.reason,
                )
            )
        out.sort(key=lambda a: a.value, reverse=True)
        return out[:3]

    def _dev_cards(
        self, state: GameState, player: int, vp: float
    ) -> List[Advice]:
        left = state.dev_deck_left()
        if left <= 0:
            return []
        afford = rules.can_afford(state, player, "dev_card")

        # Expected value of an unseen card, from what is left in the deck.
        vp_chance = DEV_DECK[DevCard.VICTORY_POINT] / DEV_DECK_SIZE
        knight_chance = DEV_DECK[DevCard.KNIGHT] / DEV_DECK_SIZE

        value = vp_chance * vp
        note = f"About {vp_chance:.0%} chance of a victory point"

        holder = rules.largest_army_holder(state)
        knights = state.players[player].knights_played
        if holder != player:
            best_rival = max(
                (state.players[p].knights_played for p in state.opponents(player)),
                default=0,
            )
            needed = max(rules.LARGEST_ARMY_MINIMUM, best_rival + 1) - knights
            if needed <= 1:
                value += knight_chance * 2 * vp * 0.8
                note += "; one knight from Largest Army"
            elif needed == 2:
                value += knight_chance * 2 * vp * 0.35
                note += f"; {needed} knights from Largest Army"
        value += knight_chance * 1.5  # knights are useful even without the card

        return [
            Advice(
                action="buy_dev_card",
                label="Buy a development card",
                value=round(value, 2),
                reason=note + f". {left} cards left in the deck.",
                cost=COSTS["dev_card"],
                affordable=bool(afford),
                missing=afford.reason,
            )
        ]

    def _trades(self, state: GameState, player: int) -> List[Advice]:
        """Suggest a bank/port trade when a surplus is blocking a purchase."""
        hand = state.players[player].hand
        rates = state.ports_of(player)
        out: List[Advice] = []

        for item in ("city", "settlement", "road", "dev_card"):
            cost = COSTS[item]
            short = {r: n - hand.cards[r] for r, n in cost.items()
                     if hand.cards[r] < n}
            if not short or sum(short.values()) > 1:
                continue  # only suggest trades that complete a purchase now
            need = next(iter(short))
            spare = [
                r for r in RESOURCES
                if r not in cost and hand.cards[r] >= rates[r]
            ] + [
                r for r in RESOURCES
                if r in cost and hand.cards[r] - cost[r] >= rates[r]
            ]
            if not spare:
                continue
            give = min(spare, key=lambda r: rates[r])
            out.append(
                Advice(
                    action="trade_bank",
                    label=f"Trade {rates[give]}×{give.value} → {need.value}",
                    value=6.0,
                    reason=(
                        f"That completes a {item.replace('_', ' ')} this turn "
                        f"at {rates[give]}:1."
                    ),
                    cost=None,
                    affordable=True,
                    urgency="high",
                )
            )
            break
        return out

    # -- robber ------------------------------------------------------------

    def robber_advice(
        self, state: GameState, player: Optional[int] = None
    ) -> Optional[Advice]:
        """
        Where to put the robber and who to rob.

        Scores each candidate tile by the production it denies — weighted by how
        close that opponent is to winning — plus the chance of a useful steal.
        """
        player = state.me if player is None else player
        board = self.board
        best: Optional[Advice] = None
        best_value = 0.0

        my_vp = rules.victory_points(state, player)
        threat = {
            p: max(1.0, rules.victory_points(state, p) / max(1, state.target_vp - 2))
            for p in state.opponents(player)
        }

        for coord, tile in board.tiles.items():
            if coord == state.robber or tile.resource is None:
                continue

            denied: Dict[int, float] = {}
            hits_me = False
            for node_id in tile.nodes:
                entry = state.buildings.get(node_id)
                if entry is None:
                    continue
                owner, kind = entry
                amount = probability(tile.number) * (2 if kind is Building.CITY else 1)
                if owner == player:
                    hits_me = True
                    continue
                denied[owner] = denied.get(owner, 0.0) + amount

            if not denied or hits_me:
                continue

            target = max(denied, key=lambda p: denied[p] * threat[p])
            value = sum(
                self.scorer.values[tile.resource] * amount * threat[p] * 100
                for p, amount in denied.items()
            )
            # Robbing someone holding cards is worth more than robbing an empty hand.
            cards = state.players[target].hand.total()
            value += min(cards, 4) * 1.5

            if value > best_value:
                best_value = value
                victim = max(
                    denied, key=lambda p: (state.players[p].hand.total(), denied[p])
                )
                likely = max(
                    RESOURCES, key=lambda r: state.players[victim].hand.cards[r]
                )
                held = state.players[victim].hand.total()
                reason = (
                    f"Blocks {tile.resource.value} on {tile.number} for "
                    + ", ".join(
                        f"Player {p} ({rules.victory_points(state, p)} VP)"
                        for p in sorted(denied, key=lambda p: -denied[p])
                    )
                    + "."
                )
                if held:
                    reason += (
                        f" Rob Player {victim} — about {held} card"
                        f"{'s' if held != 1 else ''}, most likely {likely.value}."
                    )
                else:
                    reason += f" Player {victim} looks empty, so expect nothing from the steal."
                best = Advice(
                    action="move_robber",
                    label=f"Robber → {tile.resource.value} {tile.number}",
                    value=round(value, 2),
                    reason=reason,
                    coord=coord,
                    urgency="high",
                )
        return best

    def discard_advice(
        self, state: GameState, player: Optional[int] = None
    ) -> Optional[Advice]:
        """Which cards to throw away on a 7."""
        player = state.me if player is None else player
        n = rules.must_discard(state, player)
        if n <= 0:
            return None

        hand = state.players[player].hand
        # Keep what the next purchase needs; shed the rest, cheapest first.
        wanted: Dict[Resource, int] = {}
        for advice in self.recommend(state, player, top=3):
            if advice.cost:
                for r, count in advice.cost.items():
                    wanted[r] = max(wanted.get(r, 0), count)

        pool = []
        for r in RESOURCES:
            spare = hand.cards[r] - wanted.get(r, 0)
            for i in range(hand.cards[r]):
                # Cards beyond what the plan needs go first, and within each
                # group the cheapest card goes before the dearer one.
                priority = (0 if i < max(0, spare) else 1, self.scorer.values[r])
                pool.append((priority, r))
        # Sort on the priority alone: Resource is an Enum and does not order,
        # so letting the tuple fall through to it raises on any tie.
        pool.sort(key=lambda item: item[0])
        picks = [r for _, r in pool[:n]]

        counts: Dict[Resource, int] = {}
        for r in picks:
            counts[r] = counts.get(r, 0) + 1
        text = ", ".join(f"{c}×{r.value}" for r, c in counts.items())
        return Advice(
            action="discard",
            label=f"Discard {n}",
            value=0.0,
            reason=f"Drop {text}; that keeps what your next build needs.",
            urgency="high",
        )

    # -- situational warnings ---------------------------------------------

    def alerts(self, state: GameState, player: Optional[int] = None) -> List[str]:
        player = state.me if player is None else player
        out: List[str] = []
        hand = state.players[player].hand

        total = hand.total()
        if total >= 8:
            out.append(
                f"{total} cards in hand — a 7 costs you {total // 2}. Spend some."
            )

        rates = state.ports_of(player)
        for r in RESOURCES:
            if hand.cards[r] >= rates[r] and rates[r] < 4:
                out.append(
                    f"{hand.cards[r]} {r.value} and a {rates[r]}:1 port — trade some."
                )

        produced = rules.expected_yield(state, player, ignore_robber=True)
        missing = [r.value for r in RESOURCES if produced[r] < COVERAGE_FLOOR]
        if missing:
            out.append(f"You produce no {', '.join(missing)}. Trade or expand for it.")

        my_tiles = {
            c for nid in state.nodes_of(player)
            for c in self.board.node(nid).tiles
        }
        if state.robber in my_tiles:
            tile = self.board.tiles[state.robber]
            out.append(
                f"The robber is on your {tile.resource.value if tile.resource else 'desert'} "
                f"{tile.number} — move it with a knight."
            )

        for p in state.opponents(player):
            vp = rules.victory_points(state, p, include_hidden=False)
            if vp >= state.target_vp - 2:
                out.append(f"Player {p} is on {vp} visible VP. Block, don't build.")

        return out
