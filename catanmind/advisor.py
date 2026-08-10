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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from catanmind.board import (
    Board,
    Building,
    COSTS,
    DevCard,
    DEV_DECK,
    DEV_DECK_SIZE,
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
from catanmind.tracker import Tracker
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

#: How much a trade is worth by what it buys. A trade that completes a city
#: matters more than one that completes a road, so the routes stay
#: comparable across different goals.
TRADE_GOAL_WEIGHT: Dict[str, float] = {
    "city": 1.30, "settlement": 1.25, "dev_card": 1.00, "road": 0.90,
}

#: Asking a player costs nothing if they refuse — the bank is still there
#: afterwards. So an unlikely ask keeps this share of its value rather
#: than being scaled away by the odds, and the ask is tried first.
ASK_FLOOR = 0.62

#: What one spare card is worth giving up. Trades are priced net of this, which
#: is why asking a player for a card beats paying the bank four for the same
#: card: the outcome is identical and it costs three cards less.
CARD_COST = 1.2

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


def _same_board(board: Board, state: GameState) -> None:
    """
    Guard against advising on a board that is no longer in play.

    Editing the layout builds a *new* :class:`Board`, and a
    :class:`~catanmind.scoring.Scorer` holds precomputed per-node values for
    the board it was constructed with. Reusing a stale one produces advice for
    tiles that are not on the table any more — confidently, and with no error.
    Cheap identity check, once per call, so it fails loudly instead.
    """
    if board is not state.board:
        raise ValueError(
            "this advisor was built for a different board; rebuild it after "
            "the layout changes"
        )


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
# The through-line
# --------------------------------------------------------------------------


@dataclass
class Plan:
    """
    What this position is actually trying to do.

    Ranking moves one at a time produces a shopping list, not a strategy. Catan
    positions fall into a handful of recognisable shapes — an ore-and-wheat
    engine wants cities and development cards, a wood-and-brick engine wants
    to sprawl — and knowing which one you are in changes which of two
    similarly-scored moves is the right one.

    :attr:`emphasis` is what makes this more than a caption: it tilts the
    ranking toward the moves the plan is built on.
    """

    key: str
    title: str
    focus: str
    reason: str
    emphasis: Dict[str, float] = field(default_factory=dict)


#: How strongly a plan favours its own moves. Deliberately gentle — the plan
#: breaks ties between comparable moves, it does not override a much better one.
PLAN_EMPHASIS: Dict[str, Dict[str, float]] = {
    "cities": {"upgrade_city": 1.30, "buy_dev_card": 1.10},
    "expansion": {"build_settlement": 1.25, "build_road": 1.20},
    "development": {"buy_dev_card": 1.35},
    "longest_road": {"build_road": 1.40},
    "finish": {
        "upgrade_city": 1.25, "build_settlement": 1.25, "buy_dev_card": 1.20,
    },
    "balanced": {},
}


def strategic_plan(state: GameState, player: int) -> Plan:
    """
    Read the position and name the strategy it supports.

    Based on what the player's *engine* produces rather than what happens to
    be in hand this turn, because a hand empties every time you build and the
    engine is what you actually have to work with.
    """
    income = income_per_round(state, player)
    ore_wheat = income[Resource.ORE] + income[Resource.WHEAT]
    wood_brick = income[Resource.WOOD] + income[Resource.BRICK]
    vp = rules.victory_points(state, player)

    upgradable = len(rules.legal_cities(state, player))
    room = len(rules.legal_settlements(state, player)) > 0
    settlements_left = state.remaining(player, "settlement")

    # Within two points of winning, nothing matters except the fastest points.
    if vp >= state.target_vp - 2:
        return Plan(
            key="finish",
            title="Close it out",
            focus="Take the fastest points on the board.",
            reason=(
                f"You are on {vp} of {state.target_vp}. Stop building the "
                "engine and buy the points."
            ),
            emphasis=PLAN_EMPHASIS["finish"],
        )

    if ore_wheat > wood_brick * 1.35 and upgradable:
        return Plan(
            key="cities",
            title="Cities and development cards",
            focus="Upgrade settlements; buy cards with the spare ore.",
            reason=(
                "Your engine leans on ore and wheat, which is exactly what "
                "cities and development cards cost."
            ),
            emphasis=PLAN_EMPHASIS["cities"],
        )

    if ore_wheat > wood_brick * 1.35:
        return Plan(
            key="development",
            title="Development cards",
            focus="Turn the ore into cards and hunt Largest Army.",
            reason=(
                "You produce ore and wheat but have nothing worth upgrading "
                "yet, so the cards are the better use of it."
            ),
            emphasis=PLAN_EMPHASIS["development"],
        )

    if wood_brick > ore_wheat * 1.35 and (room and settlements_left):
        return Plan(
            key="expansion",
            title="Expand",
            focus="Lay roads and claim more spots.",
            reason=(
                "Wood and brick are what roads and settlements cost, and "
                "there is still room to build."
            ),
            emphasis=PLAN_EMPHASIS["expansion"],
        )

    if wood_brick > ore_wheat * 1.35:
        return Plan(
            key="longest_road",
            title="Longest Road",
            focus="Push the road network; the board is full.",
            reason=(
                "You make wood and brick but have nowhere left to settle, so "
                "the two points are in the road."
            ),
            emphasis=PLAN_EMPHASIS["longest_road"],
        )

    return Plan(
        key="balanced",
        title="Balanced",
        focus="Take whichever build is worth most right now.",
        reason=(
            "Your production is even, so nothing forces a particular route."
        ),
        emphasis=PLAN_EMPHASIS["balanced"],
    )


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
        _same_board(self.board, state)
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
        self.board
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
        _same_board(self.board, state)
        player = state.me if player is None else player
        phase = phase_of(state, player)
        weights = PHASE_WEIGHTS[phase]
        vp = vp_value(state, player)

        plan = strategic_plan(state, player)

        out: List[Advice] = []
        out += self._settlements(state, player, weights, vp)
        out += self._cities(state, player, weights, vp)
        out += self._roads(state, player, weights, vp)
        out += self._dev_cards(state, player, vp)
        out += self.trade_advice(state, player)

        # The plan tilts the ranking toward the moves it is built on, so two
        # comparable options resolve the same way a coherent game would.
        for advice in out:
            advice.value = round(
                advice.value * plan.emphasis.get(advice.action, 1.0), 2
            )

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

    def plan(self, state: GameState, player: Optional[int] = None) -> Plan:
        """The strategy the current position supports."""
        _same_board(self.board, state)
        return strategic_plan(state, state.me if player is None else player)

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
                    reason=spot.explain() + " Worth 1 point.",
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
            reason = (
                "Doubles what this spot already makes "
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

    # -- trading -----------------------------------------------------------

    def _goals(
        self, state: GameState, player: int
    ) -> List[Tuple[str, Dict[Resource, int]]]:
        """
        Every purchase the player is nearly able to make, best first.

        A list rather than a single target: the most valuable goal is often the
        one a trade cannot reach, and stopping there would leave the player
        with no advice at all when a cheaper purchase was one card away.
        """
        hand = state.players[player].hand
        plan = strategic_plan(state, player)
        order = {
            "cities": ("city", "dev_card", "settlement", "road"),
            "development": ("dev_card", "city", "settlement", "road"),
            "expansion": ("settlement", "road", "city", "dev_card"),
            "longest_road": ("road", "settlement", "city", "dev_card"),
            "finish": ("city", "settlement", "dev_card", "road"),
            "balanced": ("city", "settlement", "dev_card", "road"),
        }[plan.key]

        out: List[Tuple[str, Dict[Resource, int]]] = []
        for item in order:
            if item in ("settlement", "city", "road"):
                if state.remaining(player, item) <= 0:
                    continue
                spots = {
                    "settlement": rules.legal_settlements,
                    "city": rules.legal_cities,
                    "road": rules.legal_roads,
                }[item](state, player)
                if not spots:
                    continue
            if item == "dev_card" and state.dev_deck_left() <= 0:
                continue
            cost = COSTS[item]
            short = {
                r: n - hand.cards[r] for r, n in cost.items()
                if hand.cards[r] < n
            }
            if short and sum(short.values()) <= 2:
                out.append((item, short))
        return out

    def _threat_level(
        self, state: GameState, other: int, their_vp: int, leader: int
    ) -> float:
        """
        0..1 — how much it matters that this player gains ground.

        Distance from the target, not a cliff at one score: a player on 7 of
        10 is already worth being careful with, and one on 3 is not.
        """
        # Two points from winning is nearly maximum alarm, not two thirds of
        # it: at that range one good turn ends the game.
        remaining = max(1, state.target_vp - their_vp)
        level = max(0.0, min(1.0, (5.0 - remaining) / 3.5))
        if their_vp >= leader and their_vp >= state.target_vp - 4:
            level = max(level, 0.5)
        return level

    def _wants(
        self, state: GameState, other: int, resource: Resource
    ) -> float:
        """
        0..1 — how badly ``other`` needs one card of ``resource``.

        Driven by what their engine produces rather than by what they hold: a
        player who makes no ore wants ore every turn of the game, whether or
        not they happen to have one in hand right now. This is the number that
        decides both whether they will accept a trade and whether accepting
        it helps them, which is why it is one function and not two.
        """
        produced = rules.expected_yield(state, other, ignore_robber=True)
        rate = produced.get(resource, 0.0)
        if rate < COVERAGE_FLOOR:
            scarcity = 1.0
        else:
            scarcity = max(0.0, 1.0 - rate / SATURATION)
        value = self.scorer.values[resource]
        return max(0.0, min(1.0, scarcity * 0.75 + (value - 0.5) * 0.25))

    def _spare_cards(
        self, state: GameState, player: int, cost: Dict[Resource, int]
    ) -> Dict[Resource, int]:
        """Cards the player can give away without breaking the purchase."""
        hand = state.players[player].hand
        return {
            r: hand.cards[r] - cost.get(r, 0)
            for r in RESOURCES
            if hand.cards[r] - cost.get(r, 0) > 0
        }

    def trade_advice(
        self, state: GameState, player: Optional[int] = None
    ) -> List[Advice]:
        """
        How to get the card that is blocking the plan.

        Two routes. The bank is certain but expensive; another player is cheap
        but has to agree, so the recommendation weighs how likely they are to
        hold the card and whether handing them your surplus is safe. Trading
        the leader exactly what they need is how games are lost, so it is
        priced in rather than left to the player to notice.
        """
        _same_board(self.board, state)
        player = state.me if player is None else player

        offers: List[Advice] = []
        for item, short in self._goals(state, player):
            offers += self._offers_for(state, player, item, short)
        if not offers:
            return []

        # Rank across every goal rather than committing to the first one. The
        # most valuable purchase is often the one no single trade can reach,
        # and a trade that actually completes a cheaper build beats a trade
        # that only chips away at an expensive one.
        offers.sort(key=lambda a: a.value, reverse=True)
        best_by_route: Dict[str, Advice] = {}
        for offer in offers:
            best_by_route.setdefault(offer.action, offer)
        return sorted(
            best_by_route.values(), key=lambda a: a.value, reverse=True
        )

    def _offers_for(
        self, state: GameState, player: int, item: str,
        short: Dict[Resource, int],
    ) -> List[Advice]:
        """Ways to cover ``short`` and complete ``item`` this turn."""
        cost = COSTS[item]
        spare = self._spare_cards(state, player, cost)
        if not spare:
            return []

        rates = state.ports_of(player)
        item_name = item.replace("_", " ")
        weight = TRADE_GOAL_WEIGHT[item]
        out: List[Advice] = []

        def effect(need: Resource) -> Tuple[str, float, str]:
            """
            How much of the purchase this one trade actually unblocks.

            Claiming a trade "finishes a city" when two different cards are
            missing is simply wrong, and a player who acts on it ends the turn
            having spent a card for nothing.
            """
            left = {r: n for r, n in short.items() if r is not need}
            if not left:
                return (f"finishes a {item_name} this turn", 1.0, "high")
            remaining = _join([f"{n} {r.value}" for r, n in left.items()])
            return (
                f"leaves you needing {remaining} for a {item_name}",
                0.55,
                "normal",
            )

        # -- the bank ------------------------------------------------------
        for need, missing in short.items():
            affordable_bank = [
                r for r, count in spare.items() if count >= rates[r] * missing
            ]
            if affordable_bank:
                give = min(affordable_bank, key=lambda r: rates[r])
                phrase, scale, urgency = effect(need)
                out.append(
                    Advice(
                        action="trade_bank",
                        label=(
                            f"Trade {rates[give] * missing}×{give.value} "
                            f"for {missing}×{need.value}"
                        ),
                        value=round(
                            14.0 * scale * weight
                            - rates[give] * missing * CARD_COST, 2
                        ),
                        reason=(
                            f"That {phrase} at {rates[give]}:1, and you can "
                            f"spare the {give.value}."
                        ),
                        affordable=True,
                        urgency=urgency,
                    )
                )
                break

        # -- another player -------------------------------------------------
        tracker = Tracker(state)
        leader = max(
            (rules.victory_points(state, p, include_hidden=False)
             for p in state.opponents(player)),
            default=0,
        )

        best: Optional[Advice] = None
        best_score = 0.0
        for need, missing in short.items():
            for estimate in tracker.all_estimates(exclude=player):
                other = estimate.player
                if estimate.total == 0:
                    continue
                chance = estimate.chance_of(need)
                if chance <= 0:
                    continue

                their_vp = rules.victory_points(state, other, include_hidden=False)
                threat = self._threat_level(state, other, their_vp, leader)

                candidates = [r for r, count in spare.items() if count >= missing]
                if not candidates:
                    continue

                # What to offer is a three-way trade-off: they have to want it
                # enough to say yes, it must not hand a dangerous opponent the
                # card they were missing, and it should be the one we mind
                # losing least. Against a safe player the first term wins;
                # against the leader the second does, which is the whole point.
                def offer_score(resource: Resource) -> float:
                    wanted = self._wants(state, other, resource)
                    return (
                        wanted * (1.0 - threat)
                        - wanted * threat * 2.2
                        - self.scorer.values[resource] * 0.35
                    )

                give = max(candidates, key=offer_score)
                helps_them = self._wants(state, other, give)
                danger = threat * helps_them

                phrase, scale, urgency = effect(need)
                # Asking costs nothing if they refuse — the bank is still
                # there afterwards — so an unlikely ask keeps most of its
                # value rather than being scaled away by the odds.
                confidence = ASK_FLOOR + (1.0 - ASK_FLOOR) * chance
                score = (
                    12.0 * confidence * (1.0 - danger) * scale * weight
                    - missing * CARD_COST
                )
                if score <= best_score:
                    continue

                reason = (
                    f"That {phrase}. Player {other} shows "
                    f"{estimate.describe()}, so about a {chance:.0%} chance "
                    f"they hold {need.value}."
                )
                if helps_them > 0.6:
                    reason += (
                        f" They produce little {give.value}, so they have a "
                        "reason to say yes."
                    )
                elif helps_them < 0.3:
                    reason += (
                        f" They already make plenty of {give.value}, so it "
                        "costs them nothing to hand over — and gains them "
                        "nothing either."
                    )
                if threat >= 0.7 and danger >= 0.4:
                    reason += (
                        f" Careful — Player {other} is on {their_vp} points "
                        f"and needs {give.value}. Only trade if it wins you "
                        "the game first."
                    )
                elif threat >= 0.7:
                    reason += (
                        f" Player {other} is on {their_vp} points, but "
                        f"{give.value} is no use to them, so this is safe."
                    )

                best = Advice(
                    action="trade_player",
                    label=(
                        f"Ask Player {other} for {missing}×{need.value}, "
                        f"offer {missing}×{give.value}"
                    ),
                    value=round(score, 2),
                    reason=reason,
                    affordable=True,
                    urgency=urgency if danger < 0.75 else "normal",
                )
                best_score = score

        if best is not None:
            out.append(best)
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
        _same_board(self.board, state)
        player = state.me if player is None else player
        board = self.board
        best: Optional[Advice] = None
        best_value = 0.0

        rules.victory_points(state, player)
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
