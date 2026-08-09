"""
What are the opponents holding?

The old tracker pretended to know each opponent's hand exactly, then quietly
corrupted itself: ``remove`` clamped at zero without recording that the card
had to have come from somewhere, and ``halve`` divided the discard evenly
across five resources with integer division, so a 4-wheat/4-ore hand discarded
nothing at all.

This version keeps an honest split. Cards whose type we watched arrive are
tracked per resource; cards we only know the *count* of (a steal, an
unobserved trade) live in ``unknown``. Every estimate that comes out of here
says how confident it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from catanmind.board import Building, COSTS, Resource, RESOURCES
from catanmind.state import GameState
from catanmind import rules


@dataclass
class HandEstimate:
    """What we believe one opponent is holding."""

    player: int
    known: Dict[Resource, int]
    unknown: int
    total: int

    def likeliest(self) -> Optional[Resource]:
        """The resource most likely to be pulled by a random steal."""
        if self.total == 0:
            return None
        return max(RESOURCES, key=lambda r: self.known.get(r, 0))

    def chance_of(self, resource: Resource) -> float:
        """Probability a random card from this hand is ``resource``."""
        if self.total == 0:
            return 0.0
        # Unknown cards are spread evenly across the five resources.
        return (self.known.get(resource, 0) + self.unknown / 5.0) / self.total

    def confidence(self) -> float:
        """0..1 — how much of the hand we actually watched arrive."""
        if self.total == 0:
            return 1.0
        return 1.0 - self.unknown / self.total

    def describe(self) -> str:
        if self.total == 0:
            return "empty"
        parts = [f"{n} {r.value}" for r, n in self.known.items() if n]
        if self.unknown:
            parts.append(f"{self.unknown} unknown")
        return ", ".join(parts)


class Tracker:
    """
    Derives every opponent's likely hand from the game's event log.

    It does not keep its own mutable copy of the world — it reads
    :class:`~catanmind.state.GameState`, which already replays the log. That
    means the tracker cannot drift out of sync with the board, and undo works
    on it for free.
    """

    def __init__(self, state: GameState):
        self.state = state

    def estimate(self, player: int) -> HandEstimate:
        hand = self.state.players[player].hand
        known = dict(hand.cards)
        # The state's steal/discard handlers guess a specific card when the real
        # one was hidden. Count those guesses as uncertainty rather than fact.
        unknown = self._uncertain_cards(player)
        total = sum(known.values())
        unknown = min(unknown, total)
        for _ in range(unknown):
            best = max(RESOURCES, key=lambda r: known[r])
            if known[best] == 0:
                break
            known[best] -= 1
        return HandEstimate(
            player=player, known=known, unknown=unknown, total=total
        )

    def _uncertain_cards(self, player: int) -> int:
        """How many of this player's cards we had to guess at."""
        n = 0
        for event in self.state.log:
            data = event.data
            if event.kind == "steal":
                if data.get("resource") is None:
                    if data.get("thief") == player or data.get("victim") == player:
                        n += 1
            elif event.kind == "discard":
                if data.get("player") == player and not data.get("cards"):
                    n += int(data.get("count", 0))  # type: ignore[arg-type]
            elif event.kind == "buy_dev" and data.get("player") == player:
                pass
        return n

    def all_estimates(
        self, exclude: Optional[int] = None
    ) -> List[HandEstimate]:
        return [
            self.estimate(p) for p in sorted(self.state.players)
            if p != exclude
        ]

    # -- robbery -----------------------------------------------------------

    def steal_target(
        self, thief: int, want: Optional[Resource] = None
    ) -> Optional[Tuple[int, Resource, str]]:
        """
        Who to steal from and what you are likely to get.

        Returns ``(player, likely_resource, reasoning)``, or ``None`` when every
        opponent is empty — the old version returned player 1 with a wood card
        in that case, which meant it told you to rob yourself.
        """
        best: Optional[Tuple[float, int, Resource, str]] = None
        for est in self.all_estimates(exclude=thief):
            if est.total == 0:
                continue
            resource = (
                want if want and est.chance_of(want) > 0 else est.likeliest()
            )
            if resource is None:
                continue
            score = est.total * 1.0
            if want:
                score += est.chance_of(want) * 6.0
            vp = rules.victory_points(self.state, est.player)
            score += vp * 0.5
            reason = (
                f"Player {est.player} holds about {est.total} card"
                f"{'s' if est.total != 1 else ''} ({est.describe()}); "
                f"{est.chance_of(resource):.0%} chance of {resource.value}"
            )
            if est.confidence() < 0.6:
                reason += " — low confidence, several cards untracked"
            if best is None or score > best[0]:
                best = (score, est.player, resource, reason)

        if best is None:
            return None
        _, player, resource, reason = best
        return (player, resource, reason)

    # -- reading the opponents --------------------------------------------

    def threats(self, me: int) -> List[str]:
        """Short warnings about what opponents are about to be able to do."""
        out: List[str] = []
        for p in self.state.opponents(me):
            est = self.estimate(p)
            if est.total == 0:
                continue
            for item in ("city", "settlement"):
                cost = COSTS[item]
                if all(est.known.get(r, 0) >= n for r, n in cost.items()):
                    out.append(
                        f"Player {p} can afford a {item} "
                        f"({rules.victory_points(self.state, p)} VP)."
                    )
                    break
            if est.total >= 8:
                out.append(
                    f"Player {p} is holding {est.total} cards — a 7 hurts them."
                )
        return out

    def production_forecast(self, player: int) -> Dict[int, Dict[Resource, int]]:
        """
        What each dice number would pay this player right now.

        Used by the UI to show "if a 6 comes up, Player 2 gets 2 ore".
        """
        state = self.state
        out: Dict[int, Dict[Resource, int]] = {}
        for number in range(2, 13):
            if number == 7:
                continue
            gains: Dict[Resource, int] = {}
            for tile in state.board.tiles_by_number.get(number, ()):
                if tile.coord == state.robber or tile.resource is None:
                    continue
                for node_id in tile.nodes:
                    entry = state.buildings.get(node_id)
                    if entry is None or entry[0] != player:
                        continue
                    amount = 2 if entry[1] is Building.CITY else 1
                    gains[tile.resource] = gains.get(tile.resource, 0) + amount
            if gains:
                out[number] = gains
        return out
