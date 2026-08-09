"""
Whose turn it is, and what they are allowed to do right now.

This is the piece the app was missing. Without it the screen was a free-form
recorder: any player could be given four settlements at once, for free, in any
order, and every action was offered at every moment. Catan has a strict
sequence, and an advisor that does not follow it cannot advise.

The flow is **derived from the event log**, never stored. ``TurnFlow(state)``
walks ``state.log`` and works out where the game stands. That is the same
discipline :class:`~catanmind.state.GameState` uses for undo, and it has the
same payoff: undo, replay and the turn machine cannot drift apart, because
there is only one fact — the log — and everything else is a function of it.

The sequence
------------
*Setup* runs in snake order (1..N then N..1). Each seat places one settlement
and then one road touching it, and the board advances on its own. The second
settlement pays out its surrounding tiles, as the rules require.

*A turn* is: play a development card or roll; on a 7 everyone over the hand
limit discards, then the robber moves and steals; then build, trade and buy;
then end the turn and the next player is up.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

from catanmind.board import COSTS, DevCard, Resource
from catanmind.state import Event, GameState, Phase
from catanmind import rules


class Step(Enum):
    """Exactly where in the sequence the game is standing."""

    SETUP_SETTLEMENT = "setup_settlement"
    SETUP_ROAD = "setup_road"
    PRE_ROLL = "pre_roll"
    DISCARD = "discard"
    MOVE_ROBBER = "move_robber"
    STEAL = "steal"
    ROAD_BUILDING = "road_building"
    MAIN = "main"
    OVER = "over"


#: What each step is called on screen.
STEP_TITLE: Dict[Step, str] = {
    Step.SETUP_SETTLEMENT: "Place a settlement",
    Step.SETUP_ROAD: "Place a road",
    Step.PRE_ROLL: "Roll the dice",
    Step.DISCARD: "Discard down to the limit",
    Step.MOVE_ROBBER: "Move the robber",
    Step.STEAL: "Steal a card",
    Step.ROAD_BUILDING: "Place your free roads",
    Step.MAIN: "Build, trade, or end the turn",
    Step.OVER: "Game over",
}


@dataclass(frozen=True)
class Action:
    """
    One thing that may be done right now.

    ``target`` tells the screen what the player has to tap next: an
    intersection, a path, a tile, an opponent, or nothing at all. That is what
    lets the UI offer only legal moves instead of a fixed row of buttons.
    """

    id: str
    label: str
    target: Optional[str] = None      # "node" | "edge" | "tile" | "player" | None
    enabled: bool = True
    hint: str = ""
    primary: bool = False             # the expected move, highlighted


def snake_order(num_players: int) -> List[int]:
    """Setup placement order: 1..N then N..1."""
    forward = list(range(1, num_players + 1))
    return forward + forward[::-1]


class TurnFlow:
    """
    The rules of sequence, derived fresh from the log every time.

    Cheap enough to rebuild on every redraw — a full game is a few hundred
    events — and correct by construction after an undo.
    """

    def __init__(self, state: GameState):
        self.state = state
        self.order = snake_order(state.num_players)
        self._derive()

    # -- derivation --------------------------------------------------------

    def _derive(self) -> None:
        state = self.state
        self.setup_index = 0
        self.step = Step.SETUP_SETTLEMENT
        self.current = self.order[0]
        self.has_rolled = False
        self.dev_played = False
        self.last_roll: Optional[int] = None
        self.robber_from_knight = False
        #: Cards drawn on this turn — the rules forbid playing them yet.
        self.bought_this_turn: Dict[DevCard, int] = {}
        #: Free roads still owed by a Road Building card.
        self.free_roads = 0
        in_setup = True

        for event in state.log:
            kind = event.kind
            data = event.data

            if in_setup:
                if kind == "build_settlement":
                    self.step = Step.SETUP_ROAD
                elif kind == "build_road":
                    self.setup_index += 1
                    if self.setup_index >= len(self.order):
                        in_setup = False
                        self.current = 1
                        self.step = Step.PRE_ROLL
                    else:
                        self.current = self.order[self.setup_index]
                        self.step = Step.SETUP_SETTLEMENT
                continue

            if kind == "roll":
                self.has_rolled = True
                self.last_roll = int(data.get("number", 0))  # type: ignore[arg-type]
                self.step = (
                    Step.DISCARD if self.last_roll == 7 else Step.MAIN
                )
            elif kind == "discard":
                self.step = Step.DISCARD
            elif kind in ("play_knight", "play_dev"):
                card = (
                    DevCard(data.get("card", DevCard.KNIGHT.value))
                    if kind == "play_dev" else DevCard.KNIGHT
                )
                if card is not DevCard.VICTORY_POINT:
                    self.dev_played = True
                if card is DevCard.KNIGHT:
                    self.robber_from_knight = True
                    self.step = Step.MOVE_ROBBER
                elif card is DevCard.ROAD_BUILDING:
                    self.free_roads = 2
                    self.step = Step.ROAD_BUILDING
            elif kind == "move_robber":
                self.step = Step.STEAL
            elif kind in ("steal", "skip_steal"):
                self.step = Step.MAIN if self.has_rolled else Step.PRE_ROLL
            elif kind == "build_road":
                if self.free_roads > 0:
                    self.free_roads -= 1
                    self.step = (
                        Step.ROAD_BUILDING if self.free_roads
                        else (Step.MAIN if self.has_rolled else Step.PRE_ROLL)
                    )
            elif kind == "buy_dev":
                bought = data.get("card")
                if bought:
                    key = DevCard(bought)
                    self.bought_this_turn[key] = (
                        self.bought_this_turn.get(key, 0) + 1
                    )
            elif kind == "end_turn":
                self.current = self.current % state.num_players + 1
                self.step = Step.PRE_ROLL
                self.has_rolled = False
                self.dev_played = False
                self.robber_from_knight = False
                self.last_roll = None
                self.bought_this_turn = {}
                self.free_roads = 0

        # A pending discard outranks whatever came next: nobody moves the
        # robber while a player is still over the hand limit.
        if self.step in (Step.DISCARD, Step.MOVE_ROBBER) and not in_setup:
            if self.step is Step.DISCARD:
                self.step = (
                    Step.DISCARD if self.pending_discards() else Step.MOVE_ROBBER
                )

        # Nothing to steal means the steal step resolves itself.
        if self.step is Step.STEAL and not self.steal_victims():
            self.step = Step.MAIN if self.has_rolled else Step.PRE_ROLL

        # Road Building on a board with nowhere left to build would otherwise
        # strand the turn on a step whose only action is impossible.
        if self.step is Step.ROAD_BUILDING and not rules.legal_roads(
            state, self.current
        ):
            self.free_roads = 0
            self.step = Step.MAIN if self.has_rolled else Step.PRE_ROLL

        # You win on your own turn. Points only ever arrive on your turn, and
        # announcing someone else's victory mid-turn would freeze the screen
        # before the current player had finished recording their move.
        if not in_setup:
            champion = rules.winner(state)
            if champion is not None and champion == self.current:
                self.step = Step.OVER

        self.in_setup = in_setup

    # -- questions the screen asks ----------------------------------------

    @property
    def setup_round(self) -> int:
        """1 during the first setup round, 2 during the reverse round."""
        return 1 if self.setup_index < self.state.num_players else 2

    def is_second_settlement(self) -> bool:
        return self.in_setup and self.setup_round == 2

    def is_my_turn(self) -> bool:
        return self.current == self.state.me

    def pending_discards(self) -> List[int]:
        """
        Players still holding more than the limit after a 7.

        Derivable from the live hands: a correct discard drops a player to at
        most seven cards, so anyone still above it has not discarded yet.
        """
        if self.last_roll != 7:
            return []
        return [
            p for p in sorted(self.state.players)
            if self.state.players[p].hand.total() > 7
        ]

    def steal_victims(self) -> List[int]:
        """Opponents with a building on the robber's tile and a card to lose."""
        state = self.state
        tile = state.board.tiles[state.robber]
        out: Set[int] = set()
        for node_id in tile.nodes:
            entry = state.buildings.get(node_id)
            if entry is None:
                continue
            owner = entry[0]
            if owner != self.current and state.players[owner].hand.total() > 0:
                out.add(owner)
        return sorted(out)

    def playable_dev_cards(self) -> List[DevCard]:
        """
        Cards the current player may actually play right now.

        Three rules apply at once: one development card per turn, never a card
        drawn on this same turn, and a victory point is revealed rather than
        played. The old screen ignored all three and offered "play a knight"
        the moment anything was bought.
        """
        if self.dev_played or self.step not in (Step.PRE_ROLL, Step.MAIN):
            return []
        held = self.state.players[self.current].dev_cards
        out: List[DevCard] = []
        for card, count in held.items():
            if not card.playable:
                continue
            if count - self.bought_this_turn.get(card, 0) > 0:
                out.append(card)
        return sorted(out, key=lambda c: c.value)

    def revealable_vp(self) -> bool:
        """A victory point card can be revealed at any time on your turn."""
        return (
            self.step in (Step.PRE_ROLL, Step.MAIN)
            and self.state.players[self.current].holds(DevCard.VICTORY_POINT) > 0
        )

    def setup_settlement_node(self) -> Optional[int]:
        """The settlement the current setup road must touch."""
        if self.step is not Step.SETUP_ROAD:
            return None
        for event in reversed(self.state.log):
            if event.kind == "build_settlement":
                return int(event.data["node"])  # type: ignore[arg-type]
        return None

    def winner(self) -> Optional[int]:
        return rules.winner(self.state)

    # -- what may be done now ---------------------------------------------

    def available_actions(self) -> List[Action]:
        step = self.step
        if step is Step.SETUP_SETTLEMENT:
            return [
                Action(
                    "setup_settlement",
                    f"Place Player {self.current}'s settlement",
                    target="node", primary=True,
                    hint="Tap an intersection. Free, and it needs no road.",
                )
            ]

        if step is Step.SETUP_ROAD:
            self.setup_settlement_node()
            return [
                Action(
                    "setup_road",
                    f"Place Player {self.current}'s road",
                    target="edge", primary=True,
                    hint="Tap a path touching the settlement you just placed.",
                )
            ]

        if step is Step.DISCARD:
            waiting = self.pending_discards()
            return [
                Action(
                    "discard", f"Player {p} discards "
                    f"{rules.must_discard(self.state, p)}",
                    target="player", primary=(p == self.state.me),
                    hint="A 7 was rolled and this hand is over the limit.",
                )
                for p in waiting
            ]

        if step is Step.MOVE_ROBBER:
            return [
                Action(
                    "move_robber", "Move the robber",
                    target="tile", primary=True,
                    hint="Tap the tile to block.",
                )
            ]

        if step is Step.STEAL:
            victims = self.steal_victims()
            out = [
                Action(
                    "steal", f"Steal from Player {p}", target="player",
                    primary=True, hint="Pick who was robbed.",
                )
                for p in victims
            ]
            out.append(Action("skip_steal", "Nobody was robbed"))
            return out

        if step is Step.ROAD_BUILDING:
            return [
                Action(
                    "free_road",
                    f"Place free road ({self.free_roads} left)",
                    target="edge", primary=True,
                    hint="Road building — these two cost nothing.",
                )
            ]

        if step is Step.PRE_ROLL:
            out = [Action("roll", "Roll the dice", primary=True,
                          hint="Enter what the real dice showed.")]
            out += self._dev_actions()
            return out

        if step is Step.MAIN:
            return self._main_actions()

        if step is Step.OVER:
            return [Action("new_game", "Start a new game", primary=True)]

        return []

    def _main_actions(self) -> List[Action]:
        state = self.state
        player = self.current
        out: List[Action] = []

        for item, action_id, target, label in (
            ("road", "build_road", "edge", "Build a road"),
            ("settlement", "build_settlement", "node", "Build a settlement"),
            ("city", "city", "node", "Upgrade to a city"),
        ):
            afford = rules.can_afford(state, player, item)
            if item == "city":
                spots = rules.legal_cities(state, player)
            elif item == "settlement":
                spots = rules.legal_settlements(state, player)
            else:
                spots = rules.legal_roads(state, player)
            supply = state.remaining(player, item)

            if supply <= 0:
                hint = f"No {item} pieces left."
                enabled = False
            elif not spots:
                hint = "Nowhere legal to put one."
                enabled = False
            elif not afford:
                hint = afford.reason or ""
                enabled = False
            else:
                hint = f"Costs {_cost_text(COSTS[item])}."
                enabled = True
            out.append(
                Action(action_id, label, target=target, enabled=enabled,
                       hint=hint)
            )

        afford_dev = rules.can_afford(state, player, "dev_card")
        left = state.dev_deck_left()
        out.append(
            Action(
                "buy_dev", "Buy a development card",
                enabled=bool(afford_dev) and left > 0,
                hint=(
                    f"{left} left. Costs {_cost_text(COSTS['dev_card'])}."
                    if left else "The deck is empty."
                ) if afford_dev else (afford_dev.reason or ""),
            )
        )

        out += self._dev_actions()

        out.append(
            Action("trade", "Trade", hint="Bank, port, or another player.")
        )
        out.append(
            Action("end_turn", "End turn", primary=True,
                   hint=f"Passes to Player {player % state.num_players + 1}.")
        )
        return out

    def _dev_actions(self) -> List[Action]:
        """One button per card actually in hand, plus revealing a point."""
        out: List[Action] = []
        for card in self.playable_dev_cards():
            out.append(
                Action(
                    f"play_dev:{card.value}",
                    f"Play {card.label}",
                    target="tile" if card is DevCard.KNIGHT else None,
                    hint=card.blurb,
                )
            )
        if self.revealable_vp():
            out.append(
                Action("reveal_vp", "Reveal a victory point",
                       hint="Adds 1 to your score.")
            )
        holding = self.state.players[self.current].dev_cards_held
        if holding and not out and not self.dev_played:
            out.append(
                Action(
                    "play_dev_blocked", "Development cards", enabled=False,
                    hint="Cards drawn this turn cannot be played until next turn.",
                )
            )
        return out

    def can(self, action_id: str) -> bool:
        return any(a.id == action_id and a.enabled
                   for a in self.available_actions())

    # -- performing --------------------------------------------------------

    def place_setup_settlement(self, node: int) -> rules.Legality:
        if self.step is not Step.SETUP_SETTLEMENT:
            return rules.Legality(False, "Not the settlement step")
        ok = rules.can_place_settlement(
            self.state, self.current, node, setup=True
        )
        if not ok:
            return ok
        second = self.is_second_settlement()
        self.state.build_settlement(self.current, node, free=True)
        if second:
            # The rules pay out the second settlement immediately.
            self.state.apply(
                Event.make("setup_collect", player=self.current, node=node)
            )
        self._derive()
        return rules.OK

    def place_setup_road(self, edge: int) -> rules.Legality:
        if self.step is not Step.SETUP_ROAD:
            return rules.Legality(False, "Not the road step")
        node = self.setup_settlement_node()
        if node is None:
            return rules.Legality(False, "No settlement to build from")
        ok = rules.can_place_setup_road(self.state, self.current, edge, node)
        if not ok:
            return ok
        self.state.build_road(self.current, edge, free=True)
        was_last = self.setup_index + 1 >= len(self.order)
        self._derive()
        if was_last and self.state.phase is not Phase.PLAY:
            self.state.set_phase(Phase.PLAY)
            self._derive()
        return rules.OK

    def roll(self, number: int) -> rules.Legality:
        if self.step is not Step.PRE_ROLL:
            return rules.Legality(False, "Already rolled this turn")
        self.state.roll(number)
        self._derive()
        return rules.OK

    def discard(self, player: int, cards: Sequence[Resource]) -> rules.Legality:
        owed = rules.must_discard(self.state, player)
        if owed <= 0:
            return rules.Legality(False, f"Player {player} owes nothing")
        if len(cards) != owed:
            return rules.Legality(False, f"Discard exactly {owed}")
        self.state.apply(
            Event.make(
                "discard", player=player, count=owed,
                cards=",".join(c.value for c in cards),
            )
        )
        self._derive()
        return rules.OK

    def move_robber(self, coord: Tuple[int, int]) -> rules.Legality:
        if self.step is not Step.MOVE_ROBBER:
            return rules.Legality(False, "Not the robber step")
        if tuple(coord) == tuple(self.state.robber):
            return rules.Legality(False, "The robber must actually move")
        self.state.move_robber(coord)
        self._derive()
        return rules.OK

    def steal(
        self, victim: int, resource: Optional[Resource] = None
    ) -> rules.Legality:
        if self.step is not Step.STEAL:
            return rules.Legality(False, "Not the steal step")
        if victim not in self.steal_victims():
            return rules.Legality(False, f"Player {victim} cannot be robbed here")
        self.state.steal(thief=self.current, victim=victim, resource=resource)
        self._derive()
        return rules.OK

    def skip_steal(self) -> rules.Legality:
        self.state.apply(Event.make("skip_steal"))
        self._derive()
        return rules.OK

    def build(self, item: str, target: int) -> rules.Legality:
        """Build in the main phase, paying for it."""
        if self.step is not Step.MAIN:
            return rules.Legality(False, "Not the building step")
        state, player = self.state, self.current

        if item == "settlement":
            ok = rules.can_place_settlement(state, player, target)
        elif item == "city":
            ok = rules.can_upgrade_city(state, player, target)
        elif item == "road":
            ok = rules.can_place_road(state, player, target)
        else:
            return rules.Legality(False, f"Unknown item {item!r}")
        if not ok:
            return ok

        afford = rules.can_afford(state, player, item)
        if not afford:
            return afford

        if item == "settlement":
            state.build_settlement(player, target)
        elif item == "city":
            state.build_city(player, target)
        else:
            state.build_road(player, target)
        self._derive()
        return rules.OK

    def buy_dev(self, card: Optional[DevCard] = None) -> rules.Legality:
        """
        Draw a development card.

        ``card`` is the type actually drawn, which the player reads off the
        real card. It is optional so an opponent's purchase can be recorded
        as a face-down unknown.
        """
        if self.step is not Step.MAIN:
            return rules.Legality(False, "Not the building step")
        if self.state.dev_deck_left() <= 0:
            return rules.Legality(False, "The development deck is empty")
        afford = rules.can_afford(self.state, self.current, "dev_card")
        if not afford:
            return afford
        self.state.apply(
            Event.make(
                "buy_dev", player=self.current,
                card=card.value if card else None,
            )
        )
        self._derive()
        return rules.OK

    def play_dev(
        self,
        card: DevCard,
        *,
        resource: Optional[Resource] = None,
        cards: Optional[Sequence[Resource]] = None,
    ) -> rules.Legality:
        """Play one development card, enforcing the timing rules."""
        if card is DevCard.VICTORY_POINT:
            return self.reveal_vp()
        if self.step not in (Step.PRE_ROLL, Step.MAIN):
            return rules.Legality(
                False, "Development cards are played before or after the roll"
            )
        if self.dev_played:
            return rules.Legality(False, "Only one development card per turn")
        if card not in self.playable_dev_cards():
            held = self.state.players[self.current].holds(card)
            if held and self.bought_this_turn.get(card, 0) >= held:
                return rules.Legality(
                    False, "You drew that card this turn — playable next turn"
                )
            return rules.Legality(False, f"No {card.label} card in hand")

        if card is DevCard.MONOPOLY and resource is None:
            return rules.Legality(False, "Name the resource to monopolise")
        if card is DevCard.YEAR_OF_PLENTY and len(list(cards or ())) != 2:
            return rules.Legality(False, "Pick exactly two resources")

        self.state.apply(
            Event.make(
                "play_dev", player=self.current, card=card.value,
                resource=resource.value if resource else None,
                cards=",".join(c.value for c in cards) if cards else None,
            )
        )
        self._derive()
        return rules.OK

    def play_knight(self) -> rules.Legality:
        return self.play_dev(DevCard.KNIGHT)

    def reveal_vp(self) -> rules.Legality:
        if self.state.players[self.current].holds(DevCard.VICTORY_POINT) <= 0:
            return rules.Legality(False, "No victory point card in hand")
        self.state.apply(
            Event.make(
                "play_dev", player=self.current,
                card=DevCard.VICTORY_POINT.value,
            )
        )
        self._derive()
        return rules.OK

    def place_free_road(self, edge: int) -> rules.Legality:
        """One of the two roads granted by a Road Building card."""
        if self.step is not Step.ROAD_BUILDING:
            return rules.Legality(False, "No free roads owed")
        ok = rules.can_place_road(self.state, self.current, edge)
        if not ok:
            return ok
        self.state.build_road(self.current, edge, free=True)
        self._derive()
        return rules.OK

    def trade_bank(
        self, give: Resource, get: Resource
    ) -> rules.Legality:
        if self.step is not Step.MAIN:
            return rules.Legality(False, "Trade during your building step")
        rate = self.state.ports_of(self.current)[give]
        if self.state.players[self.current].hand.cards[give] < rate:
            return rules.Legality(
                False, f"Need {rate} {give.value} to trade at {rate}:1"
            )
        self.state.apply(
            Event.make(
                "trade_bank", player=self.current,
                give=give.value, get=get.value, rate=rate,
            )
        )
        self._derive()
        return rules.OK

    def trade_player(
        self,
        other: int,
        give: Sequence[Resource],
        get: Sequence[Resource],
    ) -> rules.Legality:
        """
        Record a trade with another player.

        Only the current player's side is checked strictly — that is the hand
        we actually know. What the opponent hands over is taken on trust,
        because our picture of their cards is an estimate and refusing a trade
        that really happened would make the app argue with the table.

        Trading is only legal on your own turn, and both sides must give
        something: the rules do not allow handing cards over for nothing.
        """
        if self.step is not Step.MAIN:
            return rules.Legality(False, "Trade during your building step")
        if other == self.current:
            return rules.Legality(False, "Pick a different player")
        if other not in self.state.players:
            return rules.Legality(False, f"No player {other}")
        if not give or not get:
            return rules.Legality(
                False, "A trade has to go both ways — pick cards on both sides"
            )

        hand = self.state.players[self.current].hand
        needed: Dict[Resource, int] = {}
        for resource in give:
            needed[resource] = needed.get(resource, 0) + 1
        short = [
            f"{count - hand.cards[r]} {r.value}"
            for r, count in needed.items() if hand.cards[r] < count
        ]
        if short:
            return rules.Legality(False, "You are short " + ", ".join(short))

        self.state.apply(
            Event.make(
                "trade_player",
                player=self.current,
                other=other,
                give=",".join(r.value for r in give),
                get=",".join(r.value for r in get),
            )
        )
        self._derive()
        return rules.OK

    def end_turn(self) -> rules.Legality:
        if self.step is not Step.MAIN:
            return rules.Legality(False, "Finish the current step first")
        self.state.apply(Event.make("end_turn"))
        self._derive()
        return rules.OK

    def undo(self) -> bool:
        done = self.state.undo()
        self._derive()
        return done

    # -- description -------------------------------------------------------

    def banner(self) -> str:
        """One line describing whose move it is and what is expected."""
        if self.step is Step.OVER:
            winner = self.winner()
            return "You win!" if winner == self.state.me else f"Player {winner} wins."

        mine = self.is_my_turn()
        subject = "You" if mine else f"Player {self.current}"
        possessive = "Your" if mine else f"Player {self.current}'s"
        task = STEP_TITLE[self.step].lower()

        if self.in_setup:
            return f"Setup round {self.setup_round} — {subject}: {task}"
        if self.step is Step.DISCARD:
            waiting = ", ".join(f"P{p}" for p in self.pending_discards())
            return f"Rolled 7 — {waiting} must discard"
        return f"{possessive} turn — {task}"


def _cost_text(cost: Dict[Resource, int]) -> str:
    return " + ".join(
        f"{n}×{r.value}" if n > 1 else r.value for r, n in cost.items()
    )
