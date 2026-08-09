"""
Game state: the single source of truth.

The previous design kept resources in two places — a ``Player.resources`` dict
and a separate tracker — and only ever wrote to one of them, so every
affordability check saw an empty hand. Here there is exactly one place a card
can live: ``GameState.players[pid].hand``.

Undo is implemented by *replaying the event log from scratch*. A full replay of
a finished game costs well under a millisecond, and it cannot drift out of sync
with the forward path the way hand-written inverse operations do.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from catanmind.board import (
    Board,
    Building,
    COSTS,
    DevCard,
    DEV_DECK_SIZE,
    Layout,
    Port,
    Resource,
    RESOURCES,
    SUPPLY,
)


class Phase(Enum):
    """Which part of the game we are in."""

    SETUP = "setup"      # placing the two starting settlements and roads
    PLAY = "play"        # normal turns
    OVER = "over"        # someone reached the target score


@dataclass
class Hand:
    """A player's resource cards. Counts are never allowed to go negative."""

    cards: Dict[Resource, int] = field(
        default_factory=lambda: {r: 0 for r in RESOURCES}
    )

    def total(self) -> int:
        return sum(self.cards.values())

    def add(self, resource: Resource, n: int = 1) -> None:
        self.cards[resource] += n

    def take(self, resource: Resource, n: int = 1) -> int:
        """Remove up to ``n`` cards; return how many were actually removed."""
        removed = min(n, self.cards[resource])
        self.cards[resource] -= removed
        return removed

    def can_pay(self, cost: Dict[Resource, int]) -> bool:
        return all(self.cards[r] >= n for r, n in cost.items())

    def pay(self, cost: Dict[Resource, int]) -> bool:
        """Deduct ``cost`` if affordable. Returns whether it was paid."""
        if not self.can_pay(cost):
            return False
        for r, n in cost.items():
            self.cards[r] -= n
        return True

    def copy(self) -> "Hand":
        return Hand(cards=dict(self.cards))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        held = ", ".join(f"{n}{r.value[0]}" for r, n in self.cards.items() if n)
        return f"<Hand {held or 'empty'}>"


@dataclass
class PlayerState:
    """
    One player's holdings.

    Development cards are split the way knowledge actually is: ``dev_cards``
    holds cards whose type we know — your own draws, and anything an opponent
    has revealed — while ``unknown_dev`` counts an opponent's face-down cards.
    The old model kept a single ``dev_cards_held`` integer, which is why the
    screen offered "play a knight" the instant anything was bought.
    """

    id: int
    name: str
    hand: Hand = field(default_factory=Hand)
    knights_played: int = 0
    dev_cards: Dict["DevCard", int] = field(default_factory=dict)
    unknown_dev: int = 0           # face-down cards of unknown type
    vp_cards: int = 0              # revealed victory-point cards
    has_longest_road: bool = False
    has_largest_army: bool = False

    @property
    def dev_cards_held(self) -> int:
        """Every unplayed development card, known type or not."""
        return sum(self.dev_cards.values()) + self.unknown_dev

    def holds(self, card: "DevCard") -> int:
        return self.dev_cards.get(card, 0)

    def add_dev(self, card: Optional["DevCard"]) -> None:
        if card is None:
            self.unknown_dev += 1
        else:
            self.dev_cards[card] = self.dev_cards.get(card, 0) + 1

    def take_dev(self, card: "DevCard") -> bool:
        """Spend one card of that type, falling back to an unknown one."""
        if self.dev_cards.get(card, 0) > 0:
            self.dev_cards[card] -= 1
            return True
        if self.unknown_dev > 0:
            # An opponent just revealed what one of their face-down cards was.
            self.unknown_dev -= 1
            return True
        return False

    def copy(self) -> "PlayerState":
        return PlayerState(
            id=self.id,
            name=self.name,
            hand=self.hand.copy(),
            knights_played=self.knights_played,
            dev_cards=dict(self.dev_cards),
            unknown_dev=self.unknown_dev,
            vp_cards=self.vp_cards,
            has_longest_road=self.has_longest_road,
            has_largest_army=self.has_largest_army,
        )


@dataclass(frozen=True)
class Event:
    """One recorded thing that happened. The log of these *is* the game."""

    kind: str
    payload: Tuple[Tuple[str, object], ...] = ()

    @staticmethod
    def make(kind: str, **payload) -> "Event":
        return Event(kind=kind, payload=tuple(sorted(payload.items())))

    @property
    def data(self) -> Dict[str, object]:
        return dict(self.payload)

    def describe(self) -> str:
        d = self.data
        p = d.get("player")
        if self.kind == "roll":
            return f"Rolled {d['number']}"
        if self.kind == "build_settlement":
            return f"P{p} settled #{d['node']}"
        if self.kind == "build_city":
            return f"P{p} upgraded #{d['node']}"
        if self.kind == "build_road":
            return f"P{p} road #{d['edge']}"
        if self.kind == "move_robber":
            return f"Robber to {d['coord']}"
        if self.kind == "steal":
            return f"P{d['thief']} stole from P{d['victim']}"
        if self.kind == "discard":
            return f"P{p} discarded {d['count']}"
        if self.kind == "trade_bank":
            return f"P{p} traded with the bank"
        if self.kind == "trade_player":
            return f"P{p} traded with P{d['other']}"
        if self.kind == "buy_dev":
            return f"P{p} bought a development card"
        if self.kind == "play_knight":
            return f"P{p} played a knight"
        if self.kind == "reveal_vp":
            return f"P{p} revealed a victory point card"
        if self.kind == "end_turn":
            return "End of turn"
        if self.kind == "adjust":
            return f"P{p} hand corrected"
        return self.kind


class GameState:
    """
    Everything mutable about a game in progress.

    Mutation happens only through :meth:`apply`, which appends to the event log
    so that :meth:`undo` can rebuild the state by replaying.
    """

    def __init__(
        self,
        board: Board,
        num_players: int = 4,
        me: int = 1,
        target_vp: int = 10,
    ):
        self.board = board
        self.num_players = num_players
        self.me = me
        self.target_vp = target_vp
        self.log: List[Event] = []
        self._reset()

    # -- lifecycle ---------------------------------------------------------

    def _reset(self) -> None:
        self.players: Dict[int, PlayerState] = {
            i: PlayerState(id=i, name=f"Player {i}")
            for i in range(1, self.num_players + 1)
        }
        #: node id -> (player id, Building)
        self.buildings: Dict[int, Tuple[int, Building]] = {}
        #: edge id -> player id
        self.roads: Dict[int, int] = {}
        self.robber: Tuple[int, int] = self._desert_coord()
        self.phase: Phase = Phase.SETUP
        self.turn: int = 1
        self.last_roll: Optional[int] = None
        self.dev_cards_drawn: int = 0
        # Cached derived values, invalidated on every mutation.
        self._derived_dirty = True

    def _desert_coord(self) -> Tuple[int, int]:
        for coord, tile in self.board.tiles.items():
            if tile.is_desert:
                return coord
        return (0, 0)  # pragma: no cover - every legal board has a desert

    def set_layout(self, layout: Layout) -> None:
        """Repaint the board. Clears the game, since node ids keep meaning
        but the resources under them do not."""
        self.board = self.board.with_layout(layout)
        self.log.clear()
        self._reset()

    # -- event application -------------------------------------------------

    def apply(self, event: Event) -> None:
        """Record and perform an event."""
        self._perform(event)
        self.log.append(event)
        self._derived_dirty = True

    def undo(self) -> bool:
        """Undo exactly one event. Returns False if the log is empty."""
        if not self.log:
            return False
        self.log.pop()
        self._replay()
        return True

    def _replay(self) -> None:
        log = list(self.log)
        self._reset()
        for event in log:
            self._perform(event)
        self.log = log
        self._derived_dirty = True

    def _perform(self, event: Event) -> None:
        handler = getattr(self, f"_do_{event.kind}", None)
        if handler is None:
            raise ValueError(f"unknown event kind: {event.kind!r}")
        handler(**event.data)  # type: ignore[misc]

    # -- individual event handlers ----------------------------------------

    def _do_roll(self, number: int) -> None:
        self.last_roll = number
        if number == 7:
            return
        for tile in self.board.tiles_by_number.get(number, ()):
            if tile.coord == self.robber or tile.resource is None:
                continue
            for node_id in tile.nodes:
                owner = self.buildings.get(node_id)
                if owner is None:
                    continue
                pid, kind = owner
                self.players[pid].hand.add(
                    tile.resource, 2 if kind is Building.CITY else 1
                )

    def _do_build_settlement(self, player: int, node: int, free: bool = False) -> None:
        self.buildings[node] = (player, Building.SETTLEMENT)
        if not free:
            self.players[player].hand.pay(COSTS["settlement"])

    def _do_build_city(self, player: int, node: int, free: bool = False) -> None:
        self.buildings[node] = (player, Building.CITY)
        if not free:
            self.players[player].hand.pay(COSTS["city"])

    def _do_build_road(self, player: int, edge: int, free: bool = False) -> None:
        self.roads[edge] = player
        if not free:
            self.players[player].hand.pay(COSTS["road"])

    def _do_move_robber(self, coord: Tuple[int, int]) -> None:
        self.robber = tuple(coord)  # type: ignore[assignment]

    def _do_steal(
        self, thief: int, victim: int, resource: Optional[str] = None
    ) -> None:
        hand = self.players[victim].hand
        if resource is not None:
            res = Resource(resource)
            if hand.take(res, 1):
                self.players[thief].hand.add(res, 1)
            return
        # Unknown card: remove the victim's most likely resource so totals stay
        # honest, and credit the thief with the same guess.
        best = max(RESOURCES, key=lambda r: hand.cards[r])
        if hand.cards[best] > 0:
            hand.take(best, 1)
            self.players[thief].hand.add(best, 1)

    def _do_discard(self, player: int, count: int, cards: Optional[str] = None) -> None:
        hand = self.players[player].hand
        if cards:
            for token in cards.split(","):
                if token:
                    hand.take(Resource(token), 1)
            return
        # Unknown discard: shed from the largest stacks first, which is both the
        # usual human choice and the least-surprising estimate.
        for _ in range(count):
            best = max(RESOURCES, key=lambda r: hand.cards[r])
            if hand.cards[best] == 0:
                break
            hand.take(best, 1)

    def _do_trade_bank(self, player: int, give: str, get: str, rate: int) -> None:
        hand = self.players[player].hand
        if hand.take(Resource(give), rate) == rate:
            hand.add(Resource(get), 1)

    def _do_trade_player(
        self, player: int, other: int, give: str, get: str
    ) -> None:
        a, b = self.players[player].hand, self.players[other].hand
        for token in give.split(","):
            if token and a.take(Resource(token), 1):
                b.add(Resource(token), 1)
        for token in get.split(","):
            if token and b.take(Resource(token), 1):
                a.add(Resource(token), 1)

    def _do_buy_dev(self, player: int, card: Optional[str] = None) -> None:
        """
        Draw a development card. ``card`` is the type when it is known — your
        own draw — and ``None`` for an opponent's face-down card.
        """
        if self.players[player].hand.pay(COSTS["dev_card"]):
            self.players[player].add_dev(DevCard(card) if card else None)
            self.dev_cards_drawn += 1

    def _do_play_dev(
        self,
        player: int,
        card: str,
        resource: Optional[str] = None,
        cards: Optional[str] = None,
    ) -> None:
        kind = DevCard(card)
        p = self.players[player]
        # An event records something that happened at the table. Spend the
        # card if we were tracking it, but apply the effect either way —
        # legality is the turn machine's job, not the log's.
        p.take_dev(kind)

        if kind is DevCard.KNIGHT:
            p.knights_played += 1
        elif kind is DevCard.VICTORY_POINT:
            p.vp_cards += 1
        elif kind is DevCard.YEAR_OF_PLENTY:
            for token in (cards or "").split(","):
                if token:
                    p.hand.add(Resource(token), 1)
        elif kind is DevCard.MONOPOLY:
            if resource:
                res = Resource(resource)
                taken = 0
                for other, victim in self.players.items():
                    if other != player:
                        taken += victim.hand.take(res, victim.hand.cards[res])
                p.hand.add(res, taken)
        # ROAD_BUILDING grants two free roads, recorded as ordinary road
        # events by the turn machine so the board and supply stay honest.

    def _do_play_knight(self, player: int) -> None:
        """Kept so older logs still replay."""
        self._do_play_dev(player, DevCard.KNIGHT.value)

    def _do_reveal_vp(self, player: int) -> None:
        """Kept so older logs still replay."""
        self._do_play_dev(player, DevCard.VICTORY_POINT.value)

    def _do_setup_collect(self, player: int, node: int) -> None:
        """
        The second setup settlement pays out its surrounding tiles.

        Recorded as its own event so a replay reproduces the opening hand
        exactly, instead of leaving the player to type it in.
        """
        for tile in self.board.node_tiles[node]:
            if tile.resource is not None:
                self.players[player].hand.add(tile.resource, 1)

    def _do_skip_steal(self) -> None:
        """The robber moved but nobody was robbed. Recorded so the turn
        machine knows the steal step is resolved rather than pending."""

    def _do_end_turn(self) -> None:
        self.turn = self.turn % self.num_players + 1

    def _do_adjust(self, player: int, resource: str, delta: int) -> None:
        hand = self.players[player].hand
        res = Resource(resource)
        if delta >= 0:
            hand.add(res, delta)
        else:
            hand.take(res, -delta)

    def _do_set_phase(self, phase: str) -> None:
        self.phase = Phase(phase)

    # -- convenience constructors -----------------------------------------

    def roll(self, number: int) -> None:
        self.apply(Event.make("roll", number=number))

    def build_settlement(self, player: int, node: int, free: bool = False) -> None:
        self.apply(
            Event.make("build_settlement", player=player, node=node, free=free)
        )

    def build_city(self, player: int, node: int, free: bool = False) -> None:
        self.apply(Event.make("build_city", player=player, node=node, free=free))

    def build_road(self, player: int, edge: int, free: bool = False) -> None:
        self.apply(Event.make("build_road", player=player, edge=edge, free=free))

    def move_robber(self, coord: Tuple[int, int]) -> None:
        self.apply(Event.make("move_robber", coord=tuple(coord)))

    def steal(self, thief: int, victim: int, resource: Optional[Resource] = None) -> None:
        self.apply(
            Event.make(
                "steal", thief=thief, victim=victim,
                resource=resource.value if resource else None,
            )
        )

    def set_phase(self, phase: Phase) -> None:
        self.apply(Event.make("set_phase", phase=phase.value))

    def adjust(self, player: int, resource: Resource, delta: int) -> None:
        self.apply(
            Event.make("adjust", player=player, resource=resource.value, delta=delta)
        )

    # -- queries -----------------------------------------------------------

    def owner_of(self, node_id: int) -> Optional[int]:
        entry = self.buildings.get(node_id)
        return None if entry is None else entry[0]

    def building_at(self, node_id: int) -> Optional[Building]:
        entry = self.buildings.get(node_id)
        return None if entry is None else entry[1]

    def road_owner(self, edge_id: int) -> Optional[int]:
        return self.roads.get(edge_id)

    def nodes_of(self, player: int) -> List[int]:
        return [n for n, (p, _) in self.buildings.items() if p == player]

    def settlements_of(self, player: int) -> List[int]:
        return [
            n for n, (p, k) in self.buildings.items()
            if p == player and k is Building.SETTLEMENT
        ]

    def cities_of(self, player: int) -> List[int]:
        return [
            n for n, (p, k) in self.buildings.items()
            if p == player and k is Building.CITY
        ]

    def edges_of(self, player: int) -> List[int]:
        return [e for e, p in self.roads.items() if p == player]

    def remaining(self, player: int, item: str) -> int:
        """How many of ``item`` the player still has in their supply."""
        if item == "settlement":
            return SUPPLY["settlement"] - len(self.settlements_of(player))
        if item == "city":
            return SUPPLY["city"] - len(self.cities_of(player))
        if item == "road":
            return SUPPLY["road"] - len(self.edges_of(player))
        return 0

    def ports_of(self, player: int) -> Dict[Optional[Resource], int]:
        """Best trade rate this player has for each resource (4 if no port)."""
        rates: Dict[Optional[Resource], int] = {r: 4 for r in RESOURCES}
        for node_id in self.nodes_of(player):
            port = self.board.node(node_id).port
            if port is None:
                continue
            if port is Port.GENERIC:
                for r in RESOURCES:
                    rates[r] = min(rates[r], 3)
            else:
                res = port.resource
                assert res is not None
                rates[res] = min(rates[res], 2)
        return rates

    def dev_deck_left(self) -> int:
        return max(0, DEV_DECK_SIZE - self.dev_cards_drawn)

    def opponents(self, player: Optional[int] = None) -> List[int]:
        player = self.me if player is None else player
        return [p for p in self.players if p != player]

    def snapshot(self) -> Dict[str, object]:
        """A plain-data view, handy for tests and debugging."""
        return {
            "phase": self.phase.value,
            "turn": self.turn,
            "robber": self.robber,
            "buildings": dict(self.buildings),
            "roads": dict(self.roads),
            "hands": {
                pid: {r.value: n for r, n in p.hand.cards.items() if n}
                for pid, p in self.players.items()
            },
        }
