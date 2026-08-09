"""
Catan rules: legality, longest road, largest army, victory points.

Every function here is a pure query over a :class:`~catanmind.state.GameState`.
Nothing in this module mutates anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from catanmind.board import Building, COSTS, Resource, SUPPLY
from catanmind.state import GameState

#: A player needs at least this many connected roads before Longest Road is
#: awarded at all.
LONGEST_ROAD_MINIMUM = 5

#: Knights required before Largest Army is awarded.
LARGEST_ARMY_MINIMUM = 3


@dataclass(frozen=True)
class Legality:
    ok: bool
    reason: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok


OK = Legality(True)


def _no(reason: str) -> Legality:
    return Legality(False, reason)


# --------------------------------------------------------------------------
# Settlements
# --------------------------------------------------------------------------


def can_place_settlement(
    state: GameState, player: int, node_id: int, *, setup: bool = False
) -> Legality:
    """
    Check the four settlement rules: the node exists, is free, respects the
    distance rule, and (outside setup) touches the player's road network.
    """
    if not 0 <= node_id < len(state.board.nodes):
        return _no(f"No intersection #{node_id}")

    if node_id in state.buildings:
        owner = state.buildings[node_id][0]
        return _no(f"#{node_id} is already taken by Player {owner}")

    for neighbour in state.board.node(node_id).neighbors:
        if neighbour in state.buildings:
            return _no(f"Distance rule: #{neighbour} next door is built on")

    if state.remaining(player, "settlement") <= 0:
        return _no(f"Player {player} has no settlements left ({SUPPLY['settlement']} max)")

    if not setup:
        touches_road = any(
            state.roads.get(eid) == player
            for eid in state.board.node_edges[node_id]
        )
        if not touches_road:
            return _no("Must connect to your own road")

    return OK


def can_upgrade_city(state: GameState, player: int, node_id: int) -> Legality:
    entry = state.buildings.get(node_id)
    if entry is None:
        return _no(f"Nothing built at #{node_id}")
    owner, kind = entry
    if owner != player:
        return _no(f"#{node_id} belongs to Player {owner}")
    if kind is Building.CITY:
        return _no(f"#{node_id} is already a city")
    if state.remaining(player, "city") <= 0:
        return _no(f"Player {player} has no cities left ({SUPPLY['city']} max)")
    return OK


# --------------------------------------------------------------------------
# Roads
# --------------------------------------------------------------------------


def can_place_road(state: GameState, player: int, edge_id: int) -> Legality:
    """
    A road is legal when the edge is free and at least one of its endpoints is
    a node the player can build *out of*: their own building, or a node their
    road already reaches that is not occupied by an opponent.
    """
    if not 0 <= edge_id < len(state.board.edges):
        return _no(f"No path #{edge_id}")
    if edge_id in state.roads:
        return _no(f"Path #{edge_id} already has a road")
    if state.remaining(player, "road") <= 0:
        return _no(f"Player {player} has no roads left ({SUPPLY['road']} max)")

    edge = state.board.edge(edge_id)
    for endpoint in (edge.a, edge.b):
        entry = state.buildings.get(endpoint)
        if entry is not None:
            if entry[0] == player:
                return OK
            # An opponent's building blocks the network *through* this node.
            continue
        touching = any(
            state.roads.get(eid) == player
            for eid in state.board.node_edges[endpoint]
            if eid != edge_id
        )
        if touching:
            return OK

    return _no("Must extend your own road network")


def can_place_setup_road(
    state: GameState, player: int, edge_id: int, settlement_node: int
) -> Legality:
    """During setup a road must touch the settlement just placed."""
    if not 0 <= edge_id < len(state.board.edges):
        return _no(f"No path #{edge_id}")
    if edge_id in state.roads:
        return _no(f"Path #{edge_id} already has a road")
    edge = state.board.edge(edge_id)
    if settlement_node not in (edge.a, edge.b):
        return _no(f"Setup road must touch your new settlement #{settlement_node}")
    return OK


# --------------------------------------------------------------------------
# Legal-move enumeration
# --------------------------------------------------------------------------


def legal_settlements(
    state: GameState, player: int, *, setup: bool = False
) -> List[int]:
    return [
        n.id for n in state.board.nodes
        if can_place_settlement(state, player, n.id, setup=setup)
    ]


def legal_cities(state: GameState, player: int) -> List[int]:
    return [
        n for n in state.settlements_of(player)
        if can_upgrade_city(state, player, n)
    ]


def legal_roads(state: GameState, player: int) -> List[int]:
    return [
        e.id for e in state.board.edges if can_place_road(state, player, e.id)
    ]


def can_afford(state: GameState, player: int, item: str) -> Legality:
    cost = COSTS.get(item)
    if cost is None:
        return _no(f"Unknown item {item!r}")
    hand = state.players[player].hand
    missing = [
        f"{n - hand.cards[r]} {r.value}" for r, n in cost.items()
        if hand.cards[r] < n
    ]
    if missing:
        return _no("Short " + ", ".join(missing))
    return OK


# --------------------------------------------------------------------------
# Longest road
# --------------------------------------------------------------------------


def longest_road(state: GameState, player: int) -> int:
    """
    Length of the player's longest continuous road.

    This is the longest *trail* in the player's road subgraph: no road may be
    used twice, but the route may revisit an intersection. An opponent's
    building breaks the chain — a route may end on it but not pass through it.

    The naive approach of walking the line graph (treating two roads as
    connected whenever they share any endpoint) overcounts: three roads meeting
    at one intersection form a triangle in the line graph and report length 3,
    when the real answer is 2. This walks the original graph instead, tracking
    which node we are standing on.
    """
    owned = set(state.edges_of(player))
    if not owned:
        return 0

    board = state.board
    # node -> the player's roads leaving it
    out: Dict[int, List[int]] = {}
    for eid in owned:
        edge = board.edge(eid)
        out.setdefault(edge.a, []).append(eid)
        out.setdefault(edge.b, []).append(eid)

    def blocked(node_id: int) -> bool:
        entry = state.buildings.get(node_id)
        return entry is not None and entry[0] != player

    best = 0

    def walk(node: int, used: Set[int], length: int) -> None:
        nonlocal best
        if length > best:
            best = length
        if blocked(node):
            return  # may arrive here, may not continue through
        for eid in out.get(node, ()):
            if eid in used:
                continue
            used.add(eid)
            walk(board.edge(eid).other(node), used, length + 1)
            used.remove(eid)

    for start in out:
        walk(start, set(), 0)

    return best


def longest_road_holder(state: GameState) -> Optional[int]:
    """
    Who holds Longest Road, or ``None``.

    The card is *sticky*: whoever takes it keeps it until another player
    strictly beats their length, so a tie does not hand it over. That is
    history rather than a property of the current position, which is why the
    answer is maintained by :class:`~catanmind.state.GameState` as roads are
    built and read back here.
    """
    return state.longest_road_holder


def largest_army_holder(state: GameState) -> Optional[int]:
    """Who holds Largest Army. Sticky, exactly like Longest Road."""
    return state.largest_army_holder


# --------------------------------------------------------------------------
# Victory points
# --------------------------------------------------------------------------


def victory_points(
    state: GameState, player: int, *, include_hidden: bool = True
) -> int:
    """
    Victory points for ``player``.

    ``include_hidden=False`` gives the *public* score — what opponents can see —
    by leaving out unrevealed victory-point cards.
    """
    vp = 0
    for node_id, (owner, kind) in state.buildings.items():
        if owner == player:
            vp += 2 if kind is Building.CITY else 1
    if longest_road_holder(state) == player:
        vp += 2
    if largest_army_holder(state) == player:
        vp += 2
    if include_hidden:
        vp += state.players[player].vp_cards
    return vp


def scores(state: GameState, *, include_hidden: bool = True) -> Dict[int, int]:
    return {
        p: victory_points(state, p, include_hidden=include_hidden)
        for p in state.players
    }


def winner(state: GameState) -> Optional[int]:
    for p, vp in scores(state).items():
        if vp >= state.target_vp:
            return p
    return None


def sync_awards(state: GameState) -> None:
    """Write the current Longest Road / Largest Army holders onto the players."""
    road = longest_road_holder(state)
    army = largest_army_holder(state)
    for pid, p in state.players.items():
        p.has_longest_road = pid == road
        p.has_largest_army = pid == army


# --------------------------------------------------------------------------
# Production
# --------------------------------------------------------------------------


def expected_yield(
    state: GameState, player: int, *, ignore_robber: bool = False
) -> Dict[Resource, float]:
    """
    Expected cards per roll, by resource, from everything ``player`` owns.

    Cities count double. The tile under the robber contributes nothing unless
    ``ignore_robber`` is set (useful when comparing spots in the abstract).
    """
    from catanmind.board import probability

    out = {r: 0.0 for r in Resource}
    for node_id, (owner, kind) in state.buildings.items():
        if owner != player:
            continue
        multiplier = 2 if kind is Building.CITY else 1
        for tile in state.board.node_tiles[node_id]:
            if tile.resource is None:
                continue
            if not ignore_robber and tile.coord == state.robber:
                continue
            out[tile.resource] += probability(tile.number) * multiplier
    return out


def node_yield(
    state: GameState, node_id: int, *, as_city: bool = False,
    ignore_robber: bool = False,
) -> Dict[Resource, float]:
    """Expected cards per roll a single node would produce."""
    from catanmind.board import probability

    multiplier = 2 if as_city else 1
    out = {r: 0.0 for r in Resource}
    for tile in state.board.node_tiles[node_id]:
        if tile.resource is None:
            continue
        if not ignore_robber and tile.coord == state.robber:
            continue
        out[tile.resource] += probability(tile.number) * multiplier
    return out


def must_discard(state: GameState, player: int) -> int:
    """Cards this player must discard on a 7 (half, rounded down, over 7)."""
    total = state.players[player].hand.total()
    return total // 2 if total > 7 else 0


def setup_placements_done(state: GameState, player: int) -> int:
    """How many setup settlements this player has placed (0, 1 or 2)."""
    return min(2, len(state.nodes_of(player)))
