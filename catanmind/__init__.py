"""CatanMind — a Catan advisor.

The engine is split into layers, each usable without the one above it:

``board``    geometry and graph; immutable topology plus the painted layout
``state``    every mutable fact about a game, and the event log that produced it
``rules``    legality, longest road, largest army, victory points
``scoring``  how good a spot is
``advisor``  what to do next
``tracker``  what the opponents are probably holding
"""

from catanmind.board import (
    Board,
    Building,
    COSTS,
    Layout,
    Node,
    Edge,
    Port,
    Resource,
    RESOURCES,
    Tile,
    pips,
    probability,
)

__all__ = [
    "Board", "Building", "COSTS", "Layout", "Node", "Edge", "Port",
    "Resource", "RESOURCES", "Tile", "pips", "probability",
]
