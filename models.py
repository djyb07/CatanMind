"""
CatanMind - Core Data Models
Graph-based representation of the Catan board using axial coordinates.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
import math


class ResourceType(Enum):
    """Resource types available in Catan."""
    WOOD = "wood"
    BRICK = "brick"
    SHEEP = "sheep"
    WHEAT = "wheat"
    ORE = "ore"
    DESERT = "desert"


class BuildingType(Enum):
    """Building types that can be placed on intersections."""
    SETTLEMENT = "settlement"
    CITY = "city"


class PortType(Enum):
    """Port types for trading."""
    GENERIC = "3:1"
    WOOD = "wood_2:1"
    BRICK = "brick_2:1"
    SHEEP = "sheep_2:1"
    WHEAT = "wheat_2:1"
    ORE = "ore_2:1"


# Dice roll probabilities (dots/pips)
DICE_PROBABILITIES = {
    2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
    7: 0,  # Robber - no production
    8: 5, 9: 4, 10: 3, 11: 2, 12: 1
}


@dataclass
class Tile:
    """
    Represents a hexagonal tile on the board.
    Uses axial coordinates (q, r) for positioning.
    """
    q: int  # Axial coordinate q
    r: int  # Axial coordinate r
    resource_type: ResourceType
    dice_number: int  # 2-12, or 0 for desert
    has_robber: bool = False
    
    @property
    def coordinates(self) -> Tuple[int, int]:
        return (self.q, self.r)
    
    @property
    def probability(self) -> int:
        """Returns the pip count (dots) for this tile's number."""
        return DICE_PROBABILITIES.get(self.dice_number, 0)
    
    def cube_coordinates(self) -> Tuple[int, int, int]:
        """Convert axial to cube coordinates for distance calculations."""
        x = self.q
        z = self.r
        y = -x - z
        return (x, y, z)


@dataclass
class Intersection:
    """
    Represents a node in the board graph (corner of hexes).
    Settlements and cities are placed on intersections.
    """
    id: int
    neighbors: List[int] = field(default_factory=list)  # Adjacent intersection IDs
    touching_tiles: List[Tile] = field(default_factory=list)  # Up to 3 tiles
    owner: Optional[int] = None  # Player ID (1-4) or None
    building_type: Optional[BuildingType] = None
    port_type: Optional[PortType] = None
    
    # Pixel position for rendering
    x: float = 0.0
    y: float = 0.0
    
    def is_buildable(self) -> bool:
        """Check if this intersection can have a building placed."""
        return self.owner is None
    
    def get_resources(self) -> List[ResourceType]:
        """Get list of resources this intersection produces."""
        return [t.resource_type for t in self.touching_tiles 
                if t.resource_type != ResourceType.DESERT]
    
    def get_total_probability(self) -> int:
        """Get sum of pip probabilities from all touching tiles."""
        return sum(t.probability for t in self.touching_tiles)


@dataclass
class Path:
    """
    Represents an edge in the board graph (side of hexes).
    Roads are placed on paths.
    """
    node_a: int  # Intersection ID
    node_b: int  # Intersection ID
    owner: Optional[int] = None  # Player ID or None
    
    def __hash__(self):
        # Ensure (a, b) and (b, a) hash the same
        return hash(tuple(sorted([self.node_a, self.node_b])))
    
    def __eq__(self, other):
        if not isinstance(other, Path):
            return False
        return set([self.node_a, self.node_b]) == set([other.node_a, other.node_b])
    
    def connects(self, node_id: int) -> bool:
        """Check if this path connects to a given node."""
        return node_id in (self.node_a, self.node_b)
    
    def other_node(self, node_id: int) -> int:
        """Get the other node connected by this path."""
        if node_id == self.node_a:
            return self.node_b
        return self.node_a


class Board:
    """
    Complete Catan board representation.
    Standard layout: 19 hexes, 54 intersections, 72 paths.
    """
    
    # Standard Catan hex layout in axial coordinates
    # Center hex is (0, 0)
    STANDARD_HEX_POSITIONS = [
        # Center
        (0, 0),
        # Inner ring
        (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1),
        # Outer ring
        (2, -2), (2, -1), (2, 0), (1, 1), (0, 2), (-1, 2),
        (-2, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2)
    ]
    
    # Default resource distribution (standard setup)
    STANDARD_RESOURCES = [
        ResourceType.ORE, ResourceType.SHEEP, ResourceType.WOOD,
        ResourceType.WHEAT, ResourceType.BRICK, ResourceType.SHEEP,
        ResourceType.BRICK, ResourceType.WHEAT, ResourceType.WOOD,
        ResourceType.DESERT, ResourceType.WOOD, ResourceType.ORE,
        ResourceType.WOOD, ResourceType.ORE, ResourceType.WHEAT,
        ResourceType.SHEEP, ResourceType.BRICK, ResourceType.WHEAT,
        ResourceType.SHEEP
    ]
    
    # Standard dice number placement (spiral from edge)
    STANDARD_NUMBERS = [
        10, 2, 9, 12, 6, 4, 10, 9, 11,
        0,  # Desert
        3, 8, 8, 3, 4, 5, 5, 6, 11
    ]
    
    def __init__(self):
        self.tiles: Dict[Tuple[int, int], Tile] = {}
        self.intersections: Dict[int, Intersection] = {}
        self.paths: Set[Path] = set()
        self.players: Dict[int, 'Player'] = {}
        
        self._build_tiles()
        self._build_graph()
        self._assign_ports()
    
    def _build_tiles(self):
        """Create all 19 hex tiles."""
        for i, (q, r) in enumerate(self.STANDARD_HEX_POSITIONS):
            self.tiles[(q, r)] = Tile(
                q=q,
                r=r,
                resource_type=self.STANDARD_RESOURCES[i],
                dice_number=self.STANDARD_NUMBERS[i],
                has_robber=(self.STANDARD_RESOURCES[i] == ResourceType.DESERT)
            )
    
    def _hex_to_pixel(self, q: int, r: int, size: float = 50.0) -> Tuple[float, float]:
        """Convert axial coordinates to pixel position."""
        x = size * (3/2 * q)
        y = size * (math.sqrt(3)/2 * q + math.sqrt(3) * r)
        return (x, y)
    
    def _get_hex_corners(self, q: int, r: int, size: float = 50.0) -> List[Tuple[float, float]]:
        """Get the 6 corner positions of a hex."""
        cx, cy = self._hex_to_pixel(q, r, size)
        corners = []
        for i in range(6):
            angle = math.pi / 3 * i - math.pi / 6
            corner_x = cx + size * math.cos(angle)
            corner_y = cy + size * math.sin(angle)
            corners.append((round(corner_x, 2), round(corner_y, 2)))
        return corners
    
    def _build_graph(self):
        """
        Build the intersection graph by finding unique corners.
        Each hex has 6 corners; shared corners become single intersections.
        """
        # Collect all corner positions and map to intersection IDs
        corner_to_id: Dict[Tuple[float, float], int] = {}
        next_id = 0
        
        # For each tile, get its corners
        tile_corners: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        for (q, r), tile in self.tiles.items():
            corners = self._get_hex_corners(q, r)
            tile_corners[(q, r)] = corners
            
            for corner in corners:
                # Round to handle floating point comparison
                key = (round(corner[0], 1), round(corner[1], 1))
                if key not in corner_to_id:
                    corner_to_id[key] = next_id
                    self.intersections[next_id] = Intersection(
                        id=next_id,
                        x=corner[0],
                        y=corner[1]
                    )
                    next_id += 1
        
        # Assign touching tiles to each intersection
        for (q, r), corners in tile_corners.items():
            tile = self.tiles[(q, r)]
            for corner in corners:
                key = (round(corner[0], 1), round(corner[1], 1))
                node_id = corner_to_id[key]
                if tile not in self.intersections[node_id].touching_tiles:
                    self.intersections[node_id].touching_tiles.append(tile)
        
        # Build neighbor connections (edges of hexes)
        for (q, r), corners in tile_corners.items():
            for i in range(6):
                corner_a = corners[i]
                corner_b = corners[(i + 1) % 6]
                
                key_a = (round(corner_a[0], 1), round(corner_a[1], 1))
                key_b = (round(corner_b[0], 1), round(corner_b[1], 1))
                
                id_a = corner_to_id[key_a]
                id_b = corner_to_id[key_b]
                
                # Add neighbor relationship
                if id_b not in self.intersections[id_a].neighbors:
                    self.intersections[id_a].neighbors.append(id_b)
                if id_a not in self.intersections[id_b].neighbors:
                    self.intersections[id_b].neighbors.append(id_a)
                
                # Add path
                path = Path(node_a=id_a, node_b=id_b)
                self.paths.add(path)
    
    def _assign_ports(self):
        """Assign ports to edge intersections."""
        # Find edge intersections (those with fewer than 3 touching tiles)
        edge_nodes = [n for n in self.intersections.values() 
                      if len(n.touching_tiles) < 3]
        
        # Standard port distribution
        port_types = [
            PortType.GENERIC, PortType.WHEAT, PortType.ORE,
            PortType.GENERIC, PortType.SHEEP, PortType.GENERIC,
            PortType.GENERIC, PortType.BRICK, PortType.WOOD
        ]
        
        # Assign ports to pairs of adjacent edge nodes
        assigned = set()
        port_idx = 0
        
        for node in sorted(edge_nodes, key=lambda n: (n.y, n.x)):
            if node.id in assigned or port_idx >= len(port_types):
                continue
                
            # Find adjacent edge node
            for neighbor_id in node.neighbors:
                neighbor = self.intersections[neighbor_id]
                if len(neighbor.touching_tiles) < 3 and neighbor_id not in assigned:
                    # Assign port to both
                    node.port_type = port_types[port_idx]
                    neighbor.port_type = port_types[port_idx]
                    assigned.add(node.id)
                    assigned.add(neighbor_id)
                    port_idx += 1
                    break
    
    def get_path(self, node_a: int, node_b: int) -> Optional[Path]:
        """Get the path between two nodes if it exists."""
        for path in self.paths:
            if set([path.node_a, path.node_b]) == set([node_a, node_b]):
                return path
        return None
    
    def get_valid_settlement_spots(self, player_id: Optional[int] = None, 
                                    initial_phase: bool = False) -> List[int]:
        """
        Get all valid spots for placing a settlement.
        Respects distance rule and connectivity (unless initial phase).
        """
        valid = []
        for node_id, node in self.intersections.items():
            if not node.is_buildable():
                continue
            
            # Distance rule: no adjacent settlements
            has_adjacent_building = False
            for neighbor_id in node.neighbors:
                if self.intersections[neighbor_id].owner is not None:
                    has_adjacent_building = True
                    break
            
            if has_adjacent_building:
                continue
            
            # Connectivity check (skip for initial phase)
            if not initial_phase and player_id is not None:
                has_connected_road = False
                for neighbor_id in node.neighbors:
                    path = self.get_path(node_id, neighbor_id)
                    if path and path.owner == player_id:
                        has_connected_road = True
                        break
                if not has_connected_road:
                    continue
            
            valid.append(node_id)
        
        return valid
    
    def get_valid_road_spots(self, player_id: int) -> List[Tuple[int, int]]:
        """Get all valid spots for placing a road."""
        valid = []
        
        # Find all nodes owned by player or connected by player's roads
        player_nodes = set()
        for node in self.intersections.values():
            if node.owner == player_id:
                player_nodes.add(node.id)
        
        for path in self.paths:
            if path.owner == player_id:
                player_nodes.add(path.node_a)
                player_nodes.add(path.node_b)
        
        # Find unbuilt paths adjacent to player's network
        for path in self.paths:
            if path.owner is not None:
                continue
            
            # Check if either endpoint connects to player's network
            a_connects = path.node_a in player_nodes
            b_connects = path.node_b in player_nodes
            
            # Also check if blocked by opponent's settlement
            a_blocked = (self.intersections[path.node_a].owner is not None and
                        self.intersections[path.node_a].owner != player_id)
            b_blocked = (self.intersections[path.node_b].owner is not None and
                        self.intersections[path.node_b].owner != player_id)
            
            if (a_connects and not a_blocked) or (b_connects and not b_blocked):
                valid.append((path.node_a, path.node_b))
        
        return valid
    
    def place_settlement(self, node_id: int, player_id: int) -> bool:
        """Place a settlement at the given intersection."""
        if node_id not in self.intersections:
            return False
        
        node = self.intersections[node_id]
        if not node.is_buildable():
            return False
        
        node.owner = player_id
        node.building_type = BuildingType.SETTLEMENT
        return True
    
    def place_road(self, node_a: int, node_b: int, player_id: int) -> bool:
        """Place a road on the path between two nodes."""
        path = self.get_path(node_a, node_b)
        if path is None or path.owner is not None:
            return False
        
        path.owner = player_id
        return True
    
    def upgrade_to_city(self, node_id: int, player_id: int) -> bool:
        """Upgrade a settlement to a city."""
        if node_id not in self.intersections:
            return False
        
        node = self.intersections[node_id]
        if node.owner != player_id or node.building_type != BuildingType.SETTLEMENT:
            return False
        
        node.building_type = BuildingType.CITY
        return True
    
    def get_player_buildings(self, player_id: int) -> List[Intersection]:
        """Get all buildings owned by a player."""
        return [n for n in self.intersections.values() if n.owner == player_id]
    
    def get_player_roads(self, player_id: int) -> List[Path]:
        """Get all roads owned by a player."""
        return [p for p in self.paths if p.owner == player_id]
    
    def calculate_longest_road(self, player_id: int) -> int:
        """Calculate the longest continuous road for a player using DFS."""
        player_roads = self.get_player_roads(player_id)
        if not player_roads:
            return 0
        
        # Build adjacency for road segments
        def get_connected_roads(road: Path) -> List[Path]:
            connected = []
            for other_road in player_roads:
                if other_road == road:
                    continue
                # Check if roads share an endpoint not blocked by enemy
                shared_nodes = set([road.node_a, road.node_b]) & set([other_road.node_a, other_road.node_b])
                for shared in shared_nodes:
                    node = self.intersections[shared]
                    # Can pass through own settlement or empty intersection
                    if node.owner is None or node.owner == player_id:
                        connected.append(other_road)
                        break
            return connected
        
        def dfs(road: Path, visited: Set[Path]) -> int:
            visited.add(road)
            max_length = 1
            
            for next_road in get_connected_roads(road):
                if next_road not in visited:
                    length = 1 + dfs(next_road, visited.copy())
                    max_length = max(max_length, length)
            
            return max_length
        
        longest = 0
        for road in player_roads:
            length = dfs(road, set())
            longest = max(longest, length)
        
        return longest


@dataclass
class Player:
    """Represents a player in the game."""
    id: int
    name: str
    color: str
    victory_points: int = 0
    resources: Dict[ResourceType, int] = field(default_factory=lambda: {
        ResourceType.WOOD: 0,
        ResourceType.BRICK: 0,
        ResourceType.SHEEP: 0,
        ResourceType.WHEAT: 0,
        ResourceType.ORE: 0
    })
    development_cards: int = 0
    knights_played: int = 0
    has_longest_road: bool = False
    has_largest_army: bool = False
    
    def get_total_resources(self) -> int:
        """Get total number of resource cards."""
        return sum(self.resources.values())
    
    def can_afford(self, cost: Dict[ResourceType, int]) -> bool:
        """Check if player can afford a given cost."""
        for resource, amount in cost.items():
            if self.resources.get(resource, 0) < amount:
                return False
        return True
    
    def calculate_vp(self, board: Board) -> int:
        """Calculate current victory points."""
        vp = 0
        
        # Buildings
        for node in board.get_player_buildings(self.id):
            if node.building_type == BuildingType.SETTLEMENT:
                vp += 1
            elif node.building_type == BuildingType.CITY:
                vp += 2
        
        # Longest Road (2 VP)
        if self.has_longest_road:
            vp += 2
        
        # Largest Army (2 VP)  
        if self.has_largest_army:
            vp += 2
        
        # VP from dev cards would be tracked separately
        
        self.victory_points = vp
        return vp


# Building costs
BUILDING_COSTS = {
    "road": {ResourceType.WOOD: 1, ResourceType.BRICK: 1},
    "settlement": {ResourceType.WOOD: 1, ResourceType.BRICK: 1, 
                   ResourceType.SHEEP: 1, ResourceType.WHEAT: 1},
    "city": {ResourceType.WHEAT: 2, ResourceType.ORE: 3},
    "development_card": {ResourceType.SHEEP: 1, ResourceType.WHEAT: 1, ResourceType.ORE: 1}
}
