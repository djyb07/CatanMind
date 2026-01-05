"""
CatanMind - Game Rule Validators
Strict enforcement of Catan rules.
"""

from typing import List, Tuple, Optional
from models import Board, Player, BuildingType


class ValidationError(Exception):
    """Exception raised when a game rule is violated."""
    def __init__(self, message: str, rule: str):
        self.message = message
        self.rule = rule
        super().__init__(f"{rule}: {message}")


class Validator:
    """
    Validates all game actions against official Catan rules.
    """
    
    def __init__(self, board: Board):
        self.board = board
    
    def validate_settlement_placement(self, node_id: int, player_id: int,
                                       initial_phase: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Validate settlement placement.
        
        Rules:
        1. Node must exist
        2. Node must be unoccupied
        3. Distance Rule: No adjacent settlements
        4. Connectivity: Must connect to player's road (except initial phase)
        
        Returns:
            (is_valid, error_message)
        """
        # Rule 1: Node must exist
        if node_id not in self.board.intersections:
            return False, f"Invalid intersection ID: {node_id}"
        
        node = self.board.intersections[node_id]
        
        # Rule 2: Node must be unoccupied
        if node.owner is not None:
            return False, f"Intersection {node_id} is already occupied"
        
        # Rule 3: Distance Rule
        for neighbor_id in node.neighbors:
            neighbor = self.board.intersections.get(neighbor_id)
            if neighbor and neighbor.owner is not None:
                return False, f"Distance Rule: Adjacent intersection {neighbor_id} has a building"
        
        # Rule 4: Connectivity (skip for initial phase)
        if not initial_phase:
            has_connected_road = False
            for neighbor_id in node.neighbors:
                path = self.board.get_path(node_id, neighbor_id)
                if path and path.owner == player_id:
                    has_connected_road = True
                    break
            
            if not has_connected_road:
                return False, "Connectivity: Settlement must connect to your road network"
        
        return True, None
    
    def validate_city_upgrade(self, node_id: int, player_id: int) -> Tuple[bool, Optional[str]]:
        """
        Validate city upgrade.
        
        Rules:
        1. Node must have player's settlement
        2. Cannot upgrade a city (already upgraded)
        
        Returns:
            (is_valid, error_message)
        """
        if node_id not in self.board.intersections:
            return False, f"Invalid intersection ID: {node_id}"
        
        node = self.board.intersections[node_id]
        
        # Rule 1: Must own the settlement
        if node.owner != player_id:
            return False, f"You don't own the building at {node_id}"
        
        # Rule 2: Must be a settlement
        if node.building_type != BuildingType.SETTLEMENT:
            return False, f"Intersection {node_id} is not a settlement (already a city?)"
        
        return True, None
    
    def validate_road_placement(self, node_a: int, node_b: int,
                                 player_id: int) -> Tuple[bool, Optional[str]]:
        """
        Validate road placement.
        
        Rules:
        1. Path must exist between nodes
        2. Path must be unoccupied
        3. Road Continuity: Must connect to player's road/building
        4. Cannot pass through enemy settlement
        
        Returns:
            (is_valid, error_message)
        """
        # Rule 1: Path must exist
        path = self.board.get_path(node_a, node_b)
        if path is None:
            return False, f"No path exists between {node_a} and {node_b}"
        
        # Rule 2: Path must be unoccupied
        if path.owner is not None:
            return False, f"Path {node_a}-{node_b} already has a road"
        
        # Rule 3: Road Continuity
        connects_to_network = False
        
        for endpoint in [node_a, node_b]:
            node = self.board.intersections[endpoint]
            
            # Check if player has building here
            if node.owner == player_id:
                connects_to_network = True
                break
            
            # Check if player has road connecting here
            for neighbor_id in node.neighbors:
                if neighbor_id in [node_a, node_b]:
                    continue  # Skip the path we're trying to build
                
                adjacent_path = self.board.get_path(endpoint, neighbor_id)
                if adjacent_path and adjacent_path.owner == player_id:
                    connects_to_network = True
                    break
            
            if connects_to_network:
                break
        
        if not connects_to_network:
            return False, "Road Continuity: Road must connect to your network"
        
        # Rule 4: Cannot pass through enemy settlement
        for endpoint in [node_a, node_b]:
            node = self.board.intersections[endpoint]
            if node.owner is not None and node.owner != player_id:
                # Check if this blocks the connection
                # A road can still be built if ONE endpoint is blocked,
                # but the connecting endpoint must be valid
                other_endpoint = node_b if endpoint == node_a else node_a
                other_node = self.board.intersections[other_endpoint]
                
                # If connecting from the blocked endpoint, it's invalid
                has_road_from_blocked = False
                for neighbor_id in node.neighbors:
                    if neighbor_id == other_endpoint:
                        continue
                    adj_path = self.board.get_path(endpoint, neighbor_id)
                    if adj_path and adj_path.owner == player_id:
                        has_road_from_blocked = True
                        break
                
                if has_road_from_blocked:
                    return False, f"Cannot build through enemy settlement at {endpoint}"
        
        return True, None
    
    def validate_initial_road(self, settlement_id: int, node_a: int, node_b: int,
                               player_id: int) -> Tuple[bool, Optional[str]]:
        """
        Validate initial phase road placement.
        
        Rules:
        1. Road must connect to the settlement just placed
        2. Standard road placement rules apply
        
        Returns:
            (is_valid, error_message)
        """
        # Check connection to settlement
        if settlement_id not in [node_a, node_b]:
            return False, "Initial road must connect to your settlement"
        
        # Path must exist
        path = self.board.get_path(node_a, node_b)
        if path is None:
            return False, f"No path exists between {node_a} and {node_b}"
        
        if path.owner is not None:
            return False, f"Path {node_a}-{node_b} already has a road"
        
        return True, None
    
    def get_valid_settlement_spots(self, player_id: int,
                                    initial_phase: bool = False) -> List[int]:
        """Get all valid spots for settlement placement."""
        valid = []
        for node_id in self.board.intersections:
            is_valid, _ = self.validate_settlement_placement(
                node_id, player_id, initial_phase
            )
            if is_valid:
                valid.append(node_id)
        return valid
    
    def get_valid_road_spots(self, player_id: int) -> List[Tuple[int, int]]:
        """Get all valid spots for road placement."""
        valid = []
        for path in self.board.paths:
            is_valid, _ = self.validate_road_placement(
                path.node_a, path.node_b, player_id
            )
            if is_valid:
                valid.append((path.node_a, path.node_b))
        return valid
    
    def can_afford(self, player: Player, building_type: str) -> Tuple[bool, Optional[str]]:
        """Check if player can afford a building."""
        from models import BUILDING_COSTS
        
        if building_type not in BUILDING_COSTS:
            return False, f"Unknown building type: {building_type}"
        
        cost = BUILDING_COSTS[building_type]
        
        for resource, amount in cost.items():
            have = player.resources.get(resource, 0)
            if have < amount:
                return False, f"Need {amount} {resource.value}, have {have}"
        
        return True, None
    
    def check_victory(self, player: Player) -> bool:
        """Check if player has reached 10 VP."""
        return player.calculate_vp(self.board) >= 10
    
    def validate_robber_placement(self, tile_coords: Tuple[int, int],
                                   current_robber: Tuple[int, int]) -> Tuple[bool, Optional[str]]:
        """
        Validate robber movement.
        
        Rules:
        1. Must move to different tile
        2. Cannot place on desert initially (already there)
        
        Returns:
            (is_valid, error_message)
        """
        if tile_coords == current_robber:
            return False, "Robber must move to a different tile"
        
        if tile_coords not in self.board.tiles:
            return False, f"Invalid tile coordinates: {tile_coords}"
        
        return True, None
