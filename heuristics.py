"""
CatanMind - Heuristic Engines
The "Brain" - scoring algorithms for optimal decision making.
"""

from collections import deque
from typing import Dict, List, Set, Tuple, Optional
from models import Board, Intersection, ResourceType, DICE_PROBABILITIES


class HeuristicEngine:
    """
    Calculates scores for intersections based on multiple factors:
    - Production value (pip probabilities)
    - Resource diversity
    - Synergy bonuses (Wood+Brick, Ore+Wheat)
    - Expansion potential (BFS)
    - Scarcity adjustment
    """
    
    # Synergy pairs that work well together
    SYNERGY_PAIRS = [
        ({ResourceType.WOOD, ResourceType.BRICK}, 1.5),  # Roads & Settlements
        ({ResourceType.ORE, ResourceType.WHEAT}, 1.4),   # Cities & Dev Cards
        ({ResourceType.SHEEP, ResourceType.WHEAT, ResourceType.ORE}, 1.3),  # Dev Cards
    ]
    
    # Base resource values (can be adjusted by scarcity)
    BASE_RESOURCE_VALUES = {
        ResourceType.WOOD: 1.0,
        ResourceType.BRICK: 1.0,
        ResourceType.SHEEP: 0.8,
        ResourceType.WHEAT: 1.1,
        ResourceType.ORE: 1.2,
        ResourceType.DESERT: 0.0
    }
    
    def __init__(self, board: Board):
        self.board = board
        self._resource_scarcity = self._calculate_global_scarcity()
    
    def _calculate_global_scarcity(self) -> Dict[ResourceType, float]:
        """
        Calculate how scarce each resource is on the board.
        Returns a multiplier (higher = more scarce = more valuable).
        """
        resource_pips: Dict[ResourceType, int] = {r: 0 for r in ResourceType}
        
        for tile in self.board.tiles.values():
            if tile.resource_type != ResourceType.DESERT:
                resource_pips[tile.resource_type] += tile.probability
        
        total_pips = sum(resource_pips.values())
        if total_pips == 0:
            return {r: 1.0 for r in ResourceType}
        
        # Average pips per resource type
        avg_pips = total_pips / 5  # 5 resource types (excluding desert)
        
        scarcity = {}
        for resource, pips in resource_pips.items():
            if resource == ResourceType.DESERT:
                scarcity[resource] = 0.0
            elif pips == 0:
                scarcity[resource] = 2.5  # Very scarce
            else:
                # Inverse relationship: fewer pips = higher scarcity multiplier
                scarcity[resource] = min(2.0, max(0.5, avg_pips / pips))
        
        return scarcity
    
    def calculate_production(self, node_id: int) -> float:
        """
        Calculate production score based on pip probabilities of adjacent tiles.
        Adjusted for resource scarcity.
        """
        if node_id not in self.board.intersections:
            return 0.0
        
        node = self.board.intersections[node_id]
        score = 0.0
        
        for tile in node.touching_tiles:
            if tile.resource_type == ResourceType.DESERT:
                continue
            
            # Base probability (pips)
            pips = tile.probability
            
            # Apply scarcity multiplier
            scarcity_mult = self._resource_scarcity.get(tile.resource_type, 1.0)
            
            # Apply base resource value
            base_value = self.BASE_RESOURCE_VALUES.get(tile.resource_type, 1.0)
            
            score += pips * scarcity_mult * base_value
        
        return round(score, 2)
    
    def calculate_diversity(self, node_id: int) -> float:
        """
        Calculate diversity bonus for unique resource types.
        Having access to more different resources is valuable.
        """
        if node_id not in self.board.intersections:
            return 0.0
        
        node = self.board.intersections[node_id]
        resources = set()
        
        for tile in node.touching_tiles:
            if tile.resource_type != ResourceType.DESERT:
                resources.add(tile.resource_type)
        
        # Bonus scale: 1 resource = 0, 2 resources = 1.0, 3 resources = 2.5
        diversity_bonuses = {1: 0.0, 2: 1.0, 3: 2.5}
        return diversity_bonuses.get(len(resources), 0.0)
    
    def calculate_synergy(self, node_id: int, existing_resources: Set[ResourceType] = None) -> float:
        """
        Calculate synergy bonus for resource combinations.
        Considers both resources at this node and existing player resources.
        """
        if node_id not in self.board.intersections:
            return 0.0
        
        node = self.board.intersections[node_id]
        node_resources = set(tile.resource_type for tile in node.touching_tiles 
                            if tile.resource_type != ResourceType.DESERT)
        
        # Combine with existing resources
        all_resources = node_resources.copy()
        if existing_resources:
            all_resources |= existing_resources
        
        synergy_bonus = 0.0
        
        for pair_resources, bonus in self.SYNERGY_PAIRS:
            # Check if we complete or contribute to this synergy
            if pair_resources.issubset(all_resources):
                synergy_bonus += bonus
            elif len(pair_resources & node_resources) > 0 and len(pair_resources & all_resources) > len(pair_resources & (all_resources - node_resources)):
                # Partial bonus for contributing to synergy
                synergy_bonus += bonus * 0.5
        
        return round(synergy_bonus, 2)
    
    def calculate_expansion_potential(self, node_id: int, max_depth: int = 3) -> float:
        """
        Run BFS to depth 3 to calculate expansion potential.
        Counts valid future building spots respecting distance rule.
        Penalizes blocked paths.
        """
        if node_id not in self.board.intersections:
            return 0.0
        
        visited: Set[int] = set()
        queue = deque([(node_id, 0)])  # (node_id, depth)
        valid_spots = 0
        blocked_penalty = 0
        
        while queue:
            current_id, depth = queue.popleft()
            
            if current_id in visited:
                continue
            visited.add(current_id)
            
            if depth > max_depth:
                continue
            
            current_node = self.board.intersections[current_id]
            
            # Check if this is a valid future building spot
            if depth > 0:  # Don't count the starting node
                if current_node.owner is None:
                    # Check distance rule
                    is_valid = True
                    for neighbor_id in current_node.neighbors:
                        neighbor = self.board.intersections[neighbor_id]
                        if neighbor.owner is not None:
                            is_valid = False
                            break
                    
                    if is_valid:
                        # Value decreases with distance
                        valid_spots += (max_depth - depth + 1) * 0.5
                else:
                    # Occupied by someone - potential block
                    blocked_penalty += 0.5
            
            # Explore neighbors
            for neighbor_id in current_node.neighbors:
                if neighbor_id not in visited:
                    # Check if path is blocked by enemy road
                    path = self.board.get_path(current_id, neighbor_id)
                    if path and path.owner is not None:
                        # Path is owned - check if it's enemy
                        # For now, treat any owned path as potential block
                        blocked_penalty += 0.3
                    else:
                        queue.append((neighbor_id, depth + 1))
        
        return round(max(0, valid_spots - blocked_penalty), 2)
    
    def calculate_port_value(self, node_id: int) -> float:
        """Calculate value of port access at this node."""
        if node_id not in self.board.intersections:
            return 0.0
        
        node = self.board.intersections[node_id]
        
        if node.port_type is None:
            return 0.0
        
        # 3:1 ports are good, 2:1 ports are excellent if matching resources
        from models import PortType
        
        if node.port_type == PortType.GENERIC:
            return 1.5
        
        # Check if we have matching resource production
        port_resource_map = {
            PortType.WOOD: ResourceType.WOOD,
            PortType.BRICK: ResourceType.BRICK,
            PortType.SHEEP: ResourceType.SHEEP,
            PortType.WHEAT: ResourceType.WHEAT,
            PortType.ORE: ResourceType.ORE,
        }
        
        matching_resource = port_resource_map.get(node.port_type)
        if matching_resource:
            # Check if this node produces the matching resource
            for tile in node.touching_tiles:
                if tile.resource_type == matching_resource:
                    return 3.0  # Excellent synergy
            return 1.0  # Still useful
        
        return 0.0
    
    def calculate_spot_score(self, node_id: int, 
                             game_phase: str = "early",
                             existing_resources: Set[ResourceType] = None,
                             player_id: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate comprehensive score for a spot.
        Returns breakdown of all factors.
        
        Args:
            node_id: The intersection to score
            game_phase: 'early', 'mid', or 'late'
            existing_resources: Resources player already has access to
            player_id: Current player ID for blocking calculations
        """
        if node_id not in self.board.intersections:
            return {"total": 0.0}
        
        node = self.board.intersections[node_id]
        
        # Cannot score occupied spots
        if node.owner is not None:
            return {"total": 0.0}
        
        # Calculate individual components
        production = self.calculate_production(node_id)
        diversity = self.calculate_diversity(node_id)
        synergy = self.calculate_synergy(node_id, existing_resources)
        expansion = self.calculate_expansion_potential(node_id)
        port = self.calculate_port_value(node_id)
        
        # Apply phase-based weights
        weights = self._get_phase_weights(game_phase)
        
        weighted_score = (
            production * weights['production'] +
            diversity * weights['diversity'] +
            synergy * weights['synergy'] +
            expansion * weights['expansion'] +
            port * weights['port']
        )
        
        return {
            "production": production,
            "diversity": diversity,
            "synergy": synergy,
            "expansion": expansion,
            "port": port,
            "total": round(weighted_score, 2)
        }
    
    def _get_phase_weights(self, phase: str) -> Dict[str, float]:
        """Get scoring weights based on game phase."""
        weights = {
            "early": {
                "production": 1.0,
                "diversity": 0.4,
                "synergy": 0.3,
                "expansion": 0.5,
                "port": 0.2
            },
            "mid": {
                "production": 0.6,
                "diversity": 0.3,
                "synergy": 0.5,
                "expansion": 0.3,
                "port": 0.6
            },
            "late": {
                "production": 0.3,
                "diversity": 0.1,
                "synergy": 0.2,
                "expansion": 0.1,
                "port": 0.4
            }
        }
        return weights.get(phase, weights["early"])
    
    def get_ranked_spots(self, game_phase: str = "early",
                         existing_resources: Set[ResourceType] = None,
                         player_id: Optional[int] = None,
                         top_n: int = 10) -> List[Tuple[int, Dict[str, float]]]:
        """
        Get ranked list of best available spots.
        
        Returns:
            List of (node_id, score_breakdown) tuples sorted by total score.
        """
        valid_spots = self.board.get_valid_settlement_spots(player_id, initial_phase=True)
        
        scored_spots = []
        for node_id in valid_spots:
            score = self.calculate_spot_score(
                node_id, game_phase, existing_resources, player_id
            )
            if score["total"] > 0:
                scored_spots.append((node_id, score))
        
        # Sort by total score descending
        scored_spots.sort(key=lambda x: x[1]["total"], reverse=True)
        
        return scored_spots[:top_n]
    
    def explain_spot(self, node_id: int, score: Dict[str, float]) -> str:
        """Generate human-readable explanation for a spot's score."""
        if node_id not in self.board.intersections:
            return "Invalid spot"
        
        node = self.board.intersections[node_id]
        resources = [t.resource_type.value for t in node.touching_tiles 
                    if t.resource_type != ResourceType.DESERT]
        numbers = [t.dice_number for t in node.touching_tiles 
                  if t.dice_number > 0]
        
        explanation = f"Spot #{node_id}: "
        explanation += f"Resources: {', '.join(resources)} | "
        explanation += f"Numbers: {numbers} | "
        explanation += f"Score: {score['total']:.1f}\n"
        
        reasons = []
        if score.get('production', 0) > 5:
            reasons.append("High production")
        if score.get('diversity', 0) > 1:
            reasons.append("Good diversity")
        if score.get('synergy', 0) > 1:
            reasons.append("Strong synergy")
        if score.get('expansion', 0) > 2:
            reasons.append("Great expansion")
        if score.get('port', 0) > 0:
            reasons.append("Port access")
        
        if reasons:
            explanation += "Why: " + ", ".join(reasons)
        
        return explanation
