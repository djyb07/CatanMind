"""
CatanMind - Strategy Manager
Dynamic strategy phases based on Victory Points (VP).
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from models import Board, Player, ResourceType, BuildingType, BUILDING_COSTS
from heuristics import HeuristicEngine


class GamePhase(Enum):
    """Game phases based on VP count."""
    EARLY = "early"      # 0-4 VP
    MID = "mid"          # 5-7 VP
    LATE = "late"        # 8+ VP


@dataclass
class StrategyRecommendation:
    """A strategic recommendation with reasoning."""
    action: str
    target: Optional[int]  # Node ID or None
    priority: float
    reasoning: str
    cost: Optional[Dict[ResourceType, int]] = None


class StrategyManager:
    """
    Manages game strategy based on current VP and board state.
    Adapts recommendations as the game progresses through phases.
    """
    
    # Phase weights for different factors
    PHASE_WEIGHTS = {
        GamePhase.EARLY: {
            "production": 0.6,
            "expansion": 0.3,
            "port_access": 0.1,
            "city_upgrade": 0.0,
            "blocking": 0.0,
            "dev_cards": 0.0
        },
        GamePhase.MID: {
            "production": 0.2,
            "expansion": 0.1,
            "port_access": 0.3,
            "city_upgrade": 0.5,
            "blocking": 0.2,
            "dev_cards": 0.1
        },
        GamePhase.LATE: {
            "production": 0.0,
            "expansion": 0.0,
            "port_access": 0.1,
            "city_upgrade": 0.4,
            "blocking": 0.1,
            "dev_cards": 0.4
        }
    }
    
    def __init__(self, board: Board):
        self.board = board
        self.heuristics = HeuristicEngine(board)
    
    def get_phase(self, player: Player) -> GamePhase:
        """Determine current game phase based on VP."""
        vp = player.calculate_vp(self.board)
        
        if vp <= 4:
            return GamePhase.EARLY
        elif vp <= 7:
            return GamePhase.MID
        else:
            return GamePhase.LATE
    
    def get_phase_description(self, phase: GamePhase) -> str:
        """Get human-readable description of the current phase."""
        descriptions = {
            GamePhase.EARLY: "Early Game (0-4 VP): Focus on production & expansion",
            GamePhase.MID: "Mid Game (5-7 VP): Port access, cities, and blocking",
            GamePhase.LATE: "Late Game (8+ VP): Pure VP focus - finish strong!"
        }
        return descriptions.get(phase, "Unknown phase")
    
    def get_recommendations(self, player: Player,
                            opponents: List[Player] = None) -> List[StrategyRecommendation]:
        """
        Get ranked list of strategic recommendations for the current turn.
        
        Args:
            player: The current player
            opponents: List of opponent players for blocking analysis
        
        Returns:
            Sorted list of recommendations by priority.
        """
        phase = self.get_phase(player)
        weights = self.PHASE_WEIGHTS[phase]
        recommendations = []
        
        # Analyze possible actions
        if weights["expansion"] > 0 or weights["production"] > 0:
            recommendations.extend(self._get_settlement_recommendations(player, phase, weights))
        
        if weights["city_upgrade"] > 0:
            recommendations.extend(self._get_city_recommendations(player, weights))
        
        recommendations.extend(self._get_road_recommendations(player, phase, weights))
        
        if weights["dev_cards"] > 0:
            recommendations.extend(self._get_dev_card_recommendations(player, weights))
        
        if weights["blocking"] > 0 and opponents:
            recommendations.extend(self._get_blocking_recommendations(player, opponents, weights))
        
        # Sort by priority
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        
        return recommendations
    
    def _get_settlement_recommendations(self, player: Player,
                                         phase: GamePhase,
                                         weights: Dict) -> List[StrategyRecommendation]:
        """Get recommendations for building settlements."""
        recs = []
        
        # Check if player can afford a settlement
        if not player.can_afford(BUILDING_COSTS["settlement"]):
            return recs
        
        # Get valid spots
        valid_spots = self.board.get_valid_settlement_spots(player.id, initial_phase=False)
        
        if not valid_spots:
            return recs
        
        # Get existing resources for synergy calculation
        existing_resources = self._get_player_resources(player)
        
        # Score top spots
        scored_spots = self.heuristics.get_ranked_spots(
            phase.value, existing_resources, player.id, top_n=3
        )
        
        for node_id, score in scored_spots:
            node = self.board.intersections[node_id]
            
            priority = score["total"] * (weights["production"] + weights["expansion"])
            
            # Bonus for port access in mid/late game
            if node.port_type and weights["port_access"] > 0:
                priority += 3.0 * weights["port_access"]
            
            resources = ", ".join(r.value for r in self._get_spot_resources(node_id))
            reasoning = f"Build settlement at #{node_id} ({resources})"
            
            if score.get("expansion", 0) > 2:
                reasoning += " - Good expansion"
            if node.port_type:
                reasoning += f" - {node.port_type.value} port"
            
            recs.append(StrategyRecommendation(
                action="build_settlement",
                target=node_id,
                priority=priority,
                reasoning=reasoning,
                cost=BUILDING_COSTS["settlement"]
            ))
        
        return recs
    
    def _get_city_recommendations(self, player: Player,
                                   weights: Dict) -> List[StrategyRecommendation]:
        """Get recommendations for upgrading to cities."""
        recs = []
        
        if not player.can_afford(BUILDING_COSTS["city"]):
            return recs
        
        # Find settlements to upgrade
        settlements = [n for n in self.board.get_player_buildings(player.id)
                      if n.building_type == BuildingType.SETTLEMENT]
        
        for node in settlements:
            # Score based on production value (cities double production)
            production = node.get_total_probability()
            
            # Prioritize high-ore/wheat spots for snowball effect
            ore_wheat_bonus = 0
            for tile in node.touching_tiles:
                if tile.resource_type == ResourceType.ORE:
                    ore_wheat_bonus += tile.probability * 0.5
                elif tile.resource_type == ResourceType.WHEAT:
                    ore_wheat_bonus += tile.probability * 0.3
            
            priority = (production + ore_wheat_bonus) * weights["city_upgrade"]
            
            resources = ", ".join(t.resource_type.value for t in node.touching_tiles
                                 if t.resource_type != ResourceType.DESERT)
            
            recs.append(StrategyRecommendation(
                action="upgrade_city",
                target=node.id,
                priority=priority,
                reasoning=f"Upgrade #{node.id} to city ({resources}) - doubles production",
                cost=BUILDING_COSTS["city"]
            ))
        
        return recs
    
    def _get_road_recommendations(self, player: Player,
                                   phase: GamePhase,
                                   weights: Dict) -> List[StrategyRecommendation]:
        """Get recommendations for building roads."""
        recs = []
        
        if not player.can_afford(BUILDING_COSTS["road"]):
            return recs
        
        valid_roads = self.board.get_valid_road_spots(player.id)
        
        if not valid_roads:
            return recs
        
        # Analyze road options
        scored_roads = []
        
        for node_a, node_b in valid_roads:
            score = 0
            reasoning_parts = []
            
            # Check if road leads to good settlement spots
            for endpoint in [node_a, node_b]:
                node = self.board.intersections[endpoint]
                if node.owner is None:
                    spot_score = self.heuristics.calculate_spot_score(endpoint, phase.value)
                    if spot_score["total"] > 4:
                        score += spot_score["total"] * 0.5
                        reasoning_parts.append(f"leads to good spot #{endpoint}")
            
            # Check for longest road potential
            current_road_length = self.board.calculate_longest_road(player.id)
            if current_road_length >= 4:
                # Close to longest road
                score += 2.0
                reasoning_parts.append("longest road potential")
            
            if score > 0:
                scored_roads.append((score, node_a, node_b, reasoning_parts))
        
        # Sort and take top 2
        scored_roads.sort(reverse=True)
        
        for score, node_a, node_b, reasons in scored_roads[:2]:
            priority = score * weights["expansion"]
            
            recs.append(StrategyRecommendation(
                action="build_road",
                target=node_a,  # Primary endpoint
                priority=priority,
                reasoning=f"Road {node_a}-{node_b}: " + ", ".join(reasons),
                cost=BUILDING_COSTS["road"]
            ))
        
        return recs
    
    def _get_dev_card_recommendations(self, player: Player,
                                       weights: Dict) -> List[StrategyRecommendation]:
        """Get recommendations for buying development cards."""
        recs = []
        
        if not player.can_afford(BUILDING_COSTS["development_card"]):
            return recs
        
        # Calculate expected value
        # Assume: ~5 VP cards, ~14 knights, ~6 other in deck
        priority = 2.0 * weights["dev_cards"]
        
        # Boost if close to largest army
        if player.knights_played >= 2:
            priority += 1.5
            if player.knights_played == 2:
                reasoning = "Buy dev card - 1 knight from Largest Army!"
            else:
                reasoning = "Buy dev card - maintain Largest Army"
        else:
            reasoning = "Buy dev card - chance for VP or knights"
        
        recs.append(StrategyRecommendation(
            action="buy_dev_card",
            target=None,
            priority=priority,
            reasoning=reasoning,
            cost=BUILDING_COSTS["development_card"]
        ))
        
        return recs
    
    def _get_blocking_recommendations(self, player: Player,
                                       opponents: List[Player],
                                       weights: Dict) -> List[StrategyRecommendation]:
        """Get recommendations for blocking opponent strategies."""
        recs = []
        
        for opponent in opponents:
            # Check opponent's road length
            road_length = self.board.calculate_longest_road(opponent.id)
            
            if road_length >= 4:
                # Opponent is close to or has longest road
                # Find where to block
                block_spots = self._find_road_block_spots(player, opponent)
                
                for spot_id, effectiveness in block_spots[:1]:  # Top blocking spot
                    priority = effectiveness * weights["blocking"]
                    
                    recs.append(StrategyRecommendation(
                        action="block_road",
                        target=spot_id,
                        priority=priority,
                        reasoning=f"Block opponent's road at #{spot_id} (road length: {road_length})",
                        cost=BUILDING_COSTS["settlement"]
                    ))
        
        return recs
    
    def _find_road_block_spots(self, player: Player,
                                opponent: Player) -> List[Tuple[int, float]]:
        """Find spots that can break opponent's road chain."""
        block_spots = []
        
        opponent_roads = self.board.get_player_roads(opponent.id)
        
        # Find all intersections along opponent's roads
        road_nodes = set()
        for road in opponent_roads:
            road_nodes.add(road.node_a)
            road_nodes.add(road.node_b)
        
        # Check which nodes we can build on
        valid_spots = self.board.get_valid_settlement_spots(player.id, initial_phase=False)
        
        for node_id in valid_spots:
            if node_id in road_nodes:
                # This spot is on opponent's road network
                node = self.board.intersections[node_id]
                
                # Calculate how many road segments this would break
                breaks = 0
                for neighbor_id in node.neighbors:
                    path = self.board.get_path(node_id, neighbor_id)
                    if path and path.owner == opponent.id:
                        breaks += 1
                
                if breaks > 0:
                    effectiveness = breaks * 2.0
                    block_spots.append((node_id, effectiveness))
        
        block_spots.sort(key=lambda x: x[1], reverse=True)
        return block_spots
    
    def _get_player_resources(self, player: Player) -> Set[ResourceType]:
        """Get set of resources player has access to."""
        resources = set()
        for node in self.board.get_player_buildings(player.id):
            for tile in node.touching_tiles:
                if tile.resource_type != ResourceType.DESERT:
                    resources.add(tile.resource_type)
        return resources
    
    def _get_spot_resources(self, spot_id: int) -> Set[ResourceType]:
        """Get resources at a spot."""
        node = self.board.intersections[spot_id]
        return set(t.resource_type for t in node.touching_tiles 
                  if t.resource_type != ResourceType.DESERT)
    
    def get_resource_alerts(self, player: Player) -> List[str]:
        """
        Generate alerts about resource situations.
        
        Returns:
            List of alert messages.
        """
        alerts = []
        
        # Port alert: lots of one resource
        for resource, count in player.resources.items():
            if count >= 4:
                # Check if player has matching port
                has_matching_port = False
                for node in self.board.get_player_buildings(player.id):
                    if node.port_type and resource.value in node.port_type.value:
                        has_matching_port = True
                        break
                
                if has_matching_port:
                    alerts.append(f"⚠️ PORT TRADE: You have {count} {resource.value} and a matching port!")
                else:
                    alerts.append(f"💡 Consider building toward a {resource.value} port (have {count})")
        
        # Scarcity alert
        resources_player_produces = self._get_player_resources(player)
        all_resources = {ResourceType.WOOD, ResourceType.BRICK, ResourceType.SHEEP,
                        ResourceType.WHEAT, ResourceType.ORE}
        missing = all_resources - resources_player_produces
        
        if missing:
            missing_str = ", ".join(r.value for r in missing)
            alerts.append(f"⚠️ You don't produce: {missing_str}")
        
        return alerts
    
    def get_next_best_move(self, player: Player,
                           opponents: List[Player] = None) -> Optional[StrategyRecommendation]:
        """Get the single best recommended action."""
        recommendations = self.get_recommendations(player, opponents)
        return recommendations[0] if recommendations else None
