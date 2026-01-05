"""
CatanMind - Mid-Game ROI Calculator
Calculates Return on Investment for all possible actions.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from models import Board, Player, ResourceType, BuildingType, BUILDING_COSTS
from heuristics import HeuristicEngine


@dataclass
class ROIAnalysis:
    """Analysis of a potential action's ROI."""
    action: str
    target: Optional[int]
    vp_gain: float
    production_gain: float
    cost_value: float
    roi: float
    reasoning: str
    alert: Optional[str] = None


class MidGameSolver:
    """
    Calculates ROI for all possible actions and generates alerts.
    
    ROI = (VP_Gain + Production_Gain) / Cost
    """
    
    # Resource values for cost calculation
    RESOURCE_VALUES = {
        ResourceType.WOOD: 1.0,
        ResourceType.BRICK: 1.0,
        ResourceType.SHEEP: 1.0,
        ResourceType.WHEAT: 1.2,
        ResourceType.ORE: 1.3
    }
    
    def __init__(self, board: Board):
        self.board = board
        self.heuristics = HeuristicEngine(board)
    
    def calculate_cost_value(self, cost: Dict[ResourceType, int]) -> float:
        """Calculate the total value of a cost."""
        total = 0.0
        for resource, amount in cost.items():
            total += amount * self.RESOURCE_VALUES.get(resource, 1.0)
        return total
    
    def calculate_production_gain(self, node_id: int, is_city: bool = False) -> float:
        """
        Calculate production gain from building at a spot.
        
        Args:
            node_id: The intersection
            is_city: True if upgrading to city (doubles existing production)
        """
        node = self.board.intersections[node_id]
        
        if is_city:
            # City doubles production from existing settlement
            return node.get_total_probability()
        else:
            # New settlement
            return node.get_total_probability()
    
    def recommend_move(self, player: Player,
                       opponents: List[Player] = None) -> List[ROIAnalysis]:
        """
        Calculate ROI for all possible actions and rank them.
        
        Args:
            player: Current player
            opponents: List of opponents for blocking analysis
        
        Returns:
            Sorted list of ROI analyses.
        """
        analyses = []
        
        # Analyze settlement options
        analyses.extend(self._analyze_settlements(player))
        
        # Analyze city upgrades
        analyses.extend(self._analyze_cities(player))
        
        # Analyze roads
        analyses.extend(self._analyze_roads(player))
        
        # Analyze development cards
        analyses.extend(self._analyze_dev_cards(player))
        
        # Check for special alerts
        alerts = self.generate_alerts(player, opponents or [])
        
        # Attach relevant alerts to top analyses
        for analysis in analyses:
            for alert in alerts:
                if alert["type"] == "port" and analysis.action == "build_settlement":
                    analysis.alert = alert["message"]
                elif alert["type"] == "road_break" and analysis.action in ["build_settlement", "block"]:
                    if analysis.target == alert.get("target"):
                        analysis.alert = alert["message"]
        
        # Sort by ROI descending
        analyses.sort(key=lambda a: a.roi, reverse=True)
        
        return analyses
    
    def _analyze_settlements(self, player: Player) -> List[ROIAnalysis]:
        """Analyze ROI for settlement building."""
        analyses = []
        
        if not player.can_afford(BUILDING_COSTS["settlement"]):
            return analyses
        
        cost_value = self.calculate_cost_value(BUILDING_COSTS["settlement"])
        valid_spots = self.board.get_valid_settlement_spots(player.id, initial_phase=False)
        
        for node_id in valid_spots[:10]:  # Analyze top 10 spots
            node = self.board.intersections[node_id]
            
            # VP gain = 1 (settlement = 1 VP)
            vp_gain = 1.0
            
            # Production gain
            prod_gain = self.calculate_production_gain(node_id)
            
            # Bonus for port
            port_bonus = 0
            if node.port_type:
                port_bonus = 0.5  # Additional value
            
            # Calculate ROI
            total_gain = vp_gain + (prod_gain * 0.3) + port_bonus
            roi = total_gain / cost_value if cost_value > 0 else 0
            
            resources = ", ".join(t.resource_type.value for t in node.touching_tiles
                                 if t.resource_type != ResourceType.DESERT)
            
            reasoning = f"Settlement at #{node_id}: {resources}"
            if node.port_type:
                reasoning += f" + {node.port_type.value} port"
            
            analyses.append(ROIAnalysis(
                action="build_settlement",
                target=node_id,
                vp_gain=vp_gain,
                production_gain=prod_gain,
                cost_value=cost_value,
                roi=round(roi, 3),
                reasoning=reasoning
            ))
        
        return analyses
    
    def _analyze_cities(self, player: Player) -> List[ROIAnalysis]:
        """Analyze ROI for city upgrades."""
        analyses = []
        
        if not player.can_afford(BUILDING_COSTS["city"]):
            return analyses
        
        cost_value = self.calculate_cost_value(BUILDING_COSTS["city"])
        
        settlements = [n for n in self.board.get_player_buildings(player.id)
                      if n.building_type == BuildingType.SETTLEMENT]
        
        for node in settlements:
            # VP gain = 1 (going from 1 VP to 2 VP)
            vp_gain = 1.0
            
            # Production gain = doubles the existing production
            prod_gain = self.calculate_production_gain(node.id, is_city=True)
            
            # Calculate ROI
            total_gain = vp_gain + (prod_gain * 0.3)
            roi = total_gain / cost_value if cost_value > 0 else 0
            
            resources = ", ".join(t.resource_type.value for t in node.touching_tiles
                                 if t.resource_type != ResourceType.DESERT)
            numbers = [t.dice_number for t in node.touching_tiles if t.dice_number > 0]
            
            analyses.append(ROIAnalysis(
                action="upgrade_city",
                target=node.id,
                vp_gain=vp_gain,
                production_gain=prod_gain,
                cost_value=cost_value,
                roi=round(roi, 3),
                reasoning=f"City at #{node.id}: {resources} {numbers} - doubles production"
            ))
        
        return analyses
    
    def _analyze_roads(self, player: Player) -> List[ROIAnalysis]:
        """Analyze ROI for road building."""
        analyses = []
        
        if not player.can_afford(BUILDING_COSTS["road"]):
            return analyses
        
        cost_value = self.calculate_cost_value(BUILDING_COSTS["road"])
        valid_roads = self.board.get_valid_road_spots(player.id)
        
        current_road_length = self.board.calculate_longest_road(player.id)
        
        for node_a, node_b in valid_roads[:8]:  # Top 8 road options
            # VP potential (longest road)
            vp_gain = 0.0
            if current_road_length == 4:
                vp_gain = 0.5  # Close to longest road
            elif current_road_length >= 5:
                vp_gain = 0.2  # Extending longest
            
            # Strategic value - does it lead somewhere good?
            strategic_value = 0.0
            for endpoint in [node_a, node_b]:
                node = self.board.intersections[endpoint]
                if node.owner is None:
                    spot_score = self.heuristics.calculate_spot_score(endpoint, "mid")
                    if spot_score["total"] > 5:
                        strategic_value += 0.5
            
            total_gain = vp_gain + strategic_value
            roi = total_gain / cost_value if cost_value > 0 else 0
            
            reasoning = f"Road {node_a}-{node_b}"
            if vp_gain > 0:
                reasoning += f" (Longest Road: {current_road_length}+)"
            if strategic_value > 0:
                reasoning += " - leads to good spots"
            
            analyses.append(ROIAnalysis(
                action="build_road",
                target=node_a,
                vp_gain=vp_gain,
                production_gain=0,
                cost_value=cost_value,
                roi=round(roi, 3),
                reasoning=reasoning
            ))
        
        return analyses
    
    def _analyze_dev_cards(self, player: Player) -> List[ROIAnalysis]:
        """Analyze ROI for buying development cards."""
        analyses = []
        
        if not player.can_afford(BUILDING_COSTS["development_card"]):
            return analyses
        
        cost_value = self.calculate_cost_value(BUILDING_COSTS["development_card"])
        
        # Expected value calculation
        # Assume standard deck: ~5 VP cards, ~14 knights, ~6 other
        # Expected VP = 5/25 * 1 = 0.2
        # Knight value = helps toward largest army (2 VP for having most)
        
        expected_vp = 0.2
        
        # Bonus if close to largest army
        knight_bonus = 0
        if player.knights_played >= 2:
            knight_bonus = 0.5  # High chance of getting there
            if player.knights_played == 2:
                knight_bonus = 1.0  # One knight away!
        
        total_gain = expected_vp + knight_bonus
        roi = total_gain / cost_value if cost_value > 0 else 0
        
        reasoning = f"Dev Card (Expected VP: {expected_vp:.1f})"
        if knight_bonus > 0:
            reasoning += f" + Largest Army potential (knights: {player.knights_played})"
        
        analyses.append(ROIAnalysis(
            action="buy_dev_card",
            target=None,
            vp_gain=expected_vp + knight_bonus,
            production_gain=0,
            cost_value=cost_value,
            roi=round(roi, 3),
            reasoning=reasoning
        ))
        
        return analyses
    
    def generate_alerts(self, player: Player,
                        opponents: List[Player]) -> List[Dict]:
        """
        Generate special situation alerts.
        
        Returns:
            List of alert dictionaries.
        """
        alerts = []
        
        # PORT ALERT: Player has excess of one resource
        for resource, count in player.resources.items():
            if count >= 4:
                # Check for matching port in network
                has_matching_port = False
                matching_port_node = None
                
                for node in self.board.get_player_buildings(player.id):
                    if node.port_type and resource.value in node.port_type.value:
                        has_matching_port = True
                        break
                
                if has_matching_port:
                    alerts.append({
                        "type": "port",
                        "priority": "high",
                        "message": f"🚢 PORT ALERT: Trade {resource.value} at 2:1! (have {count})"
                    })
                else:
                    # Suggest building toward a port
                    port_spots = self._find_port_spots(resource)
                    if port_spots:
                        alerts.append({
                            "type": "port",
                            "priority": "medium",
                            "target": port_spots[0],
                            "message": f"💡 Build toward {resource.value} port (have {count} {resource.value})"
                        })
        
        # ROAD BREAK: Opponent has long road
        for opponent in opponents:
            road_length = self.board.calculate_longest_road(opponent.id)
            
            if road_length >= 4:
                # Find blocking spots
                break_spots = self._find_road_break_spots(player, opponent)
                
                if break_spots:
                    spot_id, effectiveness = break_spots[0]
                    
                    if effectiveness >= 2:
                        alerts.append({
                            "type": "road_break",
                            "priority": "high",
                            "target": spot_id,
                            "message": f"⚔️ ROAD BREAK: Block opponent's {road_length}-road at #{spot_id}!"
                        })
        
        return alerts
    
    def _find_port_spots(self, resource: ResourceType) -> List[int]:
        """Find spots with matching port."""
        from models import PortType
        
        port_map = {
            ResourceType.WOOD: PortType.WOOD,
            ResourceType.BRICK: PortType.BRICK,
            ResourceType.SHEEP: PortType.SHEEP,
            ResourceType.WHEAT: PortType.WHEAT,
            ResourceType.ORE: PortType.ORE
        }
        
        target_port = port_map.get(resource)
        spots = []
        
        for node in self.board.intersections.values():
            if node.port_type == target_port and node.owner is None:
                spots.append(node.id)
        
        return spots
    
    def _find_road_break_spots(self, player: Player,
                                opponent: Player) -> List[Tuple[int, float]]:
        """Find spots that break opponent's road chain."""
        break_spots = []
        
        opponent_roads = self.board.get_player_roads(opponent.id)
        
        # Find intersections along opponent's roads
        road_nodes = set()
        for road in opponent_roads:
            road_nodes.add(road.node_a)
            road_nodes.add(road.node_b)
        
        # Check which we can build on
        valid_spots = self.board.get_valid_settlement_spots(player.id, initial_phase=False)
        
        for node_id in valid_spots:
            if node_id in road_nodes:
                node = self.board.intersections[node_id]
                
                # Count how many road segments this breaks
                breaks = sum(1 for n_id in node.neighbors
                           if self.board.get_path(node_id, n_id) and
                           self.board.get_path(node_id, n_id).owner == opponent.id)
                
                if breaks > 0:
                    break_spots.append((node_id, breaks))
        
        break_spots.sort(key=lambda x: x[1], reverse=True)
        return break_spots
    
    def get_best_move(self, player: Player,
                      opponents: List[Player] = None) -> Optional[ROIAnalysis]:
        """Get the single best move by ROI."""
        analyses = self.recommend_move(player, opponents)
        return analyses[0] if analyses else None
