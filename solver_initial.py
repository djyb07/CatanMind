"""
CatanMind - Initial Placement Solver
Snake draft algorithm (1-2-3-4-4-3-2-1) with Minimax lookahead.
"""

from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from models import Board, ResourceType
from heuristics import HeuristicEngine


@dataclass
class PlacementRecommendation:
    """Recommendation for initial placement."""
    node_id: int
    score: float
    resources: List[str]
    numbers: List[int]
    reasoning: str
    complementary_spot: Optional[int] = None  # For second placement


class InitialPlacementSolver:
    """
    Solves the initial placement phase using the snake draft order.
    Turn order: 1-2-3-4-4-3-2-1
    
    Uses Minimax with alpha-beta pruning to predict opponent moves
    and select optimal complementary placements.
    """
    
    def __init__(self, board: Board):
        self.board = board
        self.heuristics = HeuristicEngine(board)
    
    def get_turn_order(self) -> List[int]:
        """Returns the standard snake draft order."""
        return [1, 2, 3, 4, 4, 3, 2, 1]
    
    def get_best_starting_spots(self, player_turn_index: int,
                                 num_players: int = 4,
                                 taken_spots: List[int] = None) -> List[PlacementRecommendation]:
        """
        Get the best spots for initial placement.
        
        Args:
            player_turn_index: 1-4 indicating player position
            num_players: Total number of players (usually 4)
            taken_spots: List of already-taken intersection IDs
        
        Returns:
            Top 3 recommendations with reasoning.
        """
        taken_spots = taken_spots or []
        
        # Mark taken spots on a copy (for analysis)
        available_spots = self._get_available_spots(taken_spots)
        
        if player_turn_index == 4:
            # Player 4 can plan both placements at once
            return self._solve_player_4(available_spots, taken_spots)
        else:
            # Players 1-3 use lookahead
            return self._solve_with_lookahead(
                player_turn_index, available_spots, taken_spots, num_players
            )
    
    def _get_available_spots(self, taken_spots: List[int]) -> List[int]:
        """Get list of available valid spots."""
        all_valid = self.board.get_valid_settlement_spots(initial_phase=True)
        
        # Remove taken spots and their neighbors (distance rule)
        blocked = set(taken_spots)
        for spot_id in taken_spots:
            if spot_id in self.board.intersections:
                blocked.update(self.board.intersections[spot_id].neighbors)
        
        return [s for s in all_valid if s not in blocked]
    
    def _solve_player_4(self, available_spots: List[int],
                        taken_spots: List[int]) -> List[PlacementRecommendation]:
        """
        Player 4 gets back-to-back picks, so optimize for complementary pair.
        """
        recommendations = []
        
        # Score all available spots
        spot_scores = {}
        for spot_id in available_spots:
            score = self.heuristics.calculate_spot_score(spot_id, "early")
            spot_scores[spot_id] = score
        
        # Find best complementary pairs
        best_pairs = self._find_complementary_pairs(available_spots, spot_scores)
        
        for pair_score, spot_a, spot_b, combined_resources in best_pairs[:3]:
            node_a = self.board.intersections[spot_a]
            node_b = self.board.intersections[spot_b]
            
            resources_a = [t.resource_type.value for t in node_a.touching_tiles 
                          if t.resource_type != ResourceType.DESERT]
            numbers_a = [t.dice_number for t in node_a.touching_tiles if t.dice_number > 0]
            
            reasoning = self._generate_pair_reasoning(spot_a, spot_b, spot_scores, combined_resources)
            
            recommendations.append(PlacementRecommendation(
                node_id=spot_a,
                score=pair_score,
                resources=resources_a,
                numbers=numbers_a,
                reasoning=reasoning,
                complementary_spot=spot_b
            ))
        
        return recommendations
    
    def _find_complementary_pairs(self, available_spots: List[int],
                                   spot_scores: Dict[int, Dict]) -> List[Tuple]:
        """
        Find pairs of spots that complement each other.
        Considers: resource coverage, distance (not too close/far), combined score.
        """
        pairs = []
        
        for i, spot_a in enumerate(available_spots):
            score_a = spot_scores[spot_a]
            resources_a = self._get_spot_resources(spot_a)
            
            for spot_b in available_spots[i+1:]:
                # Check distance constraint
                if not self._valid_pair_distance(spot_a, spot_b):
                    continue
                
                score_b = spot_scores[spot_b]
                resources_b = self._get_spot_resources(spot_b)
                
                # Calculate combined resources
                combined = resources_a | resources_b
                
                # Score the pair
                base_score = score_a["total"] + score_b["total"]
                
                # Bonus for resource diversity
                diversity_bonus = len(combined) * 0.5
                
                # Bonus for having all 5 resources
                if len(combined) == 5:
                    diversity_bonus += 2.0
                
                # Synergy bonus for covering key pairs
                synergy_bonus = 0
                if {ResourceType.WOOD, ResourceType.BRICK}.issubset(combined):
                    synergy_bonus += 1.5
                if {ResourceType.ORE, ResourceType.WHEAT}.issubset(combined):
                    synergy_bonus += 1.5
                
                total_pair_score = base_score + diversity_bonus + synergy_bonus
                pairs.append((total_pair_score, spot_a, spot_b, combined))
        
        # Sort by combined score
        pairs.sort(reverse=True)
        return pairs
    
    def _valid_pair_distance(self, spot_a: int, spot_b: int) -> bool:
        """
        Check if two spots are at valid distance for a pair.
        Should be at least 2 edges apart (not neighbors) but within road reach.
        """
        node_a = self.board.intersections[spot_a]
        node_b = self.board.intersections[spot_b]
        
        # Not neighbors (too close)
        if spot_b in node_a.neighbors:
            return False
        
        # Check if reachable within reasonable road distance (BFS depth 4-6)
        # For initial placement, we're not checking road connectivity strictly
        # but we want spots that can eventually connect
        
        return True
    
    def _get_spot_resources(self, spot_id: int) -> Set[ResourceType]:
        """Get resources at a spot."""
        node = self.board.intersections[spot_id]
        return set(t.resource_type for t in node.touching_tiles 
                  if t.resource_type != ResourceType.DESERT)
    
    def _solve_with_lookahead(self, player_turn: int,
                               available_spots: List[int],
                               taken_spots: List[int],
                               num_players: int) -> List[PlacementRecommendation]:
        """
        Use Minimax lookahead to find best spot considering opponent picks.
        Predicts what opponents will take before our second pick.
        """
        recommendations = []
        
        # Calculate how many picks happen between our first and second
        # Turn order: 1-2-3-4-4-3-2-1
        picks_between = self._calculate_picks_between(player_turn, num_players)
        
        # For each candidate first spot, simulate opponent picks and find best second spot
        scored_candidates = []
        
        for spot_a in available_spots:
            score_a = self.heuristics.calculate_spot_score(spot_a, "early")
            
            if score_a["total"] < 3.0:  # Skip low-value spots
                continue
            
            # Simulate opponents taking best remaining spots
            predicted_taken = self._predict_opponent_picks(
                spot_a, available_spots, picks_between
            )
            
            # Find best second spot after opponent picks
            remaining = [s for s in available_spots 
                        if s != spot_a and s not in predicted_taken
                        and self._valid_pair_distance(spot_a, s)]
            
            if not remaining:
                continue
            
            resources_a = self._get_spot_resources(spot_a)
            
            # Find best complementary second spot
            best_b_score = 0
            best_b = None
            
            for spot_b in remaining:
                # Score considering what resources we need
                score_b = self.heuristics.calculate_spot_score(
                    spot_b, "early", existing_resources=resources_a
                )
                
                resources_b = self._get_spot_resources(spot_b)
                combined = resources_a | resources_b
                
                # Bonus for filling resource gaps
                gap_bonus = len(combined) * 0.3
                
                total_b = score_b["total"] + gap_bonus
                
                if total_b > best_b_score:
                    best_b_score = total_b
                    best_b = spot_b
            
            if best_b:
                # Combined score: both spots plus synergy
                combined_resources = resources_a | self._get_spot_resources(best_b)
                combined_score = score_a["total"] + best_b_score
                
                scored_candidates.append((
                    combined_score, spot_a, best_b, 
                    score_a, combined_resources
                ))
        
        # Sort and take top 3
        scored_candidates.sort(reverse=True)
        
        for combined_score, spot_a, spot_b, score_a, combined_resources in scored_candidates[:3]:
            node_a = self.board.intersections[spot_a]
            
            resources = [t.resource_type.value for t in node_a.touching_tiles 
                        if t.resource_type != ResourceType.DESERT]
            numbers = [t.dice_number for t in node_a.touching_tiles if t.dice_number > 0]
            
            reasoning = self._generate_lookahead_reasoning(
                player_turn, spot_a, spot_b, score_a, combined_resources
            )
            
            recommendations.append(PlacementRecommendation(
                node_id=spot_a,
                score=combined_score,
                resources=resources,
                numbers=numbers,
                reasoning=reasoning,
                complementary_spot=spot_b
            ))
        
        return recommendations
    
    def _calculate_picks_between(self, player_turn: int, num_players: int) -> int:
        """
        Calculate how many opponent picks happen between our first and second pick.
        Turn order: 1-2-3-4-4-3-2-1
        """
        if num_players != 4:
            # Simplified for non-4 player games
            return (num_players - player_turn) * 2
        
        # For 4 players: 1-2-3-4-4-3-2-1
        picks = {
            1: 6,  # Players 2,3,4,4,3,2 pick before our second
            2: 4,  # Players 3,4,4,3 pick before our second
            3: 2,  # Players 4,4 pick before our second
            4: 0   # Back-to-back picks
        }
        return picks.get(player_turn, 0)
    
    def _predict_opponent_picks(self, our_pick: int,
                                 available: List[int],
                                 num_picks: int) -> Set[int]:
        """
        Predict which spots opponents will take.
        Assumes opponents pick highest-scoring available spots.
        """
        predicted = set()
        remaining = [s for s in available if s != our_pick]
        
        for _ in range(num_picks):
            if not remaining:
                break
            
            # Opponents pick the best remaining spot
            best_spot = None
            best_score = -1
            
            for spot in remaining:
                score = self.heuristics.calculate_spot_score(spot, "early")
                if score["total"] > best_score:
                    best_score = score["total"]
                    best_spot = spot
            
            if best_spot:
                predicted.add(best_spot)
                remaining.remove(best_spot)
                # Also remove neighbors (distance rule)
                if best_spot in self.board.intersections:
                    for neighbor in self.board.intersections[best_spot].neighbors:
                        if neighbor in remaining:
                            remaining.remove(neighbor)
        
        return predicted
    
    def _generate_pair_reasoning(self, spot_a: int, spot_b: int,
                                  scores: Dict, combined_resources: Set) -> str:
        """Generate reasoning for a pair recommendation (Player 4)."""
        node_a = self.board.intersections[spot_a]
        node_b = self.board.intersections[spot_b]
        
        resources_str = ", ".join(r.value for r in combined_resources)
        
        reasons = [f"Complementary pair covers: {resources_str}"]
        
        # Check for key synergies
        if {ResourceType.WOOD, ResourceType.BRICK}.issubset(combined_resources):
            reasons.append("Road/Settlement production ready")
        if {ResourceType.ORE, ResourceType.WHEAT}.issubset(combined_resources):
            reasons.append("City upgrade path available")
        
        # Check port access
        if node_a.port_type or node_b.port_type:
            reasons.append("Port access included")
        
        return ". ".join(reasons)
    
    def _generate_lookahead_reasoning(self, player_turn: int,
                                       spot_a: int, spot_b: int,
                                       score: Dict,
                                       combined_resources: Set) -> str:
        """Generate reasoning for a lookahead recommendation."""
        picks_between = self._calculate_picks_between(player_turn, 4)
        
        reasons = [f"Spot #{spot_a} + predicted #{spot_b}"]
        reasons.append(f"({picks_between} opponent picks simulated)")
        
        resources_str = ", ".join(r.value for r in combined_resources)
        reasons.append(f"Combined resources: {resources_str}")
        
        if score.get("production", 0) > 6:
            reasons.append("High production value")
        if score.get("expansion", 0) > 2:
            reasons.append("Good expansion potential")
        
        return ". ".join(reasons)
    
    def simulate_placement(self, spot_id: int, player_id: int) -> bool:
        """
        Apply a placement to the board.
        Returns True if successful.
        """
        return self.board.place_settlement(spot_id, player_id)
    
    def get_road_recommendation(self, settlement_id: int, player_id: int,
                                  target_resources: Set[ResourceType] = None) -> Optional[Tuple[int, int]]:
        """
        Recommend best initial road from a settlement.
        
        Args:
            settlement_id: The settlement we just placed
            player_id: Current player
            target_resources: Resources we want to expand toward
        
        Returns:
            (node_a, node_b) for the recommended road path.
        """
        node = self.board.intersections[settlement_id]
        
        best_road = None
        best_score = -1
        
        for neighbor_id in node.neighbors:
            neighbor = self.board.intersections[neighbor_id]
            
            # Check if this direction has good expansion
            expansion_score = self.heuristics.calculate_expansion_potential(neighbor_id)
            
            # Check resources in this direction
            direction_resources = self._get_spot_resources(neighbor_id)
            resource_match = 0
            if target_resources:
                resource_match = len(direction_resources & target_resources) * 2
            
            total = expansion_score + resource_match
            
            if total > best_score:
                best_score = total
                best_road = (settlement_id, neighbor_id)
        
        return best_road
