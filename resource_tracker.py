"""
CatanMind - Resource Tracker ("The Game Beast")
Card counting and robber recommendation engine.
Tracks predicted resource hands for all players based on game events.
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from models import Board, ResourceType, Player, BuildingType, BUILDING_COSTS


@dataclass
class PlayerHand:
    """Predicted hand state for a player."""
    resources: Dict[ResourceType, int] = field(default_factory=lambda: {
        ResourceType.WOOD: 0,
        ResourceType.BRICK: 0,
        ResourceType.SHEEP: 0,
        ResourceType.WHEAT: 0,
        ResourceType.ORE: 0
    })
    total_unknown: int = 0  # Cards we couldn't track
    
    def total_cards(self) -> int:
        """Total known cards."""
        return sum(self.resources.values()) + self.total_unknown
    
    def add(self, resource: ResourceType, amount: int = 1):
        """Add resources to hand."""
        if resource in self.resources:
            self.resources[resource] += amount
    
    def remove(self, resource: ResourceType, amount: int = 1) -> bool:
        """Remove resources from hand. Returns False if insufficient."""
        if resource in self.resources:
            if self.resources[resource] >= amount:
                self.resources[resource] -= amount
                return True
            else:
                # Take what we can, mark rest as unknown loss
                self.resources[resource] = 0
                return False
        return False
    
    def halve(self):
        """Halve the hand (robber penalty for >7 cards)."""
        if self.total_cards() > 7:
            discard_count = self.total_cards() // 2
            # Simple heuristic: remove equally from all
            per_resource = discard_count // 5
            remainder = discard_count % 5
            
            for resource in self.resources:
                to_remove = min(self.resources[resource], per_resource)
                self.resources[resource] -= to_remove
            
            # Mark remainder as unknown discard
            self.total_unknown = max(0, self.total_unknown - remainder)
    
    def get_majority_resource(self) -> Tuple[ResourceType, int]:
        """Get the resource type with the most cards."""
        max_resource = ResourceType.WOOD
        max_count = 0
        
        for resource, count in self.resources.items():
            if count > max_count:
                max_count = count
                max_resource = resource
        
        return (max_resource, max_count)


class ResourceTracker:
    """
    Tracks predicted resource hands for all players.
    
    Usage:
        tracker = ResourceTracker(board, num_players=4)
        tracker.on_dice_roll(8)  # Players get resources from 8-tiles
        tracker.on_build(2, "road")  # Player 2 built a road
        tracker.on_robber(3)  # Player 3 has >7 cards, must discard
        
        # Get advice
        who, resource = tracker.get_rob_recommendation(my_id=1)
    """
    
    def __init__(self, board: Board, num_players: int = 4):
        self.board = board
        self.num_players = num_players
        
        # Player hands (1-indexed)
        self.hands: Dict[int, PlayerHand] = {
            i: PlayerHand() for i in range(1, num_players + 1)
        }
        
        # Track which hex has the robber
        self.robber_hex: Optional[Tuple[int, int]] = None
        for (q, r), tile in board.tiles.items():
            if tile.has_robber:
                self.robber_hex = (q, r)
                break
        
        # History for undo
        self.history: List[Dict] = []
    
    def on_dice_roll(self, number: int) -> Dict[int, List[ResourceType]]:
        """
        Process a dice roll - distribute resources to players.
        
        Args:
            number: The dice roll result (2-12)
        
        Returns:
            Dict mapping player_id to list of resources gained.
        """
        if number == 7:
            # No resources on 7 - robber activation handled separately
            return {}
        
        # Save state for undo
        self._save_state("dice_roll", {"number": number})
        
        gains: Dict[int, List[ResourceType]] = {i: [] for i in range(1, self.num_players + 1)}
        
        # Find tiles with this number
        producing_tiles = [
            tile for tile in self.board.tiles.values()
            if tile.dice_number == number and not tile.has_robber
        ]
        
        # For each producing tile, find settlements/cities on its corners
        for tile in producing_tiles:
            resource = tile.resource_type
            if resource == ResourceType.DESERT:
                continue
            
            # Find intersections touching this tile
            for node in self.board.intersections.values():
                if tile in node.touching_tiles and node.owner is not None:
                    player_id = node.owner
                    
                    # Settlement = 1 resource, City = 2 resources
                    amount = 2 if node.building_type == BuildingType.CITY else 1
                    
                    for _ in range(amount):
                        self.hands[player_id].add(resource)
                        gains[player_id].append(resource)
        
        return gains
    
    def on_build(self, player_id: int, building_type: str) -> bool:
        """
        Process a build action - deduct resources from player's hand.
        
        Args:
            player_id: The player who built
            building_type: "road", "settlement", "city", or "development_card"
        
        Returns:
            True if player had enough resources (tracked).
        """
        if building_type not in BUILDING_COSTS:
            return False
        
        cost = BUILDING_COSTS[building_type]
        hand = self.hands[player_id]
        
        # Save state for undo
        self._save_state("build", {"player_id": player_id, "type": building_type})
        
        success = True
        for resource, amount in cost.items():
            if not hand.remove(resource, amount):
                success = False
        
        return success
    
    def on_robber(self, player_id: int):
        """
        Process robber penalty - player with >7 cards must discard half.
        
        Args:
            player_id: The player who rolled 7 and has >7 cards
        """
        self._save_state("robber", {"player_id": player_id})
        self.hands[player_id].halve()
    
    def on_robber_move(self, hex_q: int, hex_r: int, 
                       steal_from: Optional[int] = None,
                       resource_stolen: Optional[ResourceType] = None):
        """
        Process robber being moved to a new hex.
        
        Args:
            hex_q, hex_r: Axial coordinates of new robber position
            steal_from: Player ID we stole from (if any)
            resource_stolen: The resource we stole (if known)
        """
        # Update robber position on tiles
        if self.robber_hex:
            old_tile = self.board.tiles.get(self.robber_hex)
            if old_tile:
                old_tile.has_robber = False
        
        new_tile = self.board.tiles.get((hex_q, hex_r))
        if new_tile:
            new_tile.has_robber = True
            self.robber_hex = (hex_q, hex_r)
        
        # Process steal
        if steal_from is not None and resource_stolen is not None:
            self.hands[steal_from].remove(resource_stolen)
    
    def on_trade_with_bank(self, player_id: int, 
                            gave: Dict[ResourceType, int],
                            received: Dict[ResourceType, int]):
        """Process a bank/port trade."""
        hand = self.hands[player_id]
        
        for resource, amount in gave.items():
            hand.remove(resource, amount)
        
        for resource, amount in received.items():
            hand.add(resource, amount)
    
    def on_trade_with_player(self, player_a: int, player_b: int,
                              a_gives: Dict[ResourceType, int],
                              b_gives: Dict[ResourceType, int]):
        """Process a player-to-player trade."""
        for resource, amount in a_gives.items():
            self.hands[player_a].remove(resource, amount)
            self.hands[player_b].add(resource, amount)
        
        for resource, amount in b_gives.items():
            self.hands[player_b].remove(resource, amount)
            self.hands[player_a].add(resource, amount)
    
    def get_rob_recommendation(self, my_id: int, 
                                 needed_resource: Optional[ResourceType] = None
                                 ) -> Tuple[int, ResourceType, str]:
        """
        Get recommendation for who to rob.
        
        Args:
            my_id: The current player's ID (to exclude from targets)
            needed_resource: Optional - prefer stealing this resource
        
        Returns:
            Tuple of (player_id, likely_resource, reasoning)
        """
        best_target = None
        best_score = -1
        best_resource = ResourceType.WOOD
        reasoning = ""
        
        for player_id, hand in self.hands.items():
            if player_id == my_id:
                continue
            
            total_cards = hand.total_cards()
            if total_cards == 0:
                continue
            
            majority_resource, majority_count = hand.get_majority_resource()
            
            # Score based on card count and resource match
            score = total_cards * 2
            
            if needed_resource and hand.resources.get(needed_resource, 0) > 0:
                score += 5  # Bonus for having what we need
                majority_resource = needed_resource
            
            if majority_count >= 3:
                score += 3  # Bonus for high concentration
            
            if score > best_score:
                best_score = score
                best_target = player_id
                best_resource = majority_resource
                
                # Build reasoning
                reasoning = f"Player {player_id} has ~{total_cards} cards"
                if majority_count >= 2:
                    reasoning += f" ({majority_count} {majority_resource.value})"
        
        if best_target is None:
            return (1, ResourceType.WOOD, "No good targets")
        
        return (best_target, best_resource, reasoning)
    
    def get_robber_hex_recommendation(self, my_id: int) -> Tuple[Tuple[int, int], int, str]:
        """
        Get recommendation for where to place the robber.
        
        Returns:
            Tuple of (hex_coords, target_player_id, reasoning)
        """
        best_hex = None
        best_target = None
        best_score = -1
        reasoning = ""
        
        for (q, r), tile in self.board.tiles.items():
            if tile.resource_type == ResourceType.DESERT:
                continue
            if tile.has_robber:
                continue  # Can't place on current position
            
            # Find players on this hex
            players_here: Dict[int, int] = {}  # player_id -> building count
            
            for node in self.board.intersections.values():
                if tile in node.touching_tiles and node.owner is not None:
                    if node.owner != my_id:
                        multiplier = 2 if node.building_type == BuildingType.CITY else 1
                        players_here[node.owner] = players_here.get(node.owner, 0) + multiplier
            
            if not players_here:
                continue
            
            # Score this hex
            hex_score = tile.probability  # Higher probability = better block
            
            # Pick main target (player with most buildings here)
            target_player = max(players_here, key=players_here.get)
            target_building_score = players_here[target_player]
            
            # Add value of blocking their production
            target_hand = self.hands.get(target_player)
            if target_hand:
                hex_score += target_hand.total_cards() * 0.5
            
            hex_score += target_building_score * 2
            
            if hex_score > best_score:
                best_score = hex_score
                best_hex = (q, r)
                best_target = target_player
                reasoning = f"Block Player {target_player}'s {tile.resource_type.value} ({tile.dice_number})"
        
        if best_hex is None:
            # Fallback to any non-desert
            for (q, r), tile in self.board.tiles.items():
                if tile.resource_type != ResourceType.DESERT:
                    return ((q, r), 0, "No significant targets")
        
        return (best_hex, best_target, reasoning)
    
    def get_player_summary(self, player_id: int) -> str:
        """Get a summary of a player's predicted hand."""
        hand = self.hands[player_id]
        
        resources = []
        for resource, count in hand.resources.items():
            if count > 0:
                resources.append(f"{count} {resource.value}")
        
        if not resources:
            return f"Player {player_id}: Empty hand"
        
        return f"Player {player_id}: {', '.join(resources)}"
    
    def get_all_summaries(self, exclude_id: Optional[int] = None) -> List[str]:
        """Get summaries for all players."""
        summaries = []
        for player_id in sorted(self.hands.keys()):
            if player_id != exclude_id:
                summaries.append(self.get_player_summary(player_id))
        return summaries
    
    def _save_state(self, action: str, data: Dict):
        """Save state for undo functionality."""
        state = {
            "action": action,
            "data": data,
            "hands": {
                pid: {
                    "resources": dict(h.resources),
                    "total_unknown": h.total_unknown
                }
                for pid, h in self.hands.items()
            }
        }
        self.history.append(state)
        
        # Keep only last 20 states
        if len(self.history) > 20:
            self.history.pop(0)
    
    def undo(self) -> bool:
        """Undo the last action. Returns True if successful."""
        if len(self.history) < 2:
            return False
        
        # Remove current state
        self.history.pop()
        
        # Restore previous state
        prev_state = self.history[-1]
        for pid, hand_data in prev_state["hands"].items():
            self.hands[pid].resources = {
                ResourceType(k) if isinstance(k, str) else k: v 
                for k, v in hand_data["resources"].items()
            }
            self.hands[pid].total_unknown = hand_data["total_unknown"]
        
        return True
    
    def reset(self):
        """Reset all hands to empty."""
        for hand in self.hands.values():
            hand.resources = {r: 0 for r in hand.resources}
            hand.total_unknown = 0
        self.history.clear()
