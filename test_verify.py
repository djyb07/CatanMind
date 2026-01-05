"""Quick verification test for CatanMind."""

from models import Board
from heuristics import HeuristicEngine
from solver_initial import InitialPlacementSolver
from strategy_manager import StrategyManager
from solver_midgame import MidGameSolver
from validators import Validator

# Test Board creation
board = Board()
print(f"[OK] Board created: {len(board.tiles)} tiles, "
      f"{len(board.intersections)} intersections, {len(board.paths)} paths")

# Test Heuristics
heuristics = HeuristicEngine(board)
spots = heuristics.get_ranked_spots("early", top_n=5)
print(f"[OK] Heuristics: Top 5 spots = {[s[0] for s in spots]}")

# Test Initial Solver
solver = InitialPlacementSolver(board)
recs = solver.get_best_starting_spots(player_turn_index=1)
print(f"[OK] Initial Solver: Player 1 recommendations = "
      f"{[(r.node_id, r.complementary_spot) for r in recs[:3]]}")

# Test Player 4 (back-to-back picks)
recs4 = solver.get_best_starting_spots(player_turn_index=4)
print(f"[OK] Player 4 recommendations = "
      f"{[(r.node_id, r.complementary_spot) for r in recs4[:2]]}")

# Test Strategy Manager
from models import Player
player = Player(id=1, name="Test", color="red")
strategy = StrategyManager(board)
phase = strategy.get_phase(player)
print(f"[OK] Strategy Manager: Current phase = {phase.value}")

# Test Validator
validator = Validator(board)
valid_spots = validator.get_valid_settlement_spots(1, initial_phase=True)
print(f"[OK] Validator: {len(valid_spots)} valid initial settlement spots")

# Test Mid-Game Solver
midgame = MidGameSolver(board)
print("[OK] Mid-Game Solver initialized")

# Test Resource Tracker
from resource_tracker import ResourceTracker
tracker = ResourceTracker(board)
gains = tracker.on_dice_roll(8)
print(f"[OK] ResourceTracker: Dice roll 8 processed, gains: {len(gains)} players")

# Test robber recommendation
target, resource, reason = tracker.get_rob_recommendation(my_id=1)
print(f"[OK] ResourceTracker: Rob recommendation = Player {target}, {resource.value}")

# Test hex recommendation
hex_coords, hex_target, hex_reason = tracker.get_robber_hex_recommendation(my_id=1)
print(f"[OK] ResourceTracker: Robber hex = {hex_coords}, reason: {hex_reason[:30]}...")

print("\n" + "="*50)
print("All core modules verified successfully!")
print("="*50)


print("\n" + "="*50)
print("All core modules verified successfully!")
print("="*50)
