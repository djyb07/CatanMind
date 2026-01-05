#!/usr/bin/env python3
"""
CatanMind Coordinate Diagnostic Script
Checks alignment between logical board coordinates and visual renderer.
"""

from models import Board
from app import BoardRenderer

def main():
    print("=" * 60)
    print("CatanMind Coordinate Diagnostic")
    print("=" * 60)
    
    # 1. Instantiate Board and Renderer
    board = Board()
    renderer = BoardRenderer(board)
    
    # 2. Config Check
    print("\n📐 CONFIGURATION CHECK:")
    print("-" * 40)
    print(f"models.py _get_hex_corners default size: 50.0 (hardcoded)")
    print(f"BoardRenderer.HEX_SIZE: {renderer.HEX_SIZE}")
    print(f"BoardRenderer.BOARD_SIZE: {renderer.BOARD_SIZE}")
    
    if renderer.HEX_SIZE == 50:
        print("✅ HEX_SIZE matches!")
    else:
        print("❌ HEX_SIZE MISMATCH - This causes scaling drift!")
    
    # 3. Viewport Check
    print("\n📏 VIEWPORT CHECK:")
    print("-" * 40)
    print(f"view_min_x: {renderer.view_min_x}")
    print(f"view_max_x: {renderer.view_max_x}")
    print(f"view_min_y: {renderer.view_min_y}")
    print(f"view_max_y: {renderer.view_max_y}")
    print(f"view_size:  {renderer.view_size}")
    
    # 4. Board Stats
    print("\n📊 BOARD STATS:")
    print("-" * 40)
    print(f"Tiles: {len(board.tiles)}")
    print(f"Intersections: {len(board.intersections)}")
    print(f"Paths: {len(board.paths)}")
    
    # Calculate actual node bounds
    all_x = [n.x for n in board.intersections.values()]
    all_y = [n.y for n in board.intersections.values()]
    print(f"\nActual Node X range: [{min(all_x):.1f}, {max(all_x):.1f}]")
    print(f"Actual Node Y range: [{min(all_y):.1f}, {max(all_y):.1f}]")
    
    # 5. Coordinate Mapping Table
    print("\n🗺️  COORDINATE MAPPING TABLE:")
    print("-" * 60)
    print(f"{'Node ID':<10} {'Logical (x, y)':<25} {'Screen (x, y)':<25}")
    print("-" * 60)
    
    # Sample nodes: 0, 10, 20, 30, 40, and last
    node_ids = list(board.intersections.keys())
    sample_ids = [node_ids[0], node_ids[10], node_ids[20], node_ids[30], node_ids[40], node_ids[-1]]
    
    for nid in sample_ids:
        node = board.intersections[nid]
        logical = (node.x, node.y)
        screen = renderer.node_to_screen(nid)
        print(f"{nid:<10} ({logical[0]:>8.1f}, {logical[1]:>8.1f})   ({screen[0]:>8.1f}, {screen[1]:>8.1f})")
    
    # 6. Screen Space Analysis
    print("\n📱 SCREEN SPACE ANALYSIS:")
    print("-" * 40)
    screen_coords = [renderer.node_to_screen(nid) for nid in node_ids]
    screen_x = [c[0] for c in screen_coords]
    screen_y = [c[1] for c in screen_coords]
    
    print(f"Screen X range: [{min(screen_x):.1f}, {max(screen_x):.1f}]")
    print(f"Screen Y range: [{min(screen_y):.1f}, {max(screen_y):.1f}]")
    print(f"Expected range: [0, {renderer.BOARD_SIZE}]")
    
    # Check if nodes are within bounds
    out_of_bounds = sum(1 for x, y in screen_coords if x < 0 or x > renderer.BOARD_SIZE or y < 0 or y > renderer.BOARD_SIZE)
    if out_of_bounds == 0:
        print("✅ All nodes within screen bounds!")
    else:
        print(f"❌ {out_of_bounds} nodes out of screen bounds!")
    
    print("\n" + "=" * 60)
    print("Diagnostic Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
