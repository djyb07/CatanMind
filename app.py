"""
CatanMind Mobile - Flet UI Application
Mobile-friendly Catan AI Assistant with visual board and recommendations.
"""

import flet as ft
import io
import base64
import math
from typing import List, Dict, Optional, Set

# Import our logic modules
from models import Board, Player, ResourceType, Tile, BuildingType, PortType
from heuristics import HeuristicEngine
from solver_initial import InitialPlacementSolver, PlacementRecommendation
from strategy_manager import StrategyManager, GamePhase
from solver_midgame import MidGameSolver
from validators import Validator


# Color scheme
COLORS = {
    "background": "#1a1a2e",
    "surface": "#16213e",
    "primary": "#e94560",
    "secondary": "#0f3460",
    "text": "#eaeaea",
    "wood": "#228B22",
    "brick": "#B22222",
    "sheep": "#90EE90",
    "wheat": "#FFD700",
    "ore": "#708090",
    "desert": "#DEB887",
}

RESOURCE_COLORS = {
    ResourceType.WOOD: "#2d5a27",
    ResourceType.BRICK: "#8b3a3a",
    ResourceType.SHEEP: "#7cb342",
    ResourceType.WHEAT: "#ffc107",
    ResourceType.ORE: "#546e7a",
    ResourceType.DESERT: "#d4a574",
}


class BoardRenderer:
    """Renders the board as a matplotlib figure and returns a base64 PNG."""
    
    def __init__(self, board: Board):
        self.board = board
    
    def render_to_base64(self, 
                         highlighted_nodes: List[int] = None,
                         recommendations: List[PlacementRecommendation] = None) -> str:
        """
        Render board to base64 PNG string.
        
        Args:
            highlighted_nodes: Node IDs to highlight
            recommendations: Placement recommendations to display
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import RegularPolygon
        
        fig, ax = plt.subplots(figsize=(8, 8), facecolor=COLORS["background"])
        ax.set_facecolor(COLORS["background"])
        ax.set_aspect('equal')
        
        # Draw hexagonal tiles
        for (q, r), tile in self.board.tiles.items():
            x, y = self._hex_to_pixel(q, r)
            color = RESOURCE_COLORS.get(tile.resource_type, "#888888")
            
            hex_patch = RegularPolygon(
                (x, y), numVertices=6, radius=48,
                orientation=math.pi/6,
                facecolor=color, edgecolor='#333333', linewidth=2
            )
            ax.add_patch(hex_patch)
            
            # Draw dice number
            if tile.dice_number > 0:
                # Circle background for number
                circle = plt.Circle((x, y), 15, color='white', zorder=5)
                ax.add_patch(circle)
                
                # Red for 6 and 8
                num_color = '#e94560' if tile.dice_number in [6, 8] else '#333333'
                ax.text(x, y, str(tile.dice_number), 
                       ha='center', va='center', fontsize=14, 
                       fontweight='bold', color=num_color, zorder=6)
                
                # Pips (dots)
                pips = self._get_pips(tile.dice_number)
                ax.text(x, y-22, '•' * pips, ha='center', va='center',
                       fontsize=8, color='#666666')
        
        # Draw paths (roads)
        for path in self.board.paths:
            node_a = self.board.intersections[path.node_a]
            node_b = self.board.intersections[path.node_b]
            
            color = '#444444'
            width = 1
            if path.owner is not None:
                color = self._get_player_color(path.owner)
                width = 4
            
            ax.plot([node_a.x, node_b.x], [node_a.y, node_b.y],
                   color=color, linewidth=width, zorder=2)
        
        # Draw intersections
        recommended_nodes = set()
        if recommendations:
            for rec in recommendations[:3]:
                recommended_nodes.add(rec.node_id)
                if rec.complementary_spot:
                    recommended_nodes.add(rec.complementary_spot)
        
        for node_id, node in self.board.intersections.items():
            x, y = node.x, node.y
            
            if node.owner is not None:
                # Player building
                color = self._get_player_color(node.owner)
                if node.building_type == BuildingType.CITY:
                    marker = plt.Circle((x, y), 10, color=color, zorder=10)
                else:
                    marker = plt.Circle((x, y), 7, color=color, zorder=10)
                ax.add_patch(marker)
            elif node_id in recommended_nodes:
                # Recommended spot - highlight
                marker = plt.Circle((x, y), 8, color='#e94560', 
                                   fill=False, linewidth=3, zorder=10)
                ax.add_patch(marker)
                ax.text(x+12, y+12, f"#{node_id}", fontsize=8, 
                       color='#e94560', zorder=11)
            elif highlighted_nodes and node_id in highlighted_nodes:
                # Highlighted spot
                marker = plt.Circle((x, y), 6, color='#ffc107', zorder=10)
                ax.add_patch(marker)
            else:
                # Normal intersection
                marker = plt.Circle((x, y), 3, color='#666666', zorder=8)
                ax.add_patch(marker)
            
            # Port indicator
            if node.port_type:
                port_text = "3:1" if node.port_type == PortType.GENERIC else "2:1"
                ax.text(x, y-15, port_text, ha='center', va='center',
                       fontsize=6, color='#00bcd4', zorder=12)
        
        # Set axis limits
        all_x = [n.x for n in self.board.intersections.values()]
        all_y = [n.y for n in self.board.intersections.values()]
        margin = 60
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        ax.axis('off')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                   facecolor=COLORS["background"], edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        
        return base64.b64encode(buf.read()).decode('utf-8')
    
    def _hex_to_pixel(self, q: int, r: int, size: float = 50.0) -> tuple:
        x = size * (3/2 * q)
        y = size * (math.sqrt(3)/2 * q + math.sqrt(3) * r)
        return (x, y)
    
    def _get_pips(self, number: int) -> int:
        pips = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
        return pips.get(number, 0)
    
    def _get_player_color(self, player_id: int) -> str:
        colors = {1: "#e94560", 2: "#0096FF", 3: "#00c853", 4: "#ff9800"}
        return colors.get(player_id, "#ffffff")


class CatanMindApp:
    """Main Flet application for CatanMind."""
    
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "CatanMind"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.bgcolor = COLORS["background"]
        self.page.padding = 10
        
        # Game state
        self.board = Board()
        self.players: Dict[int, Player] = {}
        self.current_player_id = 1
        self.game_phase = "setup"  # setup, initial, playing
        self.turn_order = 1
        
        # Engines
        self.heuristics = HeuristicEngine(self.board)
        self.initial_solver = InitialPlacementSolver(self.board)
        self.strategy_manager = StrategyManager(self.board)
        self.midgame_solver = MidGameSolver(self.board)
        self.validator = Validator(self.board)
        self.renderer = BoardRenderer(self.board)
        
        # UI components
        self.board_image = ft.Image(
            src_base64=self.renderer.render_to_base64(),
            fit=ft.ImageFit.CONTAIN,
            width=350,
            height=350
        )
        
        self.recommendation_text = ft.Text(
            "Configure your turn order and press START GAME",
            size=14,
            color=COLORS["text"],
            text_align=ft.TextAlign.CENTER
        )
        
        self.phase_text = ft.Text(
            "Setup Phase",
            size=12,
            color=COLORS["primary"],
            weight=ft.FontWeight.BOLD
        )
        
        self.alerts_column = ft.Column(spacing=5)
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the main UI layout."""
        
        # Header
        header = ft.Container(
            content=ft.Row([
                ft.Icon(ft.icons.PSYCHOLOGY, color=COLORS["primary"], size=30),
                ft.Text("CatanMind", size=24, weight=ft.FontWeight.BOLD,
                       color=COLORS["text"]),
            ], alignment=ft.MainAxisAlignment.CENTER),
            padding=10
        )
        
        # Turn order selector
        self.turn_dropdown = ft.Dropdown(
            label="Your Turn Order",
            options=[
                ft.dropdown.Option("1", "Player 1 (First)"),
                ft.dropdown.Option("2", "Player 2"),
                ft.dropdown.Option("3", "Player 3"),
                ft.dropdown.Option("4", "Player 4 (Last)"),
            ],
            value="1",
            width=200,
            on_change=self._on_turn_change
        )
        
        # Start button
        self.start_button = ft.ElevatedButton(
            "START GAME",
            icon=ft.icons.PLAY_ARROW,
            on_click=self._on_start_game,
            style=ft.ButtonStyle(
                bgcolor=COLORS["primary"],
                color=COLORS["text"],
                padding=20
            ),
            width=200,
            height=50
        )
        
        # Setup section
        setup_section = ft.Container(
            content=ft.Column([
                self.turn_dropdown,
                ft.Container(height=10),
                self.start_button
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=10,
            visible=True
        )
        self.setup_section = setup_section
        
        # Board display
        board_card = ft.Container(
            content=ft.Column([
                self.phase_text,
                self.board_image
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
            bgcolor=COLORS["surface"],
            border_radius=10,
            padding=10
        )
        
        # Recommendation card
        recommendation_card = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.icons.LIGHTBULB, color="#ffc107"),
                    ft.Text("Recommendation", weight=ft.FontWeight.BOLD,
                           color=COLORS["text"])
                ]),
                ft.Divider(color=COLORS["secondary"]),
                self.recommendation_text
            ]),
            bgcolor=COLORS["surface"],
            border_radius=10,
            padding=15
        )
        
        # Alerts section
        alerts_card = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.icons.WARNING, color="#ff9800"),
                    ft.Text("Alerts", weight=ft.FontWeight.BOLD,
                           color=COLORS["text"])
                ]),
                self.alerts_column
            ]),
            bgcolor=COLORS["surface"],
            border_radius=10,
            padding=15
        )
        self.alerts_card = alerts_card
        
        # Action buttons
        self.action_buttons = ft.Column([
            ft.ElevatedButton(
                "I Built Settlement",
                icon=ft.icons.HOME,
                on_click=self._on_built_settlement,
                width=200,
                style=ft.ButtonStyle(bgcolor=COLORS["secondary"])
            ),
            ft.ElevatedButton(
                "I Built Road",
                icon=ft.icons.LINEAR_SCALE,
                on_click=self._on_built_road,
                width=200,
                style=ft.ButtonStyle(bgcolor=COLORS["secondary"])
            ),
            ft.ElevatedButton(
                "Enemy Built",
                icon=ft.icons.PERSON_OFF,
                on_click=self._on_enemy_built,
                width=200,
                style=ft.ButtonStyle(bgcolor="#8b0000")
            ),
            ft.ElevatedButton(
                "Next Turn",
                icon=ft.icons.SKIP_NEXT,
                on_click=self._on_next_turn,
                width=200,
                style=ft.ButtonStyle(bgcolor=COLORS["primary"])
            ),
        ], spacing=10, visible=False)
        
        # Main scrollable content
        content = ft.Column([
            header,
            setup_section,
            board_card,
            recommendation_card,
            alerts_card,
            ft.Container(height=10),
            ft.Container(
                content=self.action_buttons,
                alignment=ft.alignment.center
            ),
            ft.Container(height=20),
        ],
        scroll=ft.ScrollMode.AUTO,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        expand=True)
        
        self.page.add(content)
    
    def _on_turn_change(self, e):
        """Handle turn order selection change."""
        self.turn_order = int(self.turn_dropdown.value)
    
    def _on_start_game(self, e):
        """Start the game and calculate initial placements."""
        self.turn_order = int(self.turn_dropdown.value)
        self.game_phase = "initial"
        
        # Initialize players
        for i in range(1, 5):
            self.players[i] = Player(
                id=i, 
                name=f"Player {i}",
                color=["#e94560", "#0096FF", "#00c853", "#ff9800"][i-1]
            )
        
        self.current_player_id = self.turn_order
        
        # Hide setup, show action buttons
        self.setup_section.visible = False
        self.action_buttons.visible = True
        
        # Calculate recommendations
        self._update_recommendations()
        
        self.page.update()
    
    def _update_recommendations(self):
        """Update recommendations based on current game state."""
        
        if self.game_phase == "initial":
            self.phase_text.value = "Initial Placement Phase"
            
            # Get initial placement recommendations
            recommendations = self.initial_solver.get_best_starting_spots(
                self.turn_order
            )
            
            if recommendations:
                rec = recommendations[0]
                text = f"🎯 **Best Spot: #{rec.node_id}**\n"
                text += f"Resources: {', '.join(rec.resources)}\n"
                text += f"Numbers: {rec.numbers}\n"
                text += f"Score: {rec.score:.1f}\n\n"
                text += f"💡 {rec.reasoning}"
                
                if rec.complementary_spot:
                    text += f"\n\n📍 Complementary 2nd spot: #{rec.complementary_spot}"
                
                self.recommendation_text.value = text
                
                # Update board image with highlighted recommendations
                self.board_image.src_base64 = self.renderer.render_to_base64(
                    recommendations=recommendations
                )
            
        elif self.game_phase == "playing":
            player = self.players[self.current_player_id]
            phase = self.strategy_manager.get_phase(player)
            
            self.phase_text.value = self.strategy_manager.get_phase_description(phase)
            
            # Get best move
            opponents = [p for pid, p in self.players.items() if pid != self.current_player_id]
            best_move = self.strategy_manager.get_next_best_move(player, opponents)
            
            if best_move:
                text = f"🎯 **{best_move.action.replace('_', ' ').title()}**\n"
                if best_move.target:
                    text += f"Target: #{best_move.target}\n"
                text += f"Priority: {best_move.priority:.1f}\n\n"
                text += f"💡 {best_move.reasoning}"
                
                self.recommendation_text.value = text
            else:
                self.recommendation_text.value = "No immediate recommendations. Consider trading or saving resources."
            
            # Update alerts
            self.alerts_column.controls.clear()
            alerts = self.strategy_manager.get_resource_alerts(player)
            for alert in alerts:
                self.alerts_column.controls.append(
                    ft.Text(alert, size=12, color="#ff9800")
                )
            
            self.board_image.src_base64 = self.renderer.render_to_base64()
        
        self.page.update()
    
    def _on_built_settlement(self, e):
        """Handle player building a settlement."""
        # Show dialog to input node ID
        node_input = ft.TextField(label="Node ID", keyboard_type=ft.KeyboardType.NUMBER)
        
        def close_dlg(e):
            dlg.open = False
            self.page.update()
        
        def confirm(e):
            try:
                node_id = int(node_input.value)
                success = self.board.place_settlement(node_id, self.current_player_id)
                if success:
                    self._update_recommendations()
                    if self.game_phase == "initial":
                        self.game_phase = "playing"
            except ValueError:
                pass
            dlg.open = False
            self.page.update()
        
        dlg = ft.AlertDialog(
            title=ft.Text("Build Settlement"),
            content=node_input,
            actions=[
                ft.TextButton("Cancel", on_click=close_dlg),
                ft.TextButton("Build", on_click=confirm)
            ]
        )
        
        self.page.dialog = dlg
        dlg.open = True
        self.page.update()
    
    def _on_built_road(self, e):
        """Handle player building a road."""
        node_a_input = ft.TextField(label="From Node", keyboard_type=ft.KeyboardType.NUMBER)
        node_b_input = ft.TextField(label="To Node", keyboard_type=ft.KeyboardType.NUMBER)
        
        def close_dlg(e):
            dlg.open = False
            self.page.update()
        
        def confirm(e):
            try:
                node_a = int(node_a_input.value)
                node_b = int(node_b_input.value)
                success = self.board.place_road(node_a, node_b, self.current_player_id)
                if success:
                    self._update_recommendations()
            except ValueError:
                pass
            dlg.open = False
            self.page.update()
        
        dlg = ft.AlertDialog(
            title=ft.Text("Build Road"),
            content=ft.Column([node_a_input, node_b_input], height=120),
            actions=[
                ft.TextButton("Cancel", on_click=close_dlg),
                ft.TextButton("Build", on_click=confirm)
            ]
        )
        
        self.page.dialog = dlg
        dlg.open = True
        self.page.update()
    
    def _on_enemy_built(self, e):
        """Handle enemy building."""
        player_dropdown = ft.Dropdown(
            label="Enemy Player",
            options=[ft.dropdown.Option(str(i), f"Player {i}") 
                    for i in range(1, 5) if i != self.current_player_id],
            width=150
        )
        type_dropdown = ft.Dropdown(
            label="Building Type",
            options=[
                ft.dropdown.Option("settlement", "Settlement"),
                ft.dropdown.Option("road", "Road"),
                ft.dropdown.Option("city", "City")
            ],
            width=150
        )
        node_input = ft.TextField(label="Node ID(s)", hint_text="e.g., 15 or 15,16 for road")
        
        def close_dlg(e):
            dlg.open = False
            self.page.update()
        
        def confirm(e):
            try:
                player_id = int(player_dropdown.value)
                building_type = type_dropdown.value
                
                if building_type == "road":
                    nodes = [int(n.strip()) for n in node_input.value.split(",")]
                    if len(nodes) >= 2:
                        self.board.place_road(nodes[0], nodes[1], player_id)
                elif building_type == "settlement":
                    node_id = int(node_input.value)
                    self.board.place_settlement(node_id, player_id)
                elif building_type == "city":
                    node_id = int(node_input.value)
                    self.board.upgrade_to_city(node_id, player_id)
                
                self._update_recommendations()
            except (ValueError, IndexError):
                pass
            dlg.open = False
            self.page.update()
        
        dlg = ft.AlertDialog(
            title=ft.Text("Enemy Built"),
            content=ft.Column([player_dropdown, type_dropdown, node_input], height=180),
            actions=[
                ft.TextButton("Cancel", on_click=close_dlg),
                ft.TextButton("Confirm", on_click=confirm)
            ]
        )
        
        self.page.dialog = dlg
        dlg.open = True
        self.page.update()
    
    def _on_next_turn(self, e):
        """Advance to next turn and recalculate."""
        if self.game_phase == "initial":
            self.game_phase = "playing"
        
        self._update_recommendations()


def main(page: ft.Page):
    """Main entry point for Flet app."""
    page.window_width = 400
    page.window_height = 800
    CatanMindApp(page)


if __name__ == "__main__":
    ft.app(target=main)
