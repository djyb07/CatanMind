"""
CatanMind Mobile - Flet UI Application
Mobile-friendly Catan AI Assistant with INTERACTIVE CLICKABLE MAP.
No typing node IDs - just tap on the board!
"""

import flet as ft
import io
import base64
import math
from typing import List, Dict, Optional, Set, Tuple

# Import our logic modules
from models import Board, Player, ResourceType, Tile, BuildingType, PortType
from heuristics import HeuristicEngine
from solver_initial import InitialPlacementSolver, PlacementRecommendation
from strategy_manager import StrategyManager, GamePhase
from solver_midgame import MidGameSolver
from validators import Validator
from resource_tracker import ResourceTracker


# Color scheme
COLORS = {
    "background": "#1a1a2e",
    "surface": "#16213e",
    "primary": "#e94560",
    "secondary": "#0f3460",
    "text": "#eaeaea",
    "accent": "#ffc107",
    "success": "#00c853",
    "danger": "#ff5252",
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

PLAYER_COLORS = {
    1: "#e94560",
    2: "#0096FF", 
    3: "#00c853",
    4: "#ff9800"
}


class BoardRenderer:
    """Renders the board as a matplotlib figure and returns a base64 PNG."""
    
    BOARD_SIZE = 350
    HEX_SIZE = 48
    
    def __init__(self, board: Board):
        self.board = board
        self._calculate_absolute_bounds()
    
    def _calculate_absolute_bounds(self):
        """Calculate bounds based on vertices to ensure perfect fit."""
        all_x = []
        all_y = []
        
        # Calculate bounds using the CORRECTED coordinate system
        for (q, r) in self.board.tiles.keys():
            corners = self.board._get_hex_corners(q, r, self.HEX_SIZE)
            for cx, cy in corners:
                all_x.append(cx)
                all_y.append(cy)
        
        if not all_x:  # Safety check
            self.view_min_x = -100
            self.view_max_x = 100
            self.view_min_y = -100
            self.view_max_y = 100
            self.view_size = 200
            return

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Add padding (stroke width + margin)
        padding = self.HEX_SIZE * 0.6
        
        width = (max_x - min_x) + (2 * padding)
        height = (max_y - min_y) + (2 * padding)
        
        # Square the viewport
        max_dim = max(width, height)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        self.view_min_x = center_x - (max_dim / 2)
        self.view_max_x = center_x + (max_dim / 2)
        self.view_min_y = center_y - (max_dim / 2)
        self.view_max_y = center_y + (max_dim / 2)
        self.view_size = max_dim

    def get_render_params(self) -> Dict:
        """Required by UI to set container size."""
        return {
            "width": self.BOARD_SIZE,
            "height": self.BOARD_SIZE
        }

    def node_to_screen(self, node_id: int) -> Tuple[float, float]:
        """Convert node to screen coordinates."""
        if node_id not in self.board.intersections:
            return (0, 0)
            
        node = self.board.intersections[node_id]
        
        rel_x = (node.x - self.view_min_x) / self.view_size
        rel_y = (node.y - self.view_min_y) / self.view_size
        
        screen_x = rel_x * self.BOARD_SIZE
        screen_y = (1.0 - rel_y) * self.BOARD_SIZE
        return (screen_x, screen_y)
    
    def _hex_to_pixel(self, q: int, r: int, size: float = 50.0) -> tuple:
        """
        FIX: Match the Pointy-Top logic from models.py.
        Old Flat-Top logic caused the 30-degree rotation error.
        """
        x = size * (math.sqrt(3) * q + math.sqrt(3)/2 * r)
        y = size * (3/2 * r)
        return (x, y)
    
    def _get_pips(self, number: int) -> int:
        pips = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
        return pips.get(number, 0)

    def render_to_base64(self, 
                         highlighted_nodes: List[int] = None,
                         recommendations: List[PlacementRecommendation] = None) -> str:
        """Render board with corrected Pointy-Top orientation."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import RegularPolygon
        
        fig = plt.figure(figsize=(5, 5), facecolor=COLORS["background"])
        ax = fig.add_axes([0, 0, 1, 1], facecolor=COLORS["background"])
        ax.set_aspect('equal')
        
        # 1. Draw Tiles
        for (q, r), tile in self.board.tiles.items():
            x, y = self._hex_to_pixel(q, r, self.HEX_SIZE)
            color = RESOURCE_COLORS.get(tile.resource_type, "#888888")
            if tile.has_robber:
                color = "#333333"
            
            # Pointy Top orientation
            ax.add_patch(RegularPolygon(
                (x, y), numVertices=6, radius=self.HEX_SIZE,
                orientation=math.pi/6, facecolor=color, edgecolor='#222222', linewidth=2
            ))
            
            if tile.dice_number > 0 and not tile.has_robber:
                ax.add_patch(plt.Circle((x, y), 15, color='white', zorder=5))
                c = '#e94560' if tile.dice_number in [6, 8] else '#333333'
                ax.text(x, y, str(tile.dice_number), ha='center', va='center',
                       fontsize=13, fontweight='bold', color=c, zorder=6)
            if tile.has_robber:
                ax.text(x, y, "🏴‍☠️", ha='center', va='center', fontsize=18, zorder=6)

        # 2. Draw Roads
        for path in self.board.paths:
            n_a = self.board.intersections[path.node_a]
            n_b = self.board.intersections[path.node_b]
            if path.owner is not None:
                c = PLAYER_COLORS.get(path.owner, "#ffffff")
                ax.plot([n_a.x, n_b.x], [n_a.y, n_b.y], color=c, linewidth=5, zorder=4)
                ax.plot([n_a.x, n_b.x], [n_a.y, n_b.y], color='black', linewidth=7, zorder=3)
            else:
                ax.plot([n_a.x, n_b.x], [n_a.y, n_b.y], color='#333333', linewidth=1, zorder=1, alpha=0.3)

        # 3. Draw Nodes
        rec_ids = set()
        if recommendations:
            for r in recommendations[:3]:
                rec_ids.add(r.node_id)
                if r.complementary_spot:
                    rec_ids.add(r.complementary_spot)

        for nid, node in self.board.intersections.items():
            if node.owner is not None:
                c = PLAYER_COLORS.get(node.owner, "white")
                r = 12 if node.building_type == BuildingType.CITY else 8
                ax.add_patch(plt.Circle((node.x, node.y), r + 2, color='black', zorder=9))
                ax.add_patch(plt.Circle((node.x, node.y), r, color=c, zorder=10))
            elif nid in rec_ids:
                ax.add_patch(plt.Circle((node.x, node.y), 12, color='#e94560', alpha=0.6, zorder=15))
            elif highlighted_nodes and nid in highlighted_nodes:
                ax.add_patch(plt.Circle((node.x, node.y), 8, color='#ffc107', zorder=15))

        ax.set_xlim(self.view_min_x, self.view_max_x)
        ax.set_ylim(self.view_min_y, self.view_max_y)
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor=COLORS["background"], edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')


class InteractiveBoard:
    """
    Interactive board with clickable node overlays.
    Uses ft.Stack to layer clickable buttons over the board image.
    Note: This is now a simple data class - the actual Stack is built in CatanMindApp.
    """
    
    def __init__(self, board: Board, renderer: BoardRenderer, 
                 on_node_click, on_road_click=None):
        self.board = board
        self.renderer = renderer
        self.on_node_click = on_node_click
        self.on_road_click = on_road_click
        self.recommendations: List[PlacementRecommendation] = []


class CatanMindApp:
    """Main Flet application for CatanMind with Interactive Clickable Map."""
    
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
        self.game_phase = "setup"
        self.turn_order = 1
        
        # Engines
        self.heuristics = HeuristicEngine(self.board)
        self.initial_solver = InitialPlacementSolver(self.board)
        self.strategy_manager = StrategyManager(self.board)
        self.midgame_solver = MidGameSolver(self.board)
        self.validator = Validator(self.board)
        self.renderer = BoardRenderer(self.board)
        self.resource_tracker = ResourceTracker(self.board, num_players=4)
        
        # Track current recommendations
        self.current_recommendations: List[PlacementRecommendation] = []
        
        # Build UI
        self._build_ui()
    
    def _build_ui(self):
        """Build the main UI layout."""
        
        # Header
        header = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.PSYCHOLOGY, color=COLORS["primary"], size=30),
                ft.Text("CatanMind", size=24, weight=ft.FontWeight.BOLD,
                       color=COLORS["text"]),
            ], alignment=ft.MainAxisAlignment.CENTER),
            padding=10
        )
        
        # Phase indicator
        self.phase_text = ft.Text(
            "Setup Phase",
            size=14,
            color=COLORS["primary"],
            weight=ft.FontWeight.BOLD
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
        )
        # Assign event handler separately (fixes mobile crash)
        self.turn_dropdown.on_change = self._on_turn_change
        
        # Start button
        self.start_button = ft.ElevatedButton(
            "START GAME",
            icon=ft.Icons.PLAY_ARROW,
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
        self.setup_section = ft.Container(
            content=ft.Column([
                self.turn_dropdown,
                ft.Container(height=10),
                self.start_button
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=10,
            visible=True
        )
        
        # Interactive Board
        self.interactive_board = ft.Container(
            content=ft.Stack([
                ft.Image(
                    src=f"data:image/png;base64,{self.renderer.render_to_base64()}",
                    width=350,
                    height=350,
                    fit="contain"
                )
            ]),
            bgcolor=COLORS["surface"],
            border_radius=10,
            padding=10,
        )
        self.board_image = self.interactive_board.content.controls[0]
        
        # Recommendation card
        self.recommendation_text = ft.Text(
            "Tap START GAME to begin. Select your turn order first.",
            size=14,
            color=COLORS["text"],
            text_align=ft.TextAlign.CENTER
        )
        
        recommendation_card = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.LIGHTBULB, color=COLORS["accent"]),
                    ft.Text("AI Recommendation", weight=ft.FontWeight.BOLD,
                           color=COLORS["text"])
                ]),
                ft.Divider(color=COLORS["secondary"], height=1),
                self.recommendation_text
            ]),
            bgcolor=COLORS["surface"],
            border_radius=10,
            padding=15
        )
        
        # Resource Tracker Display
        self.tracker_text = ft.Text(
            "",
            size=12,
            color=COLORS["text"],
        )
        
        tracker_card = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.ANALYTICS, color=COLORS["success"]),
                    ft.Text("Card Counter", weight=ft.FontWeight.BOLD,
                           color=COLORS["text"])
                ]),
                self.tracker_text
            ]),
            bgcolor=COLORS["surface"],
            border_radius=10,
            padding=15,
            visible=False
        )
        self.tracker_card = tracker_card
        
        # Alerts section
        self.alerts_column = ft.Column(spacing=5)
        alerts_card = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.WARNING, color=COLORS["accent"]),
                    ft.Text("Alerts", weight=ft.FontWeight.BOLD,
                           color=COLORS["text"])
                ]),
                self.alerts_column
            ]),
            bgcolor=COLORS["surface"],
            border_radius=10,
            padding=15,
            visible=False
        )
        self.alerts_card = alerts_card
        
        # Dice Roll Button (always visible during play)
        self.dice_button = ft.ElevatedButton(
            "🎲 DICE ROLL",
            on_click=self._on_dice_roll,
            style=ft.ButtonStyle(
                bgcolor=COLORS["accent"],
                color="#000000",
                padding=15,
                text_style=ft.TextStyle(size=18, weight=ft.FontWeight.BOLD)
            ),
            width=200,
            height=60,
            visible=False
        )
        
        # Action buttons (Safe Hex Colors)
        self.action_buttons = ft.Column([
            ft.Row([
                ft.ElevatedButton(
                    "My Settlement",
                    icon=ft.Icons.HOME,
                    on_click=lambda e: self._show_node_action_sheet("settlement_self"),
                    style=ft.ButtonStyle(bgcolor="#2E7D32", color="white"),  # Green
                    expand=True
                ),
                ft.ElevatedButton(
                    "Enemy Build",
                    icon=ft.Icons.PERSON_OFF,
                    on_click=lambda e: self._show_enemy_build_dialog(),
                    style=ft.ButtonStyle(bgcolor="#C62828", color="white"),  # Red
                    expand=True
                ),
            ]),
            ft.Row([
                ft.ElevatedButton(
                    "My City",
                    icon=ft.Icons.LOCATION_CITY,
                    on_click=lambda e: self._show_node_action_sheet("city_self"),
                    style=ft.ButtonStyle(bgcolor=COLORS["secondary"], color="white"),
                    expand=True
                ),
                ft.ElevatedButton(
                    "My Road",
                    icon=ft.Icons.LINEAR_SCALE,
                    on_click=lambda e: self._show_road_dialog(),
                    style=ft.ButtonStyle(bgcolor=COLORS["secondary"], color="white"),
                    expand=True
                ),
            ]),
            ft.OutlinedButton(
                "↩️ Undo Last",
                on_click=self._on_undo,
                style=ft.ButtonStyle(color="grey"),
                width=200
            )
        ], spacing=10, visible=False)
        
        # Main scrollable content
        content = ft.Column([
            header,
            self.phase_text,
            self.setup_section,
            ft.Container(
                content=ft.Column([
                    ft.Text("Tap a node on the board to interact", 
                           size=12, color="#888888", italic=True),
                    self.interactive_board,
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
                alignment=ft.Alignment(0, 0),
            ),
            self.dice_button,
            recommendation_card,
            tracker_card,
            alerts_card,
            ft.Container(height=10),
            self.action_buttons,
            ft.Container(height=20),
        ],
        scroll=ft.ScrollMode.AUTO,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        expand=True)
        
        self.page.add(content)
        
        # Build the interactive overlay
        self._rebuild_board_overlay()
    
    def _rebuild_board_overlay(self):
        """Rebuild the interactive board overlay with clickable nodes."""
        params = self.renderer.get_render_params()
        
        # Start with the base image
        controls = [self.board_image]
        
        # Get recommended nodes
        recommended_ids = set()
        if self.current_recommendations:
            for rec in self.current_recommendations[:3]:
                recommended_ids.add(rec.node_id)
                if rec.complementary_spot:
                    recommended_ids.add(rec.complementary_spot)
        
        # Add clickable overlays for each node
        for node_id, node in self.board.intersections.items():
            screen_x, screen_y = self.renderer.node_to_screen(node_id)
            
            # Button size based on state
            if node.owner is not None:
                btn_size = 24 if node.building_type == BuildingType.CITY else 20
                btn_color = PLAYER_COLORS.get(node.owner, "#ffffff")
                opacity = 0.7
            elif node_id in recommended_ids:
                btn_size = 26
                btn_color = COLORS["primary"]
                opacity = 0.5
            else:
                btn_size = 20
                btn_color = "transparent"
                opacity = 0.01  # Nearly invisible but still clickable
            
            # Create overlay button
            overlay = ft.Container(
                content=None,
                width=btn_size,
                height=btn_size,
                border_radius=btn_size // 2,
                bgcolor=btn_color if btn_color != "transparent" else None,
                border=ft.border.all(3, COLORS["primary"]) if node_id in recommended_ids else None,
                left=screen_x - btn_size / 2,
                top=screen_y - btn_size / 2,
                on_click=lambda e, nid=node_id: self._on_node_clicked(nid),
                opacity=opacity,
            )
            controls.append(overlay)
        
        # Update the stack
        self.interactive_board.content = ft.Stack(
            controls,
            width=params["width"],
            height=params["height"],
        )
    
    def _on_turn_change(self, e):
        """Handle turn order selection change."""
        self.turn_order = int(self.turn_dropdown.value)
    
    def _on_start_game(self, e):
        """Start the game and show recommendations."""
        self.turn_order = int(self.turn_dropdown.value)
        self.game_phase = "initial"
        
        # Initialize players
        for i in range(1, 5):
            self.players[i] = Player(
                id=i, 
                name=f"Player {i}",
                color=list(PLAYER_COLORS.values())[i-1]
            )
        
        self.current_player_id = self.turn_order
        
        # Hide setup, show game controls
        self.setup_section.visible = False
        self.action_buttons.visible = True
        self.dice_button.visible = True
        self.tracker_card.visible = True
        self.alerts_card.visible = True
        
        # Calculate recommendations
        self._update_recommendations()
        
        self.page.update()
    
    def _update_recommendations(self):
        """Update AI recommendations based on current state."""
        
        if self.game_phase == "initial":
            self.phase_text.value = "🏠 Initial Placement Phase"
            
            recommendations = self.initial_solver.get_best_starting_spots(
                self.turn_order
            )
            self.current_recommendations = recommendations
            
            if recommendations:
                rec = recommendations[0]
                text = f"🎯 Best Spot: Node #{rec.node_id}\n"
                text += f"📦 {', '.join(rec.resources)}\n"
                text += f"🎲 Numbers: {rec.numbers}\n"
                text += f"⭐ Score: {rec.score:.1f}\n\n"
                text += f"💡 {rec.reasoning}"
                
                if rec.complementary_spot:
                    text += f"\n\n📍 2nd spot: #{rec.complementary_spot}"
                
                self.recommendation_text.value = text
            
        else:  # playing phase
            player = self.players[self.current_player_id]
            phase = self.strategy_manager.get_phase(player)
            
            phase_names = {
                GamePhase.EARLY: "🌱 Early Game",
                GamePhase.MID: "⚔️ Mid Game", 
                GamePhase.LATE: "🏆 Late Game"
            }
            self.phase_text.value = phase_names.get(phase, "Playing")
            
            opponents = [p for pid, p in self.players.items() if pid != self.current_player_id]
            best_move = self.strategy_manager.get_next_best_move(player, opponents)
            
            if best_move:
                text = f"🎯 {best_move.action.replace('_', ' ').title()}\n"
                if best_move.target:
                    text += f"📍 Target: #{best_move.target}\n"
                text += f"⭐ Priority: {best_move.priority:.1f}\n\n"
                text += f"💡 {best_move.reasoning}"
                
                self.recommendation_text.value = text
            else:
                self.recommendation_text.value = "No immediate recommendations. Trade or save resources."
            
            # Update alerts
            self.alerts_column.controls.clear()
            alerts = self.strategy_manager.get_resource_alerts(player)
            for alert in alerts:
                self.alerts_column.controls.append(
                    ft.Text(f"⚠️ {alert}", size=12, color=COLORS["accent"])
                )
        
        # Update tracker display
        summaries = self.resource_tracker.get_all_summaries(exclude_id=self.current_player_id)
        self.tracker_text.value = "\n".join(summaries) if summaries else "No data yet. Roll dice to start tracking."
        
        # Refresh board
        self.board_image.src = f"data:image/png;base64,{self.renderer.render_to_base64(recommendations=self.current_recommendations)}"
        self._rebuild_board_overlay()
        
        self.page.update()
    
    def _on_node_clicked(self, node_id: int):
        """Handle click on a board node - show bottom sheet with options."""
        node = self.board.intersections[node_id]
        
        # Build info about this node
        resources = [t.resource_type.value for t in node.touching_tiles 
                    if t.resource_type != ResourceType.DESERT]
        numbers = [t.dice_number for t in node.touching_tiles if t.dice_number > 0]
        
        info_text = f"Node #{node_id}\n"
        info_text += f"Resources: {', '.join(resources)}\n" if resources else "No resources\n"
        info_text += f"Numbers: {numbers}" if numbers else ""
        
        if node.port_type:
            info_text += f"\n🚢 Port: {node.port_type.value}"
        
        # Build action options based on node state
        actions = []
        
        if node.owner is None:
            # Empty node - can build
            actions.append(
                ft.ListTile(
                    leading=ft.Icon(ft.Icons.HOME, color=COLORS["success"]),
                    title=ft.Text("I Built Settlement"),
                    on_click=lambda e, nid=node_id: self._build_settlement(nid, self.current_player_id)
                )
            )
            actions.append(
                ft.ListTile(
                    leading=ft.Icon(ft.Icons.PERSON_OFF, color=COLORS["danger"]),
                    title=ft.Text("Enemy Built Settlement"),
                    on_click=lambda e, nid=node_id: self._show_enemy_player_picker(nid)
                )
            )
        elif node.owner == self.current_player_id:
            # My building
            if node.building_type == BuildingType.SETTLEMENT:
                actions.append(
                    ft.ListTile(
                        leading=ft.Icon(ft.Icons.LOCATION_CITY, color=COLORS["accent"]),
                        title=ft.Text("Upgrade to City"),
                        on_click=lambda e, nid=node_id: self._upgrade_to_city(nid)
                    )
                )
        else:
            # Enemy building
            info_text += f"\n👤 Owner: Player {node.owner}"
        
        actions.append(
            ft.ListTile(
                leading=ft.Icon(ft.Icons.CLOSE, color="#888888"),
                title=ft.Text("Cancel"),
                on_click=lambda e: self._close_bottom_sheet()
            )
        )
        
        # Show bottom sheet
        self.page.bottom_sheet = ft.BottomSheet(
            content=ft.Container(
                content=ft.Column([
                    ft.Text(info_text, size=14, color=COLORS["text"]),
                    ft.Divider(color=COLORS["secondary"]),
                    *actions
                ], tight=True),
                padding=20,
                bgcolor=COLORS["surface"],
            ),
            open=True
        )
        self.page.update()
    
    def _close_bottom_sheet(self):
        """Close the bottom sheet."""
        if self.page.bottom_sheet:
            self.page.bottom_sheet.open = False
            self.page.update()
    
    def _build_settlement(self, node_id: int, player_id: int):
        """Build a settlement at the given node."""
        self._close_bottom_sheet()
        
        success = self.board.place_settlement(node_id, player_id)
        if success:
            # Deduct resources from tracker
            self.resource_tracker.on_build(player_id, "settlement")
            
            if self.game_phase == "initial":
                self.game_phase = "playing"
            
            self._update_recommendations()
    
    def _upgrade_to_city(self, node_id: int):
        """Upgrade a settlement to city."""
        self._close_bottom_sheet()
        
        success = self.board.upgrade_to_city(node_id, self.current_player_id)
        if success:
            self.resource_tracker.on_build(self.current_player_id, "city")
            self._update_recommendations()
    
    def _show_enemy_player_picker(self, node_id: int):
        """Show picker for which enemy player built here."""
        self._close_bottom_sheet()
        
        options = []
        for i in range(1, 5):
            if i != self.current_player_id:
                options.append(
                    ft.ListTile(
                        leading=ft.Container(
                            width=20, height=20, 
                            bgcolor=PLAYER_COLORS[i],
                            border_radius=10
                        ),
                        title=ft.Text(f"Player {i}"),
                        on_click=lambda e, pid=i, nid=node_id: self._build_enemy_settlement(nid, pid)
                    )
                )
        
        self.page.bottom_sheet = ft.BottomSheet(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Which player built here?", size=14, weight=ft.FontWeight.BOLD),
                    *options,
                    ft.ListTile(
                        leading=ft.Icon(ft.Icons.CLOSE),
                        title=ft.Text("Cancel"),
                        on_click=lambda e: self._close_bottom_sheet()
                    )
                ], tight=True),
                padding=20,
                bgcolor=COLORS["surface"],
            ),
            open=True
        )
        self.page.update()
    
    def _build_enemy_settlement(self, node_id: int, player_id: int):
        """Record an enemy settlement."""
        self._close_bottom_sheet()
        self._build_settlement(node_id, player_id)
    
    def _on_dice_roll(self, e):
        """Handle dice roll button - show number picker."""
        number_buttons = []
        
        for num in range(2, 13):
            if num == 7:
                # Special styling for 7 (robber)
                btn = ft.ElevatedButton(
                    "7 🏴‍☠️",
                    on_click=lambda e, n=7: self._process_dice_roll(n),
                    style=ft.ButtonStyle(bgcolor=COLORS["danger"]),
                    width=70
                )
            else:
                pips = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
                btn = ft.ElevatedButton(
                    str(num),
                    on_click=lambda e, n=num: self._process_dice_roll(n),
                    style=ft.ButtonStyle(
                        bgcolor=COLORS["primary"] if num in [6, 8] else COLORS["secondary"]
                    ),
                    width=60
                )
            number_buttons.append(btn)
        
        # Arrange in rows
        row1 = ft.Row(number_buttons[:4], alignment=ft.MainAxisAlignment.CENTER)
        row2 = ft.Row(number_buttons[4:8], alignment=ft.MainAxisAlignment.CENTER)
        row3 = ft.Row(number_buttons[8:], alignment=ft.MainAxisAlignment.CENTER)
        
        self.page.bottom_sheet = ft.BottomSheet(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("What number was rolled?", size=16, weight=ft.FontWeight.BOLD),
                    ft.Container(height=10),
                    row1, row2, row3,
                    ft.Container(height=10),
                    ft.TextButton("Cancel", on_click=lambda e: self._close_bottom_sheet())
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                padding=20,
                bgcolor=COLORS["surface"],
            ),
            open=True
        )
        self.page.update()
    
    def _process_dice_roll(self, number: int):
        """Process the dice roll."""
        self._close_bottom_sheet()
        
        if number == 7:
            # Robber! Show robber advice
            self._show_robber_advice()
        else:
            # Normal roll - update resource tracker
            gains = self.resource_tracker.on_dice_roll(number)
            
            # Show what was gained
            gain_text = []
            for player_id, resources in gains.items():
                if resources:
                    resource_counts = {}
                    for r in resources:
                        resource_counts[r.value] = resource_counts.get(r.value, 0) + 1
                    gain_str = ", ".join(f"{v} {k}" for k, v in resource_counts.items())
                    gain_text.append(f"P{player_id}: +{gain_str}")
            
            if gain_text:
                self.page.snack_bar = ft.SnackBar(
                    content=ft.Text(f"🎲 {number} | " + " | ".join(gain_text)),
                    bgcolor=COLORS["surface"]
                )
                self.page.snack_bar.open = True
        
        self._update_recommendations()
    
    def _show_robber_advice(self):
        """Show robber placement advice."""
        # Get recommendation
        hex_coords, target_player, hex_reason = self.resource_tracker.get_robber_hex_recommendation(
            self.current_player_id
        )
        steal_player, steal_resource, steal_reason = self.resource_tracker.get_rob_recommendation(
            self.current_player_id
        )
        
        self.page.dialog = ft.AlertDialog(
            title=ft.Text("🏴‍☠️ Robber!", color=COLORS["danger"]),
            content=ft.Column([
                ft.Text("Placement Advice:", weight=ft.FontWeight.BOLD),
                ft.Text(hex_reason, size=14),
                ft.Divider(),
                ft.Text("Steal From:", weight=ft.FontWeight.BOLD),
                ft.Text(steal_reason, size=14),
                ft.Container(
                    content=ft.Text(f"Take {steal_resource.value.upper()} from Player {steal_player}",
                                   color=COLORS["primary"], weight=ft.FontWeight.BOLD),
                    bgcolor=COLORS["secondary"],
                    padding=10,
                    border_radius=5
                ),
            ], tight=True),
            actions=[
                ft.TextButton("Got it!", on_click=lambda e: self._close_dialog())
            ]
        )
        self.page.dialog.open = True
        self.page.update()
    
    def _close_dialog(self):
        """Close dialog."""
        if self.page.dialog:
            self.page.dialog.open = False
            self.page.update()
    
    def _show_node_action_sheet(self, action_type: str):
        """Show a sheet to pick a node for the action."""
        # For now, use a simple text input
        node_input = ft.TextField(label="Node ID", keyboard_type=ft.KeyboardType.NUMBER, width=100)
        
        def confirm(e):
            try:
                node_id = int(node_input.value)
                if action_type == "settlement_self":
                    self._build_settlement(node_id, self.current_player_id)
                elif action_type == "city_self":
                    self._upgrade_to_city(node_id)
                self._close_dialog()
            except ValueError:
                pass
        
        self.page.dialog = ft.AlertDialog(
            title=ft.Text("Enter Node ID"),
            content=node_input,
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: self._close_dialog()),
                ft.TextButton("Confirm", on_click=confirm)
            ]
        )
        self.page.dialog.open = True
        self.page.update()
    
    def _show_road_dialog(self):
        """Show dialog to enter road nodes."""
        node_a = ft.TextField(label="From Node", keyboard_type=ft.KeyboardType.NUMBER, width=100)
        node_b = ft.TextField(label="To Node", keyboard_type=ft.KeyboardType.NUMBER, width=100)
        
        def confirm(e):
            try:
                a = int(node_a.value)
                b = int(node_b.value)
                success = self.board.place_road(a, b, self.current_player_id)
                if success:
                    self.resource_tracker.on_build(self.current_player_id, "road")
                    self._update_recommendations()
                self._close_dialog()
            except ValueError:
                pass
        
        self.page.dialog = ft.AlertDialog(
            title=ft.Text("Build Road"),
            content=ft.Row([node_a, ft.Text("→"), node_b]),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: self._close_dialog()),
                ft.TextButton("Build", on_click=confirm)
            ]
        )
        self.page.dialog.open = True
        self.page.update()
    
    def _show_enemy_build_dialog(self):
        """Show dialog for enemy building."""
        player_dd = ft.Dropdown(
            label="Player",
            options=[ft.dropdown.Option(str(i), f"Player {i}") 
                    for i in range(1, 5) if i != self.current_player_id],
            width=120
        )
        type_dd = ft.Dropdown(
            label="Type",
            options=[
                ft.dropdown.Option("settlement", "Settlement"),
                ft.dropdown.Option("city", "City"),
                ft.dropdown.Option("road", "Road")
            ],
            width=120
        )
        node_input = ft.TextField(label="Node(s)", hint_text="42 or 42,43", width=100)
        
        def confirm(e):
            try:
                player_id = int(player_dd.value)
                build_type = type_dd.value
                
                if build_type == "road":
                    nodes = [int(n.strip()) for n in node_input.value.split(",")]
                    if len(nodes) >= 2:
                        self.board.place_road(nodes[0], nodes[1], player_id)
                        self.resource_tracker.on_build(player_id, "road")
                elif build_type == "settlement":
                    node_id = int(node_input.value)
                    self.board.place_settlement(node_id, player_id)
                    self.resource_tracker.on_build(player_id, "settlement")
                elif build_type == "city":
                    node_id = int(node_input.value)
                    self.board.upgrade_to_city(node_id, player_id)
                    self.resource_tracker.on_build(player_id, "city")
                
                self._update_recommendations()
                self._close_dialog()
            except (ValueError, IndexError):
                pass
        
        self.page.dialog = ft.AlertDialog(
            title=ft.Text("Enemy Built"),
            content=ft.Column([
                ft.Row([player_dd, type_dd]),
                node_input
            ], tight=True),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: self._close_dialog()),
                ft.TextButton("Confirm", on_click=confirm)
            ]
        )
        self.page.dialog.open = True
        self.page.update()
    
    def _on_undo(self, e):
        """Handle undo button - revert last ResourceTracker action."""
        success = self.resource_tracker.undo()
        if success:
            self.page.snack_bar = ft.SnackBar(
                content=ft.Text("↩️ Last action undone"),
                bgcolor=COLORS["surface"]
            )
            self.page.snack_bar.open = True
            self._update_recommendations()
        else:
            self.page.snack_bar = ft.SnackBar(
                content=ft.Text("Nothing to undo"),
                bgcolor=COLORS["danger"]
            )
            self.page.snack_bar.open = True
            self.page.update()


def main(page: ft.Page):
    """Main entry point with Error Reporting."""
    page.window.width = 400
    page.window.height = 850
    
    try:
        # Attempt to launch the app
        CatanMindApp(page)
    except Exception as e:
        # If it crashes, show the error on screen!
        import traceback
        error_msg = traceback.format_exc()
        
        page.bgcolor = "black"
        page.scroll = ft.ScrollMode.AUTO
        page.clean()
        page.add(
            ft.Column([
                ft.Text("⚠️ APP CRASHED ⚠️", size=30, color="red", weight="bold"),
                ft.Text("Please screenshot this and send to developer:", color="white"),
                ft.Divider(color="white"),
                ft.Text(error_msg, color="red", font_family="monospace", selectable=True)
            ])
        )
        page.update()
        print(error_msg)  # Fallback for local run


if __name__ == "__main__":
    ft.app(target=main)
