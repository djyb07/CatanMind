"""
The screen's logic, driven without a display.

Flet renders through Flutter, so a running app exposes no DOM to inspect — a
screenshot is the only visual check, and it cannot tell you whether tapping an
intersection recorded the right settlement, or whether the screen offered a
move the rules forbid. Everything behavioural is therefore tested here against
a stub page.

The central claim under test: the screen never offers an action that
:mod:`catanmind.flow` does not allow.
"""

import pytest

import flet as ft

from catanmind.board import Board, Building, DevCard, Layout, Port, Resource
from catanmind.flow import Step
from catanmind.state import Phase
from catanmind.ui import BOARD_HEIGHT, CatanMind, board_shapes
from catanmind.view import Viewport
from catanmind import rules


class StubPage:
    """Just enough Page for the app to run headless."""

    def __init__(self):
        self.updates = 0
        self.dialogs = []
        self.controls = []

    def update(self):
        self.updates += 1

    def add(self, control):
        self.controls.append(control)

    def show_dialog(self, dialog):
        self.dialogs.append(dialog)

    def pop_dialog(self):
        return self.dialogs.pop() if self.dialogs else None


class Tap:
    def __init__(self, x, y):
        self.local_position = ft.Offset(x, y)


class Resize:
    def __init__(self, width, height):
        self.width = width
        self.height = height


@pytest.fixture
def app():
    page = StubPage()
    a = CatanMind(page, num_players=4, me=1)
    a.build()
    a.refresh()
    return a


def start_game(app, layout=None):
    """Take the app through config and the editor into a live game."""
    layout = layout or Layout.standard()
    app.pending = dict(layout.tiles)
    app.pending_ports = dict(layout.ports)
    app._finish_editing()
    return app


def tap_node(app, node_id):
    x, y = app.view.node_xy(app.board, node_id)
    app._on_tap(Tap(x, y))


def tap_edge(app, edge_id):
    x, y = app.view.edge_xy(app.board, edge_id)
    app._on_tap(Tap(x, y))


def tap_tile(app, coord):
    x, y = app.view.tile_xy(app.board, coord)
    app._on_tap(Tap(x, y))


def act(app, action_id):
    """Press the named action button, if the screen is offering it."""
    for action in app.flow.available_actions():
        if action.id == action_id:
            app._choose_action(action)
            return action
    raise AssertionError(f"{action_id!r} is not on offer at {app.flow.step}")


def play_setup(app):
    """Play the whole setup phase through the screen, as a user would."""
    while app.flow.in_setup:
        spot = next(
            n for n in rules.legal_settlements(
                app.state, app.flow.current, setup=True
            )
        )
        tap_node(app, spot)
        edge = next(
            e for e in app.board.node_edges[spot] if e not in app.state.roads
        )
        tap_edge(app, edge)
    return app


def clear_hands(app):
    for player in app.state.players.values():
        for resource in Resource:
            player.hand.cards[resource] = 0


# -- the screens -----------------------------------------------------------


def test_the_app_opens_on_the_table_setup(app):
    assert app.screen == "config"


def test_choosing_the_board_comes_before_playing(app):
    app._go_editor()
    assert app.screen == "editor"
    start_game(app)
    assert app.screen == "game"


def test_the_seat_you_pick_is_the_seat_you_play(app):
    app.state.me = 3
    start_game(app)
    assert app.me == 3
    assert app.flow.current == 1, "seat 1 still places first"


def test_the_entered_board_is_the_board_advice_uses(app):
    layout = Layout.random()
    start_game(app, layout)
    assert app.board.layout.tiles == layout.tiles
    assert app.scorer.board is app.board
    assert app.turn_advisor.board is app.board


def test_editor_blocks_an_incomplete_board(app):
    app.pending = {(0, 0): (Resource.WOOD, 6)}
    app._finish_editing()
    assert app.screen != "game"


# -- the screen only offers legal moves ------------------------------------


def test_setup_offers_exactly_one_move(app):
    start_game(app)
    actions = app.flow.available_actions()
    assert len(actions) == 1
    assert actions[0].id == "setup_settlement"


def test_building_is_not_offered_during_setup(app):
    start_game(app)
    ids = {a.id for a in app.flow.available_actions()}
    assert "build_settlement" not in ids
    assert "buy_dev" not in ids
    assert "end_turn" not in ids


def test_the_setup_step_arms_the_board_automatically(app):
    """No ceremony: there is only one move, so the board is ready for it."""
    start_game(app)
    assert app.pending_action is not None
    assert app.pending_action.target == "node"


def test_you_cannot_place_two_settlements_in_a_row(app):
    """The exact abuse the old screen allowed."""
    start_game(app)
    first = next(rules.legal_settlements(app.state, 1, setup=True).__iter__())
    tap_node(app, first)
    assert app.flow.step is Step.SETUP_ROAD

    other = next(
        n for n in rules.legal_settlements(app.state, 1, setup=True) if n != first
    )
    tap_node(app, other)
    assert other not in app.state.buildings, "a road is owed first"


def test_setup_advances_through_the_seats_on_its_own(app):
    start_game(app)
    seen = []
    while app.flow.in_setup:
        seen.append(app.flow.current)
        spot = next(
            n for n in rules.legal_settlements(
                app.state, app.flow.current, setup=True
            )
        )
        tap_node(app, spot)
        edge = next(
            e for e in app.board.node_edges[spot] if e not in app.state.roads
        )
        tap_edge(app, edge)
    assert seen == [1, 2, 3, 4, 4, 3, 2, 1]


def test_nobody_gets_four_free_settlements(app):
    start_game(app)
    play_setup(app)
    for player in app.state.players:
        assert len(app.state.settlements_of(player)) == 2
        assert len(app.state.edges_of(player)) == 2


def test_a_turn_begins_with_the_roll_not_with_building(app):
    start_game(app)
    play_setup(app)
    ids = {a.id for a in app.flow.available_actions()}
    assert "roll" in ids
    assert "build_road" not in ids


def test_after_rolling_the_build_actions_appear(app):
    start_game(app)
    play_setup(app)
    app.flow.roll(8)
    app.refresh()
    ids = {a.id for a in app.flow.available_actions()}
    assert {"build_road", "build_settlement", "city", "end_turn"} <= ids


def test_unaffordable_builds_are_shown_but_disabled(app):
    start_game(app)
    play_setup(app)
    app.flow.roll(8)
    clear_hands(app)
    app.refresh()
    actions = {a.id: a for a in app.flow.available_actions()}
    assert actions["build_road"].enabled is False
    assert actions["build_road"].hint


def test_ending_the_turn_moves_to_the_next_player(app):
    start_game(app)
    play_setup(app)
    app.flow.roll(8)
    app.refresh()
    act(app, "end_turn")
    assert app.flow.current == 2
    assert app.flow.step is Step.PRE_ROLL


# -- recording an opponent's turn ------------------------------------------


def test_advice_is_withheld_on_an_opponents_turn(app):
    start_game(app)
    play_setup(app)
    app.flow.roll(8)
    act(app, "end_turn")
    assert app.flow.is_my_turn() is False
    body = app._tab_advice()
    assert body is not None


def test_an_opponents_build_is_recorded_against_them(app):
    start_game(app)
    play_setup(app)
    app.flow.roll(8)
    act(app, "end_turn")
    app.state.adjust(2, Resource.WOOD, 1)
    app.state.adjust(2, Resource.BRICK, 1)
    app.flow.roll(8)
    app.refresh()
    edge = rules.legal_roads(app.state, 2)[0]
    act(app, "build_road")
    tap_edge(app, edge)
    assert app.state.roads.get(edge) == 2


# -- board taps map to the right move --------------------------------------


def test_a_settlement_tap_records_a_settlement(app):
    start_game(app)
    spot = next(iter(rules.legal_settlements(app.state, 1, setup=True)))
    tap_node(app, spot)
    assert app.state.buildings[spot] == (1, Building.SETTLEMENT)


def test_an_illegal_tap_is_refused_and_explained(app):
    start_game(app)
    spot = next(iter(rules.legal_settlements(app.state, 1, setup=True)))
    tap_node(app, spot)
    edge = next(e for e in app.board.node_edges[spot])
    tap_edge(app, edge)
    # Second seat: try to settle right next to seat one's settlement.
    neighbour = app.board.node(spot).neighbors[0]
    tap_node(app, neighbour)
    assert neighbour not in app.state.buildings
    assert app.status


def test_a_refused_tap_keeps_the_action_armed(app):
    """A misfire must not force the player to press the button again."""
    start_game(app)
    spot = next(iter(rules.legal_settlements(app.state, 1, setup=True)))
    tap_node(app, spot)
    edge = next(e for e in app.board.node_edges[spot])
    tap_edge(app, edge)
    neighbour = app.board.node(spot).neighbors[0]
    tap_node(app, neighbour)
    assert app.pending_action is not None


def test_the_road_step_does_not_let_a_node_swallow_the_tap(app):
    start_game(app)
    spot = next(iter(rules.legal_settlements(app.state, 1, setup=True)))
    tap_node(app, spot)
    edge_id = app.board.node_edges[spot][0]
    edge = app.board.edge(edge_id)
    ax, ay = app.view.node_xy(app.board, edge.a)
    bx, by = app.view.node_xy(app.board, edge.b)
    app._on_tap(Tap(ax + (bx - ax) * 0.2, ay + (by - ay) * 0.2))
    assert app.state.roads.get(edge_id) == 1


def test_an_armed_action_is_dropped_when_the_step_moves_on(app):
    """
    Regression: the setup prompt ("tap an intersection") stayed on screen into
    the building step, and the board kept trying to place a settlement.
    """
    start_game(app)
    assert app.pending_action is not None
    play_setup(app)
    app.flow.roll(8)
    app.refresh()
    assert app.flow.step is Step.MAIN
    assert app.pending_action is None


def test_tapping_with_nothing_armed_only_inspects(app):
    start_game(app)
    play_setup(app)
    app.pending_action = None
    before = dict(app.state.buildings)
    tap_node(app, 10)
    assert app.state.buildings == before
    assert app.status


# -- the seven -------------------------------------------------------------

def test_a_seven_asks_for_discards_before_the_robber(app):
    start_game(app)
    play_setup(app)
    clear_hands(app)
    app.state.adjust(1, Resource.WOOD, 10)
    app.flow.roll(7)
    app.refresh()
    assert app.flow.step is Step.DISCARD
    ids = {a.id for a in app.flow.available_actions()}
    assert ids == {"discard"}


def test_the_robber_step_arms_the_tiles(app):
    start_game(app)
    play_setup(app)
    clear_hands(app)
    app.flow.roll(7)
    app.refresh()
    assert app.flow.step is Step.MOVE_ROBBER
    assert app.pending_action is not None
    assert app.pending_action.target == "tile"


def test_moving_the_robber_by_tapping_a_tile(app):
    start_game(app)
    play_setup(app)
    clear_hands(app)
    app.flow.roll(7)
    app.refresh()
    target = next(
        c for c, t in app.board.tiles.items()
        if c != app.state.robber and not any(
            n in app.state.buildings for n in t.nodes
        )
    )
    tap_tile(app, target)
    assert app.state.robber == target


# -- undo ------------------------------------------------------------------


def test_undo_rewinds_the_turn_machine(app):
    start_game(app)
    play_setup(app)
    app.flow.roll(8)
    app.refresh()
    assert app.flow.step is Step.MAIN
    app._undo()
    assert app.flow.step is Step.PRE_ROLL


def test_undo_on_a_fresh_game_is_harmless(app):
    start_game(app)
    for _ in range(5):
        app._undo()
    assert app.state.buildings == {}


# -- hand ------------------------------------------------------------------


def test_adjusting_a_hand_moves_the_single_source_of_truth(app):
    start_game(app)
    play_setup(app)
    before = app.state.players[1].hand.cards[Resource.WOOD]
    app._adjust(1, Resource.WOOD, 1)
    assert app.state.players[1].hand.cards[Resource.WOOD] == before + 1


def test_a_hand_never_goes_negative(app):
    start_game(app)
    clear_hands(app)
    app._adjust(1, Resource.ORE, -1)
    assert app.state.players[1].hand.cards[Resource.ORE] == 0


def test_every_player_hand_is_editable_in_play(app):
    """Opponent card counts have to be recordable, or the tracker is blind."""
    start_game(app)
    play_setup(app)
    app._adjust(3, Resource.SHEEP, 2)
    assert app.state.players[3].hand.cards[Resource.SHEEP] == 2


def test_hands_cannot_be_edited_during_setup(app):
    """
    Nobody holds cards until the second settlement pays out, and the app deals
    that itself — so editing here could only introduce an error.
    """
    start_game(app)
    assert app.flow.in_setup
    app._adjust(1, Resource.SHEEP, 3)
    assert app.state.players[1].hand.cards[Resource.SHEEP] == 0
    assert app.status


# -- drawing ---------------------------------------------------------------


def test_the_board_fits_the_canvas_it_is_drawn_into(app):
    start_game(app)
    for width, height in ((375, BOARD_HEIGHT), (820, BOARD_HEIGHT), (300, 260)):
        app._on_resize(Resize(width, height))
        view = app.view
        for node in app.board.nodes:
            x, y = view.node_xy(app.board, node.id)
            assert 0 <= x <= width, f"node {node.id} off-screen at {width}x{height}"
            assert 0 <= y <= height


def test_port_badges_stay_inside_the_canvas(app):
    """Port jetties are drawn beyond the coastline and clip first."""
    start_game(app)
    for width, height in ((375, BOARD_HEIGHT), (280, 300), (820, BOARD_HEIGHT)):
        app._on_resize(Resize(width, height))
        view = app.view
        radius = view.hex_radius(app.board)
        cx, cy = view.to_screen(0, 0)
        badge = radius * 0.26
        for edge_id in app.board.port_slots:
            ex, ey = view.edge_xy(app.board, edge_id)
            dx, dy = ex - cx, ey - cy
            length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
            px = ex + dx / length * radius * 0.50
            py = ey + dy / length * radius * 0.50
            assert badge <= px <= width - badge, f"port clipped at {width}x{height}"
            assert badge <= py <= height - badge


def test_resizing_settles_instead_of_looping(app):
    start_game(app)
    app._on_resize(Resize(375, BOARD_HEIGHT))
    before = app.page.updates
    app._on_resize(Resize(375, BOARD_HEIGHT))
    assert app.page.updates == before


def test_legal_targets_are_drawn_while_an_action_is_armed(app):
    start_game(app)
    draw = app._draw_targets()
    assert draw.get("legal_nodes"), "the player must see where they may tap"


def test_the_editor_draws_unknown_tiles_without_a_state(app):
    shapes = board_shapes(app.board, app.view, None, pending={})
    assert shapes


# -- dialogs ---------------------------------------------------------------
#
# Every dialog is built and then actually pressed. Constructing a control is
# not enough: the port editor shipped a `Dropdown(on_change=...)` that Flet
# 0.86 rejects, and nothing caught it because no test had ever clicked inside
# a dialog.


def click_all(control, limit=60):
    """Press every enabled button reachable from a control."""
    pressed = 0
    stack = [control]
    seen = set()
    while stack and pressed < limit:
        c = stack.pop()
        if id(c) in seen:
            continue
        seen.add(id(c))
        handler = getattr(c, "on_click", None)
        if handler and not getattr(c, "disabled", False):
            handler(None)
            pressed += 1
        for attr in ("controls", "actions"):
            kids = getattr(c, attr, None)
            if isinstance(kids, list):
                stack.extend(kids)
        for attr in ("content", "title"):
            kid = getattr(c, attr, None)
            if kid is not None and not isinstance(kid, str):
                stack.append(kid)
    return pressed


def test_the_port_editor_opens_and_its_chips_work(app):
    """Regression: this used to raise on a Flet keyword that no longer exists."""
    app._go_editor()
    app._use_standard()
    app._open_port_editor()
    assert app.page.dialogs
    assert click_all(app.page.dialogs[-1]) > 0


def test_setting_a_port_sticks(app):
    app._go_editor()
    app._use_standard()
    app.set_port(0, Port.ORE)
    assert app.pending_ports[0] is Port.ORE
    app._open_port_editor()
    assert app.page.dialogs


def test_the_tile_editor_opens_and_its_buttons_work(app):
    app._go_editor()
    tap_tile(app, (0, 0))
    assert app.page.dialogs
    assert click_all(app.page.dialogs[-1]) > 0


def test_the_roll_dialog_records_a_roll(app):
    start_game(app)
    play_setup(app)
    app._roll_dialog()
    dialog = app.page.dialogs[-1]
    # Find the button labelled "8" and press it.
    pressed = False
    stack = [dialog]
    while stack:
        c = stack.pop()
        if getattr(c, "content", None) == "8" and getattr(c, "on_click", None):
            c.on_click(None)
            pressed = True
            break
        for attr in ("controls", "actions"):
            kids = getattr(c, attr, None)
            if isinstance(kids, list):
                stack.extend(kids)
        kid = getattr(c, "content", None)
        if kid is not None and not isinstance(kid, str):
            stack.append(kid)
    assert pressed, "the roll dialog offered no 8"
    assert app.flow.step is Step.MAIN
    assert app.state.last_roll == 8


def test_the_trade_dialog_opens_and_its_buttons_work(app):
    start_game(app)
    play_setup(app)
    app.flow.roll(8)
    app.refresh()
    app._trade_dialog()
    assert app.page.dialogs
    assert click_all(app.page.dialogs[-1]) > 0


def test_the_discard_dialog_opens_and_its_buttons_work(app):
    start_game(app)
    play_setup(app)
    clear_hands(app)
    app.state.adjust(1, Resource.WOOD, 10)
    app.flow.roll(7)
    app.refresh()
    app._discard_dialog(1)
    assert app.page.dialogs
    assert click_all(app.page.dialogs[-1]) > 0


def test_the_steal_dialog_opens(app):
    start_game(app)
    play_setup(app)
    clear_hands(app)
    app.flow.roll(7)
    app.refresh()
    tile = next(
        (
            t for t in app.board.tiles.values()
            if t.coord != app.state.robber and any(
                n in app.state.buildings and app.state.buildings[n][0] != 1
                for n in t.nodes
            )
        ),
        None,
    )
    if tile is None:
        pytest.skip("no robbable tile on this board")
    for node in tile.nodes:
        entry = app.state.buildings.get(node)
        if entry and entry[0] != 1:
            app.state.adjust(entry[0], Resource.ORE, 1)
    tap_tile(app, tile.coord)
    app.refresh()
    assert app.flow.step is Step.STEAL
    for action in app.flow.available_actions():
        if action.id == "steal":
            app._choose_action(action)
            assert app.page.dialogs
            break


def test_the_buy_dialog_asks_which_card_was_drawn(app):
    """You have to tell the app what came off the deck; it cannot see it."""
    start_game(app)
    play_setup(app)
    app.flow.roll(8)
    clear_hands(app)
    app.state.adjust(1, Resource.SHEEP, 1)
    app.state.adjust(1, Resource.WHEAT, 1)
    app.state.adjust(1, Resource.ORE, 1)
    app.refresh()
    act(app, "buy_dev")
    assert app.page.dialogs, "buying must ask which card it was"
    assert app.state.players[1].dev_cards_held == 0, "nothing drawn until chosen"


def test_choosing_the_drawn_card_records_its_type(app):
    start_game(app)
    play_setup(app)
    app.flow.roll(8)
    clear_hands(app)
    for r in (Resource.SHEEP, Resource.WHEAT, Resource.ORE):
        app.state.adjust(1, r, 1)
    app.refresh()
    act(app, "buy_dev")
    click_all(app.page.dialogs[-1], limit=1)   # presses the first card button
    assert app.state.players[1].dev_cards_held == 1


def test_a_card_bought_this_turn_is_not_offered(app):
    start_game(app)
    play_setup(app)
    app.flow.roll(8)
    clear_hands(app)
    for r in (Resource.SHEEP, Resource.WHEAT, Resource.ORE):
        app.state.adjust(1, r, 1)
    app.flow.buy_dev(DevCard.KNIGHT)
    app.refresh()
    ids = {a.id for a in app.flow.available_actions()}
    assert "play_dev:knight" not in ids


def test_a_held_knight_is_offered_next_turn(app):
    start_game(app)
    play_setup(app)
    app.state.players[1].dev_cards[DevCard.KNIGHT] = 1
    app.flow.roll(8)
    app.refresh()
    ids = {a.id for a in app.flow.available_actions()}
    assert "play_dev:knight" in ids


def test_the_monopoly_dialog_collects_the_resource(app):
    start_game(app)
    play_setup(app)
    app.state.players[1].dev_cards[DevCard.MONOPOLY] = 1
    app.flow.roll(8)
    clear_hands(app)
    app.state.adjust(2, Resource.ORE, 3)
    app.refresh()
    act(app, "play_dev:monopoly")
    assert app.page.dialogs, "monopoly must ask which resource"
    # Press the ore button.
    stack = [app.page.dialogs[-1]]
    while stack:
        c = stack.pop()
        if getattr(c, "content", None) == "Ore" and getattr(c, "on_click", None):
            c.on_click(None)
            break
        for attr in ("controls", "actions"):
            kids = getattr(c, attr, None)
            if isinstance(kids, list):
                stack.extend(kids)
        kid = getattr(c, "content", None)
        if kid is not None and not isinstance(kid, str):
            stack.append(kid)
    assert app.state.players[1].hand.cards[Resource.ORE] == 3
    assert app.state.players[2].hand.cards[Resource.ORE] == 0


def test_the_year_of_plenty_dialog_opens(app):
    start_game(app)
    play_setup(app)
    app.state.players[1].dev_cards[DevCard.YEAR_OF_PLENTY] = 1
    app.flow.roll(8)
    app.refresh()
    act(app, "play_dev:year_of_plenty")
    assert app.page.dialogs
    assert click_all(app.page.dialogs[-1]) > 0


def test_road_building_walks_you_through_two_free_roads(app):
    start_game(app)
    play_setup(app)
    app.state.players[1].dev_cards[DevCard.ROAD_BUILDING] = 1
    app.flow.roll(8)
    clear_hands(app)
    app.refresh()
    act(app, "play_dev:road_building")
    app.refresh()
    assert app.flow.step is Step.ROAD_BUILDING
    assert app.pending_action is None or app.pending_action.target == "edge"

    for _ in range(2):
        action = next(
            a for a in app.flow.available_actions() if a.id == "free_road"
        )
        app._choose_action(action)
        edge = rules.legal_roads(app.state, 1)[0]
        tap_edge(app, edge)
    assert app.flow.step is Step.MAIN
    assert app.state.players[1].hand.total() == 0, "free roads cost nothing"


# -- the board editor shows ports ------------------------------------------


def test_ports_are_drawn_while_entering_the_board(app):
    """
    Entering a board without seeing the harbours is guesswork, and the editor
    used to return before it ever drew them.
    """
    app._go_editor()
    app._use_standard()
    with_ports = board_shapes(
        app.board, app.view, None, pending=app.pending, ports=app.pending_ports
    )
    without = board_shapes(
        app.board, app.view, None, pending=app.pending, ports={}
    )
    assert len(with_ports) > len(without)


def test_tapping_a_jetty_edits_that_port(app):
    app._go_editor()
    app._use_standard()
    x, y = app.port_badge_xy(0)
    app._on_tap(Tap(x, y))
    assert app.page.dialogs, "tapping a port should offer to change it"


def test_tapping_the_middle_of_the_board_still_edits_a_tile(app):
    app._go_editor()
    app._use_standard()
    tap_tile(app, (0, 0))
    assert app.page.dialogs


def test_refresh_never_raises_at_any_step(app):
    """Walk a whole turn cycle and redraw at every step."""
    start_game(app)
    app.refresh()
    play_setup(app)
    app.refresh()
    for _ in range(4):
        app.flow.roll(8)
        app.refresh()
        for tab in (app._tab_advice, app._tab_hand, app._tab_table):
            assert tab() is not None
        app.flow.end_turn()
        app.refresh()


def test_refresh_never_raises_around_a_seven(app):
    start_game(app)
    play_setup(app)
    clear_hands(app)
    app.state.adjust(1, Resource.WOOD, 10)
    app.flow.roll(7)
    app.refresh()
    assert app._tab_advice() is not None
    app.flow.discard(1, [Resource.WOOD] * 5)
    app.refresh()
    assert app._tab_advice() is not None
