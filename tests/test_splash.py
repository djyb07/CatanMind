"""
The opening sequence.

A splash is decoration, so the things worth testing are the ones that would
make it worse than none at all: holding the player hostage, being unskippable,
or throwing an exception on the way into the app.
"""

import asyncio

import pytest

from catanmind import splash
from catanmind.board import Resource


class StubPage:
    def __init__(self, fail=False):
        self.updates = 0
        self.fail = fail

    def update(self):
        if self.fail:
            raise RuntimeError("window is gone")
        self.updates += 1


@pytest.fixture
def palette():
    return {
        Resource.WOOD: "#2f6b3a", Resource.BRICK: "#b8532c",
        Resource.SHEEP: "#8dc63f", Resource.WHEAT: "#e8b23a",
        Resource.ORE: "#6b8095", None: "#d9c39a",
    }


@pytest.fixture(autouse=True)
def with_palette(palette):
    splash.set_palette(palette)


def make(page=None, **kwargs):
    calls = []
    opening = splash.Splash(
        page or StubPage(), lambda: calls.append(1), **kwargs
    )
    return opening, calls


# -- handover --------------------------------------------------------------


def test_the_app_starts_when_the_sequence_ends():
    opening, calls = make()
    asyncio.run(opening.play())
    assert calls == [1]
    assert opening.finished


def test_tapping_skips_straight_to_the_app():
    opening, calls = make()
    opening.finish()
    assert calls == [1]


def test_the_app_is_only_ever_started_once():
    """A tap during the last frame must not start the app twice."""
    opening, calls = make()
    opening.finish()
    opening.finish()
    asyncio.run(opening.play())
    assert calls == [1]


def test_skipping_ends_the_sequence_early():
    opening, calls = make()

    async def skip_then_play():
        task = asyncio.ensure_future(opening.play())
        await asyncio.sleep(0)
        opening.finish()
        await task

    asyncio.run(skip_then_play())
    assert calls == [1]


def test_a_closed_window_does_not_take_the_app_down_with_it():
    """The page can vanish mid-sequence; a splash is never worth a crash."""
    opening, calls = make(page=StubPage(fail=True))
    asyncio.run(opening.play())
    assert calls == [1]


def test_the_whole_thing_is_over_quickly():
    """Decoration must never be the reason someone is waiting."""
    assert splash.TOTAL_SECONDS <= 3.0


# -- what it draws ---------------------------------------------------------


def test_the_island_arrives_one_ring_at_a_time():
    opening, _ = make()
    counts = []
    for step in range(4):
        opening.revealed = step
        opening._draw()
        counts.append(len(opening.canvas.shapes))
    assert counts[0] == 0, "nothing is drawn before the sequence starts"
    assert counts == sorted(counts), "each ring adds to the board"
    assert counts[-1] > counts[1]


def test_the_rings_cover_every_tile():
    opening, _ = make()
    covered = sum(len(ring) for ring in opening.rings)
    assert covered == len(opening.board.tiles) == 19


def test_rings_are_grouped_by_distance_from_the_centre():
    opening, _ = make()
    assert len(opening.rings[0]) == 1      # the centre tile
    assert len(opening.rings[1]) == 6
    assert len(opening.rings[2]) == 12


def test_it_draws_at_any_screen_size():
    for width, height in ((360, 640), (390, 844), (768, 1024), (1280, 800)):
        opening, _ = make(width=width, height=height)
        opening.revealed = 3
        opening._draw()
        assert opening.canvas.shapes


def island_points(opening):
    return [
        (e.x, e.y)
        for s in opening.canvas.shapes if hasattr(s, "elements")
        for e in s.elements if hasattr(e, "x")
    ]


@pytest.mark.parametrize(
    "width,height", [(360, 640), (390, 844), (430, 932), (768, 1024)]
)
def test_the_island_stays_inside_its_own_band(width, height):
    """
    The board and the name each get a band of the screen. Overlapping them
    put the wordmark straight across the middle of the island.
    """
    opening, _ = make(width=width, height=height)
    opening.revealed = 3
    opening._draw()
    points = island_points(opening)
    assert points
    assert all(-2 <= x <= width + 2 for x, _ in points), "island off the side"
    assert all(
        -2 <= y <= opening.island_height + 2 for _, y in points
    ), "island spills into the space the wordmark occupies"


def test_the_wordmark_has_room_below_the_island():
    opening, _ = make(width=390, height=844)
    assert opening.island_height < 844, "no room left for the name"
    assert 844 - opening.island_height > 100


def test_the_wordmark_starts_hidden_and_is_animated():
    opening, _ = make()
    assert opening.wordmark.opacity == 0
    assert opening.wordmark.animate_opacity is not None


def test_the_wordmark_is_showing_by_the_end():
    opening, _ = make()
    asyncio.run(opening.play())
    assert opening.wordmark.opacity == 1
