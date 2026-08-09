"""
CatanMind — entry point.

The application itself lives in :mod:`catanmind.ui`; this file only starts it,
so that the package stays importable (and testable) without a display.
"""

import flet as ft

from catanmind.ui import main

if __name__ == "__main__":
    ft.run(main)
