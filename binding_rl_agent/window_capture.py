from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mss
import numpy as np
import win32gui


WINDOW_TITLE_HINTS: tuple[str, ...] = (
    "binding of isaac",
    "isaac",
)


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    def as_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


def list_visible_windows() -> list[WindowInfo]:
    windows: list[WindowInfo] = []

    def callback(hwnd: int, _: int) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd).strip()
        if not title:
            return
        windows.append(WindowInfo(hwnd=hwnd, title=title))

    win32gui.EnumWindows(callback, 0)
    return windows


def find_window_by_title(
    title_substring: str | None = None,
    title_hints: Iterable[str] = WINDOW_TITLE_HINTS,
) -> WindowInfo:
    windows = list_visible_windows()
    if title_substring:
        title_substring = title_substring.lower()
        for window in windows:
            if title_substring in window.title.lower():
                return window

    lowered_hints = tuple(hint.lower() for hint in title_hints)
    for window in windows:
        title_lower = window.title.lower()
        if any(hint in title_lower for hint in lowered_hints):
            return window

    visible_titles = "\n".join(f"- {window.title}" for window in windows[:25])
    raise RuntimeError(
        "Could not find the Isaac window. Visible windows:\n"
        f"{visible_titles or '- <none>'}"
    )


def get_client_region(window: WindowInfo) -> CaptureRegion:
    left, top, right, bottom = win32gui.GetClientRect(window.hwnd)
    screen_left, screen_top = win32gui.ClientToScreen(window.hwnd, (left, top))
    screen_right, screen_bottom = win32gui.ClientToScreen(window.hwnd, (right, bottom))
    width = screen_right - screen_left
    height = screen_bottom - screen_top
    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Window '{window.title}' has an invalid client area: {width}x{height}"
        )
    return CaptureRegion(
        left=screen_left,
        top=screen_top,
        width=width,
        height=height,
    )


class IsaacWindowCapture:
    def __init__(self, title_substring: str | None = None):
        self.window = find_window_by_title(title_substring=title_substring)
        self.region = get_client_region(self.window)
        self._mss = mss.mss()

    def refresh_region(self) -> CaptureRegion:
        self.region = get_client_region(self.window)
        return self.region

    def grab(self) -> np.ndarray:
        screenshot = self._mss.grab(self.region.as_dict())
        frame = np.array(screenshot, dtype=np.uint8)
        return frame[:, :, :3]

    def is_foreground(self) -> bool:
        return win32gui.GetForegroundWindow() == self.window.hwnd

    def focus_window(self) -> None:
        try:
            win32gui.SetForegroundWindow(self.window.hwnd)
        except Exception:
            # Windows may refuse foreground changes in some contexts; callers can
            # fall back to showing focus state in the overlay.
            pass
