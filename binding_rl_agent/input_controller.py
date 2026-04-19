from __future__ import annotations

import ctypes
import time

from pynput.keyboard import Controller, Key


_keyboard = Controller()


KEY_OBJECTS: dict[str, str | Key] = {
    "r": "r",
    "w": "w",
    "a": "a",
    "s": "s",
    "d": "d",
    "e": "e",
    "up": Key.up,
    "down": Key.down,
    "left": Key.left,
    "right": Key.right,
    "space": Key.space,
}

FUNCTION_KEY_CODES: dict[str, int] = {
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
}


def tap_key(key: str, hold_seconds: float = 0.05) -> None:
    resolved = _resolve_key(key)
    _keyboard.press(resolved)
    time.sleep(hold_seconds)
    _keyboard.release(resolved)


def key_down(key: str) -> None:
    _keyboard.press(_resolve_key(key))


def key_up(key: str) -> None:
    _keyboard.release(_resolve_key(key))


def hold_keys(keys: list[str], hold_seconds: float = 0.08) -> None:
    normalized = [key.lower() for key in keys]
    for key in normalized:
        key_down(key)
    time.sleep(hold_seconds)
    for key in reversed(normalized):
        key_up(key)


def release_keys(keys: list[str]) -> None:
    for key in reversed([key.lower() for key in keys]):
        key_up(key)


def sync_pressed_keys(target_keys: list[str], active_keys: set[str]) -> set[str]:
    normalized_target = {key.lower() for key in target_keys}
    normalized_active = {key.lower() for key in active_keys}

    for key in sorted(normalized_target - normalized_active):
        key_down(key)
    for key in sorted(normalized_active - normalized_target):
        key_up(key)

    return normalized_target


def release_all_agent_keys() -> None:
    release_keys(list(KEY_OBJECTS))


def is_virtual_key_pressed(virtual_key: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(virtual_key) & 0x8000)


def is_function_key_pressed(key: str) -> bool:
    return is_virtual_key_pressed(FUNCTION_KEY_CODES[key.lower()])


def _resolve_key(key: str) -> str | Key:
    normalized = key.lower()
    try:
        return KEY_OBJECTS[normalized]
    except KeyError as error:
        raise ValueError(f"Unknown key: {key}") from error
