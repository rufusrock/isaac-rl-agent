from __future__ import annotations

import socket
from dataclasses import dataclass


# Isaac AB+ room type constants (Data.Type values from Lua)
class RoomType:
    DEFAULT     = 1
    SHOP        = 2
    ERROR       = 3
    TREASURE    = 4
    BOSS        = 5
    MINIBOSS    = 6
    SECRET      = 7
    SUPERSECRET = 8
    ARCADE      = 9
    CURSE       = 10
    CHALLENGE   = 11
    LIBRARY     = 12
    SACRIFICE   = 13
    DEVIL       = 14
    ANGEL       = 15
    DUNGEON     = 16
    BOSSRUSH    = 17
    ISAACS      = 18
    BARREN      = 19
    CHEST       = 20
    DICE        = 21


@dataclass(frozen=True)
class RoomInfo:
    grid_index: int
    room_type: int
    visited: int   # VisitedCount from Lua (0 = never visited)
    cleared: bool


@dataclass(frozen=True)
class IsaacGameState:
    frame: int
    rooms_cleared: int
    floors_cleared: int
    kills: int
    dmg_taken: float
    coins: int
    keys: int
    soul_hearts: int
    black_hearts: int
    collectibles: int
    room_index: int
    floor: int
    rooms_explored: int
    deaths: int
    floor_rooms: tuple[RoomInfo, ...]  # empty if telemetry pre-dates room graph


class IsaacUDPGameStateReceiver:
    def __init__(self, port: int = 8123):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.last_payload: str | None = None

    def get_latest(self) -> IsaacGameState | None:
        try:
            data, _ = self.sock.recvfrom(4096)
            self.last_payload = data.decode()
        except BlockingIOError:
            pass
        return self.parse(self.last_payload)

    @staticmethod
    def parse(payload: str | None) -> IsaacGameState | None:
        if not payload:
            return None
        try:
            # Split scalar section from optional room-graph section
            if "|" in payload:
                scalar_str, rooms_str = payload.split("|", 1)
            else:
                scalar_str, rooms_str = payload, ""

            parts = scalar_str.strip().split(",")
            floor_rooms = _parse_rooms(rooms_str)

            return IsaacGameState(
                frame=int(parts[0]),
                rooms_cleared=int(parts[1]),
                floors_cleared=int(parts[2]),
                kills=int(parts[3]),
                dmg_taken=float(parts[4]),
                coins=int(parts[5]),
                keys=int(parts[6]),
                soul_hearts=int(parts[7]),
                black_hearts=int(parts[10]),
                collectibles=int(parts[11]),
                room_index=int(parts[12]),
                floor=int(parts[13]),
                rooms_explored=int(parts[14]),
                deaths=int(parts[15]),
                floor_rooms=floor_rooms,
            )
        except Exception:
            return None


def _parse_rooms(rooms_str: str) -> tuple[RoomInfo, ...]:
    if not rooms_str:
        return ()
    rooms = []
    for token in rooms_str.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 4:
            continue
        rooms.append(RoomInfo(
            grid_index=int(parts[0]),
            room_type=int(parts[1]),
            visited=int(parts[2]),
            cleared=parts[3] == "1",
        ))
    return tuple(rooms)
