"""
Room graph and BFS-based navigation hint.

Isaac AB+ lays floors on a 13-column grid (indices 0-168).
Two rooms are adjacent when their grid indices differ by 1 (E/W)
or by GRID_WIDTH=13 (N/S), with a boundary check so rows don't wrap.

The navigation hint is a 5-way signal consumed by the policy as
an auxiliary input (one-hot vector of length 5):
  0 = STAY   current room not yet cleared; fight first
  1 = NORTH  move toward grid_index - 13
  2 = SOUTH  move toward grid_index + 13
  3 = WEST   move toward grid_index - 1
  4 = EAST   move toward grid_index + 1
"""
from __future__ import annotations

from collections import deque
from enum import IntEnum

from binding_rl_agent.game_state import IsaacGameState, RoomInfo, RoomType

GRID_WIDTH = 13
GRID_SIZE  = GRID_WIDTH * GRID_WIDTH  # 169


class NavHint(IntEnum):
    STAY  = 0
    NORTH = 1
    SOUTH = 2
    WEST  = 3
    EAST  = 4


# Priority order when choosing a navigation target.
# Lower number = higher priority.
_TARGET_PRIORITY: dict[int, int] = {
    RoomType.BOSS:      0,
    RoomType.TREASURE:  1,
    RoomType.DEVIL:     2,
    RoomType.ANGEL:     2,
    RoomType.DEFAULT:   3,
    RoomType.MINIBOSS:  3,
    RoomType.CHALLENGE: 3,
    RoomType.SACRIFICE: 4,
    RoomType.LIBRARY:   4,
    RoomType.ARCADE:    4,
    RoomType.DICE:      4,
    RoomType.BARREN:    5,
    RoomType.SHOP:      6,   # low priority — agent can't use it without coins
    RoomType.ERROR:     7,
    # SECRET and SUPERSECRET excluded from targets entirely (_UNBOMBABLE_TYPES)
}


class RoomGraph:
    """Immutable snapshot of a floor's room graph built from one game state."""

    def __init__(self, rooms: tuple[RoomInfo, ...]) -> None:
        self._rooms: dict[int, RoomInfo] = {r.grid_index: r for r in rooms}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def nav_hint(self, current_index: int) -> NavHint:
        """Return the navigation hint for the current room."""
        current = self._rooms.get(current_index)
        if current is None or not self._rooms:
            return NavHint.STAY

        # Stay and fight if current room isn't cleared yet.
        if not current.cleared:
            return NavHint.STAY

        target = self._choose_target(current_index)
        if target is None or target == current_index:
            return NavHint.STAY

        path = self._bfs_path(current_index, target)
        if path is None or len(path) < 2:
            return NavHint.STAY

        return _step_direction(path[0], path[1])

    def as_one_hot(self, current_index: int) -> list[float]:
        """5-element one-hot suitable for concatenating to model features."""
        hint = self.nav_hint(current_index)
        vec = [0.0] * len(NavHint)
        vec[int(hint)] = 1.0
        return vec

    @classmethod
    def from_game_state(cls, state: IsaacGameState) -> RoomGraph:
        return cls(state.floor_rooms)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _neighbors(self, idx: int) -> list[int]:
        col = idx % GRID_WIDTH
        candidates: list[tuple[int, bool]] = [
            (idx - GRID_WIDTH, True),          # north — always valid column
            (idx + GRID_WIDTH, True),          # south
            (idx - 1,          col > 0),       # west  — not on left edge
            (idx + 1,          col < GRID_WIDTH - 1),  # east — not on right edge
        ]
        return [
            n for n, valid in candidates
            if valid and 0 <= n < GRID_SIZE and n in self._rooms
        ]

    # Room types excluded from navigation targets:
    # - SECRET/SUPERSECRET require bombing a wall to enter
    # - CURSE costs health to enter (agent has no health awareness)
    _EXCLUDED_TYPES: frozenset[int] = frozenset({
        RoomType.SECRET,
        RoomType.SUPERSECRET,
        RoomType.CURSE,
    })

    def _navigable(self, room: RoomInfo) -> bool:
        return room.room_type not in self._EXCLUDED_TYPES

    def _choose_target(self, current_index: int) -> int | None:
        """
        Priority:
          1. Nearest unvisited navigable room (explore first), by room type priority
          2. Nearest uncleared navigable room (clear remaining enemies)
          3. Boss room (fallback when floor is fully explored and cleared)
        """
        unvisited = [r for r in self._rooms.values() if r.visited == 0 and self._navigable(r)]
        if unvisited:
            return self._nearest_by_priority(current_index, unvisited)

        uncleared = [r for r in self._rooms.values() if not r.cleared and self._navigable(r)]
        if uncleared:
            return self._nearest_by_priority(current_index, uncleared)

        boss_rooms = [r for r in self._rooms.values() if r.room_type == RoomType.BOSS]
        if boss_rooms:
            return boss_rooms[0].grid_index

        return None

    def _nearest_by_priority(
        self, start: int, candidates: list[RoomInfo]
    ) -> int | None:
        """
        BFS outward from start; return the first candidate reached.
        Among candidates at the same BFS distance, prefer by room type priority.
        """
        candidate_set = {r.grid_index: r for r in candidates}
        visited: set[int] = {start}
        # BFS layer by layer so we can compare within the same distance
        current_layer = [start]
        while current_layer:
            hits = [candidate_set[n] for n in current_layer if n in candidate_set]
            if hits:
                hits.sort(key=lambda r: _TARGET_PRIORITY.get(r.room_type, 99))
                return hits[0].grid_index
            next_layer: list[int] = []
            for idx in current_layer:
                for n in self._neighbors(idx):
                    if n not in visited:
                        visited.add(n)
                        next_layer.append(n)
            current_layer = next_layer
        return None

    def _bfs_path(self, start: int, goal: int) -> list[int] | None:
        """Return the shortest path from start to goal, or None if unreachable."""
        if start == goal:
            return [start]
        parent: dict[int, int | None] = {start: None}
        queue: deque[int] = deque([start])
        while queue:
            idx = queue.popleft()
            for n in self._neighbors(idx):
                if n not in parent:
                    parent[n] = idx
                    if n == goal:
                        return _reconstruct_path(parent, start, goal)
                    queue.append(n)
        return None


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _step_direction(from_idx: int, to_idx: int) -> NavHint:
    diff = to_idx - from_idx
    if diff == -GRID_WIDTH:
        return NavHint.NORTH
    if diff == GRID_WIDTH:
        return NavHint.SOUTH
    if diff == -1:
        return NavHint.WEST
    if diff == 1:
        return NavHint.EAST
    return NavHint.STAY


def _reconstruct_path(
    parent: dict[int, int | None], start: int, goal: int
) -> list[int]:
    path: list[int] = []
    cur: int | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path
