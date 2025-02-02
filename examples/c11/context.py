from __future__ import annotations

from collections import deque


class CNameContext:
    def __init__(self, *sets: set[str]):
        self.sets = deque(sets or [set()])

    def __contains__(self, key: str, /) -> bool:
        return key in self.sets[-1]

    def declare_typedef_name(self, name: str, /) -> None:
        self.sets[-1].add(name)

    def declare_var_name(self, name: str, /) -> None:
        self.sets[-1].remove(name)

    def save_context(self) -> None:
        self.sets.append(set())

    def restore_context(self) -> None:
        self.sets.pop()
