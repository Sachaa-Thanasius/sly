from __future__ import annotations


class Token:
    """Representation of a single token."""

    __slots__ = ("kind", "value", "lineno", "column")

    kind: str
    value: str
    lineno: int
    column: int

    def __init__(self, kind: str, value: str, lineno: int, column: int, /):
        self.kind = kind
        self.value = value
        self.lineno = lineno
        self.column = column

    def __repr__(self, /):
        return (
            f"{self.__class__.__name__}({self.kind!r}, {self.value!r}, lineno={self.lineno!r}, column={self.column!r})"
        )
