"""Some utilities for internal use."""

import dataclasses
from typing import Any, Optional

from sly import Parser
from sly._misc import Self

__all__ = ("Coord",)


@dataclasses.dataclass
class Coord:
    filename: str
    line_start: int
    col_start: int
    line_end: Optional[int] = None
    col_end: Optional[int] = None

    @classmethod
    def from_literal(cls, p: Any, literal: str, filename: str = "") -> Self:
        return cls(filename, p.lineno, p.index, None, None)

    @classmethod
    def from_prod(cls, parser: Parser, p: Any, filename: str = "") -> Self:
        lineno = parser.line_position(p)
        col_start, col_end = parser.index_position(p)
        return cls(filename, lineno, col_start, None, col_end)

    @classmethod
    def combine_coords(cls, coord1: Self, coord2: Self) -> Self:
        line_start = coord1.line_start
        line_end = coord2.line_end or coord2.line_start
        col_start = coord1.col_start
        col_end = coord2.col_start or coord2.col_end

        return cls("", line_start, line_end, col_start, col_end)

    def __str__(self):
        return f"{self.filename}:{self.line_start}:{(self.col_start, self.col_end)}"
