"""Some utilities for internal use."""

from typing import TYPE_CHECKING, Any, Optional

from ._cluegen import Datum
from ._typing_compat import Self, override

if TYPE_CHECKING:
    from .c_parser import CParser

__all__ = ("Coord",)


class Coord(Datum):
    line_start: int
    col_start: int
    line_end: Optional[int] = None
    col_end: Optional[int] = None
    filename: str = "<unknown>"

    @classmethod
    def from_literal(cls, p: Any, literal: str) -> Self:
        return cls(p.lineno, p.index, None, None)

    @classmethod
    def from_prod(cls, parser: "CParser", p: Any) -> Self:
        lineno = parser.line_position(p)
        assert lineno

        col_start, col_end = parser.index_position(p)
        assert col_start
        assert col_end

        return cls(lineno, col_start, None, col_end)

    @override
    def __str__(self) -> str:
        fmt = f"filename={self.filename} | start=({self.line_start}, {self.col_start})"
        if self.line_end is not None or self.col_end is not None:
            fmt += f" | end=({self.line_end}, {self.col_end})"
        return fmt
