from __future__ import annotations


__all__ = ("TokenError",)


class TokenError(Exception):
    """Exception raised if an invalid character is encountered."""

    value: str
    lineno: int
    column: int

    def __init__(self, msg: str, value: str, lineno: int, column: int, /) -> None:
        super().__init__(msg)
        self.value = value
        self.lineno = lineno
        self.column = column
