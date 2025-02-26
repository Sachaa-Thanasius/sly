from __future__ import annotations


__all__ = ("TokenError",)


class TokenError(Exception):
    """Exception raised if an invalid character is encountered."""

    def __init__(self, msg: str, value: str, lineno: int, column: int, /) -> None:
        super().__init__(msg)
        self.value: str = value
        self.lineno: int = lineno
        self.column: int = column
