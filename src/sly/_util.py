"""Bits and bobs for internal use."""

from __future__ import annotations

from . import _typing_compat as _t


__all__ = ("MISSING",)


@_t.final
class _Missing:
    __slots__ = ()

    def __repr__(self, /) -> str:
        return "<MISSING>"


MISSING: _t.Final[_t.Any] = _Missing()
"""Internal sentinel. It should NEVER be seen in user space."""
