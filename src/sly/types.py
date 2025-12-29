"""Types for symbols that only exist during sly class creation, i.e. the  `@_` decorator. Do not import at runtime:
this will intentionally raise an ImportError.

Because the `_` decorator doesn't exist outside of the body of a `sly.Lexer` or `sly.Parser` subclass body, it cannot be
imported at runtime. However, it can still provide typing and intellisense support if "fake" imported such that
type-checkers and IDEs can see it but the Python runtime doesn't, e.g. within an `if typing.TYPE_CHECKING: ...` block.
That's what this module provides.
"""

from __future__ import annotations


TYPE_CHECKING = False

if not TYPE_CHECKING:
    msg = "This module is not meant to be imported at runtime; see docstring for more details."
    raise ImportError(msg, name=__spec__.name)

from collections.abc import Callable  # noqa: E402
from typing import Any, Final, Protocol, TypeVar, cast, type_check_only  # noqa: E402


__all__ = ("_",)

_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])


@type_check_only
class _RuleDecorator(Protocol):
    # Technically, the `@_` for Lexer has a different first parameter name: "pattern". However, since `@_` in Lexer
    # and Parser both only accept positional-only strings, it shouldn't matter.
    def __call__(self, rule: str, *extras: str) -> Callable[[_CallableT], _CallableT]: ...


_: Final = cast("_RuleDecorator", object())
"""Typing aid for `@_` within `sly.Lexer` and `sly.Parser` subclasses. Do not import at runtime."""
