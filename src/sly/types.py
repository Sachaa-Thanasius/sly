"""Types for functions that only exist during sly class creation, i.e. the decorators `@_` and `@subst`.

Extended Summary
----------------
Because the objects represented here don't technically exist outside of the body of a `sly.Lexer` or `sly.Parser`
subclass body, they cannot be imported at runtime. However, they can still provide typing and intellisense support if
"fake" imported in such a way that type-checkers and IDEs can see them, e.g. within an `if typing.TYPE_CHECKING: ...`
block, that doesn't execute at runtime.

Raises
------
ImportError
    If the module is imported at runtime. This is intentionally done by using `typing.type_check_only()`, which doesn't
    exist at runtime.
"""

from collections.abc import Callable
from typing import Any, Final, Protocol, TypeVar, cast, type_check_only

__all__ = ("_", "subst")

_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])


@type_check_only
class _RuleDecorator(Protocol):
    # Technically, the `@_` for lex has a first parameter name: "pattern". However, since `@_` for lex and yacc both
    # only accept positional-only strings, it shouldn't matter.
    def __call__(self, rule: str, *extras: str) -> Callable[[_CallableT], _CallableT]: ...


@type_check_only
class _SubstitutionDecorator(Protocol):
    def __call__(self, sub: dict[str, str], *extras: dict[str, str]) -> Callable[[_CallableT], _CallableT]: ...


_: Final = cast(_RuleDecorator, object())
"""Typing aid for `@_` within `sly.Lexer` and `sly.Parser` subclasses. Do not import at runtime."""

subst: Final = cast(_SubstitutionDecorator, object())
"""Typing aid for `@subst` within `sly.Parser` subclasses. Do not import at runtime."""
