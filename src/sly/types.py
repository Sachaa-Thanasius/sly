"""Types for functions that only exist during sly class creation, i.e. the  `@_` decorator.

Extended Summary
----------------
Because `@_` doesn't technically exist outside of the body of a `sly.Lexer` or `sly.Parser` subclass body, it cannot
be imported at runtime. However, it can still provide typing and intellisense support if "fake" imported in such a way
that type-checkers and IDEs can see it, e.g. within an `if typing.TYPE_CHECKING: ...` block, that doesn't execute at
runtime.

Raises
------
ImportError
    If `_` is imported at runtime.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeVar

__all__ = ("_",)

_CallableT = TypeVar("_CallableT", bound=Callable[..., object])


class _RuleDecorator(Protocol):
    # Technically, the `@_` for Lexer has a first parameter name: "pattern". However, since `@_` for lex and yacc both
    # only accept positional-only strings, it shouldn't matter.
    def __call__(self, rule: str, *extras: str) -> Callable[[_CallableT], _CallableT]: ...


if TYPE_CHECKING:
    from typing import Final, cast

    _: Final = cast(_RuleDecorator, object())
    """Typing aid for `@_` within `sly.Lexer` and `sly.Parser` subclasses. Do not import at runtime."""
