"""Types for objects that only exist during class creation, i.e. the sly decorators `@_` and `@subst`.

Extended Summary
----------------
Because the objects represented here don't technically exist outside of the body of a `sly.Lexer` or `sly.Parser`
subclass body, these cannot be imported at runtime. Only import from this module in places where the import won't
execute, e.g. within an `if typing.TYPE_CHECKING: ...` block.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Protocol, TypeVar, cast, type_check_only

    __all__ = ("_", "subst")

    _CallableT = TypeVar("_CallableT", bound=Callable[..., Any])

    @type_check_only
    class _RuleDecorator(Protocol):
        def __call__(self, rule: str, *extras: str) -> Callable[[_CallableT], _CallableT]: ...

    @type_check_only
    class _SubstitutionDecorator(Protocol):
        def __call__(self, sub: dict[str, str], *extras: dict[str, str]) -> Callable[[_CallableT], _CallableT]: ...

    # Typing hack to account for `_` existing in a sly.Lexer or sly.Parser subclass's namespace (and `subst` in a
    # sly.Parser subclass's namespace only during class creation.
    # Should only ever be imported within an `if TYPE_CHECKING` block.
    _ = cast(_RuleDecorator, object())
    subst = cast(_SubstitutionDecorator, object())
