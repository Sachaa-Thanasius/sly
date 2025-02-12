# ruff: noqa: PLW0603

"""Shim for typing- and annotation-related symbols to avoid runtime dependencies on `typing` or `typing-extensions`.

Warning: Do not directly import annotation-related symbols from this module (e.g. `from ._typing_compat import Any`)!
Doing so will trigger the module-level `__getattr__`, causing `typing` to get imported. Instead, import the module and
use symbols via attribute access as needed (e.g. `from . import _typing_compat [as _t]`). To avoid those symbols being
evaluated at runtime, which would also cause `typing` to get imported, make sure to put
`from __future__ import annotations` at the top of the module.
"""

from __future__ import annotations

import sys


TYPE_CHECKING = False


class _PlaceholderMeta(type):
    _source_module: str

    def __init__(self, *args: object, **kwargs: object):
        super().__init__(*args, **kwargs)

        if not hasattr(self, "_source_module"):
            msg = "A placeholder must indicate the source of the original with a `_source_module` variable."
            raise ValueError(msg)

        self.__doc__ = f"Placeholder for {self._source_module}.{self.__name__}."

    def __repr__(self, /):
        return f"<import placeholder for {self._source_module}.{self.__name__}>"


__all__ = (
    # Annotation/typing symbols.
    "Callable",
    "Collection",
    "Generator",
    "Iterator",
    "Any",
    "ClassVar",
    "Final",
    "Literal",
    "Optional",
    "TextIO",
    "Union",
    # Annotation/typing symbols with version-dependent handling.
    "TypeAlias",
    "Self",
    # Used at runtime.
    "TYPE_CHECKING",
    "cast",
    "final",
    # Other.
    "CallableT",
    "LoggerLike",
)


def __getattr__(name: str, /) -> object:
    # Save the imported symbols in the global namespace to avoid re-importing in the future.

    if name in {"Callable", "Collection", "Generator", "Iterator"}:
        global Callable, Collection, Generator, Iterator

        from collections.abc import Callable, Collection, Generator, Iterator

        return globals()[name]

    if name in {"Any", "ClassVar", "Final", "Literal", "Optional", "Union"}:
        global Any, ClassVar, Final, Literal, Optional, TextIO, Union

        from typing import Any, ClassVar, Final, Literal, Optional, TextIO, Union

        return globals()[name]

    if name == "TypeAlias" and sys.version_info >= (3, 10):
        global TypeAlias

        from typing import TypeAlias

        return globals()[name]

    if name == "Self" and sys.version_info >= (3, 11):
        global Self

        from typing import Self

        return globals()[name]

    if name == "CallableT":
        global CallableT

        from collections.abc import Callable
        from typing import Any, TypeVar

        CallableT = TypeVar("CallableT", bound=Callable[..., Any])
        return CallableT

    if name == "LoggerLike":
        global LoggerLike

        from typing import Any, Protocol

        class LoggerLike(Protocol):
            def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None: ...
            def info(self, msg: Any, *args: Any, **kwargs: Any) -> None: ...
            def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None: ...
            def error(self, msg: Any, *args: Any, **kwargs: Any) -> None: ...
            def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None: ...

        return LoggerLike

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))


# TypeAlias: Below 3.10, create a placeholder.
if TYPE_CHECKING:
    from typing_extensions import TypeAlias
elif sys.version_info < (3, 10):  # pragma: <3.10 cover

    class TypeAlias(metaclass=_PlaceholderMeta):
        _source_module = "typing"


# Self: Below 3.11, create a placeholder.
if TYPE_CHECKING:
    from typing_extensions import Self
elif sys.version_info < (3, 11):  # pragma: <3.11 cover

    class Self(metaclass=_PlaceholderMeta):
        _source_module = "typing"


# cast: Used at runtime.
if TYPE_CHECKING:
    from typing import cast
else:

    def cast(typ: object, val: object) -> object:
        return val


# final: Used at runtime.
if TYPE_CHECKING:
    from typing import final
else:

    def final(f: object) -> object:
        try:
            f.__final__ = True
        except (AttributeError, TypeError):  # pragma: no cover
            # Skip the attributes silently if they are not writable.
            # AttributeError happens if the object has __slots__ or a
            # read-only property, TypeError if it's a builtin class.
            pass

        return f
