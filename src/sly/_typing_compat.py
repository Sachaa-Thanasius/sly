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
        self.__doc__ = f"Placeholder for {self._source_module}.{self.__name__}."

    def __repr__(self):
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
    "Self",
    "TypeAlias",
    # Used at runtime.
    "TYPE_CHECKING",
    "cast",
    # Other.
    "CallableT",
    "LoggerLike",
)


def __getattr__(name: str, /) -> object:
    # Save the imported symbols in the globals to avoid future imports.

    if name in {"Callable", "Collection", "Generator", "Iterator"}:
        global Callable, Collection, Generator, Iterator  # noqa: PLW0603

        from collections.abc import Callable, Collection, Generator, Iterator

        return globals()[name]

    if name in {"Any", "ClassVar", "Final", "Literal", "Optional", "Union"}:
        global Any, ClassVar, Final, Literal, Optional, TextIO, Union  # noqa: PLW0603

        from typing import Any, ClassVar, Final, Literal, Optional, TextIO, Union

        return globals()[name]

    if (
        (name == "TypeAlias" and sys.version_info >= (3, 10))
        or (name == "Self" and sys.version_info >= (3, 11))
    ):  # fmt: skip
        import typing

        symbol = getattr(typing, name)
        globals()[name] = symbol

        return symbol

    if name == "CallableT":
        global CallableT  # noqa: PLW0603

        from collections.abc import Callable
        from typing import Any, TypeVar

        CallableT = TypeVar("CallableT", bound=Callable[..., Any])
        return CallableT

    if name == "LoggerLike":
        global LoggerLike  # noqa: PLW0603

        from typing import Any, Protocol

        class LoggerLike(Protocol):
            def debug(self, msg: Any, *args: object, **kwargs: object) -> None: ...
            def info(self, msg: Any, *args: object, **kwargs: object) -> None: ...
            def warning(self, msg: Any, *args: object, **kwargs: object) -> None: ...
            def error(self, msg: Any, *args: object, **kwargs: object) -> None: ...
            def critical(self, msg: Any, *args: object, **kwargs: object) -> None: ...

        return LoggerLike

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))


# TypeAlias: Below 3.10, create a placeholder.
if TYPE_CHECKING:
    from typing_extensions import TypeAlias
elif sys.version_info < (3, 10):

    class TypeAlias(metaclass=_PlaceholderMeta):
        _source_module = "typing"


# Self: Below 3.11, create a placeholder.
if TYPE_CHECKING:
    from typing_extensions import Self
elif sys.version_info < (3, 11):

    class Self(metaclass=_PlaceholderMeta):
        _source_module = "typing"


# cast: Used at runtime.
if TYPE_CHECKING:
    from typing import cast
else:

    def cast(typ: object, val: object) -> object:
        return val
