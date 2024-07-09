"""Bits and bobs, like typing-related shims and internal sentinels."""

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final, TypeVar

if sys.version_info >= (3, 9, 2):  # noqa: UP036 # Users might still be on 3.9.0.
    from types import GenericAlias as _GenericAlias
elif TYPE_CHECKING:

    class _GenericAlias:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

else:  # pragma: no cover
    from typing import _GenericAlias


__all__ = ("MISSING", "Self", "TypeAlias", "override")


CallableT = TypeVar("CallableT", bound=Callable[..., Any])

if sys.version_info >= (3, 12):  # pragma: >=3.12 cover
    from typing import override
elif TYPE_CHECKING:
    from typing_extensions import override
else:  # pragma: <3.12 cover

    def override(arg: object) -> Any:
        try:
            arg.__override__ = True
        except AttributeError:  # pragma: no cover
            pass
        return arg


class _PlaceholderGenericAlias(_GenericAlias):
    @override
    def __repr__(self) -> str:
        return f"<placeholder for {super().__repr__()}>"


class _PlaceholderMeta(type):
    def __getitem__(self, item: object) -> _PlaceholderGenericAlias:
        return _PlaceholderGenericAlias(self, item)

    @override
    def __repr__(self) -> str:
        return f"<placeholder for {super().__repr__()}>"


if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    from typing import Self
elif TYPE_CHECKING:
    from typing_extensions import Self
else:  # pragma: <3.11 cover

    class Self(metaclass=_PlaceholderMeta):
        pass


if sys.version_info >= (3, 10):  # pragma: >=3.10 cover
    from typing import TypeAlias
elif TYPE_CHECKING:
    from typing_extensions import TypeAlias
else:  # pragma: <3.10 cover

    class TypeAlias(metaclass=_PlaceholderMeta):
        pass


class _Missing:
    __slots__ = ()

    @override
    def __repr__(self) -> str:
        return "<MISSING>"


MISSING: Final[Any] = _Missing()
"""Internal sentinel."""
