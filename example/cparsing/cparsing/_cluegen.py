# region License
# -----------------------------------------------------------------------------
# cluegen.py
#
# Classes generated from type clues.
#
#     https://github.com/dabeaz/cluegen
#
# Author: David Beazley (@dabeaz).
#         http://www.dabeaz.com
#
# Copyright (C) 2018-2021.
# Copyright (C) 2024, Sachaa-Thanasius
#
# Permission is granted to use, copy, and modify this code in any
# manner as long as this copyright message and disclaimer remain in
# the source code.  There is no warranty.  Try to use the code for the
# greater good.
# -----------------------------------------------------------------------------
# endregion
"""A modified version of cluegen with typing support."""

import sys
from collections.abc import Callable
from functools import reduce
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar, get_origin

from ._typing_compat import Self, dataclass_transform

if TYPE_CHECKING:
    from types import MemberDescriptorType
else:
    MemberDescriptorType = type(type(lambda: None).__globals__)

_DBT = TypeVar("_DBT", bound="DatumBase")


__all__ = ("all_clues", "cluegen", "DatumBase", "Datum")


class _ClueGenDescriptor(Protocol[_DBT]):
    owner: type[_DBT]

    def __get__(self, instance: _DBT, owner: type[_DBT]) -> Any: ...
    def __set_name__(self, owner: type[_DBT], name: str) -> None: ...


def cluegen(func: Callable[[type[_DBT]], str]) -> _ClueGenDescriptor[_DBT]:
    """Create a custom ClueGen descriptor that will, as needed, execute and assign the code resulting from `func`."""

    def __get__(self: _ClueGenDescriptor[_DBT], instance: _DBT, owner: type[_DBT]) -> Any:
        try:
            owner_mod = sys.modules[owner.__module__]
        except KeyError:
            global_ns = {}
        else:
            global_ns = vars(owner_mod)
        local_ns: dict[str, Any] = {}
        code = func(owner)

        exec(code, global_ns, local_ns)  # noqa: S102
        method = local_ns.popitem()[1]

        setattr(owner, func.__name__, method)
        return method.__get__(instance, owner)

    def __set_name__(self: _ClueGenDescriptor[_DBT], owner: type[_DBT], name: str) -> None:
        try:
            owner.__dict__["_methods"]
        except KeyError:
            # Retrieve from superclass and assign in current class dict, in theory.
            owner._methods = list(owner._methods)
        finally:
            owner._methods.append((name, self))

    return type(f"ClueGen_{func.__name__}", (), {"__get__": __get__, "__set_name__": __set_name__})()  # pyright: ignore


def all_clues(cls: type) -> dict[str, Any]:
    """Get all annotations from a type, including from superclasses and excluding ClassVars."""

    clues = reduce(lambda x, y: getattr(y, "__annotations__", {}) | x, cls.__mro__, {})
    return {name: ann for name, ann in clues.items() if (get_origin(ann) or ann) is not ClassVar}


class DatumBase:
    """Base class for defining data structures."""

    __slots__ = ()
    _methods: ClassVar[list[tuple[str, _ClueGenDescriptor[Self]]]] = []

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        submethods: list[tuple[str, Any]] = []
        for name, val in cls._methods:
            if name not in cls.__dict__:
                setattr(cls, name, val)
                submethods.append((name, val))
            elif val is cls.__dict__[name]:
                submethods.append((name, val))

        if submethods != cls._methods:
            cls._methods = submethods


@dataclass_transform()
class Datum(DatumBase):
    __slots__ = ()

    @classmethod
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls.__match_args__ = tuple(all_clues(cls))

    @cluegen
    def __init__(cls: type[Self]) -> str:  # pyright: ignore
        _missing = object()  # sentinel
        clues = all_clues(cls)
        defaults: dict[str, Any] = {}

        for name in clues:
            attr = getattr(cls, name, _missing)
            if attr is not _missing and not isinstance(attr, MemberDescriptorType):
                defaults[name] = attr
                delattr(cls, name)

        args = ((name, f'{name}: {getattr(clue, "__name__", repr(clue))}') for name, clue in clues.items())
        args = ", ".join((f"{arg} = {defaults[name]!r}" if name in defaults else arg) for name, arg in args)
        body = "\n".join(f"   self.{name} = {name}" for name in clues)
        return f"def __init__(self, {args}):\n{body}\n"  # noqa: PLE0101

    @cluegen
    def __repr__(cls: type[Self]) -> str:  # pyright: ignore
        clues = all_clues(cls)
        fmt = ", ".join(f"{name}={{self.{name}!r}}" for name in clues)
        return f'def __repr__(self) -> str:\n    return f"{{type(self).__name__}}({fmt})"'

    @cluegen
    def __eq__(cls: type[Self]) -> str:  # pyright: ignore  # noqa: PLE0302
        clues = all_clues(cls)
        selfvals = ", ".join(f"self.{name}" for name in clues)
        othervals = ", ".join(f"other.{name}" for name in clues)
        return (
            f"def __eq__(self, other: object) -> bool:\n"
            f"    if not isinstance(self, type(other)):\n"
            f"        return NotImplemented\n"
            f"\n"
            f"    return ({selfvals},) == ({othervals},)\n"
        )
