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
"""A modified version of cluegen, with support for typing and simple defaults."""

import sys
from collections.abc import Callable
from functools import reduce
from types import MemberDescriptorType
from typing import Any, ClassVar, Final, Protocol, TypeVar, final, get_origin

from ._typing_compat import Self, dataclass_transform, override

_DBT_contra = TypeVar("_DBT_contra", bound="DatumBase", contravariant=True)


class _ClueGenDescriptor(Protocol[_DBT_contra]):
    def __get__(self, instance: _DBT_contra, owner: type[_DBT_contra]) -> Any: ...
    def __set_name__(self, owner: type[_DBT_contra], name: str) -> None: ...


__all__ = ("all_clues", "all_defaults", "cluegen", "DatumBase", "Datum")


@final
class _Nothing:
    __slots__ = ()

    @override
    def __repr__(self) -> str:
        return "CLUEGEN_NOTHING"


CLUEGEN_NOTHING: Final[Any] = _Nothing()
"""Internal sentinel that can act as a placeholder for a mutable default value in a signature."""


def cluegen(func: Callable[[type[_DBT_contra]], str]) -> _ClueGenDescriptor[_DBT_contra]:
    """Create a custom ClueGen descriptor that will, as needed, execute and assign the code resulting from `func`."""

    def __get__(self: _ClueGenDescriptor[_DBT_contra], instance: _DBT_contra, owner: type[_DBT_contra]) -> Any:
        try:
            owner_mod = sys.modules[owner.__module__]
        except KeyError:
            global_ns = {"CLUEGEN_NOTHING": CLUEGEN_NOTHING}
        else:
            global_ns = dict(owner_mod.__dict__, CLUEGEN_NOTHING=CLUEGEN_NOTHING)

        local_ns: dict[str, Any] = {}
        code = func(owner)

        exec(code, global_ns, local_ns)  # noqa: S102
        method = local_ns[func.__name__]

        setattr(owner, func.__name__, method)
        return method.__get__(instance, owner)

    def __set_name__(self: _ClueGenDescriptor[_DBT_contra], owner: type[_DBT_contra], name: str) -> None:
        try:
            owner.__dict__["_methods"]
        except KeyError:
            # Retrieve from superclass and assign in current class dict, in theory.
            owner._methods = list(owner._methods)

        owner._methods.append((name, self))

    return type(f"ClueGen_{func.__name__}", (), {"__get__": __get__, "__set_name__": __set_name__})()  # pyright: ignore


def all_clues(cls: type) -> dict[str, Any]:
    """Get all annotations from a type. This excludes ClassVars while traversing the type's mro."""

    clues = reduce(lambda x, y: getattr(y, "__annotations__", {}) | x, cls.__mro__, {})
    return {name: ann for name, ann in clues.items() if (get_origin(ann) or ann) is not ClassVar}


def all_defaults(cls: type, clues: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect and remove all default values from class-level variables with annotations, if they exist.

    Notes
    -----
    This accounts for some mutable defaults: lists, dicts, sets, and bytearrays.
    """

    defaults: dict[str, Any] = {}
    mutable_defaults: dict[str, Any] = {}

    for name in clues:
        try:
            default = getattr(cls, name)
        except AttributeError:
            pass
        else:
            if not isinstance(default, MemberDescriptorType):
                if isinstance(default, (list, dict, set, bytearray)):
                    mutable_defaults[name] = default
                    defaults[name] = CLUEGEN_NOTHING
                else:
                    defaults[name] = default

                delattr(cls, name)

    return defaults, mutable_defaults


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
    """Base data structure that automatically creates `__init__`, `__repr__`, `__eq__`, and `__match_args__` based on
    class annotations.

    Notes
    -----
    This is the class decorated with `dataclass_transform()` instead of `DatumBase` because the latter allows creation
    schemes that `dataclass_transform()` isn't specified to account for.
    """

    __slots__ = ()

    @classmethod
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls.__match_args__ = tuple(all_clues(cls))

    @cluegen
    def __init__(cls: type[Self]) -> str:  # pyright: ignore
        clues = all_clues(cls)
        defaults, mutable_defaults = all_defaults(cls, clues)

        args = ((name, f'{name}: {getattr(clue, "__name__", repr(clue))}') for name, clue in clues.items())
        args = ", ".join((f"{arg} = {defaults[name]!r}" if name in defaults else arg) for name, arg in args)
        body = "\n".join(
            (
                *(f"    self.{name} = {name}" for name in clues if name not in mutable_defaults),
                *(
                    f"    self.{name} = {name} if {name} is not CLUEGEN_NOTHING else {mutable_defaults[name]}"
                    for name in mutable_defaults
                ),
            )
        )
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
