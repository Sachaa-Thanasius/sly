"""Support creation of simple AST nodes."""

import sys

from ._misc import Self


class AST:
    # TODO: Investigate incorporating elements of dataklasses, cluegen, etc.

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        mod = sys.modules[cls.__module__]
        if not hasattr(cls, "__annotations__"):
            return

        hints = list(cls.__annotations__.items())

        def __init__(self: Self, *args: object, **kwargs: object) -> None:
            if len(hints) != len(args):
                msg = f"Expected {len(hints)} arguments"
                raise TypeError(msg)

            for arg, (name, val) in zip(args, hints):
                if isinstance(val, str):
                    val = getattr(mod, val)  # noqa: PLW2901
                if not isinstance(arg, val):
                    msg = f"{name!r} argument must be of type {val!r}."
                    raise TypeError(msg)

                setattr(self, name, arg)

        cls.__init__ = __init__
