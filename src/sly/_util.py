"""Bits and bobs for internal use."""

from __future__ import annotations

import sys

from . import _typing_compat as _t


__all__ = ("unwrap", "MISSING")


def unwrap(
    func: _t.Callable[..., _t.Any],
    *,
    stop: _t.Optional[_t.Callable[[_t.Callable[..., _t.Any]], _t.Any]] = None,
) -> _t.Any:
    """A adapted version of `inspect.unwrap()`. See the original docstring below:

    Get the object wrapped by *func*.

    Follows the chain of :attr:`__wrapped__` attributes returning the last
    object in the chain.

    *stop* is an optional callback accepting an object in the wrapper chain
    as its sole argument that allows the unwrapping to be terminated early if
    the callback returns a true value. If the callback never returns a true
    value, the last object in the chain is returned as usual. For example,
    :func:`signature` uses this to stop unwrapping if any object in the
    chain has a ``__signature__`` attribute defined.

    :exc:`ValueError` is raised if a cycle is encountered.
    """

    f = func  # remember the original func for error reporting
    # Memoise by id to tolerate non-hashable objects, but store objects to
    # ensure they aren't destroyed, which would allow their IDs to be reused.
    memo = {id(f): f}
    recursion_limit = sys.getrecursionlimit()
    while not isinstance(func, type) and hasattr(func, "__wrapped__"):
        if stop is not None and stop(func):
            break
        func = func.__wrapped__  # pyright: ignore [reportFunctionMemberAccess] # Part of the function's operation.
        id_func = id(func)
        if (id_func in memo) or (len(memo) >= recursion_limit):
            msg = f"wrapper loop when unwrapping {f!r}"
            raise ValueError(msg)
        memo[id_func] = func
    return func


@_t.final
class _Missing:
    __slots__ = ()

    def __repr__(self, /) -> str:
        return "<MISSING>"


MISSING: _t.Final[_t.Any] = _Missing()
"""Internal sentinel."""
