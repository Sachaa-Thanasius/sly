# region License
# -----------------------------------------------------------------------------
# sly: yacc.py
#
# Copyright (C) 2024, Sachaa-Thanasius
# Copyright (C) 2016-2018
# David M. Beazley (Dabeaz LLC)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the David Beazley or Dabeaz LLC may be used to
#   endorse or promote products derived from this software without
#  specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------
# endregion

from __future__ import annotations

import sys
from collections import Counter, defaultdict, deque
from itertools import count

from . import _typing_compat as _t
from .lex import Token


__all__ = ("Parser",)


@_t.final
class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<MISSING>"


MISSING: _t.Final[_t.Any] = _Missing()
"""Internal sentinel."""


def _inspect_unwrap(func: _t.Callable[..., _t.Any]) -> _t.Any:  # pragma: no cover
    """A adapted version of `inspect.unwrap()` to avoid depending on `inspect` at runtime.

    See the original docstring below:

    Get the object wrapped by *func*.

    Follows the chain of :attr:`__wrapped__` attributes returning the last
    object in the chain.

    :exc:`ValueError` is raised if a cycle is encountered.
    """

    f = func  # remember the original func for error reporting
    # Memoise by id to tolerate non-hashable objects, but store objects to
    # ensure they aren't destroyed, which would allow their IDs to be reused.
    memo = {id(f): f}
    recursion_limit = sys.getrecursionlimit()
    while not isinstance(func, type) and hasattr(func, "__wrapped__"):
        func = func.__wrapped__  # pyright: ignore [reportFunctionMemberAccess]
        id_func = id(func)
        if (id_func in memo) or (len(memo) >= recursion_limit):
            msg = f"wrapper loop when unwrapping {f!r}"
            raise ValueError(msg)
        memo[id_func] = func
    return func


class YaccError(Exception):
    """Exception raised for yacc-related build errors."""


class SlyLogger:
    """This object is a stand-in for a logging object created by the logging module.

    Extended Summary
    ----------------
    SLY will use this by default to create things such as the parser.out file. If a user wants more detailed
    information, they can create their own logging object and pass it into SLY.
    """

    def __init__(self, f: _t.TextIO) -> None:
        self.f = f

    def debug(self, msg: str, *args: object, **kwargs: object) -> None:
        self.f.write((msg % args) + "\n")

    info = debug

    def warning(self, msg: str, *args: object, **kwargs: object) -> None:
        self.f.write("WARNING: " + (msg % args) + "\n")

    def error(self, msg: str, *args: object, **kwargs: object) -> None:
        self.f.write("ERROR: " + (msg % args) + "\n")

    critical = debug


class YaccSymbol:
    """This class is used to hold non-terminal grammar symbols during parsing.

    It is intentionally duck type–compatible with `lex.Token`.

    Attributes
    ----------
    type: str
        Grammar symbol type.
    value: _t.Any
        Symbol value.
    lineno: int | None
        Starting line number.
    index: int | None
        Starting lex position.
    end: int | None
        Ending lex position.
    """

    __slots__ = ("type", "value", "lineno", "index", "end")

    def __init__(
        self,
        type: str,  # noqa: A002
        value: _t.Any = None,
        lineno: _t.Optional[int] = None,
        index: _t.Optional[int] = None,
        end: _t.Optional[int] = None,
    ) -> None:
        self.type: str = type
        self.value: _t.Any = value
        self.lineno: _t.Optional[int] = lineno
        self.index: _t.Optional[int] = index
        self.end: _t.Optional[int] = end

    def __str__(self) -> str:
        return self.type

    def __repr__(self) -> str:
        return str(self)


class YaccProduction:
    """This class is a wrapper around the objects actually passed to each grammar rule.

    Notes
    -----
    Index lookup and assignment actually assign the `.value` attribute of the underlying `YaccSymbol` object.
    """

    # In this case, slots conveniently prevent attempts to assign to proxied attributes. A much slower alternative is a
    # custom __setattr__ that calls super().__setattr__ if the attribute name begins with an underscore but otherwise
    # raises.
    __slots__ = ("_slice", "_namemap", "_stack")

    def __init__(self, s: list[YaccSymbol], stack: _t.Optional[list[YaccSymbol]] = None) -> None:
        self._slice: list[YaccSymbol] = s
        self._namemap: dict[str, _t.Callable[[list[YaccSymbol]], _t.Any]] = {}
        self._stack: list[YaccSymbol] = stack if (stack is not None) else []

    @property
    def lineno(self) -> int:
        """`int`: The line number of the given item.

        Raises
        ------
        AttributeError
            If no line number was found (or it was 0 for some reason).
        """

        for tok in self._slice:
            if tok.lineno:
                return tok.lineno
        msg = "No line number found."
        raise AttributeError(msg)

    @property
    def index(self) -> int:
        for tok in self._slice:
            if tok.index is not None:
                return tok.index
        msg = "No index attribute found."
        raise AttributeError(msg)

    @property
    def end(self) -> _t.Optional[int]:
        return next((tok.end for tok in reversed(self._slice) if tok.end), None)

    def __getitem__(self, index: int, /) -> _t.Any:
        if index >= 0:
            return self._slice[index].value
        else:
            return self._stack[index].value

    def __setitem__(self, n: int, value: _t.Any, /) -> None:
        if n >= 0:
            self._slice[n].value = value
        else:
            self._stack[n].value = value

    def __len__(self) -> int:
        return len(self._slice)

    def __getattr__(self, name: str, /) -> _t.Any:
        if name in self._namemap:
            return self._namemap[name](self._slice)
        else:
            msg = f"No symbol {name}. Must be one of {{{', '.join(self._namemap)}}}."
            raise AttributeError(msg)


# ============================================================================
# region -------- Grammar Representation --------
#
# The following functions, classes, and variables are used to represent and
# manipulate the rules that make up a grammar.
# ============================================================================


class Production:
    """This class stores the raw information about a single production or grammar rule.

    A grammar rule refers to a specification such as this: ``expr : expr PLUS term``.

    Attributes
    ----------
    number: int
        Production number.
    name: str
        Name of the production, e.g. "expr".
    prod: tuple[str, ...]
        A list of symbols on the right side, e.g. ("expr", "PLUS", "term").
    prec: tuple[str, int]
        Production precedence level.
    func: _t.Callable[[Parser, YaccProduction], _t.Any]
        Function that executes on reduce.
    file: str
        File where production function is defined.
    line: int
        Line number where production function is defined.
    len: int
        Length of the production (number of symbols on right hand side).
    usyms: set[str]
        Set of unique symbols found in the production.
    """

    __slots__ = (
        "name",
        "prod",
        "number",
        "func",
        "file",
        "line",
        "prec",
        "len",
        "usyms",
        "namemap",
        "lr_items",
        "lr_next",
        "lr0_added",
        "reduced",
    )

    def __init__(
        self,
        number: int,
        name: str,
        prod: list[str],
        func: _t.Callable[[Parser, YaccProduction], _t.Any],
        precedence: tuple[str, int] = ("right", 0),
        file: str = "",
        line: int = 0,
        *,
        name_aliases: dict[str, list[str]],
    ) -> None:
        self.name: str = name
        self.prod: tuple[str, ...] = tuple(prod)
        self.number: int = number
        self.func: _t.Callable[[Parser, YaccProduction], _t.Any] = func
        self.file: str = file
        self.line: int = line
        self.prec: tuple[str, int] = precedence

        # Internal settings used during table construction
        self.len: int = len(self.prod)

        # Create a list of unique production symbols used in the production
        self.usyms: set[str] = set(self.prod)

        # Create a name mapping
        # First determine (in advance) if there are duplicate names
        namecount: Counter[str] = Counter()
        for key in self.prod:
            namecount[key] += 1
            if key in name_aliases:
                namecount.update(name_aliases[key])

        # Now, walk through the names and generate accessor functions
        nameuse: Counter[str] = Counter()
        namemap: dict[str, _t.Callable[[list[YaccSymbol]], _t.Any]] = {}
        for index, key in enumerate(self.prod):
            if namecount[key] > 1:
                k = f"{key}{nameuse[key]}"
                nameuse[key] += 1
            else:
                k = key
            namemap[k] = lambda s, i=index: s[i].value

            if key in name_aliases:
                for n, alias in enumerate(name_aliases[key]):
                    if namecount[alias] > 1:
                        k = f"{alias}{nameuse[alias]}"
                        nameuse[alias] += 1
                    else:
                        k = alias

                    # The value is either a list (for repetition) or a tuple for optional
                    def _anon_accessor(s: list[YaccSymbol], i: int = index, n: int = n) -> _t.Any:
                        val = s[i].value
                        if isinstance(val, list):
                            return [x[n] for x in val]  # pyright: ignore [reportUnknownVariableType]
                        else:
                            return val[n]

                    namemap[k] = _anon_accessor

        self.namemap = namemap

        # List of all LR items for the production
        self.lr_items: list[LRItem] = []
        self.lr_next: _t.Optional[LRItem] = None

        self.lr0_added: int = 0

        self.reduced: int = 0

    def __str__(self) -> str:
        if self.prod:
            s = f"{self.name} -> {' '.join(self.prod)}"
        else:
            s = f"{self.name} -> <empty>"

        if self.prec[1]:
            s += f"  [precedence={self.prec[0]}, level={self.prec[1]}]"

        return s

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self})"

    def __len__(self) -> int:
        return len(self.prod)

    def __getitem__(self, index: int, /) -> str:
        return self.prod[index]

    def lr_item(self, n: int, prodnames: dict[str, list[Production]]) -> _t.Optional[LRItem]:
        """Return the nth lr_item from the production (or None if at the end)."""

        if n > len(self.prod):
            return None

        p = LRItem(self, n)

        # Precompute the list of productions immediately following.
        try:
            p.lr_after = prodnames[p.prod[n + 1]]
        except (IndexError, KeyError):
            p.lr_after = []

        try:
            p.lr_before = p.prod[n - 1]
        except IndexError:
            p.lr_before = None

        return p


class LRItem:
    """This class represents a specific stage of parsing a production rule.

    For example, ``expr : expr . PLUS term``, where the "." represents the current location of the parse.

    Attributes
    ----------
    name: str
        Name of the production, e.g. ``expr``.
    prod: tuple[str, ...]
        A list of symbols on the right side, e.g. ["expr", ".", "PLUS", "term"].
    number: int
        Production number.
    lr_next: LRItem | None
        Next LR item.

        For instance, if we are ``expr -> expr . PLUS term``, then lr_next refers to ``expr -> expr PLUS . term``.
    lr_index: int
        LR item index (location of the ".") in the prod list.
    lookaheads: dict[int, list[str]]
        LALR lookahead symbols for this item.
    len: int
        Length of the production (number of symbols on right hand side).
    lr_after: list[Production]
        List of all productions that immediately follow.
    lr_before: str | None
        Grammar symbol immediately before.
    """

    __slots__ = (
        "name",
        "prod",
        "number",
        "lr_index",
        "lookaheads",
        "len",
        "usyms",
        "lr_next",
        "lr_after",
        "lr_before",
    )

    def __init__(self, p: Production, n: int) -> None:
        self.name: str = p.name
        self.prod: tuple[str, ...] = p.prod[:n] + (".",) + p.prod[n:]
        self.number: int = p.number
        self.lr_index: int = n
        self.lookaheads: dict[int, set[str]] = {}
        self.len: int = len(self.prod)
        self.usyms: set[str] = p.usyms

        self.lr_next: _t.Optional[LRItem] = None
        self.lr_after: list[Production] = []
        self.lr_before: _t.Optional[str] = None

    def __str__(self) -> str:
        if self.prod:
            s = f"{self.name} -> {' '.join(self.prod)}"
        else:
            s = f"{self.name} -> <empty>"
        return s

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self})"


class GrammarError(YaccError):
    """Exception raised when something goes wrong in constructing the grammar."""


class Grammar:
    """This class represents the contents of the specified grammar.

    Extended Summary
    ----------------
    More specifically, it represents the contents of the grammar along with various computed properties such as first
    sets, follow sets, LR items, etc. This data is used for critical parts of the table generation process later.

    Attributes
    ----------
    Productions: list[Production]
        A list of all of the productions. The first entry is always reserved for the purpose of building an augmented
        grammar.
    Prodnames: dict[str, list[Production]]
        A dictionary mapping the names of nonterminals to a list of all productions of that nonterminal.
    Prodmap: dict[str, Production]
        A dictionary that is only used to detect duplicate productions.
    Terminals: dict[str, list[int]]
        A dictionary mapping the names of terminal symbols to a list of the rules where they are used.
    Nonterminals: dict[str, list[int]]
        A dictionary mapping names of nonterminals to a list of rule numbers where they are used.
    First: dict[str, list[str]]
        A dictionary of precomputed FIRST(x) symbols.
    Follow: dict[str, list[str]]
        A dictionary of precomputed FOLLOW(x) symbols.
    Precedence: dict[str, tuple[str, int]]
        Precedence rules for each terminal. Contains tuples of the form ("right", level) or ("nonassoc", level) or
        ("left", level).
    UsedPrecedence: set[str]
        Precedence rules that were actually used by the grammer. This is only used to provide error checking and to
        generate a warning about unused precedence rules.
    Start: str | None
        Starting symbol for the grammar.
    """

    def __init__(self, terminals: _t.Collection[str]) -> None:
        # Reserve the first entry in Productions for a start symbol (see set_start()).
        self.Productions: list[Production] = [None]  # pyright: ignore [reportAttributeAccessIssue]
        self.Prodnames: dict[str, list[Production]] = {}
        self.Prodmap: dict[str, Production] = {}
        self.Terminals: dict[str, list[int]] = dict({term: [] for term in terminals}, error=[])
        self.Nonterminals: dict[str, list[int]] = {}
        self.First: dict[str, set[str]] = {}
        self.Follow: dict[str, set[str]] = {}
        self.Precedence: dict[str, tuple[str, int]] = {}
        self.UsedPrecedence: set[str] = set()
        self.Start: _t.Optional[str] = None

    def __len__(self) -> int:
        return len(self.Productions)

    def __getitem__(self, index: int, /) -> Production:
        return self.Productions[index]

    def set_precedence(self, term: str, assoc: str, level: int) -> None:
        """Sets the precedence for a given terminal.

        Parameters
        ----------
        term: str
            The terminal.
        assoc: {"left", "right", "nonassoc"}
            The associativity of the terminal.
        level: int
            The associativity level of the terminal.

        Raises
        ------
        RuntimeError
            If `set_precedence` was called before `add_production`.
        GrammarError
            If the precedence has already been specified for `term`, or if `assoc` isn't a valid associativity value.
        """

        if self.Productions != [None]:
            msg = "Must call set_precedence() before add_production()."
            raise RuntimeError(msg)
        if term in self.Precedence:
            msg = f"Precedence already specified for terminal {term!r}."
            raise GrammarError(msg)
        if assoc not in {"left", "right", "nonassoc"}:
            msg = f'Associativity of {term!r} must be one of "left", "right", or "nonassoc".'
            raise GrammarError(msg)

        self.Precedence[term] = (assoc, level)

    def add_production(
        self,
        prodname: str,
        syms: list[str],
        func: _t.Callable[[Parser, YaccProduction], _t.Any],
        file: str = "",
        line: int = 0,
        *,
        name_aliases: dict[str, list[str]],
    ) -> None:
        """Given an action function, this function assembles a production rule and computes its precedence level.

        Precedence is determined by the precedence of the right-most non-terminal or the precedence of a terminal
        specified by ``%prec``.

        Parameters
        ----------
        prodname: str
            The name of the production, e.g. "expr" for the rule ``expr : expr PLUS term``.
        syms: list[str]
            The list of symbols representing the production, e.g. ["expr", "PLUS", "term"] for the rule
            ``expr : expr PLUS term``.
        func: _t.Callable[[Parser, YaccProduction], _t.Any]
            The action function.

        Raises
        ------
        GrammarError
            If a production symbol is invalid, or if ``%prec`` is used incorrectly.
        """

        if prodname in self.Terminals:
            msg = f"{file}:{line}: Illegal rule name {prodname!r}. Already defined as a token."
            raise GrammarError(msg)
        if prodname == "error":
            msg = f"{file}:{line}: Illegal rule name {prodname!r}. error is a reserved word."
            raise GrammarError(msg)

        # Look for literal tokens
        for n, s in enumerate(syms):
            if s[0] in "'\"" and s[0] == s[-1]:
                c = s[1:-1]
                if len(c) != 1:
                    msg = f"{file}:{line}: Literal token {s} in rule {prodname!r} may only be a single character."
                    raise GrammarError(msg)
                if c not in self.Terminals:
                    self.Terminals[c] = []
                syms[n] = c
                continue

        # Determine the precedence level
        if "%prec" in syms:
            if syms[-1] == "%prec":
                msg = f"{file}:{line}: Syntax error. Nothing follows %%prec"
                raise GrammarError(msg)
            if syms[-2] != "%prec":
                msg = f"{file}:{line}: Syntax error. %prec can only appear at the end of a grammar rule"
                raise GrammarError(msg)

            precname = syms[-1]
            prodprec = self.Precedence.get(precname)
            if not prodprec:
                msg = f"{file}:{line}: Nothing known about the precedence of {precname!r}"
                raise GrammarError(msg)

            self.UsedPrecedence.add(precname)
            del syms[-2:]  # Drop %prec from the rule
        else:
            # If no %prec, precedence is determined by the rightmost terminal symbol
            precname = next((sym for sym in reversed(syms) if sym in self.Terminals), None)
            prodprec = ("right", 0) if (precname is None) else self.Precedence.get(precname, ("right", 0))

        # See if the rule is already in the rulemap
        map_ = f"{prodname} -> {syms}"
        if map_ in self.Prodmap:
            m = self.Prodmap[map_]
            msg = f"{file}:{line}: Duplicate rule {m}. Previous definition at {m.file}:{m.line}"
            raise GrammarError(msg)

        # From this point on, everything is valid. Create a new Production instance
        pnumber = len(self.Productions)
        if prodname not in self.Nonterminals:
            self.Nonterminals[prodname] = []

        # Add the production number to Terminals and Nonterminals
        for t in syms:
            if t in self.Terminals:
                self.Terminals[t].append(pnumber)
            else:
                if t not in self.Nonterminals:
                    self.Nonterminals[t] = []
                self.Nonterminals[t].append(pnumber)

        # Create a production and add it to the list of productions
        p = Production(pnumber, prodname, syms, func, prodprec, file, line, name_aliases=name_aliases)
        self.Productions.append(p)
        self.Prodmap[map_] = p

        # Add to the productions list
        try:
            self.Prodnames[prodname].append(p)
        except KeyError:
            self.Prodnames[prodname] = [p]

    def set_start(
        self,
        start: _t.Optional[_t.Union[_t.Callable[..., _t.Any], str]] = None,
        *,
        name_aliases: dict[str, list[str]],
    ) -> None:
        """Sets the starting symbol and creates the augmented grammar.

        Production rule 0 is ``S' -> start`` where ``start`` is the start symbol.
        """

        if callable(start):
            start = start.__name__

        if not start:
            start = self.Productions[1].name

        if start not in self.Nonterminals:
            msg = f"Start symbol {start!r} undefined."
            raise GrammarError(msg)

        def _start_error(parser: Parser, prod: YaccProduction) -> None:
            """A hopefully sane default for catastrophe."""

            msg = "Catastrophic error. This should never be called."
            raise RuntimeError(msg, parser, prod)

        self.Productions[0] = Production(0, "S'", [start], _start_error, name_aliases=name_aliases)
        self.Nonterminals[start].append(0)
        self.Start = start

    def find_unreachable(self) -> set[str]:
        """Find all of the nonterminal symbols that can't be reached from the starting symbol.

        Returns
        -------
        set[str]
            A set of nonterminals that can't be reached.
        """

        reachable: set[str] = set()

        # Mark all symbols that are reachable from the start symbol.
        stack = deque([self.Productions[0].prod[0]])
        while stack:
            s = stack.popleft()
            if s in reachable:
                continue

            reachable.add(s)
            try:
                for p in self.Prodnames[s]:
                    stack.extend(p.prod)
            except KeyError:
                pass

        return set(self.Nonterminals) - reachable

    def infinite_cycles(self) -> list[str]:
        """This function looks at the various parsing rules and tries to detect infinite recursion cycles.

        Notes
        -----
        Infinite recursion cycles occur with grammar rules where there is no possible way to derive a string of only
        terminals.
        """

        terminates: dict[str, bool] = {}

        # Terminals: Initialize to true.
        for t in self.Terminals:
            terminates[t] = True
        terminates["$end"] = True

        # Nonterminals: Initialize to false.
        for n in self.Nonterminals:
            terminates[n] = False

        # Then propagate termination until no change:
        while True:
            some_change = False
            for n, pl in self.Prodnames.items():
                # Nonterminal n terminates iff any of its productions terminates.
                for p in pl:
                    # Production p terminates iff all of its rhs symbols terminate.
                    if all(map(terminates.__contains__, p.prod)):
                        # symbol n terminates!
                        if not terminates[n]:
                            terminates[n] = True
                            some_change = True
                        # Don't need to consider any more productions for this n.
                        break

            if not some_change:
                break

        infinite: list[str] = []
        for s, term in terminates.items():
            if not term:
                if s not in self.Prodnames and s not in self.Terminals and s != "error":
                    # s is used-but-not-defined, and we've already warned of that,
                    # so it would be overkill to say that it's also non-terminating.
                    pass
                else:
                    infinite.append(s)

        return infinite

    def undefined_symbols(self) -> list[tuple[str, Production]]:
        """Find all symbols that were used the grammar, but not defined as tokens or grammar rules.

        Returns
        -------
        result: list[tuple[str, Production]]
            A list of tuples (sym, prod) where sym in the symbol and prod is the production where the symbol was used.
        """

        return [
            (sym, prod)
            for prod in self.Productions
            if prod
            for sym in prod.prod
            if (sym not in self.Prodnames) and (sym not in self.Terminals) and sym != "error"
        ]

    def unused_terminals(self) -> list[str]:
        """Find all terminals that were defined, but not used by the grammar.

        Returns
        -------
        list[str]
            A list of all defined, unused symbols.
        """

        return [sym for sym, v in self.Terminals.items() if sym != "error" and not v]

    def unused_rules(self) -> list[Production]:
        """Find all grammar rules that were defined, but not used (maybe not reachable).

        Returns
        -------
        list[Production]
            A list of defined, unused productions.
        """

        return [self.Prodnames[sym][0] for sym, v in self.Nonterminals.items() if not v]

    def unused_precedence(self) -> list[tuple[str, str]]:
        """Returns a list of tuples corresponding to precedence rules that were never used by the grammar.

        Returns
        -------
        list[tuple[str, str]]
            A list of tuples representing unused precedence rules. The tuples are in the format (term, precedence),
            where term is the name of the terminal on which precedence was applied and precedence is a string such as
            'left' or 'right' corresponding to the type of precedence.
        """

        return [
            (term_name, assoc)
            for term_name, (assoc, _level) in self.Precedence.items()
            if not (term_name in self.Terminals or term_name in self.UsedPrecedence)
        ]

    def _first(self, beta: tuple[str, ...]) -> set[str]:
        """Compute the value of FIRST1(beta) where beta is a tuple of symbols.

        During execution of `compute_first()`, the result may be incomplete.
        Afterward (e.g., when called from `compute_follow()`), it will be complete.
        """

        _empty = {"<empty>"}

        # We are computing First(x1,x2,x3,...,xn)
        result: set[str] = set()

        for x in beta:
            x_produces_empty = "<empty>" in self.First[x]

            # Add all the non-<empty> symbols of First[x] to the result.
            result |= self.First[x] - _empty

            if x_produces_empty:
                # We have to consider the next x in beta, i.e. stay in the loop.
                pass
            else:
                # We don't have to consider any further symbols in beta.
                break
        else:
            # There was no 'break' from the loop,
            # so x_produces_empty was true for all x in beta,
            # so beta produces empty as well.
            result.add("<empty>")

        return result

    def build_lritems(self) -> None:
        """This function walks the list of productions and builds a complete set of the LR items.

        Notes
        -----
        The LR items are stored in two ways: First, they are uniquely numbered and placed in the list _lritems.
        Second, a linked list of LR items is built for each production. For example::

            E -> E PLUS E

        creates this list::

            [E -> . E PLUS E, E -> E . PLUS E, E -> E PLUS . E, E -> E PLUS E . ]
        """

        for p in self.Productions:
            lastlri = p
            lr_items: list[LRItem] = []
            for i in count():
                lastlri.lr_next = lri = p.lr_item(i, self.Prodnames)
                if not lri:
                    break
                lr_items.append(lri)
                lastlri = lri
            p.lr_items = lr_items

    def __str__(self) -> str:
        """Return str(self).

        Notes
        -----
        Serves as debugging output. Printing the grammar will produce a detailed description along with some
        diagnostics.
        """

        out: list[str] = []
        out.append("Grammar:\n")
        out.extend(f"Rule {n:5d} {p}" for n, p in enumerate(self.Productions))

        unused_terminals = self.unused_terminals()
        if unused_terminals:
            out.append("\nUnused terminals:\n")
            out.extend(f"    {term}" for term in unused_terminals)

        out.append("\nTerminals, with rules where they appear:\n")
        out.extend(f"{term} : {' '.join(map(str, self.Terminals[term]))}" for term in sorted(self.Terminals))

        out.append("\nNonterminals, with rules where they appear:\n")
        out.extend(
            f"{nonterm} : {' '.join(map(str, self.Nonterminals[nonterm]))}" for nonterm in sorted(self.Nonterminals)
        )

        out.append("")
        return "\n".join(out)


# endregion


# ============================================================================
# region -------- LR Generator --------
#
# The following classes and functions are used to generate LR parsing tables on
# a grammar.
# ============================================================================


_RelationFunction: _t.TypeAlias = "_t.Callable[[tuple[int, str]], list[tuple[int, str]]]"
_SetValuedFunction: _t.TypeAlias = "_t.Callable[[tuple[int, str]], set[str]]"


def digraph(
    X: set[tuple[int, str]],
    R: _RelationFunction,
    FP: _SetValuedFunction,
) -> dict[tuple[int, str], set[str]]:
    """First helper for computing set valued functions of the form ``F(x) = F'(x) U U{F(y) | x R y}``.

    This is used to compute the values of Read() sets as well as FOLLOW sets in LALR(1) generation.

    Parameters
    ----------
    X: list[tuple[int, str]]
        An input set of nodes.
    R: _RelationFunction
        A relation (i.e. a mapper from a node to a list of nodes that satisfy the relation).
    FP: _SetValuedFunction
        Set-valued function.

    See Also
    --------
    traverse
    """

    N = dict.fromkeys(X, 0)
    stack: list[tuple[int, str]] = []
    F: dict[tuple[int, str], set[str]] = {}
    for x in X:
        if N[x] == 0:
            traverse(x, N, stack, F, X, R, FP)
    return F


def traverse(
    x: tuple[int, str],
    N: dict[tuple[int, str], int],
    stack: list[tuple[int, str]],
    F: dict[tuple[int, str], set[str]],
    X: set[tuple[int, str]],
    R: _RelationFunction,
    FP: _SetValuedFunction,
) -> None:
    """Second helper for computing set valued functions of the form ``F(x) = F'(x) U U{F(y) | x R y}``.

    This is used to compute the values of Read() sets as well as FOLLOW sets in LALR(1) generation.

    See Also
    --------
    digraph
    """

    stack.append(x)
    N[x] = d = len(stack)
    F[x] = FP(x)  # F(X) <- F'(x)

    for y in R(x):  # Get y's related to x
        if N[y] == 0:
            traverse(y, N, stack, F, X, R, FP)
        N[x] = min(N[x], N[y])
        if y in F:
            F[x] |= F[y]

    if N[x] == d:
        N[stack[-1]] = sys.maxsize
        F[stack[-1]] = F[x]
        while stack.pop() != x:
            N[stack[-1]] = sys.maxsize
            F[stack[-1]] = F[x]


class LALRError(YaccError):
    pass


class LRTable:
    """This class implements the LR table generation algorithm. There are no public methods except for `write()`."""

    def __init__(self, grammar: Grammar) -> None:
        self.grammar = grammar

        # Internal attributes
        self.lr_action: dict[int, dict[str, int]] = {}  # Action table
        self.lr_goto: dict[int, dict[str, int]] = {}  # Goto table
        self.lr_productions = grammar.Productions  # Copy of grammar Production array
        # Cache of computed gotos
        self.lr_goto_cache: dict[
            _t.Union[tuple[int, str], str],
            _t.Union[list[LRItem], dict[_t.Union[int, str], list[LRItem]]],
        ] = {}
        self.lr0_cidhash: dict[int, int] = {}  # Cache of closures
        self._add_count: int = 0  # Internal counter used to detect cycles

        # Diagonistic information filled in by the table generator
        self.state_descriptions: dict[int, str] = {}
        self.sr_conflicts: list[tuple[int, str, str]] = []  # List of shift-reduce conflicts
        self.rr_conflicts: list[tuple[int, Production, Production]] = []  # List of reduce-reduce conflicts

        # Build the tables
        self.grammar.build_lritems()

        self.lr_parse_table()

        # Build default states
        # This identifies parser states where there is only one possible reduction action.
        # For such states, the parser can make a choose to make a rule reduction without consuming
        # the next look-ahead token. This delayed invocation of the tokenizer can be useful in
        # certain kinds of advanced parsing situations where the lexer and parser interact with
        # each other or change states (i.e., manipulation of scope, lexer states, etc.).
        #
        # See:  http://www.gnu.org/software/bison/manual/html_node/Default-Reductions.html#Default-Reductions
        self.defaulted_states: dict[int, int] = {}
        for state, actions in self.lr_action.items():
            rules = list(actions.values())
            if len(rules) == 1 and rules[0] < 0:
                self.defaulted_states[state] = rules[0]

    def lr0_closure(self, I: list[LRItem]) -> list[LRItem]:
        """Compute the LR(0) closure operation on a set of LR(0) items.

        Parameters
        ----------
        I: list[LRItem]
            A set of LR(0) items.
        """

        self._add_count += 1

        # Add everything in I to J
        J: list[LRItem] = I.copy()
        didadd = True
        while didadd:
            didadd = False
            for j in J:
                for x in j.lr_after:
                    if x.lr0_added == self._add_count:
                        continue
                    # Add B --> .G to J
                    assert x.lr_next is not None
                    J.append(x.lr_next)
                    x.lr0_added = self._add_count
                    didadd = True

        return J

    def lr0_goto(self, I: list[LRItem], x: str) -> _t.Optional[list[LRItem]]:
        """Compute the LR(0) goto function goto(I,X).

        Parameters
        ----------
        I: list[LRItem]
            A set of LR(0) items.
        x: str
            A grammar symbol.

        Notes
        -----
        This function is written in a way that guarantees uniqueness of the generated goto sets (i.e. the same
        goto set will never be returned as two different Python objects). With uniqueness, we can later do fast
        set comparisons using id(obj) instead of element-wise comparison.
        """

        # TODO: Understand this function. The types aren't necessarily correct.

        # First we look for a previously cached entry
        try:
            g = self.lr_goto_cache[(id(I), x)]
        except KeyError:
            pass
        else:
            assert isinstance(g, list) or (g is None)
            return g

        # Now we generate the goto set in a way that guarantees uniqueness of the result
        s = self.lr_goto_cache.setdefault(x, {})
        assert isinstance(s, dict)

        gs: list[LRItem] = []
        for p in I:
            n = p.lr_next
            if n and n.lr_before == x:
                s = s.setdefault(id(n), {})  # pyright: ignore
                gs.append(n)
        assert isinstance(s, dict)

        g = s.get("$end")
        if not g:
            if gs:
                s["$end"] = g = self.lr0_closure(gs)
            else:
                s["$end"] = gs
        self.lr_goto_cache[(id(I), x)] = g  # pyright: ignore
        return g

    def lr0_items(self) -> list[list[LRItem]]:
        """Compute the LR(0) sets of item function."""

        assert self.grammar.Productions[0].lr_next is not None

        C = [self.lr0_closure([self.grammar.Productions[0].lr_next])]

        self.lr0_cidhash |= {id(I): i for i, I in enumerate(C)}

        # Loop over the items in C and each grammar symbols
        for I in C:
            # Collect all of the symbols that could possibly be in the goto(I,X) sets
            asyms: set[str] = set().union(*[ii.usyms for ii in I])

            for x in asyms:
                g = self.lr0_goto(I, x)
                if not g or (id(g) in self.lr0_cidhash):
                    continue
                self.lr0_cidhash[id(g)] = len(C)
                C.append(g)

        return C

    # -----------------------------------------------------------------------------
    #                       ==== LALR(1) Parsing ====
    #
    # LALR(1) parsing is almost exactly the same as SLR except that instead of
    # relying upon Follow() sets when performing reductions, a more selective
    # lookahead set that incorporates the state of the LR(0) machine is utilized.
    # Thus, we mainly just have to focus on calculating the lookahead sets.
    #
    # The method used here is due to DeRemer and Pennelo (1982).
    #
    # DeRemer, F. L., and T. J. Pennelo: "Efficient Computation of LALR(1)
    #     Lookahead Sets", ACM Transactions on Programming Languages and Systems,
    #     Vol. 4, No. 4, Oct. 1982, pp. 615-649
    #
    # Further details can also be found in:
    #
    #  J. Tremblay and P. Sorenson, "The Theory and Practice of Compiler Writing",
    #      McGraw-Hill Book Company, (1985).
    #
    # -----------------------------------------------------------------------------

    def compute_nullable_nonterminals(self) -> set[str]:
        """Creates a set containing all of the non-terminals that might produce an empty production."""

        nullable: set[str] = set()
        num_nullable = 0
        while True:
            for p in self.grammar.Productions[1:]:
                if p.len == 0 or nullable.issuperset(p.prod):
                    nullable.add(p.name)
            if len(nullable) == num_nullable:
                break
            num_nullable = len(nullable)
        return nullable

    def find_nonterminal_transitions(self, C: list[list[LRItem]]) -> set[tuple[int, str]]:
        """Given a set of LR(0) items, this functions finds all of the non-terminal transitions.

        Non-terminal transitions are transitions in which a dot appears immediately before a non-terminal.

        Parameters
        ----------
        C: list[list[LRItem]]
            The set of LR(0) items.

        Returns
        -------
        set[tuple[int, str]]
            The set of nonterminal transitions, which are tuples of the form (state,N) where state is the state number
            and N is the nonterminal symbol.
        """

        return {
            tran
            for stateno, state in enumerate(C)
            for p in state
            if p.lr_index < (p.len - 1)
            and (tran := (stateno, p.prod[p.lr_index + 1]))
            and tran[1] in self.grammar.Nonterminals
        }

    def dr_relation(self, C: list[list[LRItem]], trans: tuple[int, str], nullable: set[str]) -> set[str]:
        """Computes the DR(p,A) relationships for non-terminal transitions.

        Parameters
        ----------
        C: list[list[LRItem]]
            Set of LR(0) items.
        trans: tuple[int, str]
            A tuple (state,N) where state is a number and N is a nonterminal symbol.
        nullable: set[str]
            Set of empty transitions.

        Returns
        -------
        terms: set[str]
            A set of terminals.
        """

        state, N = trans

        g = self.lr0_goto(C[state], N)
        assert g is not None
        terms = {a for p in g if p.lr_index < (p.len - 1) and (a := p.prod[p.lr_index + 1]) in self.grammar.Terminals}

        # This extra bit is to handle the start state
        if state == 0 and self.grammar.Productions[0].prod[0] == N:
            terms.add("$end")

        return terms

    def reads_relation(self, C: list[list[LRItem]], trans: tuple[int, str], empty: set[str]) -> list[tuple[int, str]]:
        """Computes the READS() relation (p,A) READS (t,C)."""

        # Look for empty transitions
        rel: list[tuple[int, str]] = []
        state, N = trans

        g = self.lr0_goto(C[state], N)
        assert g is not None
        j = self.lr0_cidhash.get(id(g), -1)
        for p in g:
            if p.lr_index < p.len - 1:
                a = p.prod[p.lr_index + 1]
                if a in empty:
                    rel.append((j, a))

        return rel

    def compute_lookback_includes(
        self,
        C: list[list[LRItem]],
        trans: set[tuple[int, str]],
        nullable: set[str],
    ) -> tuple[
        dict[tuple[int, str], list[tuple[int, LRItem]]],
        dict[tuple[int, str], list[tuple[int, str]]],
    ]:
        """Determines the lookback and includes relations.

        Notes
        -----
        LOOKBACK:

        This relation is determined by running the LR(0) state machine forward. For example, starting with a production
        ``N : . A B C``, we run it forward to obtain ``N : A B C .``. We then build a relationship between this final
        state and the starting state. These relationships are stored in a dictionary `lookdict`.

        INCLUDES:

        Computes the INCLUDE() relation ``(p,A) INCLUDES (p',B)``.

        This relation is used to determine non-terminal transitions that occur inside of other non-terminal transition
        states. ``(p,A) INCLUDES (p', B)`` if the following holds::

            B -> LAT, where T -> epsilon and p' -L-> p

        L is essentially a prefix (which may be empty), T is a suffix that must be able to derive an empty string.
        State p' must lead to state p with the string L.
        """

        # Dictionary of lookback relations
        lookdict: dict[tuple[int, str], list[tuple[int, LRItem]]] = {}
        # Dictionary of include relations
        includedict: defaultdict[tuple[int, str], list[tuple[int, str]]] = defaultdict(list)

        # Make a dictionary of non-terminal transitions
        dtrans = dict.fromkeys(trans, 1)

        # Loop over all transitions and compute lookbacks and includes
        for state, N in trans:
            lookb: list[tuple[int, LRItem]] = []
            includes: list[tuple[int, str]] = []
            for p in C[state]:
                if p.name != N:
                    continue

                # Okay, we have a name match. We now follow the production all the way
                # through the state machine until we get the . on the right hand side
                j = state
                for lr_index in range(p.lr_index + 1, p.len):
                    t = p.prod[lr_index]

                    # Check to see if this symbol and state are a non-terminal transition
                    if (j, t) in dtrans:
                        # Yes. Okay, there is some chance that this is an includes relation
                        # the only way to know for certain is whether the rest of the
                        # production derives empty

                        for li in range(lr_index + 1, p.len):
                            if p.prod[li] in self.grammar.Terminals:
                                break  # No forget it
                            if p.prod[li] not in nullable:
                                break
                        else:
                            # Appears to be a relation between (j,t) and (state,N)
                            includes.append((j, t))

                    g = self.lr0_goto(C[j], t)  # Go to next set
                    j = self.lr0_cidhash.get(id(g), -1)  # Go to next state

                # When we get here, j is the final state, now we have to locate the production
                lookb += [
                    (j, r)
                    for r in C[j]
                    if (r.name == p.name)
                    and (r.len == p.len)
                    # This look is comparing a production ". A B C" with "A B C ."
                    and (r.prod[: r.lr_index] == p.prod[1 : r.lr_index + 1])
                ]

            for i in includes:
                includedict[i].append((state, N))
            lookdict[(state, N)] = lookb

        includedict.default_factory = None

        return lookdict, includedict

    def compute_read_sets(
        self,
        C: list[list[LRItem]],
        ntrans: set[tuple[int, str]],
        nullable: set[str],
    ) -> dict[tuple[int, str], set[str]]:
        """Given a set of LR(0) items, this function computes the read sets.

        Parameters
        ----------
        C: list[list[LRItem]]
            Set of LR(0) items.
        ntrans: set[tuple[int, str]]
            Set of nonterminal transitions.
        nullable: set[str]
            Set of empty transitions.

        Returns
        -------
        F: dict[tuple[int, str], list[str]]
            A set containing the read sets.
        """

        def FP(x: tuple[int, str]) -> set[str]:
            return self.dr_relation(C, x, nullable)

        def R(x: tuple[int, str]) -> list[tuple[int, str]]:
            return self.reads_relation(C, x, nullable)

        F = digraph(ntrans, R, FP)
        return F  # noqa: RET504

    def compute_follow_sets(
        self,
        ntrans: set[tuple[int, str]],
        readsets: dict[tuple[int, str], set[str]],
        inclsets: dict[tuple[int, str], list[tuple[int, str]]],
    ) -> dict[tuple[int, str], set[str]]:
        """Given a set of LR(0) items, a set of non-terminal transitions, a readset, and an include set, this function
        computes the follow sets: ``Follow(p,A) = Read(p,A) U U {Follow(p',B) | (p,A) INCLUDES (p',B)}``.

        Parameters
        ----------
        ntrans: set[tuple[int, str]]
            Set of nonterminal transitions.
        readsets: dict[tuple[int, str], set[str]]
            Readset (previously computed).
        inclsets: dict[tuple[int, str], list[tuple[int, str]]]
            Include sets (previously computed).

        Returns
        -------
        F: dict[tuple[int, str], list[str]]
            A set containing the follow sets.
        """

        FP = readsets.__getitem__

        def R(x: tuple[int, str]) -> list[tuple[int, str]]:
            return inclsets.get(x, [])

        F = digraph(ntrans, R, FP)
        return F  # noqa: RET504

    def add_lookaheads(
        self,
        lookbacks: dict[tuple[int, str], list[tuple[int, LRItem]]],
        followset: dict[tuple[int, str], set[str]],
    ) -> None:
        """Attaches the lookahead symbols to grammar rules.

        Extended Summary
        ----------------
        This function directly attaches the lookaheads to productions contained in the lookbacks set.

        Parameters
        ----------
        lookbacks: dict[tuple[int, str], list[tuple[int, LRItem]]]
            Set of lookback relations.
        followset: dict
            Computed follow set.
        """

        for trans, lb in lookbacks.items():
            # Loop over productions in lookback
            for state, p in lb:
                try:
                    la = p.lookaheads[state]
                except KeyError:
                    la = p.lookaheads[state] = set()

                try:
                    fs = followset[trans]
                except KeyError:
                    pass
                else:
                    la |= fs

    def add_lalr_lookaheads(self, C: list[list[LRItem]]) -> None:
        """This function does all of the work of adding lookahead information for use with LALR parsing."""

        # Determine all of the nullable nonterminals
        nullable = self.compute_nullable_nonterminals()

        # Find all non-terminal transitions
        trans = self.find_nonterminal_transitions(C)

        # Compute read sets
        readsets = self.compute_read_sets(C, trans, nullable)

        # Compute lookback/includes relations
        lookd, included = self.compute_lookback_includes(C, trans, nullable)

        # Compute LALR FOLLOW sets
        followsets = self.compute_follow_sets(trans, readsets, included)

        # Add all of the lookaheads
        self.add_lookaheads(lookd, followsets)

    def lr_parse_table(self) -> None:
        """This function constructs the final LALR parse table. Touch this code and die."""

        Productions = self.grammar.Productions
        Precedence = self.grammar.Precedence
        goto = self.lr_goto  # Goto array
        action = self.lr_action  # Action array

        # Step 1: Construct C = { I0, I1, ... IN }, collection of LR(0) items
        # This determines the number of states

        C = self.lr0_items()
        self.add_lalr_lookaheads(C)

        # Build the parser table, state by state
        for st, I in enumerate(C):
            descrip: list[str] = []
            # Loop over each production in I
            actlist: list[tuple[str, LRItem, str]] = []  # List of actions
            st_action: dict[str, _t.Optional[int]] = {}
            st_actionp: dict[str, LRItem] = {}  # Action production array (temporary)
            st_goto: dict[str, int] = {}

            descrip.append(f"\nstate {st}\n")
            descrip.extend([f"    ({p.number}) {p}" for p in I])

            for p in I:
                if p.len == p.lr_index + 1:
                    if p.name == "S'":
                        # Start symbol. Accept!
                        st_action["$end"] = 0
                        st_actionp["$end"] = p
                    else:
                        # We are at the end of a production. Reduce!
                        laheads = p.lookaheads[st]
                        for a in laheads:
                            actlist.append((a, p, f"reduce using rule {p.number} ({p})"))
                            r = st_action.get(a)
                            if r is not None:
                                # Have a shift/reduce or reduce/reduce conflict
                                if r > 0:
                                    # Need to decide on shift or reduce here
                                    # By default we favor shifting. Need to add
                                    # some precedence rules here.

                                    # Shift precedence comes from the token
                                    _sprec, slevel = Precedence.get(a, ("right", 0))

                                    # Reduce precedence comes from rule being reduced (p)
                                    rprec, rlevel = Productions[p.number].prec

                                    if (slevel < rlevel) or ((slevel == rlevel) and (rprec == "left")):
                                        # We really need to reduce here.
                                        st_action[a] = -p.number
                                        st_actionp[a] = p
                                        if not slevel and not rlevel:
                                            descrip.append(f"  ! shift/reduce conflict for {a} resolved as reduce")
                                            self.sr_conflicts.append((st, a, "reduce"))
                                        Productions[p.number].reduced += 1
                                    elif (slevel == rlevel) and (rprec == "nonassoc"):
                                        st_action[a] = None
                                    else:
                                        # Hmmm. Guess we'll keep the shift
                                        if not rlevel:
                                            descrip.append(f"  ! shift/reduce conflict for {a} resolved as shift")
                                            self.sr_conflicts.append((st, a, "shift"))
                                elif r <= 0:
                                    # Reduce/reduce conflict.   In this case, we favor the rule
                                    # that was defined first in the grammar file
                                    oldp = Productions[-r]
                                    pp = Productions[p.number]
                                    if oldp.line > pp.line:
                                        st_action[a] = -p.number
                                        st_actionp[a] = p
                                        chosenp, rejectp = pp, oldp
                                        Productions[p.number].reduced += 1
                                        Productions[oldp.number].reduced -= 1
                                    else:
                                        chosenp, rejectp = oldp, pp
                                    self.rr_conflicts.append((st, chosenp, rejectp))
                                    descrip.append(
                                        f"  ! reduce/reduce conflict for {a} resolved using "
                                        f"rule {st_actionp[a].number} ({st_actionp[a]})"
                                    )
                                else:
                                    msg = f"Unknown conflict in state {st}."
                                    raise LALRError(msg)
                            else:
                                st_action[a] = -p.number
                                st_actionp[a] = p
                                Productions[p.number].reduced += 1
                else:
                    i = p.lr_index
                    a = p.prod[i + 1]  # Get symbol right after the "."
                    if a in self.grammar.Terminals:
                        g = self.lr0_goto(I, a)
                        j = self.lr0_cidhash.get(id(g), -1)
                        if j >= 0:
                            # We are in a shift state
                            actlist.append((a, p, f"shift and go to state {j}"))
                            r = st_action.get(a)
                            if r is not None:
                                # Whoa have a shift/reduce or shift/shift conflict
                                if r > 0:
                                    if r != j:
                                        msg = f"Shift/shift conflict in state {st}."
                                        raise LALRError(msg)
                                elif r <= 0:
                                    # Do a precedence check.
                                    #   -  if precedence of reduce rule is higher, we reduce.
                                    #   -  if precedence of reduce is same and left assoc, we reduce.
                                    #   -  otherwise we shift
                                    rprec, rlevel = Productions[st_actionp[a].number].prec
                                    _sprec, slevel = Precedence.get(a, ("right", 0))
                                    if (slevel > rlevel) or ((slevel == rlevel) and (rprec == "right")):
                                        # We decide to shift here... highest precedence to shift
                                        Productions[st_actionp[a].number].reduced -= 1
                                        st_action[a] = j
                                        st_actionp[a] = p
                                        if not rlevel:
                                            descrip.append(f"  ! shift/reduce conflict for {a} resolved as shift")
                                            self.sr_conflicts.append((st, a, "shift"))
                                    elif (slevel == rlevel) and (rprec == "nonassoc"):
                                        st_action[a] = None
                                    else:
                                        # Hmmm. Guess we'll keep the reduce
                                        if not slevel and not rlevel:
                                            descrip.append(f"  ! shift/reduce conflict for {a} resolved as reduce")
                                            self.sr_conflicts.append((st, a, "reduce"))

                                else:
                                    msg = f"Unknown conflict in state {st}."
                                    raise LALRError(msg)
                            else:
                                st_action[a] = j
                                st_actionp[a] = p

            # Print the actions associated with each terminal
            _actprint: set[tuple[str, str]] = set()
            for a, p, m in actlist:
                if (a in st_action) and (p is st_actionp[a]):
                    descrip.append(f"    {a:<15s} {m}")
                    _actprint.add((a, m))
            descrip.append("")

            # Print the actions that were not used. (debugging)
            used = False
            for a, p, m in actlist:
                if (a in st_action) and (p is not st_actionp[a]) and ((a, m) not in _actprint):
                    descrip.append(f"  ! {a:<15s} {m}")
                    _actprint.add((a, m))
                    used = True
            if not used:
                descrip.append("")

            # Construct the goto table for this state
            nkeys = {s: None for ii in I for s in ii.usyms if s in self.grammar.Nonterminals}

            for n in nkeys:
                g = self.lr0_goto(I, n)
                j = self.lr0_cidhash.get(id(g), -1)
                if j >= 0:
                    st_goto[n] = j
                    descrip.append(f"    {n:<30s} shift and go to state {j}")

            action[st] = st_action
            goto[st] = st_goto
            self.state_descriptions[st] = "\n".join(descrip)

    def __str__(self) -> str:
        """Return str(self).

        Notes
        -----
        Serves as debugging output. Printing the LRTable object will produce a listing of all of the states, conflicts,
        and other details.
        """

        out = list(self.state_descriptions.values())

        if self.sr_conflicts or self.rr_conflicts:
            out.append("\nConflicts:\n")
            out.extend(
                f"shift/reduce conflict for {tok} in state {state} resolved as {resolution}"
                for state, tok, resolution in self.sr_conflicts
            )

            already_reported: set[tuple[int, int, int]] = set()
            for state, rule, rejected in self.rr_conflicts:
                if (state, id(rule), id(rejected)) in already_reported:
                    continue
                out.append(f"reduce/reduce conflict in state {state} resolved using rule {rule}")
                out.append(f"rejected rule ({rejected}) in state {state}")
                already_reported.add((state, id(rule), id(rejected)))

            warned_never: set[Production] = set()
            for _, _, rejected in self.rr_conflicts:
                if not rejected.reduced and (rejected not in warned_never):
                    out.append(f"Rule ({rejected}) is never reduced")
                    warned_never.add(rejected)

        return "\n".join(out)


_RawGrammarRule: _t.TypeAlias = "tuple[_t.Callable[..., _t.Any], str, int, str, list[str]]"


class NameAliasesState:
    """State related to name aliases for repeated items in an EBNF grammar.

    Attributes
    ----------
    gen_count: int
        Generation of repetition.
    aliases: dict[str, list[str]]
        Dictionary mapping of name aliases generated by EBNF rules.
    """

    __slots__ = ("gen_count", "aliases")

    def __init__(self):
        self.gen_count: int = 0
        self.aliases: dict[str, list[str]] = {}


def _collect_grammar_rules(na_state: NameAliasesState, func: _t.Callable[..., _t.Any]) -> list[_RawGrammarRule]:
    """Collect grammar rules from a function (or class docstring)."""

    grammar: list[_RawGrammarRule] = []
    curr_func: _t.Optional[_t.Callable[..., _t.Any]] = func
    while curr_func:
        prodname = curr_func.__name__
        unwrapped = _inspect_unwrap(curr_func)
        filename: str = unwrapped.__code__.co_filename
        lineno_start: int = unwrapped.__code__.co_firstlineno

        # Pre-condition: .rules exists.
        func_rules: list[str] = curr_func.rules  # pyright: ignore [reportFunctionMemberAccess]

        for rule, lineno in zip(func_rules, range(lineno_start + len(func_rules) - 1, 0, -1)):
            syms = rule.split()
            ebnf_prod: list[_RawGrammarRule] = []

            # FIXME: This while condition will infinite loop if '"|"' is used as a literal.
            # It probably has other side effects as well.
            while ("{" in syms) or ("[" in syms) or any("|" in s for s in syms):
                for s in syms:
                    if s == "[":
                        syms, prod = _replace_ebnf_optional(na_state, syms)
                        ebnf_prod.extend(prod)
                        break
                    if s == "{":
                        syms, prod = _replace_ebnf_repeat(na_state, syms)
                        ebnf_prod.extend(prod)
                        break
                    if "|" in s:
                        syms, prod = _replace_ebnf_choice(na_state, syms)
                        ebnf_prod.extend(prod)
                        break

            if len(syms) >= 2 and syms[1] in {":", "::="}:
                grammar.append((curr_func, filename, lineno, syms[0], syms[2:]))
            else:
                grammar.append((curr_func, filename, lineno, prodname, syms))
            grammar.extend(ebnf_prod)

        curr_func = getattr(curr_func, "next_func", None)

    return grammar


def _replace_ebnf_repeat(na_state: NameAliasesState, syms: list[str]) -> tuple[list[str], list[_RawGrammarRule]]:
    """Replace EBNF repetition."""

    syms = list(syms)
    first = syms.index("{")
    end = syms.index("}", first)

    # Look for choices inside
    repeated_syms = syms[first + 1 : end]
    if any("|" in sym for sym in repeated_syms):
        repeated_syms, prods = _replace_ebnf_choice(na_state, repeated_syms)
    else:
        prods = []

    symname, moreprods = _generate_repeat_rules(na_state, repeated_syms)
    syms[first : end + 1] = [symname]
    return syms, prods + moreprods


def _replace_ebnf_optional(na_state: NameAliasesState, syms: list[str]) -> tuple[list[str], list[_RawGrammarRule]]:
    syms = list(syms)
    first = syms.index("[")
    end = syms.index("]", first)

    # Look for choices inside
    repeated_syms = syms[first + 1 : end]
    if any("|" in sym for sym in repeated_syms):
        repeated_syms, prods = _replace_ebnf_choice(na_state, repeated_syms)
    else:
        prods = []

    symname, moreprods = _generate_optional_rules(na_state, repeated_syms)
    syms[first : end + 1] = [symname]
    return syms, prods + moreprods


def _replace_ebnf_choice(na_state: NameAliasesState, syms: list[str]) -> tuple[list[str], list[_RawGrammarRule]]:
    syms = list(syms)
    newprods: list[_RawGrammarRule] = []
    for n, sym in enumerate(syms):
        if "|" in sym:
            symname, prods = _generate_choice_rules(na_state, sym.split("|"))
            syms[n] = symname
            newprods.extend(prods)

    return syms, newprods


def _sanitize_symbols(symbols: list[str]) -> _t.Generator[str]:
    for sym in symbols:
        if sym.startswith("'"):
            yield hex(ord(sym[1]))
        elif sym.isidentifier():
            yield sym
        else:
            yield sym.encode("utf-8").hex()


def _create_basename(na_state: NameAliasesState, symbols: list[str]) -> str:
    na_state.gen_count += 1
    return f"_{na_state.gen_count}_" + "_".join(_sanitize_symbols(symbols))


def _generate_repeat_rules(na_state: NameAliasesState, symbols: list[str]) -> tuple[str, list[_RawGrammarRule]]:
    """Based on a given list of grammar symbols, generate code corresponding to these grammar construction:

    .. code-block:: python

        @('repeat : many')
        def repeat(self, p):
            return p.many

        @('repeat :')
        def repeat(self, p):
            return []

        @('many : many symbols')
        def many(self, p):
            p.many.append(symbols)
            return p.many

        @('many : symbols')
        def many(self, p):
            return [ p.symbols ]
    """

    basename = _create_basename(na_state, symbols)

    name = f"{basename}_repeat"
    oname = f"{basename}_items"
    iname = f"{basename}_item"
    symtext = " ".join(symbols)

    na_state.aliases[name] = symbols

    productions: list[_RawGrammarRule] = []
    _ = _rules_decorator

    @_(f"{name} : {oname}")
    def repeat(self: Parser, p: _t.Any) -> _t.Any:
        return getattr(p, oname)

    @_(f"{name} : ")
    def repeat2(self: Parser, p: _t.Any) -> _t.Any:
        return []

    productions.extend(_collect_grammar_rules(na_state, repeat))
    productions.extend(_collect_grammar_rules(na_state, repeat2))

    @_(f"{oname} : {oname} {iname}")
    def many(self: Parser, p: _t.Any) -> _t.Any:
        items = getattr(p, oname)
        items.append(getattr(p, iname))
        return items

    @_(f"{oname} : {iname}")
    def many2(self: Parser, p: _t.Any) -> _t.Any:
        return [getattr(p, iname)]

    productions.extend(_collect_grammar_rules(na_state, many))
    productions.extend(_collect_grammar_rules(na_state, many2))

    @_(f"{iname} : {symtext}")
    def item(self: Parser, p: _t.Any) -> _t.Any:
        return tuple(p)

    productions.extend(_collect_grammar_rules(na_state, item))
    return name, productions


def _generate_optional_rules(na_state: NameAliasesState, symbols: list[str]) -> tuple[str, list[_RawGrammarRule]]:
    """Based on a given list of grammar symbols [ symbols ], generate code corresponding to these grammar
    construction:

    .. code-block:: python

        @('optional : symbols')
        def optional(self, p):
            return p.symbols

        @('optional :')
        def optional(self, p):
            return None
    """

    basename = _create_basename(na_state, symbols)

    name = f"{basename}_optional"
    symtext = " ".join(symbols)

    na_state.aliases[name] = symbols

    productions: list[_RawGrammarRule] = []
    _ = _rules_decorator

    no_values = (None,) * len(symbols)

    @_(f"{name} : {symtext}")
    def optional(self: Parser, p: _t.Any) -> _t.Any:
        return tuple(p)

    @_(f"{name} : ")
    def optional2(self: Parser, p: _t.Any) -> _t.Any:
        return no_values

    productions.extend(_collect_grammar_rules(na_state, optional))
    productions.extend(_collect_grammar_rules(na_state, optional2))
    return name, productions


def _generate_choice_rules(na_state: NameAliasesState, symbols: list[str]) -> tuple[str, list[_RawGrammarRule]]:
    """Based on a given list of grammar symbols such as [ 'PLUS', 'MINUS' ], generate code corresponding to the
    following construction:

    .. code-block:: python

        @('PLUS', 'MINUS')
        def choice(self, p):
            return p[0]
    """

    basename = _create_basename(na_state, symbols)

    name = f"{basename}_choice"

    _ = _rules_decorator
    productions: list[_RawGrammarRule] = []

    @_(*symbols)
    def choice(self: Parser, p: _t.Any) -> _t.Any:
        return p[0]

    choice.__name__ = name
    productions.extend(_collect_grammar_rules(na_state, choice))
    return name, productions


# endregion


# ============================================================================
# region -------- Parser --------
# ============================================================================


class ParserMetaDict(dict[str, object]):
    """Special dictionary that allows decorated grammar rule functions to be overloaded."""

    __slots__ = ()

    def __setitem__(self, key: str, value: _t.Any, /) -> None:
        if (key in self) and callable(value) and hasattr(value, "rules"):
            value.next_func = next_func = self[key]  # pyright: ignore [reportFunctionMemberAccess]
            if not hasattr(next_func, "rules"):
                msg = f"Redefinition of {key}. Perhaps an earlier {key} is missing `@_`."
                raise GrammarError(msg)

        return super().__setitem__(key, value)

    def __missing__(self, key: str, /) -> str:
        if key.isupper() and key[:1] != "_":
            return key.upper()
        else:
            raise KeyError(key)


def _rules_decorator(rule: str, *extra: str) -> _t.Callable[[_t.CallableT], _t.CallableT]:
    rules = [rule, *extra]

    def decorate(func: _t.CallableT) -> _t.CallableT:
        func.rules = [*getattr(func, "rules", []), *rules[::-1]]  # pyright: ignore [reportFunctionMemberAccess]
        return func

    return decorate


class ParserMeta(type):
    """Metaclass for collecting parsing rules."""

    @classmethod
    def __prepare__(cls, name: str, bases: tuple[type, ...], /, **kwargs: _t.Any) -> ParserMetaDict:
        namespace = ParserMetaDict()
        namespace["_"] = _rules_decorator
        return namespace

    def __new__(cls, name: str, bases: tuple[type, ...], namespace: ParserMetaDict, /, **kwargs: _t.Any):
        del namespace["_"]
        return super().__new__(cls, name, bases, namespace, **kwargs)


_ConcreteSeqOfStr: _t.TypeAlias = "_t.Union[list[str], tuple[str, ...]]"
_NestedConcreteSeqOfStr: _t.TypeAlias = "_t.Union[list[_ConcreteSeqOfStr], tuple[_ConcreteSeqOfStr, ...]]"


class Parser(metaclass=ParserMeta):
    """The class used for recognizing language syntax specified as a context free grammar.

    Attributes
    ----------
    token_stream: _t.Iterator[Token]
        Input tokens.
    lookahead: _t.Optional[_t.Union[Token, YaccSymbol]]
        Current lookahead symbol. Be careful with this.
    """

    __slots__ = (
        "token_stream",
        "lookahead",
        "errorok",
        "state",
        "statestack",
        "symstack",
        "_line_positions",
        "_index_positions",
        "production",
    )

    # ---- Public class attributes.
    tokens: _t.ClassVar[set[str]]
    """Lexing tokens. Must be defined in a subclass."""

    precedence: _t.ClassVar[_NestedConcreteSeqOfStr]
    """Precedence definition as a tuple/list containing tuples/lists of strings. Optional."""

    log: _t.ClassVar[_t.LoggerLike] = SlyLogger(sys.stderr)
    """Logging object where debugging/diagnostic messages are sent."""

    debugfile: _t.ClassVar[_t.Optional[str]] = None
    """Debugging filename where parsetab.out data can be written."""

    track_positions: _t.ClassVar[bool] = True
    """Whether position information is automatically tracked."""

    error_count: _t.ClassVar[int] = 3
    """Yacc config knob: The number of symbols that must be shifted to leave recovery mode."""

    expected_shift_reduce: _t.ClassVar[int] = 0
    """The exact number of shift-reduce conflicts to not report."""

    expected_reduce_reduce: _t.ClassVar[int] = 0
    """The exact number of reduce-reduce conflicts to not report."""

    def __init__(self) -> None:
        # ---- Public interface
        self.token_stream: _t.Iterator[Token] = MISSING
        self.lookahead: _t.Optional[_t.Union[Token, YaccSymbol]] = None

        # ---- Internal state
        # Error status
        self.errorok: bool = True
        # Current state
        self.state: int = 0
        # Stack of parsing states
        self.statestack: list[int] = [0]
        # Stack of grammar symbols
        self.symstack: list[YaccSymbol] = [YaccSymbol("$end")]
        # Position tracker: id -> lineno
        self._line_positions: dict[int, _t.Optional[int]] = {}
        # Position tracker: id -> (start, end)
        self._index_positions: dict[int, tuple[_t.Optional[int], _t.Optional[int]]] = {}
        # Current production
        self.production: Production = MISSING

    def __init_subclass__(cls, /, **kwargs: _t.Any) -> None:
        """Collect the parser rules, build the grammar, and build the tables."""

        super().__init_subclass__(**kwargs)
        cls._build(vars(cls).copy())

    @classmethod
    def __validate_tokens(cls) -> _t.Optional[str]:
        """Validate the tokens attribute and if that fails, return a string description of why."""

        if not hasattr(cls, "tokens"):
            return "No token list is defined"

        if not cls.tokens:
            return "tokens is empty"

        if "error" in cls.tokens:
            return "Illegal token name 'error'. Is a reserved word"

        return None

    @classmethod
    def __validate_precedence(cls) -> _t.Optional[str]:
        """Validate the precedence attribute and if that fails, return a string description of why."""

        if not hasattr(cls, "precedence"):
            cls.__preclist = []
            return None

        preclist: list[tuple[str, str, int]] = []
        if not isinstance(cls.precedence, (list, tuple)):
            return "precedence must be a list or tuple"

        for level, p in enumerate(cls.precedence, start=1):
            if not isinstance(p, (list, tuple)):
                return f"Bad precedence table entry {p!r}. Must be a list or tuple"

            if len(p) < 2:
                return f"Malformed precedence entry {p!r}. Must be (assoc, term, ..., term)"

            if not all(isinstance(term, str) for term in p):
                return "precedence items must be strings"

            assoc = p[0]
            preclist.extend((term, assoc, level) for term in p[1:])

        cls.__preclist = preclist
        return None

    @classmethod
    def __validate_specification(cls) -> _t.Optional[str]:
        """Validate various parts of the grammar specification."""

        return cls.__validate_tokens() or cls.__validate_precedence()

    @classmethod
    def __build_grammar(cls, rules: list[tuple[str, _t.Callable[..., _t.Any]]]) -> None:
        """Build the grammar from the grammar rules."""

        errors: list[str] = []
        # Check for non-empty symbols
        if not rules:
            msg = "No grammar rules are defined."
            raise YaccError(msg)

        grammar = Grammar(cls.tokens)

        # Set the precedence level for terminals
        for term, assoc, level in cls.__preclist:
            try:
                grammar.set_precedence(term, assoc, level)
            except GrammarError as e:  # noqa: PERF203
                errors.append(str(e))

        na_state = NameAliasesState()
        for _name, func in rules:
            parsed_rule = _collect_grammar_rules(na_state, func)
            for pfunc, rulefile, ruleline, prodname, syms in parsed_rule:
                try:
                    grammar.add_production(prodname, syms, pfunc, rulefile, ruleline, name_aliases=na_state.aliases)
                except GrammarError as e:  # noqa: PERF203
                    errors.append(str(e))

        # The checks following this assume there are 1 or more valid productions.
        if len(grammar.Productions) == 1:
            msg = "\n".join(["Unable to build grammar - no grammar rules were valid.", *errors])
            raise YaccError(msg)

        try:
            grammar.set_start(getattr(cls, "start", None), name_aliases=na_state.aliases)
        except GrammarError as e:
            errors.append(str(e))

        undefined_symbols = grammar.undefined_symbols()
        for sym, prod in undefined_symbols:
            errors.append(f"{prod.file}:{prod.line}: Symbol {sym!r} used, but not defined as a token or a rule")

        unused_terminals = grammar.unused_terminals()
        if unused_terminals:
            unused_str = "{" + ",".join(unused_terminals) + "}"
            cls.log.warning("Token%s %s defined, but not used", "(s)" * (len(unused_terminals) > 1), unused_str)

        unused_rules = grammar.unused_rules()
        for prod in unused_rules:
            cls.log.warning("%s:%d: Rule %r defined, but not used", prod.file, prod.line, prod.name)

        if unused_terminals:
            num_ut = len(unused_terminals)
            cls.log.warning("There %s %s unused token%s", "is" if num_ut == 1 else "are", num_ut, "s" * (num_ut > 1))

        if unused_rules:
            num_ur = len(unused_rules)
            cls.log.warning("There %s %s unused rule%s", "is" if num_ur == 1 else "are", num_ur, "s" * (num_ur > 1))

        unreachable = grammar.find_unreachable()
        for u in unreachable:
            cls.log.warning("Symbol %r is unreachable", u)

        if len(undefined_symbols) == 0:
            infinite = grammar.infinite_cycles()
            errors.extend(f"Infinite recursion detected for symbol {inf!r}\n" for inf in infinite)

        unused_prec = grammar.unused_precedence()
        for term, assoc in unused_prec:
            errors.append(f"Precedence rule {assoc!r} defined for unknown symbol {term!r}\n")

        cls._grammar = grammar
        if errors:
            msg = "\n".join(["Unable to build grammar.", *errors])
            raise YaccError(msg)

    @classmethod
    def __build_lrtables(cls) -> bool:
        """Build the LR Parsing tables from the grammar."""

        lrtable = LRTable(cls._grammar)

        # Report shift/reduce and reduce/reduce conflicts
        num_sr = len(lrtable.sr_conflicts)
        if num_sr != cls.expected_shift_reduce and num_sr >= 1:
            cls.log.warning("%d shift/reduce conflict%s", num_sr, "s" * (num_sr > 1))

        num_rr = len(lrtable.rr_conflicts)
        if num_rr != cls.expected_reduce_reduce and num_rr >= 1:
            cls.log.warning("%d reduce/reduce conflict%s", num_rr, "s" * (num_rr > 1))

        cls._lrtable = lrtable
        return True

    @classmethod
    def __collect_rules(cls, definitions: dict[str, _t.Any]) -> list[tuple[str, _t.Callable[..., _t.Any]]]:
        """Collect all of the tagged grammar rules."""

        return [(name, value) for name, value in definitions.items() if callable(value) and hasattr(value, "rules")]

    @classmethod
    def _build(cls, definitions: dict[str, _t.Any]) -> None:
        """Build the LALR(1) tables. This method is triggered by `Parser.__init_subclass__()`.

        Parameters
        ----------
        definitions: dict[str, _t.Any]
            A mapping of names to items for all definitions provided in the class, listed in the order in which they
            were defined.
        """

        # Collect all of the grammar rules from the class definition
        rules = cls.__collect_rules(definitions)

        # Validate other parts of the grammar specification
        if (spec_error := cls.__validate_specification()) is not None:
            msg = f"Invalid parser specification\n{spec_error}"
            raise YaccError(msg)

        # Build the underlying grammar object
        cls.__build_grammar(rules)

        # Build the LR tables
        if not cls.__build_lrtables():
            msg = "Can't build parsing tables."
            raise YaccError(msg)

        if cls.debugfile:
            with open(cls.debugfile, "w", encoding="utf-8") as f:
                f.write(str(cls._grammar))
                f.write("\n")
                f.write(str(cls._lrtable))
            cls.log.info("Parser debugging for %s written to %s", cls.__qualname__, cls.debugfile)

    # ----------------------------------------------------------------------
    # region ---- Parsing Support ----
    #
    # This is the parsing runtime that users use.
    # ----------------------------------------------------------------------

    def error(self, token: _t.Optional[_t.Union[Token, YaccSymbol]]) -> _t.Optional[Token]:
        """Default error handling function. This may be overridden in subclasses."""

        if token:
            if token.lineno:
                self.log.error("sly: Syntax error at line %d, token=%s\n", token.lineno, token.type)
            else:
                self.log.error("sly: Syntax error, token=%s\n", token.type)
        else:
            self.log.error("sly: Parse error in input. EOF\n")

    def errok(self) -> None:
        """Clear the error status."""

        self.errorok = True

    def restart(self) -> None:
        """Force the parser to restart from a fresh state. Clears the statestack."""

        self.statestack.clear()
        self.symstack.clear()
        self.symstack.append(YaccSymbol("$end"))
        self.statestack.append(0)
        self.state = 0

    def parse(self, tokens: _t.Iterator[Token]) -> _t.Any:
        """Parse the given input tokens."""

        # Current lookahead symbol
        self.lookahead = None
        # Stack of lookahead symbols
        lookaheadstack: list[_t.Union[Token, YaccSymbol]] = []

        # Local references (to avoid lookup on self).
        # Action table
        actions = self._lrtable.lr_action
        # Goto table
        goto = self._lrtable.lr_goto
        # Production list
        prod = self._grammar.Productions
        # Defaulted states
        defaulted_states = self._lrtable.defaulted_states

        # Production object passed to grammar rules
        pslice = YaccProduction([])
        # Used during error recovery
        errorcount = 0

        # Set up the state and symbol stacks
        self.token_stream = tokens
        statestack: list[int] = []  # Stack of parsing states
        self.statestack = statestack
        symstack: list[YaccSymbol] = []  # Stack of grammar symbols
        self.symstack = symstack
        pslice._stack = symstack  # Associate the stack with the production
        self.restart()

        # Set up position tracking
        track_positions = self.track_positions
        self._line_positions = {}  # id: -> lineno
        self._index_positions = {}  # id: -> (start, end)

        errtoken = None  # Err token
        while True:
            # Get the next symbol on the input. If a lookahead symbol
            # is already set, we just use that. Otherwise, we'll pull
            # the next token off of the lookaheadstack or from the lexer
            if self.state not in defaulted_states:
                if not self.lookahead:
                    if not lookaheadstack:
                        self.lookahead = next(tokens, None)  # Get the next token
                    else:
                        self.lookahead = lookaheadstack.pop()
                    if not self.lookahead:
                        self.lookahead = YaccSymbol("$end")

                # Check the action table
                ltype = self.lookahead.type
                t = actions[self.state].get(ltype)
            else:
                t = defaulted_states[self.state]

            if t is not None:
                if t > 0:
                    # shift a symbol on the stack
                    assert self.lookahead is not None

                    statestack.append(t)
                    self.state = t

                    symstack.append(self.lookahead)
                    self.lookahead = None

                    # Decrease error count on successful shift
                    if errorcount:
                        errorcount -= 1
                    continue

                elif t < 0:
                    # reduce a symbol on the stack, emit a production
                    self.production = p = prod[-t]
                    pname = p.name
                    plen = p.len
                    pslice._namemap = p.namemap
                    pslice._slice = symstack[-plen:] if plen else []

                    # Call the production function
                    value = p.func(self, pslice)
                    if value is pslice:
                        value = (pname, *(s.value for s in pslice._slice))

                    sym = YaccSymbol(pname, value)

                    # Record positions
                    if track_positions:
                        if plen:
                            sym.lineno = symstack[-plen].lineno
                            sym.index = symstack[-plen].index
                            sym.end = symstack[-1].end
                        else:
                            # A zero-length production  (what to put here?)
                            pass

                        _value_id = id(value)
                        self._line_positions[_value_id] = sym.lineno
                        self._index_positions[_value_id] = (sym.index, sym.end)

                    if plen:
                        del symstack[-plen:]
                        del statestack[-plen:]

                    symstack.append(sym)
                    self.state = goto[statestack[-1]][pname]
                    statestack.append(self.state)
                    continue

                else:
                    # t == 0
                    n = symstack[-1]
                    return n.value

            else:
                # t is None

                # We have some kind of parsing error here. To handle
                # this, we are going to push the current token onto
                # the tokenstack and replace it with an 'error' token.
                # If there are any synchronization rules, they may
                # catch it.
                #
                # In addition to pushing the error token, we call call
                # the user defined error() function if this is the
                # first syntax error. This function is only called if
                # errorcount == 0.

                assert self.lookahead is not None

                if errorcount == 0 or self.errorok:
                    errorcount = self.error_count
                    self.errorok = False
                    if self.lookahead.type == "$end":
                        errtoken = None  # End of file!
                    else:
                        errtoken = self.lookahead

                    if tok := self.error(errtoken):
                        # User must have done some kind of panic
                        # mode recovery on their own. The
                        # returned token is the next lookahead
                        self.lookahead = tok
                        self.errorok = True
                        continue
                    else:
                        # If at EOF. We just return. Basically dead.
                        if not errtoken:
                            return None
                else:
                    # Reset the error count. Unsuccessful token shifted
                    errorcount = self.error_count

                # case 1:  the statestack only has 1 entry on it. If we're in this state, the
                # entire parse has been rolled back and we're completely hosed.   The token is
                # discarded and we just keep going.

                if len(statestack) <= 1 and self.lookahead.type != "$end":
                    self.lookahead = None
                    self.state = 0
                    # Nuke the lookahead stack
                    lookaheadstack.clear()
                    continue

                # case 2: the statestack has a couple of entries on it, but we're
                # at the end of the file. nuke the top entry and generate an error token

                # Start nuking entries on the stack
                elif self.lookahead.type == "$end":
                    # Whoa. We're really hosed here. Bail out
                    return None

                elif self.lookahead.type != "error":
                    sym = symstack[-1]
                    if sym.type == "error":
                        # Hmmm. Error is on top of stack, we'll just nuke input symbol and continue
                        self.lookahead = None
                        continue

                    # Create the error symbol for the first time and make it the new lookahead symbol
                    t = YaccSymbol(
                        "error",
                        self.lookahead,
                        self.lookahead.lineno,
                        self.lookahead.index,
                        self.lookahead.end,
                    )

                    lookaheadstack.append(self.lookahead)
                    self.lookahead = t
                else:
                    sym = symstack.pop()
                    statestack.pop()
                    self.state = statestack[-1]
                continue

            # Call an error function here
            msg = "sly: internal parser error!!!\n"
            raise RuntimeError(msg)

    def line_position(self, value: object) -> _t.Optional[int]:
        """Get the line number of any object returned by one of the various methods in the parser definition.

        Typically, it would be a AST node.

        Notes
        -----
        The parser tracks the data using the value of id(value).
        """

        return self._line_positions[id(value)]

    def index_position(self, value: object) -> tuple[_t.Optional[int], _t.Optional[int]]:
        """Get a (start, end) index pair of any object returned by one of the various methods in the parser definition.

        Typically, it would be a AST node.

        Notes
        -----
        The parser tracks the data using the value of id(value).
        """

        return self._index_positions[id(value)]

    # endregion


# endregion
