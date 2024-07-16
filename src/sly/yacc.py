# region License
# -----------------------------------------------------------------------------
# sly: yacc.py
#
# Copyright (C) 2016-2018
# David M. Beazley (Dabeaz LLC)
# Copyright (C) 2024, Sachaa-Thanasius
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

import inspect
import sys
import threading
from collections import defaultdict
from collections.abc import Callable, Collection, Generator, Iterator
from typing import TYPE_CHECKING, Any, ClassVar, Optional, TextIO, Union, cast

from ._misc import MISSING, CallableT, TypeAlias, override
from .lex import Token

__all__ = ("Parser",)


# ============================================================================
# region -------- User configurable parameters --------
#
# Change these to modify the default behavior of yacc (if you wish).
# Move these parameters to the Yacc class itself.
# ============================================================================

MAXINT: int = sys.maxsize  # TODO: Somehow avoid this global state.

# endregion


class YaccError(Exception):
    """Exception raised for yacc-related build errors."""


class SlyLogger:
    """This object is a stand-in for a logging object created by the logging module.

    Extended Summary
    ----------------
    SLY will use this by default to create things such as the parser.out file. If a user wants more detailed
    information, they can create their own logging object and pass it into SLY.
    """

    def __init__(self, f: TextIO) -> None:
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

    Attributes
    ----------
    type: str
        Grammar symbol type.
    value: Any
        Symbol value.
    lineno: int
        Starting line number. May not exist.
    index: int
        Starting lex position. May not exist.
    end: int | None
        May not exist.

    Notes
    -----
    Instances *usually* have the documented attributes, but only the `type` and `value` attributes are guaranteed to
    exist.

    Also, this is meant be mostly duck type–compatible with `lex.Token`.
    """

    if TYPE_CHECKING:
        lineno: Optional[int]
        index: Optional[int]
        end: Optional[int]

    def __init__(self, *, type: str, value: Any = MISSING) -> None:  # noqa: A002
        self.type = type
        if value is not MISSING:
            self.value = value

    @override
    def __str__(self) -> str:
        return self.type

    @override
    def __repr__(self) -> str:
        return str(self)


class YaccProduction:
    """This class is a wrapper around the objects actually passed to each grammar rule.

    Methods
    -------
    lineno()
        Return the line number of the item if found. Otherwise, raise.

    Notes
    -----
    Index lookup and assignment actually assign the `.value` attribute of the underlying `YaccSymbol` object.
    """

    __slots__ = ("_slice", "_namemap", "_stack")

    def __init__(self, s: list[YaccSymbol], stack: Optional[list[YaccSymbol]] = None) -> None:
        self._slice = s
        self._namemap: dict[str, Callable[[list[YaccSymbol]], Any]] = {}
        self._stack = stack

    @property
    def lineno(self) -> int:
        """Get the line number of a given item.

        Returns
        -------
        lineno: int
            The line number of the item.

        Raises
        ------
        AttributeError
            If no line number was found (or it was 0 for some reason).
        """

        for tok in self._slice:
            lineno = getattr(tok, "lineno", None)
            if lineno:
                return lineno
        msg = "No line number found."
        raise AttributeError(msg)

    @property
    def index(self) -> Any:
        for tok in self._slice:
            index = getattr(tok, "index", None)
            if index is not None:
                return index
        msg = "No index attribute found."
        raise AttributeError(msg)

    @property
    def end(self) -> Any:
        result = None
        for tok in self._slice:
            r = getattr(tok, "end", None)
            if r:
                result = r
        return result

    def __getitem__(self, n: int) -> Any:
        if n >= 0:
            return self._slice[n].value
        else:
            assert self._stack is not None
            return self._stack[n].value

    def __setitem__(self, n: int, v: Any) -> None:
        if n >= 0:
            self._slice[n].value = v
        else:
            assert self._stack is not None
            self._stack[n].value = v

    def __len__(self) -> int:
        return len(self._slice)

    def __getattr__(self, name: str) -> Any:
        if name in self._namemap:
            return self._namemap[name](self._slice)
        else:
            msg = f"No symbol {name}. Must be one of {{{', '.join(self._namemap)}}}."
            raise AttributeError(msg)

    @override
    def __setattr__(self, name: str, value: object) -> None:
        if name[:1] == "_":
            super().__setattr__(name, value)
        else:
            msg = f"Can't reassign the value of attribute {name!r}."
            raise AttributeError(msg)


# ============================================================================
# region -------- Grammar Representation --------
#
# The following functions, classes, and variables are used to represent and
# manipulate the rules that make up a grammar.
# ============================================================================


class Production:
    """This class stores the raw information about a single production or grammar rule.

    Extended Summary
    ----------------
    A grammar rule refers to a specification such as this: "expr : expr PLUS term".

    Parameters
    ----------
    number: int
        Production number.
    name: str
        Name of the production, e.g. "expr".
    prod: list[str]
        A list of symbols on the right side, e.g. ["expr", "PLUS", "term"].
    prec: tuple[str, int], default=("right", 0)
        Production precedence level.
    func: Callable[..., Any], optional
        Function that executes on reduce. Defaults to None.
    file: str, default=""
        File where production function is defined.
    line: int, default=0
        Line number where production function is defined.

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
    func: Callable
        Function that executes on reduce.
    file: str
        File where production function is defined.
    line: int
        Line number where production function is defined.
    len: int
        Length of the production (number of symbols on right hand side).
    usyms: list[str]
        Set of unique symbols found in the production.
    """

    reduced: int = 0

    def __init__(
        self,
        number: int,
        name: str,
        prod: list[str],
        precedence: tuple[str, int] = ("right", 0),
        func: Optional[Callable[..., Any]] = None,
        file: str = "",
        line: int = 0,
    ) -> None:
        self.name = name
        self.prod: tuple[str, ...] = tuple(prod)
        self.number = number
        self.func = func
        self.file = file
        self.line = line
        self.prec = precedence

        # Internal settings used during table construction
        self.len = len(self.prod)

        # Create a list of unique production symbols used in the production
        self.usyms: list[str] = []
        symmap: defaultdict[str, list[int]] = defaultdict(list)
        for n, s in enumerate(self.prod):
            symmap[s].append(n)
            if s not in self.usyms:
                self.usyms.append(s)

        # Create a name mapping
        # First determine (in advance) if there are duplicate names
        namecount: defaultdict[str, int] = defaultdict(int)
        for key in self.prod:
            namecount[key] += 1
            if key in _name_aliases:
                for k in _name_aliases[key]:
                    namecount[k] += 1

        # Now, walk through the names and generate accessor functions
        nameuse: defaultdict[str, int] = defaultdict(int)
        namemap: dict[str, Callable[[list[YaccSymbol]], object]] = {}
        for index, key in enumerate(self.prod):
            if namecount[key] > 1:
                k = f"{key}{nameuse[key]}"
                nameuse[key] += 1
            else:
                k = key
            namemap[k] = lambda s, i=index: s[i].value
            if key in _name_aliases:
                for n, alias in enumerate(_name_aliases[key]):
                    if namecount[alias] > 1:
                        k = f"{alias}{nameuse[alias]}"
                        nameuse[alias] += 1
                    else:
                        k = alias
                    # The value is either a list (for repetition) or a tuple for optional
                    namemap[k] = (
                        lambda s, i=index, n=n: ([x[n] for x in s[i].value])
                        if isinstance(s[i].value, list)
                        else s[i].value[n]
                    )

        self.namemap = namemap

        # List of all LR items for the production
        self.lr_items: list[LRItem] = []
        self.lr_next: Optional[LRItem] = None

        self.lr0_added: int = 0

    @override
    def __str__(self) -> str:
        if self.prod:
            s = f'{self.name} -> {" ".join(self.prod)}'
        else:
            s = f"{self.name} -> <empty>"

        if self.prec[1]:
            s += f"  [precedence={self.prec[0]}, level={self.prec[1]}]"

        return s

    @override
    def __repr__(self) -> str:
        return f"Production({self})"

    def __len__(self) -> int:
        return len(self.prod)

    def __getitem__(self, index: int) -> str:
        return self.prod[index]

    def lr_item(self, n: int) -> Optional["LRItem"]:
        """Return the nth lr_item from the production (or None if at the end)."""

        if n > len(self.prod):
            return None
        p = LRItem(self, n)

        # Precompute the list of productions immediately following.
        # NOTE: Prodnames isn't global, so this doesn't seem possible. Leaving it just in case.
        p.lr_after = []

        try:
            p.lr_before = p.prod[n - 1]
        except IndexError:
            p.lr_before = None
        return p


class LRItem:
    """This class represents a specific stage of parsing a production rule, e.g. "expr : expr . PLUS term".

    Extended Summary
    ----------------
    In the example given in the short summary, the "." represents the current location of the parse.

    Attributes
    ----------
    name: str
        Name of the production, e.g. "expr".
    prod: tuple[str, ...]
        A list of symbols on the right side ["expr", ".", "PLUS", "term"].
    number: int
        Production number.
    lr_next: LRItem | None
        Next LR item. Example: If we are "expr -> expr . PLUS term", then lr_next refers to "expr -> expr PLUS . term".
    lr_index: int
        LR item index (location of the ".") in the prod list.
    lookaheads: dict[int, list[str]]
        LALR lookahead symbols for this item.
    len: int
        Length of the production (number of symbols on right hand side).
    lr_after: list[Production]
        List of all productions that immediately follow.
    lr_before: str
        Grammar symbol immediately before.
    """

    if TYPE_CHECKING:
        lr_next: Optional["LRItem"]
        lr_after: list[Production]
        lr_before: Optional[str]

    def __init__(self, p: Production, n: int) -> None:
        self.name: str = p.name
        self.prod: tuple[str, ...] = p.prod[:n] + (".",) + p.prod[n:]
        self.number: int = p.number
        self.lr_index: int = n
        self.lookaheads: dict[int, list[str]] = {}
        self.len: int = len(self.prod)
        self.usyms: list[str] = p.usyms

    @override
    def __str__(self) -> str:
        if self.prod:
            s = f'{self.name} -> {" ".join(self.prod)}'
        else:
            s = f"{self.name} -> <empty>"
        return s

    @override
    def __repr__(self) -> str:
        return f"LRItem({self})"


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

    def __init__(self, terminals: Collection[str]) -> None:
        # fmt: off
        self.Productions:       list[Production]            = [None]  # pyright: ignore # Reserved spot.
        self.Prodnames:         dict[str, list[Production]] = {}
        self.Prodmap:           dict[str, Production]       = {}
        self.Terminals:         dict[str, list[int]]        = dict({term: [] for term in terminals}, error=[])
        self.Nonterminals:      dict[str, list[int]]        = {}
        self.First:             dict[str, list[str]]        = {}
        self.Follow:            dict[str, list[str]]        = {}
        self.Precedence:        dict[str, tuple[str, int]]  = {}
        self.UsedPrecedence:    set[str]                    = set()
        self.Start:             Optional[str]               = None
        # fmt: on

    def __len__(self) -> int:
        return len(self.Productions)

    def __getitem__(self, index: int) -> Production:
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
        func: Optional[Callable[..., Any]] = None,
        file: str = "",
        line: int = 0,
    ) -> None:
        """Given an action function, this function assembles a production rule and computes its precedence level.

        Extended Summary
        ----------------
        Precedence is determined by the precedence of the right-most non-terminal or the precedence of a terminal
        specified by "%prec".

        Parameters
        ----------
        prodname: str
            The name of the production, e.g. "expr" for the rule "expr : expr PLUS term".
        syms: list[str]
            The list of symbols representing the production, e.g. ["expr", "PLUS", "term"] for the rule
            "expr : expr PLUS term".
        func: Callable[..., Any], optional
            The action function. Defaults to None.

        Raises
        ------
        GrammarError
            If a production symbol is invalid, or if "%prec" is used incorrectly.
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
        p = Production(pnumber, prodname, syms, prodprec, func, file, line)
        self.Productions.append(p)
        self.Prodmap[map_] = p

        # NOTE: The comment seems outdated, indicating Productions was a global in an earlier revision.
        # Add to the global productions list
        try:
            self.Prodnames[prodname].append(p)
        except KeyError:
            self.Prodnames[prodname] = [p]

    def set_start(self, start: Optional[Union[Callable[..., Any], str]] = None) -> None:
        """Sets the starting symbol and creates the augmented grammar.

        Extended Summary
        ----------------
        Production rule 0 is "S' -> start" where `start` is the start symbol.
        """

        if callable(start):
            start = start.__name__

        if not start:
            start = self.Productions[1].name

        if start not in self.Nonterminals:
            msg = f"Start symbol {start!r} undefined."
            raise GrammarError(msg)
        self.Productions[0] = Production(0, "S'", [start])
        self.Nonterminals[start].append(0)
        self.Start = start

    def find_unreachable(self) -> list[str]:
        """Find all of the nonterminal symbols that can't be reached from the starting symbol.

        Returns
        -------
        list[str]
            A list of nonterminals that can't be reached.
        """

        # Mark all symbols that are reachable from a symbol s
        def mark_reachable_from(s: str) -> None:
            if s in reachable:
                return
            reachable.add(s)
            for p in self.Prodnames.get(s, []):
                for r in p.prod:
                    mark_reachable_from(r)

        reachable: set[str] = set()
        mark_reachable_from(self.Productions[0].prod[0])
        return [s for s in self.Nonterminals if s not in reachable]

    def infinite_cycles(self) -> list[str]:
        """This function looks at the various parsing rules and tries to detect infinite recursion cycles.

        Notes
        -----
        Infinite recursion cycles occur with grammar rules where there is no possible way to derive a string of only
        terminals.
        """

        terminates: dict[str, bool] = {}

        # Terminals:
        terminates.update({t: True for t in self.Terminals})
        terminates["$end"] = True

        # Nonterminals:

        # Initialize to false:
        terminates.update({n: False for n in self.Nonterminals})

        # Then propagate termination until no change:
        while True:
            some_change = False
            for n, pl in self.Prodnames.items():
                # Nonterminal n terminates iff any of its productions terminates.
                for p in pl:
                    # Production p terminates iff all of its rhs symbols terminate.
                    for s in p.prod:
                        if not terminates[s]:
                            # The symbol s does not terminate,
                            # so production p does not terminate.
                            p_terminates = False
                            break
                    else:
                        # didn't break from the loop,
                        # so every symbol s terminates
                        # so production p terminates.
                        p_terminates = True

                    if p_terminates:
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

        result: list[tuple[str, Production]] = []
        for p in self.Productions:
            if not p:
                continue

            for s in p.prod:
                if s not in self.Prodnames and s not in self.Terminals and s != "error":
                    result.append((s, p))
        return result

    def unused_terminals(self) -> list[str]:
        """Find all terminals that were defined, but not used by the grammar.

        Returns
        -------
        list[str]
            A list of all defined, unused symbols.
        """

        return [s for s, v in self.Terminals.items() if s != "error" and not v]

    def unused_rules(self) -> list[Production]:
        """Find all grammar rules that were defined, but not used (maybe not reachable).

        Returns
        -------
        unused_prod: list[Production]
            A list of defined, unused productions.
        """

        return [self.Prodnames[s][0] for s, v in self.Nonterminals.items() if not v]

    def unused_precedence(self) -> list[tuple[str, str]]:
        """Returns a list of tuples corresponding to precedence rules that were never used by the grammar.

        Returns
        -------
        unused: list[tuple[str, str]]
            A list of tuples representing unused precedence rules. The tuples are in the format (term, precedence),
            where term is the name of the terminal on which precedence was applied and precedence is a string such as
            'left' or 'right' corresponding to the type of precedence.
        """

        unused: list[tuple[str, str]] = []
        for termname in self.Precedence:
            if not (termname in self.Terminals or termname in self.UsedPrecedence):
                unused.append((termname, self.Precedence[termname][0]))

        return unused

    def _first(self, beta: tuple[str, ...]) -> list[str]:
        """Compute the value of FIRST1(beta) where beta is a tuple of symbols.

        Extended Summary
        ----------------
        During execution of `compute_first()`, the result may be incomplete.
        Afterward (e.g., when called from `compute_follow()`), it will be complete.
        """

        # We are computing First(x1,x2,x3,...,xn)
        result: list[str] = []
        for x in beta:
            x_produces_empty = False

            # Add all the non-<empty> symbols of First[x] to the result.
            for f in self.First[x]:
                if f == "<empty>":
                    x_produces_empty = True
                else:
                    if f not in result:
                        result.append(f)

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
            result.append("<empty>")

        return result

    def compute_first(self) -> dict[str, list[str]]:
        """Compute the value of FIRST1(X) for all symbols."""

        if self.First:
            return self.First

        # Terminals:
        self.First.update({t: [t] for t in self.Terminals})
        self.First["$end"] = ["$end"]

        # Nonterminals:

        # Initialize to the empty set:
        self.First.update({n: [] for n in self.Nonterminals})

        # Then propagate symbols until no change:
        while True:
            some_change = False
            for n in self.Nonterminals:
                for p in self.Prodnames[n]:
                    for f in self._first(p.prod):
                        if f not in self.First[n]:
                            self.First[n].append(f)
                            some_change = True
            if not some_change:
                break

        return self.First

    def compute_follow(self, start: Optional[str] = None) -> dict[str, list[str]]:
        """Computes all of the follow sets for every non-terminal symbol.

        Notes
        -----
        The follow set is the set of all symbols that might follow a given non-terminal.
        See the Dragon book, 2nd Ed. p. 189.
        """

        # If already computed, return the result
        if self.Follow:
            return self.Follow

        # If first sets not computed yet, do that first.
        if not self.First:
            self.compute_first()

        # Add '$end' to the follow list of the start symbol
        for k in self.Nonterminals:
            self.Follow[k] = []

        if not start:
            start = self.Productions[1].name

        self.Follow[start] = ["$end"]

        while True:
            didadd = False
            for p in self.Productions[1:]:
                # Here is the production set
                for i, B in enumerate(p.prod):
                    if B in self.Nonterminals:
                        # Okay. We got a non-terminal in a production
                        fst = self._first(p.prod[i + 1 :])
                        hasempty = False
                        for f in fst:
                            if f != "<empty>" and f not in self.Follow[B]:
                                self.Follow[B].append(f)
                                didadd = True
                            if f == "<empty>":
                                hasempty = True
                        if hasempty or i == (len(p.prod) - 1):
                            # Add elements of follow(a) to follow(b)
                            for f in self.Follow[p.name]:
                                if f not in self.Follow[B]:
                                    self.Follow[B].append(f)
                                    didadd = True
            if not didadd:
                break
        return self.Follow

    def build_lritems(self) -> None:
        """This function walks the list of productions and builds a complete set of the LR items.

        Extended Summary
        ----------------
        The LR items are stored in two ways:  First, they are uniquely numbered and placed in the list _lritems.
        Second, a linked list of LR items is built for each production. For example:

            E -> E PLUS E

        creates the list

            [E -> . E PLUS E, E -> E . PLUS E, E -> E PLUS . E, E -> E PLUS E . ]
        """

        for p in self.Productions:
            lastlri = p
            i = 0
            lr_items: list[LRItem] = []
            while True:
                if i > len(p):
                    lri = None
                else:
                    lri = LRItem(p, i)
                    # Precompute the list of productions immediately following
                    try:
                        lri.lr_after = self.Prodnames[lri.prod[i + 1]]
                    except (IndexError, KeyError):
                        lri.lr_after = []
                    try:
                        lri.lr_before = lri.prod[i - 1]
                    except IndexError:
                        lri.lr_before = None

                lastlri.lr_next = lri
                if not lri:
                    break
                lr_items.append(lri)
                lastlri = lri
                i += 1
            p.lr_items = lr_items

    @override
    def __str__(self) -> str:
        """Return str(self).

        Extended Summary
        ----------------
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
        out.extend(f'{term} : {" ".join(str(s) for s in self.Terminals[term])}' for term in sorted(self.Terminals))

        out.append("\nNonterminals, with rules where they appear:\n")
        out.extend(
            f'{nonterm} : {" ".join(str(s) for s in self.Nonterminals[nonterm])}'
            for nonterm in sorted(self.Nonterminals)
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


_RelationFunction: TypeAlias = Callable[[tuple[int, str]], list[tuple[int, str]]]
_SetValuedFunction: TypeAlias = Callable[[tuple[int, str]], list[str]]


def digraph(X: list[tuple[int, str]], R: _RelationFunction, FP: _SetValuedFunction) -> dict[tuple[int, str], list[str]]:
    """First helper for computing set valued functions of the form `F(x) = F'(x) U U{F(y) | x R y}`.

    Extended Summary
    ----------------
    This is used to compute the values of Read() sets as well as FOLLOW sets in LALR(1) generation.

    Parameters
    ----------
    X: list[tuple[int, str]]
        An input set.
    R: _RelationFunction
        A relation.
    FP: _SetValuedFunction
        Set-valued function.

    See Also
    --------
    traverse
    """

    N = dict.fromkeys(X, 0)
    stack: list[tuple[int, str]] = []
    F: dict[tuple[int, str], list[str]] = {}
    for x in X:
        if N[x] == 0:
            traverse(x, N, stack, F, X, R, FP)
    return F


def traverse(
    x: tuple[int, str],
    N: dict[tuple[int, str], int],
    stack: list[tuple[int, str]],
    F: dict[tuple[int, str], list[str]],
    X: list[tuple[int, str]],
    R: _RelationFunction,
    FP: _SetValuedFunction,
) -> None:
    """Second helper for computing set valued functions of the form `F(x) = F'(x) U U{F(y) | x R y}`.

    Extended Summary
    ----------------
    This is used to compute the values of Read() sets as well as FOLLOW sets in LALR(1) generation.

    See Also
    --------
    digraph
    """

    stack.append(x)
    d = len(stack)
    N[x] = d
    F[x] = FP(x)  # F(X) <- F'(x)

    rel = R(x)  # Get y's related to x
    for y in rel:
        if N[y] == 0:
            traverse(y, N, stack, F, X, R, FP)
        N[x] = min(N[x], N[y])
        for a in F.get(y, []):
            if a not in F[x]:
                F[x].append(a)
    if N[x] == d:
        N[stack[-1]] = MAXINT
        F[stack[-1]] = F[x]
        element = stack.pop()
        while element != x:
            N[stack[-1]] = MAXINT
            F[stack[-1]] = F[x]
            element = stack.pop()


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
            Union[tuple[int, str], str],
            Union[list[LRItem], dict[Union[int, str], list[LRItem]]],
        ] = {}
        self.lr0_cidhash: dict[int, int] = {}  # Cache of closures
        self._add_count = 0  # Internal counter used to detect cycles

        # Diagonistic information filled in by the table generator
        self.state_descriptions: dict[int, str] = {}
        self.sr_conflict = 0
        self.rr_conflict = 0
        self.conflicts = []  # List of conflicts

        self.sr_conflicts: list[tuple[int, str, str]] = []
        self.rr_conflicts: list[tuple[int, Production, Production]] = []

        # Build the tables
        self.grammar.build_lritems()
        self.grammar.compute_first()
        self.grammar.compute_follow()
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
            A set of LR(0) items
        """

        self._add_count += 1

        # Add everything in I to J
        J: list[LRItem] = I[:]
        didadd = True
        while didadd:
            didadd = False
            for j in J:
                for x in j.lr_after:
                    if getattr(x, "lr0_added", 0) == self._add_count:
                        continue
                    # Add B --> .G to J
                    J.append(x.lr_next)
                    x.lr0_added = self._add_count
                    didadd = True

        return J

    def lr0_goto(self, I: list[LRItem], x: str) -> Optional[list[LRItem]]:
        """Compute the LR(0) goto function goto(I,X) where I is a set of LR(0) items and X is a grammar symbol.

        Notes
        -----
        This function is written in a way that guarantees uniqueness of the generated goto sets (i.e. the same
        goto set will never be returned as two different Python objects). With uniqueness, we can later do fast
        set comparisons using id(obj) instead of element-wise comparison.
        """

        # First we look for a previously cached entry
        try:
            g = self.lr_goto_cache[(id(I), x)]
        except KeyError:
            pass
        else:
            assert isinstance(g, (list, type(None)))
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

        assert self.grammar.Productions[0].lr_next

        C = [self.lr0_closure([self.grammar.Productions[0].lr_next])]

        self.lr0_cidhash.update({id(I): i for i, I in enumerate(C)})

        # Loop over the items in C and each grammar symbols
        i = 0
        while i < len(C):
            I = C[i]
            i += 1

            # Collect all of the symbols that could possibly be in the goto(I,X) sets
            asyms: dict[str, None] = dict.fromkeys([s for ii in I for s in ii.usyms])

            for x in asyms:
                g = self.lr0_goto(I, x)
                if not g or id(g) in self.lr0_cidhash:
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
        """Creates a dictionary containing all of the non-terminals that might produce an empty production."""

        nullable: set[str] = set()
        num_nullable = 0
        while True:
            for p in self.grammar.Productions[1:]:
                if p.len == 0:
                    nullable.add(p.name)
                    continue
                for t in p.prod:
                    if t not in nullable:
                        break
                else:
                    nullable.add(p.name)
            if len(nullable) == num_nullable:
                break
            num_nullable = len(nullable)
        return nullable

    def find_nonterminal_transitions(self, C: list[list[LRItem]]) -> list[tuple[int, str]]:
        """Given a set of LR(0) items, this functions finds all of the non-terminal transitions.

        Extended Summary
        ----------------
        Non-terminal transitions are transitions in which a dot appears immediately before a non-terminal.

        Parameters
        ----------
        C: list[list[LRItem]]
            The set of LR(0) items.

        Returns
        -------
        The list of nonterminal transitions, which are tuples of the form (state,N) where state is the state number
        and N is the nonterminal symbol.
        """

        trans: list[tuple[int, str]] = []
        for stateno, state in enumerate(C):
            for p in state:
                if p.lr_index < p.len - 1:
                    t = (stateno, p.prod[p.lr_index + 1])
                    if t[1] in self.grammar.Nonterminals and t not in trans:
                        trans.append(t)
        return trans

    def dr_relation(self, C: list[list[LRItem]], trans: tuple[int, str], nullable: set[str]) -> list[str]:
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
        terms: list[str]
            A list of terminals.
        """

        state, N = trans
        terms: list[str] = []

        g = self.lr0_goto(C[state], N)
        assert g
        for p in g:
            if p.lr_index < p.len - 1:
                a = p.prod[p.lr_index + 1]
                if a in self.grammar.Terminals and a not in terms:
                    terms.append(a)

        # This extra bit is to handle the start state
        if state == 0 and self.grammar.Productions[0].prod[0] == N:
            terms.append("$end")

        return terms

    def reads_relation(self, C: list[list[LRItem]], trans: tuple[int, str], empty: set[str]) -> list[tuple[int, str]]:
        """Computes the READS() relation (p,A) READS (t,C)."""

        # Look for empty transitions
        rel: list[tuple[int, str]] = []
        state, N = trans

        g = self.lr0_goto(C[state], N)
        assert g
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
        trans: list[tuple[int, str]],
        nullable: set[str],
    ) -> tuple[dict[tuple[int, str], list[tuple[int, LRItem]]], dict[tuple[int, str], list[tuple[int, str]]]]:
        """Determines the lookback and includes relations.

        Notes
        -----
        LOOKBACK:

        This relation is determined by running the LR(0) state machine forward. For example, starting with a production
        "N : . A B C", we run it forward to obtain "N : A B C ." We then build a relationship between this final state
        and the starting state. These relationships are stored in a dictionary `lookdict`.

        INCLUDES:

        Computes the INCLUDE() relation (p,A) INCLUDES (p',B).

        This relation is used to determine non-terminal transitions that occur inside of other non-terminal transition
        states. (p,A) INCLUDES (p', B) if the following holds:

            B -> LAT, where T -> epsilon and p' -L-> p

        L is essentially a prefix (which may be empty), T is a suffix that must be able to derive an empty string.
        State p' must lead to state p with the string L.
        """

        lookdict: dict[tuple[int, str], list[tuple[int, LRItem]]] = {}  # Dictionary of lookback relations
        includedict: dict[tuple[int, str], list[tuple[int, str]]] = {}  # Dictionary of include relations

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

                lr_index = p.lr_index
                j = state
                while lr_index < p.len - 1:
                    lr_index = lr_index + 1
                    t = p.prod[lr_index]

                    # Check to see if this symbol and state are a non-terminal transition
                    if (j, t) in dtrans:
                        # Yes. Okay, there is some chance that this is an includes relation
                        # the only way to know for certain is whether the rest of the
                        # production derives empty

                        li = lr_index + 1
                        while li < p.len:
                            if p.prod[li] in self.grammar.Terminals:
                                break  # No forget it
                            if p.prod[li] not in nullable:
                                break
                            li = li + 1
                        else:
                            # Appears to be a relation between (j,t) and (state,N)
                            includes.append((j, t))

                    g = self.lr0_goto(C[j], t)  # Go to next set
                    j = self.lr0_cidhash.get(id(g), -1)  # Go to next state

                # When we get here, j is the final state, now we have to locate the production
                for r in C[j]:
                    if r.name != p.name:
                        continue
                    if r.len != p.len:
                        continue
                    i = 0
                    # This look is comparing a production ". A B C" with "A B C ."
                    while i < r.lr_index:
                        if r.prod[i] != p.prod[i + 1]:
                            break
                        i = i + 1
                    else:
                        lookb.append((j, r))
            for i in includes:
                if i not in includedict:
                    includedict[i] = []
                includedict[i].append((state, N))
            lookdict[(state, N)] = lookb

        return lookdict, includedict

    def compute_read_sets(
        self,
        C: list[list[LRItem]],
        ntrans: list[tuple[int, str]],
        nullable: set[str],
    ) -> dict[tuple[int, str], list[str]]:
        """Given a set of LR(0) items, this function computes the read sets.

        Parameters
        ----------
        C: list[list[LRItem]]
            Set of LR(0) items.
        ntrans: list[tuple[int, str]]
            Set of nonterminal transitions.
        nullable: set[str]
            Set of empty transitions.

        Returns
        -------
        F: dict[tuple[int, str], list[str]]
            A set containing the read sets.
        """

        def FP(x: tuple[int, str]) -> list[str]:
            return self.dr_relation(C, x, nullable)

        def R(x: tuple[int, str]) -> list[tuple[int, str]]:
            return self.reads_relation(C, x, nullable)

        F = digraph(ntrans, R, FP)
        return F  # noqa: RET504

    def compute_follow_sets(
        self,
        ntrans: list[tuple[int, str]],
        readsets: dict[tuple[int, str], list[str]],
        inclsets: dict[tuple[int, str], list[tuple[int, str]]],
    ) -> dict[tuple[int, str], list[str]]:
        """Given a set of LR(0) items, a set of non-terminal transitions, a readset, and an include set, this function
        computes the follow sets: Follow(p,A) = Read(p,A) U U {Follow(p',B) | (p,A) INCLUDES (p',B)}.

        Parameters
        ----------
        ntrans: list[tuple[int, str]]
            Set of nonterminal transitions.
        readsets: dict[tuple[int, str], list[str]]
            Readset (previously computed).
        inclsets: dict[tuple[int, str], list[tuple[int, str]]]
            Include sets (previously computed).

        Returns
        -------
        F: dict[tuple[int, str], list[str]]
            A set containing the follow sets.
        """

        def FP(x: tuple[int, str]) -> list[str]:
            return readsets[x]

        def R(x: tuple[int, str]) -> list[tuple[int, str]]:
            return inclsets.get(x, [])

        F = digraph(ntrans, R, FP)
        return F  # noqa: RET504

    def add_lookaheads(
        self,
        lookbacks: dict[tuple[int, str], list[tuple[int, LRItem]]],
        followset: dict[tuple[int, str], list[str]],
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
                if state not in p.lookaheads:
                    p.lookaheads[state] = []
                f = followset.get(trans, [])
                for a in f:
                    if a not in p.lookaheads[state]:
                        p.lookaheads[state].append(a)

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

        actionp = {}  # Action production array (temporary)

        # Step 1: Construct C = { I0, I1, ... IN}, collection of LR(0) items
        # This determines the number of states

        C = self.lr0_items()
        self.add_lalr_lookaheads(C)

        # Build the parser table, state by state
        for st, I in enumerate(C):
            descrip: list[str] = []
            # Loop over each production in I
            actlist: list[tuple[str, LRItem, str]] = []  # List of actions
            st_action: dict[str, Optional[int]] = {}
            st_actionp: dict[str, LRItem] = {}
            st_goto: dict[str, int] = {}

            descrip.append(f"\nstate {st}\n")
            descrip.extend(f"    ({p.number}) {p}" for p in I)

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
            _actprint: dict[tuple[str, str], int] = {}
            for a, p, m in actlist:
                if (a in st_action) and (p is st_actionp[a]):
                    descrip.append(f"    {a:<15s} {m}")
                    _actprint[(a, m)] = 1
            descrip.append("")

            # Construct the goto table for this state
            nkeys: dict[str, None] = dict.fromkeys(s for ii in I for s in ii.usyms if s in self.grammar.Nonterminals)

            for n in nkeys:
                g = self.lr0_goto(I, n)
                j = self.lr0_cidhash.get(id(g), -1)
                if j >= 0:
                    st_goto[n] = j
                    descrip.append(f"    {n:<30s} shift and go to state {j}")

            action[st] = st_action
            actionp[st] = st_actionp
            goto[st] = st_goto
            self.state_descriptions[st] = "\n".join(descrip)

    @override
    def __str__(self) -> str:
        """Return str(self).

        Extended Summary
        ----------------
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


_RawGrammarRule: TypeAlias = tuple[Callable[..., Any], str, int, str, list[str]]


def _collect_grammar_rules(func: Callable[..., Any]) -> list[_RawGrammarRule]:
    """Collect grammar rules from a function."""

    grammar: list[_RawGrammarRule] = []
    curr_func: Optional[Callable[..., Any]] = func
    while curr_func:
        prodname = curr_func.__name__
        unwrapped = inspect.unwrap(curr_func)
        filename: str = unwrapped.__code__.co_filename
        lineno_start: int = unwrapped.__code__.co_firstlineno
        func_rules = cast(list[str], curr_func.rules)  # pyright: ignore # Pre-confirmed .rules exists.
        for rule, lineno in zip(func_rules, range(lineno_start + len(func_rules) - 1, 0, -1)):
            syms = rule.split()
            ebnf_prod: list[_RawGrammarRule] = []
            while ("{" in syms) or ("[" in syms):
                for s in syms:
                    if s == "[":
                        syms, prod = _replace_ebnf_optional(syms)
                        ebnf_prod.extend(prod)
                        break
                    if s == "{":
                        syms, prod = _replace_ebnf_repeat(syms)
                        ebnf_prod.extend(prod)
                        break
                    if "|" in s:
                        syms, prod = _replace_ebnf_choice(syms)
                        ebnf_prod.extend(prod)
                        break

            if syms[1:2] == [":"] or syms[1:2] == ["::="]:
                grammar.append((curr_func, filename, lineno, syms[0], syms[2:]))
            else:
                grammar.append((curr_func, filename, lineno, prodname, syms))
            grammar.extend(ebnf_prod)

        curr_func = getattr(curr_func, "next_func", None)

    return grammar


def _replace_ebnf_repeat(syms: list[str]) -> tuple[list[str], list[_RawGrammarRule]]:
    """Replace EBNF repetition."""

    syms = list(syms)
    first = syms.index("{")
    end = syms.index("}", first)

    # Look for choices inside
    repeated_syms = syms[first + 1 : end]
    if any("|" in sym for sym in repeated_syms):
        repeated_syms, prods = _replace_ebnf_choice(repeated_syms)
    else:
        prods = []

    symname, moreprods = _generate_repeat_rules(repeated_syms)
    syms[first : end + 1] = [symname]
    return syms, prods + moreprods


def _replace_ebnf_optional(syms: list[str]) -> tuple[list[str], list[_RawGrammarRule]]:
    syms = list(syms)
    first = syms.index("[")
    end = syms.index("]", first)
    symname, prods = _generate_optional_rules(syms[first + 1 : end])
    syms[first : end + 1] = [symname]
    return syms, prods


def _replace_ebnf_choice(syms: list[str]) -> tuple[list[str], list[_RawGrammarRule]]:
    syms = list(syms)
    newprods: list[_RawGrammarRule] = []
    for n, sym in enumerate(syms):
        if "|" in sym:
            symname, prods = _generate_choice_rules(sym.split("|"))
            syms[n] = symname
            newprods.extend(prods)

    return syms, newprods


_gencount = 0
"""Generate grammar rules for repeated items."""

_gencount_lock = threading.Lock()
"""Threading lock that guards _gencount."""

_name_aliases: dict[str, list[str]] = {}
"""Dictionary mapping name aliases generated by EBNF rules."""

_name_aliases_lock = threading.Lock()
"""Threading lock that guards _name_aliases."""


def _sanitize_symbols(symbols: list[str]) -> Generator[str]:
    for sym in symbols:
        if sym.startswith("'"):
            yield str(hex(ord(sym[1])))
        elif sym.isidentifier():
            yield sym
        else:
            yield sym.encode("utf-8").hex()


def _generate_repeat_rules(symbols: list[str]) -> tuple[str, list[_RawGrammarRule]]:
    """Based on a given list of grammar symbols [ symbols ], generate code corresponding to these grammar construction:

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

    with _gencount_lock:
        global _gencount  # noqa: PLW0603
        _gencount += 1
        basename = f"_{_gencount}_" + "_".join(_sanitize_symbols(symbols))

    name = f"{basename}_repeat"
    oname = f"{basename}_items"
    iname = f"{basename}_item"
    symtext = " ".join(symbols)

    with _name_aliases_lock:
        _name_aliases[name] = symbols

    productions: list[_RawGrammarRule] = []
    _ = _rules_decorator

    @_(f"{name} : {oname}")
    def repeat(self: Parser, p: Any) -> Any:
        return getattr(p, oname)

    @_(f"{name} : ")
    def repeat2(self: Parser, p: Any) -> Any:
        return []

    productions.extend(_collect_grammar_rules(repeat))
    productions.extend(_collect_grammar_rules(repeat2))

    @_(f"{oname} : {oname} {iname}")
    def many(self: Parser, p: Any) -> Any:
        items = getattr(p, oname)
        items.append(getattr(p, iname))
        return items

    @_(f"{oname} : {iname}")
    def many2(self: Parser, p: Any) -> Any:
        return [getattr(p, iname)]

    productions.extend(_collect_grammar_rules(many))
    productions.extend(_collect_grammar_rules(many2))

    @_(f"{iname} : {symtext}")
    def item(self: Parser, p: Any) -> Any:
        return tuple(p)

    productions.extend(_collect_grammar_rules(item))
    return name, productions


def _generate_optional_rules(symbols: list[str]) -> tuple[str, list[_RawGrammarRule]]:
    """Based on a given list of grammar symbols [ symbols ], generate code corresponding to these grammar construction:

    .. code-block:: python

        @('optional : symbols')
        def optional(self, p):
            return p.symbols

        @('optional :')
        def optional(self, p):
            return None
    """

    with _gencount_lock:
        global _gencount  # noqa: PLW0603
        _gencount += 1
        basename = f"_{_gencount}_" + "_".join(_sanitize_symbols(symbols))

    name = f"{basename}_optional"
    symtext = " ".join(symbols)

    with _name_aliases_lock:
        _name_aliases[name] = symbols

    productions: list[_RawGrammarRule] = []
    _ = _rules_decorator

    no_values = (None,) * len(symbols)

    @_(f"{name} : {symtext}")
    def optional(self: Parser, p: Any) -> Any:
        return tuple(p)

    @_(f"{name} : ")
    def optional2(self: Parser, p: Any) -> Any:
        return no_values

    productions.extend(_collect_grammar_rules(optional))
    productions.extend(_collect_grammar_rules(optional2))
    return name, productions


def _generate_choice_rules(symbols: list[str]) -> tuple[str, list[_RawGrammarRule]]:
    """Based on a given list of grammar symbols such as [ 'PLUS', 'MINUS' ], generate code corresponding to the
    following construction:

    .. code-block:: python

        @('PLUS', 'MINUS')
        def choice(self, p):
            return p[0]
    """

    with _gencount_lock:
        global _gencount  # noqa: PLW0603
        _gencount += 1
        basename = f"_{_gencount}_" + "_".join(_sanitize_symbols(symbols))

    name = f"{basename}_choice"

    _ = _rules_decorator
    productions: list[_RawGrammarRule] = []

    def choice(self: Parser, p: Any) -> Any:
        return p[0]

    choice.__name__ = name
    choice = _(*symbols)(choice)
    productions.extend(_collect_grammar_rules(choice))
    return name, productions


# endregion


# ============================================================================
# region -------- Parser --------
# ============================================================================


class ParserMetaDict(dict[str, Any]):
    """Special dictionary that allows decorated grammar rule functions to be overloaded."""

    @override
    def __setitem__(self, key: str, value: Any) -> None:
        if (key in self) and callable(value) and hasattr(value, "rules"):
            value.next_func = self[key]
            if not hasattr(value.next_func, "rules"):
                msg = f"Redefinition of {key}. Perhaps an earlier {key} is missing `@_`."
                raise GrammarError(msg)
        super().__setitem__(key, value)

    def __missing__(self, key: str) -> str:
        if key.isupper() and key[:1] != "_":
            return key.upper()
        raise KeyError(key)


def _substitute_decorator(sub: dict[str, str], *extra: dict[str, str]) -> Callable[[CallableT], CallableT]:
    substitutions = [sub, *extra]

    def decorate(func: CallableT) -> CallableT:
        func.substitutions = substitutions  # pyright: ignore # Runtime attribute assignment.
        return func

    return decorate


def _rules_decorator(rule: str, *extra: str) -> Callable[[CallableT], CallableT]:
    rules = [rule, *extra]

    def decorate(func: CallableT) -> CallableT:
        func.rules = [*getattr(func, "rules", []), *rules[::-1]]  # pyright: ignore # Runtime attribute assignment.
        return func

    return decorate


class ParserMeta(type):
    """Metaclass for collecting parsing rules."""

    @override
    @classmethod
    def __prepare__(cls, clsname: str, bases: tuple[type, ...], **kwds: object) -> ParserMetaDict:
        namespace = ParserMetaDict()
        namespace["_"] = _rules_decorator
        namespace["subst"] = _substitute_decorator
        return namespace

    def __new__(cls, clsname: str, bases: tuple[type, ...], namespace: ParserMetaDict, **kwds: object):
        del namespace["_"]
        del namespace["subst"]
        self = super().__new__(cls, clsname, bases, namespace, **kwds)
        self._build(list(namespace.items()))  # pyright: ignore # This method should always exist in Parser subclasses.
        return self


_ConcreteSeqOfStr: TypeAlias = Union[list[str], tuple[str, ...]]
_NestedConcreteSeqOfStr: TypeAlias = Union[list[_ConcreteSeqOfStr], tuple[_ConcreteSeqOfStr, ...]]


class Parser(metaclass=ParserMeta):
    """The class used for recognizing language syntax specified as a context free grammar.

    Attributes
    ----------
    errorok: bool
        Error status.
    given_tokens: Iterator[Token]
        Input tokens.
    lookahead: Optional[Union[Token, YaccSymbol]]
        Current lookahead symbol. Be careful with this.
    """

    # ---- These attributes may be redefined in subclasses.
    if TYPE_CHECKING:
        tokens: ClassVar[set[str]]
        """Lexing tokens. Must be assigned in a subclass by a user."""

        precedence: ClassVar[_NestedConcreteSeqOfStr]
        """Precedence definition as a tuple/list containing tuples/lists of strings. Can be assigned in a subclass by a
        user.
        """

    log = SlyLogger(sys.stderr)
    """Logging object where debugging/diagnostic messages are sent."""

    debugfile: ClassVar[Optional[str]] = None
    """Debugging filename where parsetab.out data can be written."""

    track_positions: ClassVar[bool] = True
    """Whether position information is automatically tracked."""

    error_count: ClassVar[int] = 3
    """The number of symbols that must be shifted to leave recovery mode. Yacc config knob."""

    def __init__(self) -> None:
        self.lookahead: Optional[Union[Token, YaccSymbol]] = None
        self.given_tokens: Iterator[Token] = MISSING
        self.errorok: bool = MISSING

        # ---- Internal bookkeeping attributes
        # Current state
        self.state: int = 0
        # Stack of parsing states
        self.statestack: list[int] = [0]
        # Stack of grammar symbols
        self.symstack: list[YaccSymbol] = [YaccSymbol(type="$end")]
        # Position tracker: id -> lineno
        self._line_positions: dict[int, Optional[int]] = {}
        # Position tracker: id -> (start, end)
        self._index_positions: dict[int, tuple[Optional[int], Optional[int]]] = {}
        # Current production
        self.production: Production = MISSING

    @classmethod
    def __validate_tokens(cls) -> bool:
        if not hasattr(cls, "tokens"):
            cls.log.error("No token list is defined")
            return False

        if not cls.tokens:
            cls.log.error("tokens is empty")
            return False

        if "error" in cls.tokens:
            cls.log.error("Illegal token name 'error'. Is a reserved word")
            return False

        return True

    @classmethod
    def __validate_precedence(cls) -> bool:
        if not hasattr(cls, "precedence"):
            cls.__preclist = []
            return True

        preclist: list[tuple[str, str, int]] = []
        if not isinstance(cls.precedence, (list, tuple)):
            cls.log.error("precedence must be a list or tuple")
            return False

        for level, p in enumerate(cls.precedence, start=1):
            if not isinstance(p, (list, tuple)):
                cls.log.error("Bad precedence table entry %r. Must be a list or tuple", p)
                return False

            if len(p) < 2:
                cls.log.error("Malformed precedence entry %r. Must be (assoc, term, ..., term)", p)
                return False

            if not all(isinstance(term, str) for term in p):
                cls.log.error("precedence items must be strings")
                return False

            assoc = p[0]
            preclist.extend((term, assoc, level) for term in p[1:])

        cls.__preclist = preclist
        return True

    @classmethod
    def __validate_specification(cls) -> bool:
        """Validate various parts of the grammar specification."""

        if not cls.__validate_tokens():
            return False
        return cls.__validate_precedence()

    @classmethod
    def __build_grammar(cls, rules: list[tuple[str, Callable[..., Any]]]) -> None:
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
            except GrammarError as e:
                errors.append(str(e))

        for _name, func in rules:
            try:
                parsed_rule = _collect_grammar_rules(func)
                for pfunc, rulefile, ruleline, prodname, syms in parsed_rule:
                    try:
                        grammar.add_production(prodname, syms, pfunc, rulefile, ruleline)
                    except GrammarError as e:
                        errors.append(str(e))
            except SyntaxError as e:
                errors.append(str(e))
        try:
            grammar.set_start(getattr(cls, "start", None))
        except GrammarError as e:
            errors.append(str(e))

        undefined_symbols = grammar.undefined_symbols()
        for sym, prod in undefined_symbols:
            errors.append(f"{prod.file}:{prod.line}: Symbol {sym!r} used, but not defined as a token or a rule")

        unused_terminals = grammar.unused_terminals()
        if unused_terminals:
            unused_str = "{" + ",".join(unused_terminals) + "}"
            cls.log.warning("Token%s %s defined, but not used", "(s)" if len(unused_terminals) > 1 else "", unused_str)

        unused_rules = grammar.unused_rules()
        for prod in unused_rules:
            cls.log.warning("%s:%d: Rule %r defined, but not used", prod.file, prod.line, prod.name)

        if len(unused_terminals) == 1:
            cls.log.warning("There is 1 unused token")
        if len(unused_terminals) > 1:
            cls.log.warning("There are %d unused tokens", len(unused_terminals))

        if len(unused_rules) == 1:
            cls.log.warning("There is 1 unused rule")
        if len(unused_rules) > 1:
            cls.log.warning("There are %d unused rules", len(unused_rules))

        unreachable = grammar.find_unreachable()
        for u in unreachable:
            cls.log.warning("Symbol %r is unreachable", u)

        if len(undefined_symbols) == 0:
            infinite = grammar.infinite_cycles()
            for inf in infinite:
                errors += f"Infinite recursion detected for symbol {inf!r}\n"

        unused_prec = grammar.unused_precedence()
        for term, assoc in unused_prec:
            errors += f"Precedence rule {assoc!r} defined for unknown symbol {term!r}\n"

        cls._grammar = grammar
        if errors:
            msg = "\n".join(["Unable to build grammar.", *errors])
            raise YaccError(msg)

    @classmethod
    def __build_lrtables(cls) -> bool:
        """Build the LR Parsing tables from the grammar."""

        lrtable = LRTable(cls._grammar)
        num_sr = len(lrtable.sr_conflicts)

        # Report shift/reduce and reduce/reduce conflicts
        if num_sr != getattr(cls, "expected_shift_reduce", None):
            if num_sr == 1:
                cls.log.warning("1 shift/reduce conflict")
            elif num_sr > 1:
                cls.log.warning("%d shift/reduce conflicts", num_sr)

        num_rr = len(lrtable.rr_conflicts)
        if num_rr != getattr(cls, "expected_reduce_reduce", None):
            if num_rr == 1:
                cls.log.warning("1 reduce/reduce conflict")
            elif num_rr > 1:
                cls.log.warning("%d reduce/reduce conflicts", num_rr)

        cls._lrtable = lrtable
        return True

    @classmethod
    def __collect_rules(cls, definitions: list[tuple[str, Any]]) -> list[tuple[str, Callable[..., Any]]]:
        """Collect all of the tagged grammar rules."""

        return [(name, value) for name, value in definitions if callable(value) and hasattr(value, "rules")]

    @classmethod
    def _build(cls, definitions: list[tuple[str, Any]]) -> None:
        """Build the LALR(1) tables. This method is triggered by a metaclass.

        Parameters
        ----------
        definitions: list[tuple[str, Any]]
            A list of (name, item) tuples of all definitions provided in the class, listed in the order in which they
            were defined.
        """

        if vars(cls).get("_build", False):
            return

        # Collect all of the grammar rules from the class definition
        rules = cls.__collect_rules(definitions)

        # Validate other parts of the grammar specification
        if not cls.__validate_specification():
            msg = "Invalid parser specification."
            raise YaccError(msg)

        # Build the underlying grammar object
        cls.__build_grammar(rules)

        # Build the LR tables
        if not cls.__build_lrtables():
            msg = "Can't build parsing tables."
            raise YaccError(msg)

        if cls.debugfile:
            with open(cls.debugfile, "w") as f:
                f.write(str(cls._grammar))
                f.write("\n")
                f.write(str(cls._lrtable))
            cls.log.info("Parser debugging for %s written to %s", cls.__qualname__, cls.debugfile)

    # ----------------------------------------------------------------------
    # Parsing Support. This is the parsing runtime that users use.
    # ----------------------------------------------------------------------
    def error(self, token: Optional[Union[Token, YaccSymbol]]) -> None:
        """Default error handling function. This may be subclassed."""

        if token:
            lineno = getattr(token, "lineno", 0)
            if lineno:
                sys.stderr.write(f"sly: Syntax error at line {lineno}, token={token.type}\n")
            else:
                sys.stderr.write(f"sly: Syntax error, token={token.type}")
        else:
            sys.stderr.write("sly: Parse error in input. EOF\n")

    def errok(self) -> None:
        """Clear the error status."""

        self.errorok = True

    def restart(self) -> None:
        """Force the parser to restart from a fresh state. Clears the statestack."""

        del self.statestack[:]
        del self.symstack[:]
        self.symstack.append(YaccSymbol(type="$end"))
        self.statestack.append(0)
        self.state = 0

    def parse(self, tokens: Iterator[Token]) -> Any:
        """Parse the given input tokens."""

        self.lookahead = None  # Current lookahead symbol
        lookaheadstack: list[Any] = []  # Stack of lookahead symbols
        actions = self._lrtable.lr_action  # Local reference to action table (to avoid lookup on self.)
        goto = self._lrtable.lr_goto  # Local reference to goto table (to avoid lookup on self.)
        prod = self._grammar.Productions  # Local reference to production list (to avoid lookup on self.)
        defaulted_states = self._lrtable.defaulted_states  # Local reference to defaulted states
        pslice = YaccProduction(None)  # Production object passed to grammar rules
        errorcount = 0  # Used during error recovery

        # Set up the state and symbol stacks
        self.given_tokens = tokens
        statestack: list[int] = []  # Stack of parsing states
        self.statestack = statestack
        symstack: list[YaccSymbol] = []  # Stack of grammar symbols
        self.symstack = symstack
        pslice._stack = symstack  # Associate the stack with the production
        self.restart()

        # Set up position tracking
        track_positions = self.track_positions
        if not hasattr(self, "_line_positions"):
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
                        self.lookahead = YaccSymbol(type="$end")

                # Check the action table
                ltype = self.lookahead.type
                t = actions[self.state].get(ltype)
            else:
                t = defaulted_states[self.state]

            if t is not None:
                if t > 0:
                    # shift a symbol on the stack
                    statestack.append(t)
                    self.state = t

                    symstack.append(self.lookahead)
                    self.lookahead = None

                    # Decrease error count on successful shift
                    if errorcount:
                        errorcount -= 1
                    continue

                if t < 0:
                    # reduce a symbol on the stack, emit a production
                    self.production = p = prod[-t]
                    pname = p.name
                    plen = p.len
                    pslice._namemap = p.namemap

                    # Call the production function
                    pslice._slice = symstack[-plen:] if plen else []

                    value = p.func(self, pslice)
                    if value is pslice:
                        value = (pname, *(s.value for s in pslice._slice))

                    sym = YaccSymbol(type=pname, value=value)

                    # Record positions
                    if track_positions:
                        if plen:
                            sym.lineno = symstack[-plen].lineno
                            sym.index = symstack[-plen].index
                            sym.end = symstack[-1].end
                        else:
                            # A zero-length production  (what to put here?)
                            sym.lineno = None
                            sym.index = None
                            sym.end = None

                        self._line_positions[id(value)] = sym.lineno
                        self._index_positions[id(value)] = (sym.index, sym.end)

                    if plen:
                        del symstack[-plen:]
                        del statestack[-plen:]

                    symstack.append(sym)
                    self.state = goto[statestack[-1]][pname]
                    statestack.append(self.state)
                    continue

                if t == 0:
                    n = symstack[-1]
                    result = getattr(n, "value", None)
                    return result  # noqa: RET504

            if t is None:
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
                if errorcount == 0 or self.errorok:
                    errorcount = self.error_count
                    self.errorok = False
                    if self.lookahead.type == "$end":
                        errtoken = None  # End of file!
                    else:
                        errtoken = self.lookahead

                    tok = self.error(errtoken)
                    if tok:
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

                assert self.lookahead

                if len(statestack) <= 1 and self.lookahead.type != "$end":
                    self.lookahead = None
                    self.state = 0
                    # Nuke the lookahead stack
                    del lookaheadstack[:]
                    continue

                # case 2: the statestack has a couple of entries on it, but we're
                # at the end of the file. nuke the top entry and generate an error token

                # Start nuking entries on the stack
                if self.lookahead.type == "$end":
                    # Whoa. We're really hosed here. Bail out
                    return None

                if self.lookahead.type != "error":
                    sym = symstack[-1]
                    if sym.type == "error":
                        # Hmmm. Error is on top of stack, we'll just nuke input symbol and continue
                        self.lookahead = None
                        continue

                    # Create the error symbol for the first time and make it the new lookahead symbol
                    t = YaccSymbol(type="error", value=self.lookahead)
                    if hasattr(self.lookahead, "lineno"):
                        t.lineno = self.lookahead.lineno
                    if hasattr(self.lookahead, "index"):
                        t.index = self.lookahead.index
                    if hasattr(self.lookahead, "end"):
                        t.end = self.lookahead.end

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

    # Return position tracking information
    def line_position(self, value: object) -> Optional[int]:
        return self._line_positions[id(value)]

    def index_position(self, value: object) -> tuple[Optional[int], Optional[int]]:
        return self._index_positions[id(value)]


# endregion
