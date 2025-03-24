# region License
# -----------------------------------------------------------------------------
# sly: lex.py
#
# Copyright (C) 2024, Sachaa-Thanasius
# Copyright (C) 2016 - 2018
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

import re

from . import _typing_compat as _t


__all__ = ("Lexer",)


# ============================================================================
# region -------- Exceptions --------
# ============================================================================


class LexError(Exception):
    """Exception raised if an invalid character is encountered and no default error handler function is defined.

    Parameters
    ----------
    message: str
        The message to put in the exception.
    text: str
        All remaining untokenized text.
    error_index: int
        The index location of the error.

    Attributes
    ----------
    text: str
        All remaining untokenized text.
    error_index: int
        The index location of the error.
    """

    def __init__(self, message: str, text: str, error_index: int) -> None:
        super().__init__(message)
        self.text = text
        self.error_index = error_index


class PatternError(Exception):
    """Exception raised if there's some kind of problem with the specified regex patterns in the lexer."""


class LexerBuildError(Exception):
    """Exception raised if there's some sort of problem building the lexer."""


# endregion


# ============================================================================
# region -------- Token structures --------
# ============================================================================


class Token:
    """Representation of a single token."""

    __slots__ = ("type", "value", "lineno", "index", "end")

    def __init__(self, type: str, value: _t.Any, lineno: int, index: int, end: int = -1):  # noqa: A002
        self.type: str = type
        self.value: _t.Any = value
        self.lineno: int = lineno
        self.index: int = index
        self.end: int = end

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"type={self.type!r}, value={self.value!r}, lineno={self.lineno!r}, index={self.index!r}, end={self.end}"
            ")"
        )


class TokenStr(str):
    __slots__ = ("key", "remap")

    key: str
    remap: dict[tuple[str, str], str]

    def __new__(cls, value: object, key: str, remap: dict[tuple[str, str], str]) -> _t.Self:
        self = super().__new__(cls, value)
        self.key = key
        self.remap = remap
        return self

    def __setitem__(self, key: str, value: str, /) -> None:
        # Implementation of TOKEN[value] = NEWTOKEN
        self.remap[(self.key, key)] = value

    def __delitem__(self, key: str, /) -> None:
        # Implementation of del TOKEN[value]
        self.remap[(self.key, key)] = self.key


class _Before:
    __slots__ = ("tok", "pattern")

    def __init__(self, tok: str, pattern: str) -> None:
        self.tok = tok
        self.pattern = pattern


# endregion


# ============================================================================
# region -------- Lexer --------
# ============================================================================


class LexerMetaDict(dict[str, object]):
    """Special dictionary that prohibits duplicate definitions in lexer specifications."""

    __slots__ = ("before", "delete", "remap")

    def __init__(self) -> None:
        self.before: dict[str, str] = {}
        self.delete: list[str] = []
        self.remap: dict[tuple[str, str], str] = {}

    def __setitem__(self, key: str, value: _t.Any, /) -> None:
        if isinstance(value, str):
            value = TokenStr(value, key, self.remap)

        if isinstance(value, _Before):
            self.before[key] = value.tok
            value = TokenStr(value.pattern, key, self.remap)

        if key in self and not isinstance(value, property):
            prior = self[key]
            if isinstance(prior, str):
                if callable(value):
                    value.pattern = prior  # pyright: ignore [reportFunctionMemberAccess]
                else:
                    msg = f"Name {key!r} redefined."
                    raise AttributeError(msg)  # noqa: TRY004

        super().__setitem__(key, value)

    def __delitem__(self, key: str, /) -> None:
        self.delete.append(key)
        if key not in self and key.isupper():
            return None
        else:
            return super().__delitem__(key)

    def __missing__(self, key: str, /) -> TokenStr:
        if key.split("ignore_")[-1].isupper() and key[:1] != "_":
            return TokenStr(key, key, self.remap)
        else:
            raise KeyError(key)


def _match_action_decorator(pattern: str, *extra: str) -> _t.Callable[[_t.CallableT], _t.CallableT]:
    patterns = [pattern, *extra]

    def decorate(func: _t.CallableT) -> _t.CallableT:
        pattern = "|".join(f"({pat})" for pat in patterns)

        # Runtime attribute assignment.
        try:
            old_pattern: str = func.pattern  # pyright: ignore [reportFunctionMemberAccess] # Guarded.
        except AttributeError:
            func.pattern = pattern  # pyright: ignore [reportFunctionMemberAccess]
        else:
            func.pattern = f"({pattern})|({old_pattern})"  # pyright: ignore [reportFunctionMemberAccess]

        return func

    return decorate


_TokenMatchAction: _t.TypeAlias = "_t.Callable[[Lexer, Token], _t.Optional[Token]]"


class LexerMeta(type):
    """Metaclass for collecting lexing rules."""

    @classmethod
    def __prepare__(cls, name: str, bases: tuple[type, ...], /, **kwargs: _t.Any) -> LexerMetaDict:
        namespace = LexerMetaDict()
        namespace["_"] = _match_action_decorator
        namespace["before"] = _Before
        return namespace

    def __new__(cls, name: str, bases: tuple[type, ...], namespace: LexerMetaDict, /, **kwargs: _t.Any):
        del namespace["_"]
        del namespace["before"]

        # Create attributes for use in the actual class body
        final_namespace = {str(key): (str(val) if isinstance(val, TokenStr) else val) for key, val in namespace.items()}
        return super().__new__(cls, name, bases, final_namespace, **kwargs)

    def __init__(self, name: str, bases: tuple[type, ...], namespace: LexerMetaDict, /, **kwargs: _t.Any) -> None:
        super().__init__(name, bases, namespace, **kwargs)

        # Attach various metadata to the class
        self._remap: dict[tuple[str, str], str] = namespace.remap
        self._before: dict[str, str] = namespace.before
        self._delete: list[str] = namespace.delete


class Lexer(metaclass=LexerMeta):
    """The class used to break input text into a collection of tokens specified by regular expression rules.

    If a subclass overrides the constructor, it must invoke the base class constructor (`Lexer.__init__()`) before
    doing anything else to the lexer.

    Attributes
    ----------
    text: str
        The text being lexed. Populated via `tokenize()`.
    lineno: int
        Current line number of the lexer within the text.
    index: int
        Current index of the lexer within the text.
    """

    __slots__ = ("text", "index", "lineno", "_mark_stack", "__state_stack")

    # ---- Public class attributes.
    tokens: _t.ClassVar[set[str]] = set()
    """Set of token names. Must be defined in a subclass."""

    literals: _t.ClassVar[set[str]] = set()
    """Characters serving as tokens that are always returned "as is"."""

    ignore: _t.ClassVar[str] = ""
    """String containing ignored characters between tokens."""

    reflags: _t.ClassVar[int] = 0
    """Optional flags to supply to the used regex compiler. Equivalent to the flags parameter in `re` functions."""

    regex_module = re
    """The regex module to use as the regex compiler. Defaults to `re`."""

    # ---- Internal attributes
    # These two are created by _build(), which is called in __init_subclass__().
    _rules: _t.ClassVar[list[tuple[str, _t.Union[str, _TokenMatchAction]]]]
    _master_re: _t.ClassVar[re.Pattern[str]]

    _token_names: _t.ClassVar[set[str]] = set()
    _token_funcs: _t.ClassVar[dict[str, _TokenMatchAction]] = {}
    _ignored_tokens: _t.ClassVar[set[str]] = set()
    _remapping: _t.ClassVar[dict[str, dict[str, str]]] = {}

    def __init_subclass__(cls, /, **kwargs: _t.Any) -> None:
        """Collect the lexing rules and build the master regular expression."""

        super().__init_subclass__(**kwargs)
        cls._build(vars(cls).copy())

    @classmethod
    def _collect_rules(cls, potential_rules: dict[str, _t.Any]) -> None:
        """Collect all of the rules from class definitions that look like token information.

        Notes
        -----
        There are a few things that govern this:

        1.  Any definition of the form `NAME = str` is a token if `NAME` is
            defined in the tokens set.
        2.  Any definition of the form `ignore_NAME = str` is a rule for an ignored
            token.
        3.  Any function defined with a `.pattern` attribute is treated as a rule.
            Such functions can be created with the `@_` decorator or by defining
            function with the same name as a previously defined string.

        This function is responsible for keeping rules in order.
        """

        # Collect all previous rules from base classes
        rules: list[tuple[str, _t.Any]] = [
            rule for base in cls.__bases__ if (base is not Lexer and issubclass(base, Lexer)) for rule in base._rules
        ]

        # Dictionary of previous rules
        existing = dict(rules)

        for key, value in potential_rules.items():
            if (key in cls._token_names) or key.startswith("ignore_") or hasattr(value, "pattern"):
                if callable(value) and not hasattr(value, "pattern"):
                    msg = f"function {value} doesn't have a regex pattern."
                    raise LexerBuildError(msg)

                if key in existing:
                    # The definition matches something that already existed in the base class.
                    # We replace it, but keep the original ordering.
                    n = rules.index((key, existing[key]))
                    rules[n] = (key, value)

                elif isinstance(value, TokenStr) and key in cls._before:
                    before = cls._before[key]
                    if before in existing:
                        # Position the token before another specified token.
                        n = rules.index((before, existing[before]))
                        rules.insert(n, (key, value))
                    else:
                        # Put at the end of the rule list
                        rules.append((key, value))

                else:
                    rules.append((key, value))

                existing[key] = value

            elif isinstance(value, str) and not key.startswith("_") and key not in {"ignore", "literals"}:
                msg = f"{key!r} does not match a name in tokens"
                raise LexerBuildError(msg)

        # Apply deletion rules.
        rules = [(key, value) for key, value in rules if key not in cls._delete]
        cls._rules = rules

    @classmethod
    def _build(cls, potential_rules: dict[str, _t.Any]) -> None:
        """Build the lexer object from the collected tokens and regular expressions, and validate them as sane."""

        if "tokens" not in vars(cls):
            msg = f"{cls.__qualname__} class does not define a tokens attribute."
            raise LexerBuildError(msg)

        # Pull definitions created for any parent classes
        cls._token_names = cls._token_names | set(cls.tokens)
        cls._ignored_tokens = set(cls._ignored_tokens)
        cls._token_funcs = dict(cls._token_funcs)
        cls._remapping = dict(cls._remapping)

        for (key, val), newtok in cls._remap.items():
            if key not in cls._remapping:
                cls._remapping[key] = {}
            cls._remapping[key][val] = newtok

        remapped_toks: set[str] = {val for d in cls._remapping.values() for val in d.values()}

        undefined = remapped_toks - cls._token_names
        if undefined:
            msg = f"{', '.join(undefined)} not included in token(s)."
            raise LexerBuildError(msg)

        cls._collect_rules(potential_rules)

        parts: list[str] = []
        for tokname, value in cls._rules:
            if tokname.startswith("ignore_"):
                tokname = tokname.removeprefix("ignore_")  # noqa: PLW2901
                cls._ignored_tokens.add(tokname)

            if isinstance(value, str):
                pattern = value
            elif callable(value):
                cls._token_funcs[tokname] = value
                pattern = value.pattern  # pyright: ignore [reportFunctionMemberAccess]
            else:
                msg = f"{value!r} is not a valid rule; it should be a string or a callable."
                raise LexerBuildError(msg)

            # Form the regular expression component
            part = f"(?P<{tokname}>{pattern})"

            # Make sure the individual regex compiles properly
            try:
                cpat = cls.regex_module.compile(part, cls.reflags)
            except Exception as exc:
                msg = f"Invalid regex for token {tokname}."
                raise PatternError(msg) from exc

            # Verify that the pattern doesn't match the empty string
            if cpat.match(""):
                msg = f"Regex for token {tokname} matches empty input."
                raise PatternError(msg)

            parts.append(part)

        # TODO: Is this conditional a result of _build() originally being called in the metaclass?
        # Can we remove it now?
        if not parts:
            return

        # Form the master regular expression
        cls._master_re = cls.regex_module.compile("|".join(parts), cls.reflags)

        # Verify that that ignore and literals specifiers match the input type
        if not isinstance(cls.ignore, str):
            msg = "ignore specifier must be a string."
            raise LexerBuildError(msg)

        for lit in cls.literals:
            if not isinstance(lit, str):
                msg = "literals must be specified as strings."
                raise LexerBuildError(msg)

            if len(lit) != 1:
                msg = "literals must each only be a single character."
                raise LexerBuildError(msg)

    def __init__(self, text: str = "", lineno: int = 1, index: int = 0) -> None:
        # ---- Public interface
        self.text: str = text
        self.lineno: int = lineno
        self.index: int = index

        # ---- Internal state
        self._mark_stack: list[tuple[int, int]] = []
        self.__state_stack: list[type[Lexer]] = []

    def __iter__(self, /):
        return self

    def __next__(self, /) -> Token:
        while True:
            # Case 1: Skip ignored characters.
            # At the same time, check to see if we are beyond the bounds the text and thus are done.
            try:
                if self.text[self.index] in self.ignore:
                    self.index += 1
                    continue
            except IndexError:
                raise StopIteration from None

            # Case 2: Match a specified token and call its action.
            if m := self._master_re.match(self.text, self.index):
                assert m.lastgroup is not None, "There should always be a matched named group."

                tok = Token(m.lastgroup, m.group(), self.lineno, self.index, m.end())
                self.index = tok.end

                if tok.type in self._remapping:
                    tok.type = self._remapping[tok.type].get(tok.value, tok.type)

                if tok.type in self._token_funcs:
                    tok = self._token_funcs[tok.type](self, tok)

                    if tok is None:
                        continue

                if tok.type in self._ignored_tokens:
                    continue

                return tok

            # Case 3: Match a specified character literal.
            elif (value := self.text[self.index]) in self.literals:
                tok = Token(value, value, self.lineno, self.index, self.index + 1)
                self.index += 1
                return tok

            # Case 4: Handle lexing errors by either spitting out a replacement token or moving on.
            else:
                tok = Token("ERROR", self.text[self.index :], self.lineno, self.index)
                tok = self.error(tok)
                if tok is not None:
                    tok.end = self.index
                    return tok

        msg = "Should be unreachable."
        raise RuntimeError(msg)

    def tokenize(self, text: str, lineno: int = 1, index: int = 0) -> _t.Iterator[Token]:
        """Tokenize the given text."""

        self.text = text
        self.lineno = lineno
        self.index = index

        return self

    def error(self, t: Token) -> _t.Optional[Token]:
        """Default implementation of the error handler. This may be overridden in subclasses."""

        msg = f"Illegal character {t.value[0]!r} at index {self.index}."
        raise LexError(msg, t.value, self.index)

    def mark(self) -> None:
        """Mark the current position to potentially backtrack to."""

        self._mark_stack.append((self.index, self.lineno))

    def accept(self) -> None:
        """Accept the current position."""

        self._mark_stack.pop()

    def reject(self) -> None:
        """Reject the current position and backtrack to the saved position."""

        self.index, self.lineno = self._mark_stack[-1]

    def begin(self, state: type[Lexer]) -> None:
        """Begin a new lexer state.

        Raises
        ------
        TypeError
            If `state` is not a subclass of `Lexer`.
        """

        if not issubclass(state, Lexer):
            msg = "state must be a subclass of Lexer."
            raise TypeError(msg)

        self.__class__ = state  # pyright: ignore [reportAttributeAccessIssue]

    def push_state(self, state: type[Lexer]) -> None:
        """Push a new lexer state onto the stack."""

        self.__state_stack.append(type(self))
        self.begin(state)

    def pop_state(self) -> None:
        """Pop a lexer state from the stack."""

        self.begin(self.__state_stack.pop())


# endregion
