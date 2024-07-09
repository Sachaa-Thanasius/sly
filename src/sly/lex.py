# region License
# -----------------------------------------------------------------------------
# sly: lex.py
#
# Copyright (C) 2016 - 2018
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

import re
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional, Union

from ._misc import MISSING, CallableT, Self, TypeAlias, override

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


class LexerStateChange(Exception):
    """Exception raised to force a lexing state change."""

    def __init__(self, newstate: type["Lexer"], tok: Optional["Token"] = None):
        self.newstate = newstate
        self.tok = tok


# endregion


# ============================================================================
# region -------- Token Structures --------
# ============================================================================


class Token:
    """Representation of a single token."""

    __slots__ = ("type", "value", "lineno", "index", "end")

    def __init__(self, *, lineno: int, index: int) -> None:
        self.lineno = lineno
        self.index = index

    @override
    def __repr__(self) -> str:
        return (
            f"Token(type={self.type!r}, value={self.value!r}, lineno={self.lineno}, index={self.index}, "
            f'end={getattr(self, "end", -1)})'
        )

    def update(self, *, type: str, value: Any, end: int = MISSING) -> None:  # noqa: A002
        self.type = type
        self.value = value

        # end doesn't always get initialized.
        if end is not MISSING:
            self.end = end


class TokenStr(str):
    def __new__(cls, value: object, key: str, remap: Optional[dict[tuple[str, Any], Any]] = None) -> Self:
        return super().__new__(cls, value)

    def __init__(self, value: object, key: str, remap: Optional[dict[tuple[str, Any], Any]] = None) -> None:
        self.key = key
        self.remap = remap

    def __setitem__(self, key: str, value: str) -> None:
        # Implementation of TOKEN[value] = NEWTOKEN
        if self.remap is not None:
            self.remap[(self.key, key)] = value

    def __delitem__(self, key: str) -> None:
        # Implementation of del TOKEN[value]
        if self.remap is not None:
            self.remap[(self.key, key)] = self.key


class _Before:
    def __init__(self, tok: str, pattern: str) -> None:
        self.tok = tok
        self.pattern = pattern


# endregion


# ============================================================================
# region -------- Lexer --------
# ============================================================================


class LexerMetaDict(dict[str, Any]):
    """Special dictionary that prohibits duplicate definitions in lexer specifications."""

    def __init__(self) -> None:
        self.before: dict[str, str] = {}
        self.delete: list[str] = []
        self.remap: dict[tuple[str, Any], Any] = {}

    @override
    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, str):
            value = TokenStr(value, key, self.remap)

        if isinstance(value, _Before):
            self.before[key] = value.tok
            value = TokenStr(value.pattern, key, self.remap)

        if key in self and not isinstance(value, property):
            prior = self[key]
            if isinstance(prior, str):
                if callable(value):
                    value.pattern = prior
                else:
                    msg = f"Name {key!r} redefined."
                    raise AttributeError(msg)  # noqa: TRY004

        super().__setitem__(key, value)

    @override
    def __delitem__(self, key: str) -> None:
        self.delete.append(key)
        if key not in self and key.isupper():
            return None

        return super().__delitem__(key)

    def __missing__(self, key: str) -> TokenStr:
        if key.split("ignore_")[-1].isupper() and key[:1] != "_":
            return TokenStr(key, key, self.remap)
        raise KeyError


def _match_action_decorator(pattern: str, *extra: str) -> Callable[[CallableT], CallableT]:
    patterns = [pattern, *extra]

    def decorate(func: CallableT) -> CallableT:
        pattern = "|".join(f"({pat})" for pat in patterns)
        old_pattern: str = getattr(func, "pattern", MISSING)

        if old_pattern is not MISSING:
            setattr(func, "pattern", f"{pattern}|{old_pattern}")  # noqa: B010
        else:
            setattr(func, "pattern", pattern)  # noqa: B010
        return func

    return decorate


_TokenMatchAction: TypeAlias = Callable[["Lexer", Token], Optional[Token]]


class LexerMeta(type):
    """Metaclass for collecting lexing rules."""

    _remap: dict[tuple[str, Any], Any]
    _before: dict[str, str]
    _delete: list[str]

    if TYPE_CHECKING:
        # Created by _build().
        _rules: list[tuple[str, Union[str, _TokenMatchAction]]]
        _master_re: re.Pattern[str]

    @override
    @classmethod
    def __prepare__(cls, clsname: str, bases: tuple[type, ...], **kwds: object) -> LexerMetaDict:
        namespace = LexerMetaDict()
        namespace["_"] = _match_action_decorator
        namespace["before"] = _Before
        return namespace

    def __new__(cls, clsname: str, bases: tuple[type, ...], namespace: LexerMetaDict, **kwds: object):
        del namespace["_"]
        del namespace["before"]

        # Create attributes for use in the actual class body
        real_namespace = {str(key): (str(val) if isinstance(val, TokenStr) else val) for key, val in namespace.items()}
        self = super().__new__(cls, clsname, bases, real_namespace, **kwds)

        # Attach various metadata to the class
        self._remap = namespace.remap
        self._before = namespace.before
        self._delete = namespace.delete
        self._build(dict(namespace))  # pyright: ignore # This method should always exist in Lexer subclasses.

        return self


class Lexer(metaclass=LexerMeta):
    """The class used to break input text into a collection of tokens specified by regular expression rules.

    Attributes
    ----------
    text: str
        The text being lexed. Populated via `tokenize()`.
    index: int
        Current index of the lexer within the text.
    lineno: int
        Current line number of the lexer within the text.
    """

    # ---- These attributes may be redefined in subclasses.
    tokens: ClassVar[set[str]] = set()
    """Set of token names. This is always required."""

    literals: ClassVar[set[str]] = set()
    """Characters serving as tokens that are always returned "as is"."""

    ignore: str = ""
    """String containing ignored characters between tokens."""

    reflags: int = 0
    regex_module = re

    _token_names: ClassVar[set[str]] = set()
    _token_funcs: ClassVar[dict[str, Callable[["Lexer", Token], Optional[Token]]]] = {}
    _ignored_tokens: ClassVar[set[str]] = set()
    _remapping: ClassVar[dict[str, dict[str, str]]] = {}
    _delete: ClassVar[list[str]] = []
    _remap: ClassVar[dict[tuple[str, Any], Any]] = {}

    # ---- Internal attributes
    __state_stack: Optional[list[type["Lexer"]]] = None
    __set_state: Optional[Callable[[type["Lexer"]], None]] = None

    def __init__(self) -> None:
        # ---- Public interface
        self.text: str = MISSING
        self.index: int = -1
        self.lineno: int = -1

        # ---- Internal backtracking-related functions
        self.mark: Callable[[], None] = MISSING
        self.accept: Callable[[], None] = MISSING
        self.reject: Callable[[], None] = MISSING

    @classmethod
    def _collect_rules(cls, potential_rules: dict[str, Any]) -> None:
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
        rules: list[tuple[str, Any]] = []

        for base in cls.__bases__:
            if isinstance(base, LexerMeta):
                rules.extend(base._rules)

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
                    existing[key] = value

                elif isinstance(value, TokenStr) and key in cls._before:
                    before = cls._before[key]
                    if before in existing:
                        # Position the token before another specified token.
                        n = rules.index((before, existing[before]))
                        rules.insert(n, (key, value))
                    else:
                        # Put at the end of the rule list
                        rules.append((key, value))
                    existing[key] = value
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
    def _build(cls, potential_rules: dict[str, Any]) -> None:
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

        remapped_toks: set[str] = set()
        for d in cls._remapping.values():
            remapped_toks.update(d.values())

        undefined = remapped_toks - set(cls._token_names)
        if undefined:
            missing = ", ".join(undefined)
            msg = f"{missing} not included in token(s)."
            raise LexerBuildError(msg)

        cls._collect_rules(potential_rules)

        parts: list[str] = []
        for tokname, value in cls._rules:
            if tokname.startswith("ignore_"):
                tokname = tokname[7:]  # noqa: PLW2901
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

        if not parts:
            return

        # Form the master regular expression
        cls._master_re = cls.regex_module.compile("|".join(parts), cls.reflags)

        # Verify that that ignore and literals specifiers match the input type
        if not isinstance(cls.ignore, str):
            msg = "ignore specifier must be a string."
            raise LexerBuildError(msg)

        if not all(isinstance(lit, str) for lit in cls.literals):
            msg = "literals must be specified as strings."
            raise LexerBuildError(msg)

    def begin(self, cls: type["Lexer"]) -> None:
        """Begin a new lexer state."""

        if not isinstance(cls, LexerMeta):
            msg = "state must be a subclass of Lexer."
            raise TypeError(msg)

        if self.__set_state:
            self.__set_state(cls)
        self.__class__ = cls

    def push_state(self, cls: type["Lexer"]) -> None:
        """Push a new lexer state onto the stack."""

        if self.__state_stack is None:
            self.__state_stack = []
        self.__state_stack.append(type(self))
        self.begin(cls)

    def pop_state(self) -> None:
        """Pop a lexer state from the stack."""

        assert self.__state_stack
        self.begin(self.__state_stack.pop())

    def tokenize(self, text: str, lineno: int = 1, index: int = 0) -> Generator[Token]:
        """Tokenize the given text."""

        _ignored_tokens: set[str] = MISSING
        _master_re: re.Pattern[str] = MISSING
        _ignore: str = MISSING
        _token_funcs: dict[str, Callable[[Lexer, Token], Optional[Token]]] = MISSING
        _literals: set[str] = MISSING
        _remapping: dict[str, dict[str, str]] = MISSING

        # ---- Support for state changes
        def _set_state(cls: type[Lexer]) -> None:
            nonlocal _ignored_tokens, _master_re, _ignore, _token_funcs, _literals, _remapping
            _ignored_tokens = cls._ignored_tokens
            _master_re = cls._master_re
            _ignore = cls.ignore
            _token_funcs = cls._token_funcs
            _literals = cls.literals
            _remapping = cls._remapping

        self.__set_state = _set_state
        _set_state(type(self))

        # ---- Support for backtracking
        _mark_stack: list[tuple[type[Self], int, int]] = []

        def _mark() -> None:
            _mark_stack.append((type(self), index, lineno))

        self.mark = _mark

        def _accept() -> None:
            _mark_stack.pop()

        self.accept = _accept

        def _reject() -> None:
            nonlocal index, lineno
            cls, index, lineno = _mark_stack[-1]
            _set_state(cls)

        self.reject = _reject

        # ---- Main tokenization function
        self.text = text
        try:
            while True:
                try:
                    if text[index] in _ignore:
                        index += 1
                        continue
                except IndexError:
                    return

                tok = Token(lineno=lineno, index=index)
                m = _master_re.match(text, index)
                if m:
                    index = m.end()
                    assert m.lastgroup  # The matched group will always have a name.
                    tok.update(type=m.lastgroup, value=m.group(), end=index)

                    if tok.type in _remapping:
                        tok.type = _remapping[tok.type].get(tok.value, tok.type)

                    if tok.type in _token_funcs:
                        self.index = index
                        self.lineno = lineno
                        tok = _token_funcs[tok.type](self, tok)
                        index = self.index
                        lineno = self.lineno
                        if not tok:
                            continue

                    if tok.type in _ignored_tokens:
                        continue

                    yield tok

                else:
                    # No match, see if the character is in literals
                    if text[index] in _literals:
                        value = text[index]
                        tok.update(type=value, value=value, end=index + 1)
                        index += 1
                        yield tok
                    else:
                        # A lexing error
                        self.index = index
                        self.lineno = lineno
                        tok.update(type="ERROR", value=text[index:])
                        tok = self.error(tok)
                        if tok is not None:
                            tok.end = self.index
                            yield tok

                        index = self.index
                        lineno = self.lineno

        # Set the final state of the lexer before exiting (even if exception)
        finally:
            self.text = text
            self.index = index
            self.lineno = lineno

    def error(self, t: Token) -> Optional[Token]:
        """Default implementation of the error handler. May be changed in subclasses."""

        msg = f"Illegal character {t.value[0]!r} at index {self.index}."
        raise LexError(msg, t.value, self.index)


# endregion
