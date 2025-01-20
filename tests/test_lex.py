from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sly import Lexer
from sly.lex import LexerBuildError, LexError, PatternError, Token


if TYPE_CHECKING:
    from sly.types import _


# TODO: Add more tests related to:
#   - Lexer inheritance
#       - `before` usage
#       - deletion of tokens in subclasses
#


class TestBuildErrors:
    """Tests related to errors raised when building lexers."""

    def test_undeclared_token(self):
        with pytest.raises(LexerBuildError) as exc_info:

            class MyLexer(Lexer):
                tokens = {"PLUS"}

                PLUS = r"\+"
                MINUS = "-"

        assert exc_info.value.args[0] == "'MINUS' does not match a name in tokens"

    def test_duplicated_rule_specifiers(self):
        with pytest.raises(AttributeError) as exc_info:

            class MyLexer(Lexer):
                tokens = {"PLUS", "MINUS", "NUMBER"}

                PLUS = r"\+"
                MINUS = "-"
                NUMBER = r"\d+"
                NUMBER = r"\d+"  # noqa: PIE794

        assert exc_info.value.args[0] == "Name 'NUMBER' redefined."

    def test_undefined_tokens_set(self):
        with pytest.raises(LexerBuildError) as exc_info:

            class MyLexer(Lexer):
                pass

        class_qualname = f"{self.__class__.__name__}.test_undefined_tokens_set.<locals>.MyLexer"

        assert exc_info.value.args[0] == f"{class_qualname} class does not define a tokens attribute."

    def test_invalid_regex(self):
        with pytest.raises(PatternError) as exc_info:

            class MyLexer(Lexer):
                tokens = {"PLUSPLUS"}

                PLUSPLUS = "++"

        assert exc_info.value.args[0] == "Invalid regex for token PLUSPLUS."

    def test_regex_that_matches_empty(self):
        with pytest.raises(PatternError) as exc_info:

            class MyLexer(Lexer):
                tokens = {"ANYTHING"}

                ANYTHING = ".*"

        assert exc_info.value.args[0] == "Regex for token ANYTHING matches empty input."

    def test_non_str_ignore_specifier(self):
        with pytest.raises(LexerBuildError) as exc_info:

            class MyLexer(Lexer):
                tokens = {"PLUS"}

                ignore = {" ", "\t"}

                PLUS = r"\+"

        assert exc_info.value.args[0] == "ignore specifier must be a string."

    def test_invalid_state_class(self):
        class InvalidLexerB:
            pass

        class LexerA(Lexer):
            tokens = {"PLUS", "MINUS", "NUMBER"}

            ignore = " \t"
            ignore_newline = "\n"

            PLUS = r"\+"
            MINUS = "-"
            NUMBER = r"\d+"

            @_(r"[ \t]*\#")
            def DIRECTIVE(self, t: Token):
                self.push_state(InvalidLexerB)

        source = "1 + 2 - 3\n# pragma ...\n"

        lexer = LexerA()
        with pytest.raises(TypeError) as exc_info:
            _ = list(lexer.tokenize(source))

        assert exc_info.value.args[0] == "state must be a subclass of Lexer."

    def test_invalid_multi_character_literal(self):
        with pytest.raises(LexerBuildError) as exc_info:

            class MyLexer(Lexer):
                tokens = {NAME}

                literals = {"++", "-"}

                ignore = " \t"

                NAME = r"[a-zA-Z]+"

        assert exc_info.value.args[0] == "literals must each only be a single character."


def test_empty_with_defined_tokens():
    class MyLexer(Lexer):
        tokens = {"PLUS", "MINUS", "NUMBER"}


def test_add_action_for_predefined_rule_specifier():
    class MyLexer(Lexer):
        tokens = {"PLUS", "MINUS", "NUMBER"}

        PLUS = r"\+"
        MINUS = "-"
        NUMBER = r"\d+"

        def NUMBER(self, t: Token) -> None:
            raise NotImplementedError


def test_override_rule_specifier_with_callable_1():
    class MyLexer(Lexer):
        tokens = {"PLUS", "MINUS", "NUMBER"}

        PLUS = r"\+"
        MINUS = "-"
        NUMBER = r"\d+"

        @_(r"\w+")
        def NUMBER(self, t: Token) -> None:
            raise NotImplementedError


def test_override_rule_specifier_with_callable_2():
    class MyLexer(Lexer):
        tokens = {"PLUS", "MINUS", "NUMBER"}

        PLUS = r"\+"
        MINUS = "-"

        @_(r"\d+")
        def NUMBER(self, t: Token) -> None:
            raise NotImplementedError

        @_(r"\w+")
        def NUMBER(self, t: Token) -> None:
            raise NotImplementedError


def test_default_error_on_invalid_token():
    class MyLexer(Lexer):
        tokens = {PLUS, MINUS, NUMBER}

        PLUS = r"\+"
        MINUS = "-"
        NUMBER = r"\d+"

        ignore = " \t"

    source = "1 - 2 + 3 - a"

    with pytest.raises(LexError) as exc_info:
        _ = list(MyLexer().tokenize(source))

    exc = exc_info.value
    assert exc.args[0] == "Illegal character 'a' at index 12."
    assert exc.text == "a"
    assert exc.error_index == 12


def test_state_switching():
    class LexerA(Lexer):
        tokens = {NAME, NUMBER, LBRACE}

        ignore = " \t"

        NAME = r"[a-zA-Z]+"
        NUMBER = r"\d+"

        @_(r"\{")
        def LBRACE(self, t):
            self.begin(LexerB)
            return t

    class LexerB(Lexer):
        tokens = {PLUS, MINUS, RBRACE}

        ignore = " \t"

        PLUS = r"\+"
        MINUS = r"-"

        @_(r"\}")
        def RBRACE(self, t):
            self.begin(LexerA)
            return t

    source = "a 1 {+ -}"

    lexer = LexerA()
    token_gen = lexer.tokenize(source)

    for _ in range(2):
        tok = next(token_gen)
        assert tok.type in {"NAME", "NUMBER", "RBRACE"}
        assert lexer.__class__ is LexerA

    for _ in range(3):
        tok = next(token_gen)
        assert tok.type in {"LBRACE", "PLUS", "MINUS"}
        assert lexer.__class__ is LexerB

    tok = next(token_gen)
    assert tok.type == "RBRACE"
    assert lexer.__class__ is LexerA

    with pytest.raises(StopIteration):
        next(token_gen)


class CalcLexer(Lexer):
    # Set of token names. This is always required.
    tokens = {"ID", "NUMBER", "PLUS", "MINUS", "TIMES", "DIVIDE", "ASSIGN", "LT", "LE"}

    literals = {"(", ")"}

    # String containing ignored characters between tokens
    ignore = " \t"

    # Regular expression rules for tokens
    # fmt: off
    ID      = r"[a-zA-Z_][a-zA-Z0-9_]*"
    PLUS    = r"\+"
    MINUS   = r"-"
    TIMES   = r"\*"
    DIVIDE  = r"/"
    ASSIGN  = r"="
    LE      = r"<="
    LT      = r"<"
    # fmt: on

    @_(r"\d+")
    def NUMBER(self, t: Token):
        t.value = int(t.value)
        return t

    # Ignored text
    ignore_comment = r"\#.*"

    @_(r"\n+")
    def newline(self, t: Token):
        self.lineno += len(t.value)

    # Attached rule
    def ID(self, t: Token):
        t.value = t.value.upper()
        return t

    def error(self, t: Token):
        self.errors.append(t.value)
        self.index += 1
        if hasattr(self, "return_error"):
            return t
        return None

    def __init__(self):
        self.errors: list[str] = []


class ModernCalcLexer(Lexer):
    # Set of token names. This is always required.
    tokens = {ID, NUMBER, PLUS, MINUS, TIMES, DIVIDE, ASSIGN, LT, LE, IF, ELSE}
    literals = {"(", ")"}

    # String containing ignored characters between tokens
    ignore = " \t"

    # Regular expression rules for tokens
    # fmt: off
    ID          = r"[a-zA-Z_][a-zA-Z0-9_]*"
    ID["if"]    = IF
    ID["else"]  = ELSE

    NUMBER      = r"\d+"
    PLUS        = r"\+"
    MINUS       = r"-"
    TIMES       = r"\*"
    DIVIDE      = r"/"
    ASSIGN      = r"="
    LE          = r"<="
    LT          = r"<"
    # fmt: on

    def NUMBER(self, t: Token):
        t.value = int(t.value)
        return t

    # Ignored text
    ignore_comment = r"\#.*"

    @_(r"\n+")
    def ignore_newline(self, t: Token):
        self.lineno += len(t.value)

    # Attached rule
    def ID(self, t: Token):
        t.value = t.value.upper()
        return t

    def error(self, t: Token):
        self.errors.append(t.value)
        self.index += 1
        if hasattr(self, "return_error"):
            return t
        return None

    def __init__(self):
        self.errors: list[str] = []


@pytest.mark.parametrize("lexer_type", [CalcLexer, ModernCalcLexer])
class TestRuntime:
    def test_tokens(self, lexer_type: type[Lexer]):
        """Test basic recognition of various tokens and literals."""

        lexer = lexer_type()
        toks = list(lexer.tokenize("abc 123 + - * / = < <= ( )"))
        types = [t.type for t in toks]
        vals = [t.value for t in toks]

        assert types == ["ID", "NUMBER", "PLUS", "MINUS", "TIMES", "DIVIDE", "ASSIGN", "LT", "LE", "(", ")"]
        assert vals == ["ABC", 123, "+", "-", "*", "/", "=", "<", "<=", "(", ")"]

    def test_positions(self, lexer_type: type[Lexer]):
        """Test position tracking."""

        lexer = lexer_type()
        text = "abc\n( )"
        toks = list(lexer.tokenize(text))
        lines = [t.lineno for t in toks]
        indices = [t.index for t in toks]
        ends = [t.end for t in toks]
        values = [text[t.index : t.end] for t in toks]

        assert values == ["abc", "(", ")"]
        assert lines == [1, 2, 2]
        assert indices == [0, 4, 6]
        assert ends == [3, 5, 7]

    def test_ignored(self, lexer_type: type[Lexer]):
        """Test ignored comments and newlines."""

        lexer = lexer_type()
        toks = list(lexer.tokenize("\n\n# A comment\n123\nabc\n"))
        types = [t.type for t in toks]
        vals = [t.value for t in toks]
        linenos = [t.lineno for t in toks]

        assert types == ["NUMBER", "ID"]
        assert vals == [123, "ABC"]
        assert linenos == [4, 5]
        assert lexer.lineno == 6

    def test_error(self, lexer_type: type[Lexer]):
        """Test error handling."""

        lexer = lexer_type()
        toks = list(lexer.tokenize("123 :+-"))
        types = [t.type for t in toks]
        vals = [t.value for t in toks]

        assert types == ["NUMBER", "PLUS", "MINUS"]
        assert vals == [123, "+", "-"]
        assert lexer.errors == [":+-"]

    def test_error_return(self, lexer_type: type[Lexer]):
        """Test error token return handling."""

        lexer = lexer_type()
        lexer.return_error = True
        toks = list(lexer.tokenize("123 :+-"))
        types = [t.type for t in toks]
        vals = [t.value for t in toks]

        assert types == ["NUMBER", "ERROR", "PLUS", "MINUS"]
        assert vals == [123, ":+-", "+", "-"]
        assert lexer.errors == [":+-"]
