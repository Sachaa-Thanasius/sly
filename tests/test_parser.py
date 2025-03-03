from typing import TYPE_CHECKING, Any

import pytest

from sly import Lexer, Parser
from sly.lex import Token
from sly.yacc import YaccError


if TYPE_CHECKING:
    from sly.types import _


class CalcLexer(Lexer):
    # Set of token names. This is always required.
    tokens = {ID, NUMBER, PLUS, MINUS, TIMES, DIVIDE, ASSIGN, COMMA}
    literals = {"(", ")"}

    # String containing ignored characters between tokens
    ignore = " \t"

    # Regular expression rules for tokens
    ID = r"[a-zA-Z_][a-zA-Z0-9_]*"
    PLUS = r"\+"
    MINUS = r"-"
    TIMES = r"\*"
    DIVIDE = r"/"
    ASSIGN = r"="
    COMMA = r","

    @_(r"\d+")
    def NUMBER(self, t: Token):
        t.value = int(t.value)
        return t

    # Ignored text
    ignore_comment = r"\#.*"

    @_(r"\n+")
    def newline(self, t: Token):
        self.lineno += t.value.count("\n")

    def error(self, t: Token):
        self.errors.append(t.value[0])
        self.index += 1

    def __init__(self):
        self.errors: list[str] = []


class TestBuildErrors:
    def test_no_rules(self):
        with pytest.raises(YaccError) as exc_info:

            class MyParser(Parser):
                tokens = {PLUS, MINUS, STRING}

        assert exc_info.value.args[0] == "No grammar rules are defined."

    def test_invalid_precedence_type(self):
        with pytest.raises(YaccError) as exc_info:

            class MyParser(Parser):
                tokens = {PLUS, MINUS, STRING}

                precedence = {("left", PLUS, MINUS)}

        assert exc_info.value.args[0] == "Invalid parser specification\nprecedence must be a list or tuple"

    def test_invalid_precedence_nested_type(self):
        with pytest.raises(YaccError) as exc_info:

            class MyParser(Parser):
                tokens = {PLUS, MINUS, STRING}

                precedence = [{"left": (PLUS, MINUS)}]

        assert exc_info.value.args[0] == (
            "Invalid parser specification\n"
            "Bad precedence table entry {'left': ('PLUS', 'MINUS')}. Must be a list or tuple"
        )

    def test_duplicate_precedence_values(self):
        with pytest.raises(YaccError) as exc_info:

            class MyParser(Parser):
                tokens = {PLUS, STRING}

                precedence = [("left", PLUS), ("left", PLUS)]

                @_("STRING PLUS")
                def expr(self, p):
                    pass

        assert exc_info.value.args[0] == "Unable to build grammar.\nPrecedence already specified for terminal 'PLUS'."

    def test_error_check_after_adding_productions(self):
        with pytest.raises(YaccError) as exc_info:

            class MyParser(Parser):
                tokens = {PLUS, STRING}

                @_('STRING "--" PLUS')
                def expr(self, p):
                    pass

        assert exc_info.value.args[0].startswith("Unable to build grammar - no grammar rules were valid.\n")

    def test_too_long_literal(self):
        lineno = 0

        with pytest.raises(YaccError) as exc_info:

            class MyParser(Parser):
                tokens = {PLUS, STRING}

                @_('STRING "--" PLUS')
                def expr(self, p):
                    pass

                nonlocal lineno
                lineno = expr.__code__.co_firstlineno

        assert exc_info.value.args[0].endswith(
            f"{__file__}:{lineno}: Literal token \"--\" in rule 'expr' may only be a single character."
        )

        with pytest.raises(YaccError) as exc_info:

            class MyParser(Parser):
                tokens = {PLUS, STRING}

                @_("STRING PLUS")
                def expr(self, p):
                    pass

                @_('STRING "--" PLUS')
                def expr(self, p):
                    pass

                nonlocal lineno
                lineno = expr.__code__.co_firstlineno

        assert exc_info.value.args[0].endswith(
            f"{__file__}:{lineno}: Literal token \"--\" in rule 'expr' may only be a single character."
        )


class CalcParser(Parser):
    tokens = CalcLexer.tokens

    precedence = (
        ("left", PLUS, MINUS),
        ("left", TIMES, DIVIDE),
        ("right", UMINUS),
    )

    def __init__(self):
        self.names = {}
        self.errors: list[Token] = []

    @_("ID ASSIGN expr")
    def statement(self, p: Any):
        self.names[p.ID] = p.expr

    @_('ID "(" [ arglist ] ")"')
    def statement(self, p: Any):
        return (p.ID, p.arglist)

    @_("expr { COMMA expr }")
    def arglist(self, p: Any):
        return [p.expr0, *p.expr1]

    @_("expr")
    def statement(self, p: Any):
        return p.expr

    @_("expr PLUS expr")
    def expr(self, p: Any):
        return p.expr0 + p.expr1

    @_("expr MINUS expr")
    def expr(self, p: Any):
        return p.expr0 - p.expr1

    @_("expr TIMES expr")
    def expr(self, p: Any):
        return p.expr0 * p.expr1

    @_("expr DIVIDE expr")
    def expr(self, p: Any):
        return p.expr0 / p.expr1

    @_("MINUS expr %prec UMINUS")
    def expr(self, p: Any):
        return -p.expr

    @_('"(" expr ")"')
    def expr(self, p: Any):
        return p.expr

    @_("NUMBER")
    def expr(self, p: Any):
        return p.NUMBER

    @_("ID")
    def expr(self, p: Any):
        try:
            return self.names[p.ID]
        except LookupError:
            self.errors.append(("undefined", p.ID))
            return 0

    def error(self, token: Token):
        self.errors.append(token)


def test_simple():
    """Test basic recognition of various tokens and literals."""
    lexer = CalcLexer()
    parser = CalcParser()

    result = parser.parse(lexer.tokenize("a = 3 + 4 * (5 + 6)"))
    assert result is None
    assert parser.names["a"] == 47

    result = parser.parse(lexer.tokenize("3 + 4 * (5 + 6)"))
    assert result == 47


def test_ebnf():
    lexer = CalcLexer()
    parser = CalcParser()
    result = parser.parse(lexer.tokenize("a()"))
    assert result == ("a", None)

    result = parser.parse(lexer.tokenize("a(2+3)"))
    assert result == ("a", [5])

    result = parser.parse(lexer.tokenize("a(2+3, 4+5)"))
    assert result == ("a", [5, 9])


def test_parse_error():
    lexer = CalcLexer()
    parser = CalcParser()

    result = parser.parse(lexer.tokenize("a 123 4 + 5"))
    assert result == 9
    assert len(parser.errors) == 1
    assert parser.errors[0].type == "NUMBER"
    assert parser.errors[0].value == 123


# TODO:  Add tests
# - error productions
# - embedded actions
# - lineno tracking
# - various error cases caught during parser construction
