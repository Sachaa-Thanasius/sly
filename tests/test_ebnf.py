from typing import TYPE_CHECKING, Any

from sly import Lexer, Parser
from sly.lex import Token

if TYPE_CHECKING:
    from sly.types import _


class CalcLexer(Lexer):
    # Set of token names. This is always required.
    tokens = {ID, NUMBER, PLUS, MINUS, TIMES, DIVIDE, ASSIGN, COMMA}
    literals = {"(", ")"}

    # String containing ignored characters between tokens.
    ignore = " \t"

    # Regular expression rules for tokens.
    ID = r"[a-zA-Z_][a-zA-Z0-9_]*"
    PLUS = r"\+"
    MINUS = r"-"
    TIMES = r"\*"
    DIVIDE = r"/"
    ASSIGN = r"="
    COMMA = r","

    @_(r"\d+")
    def NUMBER(self, t: Token) -> Token:
        t.value = int(t.value)
        return t

    # Ignored text
    ignore_comment = r"\#.*"

    @_(r"\n+")
    def newline(self, t: Token) -> None:
        self.lineno += t.value.count("\n")

    def error(self, t: Token) -> None:
        self.errors.append(t.value[0])
        self.index += 1

    def __init__(self) -> None:
        self.errors: list[str] = []


class CalcParser(Parser):
    tokens = CalcLexer.tokens

    def __init__(self) -> None:
        self.names: dict[str, Any] = {}
        self.errors: list[str] = []

    @_("ID ASSIGN expr")
    def statement(self, p: Any) -> None:
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

    @_("term { PLUS|MINUS term }")
    def expr(self, p: Any):
        lval = p.term0
        for op, rval in p[1]:
            if op == "+":
                lval = lval + rval
            elif op == "-":
                lval = lval - rval
        return lval

    @_("factor { TIMES|DIVIDE factor }")
    def term(self, p: Any):
        lval = p.factor0
        for op, rval in p[1]:
            if op == "*":
                lval = lval * rval
            elif op == "/":
                lval = lval / rval
        return lval

    @_("MINUS factor")
    def factor(self, p: Any):
        return -p.factor

    @_("'(' expr ')'")
    def factor(self, p: Any):
        return p.expr

    @_("NUMBER")
    def factor(self, p: Any):
        return int(p.NUMBER)

    @_("ID")
    def factor(self, p: Any):
        try:
            return self.names[p.ID]
        except LookupError:
            print(f"Undefined name {p.ID!r}")
            return 0

    def error(self, token):
        self.errors.append(token)


# Test basic recognition of various tokens and literals
def test_simple():
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
    print(type(parser.errors[0]))
    assert parser.errors[0].type == "NUMBER"
    assert parser.errors[0].value == 123


# TODO: Add tests
# - error productions
# - embedded actions
# - lineno tracking
# - various error cases caught during parser construction
