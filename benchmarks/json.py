# ruff: noqa: F811, F821, RUF012, ANN201
# pyright: basic, reportUndefinedVariable=none, reportRedeclaration=none

"""Simple benchmark adapted from python-parsing-benchmarks.

Run with the following commands::

    python -m pip install -U yelp-gprof2dot
    python -m cProfile -o benchmarks/log.pstats -m example.json run
    gprof2dot benchmarks/log.pstats [-z <module_name>:<line_no>:<function_name>] | dot -Tsvg -o benchmarks/log.svg
"""

from __future__ import annotations

from sly import Lexer, Parser
from sly.lex import Token
from sly.yacc import YaccProduction as Prod


TYPE_CHECKING = False

if TYPE_CHECKING:
    from sly.types import _


class JsonLexer(Lexer):
    tokens = {STRING, NUMBER, TRUE, FALSE, NULL}
    ignore = " \t\n\r"
    literals = {"{", "}", "[", "]", ":", ","}

    @_(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([Ee][+-]?[0-9]+)?")
    def NUMBER(self, t: Token):
        t.value = float(t.value)
        return t

    STRING = r'"([ !#-\[\]-\U0010ffff]+|\\(["\/\\bfnrt]|u[0-9A-Fa-f]{4}))*"'

    @_(r"true")
    def TRUE(self, t: Token):
        t.value = True
        return t

    @_(r"false")
    def FALSE(self, t: Token):
        t.value = False
        return t

    @_(r"null")
    def NULL(self, t: Token):
        t.value = None
        return t


class JsonParser(Parser):
    tokens = JsonLexer.tokens
    start = "value"

    @_(r'"{" [ pairs ] "}"')
    def value(self, p: Prod):
        if p.pairs:
            return dict(p.pairs)
        else:
            return {}

    @_(r'pair { "," pair }')
    def pairs(self, p: Prod):
        return [p.pair0, *p.pair1]

    @_(r'STRING ":" value')
    def pair(self, p: Prod):
        return (p.STRING, p.value)

    @_(r'value { "," value }')
    def items(self, p: Prod):
        return [p.value0, *p.value1]

    @_(r'"[" [ items ] "]"')
    def value(self, p: Prod):
        if p.items:
            return p.items
        else:
            return []

    @_("STRING", "NUMBER", "TRUE", "FALSE", "NULL")
    def value(self, p: Prod):
        return p[0]

    def error(self, token):  # noqa: ANN001
        raise ValueError(token)


lexer = JsonLexer()
parser = JsonParser()

obj = [
    r"""
{"true": true,
 "false": false,
 "null": null,
 "integer": -123,
 "float": 123.456e-7,
 "string": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
 "escaped": "this is a quote: \" and this is a slash: \\",
 "unicode": "この字は日本語の字だ\n這些字是中文字\nEstas son palabras en español",
 "escaped unicode": "\u3053\u306e\u5b57\u306f\u65e5\u672c\u8a9e\u306e\u5b57\u3060\u000a\u9019\u4e9b\u5b57\u662f\u4e2d\u6587\u5b57\u000a\u0045\u0073\u0074\u0061\u0073\u0020\u0073\u006f\u006e\u0020\u0070\u0061\u006c\u0061\u0062\u0072\u0061\u0073\u0020\u0065\u006e\u0020\u0065\u0073\u0070\u0061\u00f1\u006f\u006c",
 "mixed unicode": "この\u5b57は\u65e5\u672c\u8a9eの\u5b57だ\n\u9019\u4e9b字\u662f中文字\nEstas son palabras en espa\u00f1ol",
 "object": {"again": {"and again": {"that's": "enough"}}},
 "array": [1,[2,[3,[4,[5,[6,[7,[8,[9,[10]]]]]]]]]]
}"""
]

big = "[" + ",".join(5000 * obj) + "]"


def bench():
    parser.parse(lexer.tokenize(big))


if __name__ == "__main__":
    bench()
