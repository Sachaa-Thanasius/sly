# ruff: noqa: F811, F821, RUF012, ANN201
# pyright: basic, reportUndefinedVariable=none, reportRedeclaration=none

"""json.py: Simple json example adapted from python-parsing-benchmarks for use in benchmarking."""

from __future__ import annotations

import linecache
import tracemalloc

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

    def error(self, token):
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


def display_top(snapshot: tracemalloc.Snapshot, key_type="lineno", limit=10):
    # Pretty top: Copied from the tracemalloc docs.

    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    print(f"Top {limit} lines")
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print(f"#{index}: {frame.filename}:{frame.lineno}: {stat.size / 1024:.1f} KiB")
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print(f"    {line}")

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print(f"{len(other)} other: {size / 1024:.1f} KiB")
    total = sum(stat.size for stat in top_stats)
    print(f"Total allocated size: {total / 1024:.1f} KiB")


def profile_memory():
    tracemalloc.start()

    parser.parse(lexer.tokenize(big))

    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)


def bench():
    parser.parse(lexer.tokenize(big))


if __name__ == "__main__":
    bench()
