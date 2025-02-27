# pyright: basic, reportUndefinedVariable=none, reportRedeclaration=none

"""json.py: Simple json example adapted from python-parsing-benchmarks."""

from __future__ import annotations

import re

from sly import Lexer, Parser
from sly.lex import Token
from sly.yacc import YaccProduction as Prod


TYPE_CHECKING = False

if TYPE_CHECKING:
    from sly.types import _


_json_unesc_re = re.compile(r'\\(["/\\bfnrt]|u[0-9A-Fa-f])')
_json_unesc_map = {
    '"': '"',
    "/": "/",
    "\\": "\\",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


def _json_unescape(m: re.Match[str]):
    c = m.group(1)
    if c[0] == "u":
        return chr(int(c[1:], 16))
    c2 = _json_unesc_map.get(c)
    if not c2:
        msg = f"invalid escape sequence: {m.group(0)}"
        raise ValueError(msg)
    return c2


def json_unescape(s: str):
    return _json_unesc_re.sub(_json_unescape, s[1:-1])


class JsonLexer(Lexer):
    tokens = {STRING, NUMBER, TRUE, FALSE, NULL}
    ignore = " \t\n\r"
    literals = {"{", "}", "[", "]", ":", ","}

    @_(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([Ee][+-]?[0-9]+)?")
    def NUMBER(self, t: Token):
        t.value = float(t.value)
        return t

    @_(r'"([ !#-\[\]-\U0010ffff]+|\\(["\/\\bfnrt]|u[0-9A-Fa-f]{4}))*"')
    def STRING(self, t: Token):
        t.value = json_unescape(t.value)
        return t

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

    @_('"{" [ pairs ] "}"')
    def value(self, p: Prod):
        if p.pairs:
            return dict(p.pairs)
        else:
            return {}

    @_('pair { "," pair }')
    def pairs(self, p: Prod):
        return [p.pair0, *p.pair1]

    @_('STRING ":" value')
    def pair(self, p: Prod):
        return (p.STRING, p.value)

    @_('value { "," value }')
    def items(self, p: Prod):
        return [p.value0, *p.value1]

    @_('"[" [ items ] "]"')
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
