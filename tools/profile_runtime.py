from examples.json import JsonLexer, JsonParser


# Form the master regular expression
# master_parts: list[str] = []

# if cls.ignore:
#     master_parts.append(rf"(?P<SLY_IGNORE>[{cls.ignore}])")

# master_parts.extend(parts)

# if cls.literals:
#     master_parts.append(rf"(?P<SLY_LITERAL>[{re.escape(''.join(cls.literals))}])")

# master_parts.append(r"(?P<SLY_ERROR>.+)")

# cls._master_re = cls.regex_module.compile("|".join(master_parts), cls.reflags)

# for m in _master_re.finditer(text, index):
#     assert m.lastgroup is not None, "There should always be a matched named group."

#     if m.lastgroup not in {"SLY_IGNORE", "SLY_LITERAL", "SLY_ERROR"}:
#         tok = Token(m.lastgroup, m.group(), lineno, index, m.end())
#         index = tok.end

#         if tok.type in _remapping:
#             tok.type = _remapping[tok.type].get(tok.value, tok.type)

#         if tok.type in _token_funcs:
#             self.index, self.lineno = (index, lineno)
#             tok = _token_funcs[tok.type](self, tok)
#             index, lineno = (self.index, self.lineno)

#             if not tok:
#                 break

#         if tok.type in _ignored_tokens:
#             break

#         yield tok

#         if self._must_reset_loop:
#             self._must_reset_loop = False
#             break

#     elif m.lastgroup == "SLY_IGNORE":
#         index += 1
#         continue

#     elif m.lastgroup == "SLY_LITERAL":
#         value = m.group()
#         tok = Token(value, value, lineno, index, index + 1)
#         index += 1
#         yield tok

#     else:
#         self.index, self.lineno = (index, lineno)

#         tok = Token("ERROR", m.group(), lineno, index)
#         tok = self.error(tok)
#         if tok is not None:
#             tok.end = self.index
#             yield tok

#         index, lineno = (self.index, self.lineno)
#         break
# else:
#     break

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

lexer = JsonLexer()
parser = JsonParser()


def profile() -> None:
    # parser.parse(lexer.tokenize(big))
    parser.parse(JsonLexer(big))


if __name__ == "__main__":
    profile()
