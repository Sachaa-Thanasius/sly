from examples.json import JsonLexer, JsonParser


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
    parser.parse(lexer.tokenize(big))


if __name__ == "__main__":
    profile()
