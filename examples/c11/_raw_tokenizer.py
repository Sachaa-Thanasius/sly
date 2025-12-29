"""A tokenizer for C11 that doesn't use SLY.

Based on the tokenizer example in the Python re docs: https://docs.python.org/3/library/re.html#writing-a-tokenizer
"""

from __future__ import annotations

import re
from collections.abc import Generator

from ._regex_helpers import constant, escape_sequence, identifier, preprocessing_number
from .context import CNameContext
from .errors import TokenError
from .token import Token


# region -------- Char constant handler --------


CHAR_CONST_REST_SPEC = [
    ("CHAR_CHAR",           escape_sequence),
    ("BAD_ESCAPE_SEQ",      r"\\"),
    ("CHAR_CONSTANT_END",   r"'"),
    ("MISSING_TERMINATOR",  r"\n"),
]  # fmt: skip

CHAR_CONST_REST_REGEX = re.compile("|".join(f"(?P<{tok_name}>{tok_pat})" for tok_name, tok_pat in CHAR_CONST_REST_SPEC))


def _tokenize_char_constant_rest(code: str, /, char_start: int, line_num: int, line_start: int) -> int:
    """Find the end of a C character constant.

    Precondition: The start of the character constant was already found and consumed.

    Returns
    -------
    int
        The ending index of the character constant.

    Raises
    ------
    TokenError
        If either of the following happens:
            1. The character constant body ends, via newline or EOF, before the terminating single-quote.
            2. An invalid escape sequence is encountered.
    """

    for mo in CHAR_CONST_REST_REGEX.finditer(code, char_start):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start

        if kind is None:
            msg = "The token kind should never be none."
            raise RuntimeError(msg)

        elif kind == "CHAR_CHAR":
            continue

        elif kind == "BAD_ESCAPE_SEQ":
            msg = "Incorrect escape sequence."
            raise TokenError(msg, value, line_num, column)

        elif kind == "CHAR_CONSTANT_END":
            return mo.end()

        elif kind == "MISSING_TERMINATOR":
            msg = "Missing terminating ' character."
            raise TokenError(msg, value, line_num, column)

    msg = "Missing terminating ' character."
    raise TokenError(msg, code[char_start:], line_num, char_start - line_start)


# endregion --------


# region -------- String literal handler --------


STR_LIT_REST_SPEC = [
    ("STRING_LITERAL_END",  r'"'),
    ("MISSING_TERMINATOR",  r"\n"),
    ("STRING_CHAR",         r"."),
]  # fmt: skip

STR_LIT_REST_REGEX = re.compile("|".join(f"(?P<{tok_name}>{tok_pat})" for tok_name, tok_pat in STR_LIT_REST_SPEC))


def _tokenize_str_literal_rest(code: str, /, str_start: int, line_num: int, line_start: int) -> int:
    """Find the end of a C string literal.

    Precondition: The start of the string literal was already found and consumed.

    Returns
    -------
    int
        The ending index of the string literal.

    Raises
    ------
    TokenError
        If the string literal body ends, via newline or EOF, before the terminating double-quote.
    """

    for mo in STR_LIT_REST_REGEX.finditer(code, str_start):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start

        if kind is None:
            msg = "The token kind should never be none."
            raise RuntimeError(msg)

        elif kind == "STRING_LITERAL_END":
            return mo.end()

        elif kind == "STRING_CHAR":
            continue

        elif kind == "MISSING_TERMINATOR":
            msg = 'Missing terminating " character.'
            raise TokenError(msg, value, line_num, column)

    msg = 'Missing terminating " character.'
    raise TokenError(msg, code[str_start:], line_num, str_start - line_start)


# endregion --------


# region -------- Tokenizer --------


KEYWORDS = {
    "auto":                 "AUTO",
    "break":                "BREAK",
    "case":                 "CASE",
    "char":                 "CHAR",
    "const":                "CONST",
    "continue":             "CONTINUE",
    "default":              "DEFAULT",
    "do":                   "DO",
    "double":               "DOUBLE",
    "else":                 "ELSE",
    "enum":                 "ENUM",
    "extern":               "EXTERN",
    "float":                "FLOAT",
    "for":                  "FOR",
    "goto":                 "GOTO",
    "if":                   "IF",
    "inline":               "INLINE",
    "int":                  "INT",
    "long":                 "LONG",
    "register":             "REGISTER",
    "restrict":             "RESTRICT",
    "return":               "RETURN",
    "short":                "SHORT",
    "signed":               "SIGNED",
    "sizeof":               "SIZEOF",
    "static":               "STATIC",
    "struct":               "STRUCT",
    "switch":               "SWITCH",
    "typedef":              "TYPEDEF",
    "union":                "UNION",
    "unsigned":             "UNSIGNED",
    "void":                 "VOID",
    "volatile":             "VOLATILE",
    "while":                "WHILE",
    "_Alignas":             "ALIGNAS",
    "_Alignof":             "ALIGNOF",
    "_Atomic":              "ATOMIC",
    "_Bool":                "BOOL",
    "_Complex":             "COMPLEX",
    "_Generic":             "GENERIC",
    "_Imaginary":           "IMAGINARY",
    "_Noreturn":            "NORETURN",
    "_Static_assert":       "STATIC_ASSERT",
    "_Thread_local":        "THREAD_LOCAL",
}  # fmt: skip


# NOTE: Fake tokens or token names are lowercase.
TOKEN_SPEC = [
    # Skipped whitespace
    ("IGNORE",                  r"[ \t\v\f\r]+"),

    ("NEWLINE",                 r"\n"),
    ("CONSTANT",                constant),

    # Not an actual token; results in error.
    ("preprocessing_number",    preprocessing_number),
    # Not an actual token; results in CONSTANT or error.
    ("char_constant",           r"[LuU]?'"),
    # Not the pattern for the actual token; results in STRING_LITERAL or error.
    ("STRING_LITERAL",          r'([LuU]|u8)?"'),

    # Ellipsis
    ("ELLIPSIS",                r"\.\.\."),

    # Assignment operators
    ("PLUS_ASSIGN",             r"\+="),
    ("MINUS_ASSIGN",            r"\-="),
    ("MUL_ASSIGN",              r"\*="),
    ("DIV_EQUAL",               r"/="),
    ("MOD_ASSIGN",              r"%="),
    ("OR_ASSIGN",               r"\|="),
    ("AND_ASSIGN",              r"\&="),
    ("XOR_ASSIGN",              r"\^="),
    ("LSHIFT_ASSIGN",           r"<<="),
    ("RSHIFT_ASSIGN",           r">>="),

    # Operators
    ("LSHIFT",                  r"<<"),
    ("RSHIFT",                  r">>"),
    ("EQ",                      r"=="),
    ("NEQ",                     r"!="),
    ("LEQ",                     r"<="),
    ("GEQ",                     r">="),
    ("ASSIGN",                  r"="),
    ("LT",                      r"<"),
    ("GT",                      r">"),
    ("INC",                     r"\+\+"),
    ("DEC",                     r"\-\-"),
    ("PTR",                     r"\->"),
    ("PLUS",                    r"\+"),
    ("MINUS",                   r"\-"),
    ("STAR",                    r"\*"),
    ("SLASH",                   r"/"),
    ("PERCENT",                 r"%"),
    ("BANG",                    r"!"),
    ("ANDAND",                  r"\&\&"),
    ("BARBAR",                  r"\|\|"),
    ("AND",                     r"\&"),
    ("BAR",                     r"\|"),
    ("CARET",                   r"\^"),
    ("QUESTION",                r"\?"),
    ("COLON",                   r":"),
    ("TILDE",                   r"\~"),

    # Delimiters
    ("LBRACE",                  r"\{"),
    ("RBRACE",                  r"\}"),
    ("LBRACK",                  r"\["),
    ("RBRACK",                  r"\]"),
    ("LPAREN",                  r"\("),
    ("RPAREN",                  r"\)"),
    ("SEMICOLON",               r";"),
    ("COMMA",                   r","),
    ("DOT",                     r"\."),

    # Identifier
    ("ID",                      identifier),

    # Error
    ("ERROR",                   r"."),

]  # fmt: skip

TOKEN_REGEX = re.compile("|".join(f"(?P<{tok_name}>{tok_pat})" for tok_name, tok_pat in TOKEN_SPEC))


def tokenize(code: str, /, line_num: int = 1, line_start: int = 0, ctx: CNameContext | None = None) -> Generator[Token]:
    if ctx is None:
        ctx = CNameContext()

    for mo in TOKEN_REGEX.finditer(code):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start

        if kind is None:
            msg = "The token kind should never be None."
            raise RuntimeError(msg)

        elif kind == "IGNORE":
            continue

        elif kind == "NEWLINE":
            line_start = mo.end()
            line_num += 1
            continue

        elif kind == "preprocessing_number":
            msg = "These characters form a preprocessor number, but not a constant."
            raise TokenError(msg, value, line_num, column)

        elif kind == "char_constant":
            char_start = mo.start()
            char_end = _tokenize_char_constant_rest(code, char_start, line_num, line_start)

            kind = "CONSTANT"
            value = code[char_start:char_end]

        elif kind == "STRING_LITERAL":
            str_start = mo.start()
            str_end = _tokenize_str_literal_rest(code, str_start, line_num, line_start)
            value = code[str_start:str_end]

        elif kind == "ID":
            kind = KEYWORDS.get(value, "ID")

        elif kind == "ERROR":
            msg = f"{value!r} unexpected on line {line_num}"
            raise TokenError(msg, value, line_num, column)

        yield Token(kind, value, line_num, column)

        # NOTE: TYPE and VARIABLE are not in the token specification above, but the parser needs them to lazily
        # disambiguate typedef and variable names.
        if kind == "ID":
            id_type = "TYPE" if (value in ctx) else "VARIABLE"
            yield Token(id_type, value, line_num, column)


TOKEN_NAMES = [name for name, _ in TOKEN_SPEC if name.isupper()] + ["TYPE", "VARIABLE"]


# endregion --------
