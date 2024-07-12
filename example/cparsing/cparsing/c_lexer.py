# pyright: reportUndefinedVariable=none, reportIndexIssue=none, reportConstantRedefinition=none
"""Module for lexing C code."""

import re
from typing import TYPE_CHECKING, NoReturn, Optional

from sly import Lexer
from sly.lex import Token

from . import c_context
from ._typing_compat import override
from .utils import Coord

if TYPE_CHECKING:
    from sly.types import _


__all__ = ("CLexer",)


# ============================================================================
# region -------- Helpers
# ============================================================================


def _find_token_column(text: str, t: Token) -> int:
    last_cr = text.rfind("\n", 0, t.index)
    if last_cr < 0:
        last_cr = 0
    return (t.index - last_cr) + 1


_line_pattern = re.compile(r"([ \t]*line\W)|([ \t]*\d+)")
_pragma_pattern = re.compile(r"[ \t]*pragma\W")

_hex_prefix = "0[xX]"
_hex_digits = "[0-9a-fA-F]+"
_bin_prefix = "0[bB]"
_bin_digits = "[01]+"

# integer constants (K&R2: A.2.5.1)
_integer_suffix_opt = r"(([uU]ll)|([uU]LL)|(ll[uU]?)|(LL[uU]?)|([uU][lL])|([lL][uU]?)|[uU])?"
_decimal_constant = "(0" + _integer_suffix_opt + ")|([1-9][0-9]*" + _integer_suffix_opt + ")"

# character constants (K&R2: A.2.5.2)
# Note: a-zA-Z and '.-~^_!=&;,' are allowed as escape chars to support #line
# directives with Windows paths as filenames (..\..\dir\file)
# For the same reason, decimal_escape allows all digit sequences. We want to
# parse all correct code, even if it means to sometimes parse incorrect
# code.
#
# The original regexes were taken verbatim from the C syntax definition,
# and were later modified to avoid worst-case exponential running time.
#
#   simple_escape = r"""([a-zA-Z._~!=&\^\-\\?'"])"""
#   decimal_escape = r"""(\d+)"""
#   hex_escape = r"""(x[0-9a-fA-F]+)"""
#   bad_escape = r"""([\\][^a-zA-Z._~^!=&\^\-\\?'"x0-7])"""
#
# The following modifications were made to avoid the ambiguity that allowed backtracking:
# (https://github.com/eliben/pycparser/issues/61)
#
# - \x was removed from simple_escape, unless it was not followed by a hex digit, to avoid ambiguity with hex_escape.
# - hex_escape allows one or more hex characters, but requires that the next character(if any) is not hex
# - decimal_escape allows one or more decimal characters, but requires that the next character(if any) is not a decimal
# - bad_escape does not allow any decimals (8-9), to avoid conflicting with the permissive decimal_escape.
#
# Without this change, python's `re` module would recursively try parsing each ambiguous escape sequence in multiple ways.
# e.g. `\123` could be parsed as `\1`+`23`, `\12`+`3`, and `\123`.
_simple_escape = r"""([a-wyzA-Z._~!=&\^\-\\?'"]|x(?![0-9a-fA-F]))"""
_decimal_escape = r"""(\d+)(?!\d)"""
_hex_escape = r"""(x[0-9a-fA-F]+)(?![0-9a-fA-F])"""
_bad_escape = r"""([\\][^a-zA-Z._~^!=&\^\-\\?'"x0-9])"""

_escape_sequence = r"""(\\(""" + _simple_escape + "|" + _decimal_escape + "|" + _hex_escape + "))"

# This complicated regex with lookahead might be slow for strings, so because all of the valid escapes (including \x) allowed
# 0 or more non-escaped characters after the first character, simple_escape+decimal_escape+hex_escape got simplified to
_escape_sequence_start_in_string = r"""(\\[0-9a-zA-Z._~!=&\^\-\\?'"])"""

_string_char = r"""([^"\\\n]|""" + _escape_sequence_start_in_string + ")"
_cconst_char = r"""([^'\\\n]|""" + _escape_sequence + ")"

# floating constants (K&R2: A.2.5.3)
_exponent_part = r"""([eE][-+]?[0-9]+)"""
_fractional_constant = r"""([0-9]*\.[0-9]+)|([0-9]+\.)"""
_binary_exponent_part = r"""([pP][+-]?[0-9]+)"""
_hex_fractional_constant = "(((" + _hex_digits + r""")?\.""" + _hex_digits + ")|(" + _hex_digits + r"""\.))"""


# endregion


# ============================================================================
# region -------- Lexers
# ============================================================================


class CLexer(Lexer):
    # ---- Reserved keywords
    # fmt: off
    keywords: set[str] = {
        AUTO, BREAK, CASE, CHAR, CONST, CONTINUE, DEFAULT, DO, DOUBLE, ELSE, ENUM,
        EXTERN, FLOAT, FOR, GOTO, IF, INLINE, INT, LONG, REGISTER, OFFSETOF,
        RESTRICT, RETURN, SHORT, SIGNED, SIZEOF, STATIC, STRUCT, SWITCH, TYPEDEF,
        UNION, UNSIGNED, VOID, VOLATILE, WHILE, INT128,
    }

    keywords_new: set[str] = {
        ALIGNAS_, ALIGNOF_, ATOMIC_, BOOL_, COMPLEX_, NORETURN_, PRAGMA_, STATIC_ASSERT_, THREAD_LOCAL_,
    }

    tokens = keywords | keywords_new | {
        # Identifiers
        ID,

        # Type identifiers (identifiers previously defined as types with typedef)
        TYPEID,

        # constants
        INT_CONST_DEC, INT_CONST_OCT, INT_CONST_HEX, INT_CONST_BIN, INT_CONST_CHAR,
        FLOAT_CONST, HEX_FLOAT_CONST,
        CHAR_CONST, WCHAR_CONST, U8CHAR_CONST, U16CHAR_CONST, U32CHAR_CONST,

        # String literals
        STRING_LITERAL,
        WSTRING_LITERAL,
        U8STRING_LITERAL,
        U16STRING_LITERAL,
        U32STRING_LITERAL,

        # Operators
        PLUS, MINUS, TIMES, DIVIDE, MOD,
        OR, AND, NOT, XOR, LSHIFT, RSHIFT,
        LOR, LAND, LNOT,
        LT, LE, GT, GE, EQ, NE,

        # Assignment
        EQUALS, TIMESEQUAL, DIVEQUAL, MODEQUAL,
        PLUSEQUAL, MINUSEQUAL,
        LSHIFTEQUAL, RSHIFTEQUAL, ANDEQUAL, XOREQUAL,
        OREQUAL,

        # Increment/decrement
        PLUSPLUS, MINUSMINUS,

        # Structure dereference (->)
        ARROW,

        # Conditional operator (?)
        CONDOP,

        # Ellipsis (...)
        ELLIPSIS,

        # Scope delimiters
        LBRACE, RBRACE,

        # Pre-processor
        PP_HASH,       # "#"
        PP_PRAGMA,     # "pragma"
        PP_PRAGMASTR,
    }
    # fmt: on

    # ---- Regular delimiters
    literals = {",", ".", ";", ":", "(", ")", "[", "]"}

    ignore = " \t"

    # ---- The rest of the tokens
    @_(r"[ \t]*\#")
    def PP_HASH(self, t: Token) -> Optional[Token]:
        if _line_pattern.match(self.text, pos=t.end):
            self.push_state(PreprocessorLineLexer)
            self.pp_line = None
            self.pp_filename = None
            return None

        elif _pragma_pattern.match(self.text, pos=t.end):
            self.push_state(PreprocessorPragmaLexer)
            return None

        else:
            t.type = "PP_HASH"
            return t

    STRING_LITERAL = '"' + _string_char + '*"'

    FLOAT_CONST = "((((" + _fractional_constant + ")" + _exponent_part + "?)|([0-9]+" + _exponent_part + "))[FfLl]?)"
    HEX_FLOAT_CONST = (
        "("
        + _hex_prefix
        + "("
        + _hex_digits
        + "|"
        + _hex_fractional_constant
        + ")"
        + _binary_exponent_part
        + "[FfLl]?)"
    )
    INT_CONST_HEX = _hex_prefix + _hex_digits + _integer_suffix_opt
    INT_CONST_BIN = _bin_prefix + _bin_digits + _integer_suffix_opt

    @_("0[0-7]*[89]")
    def BAD_CONST_OCT(self, t: Token) -> None:
        self.error(t, "Invalid octal constant")

    INT_CONST_OCT = "0[0-7]*" + _integer_suffix_opt
    INT_CONST_DEC = _decimal_constant

    INT_CONST_CHAR = "'" + _cconst_char + "{2,4}'"
    CHAR_CONST = "'" + _cconst_char + "'"
    WCHAR_CONST = "L" + CHAR_CONST
    U8CHAR_CONST = "u8" + CHAR_CONST
    U16CHAR_CONST = "u" + CHAR_CONST
    U32CHAR_CONST = "U" + CHAR_CONST

    @_("('" + _cconst_char + "*\\n)|('" + _cconst_char + "*$)")
    def UNMATCHED_QUOTE(self, t: Token) -> NoReturn:
        self.error(t, "Unmatched '")

    @_(r"""('""" + _cconst_char + """[^'\n]+')|('')|('""" + _bad_escape + r"""[^'\n]*')""")
    def BAD_CHAR_CONST(self, t: Token) -> NoReturn:
        self.error(t, f"Invalid char constant {t.value!r}")

    # string literals (K&R2: A.2.6)
    WSTRING_LITERAL = "L" + STRING_LITERAL
    U8STRING_LITERAL = "u8" + STRING_LITERAL
    U16STRING_LITERAL = "u" + STRING_LITERAL
    U32STRING_LITERAL = "U" + STRING_LITERAL

    @_('"' + _string_char + "*" + _bad_escape + _string_char + '*"')
    def BAD_STRING_LITERAL(self, t: Token) -> NoReturn:
        self.error(t, "String contains invalid escape code!r")

    # Increment/decrement
    PLUSPLUS = r"\+\+"
    MINUSMINUS = r"--"

    # ->
    ARROW = r"->"

    # fmt: off
    # Assignment operators
    TIMESEQUAL  = r"\*="
    DIVEQUAL    = r"/="
    MODEQUAL    = r"%="
    PLUSEQUAL   = r"\+="
    MINUSEQUAL  = r"-="
    LSHIFTEQUAL = r"<<="
    RSHIFTEQUAL = r">>="
    ANDEQUAL    = r"&="
    OREQUAL     = r"\|="
    XOREQUAL    = r"\^="

    # Operators
    LSHIFT      = r"<<"
    RSHIFT      = r">>"
    LOR         = r"\|\|"
    LAND        = r"&&"
    LE          = r"<="
    GE          = r">="
    EQ          = r"=="
    NE          = r"!="
    EQUALS      = r"="
    LNOT        = r"!"
    LT          = r"<"
    GT          = r">"
    PLUS        = r"\+"
    MINUS       = r"-"
    TIMES       = r"\*"
    DIVIDE      = r"/"
    MOD         = r"%"
    OR          = r"\|"
    AND         = r"&"
    NOT         = r"~"
    XOR         = r"\^"

    # ?
    CONDOP      = r"\?"

    # Delimiters
    ELLIPSIS    = r"\.\.\."

    # Identifiers and keywords
    # valid C identifiers (K&R2: A.2.3), plus "$" (supported by some compilers)
    ID = r"[a-zA-Z_$][0-9a-zA-Z_$]*" # pyright: ignore [reportAssignmentType]

    ID["auto"]              = AUTO
    ID["break"]             = BREAK
    ID["case"]              = CASE
    ID["char"]              = CHAR
    ID["const"]             = CONST
    ID["continue"]          = CONTINUE
    ID["default"]           = DEFAULT
    ID["do"]                = DO
    ID["double"]            = DOUBLE
    ID["else"]              = ELSE
    ID["enum"]              = ENUM
    ID["extern"]            = EXTERN
    ID["float"]             = FLOAT
    ID["for"]               = FOR
    ID["goto"]              = GOTO
    ID["if"]                = IF
    ID["intline"]           = INLINE
    ID["int"]               = INT
    ID["long"]              = LONG
    ID["register"]          = REGISTER
    ID["offsetof"]          = OFFSETOF
    ID["restrict"]          = RESTRICT
    ID["return"]            = RETURN
    ID["short"]             = SHORT
    ID["signed"]            = SIGNED
    ID["sizeof"]            = SIZEOF
    ID["static"]            = STATIC
    ID["struct"]            = STRUCT
    ID["switch"]            = SWITCH
    ID["typedef"]           = TYPEDEF
    ID["union"]             = UNION
    ID["unsigned"]          = UNSIGNED
    ID["void"]              = VOID
    ID["volatile"]          = VOLATILE
    ID["while"]             = WHILE
    ID["__int128"]          = INT128

    ID["_Alignas"]          = ALIGNAS_
    ID["_Alignof"]          = ALIGNOF_
    ID["_Atomic"]           = ATOMIC_
    ID["_Bool"]             = BOOL_
    ID["_Complex"]          = COMPLEX_
    ID["_Noreturn"]         = NORETURN_
    ID["_Pragma"]           = PRAGMA_
    ID["_Static_assert"]    = STATIC_ASSERT_
    ID["_Thread_local"]     = THREAD_LOCAL_
    # fmt: on

    def ID(self, t: Token) -> Token:
        if self.context.scope_stack.get(t.value, False):
            t.type = "TYPEID"
        return t

    @_(r"\{")
    def LBRACE(self, t: Token) -> Token:
        self.create_scope()
        return t

    @_(r"\}")
    def RBRACE(self, t: Token) -> Token:
        self.pop_scope()
        return t

    @_(r"\n+")
    def ignore_newline(self, t: Token) -> None:
        self.lineno += t.value.count("\n")

    @override
    def error(self, t: Token, msg: Optional[str] = None) -> NoReturn:
        column = _find_token_column(self.text, t)
        msg = msg or f"Bad character {t.value[0]!r}"
        self.context.error(f"Bad character {t.value[0]!r}", Coord(self.lineno, column))

    def __init__(self, context: "c_context.CContext") -> None:
        self.context = context
        self.pp_line: Optional[str] = None
        self.pp_filename: Optional[str] = None

    def create_scope(self) -> None:
        self.context.scope_stack = self.context.scope_stack.new_child()

    def pop_scope(self) -> None:
        self.context.scope_stack = self.context.scope_stack.parents


class PreprocessorLineLexer(Lexer):
    """Lexer state that handles C's #line preprocessor directive."""

    tokens = {FILENAME, LINE_NUMBER, PP_LINE}

    ignore = " \t"

    @_('"' + _string_char + '*"')  # Same string as STRING_LITERAL.
    def FILENAME(self, t: Token) -> None:
        if self.pp_line is None:
            self.error(t, "filename before line number in #line")
        else:
            self.pp_filename = t.value.lstrip('"').rstrip('"')

    @_(_decimal_constant)  # Same string as INT_DEC_CONST.
    def LINE_NUMBER(self, t: Token) -> None:
        if self.pp_line is None:
            self.pp_line = t.value
        else:
            # Ignore: GCC's cpp sometimes inserts a numeric flag after the file name
            pass

    @_(r"line")
    def PP_LINE(self, t: Token) -> None:
        pass

    @_(r"\n")
    def ignore_newline(self, t: Token) -> None:
        if self.pp_line is None:
            self.error(t, "line number missing in #line")
        else:
            self.lineno = int(self.pp_line)

            if self.pp_filename is not None:
                self.context.filename = self.pp_filename

        self.pop_state()

    @override
    def error(self, t: Token, msg: Optional[str] = None) -> NoReturn:
        column = _find_token_column(self.text, t)
        msg = msg or f"invalid #line directive {t.value}"
        self.context.error(msg, Coord(self.lineno, column))

    def __init__(self, context: "c_context.CContext") -> None:
        self.context = context
        self.pp_line: Optional[str] = None
        self.pp_filename: Optional[str] = None


class PreprocessorPragmaLexer(Lexer):
    """Lexer state that handles C's #pragma preprocessor directive."""

    tokens = {PP_PRAGMA, STR}

    ignore = " \t"

    PP_PRAGMA = "pragma"

    @_(".+")
    def STR(self, t: Token) -> Token:
        t.type = "PP_PRAGMASTR"
        return t

    @_(r"\n")
    def ignore_newline(self, t: Token) -> None:
        self.lineno += 1
        self.pop_state()

    @override
    def error(self, t: Token, msg: Optional[str] = None) -> NoReturn:
        column = _find_token_column(self.text, t)
        msg = msg or f"invalid #pragma directive {t.value}"
        self.context.error(msg, Coord(self.lineno, column), t.value)

    def __init__(self, context: "c_context.CContext") -> None:
        self.context = context


# endregion
