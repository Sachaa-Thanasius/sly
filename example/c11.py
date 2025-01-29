# pyright: reportUndefinedVariable=none

from __future__ import annotations

from collections.abc import Generator
from typing import Optional

from sly import Lexer
from sly.lex import Token, TokenStr


TYPE_CHECKING = False

if TYPE_CHECKING:
    from sly.types import _


# region -------- Regex helper patterns --------


# Identifiers

_digit = r"[0-9]"
_hexadecimal_digit = r"[0-9A-Fa-f]"
_nondigit = r"[a-zA-Z_]"

_universal_character_name = rf"\\u{_hexadecimal_digit}{{4}}|\\U{_hexadecimal_digit}{{8}}"

_identifier_nondigit = rf"{_nondigit}|({_universal_character_name})"
_identifier = rf"({_identifier_nondigit})(({_identifier_nondigit})|{_digit})*"


# Integer constants

_nonzero_digit = r"[1-9]"
_decimal_constant = rf"{_nonzero_digit}{_digit}*"

_octal_digit = r"[0-7]"
_octal_constant = rf"0{_octal_digit}*"

_hexadecimal_prefix = r"0[xX]"
_hexadecimal_constant = rf"{_hexadecimal_prefix}{_hexadecimal_digit}+"

_unsigned_suffix = r"[uU]"
_long_suffix = r"[lL]"
_long_long_suffix = r"ll|LL"
_integer_suffix = "|".join(
    (
        rf"({_unsigned_suffix}{_long_suffix}?)",
        rf"({_unsigned_suffix}({_long_long_suffix}))",
        rf"({_long_suffix}{_unsigned_suffix}?)",
        rf"(({_long_long_suffix}){_unsigned_suffix}?)",
    )
)

_integer_constant = "|".join(
    (
        rf"({_decimal_constant}({_integer_suffix})?)",
        rf"({_octal_constant}({_integer_suffix})?)",
        rf"({_hexadecimal_constant}({_integer_suffix})?)",
    )
)


# Floating constants

_sign = r"[-+]"
_digit_sequence = rf"{_digit}+"
_floating_suffix = r"[flFL]"

_fractional_constant = "|".join(
    (
        rf"({_digit_sequence}?\.{_digit_sequence})",
        rf"({_digit_sequence}\.)",
    )
)

_exponent_part = rf"[eE]{_sign}?{_digit_sequence}"
_decimal_floating_constant = "|".join(
    (
        rf"(({_fractional_constant}){_exponent_part}?{_floating_suffix}?)",
        rf"({_digit_sequence}{_exponent_part}{_floating_suffix}?)",
    )
)

_hexadecimal_digit_sequence = rf"{_hexadecimal_digit}+"
_hexadecimal_fractional_constant = "|".join(
    (
        rf"(({_hexadecimal_digit_sequence})?\.{_hexadecimal_digit_sequence})",
        rf"({_hexadecimal_digit_sequence}\.)",
    )
)
_binary_exponent_part = rf"[pP]{_sign}?{_digit_sequence}"
_hexadecimal_floating_constant = "|".join(
    (
        rf"({_hexadecimal_prefix}({_hexadecimal_fractional_constant})({_binary_exponent_part}){_floating_suffix}?)",
        rf"({_hexadecimal_prefix}{_hexadecimal_digit_sequence}{_binary_exponent_part}{_floating_suffix}?)",
    )
)


# Constants

_constant = "|".join(
    (
        rf"({_integer_constant})",
        rf"({_decimal_floating_constant})",
        rf"({_hexadecimal_floating_constant})",
    )
)


# Preprocessing numbers

_preprocessing_number = r"\.?[0-9]([0-9A-Za-z_\.]|[eEpP][+-])*"


# Character and string constants

_simple_escape_sequence = r"""\\['"?\\abfnrtv]"""
_octal_escape_sequence = rf"\\({_octal_digit}{{1,3}})"
_hexadecimal_escape_sequence = rf"\\x{_hexadecimal_digit}+"
_escape_sequence = "|".join(
    (
        f"({_simple_escape_sequence})",
        f"({_octal_escape_sequence})",
        f"({_hexadecimal_escape_sequence})",
        f"({_universal_character_name})",
    )
)


# endregion --------


class CLexer(Lexer):
    tokens = {
        # Constant
        CONSTANT,

        # Ellipsis
        ELLIPSIS,

        # Assignment operators
        PLUS_ASSIGN, MINUS_ASSIGN, MUL_ASSIGN, DIV_EQUAL, MOD_ASSIGN,
        OR_ASSIGN, AND_ASSIGN, XOR_ASSIGN, LSHIFT_ASSIGN, RSHIFT_ASSIGN,

        # Operators
        LSHIFT, RSHIFT,
        EQ, NEQ, LEQ, GEQ,
        ASSIGN,
        LT, GT,
        INC, DEC,                           # Increment/decrement
        PTR,                                # Structure dereference
        PLUS, MINUS, STAR, SLASH, PERCENT,
        BANG,
        ANDAND, BARBAR, AND, BAR, CARET,
        QUESTION, COLON,
        TILDE,

        # Delimiters
        LBRACE, RBRACE,
        LBRACK, RBRACK,
        LPAREN, RPAREN,
        SEMICOLON, COMMA, DOT,

        # Keywords - lowercase
        AUTO, BREAK, CASE, CHAR, CONST, CONTINUE, DEFAULT, DO, DOUBLE, ELSE, ENUM, EXTERN, FLOAT, FOR, GOTO, IF, INLINE,
        INT, LONG, REGISTER, RESTRICT, RETURN, SHORT, SIGNED, SIZEOF, STATIC, STRUCT, SWITCH, TYPEDEF, UNION, UNSIGNED,
        VOID, VOLATILE, WHILE,

        # Keywords - underscore
        ALIGNAS, ALIGNOF, ATOMIC, BOOL, COMPLEX, GENERIC, IMAGINARY, NORETURN, STATIC_ASSERT, THREAD_LOCAL,

        # Identifier
        ID,
    }  # fmt: skip

    # Whitespace
    ignore = " \t\v\f\r"

    @_(r"\n+")
    def ignore_newline(self, t: Token) -> None:
        self.lineno += len(t.value)

    @_(
        _integer_constant,
        _decimal_floating_constant,
        _hexadecimal_floating_constant,
    )
    def CONSTANT(self, t: Token):
        return t

    @_(_preprocessing_number)
    def PREPROCESSING_NUMBER(self, t: Token):
        print("ERROR: These characters form a preprocessor number, but not a constant")
        self.error(t)

    @_(r"[LuU]?'")
    def CHAR_CONSTANT_START(self, t: Token):
        self._char_const_start = t
        self.push_state(CCharConstantLexer)

    @_(r'([LuU]|u8)?"')
    def STRING_LITERAL_START(self, t: Token):
        self._string_literal_start = t
        self.push_state(CStringLiteralLexer)

    # fmt: off

    # Ellipsis
    ELLIPSIS                = r"\.\.\."

    # Assignment operators
    PLUS_ASSIGN             = r"\+="
    MINUS_ASSIGN            = r"\-="
    MUL_ASSIGN              = r"\*="
    DIV_EQUAL               = r"/="
    MOD_ASSIGN              = r"%="
    OR_ASSIGN               = r"\|="
    AND_ASSIGN              = r"\&="
    XOR_ASSIGN              = r"\^="
    LSHIFT_ASSIGN           = r"<<="
    RSHIFT_ASSIGN           = r">>="

    # Operators
    LSHIFT                  = r"<<"
    RSHIFT                  = r">>"
    EQ                      = r"=="
    NEQ                     = r"!="
    LEQ                     = r"<="
    GEQ                     = r">="
    ASSIGN                  = r"="
    LT                      = r"<"
    GT                      = r">"
    INC                     = r"\+\+"       # Increment
    DEC                     = r"\-\-"       # Decrement
    PTR                     = r"\->"        # Structure dereference
    PLUS                    = r"\+"
    MINUS                   = r"\-"
    STAR                    = r"\*"
    SLASH                   = r"/"
    PERCENT                 = r"%"
    BANG                    = r"!"
    ANDAND                  = r"\&\&"
    BARBAR                  = r"\|\|"
    AND                     = r"\&"
    BAR                     = r"\|"
    CARET                   = r"\^"
    QUESTION                = r"\?"
    COLON                   = r":"
    TILDE                   = r"\~"

    # Delimiters
    LBRACE                  = r"\{"
    RBRACE                  = r"\}"
    LBRACK                  = r"\["
    RBRACK                  = r"\]"
    LPAREN                  = r"\("
    RPAREN                  = r"\)"
    SEMICOLON               = r";"
    COMMA                   = r","
    DOT                     = r"\."

    # Identifiers and keywords
    ID: TokenStr            = _identifier  # pyright: ignore [reportAssignmentType]
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
    ID["inline"]            = INLINE
    ID["int"]               = INT
    ID["long"]              = LONG
    ID["register"]          = REGISTER
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

    ID["_Alignas"]          = ALIGNAS
    ID["_Alignof"]          = ALIGNOF
    ID["_Atomic"]           = ATOMIC
    ID["_Bool"]             = BOOL
    ID["_Complex"]          = COMPLEX
    ID["_Generic"]          = GENERIC
    ID["_Imaginary"]        = IMAGINARY
    ID["_Noreturn"]         = NORETURN
    ID["_Static_assert"]    = STATIC_ASSERT
    ID["_Thread_local"]     = THREAD_LOCAL

    # fmt: on

    def tokenize(self, text: str, lineno: int = 1, index: int = 0) -> Generator[Token]:
        yield from super().tokenize(text, lineno, index)

        # Handle EOF for incomplete char constants and string literals.
        if self._char_const_start is not None:
            print("ERROR: Missing terminating ' character")
            self.error(self._char_const_start)
        elif self._string_literal_start is not None:
            print('ERROR: Missing terminating " character')
            self.error(self._string_literal_start)

    def __init__(self):
        self._char_const_start: Token | None = None
        self._string_literal_start: Token | None = None


class CCharConstantLexer(Lexer):
    _char_const_start: Optional[Token]

    tokens = {CHAR, INCORRECT_ESCAPE_SEQUENCE, CHAR_CONST_END, MISSING_TERMINATOR}

    @_(_escape_sequence)
    def CHAR(self, t: Token):
        pass

    @_(r"\\")
    def INCORRECT_ESCAPE_SEQUENCE(self, t: Token):
        print("ERROR: Incorrect escape sequence")
        self.error(t)

    @_(r"'")
    def CHAR_CONSTANT_END(self, t: Token):
        assert self._char_const_start is not None

        self.pop_state()

        start = self._char_const_start
        self._char_const_start = None
        return Token("CONSTANT", self.text[start.index : t.end], start.lineno, start.index, t.end)

    @_(r"\n")
    def MISSING_TERMINATOR(self, t: Token):
        print("ERROR: Missing terminating ' character")
        self.error(t)


class CStringLiteralLexer(Lexer):
    _string_literal_start: Optional[Token]

    tokens = {STRING_LITERAL_END, MISSING_TERMINATOR, STRING}

    @_(r'"')
    def STRING_LITERAL_END(self, t: Token):
        assert self._string_literal_start is not None

        self.pop_state()

        start = self._string_literal_start
        self._string_literal_start = None
        return Token("STRING_LITERAL", self.text[start.index : t.end], start.lineno, start.index, t.end)

    @_(r"\n")
    def MISSING_TERMINATOR(self, t: Token):
        print('ERROR: Missing terminating " character')
        self.error(t)

    @_(r".")
    def STRING(self, t: Token):
        pass
