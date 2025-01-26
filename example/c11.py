# pyright: reportUndefinedVariable=none

from sly import Lexer
from sly.lex import Token, TokenStr


TYPE_CHECKING = False

if TYPE_CHECKING:
    from sly.types import _


# region ---- Identifers

_digit = r"[0-9]"
_hexadecimal_digit = r"[0-9A-Fa-f]"
_nondigit = r"[a-zA-Z_]"

_universal_character_name = rf"\\u{_hexadecimal_digit}{{4}}|\\U{_hexadecimal_digit}{{8}}"

_identifier_nondigit = rf"{_nondigit}|({_universal_character_name})"
_identifier = rf"({_identifier_nondigit})(({_identifier_nondigit})|{_digit})*"

# endregion ----


# region ---- Integer constants

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

# endregion ----


# region ---- Floating constants

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

# endregion ----


# region ---- Constants

_constant = "|".join(
    (
        rf"({_integer_constant})",
        rf"({_decimal_floating_constant})",
        rf"({_hexadecimal_floating_constant})",
    )
)

# endregion ----


# region ---- Preprocessing numbers

_preprocessing_number = r"\.?[0-9]([0-9A-Za-z_\.]|[eEpP][+-])*"

# endregion ----


# region ---- Character and string constants

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

# endregion


class C11Lexer(Lexer):
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

    CONSTANT = _constant

    @_(_preprocessing_number)
    def PREPROCESSING_NUMBER(self, t: Token):
        print("ERROR: These characters form a preprocessor number, but not a constant")
        self.error(t)

    # | (['L' 'u' 'U']|"") "'"        { char lexbuf; char_literal_end lexbuf; CONSTANT }
    # | (['L' 'u' 'U']|""|"u8") "\""  { string_literal lexbuf; STRING_LITERAL }

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
