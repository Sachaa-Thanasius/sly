# pyright: reportUndefinedVariable=none

from __future__ import annotations

from collections.abc import Generator

from sly import Lexer
from sly.lex import Token, TokenStr

from ._regex_helpers import (
    _decimal_floating_constant,
    _escape_sequence,
    _hexadecimal_floating_constant,
    _identifier,
    _integer_constant,
    _preprocessing_number,
)
from .context import CNameContext


TYPE_CHECKING = False

if TYPE_CHECKING:
    from sly.types import _


class CLexer(Lexer):
    tokens = {
        CONSTANT,
        STRING_LITERAL,

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
        ID, TYPE, VARIABLE,
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
        for tok in super().tokenize(text, lineno, index):
            yield tok

            # Handle name ambiguity.
            if tok.type == "ID":
                id_type = "TYPE" if (tok.value in self.context) else "VARIABLE"
                yield Token(id_type, tok.value, tok.lineno, tok.index, tok.end)

        # Handle EOF for incomplete char constants and string literals.
        if self._char_const_start is not None:
            print("ERROR: Missing terminating ' character")
            self.error(self._char_const_start)

        if self._string_literal_start is not None:
            print('ERROR: Missing terminating " character')
            self.error(self._string_literal_start)

    def __init__(self, context: CNameContext):
        self.context = context

        self._char_const_start: Token | None = None
        self._string_literal_start: Token | None = None


class CCharConstantLexer(Lexer):
    _char_const_start: Token | None

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
    _string_literal_start: Token | None

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
