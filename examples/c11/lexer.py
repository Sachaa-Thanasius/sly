# pyright: reportUndefinedVariable=none

from __future__ import annotations

from collections.abc import Generator

from sly import Lexer
from sly.lex import LexError, Token, TokenStr

from ._regex_helpers import _constant, _escape_sequence, _identifier, _preprocessing_number
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
        PLUS_ASSIGN, MINUS_ASSIGN, MUL_ASSIGN, DIV_ASSIGN, MOD_ASSIGN,
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
        NAME, TYPE, VARIABLE,
    }  # fmt: skip

    # Whitespace
    ignore = " \t\v\f\r"

    @_(r"\n+")
    def ignore_newline(self, t: Token) -> None:
        self.lineno += len(t.value)

    CONSTANT = _constant

    @_(_preprocessing_number)
    def PREPROCESSING_NUMBER(self, t: Token):
        # Not an actual token; results in error.
        self.error(t, "These characters form a preprocessor number, but not a constant.")

    @_(r"[LuU]?'")
    def CHAR_CONSTANT_START(self, t: Token):
        # Not an actual token; results in CONSTANT or error.
        self._char_const_start = t
        self.push_state(CCharConstantLexer)

    @_(r'([LuU]|u8)?"')
    def STRING_LITERAL_START(self, t: Token):
        # Not an actual token; results in STRING_LITERAL or error.
        self._string_literal_start = t
        self.push_state(CStringLiteralLexer)

    # fmt: off

    # Ellipsis
    ELLIPSIS                = r"\.\.\."

    # Assignment operators
    PLUS_ASSIGN             = r"\+="
    MINUS_ASSIGN            = r"\-="
    MUL_ASSIGN              = r"\*="
    DIV_ASSIGN              = r"/="
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
    NAME: TokenStr          = _identifier  # pyright: ignore [reportAssignmentType]
    NAME["auto"]            = AUTO
    NAME["break"]           = BREAK
    NAME["case"]            = CASE
    NAME["char"]            = CHAR
    NAME["const"]           = CONST
    NAME["continue"]        = CONTINUE
    NAME["default"]         = DEFAULT
    NAME["do"]              = DO
    NAME["double"]          = DOUBLE
    NAME["else"]            = ELSE
    NAME["enum"]            = ENUM
    NAME["extern"]          = EXTERN
    NAME["float"]           = FLOAT
    NAME["for"]             = FOR
    NAME["goto"]            = GOTO
    NAME["if"]              = IF
    NAME["inline"]          = INLINE
    NAME["int"]             = INT
    NAME["long"]            = LONG
    NAME["register"]        = REGISTER
    NAME["restrict"]        = RESTRICT
    NAME["return"]          = RETURN
    NAME["short"]           = SHORT
    NAME["signed"]          = SIGNED
    NAME["sizeof"]          = SIZEOF
    NAME["static"]          = STATIC
    NAME["struct"]          = STRUCT
    NAME["switch"]          = SWITCH
    NAME["typedef"]         = TYPEDEF
    NAME["union"]           = UNION
    NAME["unsigned"]        = UNSIGNED
    NAME["void"]            = VOID
    NAME["volatile"]        = VOLATILE
    NAME["while"]           = WHILE

    NAME["_Alignas"]        = ALIGNAS
    NAME["_Alignof"]        = ALIGNOF
    NAME["_Atomic"]         = ATOMIC
    NAME["_Bool"]           = BOOL
    NAME["_Complex"]        = COMPLEX
    NAME["_Generic"]        = GENERIC
    NAME["_Imaginary"]      = IMAGINARY
    NAME["_Noreturn"]       = NORETURN
    NAME["_Static_assert"]  = STATIC_ASSERT
    NAME["_Thread_local"]   = THREAD_LOCAL

    # fmt: on

    def tokenize(self, text: str, lineno: int = 1, index: int = 0) -> Generator[Token]:
        """Tokenize the given C code.

        Raises
        ------
        LexError
            If any of the following happens:
                - An unknown character is encountered.
                - A string literal body ends, via newline or EOF, before the terminating double-quote.
                - A character constant body ends, via newline or EOF, before the terminating single-quote.
                - A character constant contains an invalid escape sequence.
        """

        for tok in super().tokenize(text, lineno, index):
            yield tok

            # TYPE or VARIABLE are emitted lazily when the parser requests an extra token to disambiguate
            # typedef and variable names.
            if tok.type == "NAME":
                id_type = "TYPE" if (tok.value in self.context) else "VARIABLE"
                yield Token(id_type, tok.value, tok.lineno, tok.index, tok.end)

        # Handle EOF for incomplete char constants and string literals.
        if self._char_const_start is not None:
            self.error(self._char_const_start, "Missing terminating ' character.")

        if self._string_literal_start is not None:
            self.error(self._string_literal_start, 'Missing terminating " character.')

    def error(self, t: Token, msg: str | None = None):
        if msg is None:
            msg = f"Illegal character {t.value[0]!r} at index {self.index}."
        raise LexError(msg, t.value, self.index)

    def __init__(self, context: CNameContext):
        self.context = context

        self._char_const_start: Token | None = None
        self._string_literal_start: Token | None = None


class CCharConstantLexer(Lexer):
    """Lexer for finding the end of a C character constant.

    Precondition for usage: The start of the character constant was already found and consumed.
    """

    _char_const_start: Token | None

    tokens = {CHAR_CHAR, INCORRECT_ESCAPE_SEQUENCE, CHAR_CONST_END, MISSING_TERMINATOR}

    ignore_CHAR_CHAR = _escape_sequence

    @_(r"\\")
    def INCORRECT_ESCAPE_SEQUENCE(self, t: Token):
        self.error(t, "Incorrect escape sequence.")

    @_(r"'")
    def CHAR_CONSTANT_END(self, t: Token):
        assert self._char_const_start is not None

        self.pop_state()
        start = self._char_const_start
        self._char_const_start = None
        return Token("CONSTANT", self.text[start.index : t.end], start.lineno, start.index, t.end)

    @_(r"\n")
    def MISSING_TERMINATOR(self, t: Token):
        self.error(t, "Missing terminating ' character.")

    def error(self, t: Token, msg: str | None = None):
        if msg is None:
            msg = f"Illegal character {t.value[0]!r} at index {self.index}."
        raise LexError(msg, t.value, self.index)


class CStringLiteralLexer(Lexer):
    """Lexer for finding the end of a C string literal.

    Precondition for usage: The start of the string literal was already found and consumed.
    """

    _string_literal_start: Token | None

    tokens = {STRING_LITERAL_END, MISSING_TERMINATOR, STRING_CHAR}

    @_(r'"')
    def STRING_LITERAL_END(self, t: Token):
        assert self._string_literal_start is not None

        self.pop_state()
        start = self._string_literal_start
        self._string_literal_start = None
        return Token("STRING_LITERAL", self.text[start.index : t.end], start.lineno, start.index, t.end)

    @_(r"\n")
    def MISSING_TERMINATOR(self, t: Token):
        self.error(t, 'Missing terminating " character.')

    ignore_STRING_CHAR = r"."

    def error(self, t: Token, msg: str | None = None):
        if msg is None:
            msg = f"Illegal character {t.value[0]!r} at index {self.index}."
        raise LexError(msg, t.value, self.index)
