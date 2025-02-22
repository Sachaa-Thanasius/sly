# pyright: reportUndefinedVariable=none, reportRedeclaration=none

from __future__ import annotations

from sly import Parser
from sly.yacc import YaccProduction as Prod

from .context import CNameContext
from .lexer import CLexer


TYPE_CHECKING = False

if TYPE_CHECKING:
    from sly.types import _


def _sanitize_symbol(s: str, /):
    return s.replace("|", "_OR_")


def list_eq1(class_ns: dict[str, object], unique: str, rest: str):
    basename = f"_list_eq1_{_sanitize_symbol(unique)}_{_sanitize_symbol(rest)}"

    if basename in class_ns:
        return basename
    else:
        source = f"""\
@_(
    "{unique} {{ {rest} }}",
    "{rest} {basename}",
)
def {basename}(self, p: Prod): ...
"""
        exec(source, {"Prod": Prod}, class_ns)  # noqa: S102
        return basename


def list_ge1(class_ns: dict[str, object], at_least_one: str, rest: str):
    basename = f"_list_ge1_{_sanitize_symbol(at_least_one)}_{_sanitize_symbol(rest)}"

    if basename in class_ns:
        return basename
    else:
        source = f"""\
@_(
    "{at_least_one} {{ {rest} }}",
    "{at_least_one} {basename}",
    "{rest} {basename}",
)
def {basename}(self, p: Prod): ...
"""
        exec(source, {"Prod": Prod}, class_ns)  # noqa: S102
        return basename


def list_eq1_eq1(class_ns: dict[str, object], unique1: str, unique2: str, rest: str):
    basename = f"_list_eq1_eq1_{_sanitize_symbol(unique1)}_{_sanitize_symbol(unique2)}_{_sanitize_symbol(rest)}"

    if basename in class_ns:
        return basename
    else:
        source = f"""\
@_(
    f"{unique1} {{list_eq1(locals(), {unique2!r}, {rest!r})}}",
    f"{unique2} {{list_eq1(locals(), {unique1!r}, {rest!r})}}",
    "{rest} {basename}",
)
def {basename}(self, p: Prod): ...
"""
        exec(source, {"list_eq1": list_eq1, "Prod": Prod}, class_ns)  # noqa: S102
        return basename


def list_eq1_ge1(class_ns: dict[str, object], unique: str, at_least_one: str, rest: str):
    basename = f"_list_eq1_ge1_{_sanitize_symbol(unique)}_{_sanitize_symbol(at_least_one)}_{_sanitize_symbol(rest)}"

    if basename in class_ns:
        return basename
    else:
        source = f"""\
@_(
    f"{unique} {{list_ge1(locals(), {at_least_one!r}, {rest!r})}}",
    f"{at_least_one} {{list_eq1(locals(), {unique!r}, {rest!r})}}",
    "{at_least_one} {basename}",
    "{rest} {basename}",
)
def {basename}(self, p: Prod): ...
"""
        exec(source, {"list_eq1": list_eq1, "list_ge1": list_ge1, "Prod": Prod}, class_ns)  # noqa: S102
        return basename


class CParser(Parser):
    # debugfile = "examples/c11/parser_debug.out"

    start = "translation_unit_file"

    tokens = CLexer.tokens | {ATOMIC_LPAREN}

    precedence = (
        ("left", BARBAR),
        ("left", ANDAND),
        ("left", BAR),
        ("left", CARET),
        ("left", AND),
        ("left", EQ, NEQ),
        ("left", GT, GEQ, LT, LEQ),
        ("left", RSHIFT, LSHIFT),
        ("left", PLUS, MINUS),
        ("left", STAR, SLASH, PERCENT),
        ("nonassoc", ELSE),
        ("nonassoc", BELOW_ELSE),
    )

    def __init__(self, ctx: CNameContext):
        super().__init__()
        self.ctx = ctx

    @_("")
    def save_context(self, p: Prod):
        self.ctx.save_context()

    @_("")
    def restore_context(self, p: Prod):
        self.ctx.restore_context()

    @_("NAME TYPE")
    def typedef_name(self, p: Prod): ...

    @_("NAME VARIABLE")
    def var_name(self, p: Prod): ...

    @_("typedef_name")
    def typedef_name_spec(self, p: Prod): ...

    @_("typedef_name", "var_name")
    def general_identifier(self, p: Prod): ...

    @_("declarator")
    def declarator_var_name(self, p: Prod):
        declarator = p.declarator
        self.ctx.declare_var_name(declarator)
        return declarator

    @_("declarator")
    def declarator_typedef_name(self, p: Prod):
        declarator = p.declarator
        self.ctx.declare_typedef_name(declarator)
        return declarator

    @_(
        "var_name",
        "CONSTANT",
        "STRING_LITERAL",
        "LPAREN expression RPAREN",
        "generic_selection",
    )
    def primary_expression(self, p: Prod): ...

    @_("GENERIC LPAREN assignment_expression COMMA generic_assoc_list RPAREN")
    def generic_selection(self, p: Prod): ...

    @_("generic_association { COMMA generic_association }")
    def generic_assoc_list(self, p: Prod): ...

    @_(
        "type_name COLON assignment_expression",
        "DEFAULT COLON assignment_expression",
    )
    def generic_association(self, p: Prod): ...

    @_(
        "primary_expression",
        "postfix_expression LBRACK expression RBRACK",
        "postfix_expression LPAREN [ argument_expression_list ] RPAREN",
        "postfix_expression DOT general_identifier",
        "postfix_expression PTR general_identifier",
        "postfix_expression INC",
        "postfix_expression DEC",
        "LPAREN type_name RPAREN LBRACE initializer_list [ COMMA ] RBRACE",
    )
    def postfix_expression(self, p: Prod): ...

    @_("assignment_expression { COMMA assignment_expression }")
    def argument_expression_list(self, p: Prod): ...

    @_(
        "postfix_expression",
        "INC unary_expression",
        "DEC unary_expression",
        "unary_operator cast_expression",
        "SIZEOF cast_expression",
        "SIZEOF LPAREN type_name RPAREN",
        "ALIGNOF LPAREN type_name RPAREN",
    )
    def unary_expression(self, p: Prod): ...

    @_(
        "AND",
        "STAR",
        "PLUS",
        "MINUS",
        "TILDE",
        "BANG",
    )
    def unary_operator(self, p: Prod): ...

    @_(
        "unary_expression",
        "LPAREN type_name RPAREN cast_expression",
    )
    def cast_expression(self, p: Prod): ...

    @_("cast_expression")
    def binary_expression(self, p: Prod): ...

    @_(
        "binary_expression STAR binary_expression",
        "binary_expression SLASH binary_expression",
        "binary_expression PERCENT binary_expression",
        "binary_expression PLUS binary_expression",
        "binary_expression MINUS binary_expression",
        "binary_expression RSHIFT binary_expression",
        "binary_expression LSHIFT binary_expression",
        "binary_expression LT binary_expression",
        "binary_expression LEQ binary_expression",
        "binary_expression GEQ binary_expression",
        "binary_expression GT binary_expression",
        "binary_expression EQ binary_expression",
        "binary_expression NEQ binary_expression",
        "binary_expression AND binary_expression",
        "binary_expression BAR binary_expression",
        "binary_expression CARET binary_expression",
        "binary_expression ANDAND binary_expression",
        "binary_expression BARBAR binary_expression",
    )
    def binary_expression(self, p: Prod): ...

    @_("binary_expression")
    def conditional_expression(self, p: Prod): ...

    @_("binary_expression QUESTION expression COLON conditional_expression")
    def conditional_expression(self, p: Prod): ...

    @_(
        "conditional_expression",
        "unary_expression assignment_operator assignment_expression",
    )
    def assignment_expression(self, p: Prod): ...

    @_(
        "ASSIGN",
        "MUL_ASSIGN",
        "DIV_ASSIGN",
        "MOD_ASSIGN",
        "PLUS_ASSIGN",
        "MINUS_ASSIGN",
        "LSHIFT_ASSIGN",
        "RSHIFT_ASSIGN",
        "AND_ASSIGN",
        "XOR_ASSIGN",
        "OR_ASSIGN",
    )
    def assignment_operator(self, p: Prod): ...

    @_("assignment_expression { COMMA assignment_expression }")
    def expression(self, p: Prod): ...

    @_("conditional_expression")
    def constant_expression(self, p: Prod): ...

    @_(
        "declaration_specifiers [ init_declarator_list ] SEMICOLON",
        "declaration_specifiers_typedef [ init_declarator_list_typedef ] SEMICOLON",
        "static_assert_declaration",
    )
    def declaration(self, p: Prod): ...

    @_(
        "storage_class_specifier",  # deprived of "typedef"
        "type_qualifier",
        "function_specifier",
        "alignment_specifier",
    )
    def declaration_specifier(self, p: Prod): ...

    @_(
        # A list of specifiers with exactly one unique type specifier.
        # "{ declaration_specifier } type_specifier_unique { declaration_specifier }",
        list_eq1(vars(), "type_specifier_unique", "declaration_specifier"),
        # A list of specifiers with one or more nonunique type specifiers.
        # "{ type_specifier_nonunique|declaration_specifier } type_specifier_nonunique { declaration_specifier }",
        list_ge1(vars(), "type_specifier_nonunique", "declaration_specifier"),
    )
    def declaration_specifiers(self, p: Prod): ...

    @_(
        # A list of declaration specifiers with exactly one TYPEDEF and one unique type specifier.
        # "{ declaration_specifier } TYPEDEF { declaration_specifier } type_specifier_unique { declaration_specifier }",
        # "{ declaration_specifier } type_specifier_unique { declaration_specifier } TYPEDEF { declaration_specifier }",
        list_eq1_eq1(vars(), "TYPEDEF", "type_specifier_unique", "declaration_specifier"),
        # A list of declaration specifiers with exactly one TYPEDEF and one or more nonunique unique type specifiers.
        # "{ type_specifier_nonunique|declaration_specifier } TYPEDEF { type_specifier_nonunique|declaration_specifier } type_specifier_nonunique { declaration_specifier }",
        # "{ type_specifier_nonunique|declaration_specifier } type_specifier_nonunique { declaration_specifier } TYPEDEF { declaration_specifier }",
        list_eq1_ge1(vars(), "TYPEDEF", "type_specifier_nonunique", "declaration_specifier"),
    )
    def declaration_specifiers_typedef(self, p: Prod): ...

    @_("init_declarator { COMMA init_declarator }")
    def init_declarator_list(self, p: Prod): ...

    @_("declarator_var_name { ASSIGN c_initializer }")
    def init_declarator(self, p: Prod): ...

    @_("init_declarator_typedef { COMMA init_declarator_typedef }")
    def init_declarator_list_typedef(self, p: Prod): ...

    @_("declarator_typedef_name { ASSIGN c_initializer }")
    def init_declarator_typedef(self, p: Prod): ...

    @_(
        "EXTERN",
        "STATIC",
        "THREAD_LOCAL",
        "AUTO",
        "REGISTER",
    )
    def storage_class_specifier(self, p: Prod): ...

    @_(
        "CHAR",
        "SHORT",
        "INT",
        "LONG",
        "FLOAT",
        "DOUBLE",
        "SIGNED",
        "UNSIGNED",
        "COMPLEX",
    )
    def type_specifier_nonunique(self, p: Prod):
        """A type specifier which can appear together with other type specifiers."""

    @_(
        "VOID",
        "BOOL",
        "INT",
        "atomic_type_specifier",
        "struct_or_union_specifier",
        "enum_specifier",
        "typedef_name_spec",
    )
    def type_specifier_unique(self, p: Prod):
        """A type specifier which cannot appear together with other type specifiers."""

    @_(
        "struct_or_union [ general_identifier ] LBRACK struct_declaration_list RBRACK",
        "struct_or_union general_identifier",
    )
    def struct_or_union_specifier(self, p: Prod): ...

    @_(
        "STRUCT",
        "UNION",
    )
    def struct_or_union(self, p: Prod): ...

    @_("struct_declaration { struct_declaration }")
    def struct_declaration_list(self, p: Prod): ...

    @_(
        "specifier_qualifier_list [ struct_declarator_list ] SEMICOLON",
        "static_assert_declaration",
    )
    def struct_declaration(self, p: Prod): ...

    @_(
        # "{ type_qualifier|alignment_specifier } type_specifier_unique { type_qualifier|alignment_specifier }",
        list_eq1(vars(), "type_specifier_unique", "type_qualifier|alignment_specifier"),
        # "{ type_specifier_nonunique|type_qualifier|alignment_specifier } type_specifier_nonunique { type_qualifier|alignment_specifier }",
        list_ge1(vars(), "type_specifier_nonunique", "type_qualifier|alignment_specifier"),
    )
    def specifier_qualifier_list(self, p: Prod): ...

    @_("struct_declarator { COMMA struct_declarator }")
    def struct_declarator_list(self, p: Prod): ...

    @_(
        "declarator",
        "[ declarator ] COLON constant_expression",
    )
    def struct_declarator(self, p: Prod): ...

    @_(
        "ENUM [ general_identifier ] LBRACE enumerator_list [ COMMA ] RBRACE",
        "ENUM general_identifier",
    )
    def enum_specifier(self, p: Prod): ...

    @_("enumerator { COMMA enumerator }")
    def enumerator_list(self, p: Prod): ...

    @_("enumeration_constant [ ASSIGN constant_expression ]")
    def enumerator(self, p: Prod):
        # | i = enumeration_constant
        # | i = enumeration_constant "=" constant_expression
        #     { declare_varname i }
        self.ctx.declare_var_name(p.enumeration_constant)

    @_("general_identifier")
    def enumeration_constant(self, p: Prod):
        # | i = general_identifier
        #     { i }
        ...

    @_(
        "ATOMIC LPAREN type_name RPAREN",
        "ATOMIC ATOMIC_LPAREN type_name RPAREN",
    )
    def atomic_type_specifier(self, p: Prod): ...

    @_(
        "CONST",
        "RESTRICT",
        "VOLATILE",
        "ATOMIC",
    )
    def type_qualifier(self, p: Prod): ...

    @_(
        "INLINE",
        "NORETURN",
    )
    def function_specifier(self, p: Prod): ...

    @_(
        "ALIGNAS LPAREN type_name RPAREN",
        "ALIGNAS LPAREN constant_expression RPAREN",
    )
    def alignment_specifier(self, p: Prod): ...

    @_("[ pointer ] direct_declarator")
    def declarator(self, p: Prod):
        # | ioption(pointer) d = direct_declarator
        #     { other_declarator d }
        ...

    @_(
        "general_identifier",
        "LPAREN save_context declarator RPAREN",
        "direct_declarator LBRACK [ type_qualifier_list ] [ assignment_expression ] RBRACK",
        "direct_declarator LBRACK STATIC [ type_qualifier_list ] assignment_expression RBRACK",
        "direct_declarator LBRACK type_qualifier_list STATIC assignment_expression RBRACK",
        "direct_declarator LBRACK [ type_qualifier_list ] STAR RBRACK",
        "direct_declarator LPAREN save_context parameter_type_list restore_context RPAREN",
        "direct_declarator LPAREN save_context [ identifier_list ] RPAREN",
    )
    def direct_declarator(self, p: Prod):
        # (* The occurrences of [save_context] inside [direct_declarator] and
        # [direct_abstract_declarator] seem to serve no purpose. In fact, they are
        # required in order to avoid certain conflicts. In other words, we must save
        # the context at this point because the LR automaton is exploring multiple
        # avenues in parallel and some of them do require saving the context. *)

        # direct_declarator:
        # | i = general_identifier
        #     { identifier_declarator i }
        # | "(" save_context d = declarator ")"
        #     { d }
        # | d = direct_declarator "[" type_qualifier_list? assignment_expression? "]"
        # | d = direct_declarator "[" "static" type_qualifier_list? assignment_expression "]"
        # | d = direct_declarator "[" type_qualifier_list "static" assignment_expression "]"
        # | d = direct_declarator "[" type_qualifier_list? "*" "]"
        #     { other_declarator d }
        # | d = direct_declarator "(" ctx = scoped(parameter_type_list) ")"
        #     { function_declarator d ctx }
        # | d = direct_declarator "(" save_context identifier_list? ")"
        #     { other_declarator d }
        ...

    @_("STAR [ type_qualifier_list ] [ pointer ]")
    def pointer(self, p: Prod): ...

    @_("[ type_qualifier_list ] type_qualifier")
    def type_qualifier_list(self, p: Prod): ...

    @_("parameter_list { COMMA ELLIPSIS } save_context")
    def parameter_type_list(self, p: Prod):
        # | parameter_list option("," "..." {}) ctx = save_context
        #     { ctx }
        self.ctx.save_context()

    @_("parameter_declaration { COMMA parameter_declaration }")
    def parameter_list(self, p: Prod): ...

    @_(
        "declaration_specifiers declarator_var_name",
        "declaration_specifiers [ abstract_declarator ]",
    )
    def parameter_declaration(self, p: Prod): ...

    @_("var_name { COMMA var_name }")
    def identifier_list(self, p: Prod): ...

    @_("specifier_qualifier_list [ abstract_declarator ]")
    def type_name(self, p: Prod): ...

    @_(
        "pointer",
        "[ pointer ] direct_abstract_declarator",
    )
    def abstract_declarator(self, p: Prod): ...

    @_(
        "LPAREN save_context abstract_declarator RPAREN",
        "[ direct_abstract_declarator ] LBRACK [ type_qualifier_list ] [ assignment_expression ] RBRACK",
        "[ direct_abstract_declarator ] LBRACK STATIC [ type_qualifier_list ] assignment_expression RBRACK",
        "[ direct_abstract_declarator ] LBRACK type_qualifier_list STATIC assignment_expression RBRACK",
        "[ direct_abstract_declarator ] LBRACK STAR RBRACK",
        "[ direct_abstract_declarator ] LPAREN save_context [ parameter_type_list ] restore_context RPAREN",
    )
    def direct_abstract_declarator(self, p: Prod): ...

    @_(
        "assignment_expression",
        "LBRACE initializer_list [ COMMA ] RBRACE",
    )
    def c_initializer(self, p: Prod): ...

    @_("[ designation ] c_initializer", "initializer_list COMMA [ designation ] c_initializer")
    def initializer_list(self, p: Prod): ...

    @_("designator_list ASSIGN")
    def designation(self, p: Prod): ...

    @_("[ designator_list ] designator")
    def designator_list(self, p: Prod): ...

    @_(
        "LBRACK constant_expression RBRACK",
        "DOT general_identifier",
    )
    def designator(self, p: Prod): ...

    @_("STATIC_ASSERT LPAREN constant_expression COMMA STRING_LITERAL RPAREN SEMICOLON")
    def static_assert_declaration(self, p: Prod): ...

    @_(
        "labeled_statement",
        "save_context compound_statement restore_context",
        "expression_statement",
        "save_context selection_statement restore_context",
        "save_context iteration_statement restore_context",
        "jump_statement",
    )
    def statement(self, p: Prod): ...

    @_(
        "general_identifier COLON statement",
        "CASE constant_expression COLON statement",
        "DEFAULT COLON statement",
    )
    def labeled_statement(self, p: Prod): ...

    @_("LBRACE [ block_item_list ] RBRACE")
    def compound_statement(self, p: Prod): ...

    @_("[ block_item_list ] block_item")
    def block_item_list(self, p: Prod): ...

    @_(
        "declaration",
        "statement",
    )
    def block_item(self, p: Prod): ...

    @_("[ expression_statement ] SEMICOLON")
    def expression_statement(self, p: Prod): ...

    @_(
        "IF LPAREN expression RPAREN save_context statement restore_context ELSE save_context statement restore_context",
        "IF LPAREN expression RPAREN save_context statement restore_context %prec BELOW_ELSE",  # TODO
        "SWITCH LPAREN expression RPAREN save_context statement",
    )
    def selection_statement(self, p: Prod): ...

    @_(
        "WHILE LPAREN expression RPAREN save_context statement restore_context",
        "DO save_context statement restore_context WHILE LPAREN expression RPAREN SEMICOLON",
        "FOR LPAREN [ expression ] SEMICOLON [ expression ] SEMICOLON [ expression ] RPAREN save_context statement restore_context",
        "FOR LPAREN declaration [ expression ] SEMICOLON [ expression ] RPAREN save_context statement restore_context",
    )
    def iteration_statement(self, p: Prod): ...

    @_(
        "GOTO general_identifier SEMICOLON",
        "CONTINUE SEMICOLON",
        "BREAK SEMICOLON",
        "RETURN [ expression ] SEMICOLON",
    )
    def jump_statement(self, p: Prod): ...

    @_("{ external_declaration }")
    def translation_unit_file(self, p: Prod): ...

    @_(
        "function_definition",
        "declaration",
    )
    def external_declaration(self, p: Prod): ...

    @_("declaration_specifiers declarator_var_name")
    def function_definition1(self, p: Prod):
        # { let ctx = save_context () in
        # reinstall_function_context d;
        # ctx }
        # TODO
        ...

    @_("function_definition1 [ declaration_list ] compound_statement")
    def function_definition(self, p: Prod):
        self.ctx.restore_context()
        # TODO

    @_("declaration { declaration }")
    def declaration_list(self, p: Prod): ...
