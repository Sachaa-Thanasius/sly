# pyright: reportRedeclaration=none, reportUndefinedVariable=none
"""Module for parsing C tokens into an AST."""

from collections.abc import Generator
from typing import TYPE_CHECKING, Any, NoReturn, Optional, TypedDict, TypeVar, Union, overload

from sly import Parser

from . import c_ast, c_context
from ._cluegen import Datum
from ._typing_compat import NotRequired, Self, override
from .c_lexer import CLexer
from .utils import Coord

if TYPE_CHECKING:
    from sly.types import _, subst

_DeclarationT = TypeVar("_DeclarationT", bound=Union[c_ast.Typedef, c_ast.Decl, c_ast.Typename])
_ModifierT = TypeVar("_ModifierT", bound=c_ast.TypeModifier)


__all__ = ("CParser",)


# ============================================================================
# region -------- Misc
# ============================================================================


class _StructDeclaratorDict(TypedDict):
    decl: Optional[Union[c_ast.TypeDecl, c_ast.TypeModifier, c_ast.Enum, c_ast.Struct, c_ast.Union, c_ast.IdType]]
    bitsize: NotRequired[Optional[c_ast.AST]]
    init: NotRequired[Optional[c_ast.AST]]


class _DeclarationSpecifiers(Datum):
    """Declaration specifiers for C declarations.

    Attributes
    ----------
    qual: list[str], default=[]
        A list of type qualifiers.
    storage: list[str], default=[]
        A list of storage type qualifiers.
    type: list[c_ast.IdType], default=[]
        A list of type specifiers.
    function: list[Any], default=[]
        A list of function specifiers.
    alignment: list[c_ast.Alignas], default=[]
        A list of alignment specifiers.
    """

    qual: list[str] = []
    storage: list[str] = []
    type: list[c_ast.IdType] = []
    function: list[Any] = []
    alignment: list[c_ast.Alignas] = []

    @classmethod
    def add(cls, decl_spec: Optional[Self], new_item: Any, kind: str, *, append: bool = False) -> Self:
        """Given a declaration specifier and a new specifier of a given kind, add the specifier to its respective list.

        If `append` is True, the new specifier is added to the end of the specifiers list, otherwise it's added at the
        beginning. Returns the declaration specifier, with the new specifier incorporated.
        """

        if decl_spec is None:
            return cls(**{kind: [new_item]})
        else:
            subspec_list: list[Any] = getattr(decl_spec, kind)
            if append:
                subspec_list.append(new_item)
            else:
                subspec_list.insert(0, new_item)

            return decl_spec


# endregion


# ============================================================================
# region -------- AST fixers
# ============================================================================


def _fix_atomic_specifiers_once(decl: Any) -> tuple[Any, bool]:
    """Performs one "fix" round of atomic specifiers.
    Returns (modified_decl, found) where found is True iff a fix was made.
    """

    parent = decl
    grandparent: Any = None
    node: Any = decl.type
    while node is not None:
        if isinstance(node, c_ast.Typename) and "_Atomic" in node.quals:
            break

        grandparent = parent
        parent = node
        try:
            node = node.type
        except AttributeError:
            # If we've reached a node without a `type` field, it means we won't
            # find what we're looking for at this point; give up the search
            # and return the original decl unmodified.
            return decl, False

    assert isinstance(parent, c_ast.TypeDecl)
    grandparent.type = node.type
    if "_Atomic" not in node.type.quals:
        node.type.quals.append("_Atomic")
    return decl, True


def fix_atomic_specifiers(decl: Any) -> Any:
    """Atomic specifiers like _Atomic(type) are unusually structured, conferring a qualifier upon the contained type.

    This function fixes a decl with atomic specifiers to have a sane AST structure, by removing spurious
    Typename->TypeDecl pairs and attaching the _Atomic qualifier in the right place.
    """

    # There can be multiple levels of _Atomic in a decl; fix them until a fixed point is reached.
    found = True
    while found:
        decl, found = _fix_atomic_specifiers_once(decl)

    # Make sure to add an _Atomic qual on the topmost decl if needed. Also
    # restore the declname on the innermost TypeDecl (it gets placed in the
    # wrong place during construction).
    typ = decl
    while not isinstance(typ, c_ast.TypeDecl):
        try:
            typ = typ.type
        except AttributeError:
            return decl
    if "_Atomic" in typ.quals and "_Atomic" not in decl.quals:
        decl.quals.append("_Atomic")
    if typ.declname is None:
        typ.declname = decl.name

    return decl


def _extract_nested_case(case_node: Union[c_ast.Case, c_ast.Default], stmts_list: list[Any]) -> None:
    """Recursively extract consecutive Case statements that are made nested by the parser and add them to
    `stmts_list`.
    """

    if isinstance(case_node.stmts[0], (c_ast.Case, c_ast.Default)):
        stmts_list.append(case_node.stmts.pop())
        _extract_nested_case(stmts_list[-1], stmts_list)


def fix_switch_cases(switch_node: c_ast.Switch) -> c_ast.Switch:
    """Fix the mess of case statements created for a switch node by default.

    Parameters
    ----------
    switch_node: c_ast.Switch
        An unfixed switch node. May be modified by the function.

    Returns
    -------
    c_ast.Switch
        The fixed switch node.

    Notes
    -----
    The "case" statements in a "switch" come out of parsing with one
    child node, so subsequent statements are just tucked to the parent
    Compound. Additionally, consecutive (fall-through) case statements
    come out messy. This is a peculiarity of the C grammar.

    The following:

        switch (myvar) {
            case 10:
                k = 10;
                p = k + 1;
                return 10;
            case 20:
            case 30:
                return 20;
            default:
                break;
        }

    creates this tree (pseudo-dump):

        Switch
            ID: myvar
            Compound:
                Case 10:
                    k = 10
                p = k + 1
                return 10
                Case 20:
                    Case 30:
                        return 20
                Default:
                    break

    The goal of this transform is to fix this mess, turning it into the
    following:

        Switch
            ID: myvar
            Compound:
                Case 10:
                    k = 10
                    p = k + 1
                    return 10
                Case 20:
                Case 30:
                    return 20
                Default:
                    break
    """

    if not isinstance(switch_node, c_ast.Switch):
        raise TypeError(switch_node)

    if not isinstance(switch_node.stmt, c_ast.Compound):
        return switch_node

    # The new Compound child for the Switch, which will collect children in the
    # correct order
    new_compound = c_ast.Compound([], coord=switch_node.stmt.coord)

    # The last Case/Default node
    last_case: Optional[Union[c_ast.Case, c_ast.Default]] = None

    # Goes over the children of the Compound below the Switch, adding them
    # either directly below new_compound or below the last Case as appropriate
    # (for `switch(cond) {}`, block_items would have been None)
    for child in switch_node.stmt.block_items or []:
        if isinstance(child, (c_ast.Case, c_ast.Default)):
            # If it's a Case/Default:
            # 1. Add it to the Compound and mark as "last case"
            # 2. If its immediate child is also a Case or Default, promote it
            #    to a sibling.
            new_compound.block_items.append(child)
            _extract_nested_case(child, new_compound.block_items)
            last_case = new_compound.block_items[-1]
        else:
            # Other statements are added as children to the last case, if it
            # exists.
            if last_case is None:
                new_compound.block_items.append(child)
            else:
                last_case.stmts.append(child)

    switch_node.stmt = new_compound
    return switch_node


# endregion


class CParser(Parser):
    debugfile = "sly_cparser.out"

    tokens = CLexer.tokens

    precedence = (
        ("left", LOR),
        ("left", LAND),
        ("left", OR),
        ("left", XOR),
        ("left", AND),
        ("left", EQ, NE),
        ("left", GT, GE, LT, LE),
        ("left", RSHIFT, LSHIFT),
        ("left", PLUS, MINUS),
        ("left", TIMES, DIVIDE, MOD),
    )

    def __init__(self, context: "c_context.CContext") -> None:
        self.context = context

    # ============================================================================
    # region ---- Scope helpers
    # ============================================================================

    def is_type_in_scope(self, name: str) -> bool:
        return self.context.scope_stack.get(name, False)

    def add_identifier_to_scope(self, name: str, coord: Optional[Coord]) -> None:
        """Add a new object, function, or enum member name (i.e. an ID) to the current scope."""

        if self.context.scope_stack.maps[0].get(name, False):
            msg = f"Non-typedef {name!r} previously declared as typedef in this scope."
            self.context.error(msg, coord)

        self.context.scope_stack[name] = False

    def add_typedef_name_to_scope(self, name: str, coord: Optional[Coord]) -> None:
        """Add a new typedef name (i.e. a TYPEID) to the current scope."""

        if not self.context.scope_stack.maps[0].get(name, True):
            msg = f"Typedef {name!r} previously declared as non-typedef in this scope."
            self.context.error(msg, coord)

        self.context.scope_stack[name] = True

    # endregion

    # ============================================================================
    # region ---- AST helpers
    # ============================================================================

    def _fix_decl_name_type(self, decl: _DeclarationT, typename: list[c_ast.IdType]) -> _DeclarationT:
        """Fixes a declaration. Modifies decl."""

        # Reach the underlying basic type
        type_ = decl
        while not isinstance(type_, c_ast.TypeDecl):
            type_ = type_.type

        decl.name = type_.declname
        type_.quals = decl.quals[:]

        # The typename is a list of types. If any type in this list isn't an IdType,
        # it must be the only type in the list (it's illegal to declare "int enum ..").
        # If all the types are basic, they're collected in the IdType holder.
        for tn in typename:
            if not isinstance(tn, c_ast.IdType):
                if len(typename) > 1:
                    msg = "Invalid multiple types specified"
                    self.context.error(msg, tn.coord)
                else:
                    type_.type = tn
                    return decl

        if not typename:
            # Functions default to returning int
            if not isinstance(decl.type, c_ast.FuncDecl):
                msg = "Missing type in declaration"
                self.context.error(msg, decl.coord)

            type_.type = c_ast.IdType(["int"], coord=decl.coord)
        else:
            # At this point, we know that typename is a list of IdType nodes.
            # Concatenate all the names into a single list.
            type_.type = c_ast.IdType([name for id_ in typename for name in id_.names], coord=typename[0].coord)
        return decl

    def _build_declarations(
        self,
        spec: _DeclarationSpecifiers,
        decls: list[_StructDeclaratorDict],
        *,
        typedef_namespace: bool = False,
    ) -> list[c_ast.Decl]:
        """Builds a list of declarations all sharing the given specifiers.

        If typedef_namespace is true, each declared name is added to the "typedef namespace", which also includes
        objects, functions, and enum constants.
        """

        is_typedef = "typedef" in spec.storage
        declarations: list[Any] = []

        decls_0 = decls[0]

        if decls_0.get("bitsize") is not None:
            # Bit-fields are allowed to be unnamed.
            pass

        elif decls_0["decl"] is None:
            # When redeclaring typedef names as identifiers in inner scopes, a problem can occur where the identifier
            # gets grouped into spec.type, leaving decl as None. This can only occur for the first declarator.
            if len(spec.type) < 2 or len(spec.type[-1].names) != 1 or not self.is_type_in_scope(spec.type[-1].names[0]):
                coord = next((t.coord for t in spec.type if hasattr(t, "coord")), "?")
                msg = "Invalid declaration"
                self.context.error(msg, coord)

            # Make this look as if it came from "direct_declarator:ID"
            decls_0["decl"] = c_ast.TypeDecl(
                declname=spec.type[-1].names[0],
                quals=None,
                align=spec.alignment,
                type=None,
                coord=spec.type[-1].coord,
            )
            # Remove the "new" type's name from the end of spec.type
            del spec.type[-1]

        elif not isinstance(decls_0["decl"], (c_ast.Enum, c_ast.Struct, c_ast.Union, c_ast.IdType)):
            # A similar problem can occur where the declaration ends up looking like an abstract declarator.
            # Give it a name if this is the case.
            decls_0_tail = decls_0["decl"]
            while not isinstance(decls_0_tail, c_ast.TypeDecl):
                decls_0_tail = decls_0_tail.type
            if decls_0_tail.declname is None:
                decls_0_tail.declname = spec.type.pop(-1).names[0]

        for decl in decls:
            assert decl["decl"] is not None
            if is_typedef:
                declaration = c_ast.Typedef(
                    name=None,
                    quals=spec.qual,
                    storage=spec.storage,
                    type=decl["decl"],
                    coord=decl["decl"].coord,
                )
            else:
                declaration = c_ast.Decl(
                    name=None,
                    quals=spec.qual,
                    align=spec.alignment,
                    storage=spec.storage,
                    funcspec=spec.function,
                    type=decl["decl"],
                    init=decl.get("init"),
                    bitsize=decl.get("bitsize"),
                    coord=decl["decl"].coord,
                )

            if isinstance(declaration.type, (c_ast.Enum, c_ast.Struct, c_ast.Union, c_ast.IdType)):
                fixed_decl = declaration
            else:
                fixed_decl = self._fix_decl_name_type(declaration, spec.type)

            # Add the type name defined by typedef to a symbol table (for usage in the lexer)
            if typedef_namespace:
                if is_typedef:
                    self.add_typedef_name_to_scope(fixed_decl.name, fixed_decl.coord)
                else:
                    self.add_identifier_to_scope(fixed_decl.name, fixed_decl.coord)

            fixed_decl = fix_atomic_specifiers(fixed_decl)
            declarations.append(fixed_decl)

        return declarations

    def _build_function_definition(self, spec: _DeclarationSpecifiers, decl: Any, param_decls: Any, body: Any) -> Any:
        """Builds a function definition."""

        if "typedef" in spec.storage:
            msg = "Invalid typedef"
            self.context.error(msg, decl.coord)

        declaration = self._build_declarations(spec, decls=[{"decl": decl, "init": None}], typedef_namespace=True)[0]

        return c_ast.FuncDef(decl=declaration, param_decls=param_decls, body=body, coord=decl.coord)

    @overload
    def _type_modify_decl(self, decl: c_ast.TypeDecl, modifier: _ModifierT) -> _ModifierT: ...
    @overload
    def _type_modify_decl(self, decl: _ModifierT, modifier: c_ast.TypeModifier) -> _ModifierT: ...
    def _type_modify_decl(
        self,
        decl: Union[c_ast.TypeDecl, _ModifierT],
        modifier: Union[c_ast.TypeModifier, _ModifierT],
    ) -> _ModifierT:
        """Tacks a type modifier on a declarator, and returns the modified declarator.

        The declarator and modifier may be modified.

        Notes
        -----
        (This is basically an insertion into a linked list.)

        To understand what's going on here, read sections A.8.5 and A.8.6 of K&R2 very carefully.

        A C type consists of a basic type declaration, with a list of modifiers. For example:

            int *c[5];

        The basic declaration here is "int c", and the pointer and the array are the modifiers.

        Basic declarations are represented by `c_ast.TypeDecl`, and the modifiers are by
        `c_ast.FuncDecl`, `c_ast.PtrDecl` and `c_ast.ArrayDecl`.

        The standard states that whenever a new modifier is parsed, it should be added to the end of the list of
        modifiers. For example:

            K&R2 A.8.6.2: Array Declarators

            In a declaration T D where D has the form
                D1 [constant-expression-opt]
            and the type of the identifier in the declaration T D1 is "type-modifier T",
            the type of the identifier of D is "type-modifier array of T".

        This is what this method does. The declarator it receives can be a list of declarators ending with
        `c_ast.TypeDecl`. It tacks the modifier to the end of this list, just before the `TypeDecl`.

        Additionally, the modifier may be a list itself. This is useful for pointers, that can come as a chain from
        the rule "pointer". In this case, the whole modifier list is spliced into the new location.
        """

        modifier_head = modifier

        # The modifier may be a nested list. Reach its tail.
        modifier_tail = modifier
        while modifier_tail.type:
            modifier_tail = modifier_tail.type

        if isinstance(decl, c_ast.TypeDecl):
            # If the decl is a basic type, just tack the modifier onto it.
            modifier_tail.type = decl
            return modifier
        else:
            # Otherwise, the decl is a list of modifiers.
            # Reach its tail and splice the modifier onto the tail, pointing to the underlying basic type.
            decl_tail = decl
            while not isinstance(decl_tail.type, c_ast.TypeDecl):
                decl_tail = decl_tail.type

            modifier_tail.type = decl_tail.type
            decl_tail.type = modifier_head
            return decl

    # endregion

    # ============================================================================
    # region ---- Grammar productions
    #
    # Implementation of the BNF defined in K&R2 A.13
    # ============================================================================

    @_("{ external_declaration }")
    def translation_unit(self, p: Any):
        """Handle a translation unit.

        Notes
        -----
        This allows empty input. Not strictly part of the C99 Grammar, but useful in practice.
        """

        # NOTE: external_declaration is already a list
        return c_ast.File([e for ext_decl in p.external_declaration for e in ext_decl])

    @_("function_definition")
    def external_declaration(self, p: Any):
        """Handle an external declaration.

        Notes
        -----
        Declarations always come as lists (because they can be several in one line), so we wrap the function definition
        into a list as well, to make the return value of external_declaration homogeneous.
        """

        return [p.function_definition]

    @_("declaration")
    def external_declaration(self, p: Any):
        return p.declaration

    @_("pp_directive", "pppragma_directive")
    def external_declaration(self, p: Any):
        return [p[0]]

    @_('";"')
    def external_declaration(self, p: Any) -> list[Any]:
        return []

    @_("static_assert")
    def external_declaration(self, p: Any):
        return [p.static_assert]

    @_('STATIC_ASSERT_ "(" constant_expression [ "," unified_string_literal ] ")"')
    def static_assert(self, p: Any):
        return c_ast.StaticAssert(p.constant_expression, p.unified_string_literal, coord=Coord.from_literal(p, p[0]))

    @_("PP_HASH")
    def pp_directive(self, p: Any):
        msg = "Directives not supported yet"
        self.context.error(msg, Coord.from_literal(p, p.PP_HASH))

    @_("PP_PRAGMA")
    def pppragma_directive(self, p: Any):
        """Handle a preprocessor pragma directive or a _Pragma operator.

        Notes
        -----
        These encompass two types of C99-compatible pragmas:
        - The #pragma directive: `# pragma character_sequence`
        - The _Pragma unary operator: `_Pragma ( " string_literal " )`
        """

        return c_ast.Pragma("", coord=Coord.from_literal(p, p.PP_PRAGMA))

    @_("PP_PRAGMA PP_PRAGMASTR")
    def pppragma_directive(self, p: Any):
        return c_ast.Pragma(p.PP_PRAGMASTR, coord=Coord.from_literal(p, p.PP_PRAGMA))

    @_('PRAGMA_ "(" unified_string_literal ")"')
    def pppragma_directive(self, p: Any):
        return c_ast.Pragma(p.unified_string_literal, coord=Coord.from_literal(p, p.PRAGMA_))

    @_("pppragma_directive { pppragma_directive }")
    def pppragma_directive_list(self, p: Any):
        return [p.pppragma_directive0, *p.pppragma_directive1]

    @_("[ declaration_specifiers ] id_declarator { declaration } compound_statement")
    def function_definition(self, p: Any):
        """Handle a function declaration.

        Notes
        -----
        In function definitions, the declarator can be followed by a declaration list, for old "K&R style"
        function definitions.
        """

        if p.declaration_specifiers:
            spec: _DeclarationSpecifiers = p.declaration_specifiers
        else:
            # no declaration specifiers - "int" becomes the default type
            spec = _DeclarationSpecifiers(type=[c_ast.IdType(["int"], coord=p.id_declarator.coord)])

        return self._build_function_definition(
            spec=spec,
            decl=p.id_declarator,
            param_decls=[decl for decl_list in p.declaration for decl in decl_list],
            body=p.compound_statement,
        )

    @_(
        "labeled_statement",
        "expression_statement",
        "compound_statement",
        "selection_statement",
        "iteration_statement",
        "jump_statement",
        "pppragma_directive",
        "static_assert",
    )
    def statement(self, p: Any):
        """Handle a statement.

        Notes
        -----
        According to C18 A.2.2 6.7.10 static_assert-declaration, _Static_assert is a declaration, not a statement.
        We additionally recognise it as a statement to fix parsing of _Static_assert inside the functions.
        """

        return p[0]

    @_("pppragma_directive_list statement")
    def pragmacomp_or_statement(self, p: Any):
        """Handles a pragma or a statement.

        Notes
        -----
        A pragma is generally considered a decorator rather than an actual statement. Still, for the purposes of
        analyzing an abstract syntax tree of C code, pragma's should not be ignored and were previously treated as a
        statement. This presents a problem for constructs that take a statement such as labeled_statements,
        selection_statements, and iteration_statements, causing a misleading structure in the AST. For example,
        consider the following C code.

            for (int i = 0; i < 3; i++)
                #pragma omp critical
                sum += 1;

        This code will compile and execute "sum += 1;" as the body of the for loop. Previous implementations of
        PyCParser would render the AST for this block of code as follows:

            For:
                DeclList:
                    Decl: i, [], [], []
                        TypeDecl: i, []
                            IdentifierType: ["int"]
                    Constant: int, 0
                BinaryOp: <
                    ID: i
                    Constant: int, 3
                UnaryOp: p++
                    ID: i
                Pragma: omp critical
            Assignment: +=
                ID: sum
                Constant: int, 1

        This AST misleadingly takes the Pragma as the body of the loop, and the assignment then becomes a sibling of
        the loop.

        To solve edge cases like these, the pragmacomp_or_statement rule groups a pragma and its following statement
        (which would otherwise be orphaned) using a compound block, effectively turning the above code into:

            for (int i = 0; i < 3; i++) {
                #pragma omp critical
                sum += 1;
            }
        """

        return c_ast.Compound(
            block_items=[*p.pppragma_directive_list, p.statement],
            coord=p.pppragma_directive_list.coord,
        )

    @_("statement")
    def pragmacomp_or_statement(self, p: Any):
        return p.statement

    @_(
        "declaration_specifiers [ init_declarator_list ]",
        "declaration_specifiers_no_type [ id_init_declarator_list ]",
    )
    def decl_body(self, p: Any):
        """Handle declaration bodies.

        Notes
        -----
        In C, declarations can come several in a line:

            int x, *px, romulo = 5;

        However, for the AST, we will split them to separate Declnodes.

        This rule splits its declarations and always returns a list of Decl nodes, even if it's one element long.
        """

        spec: _DeclarationSpecifiers = p[0]

        # p[1] is either a list or None
        #
        # NOTE: Accessing optional components via index puts the component in a 1-tuple,
        # so it's now being accessed with p[1][0].
        #
        declarator_list = p[1][0]
        if declarator_list is None:
            # By the standard, you must have at least one declarator unless
            # declaring a structure tag, a union tag, or the members of an
            # enumeration.
            #
            ty = spec.type
            if len(ty) == 1 and isinstance(ty[0], (c_ast.Struct, c_ast.Union, c_ast.Enum)):
                decls = [
                    c_ast.Decl(
                        name=None,
                        quals=spec.qual,
                        align=spec.alignment,
                        storage=spec.storage,
                        funcspec=spec.function,
                        type=ty[0],
                        init=None,
                        bitsize=None,
                        coord=ty[0].coord,
                    )
                ]

            # However, this case can also occur on redeclared identifiers in
            # an inner scope.  The trouble is that the redeclared type's name
            # gets grouped into declaration_specifiers; _build_declarations
            # compensates for this.
            #
            else:
                decls = self._build_declarations(spec, decls=[{"decl": None, "init": None}], typedef_namespace=True)

        else:
            decls = self._build_declarations(spec, decls=declarator_list, typedef_namespace=True)

        return decls

    @_('decl_body ";"')
    def declaration(self, p: Any):
        """Handle a declaration.

        Notes
        -----
        The declaration has been split to a decl_body sub-rule and ";", because having them in a single rule created a
        problem for defining typedefs.

        If a typedef line was directly followed by a line using the type defined with the typedef, the type would not
        be recognized. This is because to reduce the declaration rule, the parser's lookahead asked for the token
        after ";", which was the type from the next line, and the lexer had no chance to see the updated type symbol
        table.

        Splitting solves this problem, because after seeing ";", the parser reduces decl_body, which actually adds the
        new type into the table to be seen by the lexer before the next line is reached.
        """

        return p.decl_body

    @_(
        "type_qualifier [ declaration_specifiers_no_type ]",
        "storage_class_specifier [ declaration_specifiers_no_type ]",
        "function_specifier [ declaration_specifiers_no_type ]",
        # Without this, `typedef _Atomic(T) U` will parse incorrectly because the
        # _Atomic qualifier will match instead of the specifier.
        "atomic_specifier [ declaration_specifiers_no_type ]",
        "alignment_specifier [ declaration_specifiers_no_type ]",
    )
    def declaration_specifiers_no_type(self, p: Any):
        """Handle declaration specifiers "without a type".

        Notes
        -----
        To know when declaration-specifiers end and declarators begin,
        we require the following:

        1. declaration-specifiers must have at least one type-specifier
        2. No typedef-names are allowed after we've seen any type-specifier.

        These are both required by the spec.
        """

        # fmt: off
        qualifiers_or_specifiers = {
            "type_qualifier":           "qual",
            "storage_class_specifier":  "storage",
            "function_specifier":       "function",
            "atomic_specifier":         "type",
            "alignment_specifier":      "alignment",
        }
        # fmt: on

        decl_kind = next(kind for qual_or_spec, kind in qualifiers_or_specifiers.items() if hasattr(p, qual_or_spec))
        return _DeclarationSpecifiers.add(p.declaration_specifiers_no_type, p[0], decl_kind)

    @_(
        "declaration_specifiers type_qualifier",
        "declaration_specifiers storage_class_specifier",
        "declaration_specifiers function_specifier",
        "declaration_specifiers type_specifier_no_typeid",
        "declaration_specifiers alignment_specifier",
    )
    def declaration_specifiers(self, p: Any):
        # fmt: off
        qualifiers_or_specifiers = {
            "type_qualifier":           "qual",
            "storage_class_specifier":  "storage",
            "function_specifier":       "function",
            "type_specifier_no_typeid": "type",
            "alignment_specifier":      "alignment",
        }
        # fmt: on

        decl_kind = next(kind for qual_or_spec, kind in qualifiers_or_specifiers.items() if hasattr(p, qual_or_spec))
        return _DeclarationSpecifiers.add(p.declaration_specifiers, p[1], decl_kind, append=True)

    @_("type_specifier")
    def declaration_specifiers(self, p: Any):
        return _DeclarationSpecifiers(type=[p.type_specifier])

    @_("declaration_specifiers_no_type type_specifier")
    def declaration_specifiers(self, p: Any):
        return _DeclarationSpecifiers.add(p.declaration_specifiers_no_type, p.type_specifier, "type", append=True)

    @_("AUTO", "REGISTER", "STATIC", "EXTERN", "TYPEDEF", "THREAD_LOCAL_")
    def storage_class_specifier(self, p: Any):
        return p[0]

    @_("INLINE", "NORETURN_")
    def function_specifier(self, p: Any):
        return p[0]

    @_(
        "VOID",
        "BOOL_",
        "CHAR",
        "SHORT",
        "INT",
        "LONG",
        "FLOAT",
        "DOUBLE",
        "COMPLEX_",
        "SIGNED",
        "UNSIGNED",
        "INT128",
    )
    def type_specifier_no_typeid(self, p: Any):
        return c_ast.IdType([p[0]], coord=Coord.from_literal(p, p[0]))

    @_("typedef_name", "enum_specifier", "struct_or_union_specifier", "type_specifier_no_typeid", "atomic_specifier")
    def type_specifier(self, p: Any):
        return p[0]

    @_('ATOMIC_ "(" type_name ")"')
    def atomic_specifier(self, p: Any):
        """Handle an atomic specifier from C11.

        Notes
        -----
        See section 6.7.2.4 of the C11 standard.
        """

        typ = p.type_name
        typ.quals.append(p.ATOMIC_)
        return typ

    @_("CONST", "RESTRICT", "VOLATILE", "ATOMIC_")
    def type_qualifier(self, p: Any):
        return p[0]

    @_('init_declarator { "," init_declarator }')
    def init_declarator_list(self, p: Any):
        return [p.init_declarator0, *p.init_declarator1]

    @_("declarator [ EQUALS initializer ]")
    def init_declarator(self, p: Any):
        """Handle an init declarator.

        Returns
        -------
        dict[str, Any]
            A {decl=<declarator> : init=<initializer>} dictionary. If there's no initializer, uses None.
        """

        return {"decl": p.declarator, "init": p.initializer}

    @_('id_init_declarator { "," init_declarator }')
    def id_init_declarator_list(self, p: Any):
        return [p.id_init_declarator, *p.init_declarator]

    @_("id_declarator [ EQUALS initializer ]")
    def id_init_declarator(self, p: Any) -> dict[str, Any]:
        return {"decl": p.id_declarator, "init": p.initializer}

    @_("specifier_qualifier_list type_specifier_no_typeid")
    def specifier_qualifier_list(self, p: Any):
        """Handle a specifier qualifier list. At least one type specifier is required."""

        return _DeclarationSpecifiers.add(p.specifier_qualifier_list, p.type_specifier_no_typeid, "type", append=True)

    @_("specifier_qualifier_list type_qualifier")
    def specifier_qualifier_list(self, p: Any):
        return _DeclarationSpecifiers.add(p.specifier_qualifier_list, p.type_qualifier, "qual", append=True)

    @_("type_specifier")
    def specifier_qualifier_list(self, p: Any):
        return _DeclarationSpecifiers(type=[p.type_specifier])

    @_("type_qualifier_list type_specifier")
    def specifier_qualifier_list(self, p: Any):
        return _DeclarationSpecifiers(qual=p.type_qualifier_list, alignment=[p.type_specifier])

    @_("alignment_specifier")
    def specifier_qualifier_list(self, p: Any):
        return _DeclarationSpecifiers(alignment=[p.alignment_specifier])

    @_("specifier_qualifier_list alignment_specifier")
    def specifier_qualifier_list(self, p: Any):
        return _DeclarationSpecifiers.add(p.specifier_qualifier_list, p.alignment_specifier, "alignment")

    @_("struct_or_union ID", "struct_or_union TYPEID")
    def struct_or_union_specifier(self, p: Any):
        """Handle a struct-or-union specifier.

        Notes
        -----
        TYPEID is allowed here (and in other struct/enum related tag names), because
        struct/enum tags reside in their own namespace and can be named the same as types.
        """

        klass = c_ast.Struct if (p.struct_or_union == "struct") else c_ast.Union
        # None means no list of members
        return klass(name=p[1], decls=None, coord=Coord.from_literal(p, p.struct_or_union))

    @_("struct_or_union LBRACE struct_declaration { struct_declaration } RBRACE")
    def struct_or_union_specifier(self, p: Any):
        klass = c_ast.Struct if (p.struct_or_union == "struct") else c_ast.Union
        # Empty sequence means an empty list of members
        decls = [
            decl
            for decl_list in (p.struct_declaration0, *p.struct_declaration1)
            for decl in decl_list
            if decl is not None
        ]
        coord = Coord.from_literal(p, p.struct_or_union)
        return klass(name=None, decls=decls, coord=coord)

    @_(
        "struct_or_union ID LBRACE struct_declaration { struct_declaration } RBRACE",
        "struct_or_union TYPEID LBRACE struct_declaration { struct_declaration } RBRACE",
    )
    def struct_or_union_specifier(self, p: Any):
        klass = c_ast.Struct if (p.struct_or_union == "struct") else c_ast.Union
        # Empty sequence means an empty list of members
        decls = [
            decl
            for decl_list in (p.struct_declaration0, *p.struct_declaration1)
            for decl in decl_list
            if decl is not None
        ]
        coord = Coord.from_literal(p, p.struct_or_union)
        return klass(name=p[1], decls=decls, coord=coord)

    @_("STRUCT", "UNION")
    def struct_or_union(self, p: Any):
        return p[0]

    @_('specifier_qualifier_list [ struct_declarator_list ] ";"')
    def struct_declaration(self, p: Any):
        spec: _DeclarationSpecifiers = p.specifier_qualifier_list
        assert "typedef" not in spec.storage

        if p.struct_declarator_list is not None:
            decls = self._build_declarations(spec, decls=p.struct_declarator_list)

        elif len(spec.type) == 1:
            # Anonymous struct/union: gcc extension, C1x feature.
            # Although the standard only allows structs/unions here, I see no reason to disallow other types since
            # some compilers have typedefs here, and pycparser isn't about rejecting all invalid code.
            #
            node = spec.type[0]
            if isinstance(node, c_ast.AST):
                decl_type = node
            else:
                decl_type = c_ast.IdType(node)

            decls = self._build_declarations(spec, decls=[{"decl": decl_type}])

        else:
            # Structure/union members can have the same names as typedefs. The trouble is that the member's name gets
            # grouped into specifier_qualifier_list; _build_declarations() compensates.
            #
            decls = self._build_declarations(spec, decls=[{"decl": None, "init": None}])

        return decls

    @_('";"')
    def struct_declaration(self, p: Any):
        return None

    @_("pppragma_directive")
    def struct_declaration(self, p: Any):
        return [p.pppragma_directive]

    @_('struct_declarator { "," struct_declarator }')
    def struct_declarator_list(self, p: Any):
        return [p.struct_declarator0, *p.struct_declarator1]

    @_("declarator")
    def struct_declarator(self, p: Any) -> _StructDeclaratorDict:
        """Handle a struct declarator.

        Returns
        -------
        _StructDeclaratorDict
            A dict with the keys "decl" (for the underlying declarator) and "bitsize" (for the bitsize).
        """

        return {"decl": p.declarator, "bitsize": None}

    @_('declarator ":" constant_expression')
    def struct_declarator(self, p: Any) -> _StructDeclaratorDict:
        return {"decl": p.declarator, "bitsize": p.constant_expression}

    @_('":" constant_expression')
    def struct_declarator(self, p: Any) -> _StructDeclaratorDict:
        return {"decl": c_ast.TypeDecl(None, None, None, None), "bitsize": p.constant_expression}

    @_("ENUM ID", "ENUM TYPEID")
    def enum_specifier(self, p: Any):
        return c_ast.Enum(p[1], None, coord=Coord.from_literal(p, p.ENUM))

    @_("ENUM LBRACE enumerator_list RBRACE")
    def enum_specifier(self, p: Any):
        return c_ast.Enum(None, p.enumerator_list, coord=Coord.from_literal(p, p.ENUM))

    @_(
        "ENUM ID LBRACE enumerator_list RBRACE",
        "ENUM TYPEID LBRACE enumerator_list RBRACE",
    )
    def enum_specifier(self, p: Any):
        return c_ast.Enum(p[1], p.enumerator_list, coord=Coord.from_literal(p, p.ENUM))

    @_("enumerator")
    def enumerator_list(self, p: Any):
        return c_ast.EnumeratorList([p.enumerator], coord=p.enumerator.coord)

    @_('enumerator_list ","')
    def enumerator_list(self, p: Any):
        return p.enumerator_list

    @_('enumerator_list "," enumerator')
    def enumerator_list(self, p: Any):
        p.enumerator_list.enumerators.append(p.enumerator)
        return p.enumerator_list

    @_(
        'ALIGNAS_ "(" type_name ")"',
        'ALIGNAS_ "(" constant_expression ")"',
    )
    def alignment_specifier(self, p: Any):
        return c_ast.Alignas(p[2], coord=Coord.from_literal(p, p.ALIGNAS_))

    @_("ID [ EQUALS constant_expression ]")
    def enumerator(self, p: Any):
        enumerator = c_ast.Enumerator(p.ID, p.constant_expression, coord=Coord.from_literal(p, p.ID))
        self.add_identifier_to_scope(enumerator.name, enumerator.coord)
        return enumerator

    @_("id_declarator", "typeid_declarator")
    def declarator(self, p: Any):
        return p[0]

    # ========
    # region -- Experimental usage of `subst()` for $$$_declarator and direct_$$$_declarator rules
    #
    # Note: $$$ is substituted with id, typeid, and typeid_noparen, depending on the rule.
    # ========

    # fmt: off
    subst_ids = subst(
        {"_SUB1": "id",              "_SUB2": "ID"},
        {"_SUB1": "typeid",          "_SUB2": "TYPEID"},
        {"_SUB1": "typeid_noparen",  "_SUB2": "TYPEID"},
    )
    # fmt: on

    @subst_ids
    @_("direct_${_SUB1}_declarator")
    def _SUB1_declarator(self, p: Any) -> Union[c_ast.TypeDecl, c_ast.TypeModifier]:
        return p[0]

    @subst_ids
    @_("pointer direct_${_SUB1}_declarator")
    def _SUB1_declarator(self, p: Any) -> Union[c_ast.TypeDecl, c_ast.TypeModifier]:
        return self._type_modify_decl(p[1], p.pointer)

    @subst_ids
    @_("${_SUB2}")
    def direct__SUB1_declarator(self, p: Any):
        return c_ast.TypeDecl(declname=p[0], type=None, quals=None, align=None, coord=Coord.from_literal(p, p[0]))

    @subst({"_SUB1": "id"}, {"_SUB1": "typeid"})
    @_('"(" ${_SUB1}_declarator ")"')
    def direct__SUB1_declarator(self, p: Any):
        return p[1]

    @subst_ids
    @_('direct_${_SUB1}_declarator "[" [ type_qualifier_list ] [ assignment_expression ] "]"')
    def direct__SUB1_declarator(self, p: Any):
        if p.type_qualifier_list and p.assignment_expression:
            dim = p.assignment_expression
            dim_quals: list[Any] = p.type_qualifier_list or []
        else:
            dim = p.type_qualifier_list or p.assignment_expression
            dim_quals = []

        # Accept dimension qualifiers
        # Per C99 6.7.5.3 p7
        arr = c_ast.ArrayDecl(
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            dim=dim,
            dim_quals=dim_quals,
            coord=p[0].coord,
        )
        return self._type_modify_decl(decl=p[0], modifier=arr)

    @subst_ids
    @_('direct_${_SUB1}_declarator "[" STATIC [ type_qualifier_list ] assignment_expression "]"')
    def direct__SUB1_declarator(self, p: Any):
        listed_quals: Generator[list[Any]] = (
            (item if isinstance(item, list) else [item]) for item in [p.type_qualifier_list, p.assignment_expression]
        )
        dim_quals = [qual for sublist in listed_quals for qual in sublist if qual is not None]
        arr = c_ast.ArrayDecl(
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            dim=p.assignment_expression,
            dim_quals=dim_quals,
            coord=p[0].coord,
        )

        return self._type_modify_decl(decl=p[0], modifier=arr)

    @subst_ids
    @_('direct_${_SUB1}_declarator "[" type_qualifier_list STATIC assignment_expression "]"')
    def direct__SUB1_declarator(self, p: Any):
        listed_quals: Generator[list[Any]] = ((item if isinstance(item, list) else [item]) for item in [p[3], p[4]])
        dim_quals = [qual for sublist in listed_quals for qual in sublist if qual is not None]
        arr = c_ast.ArrayDecl(
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            dim=p.assignment_expression,
            dim_quals=dim_quals,
            coord=p[0].coord,
        )

        return self._type_modify_decl(decl=p[0], modifier=arr)

    @subst_ids
    @_('direct_${_SUB1}_declarator "[" [ type_qualifier_list ] TIMES "]"')
    def direct__SUB1_declarator(self, p: Any):
        """Special for VLAs."""

        arr = c_ast.ArrayDecl(
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            dim=c_ast.Id(p.TIMES, coord=p[0].coord),
            dim_quals=p.type_qualifier_list or [],
            coord=p[0].coord,
        )

        return self._type_modify_decl(decl=p[0], modifier=arr)

    @subst_ids
    @_(
        'direct_${_SUB1}_declarator "(" parameter_type_list ")"',
        'direct_${_SUB1}_declarator "(" [ identifier_list ] ")"',
    )
    def direct__SUB1_declarator(self, p: Any):
        # NOTE: This first line depends on an implementation detail, optional components being in tuples when accessed
        # with numerical index, to determine the difference.
        args = p[2] if not isinstance(p[2], tuple) else p[2][0]
        func = c_ast.FuncDecl(
            args=args,
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            coord=p[0].coord,
        )

        # To see why the lookahead token is needed, consider:
        #   typedef char TT;
        #   void foo(int TT) { TT = 10; }
        # Outside the function, TT is a typedef, but inside (starting and ending with the braces) it's a parameter.
        # The trouble begins with yacc's lookahead token. We don't know if we're declaring or defining a function
        # until we see LBRACE, but if we wait for yacc to trigger a rule on that token, then TT will have already been
        # read and incorrectly interpreted as TYPEID.
        # We need to add the parameters to the scope the moment the lexer sees LBRACE.
        #
        if self.lookahead and (self.lookahead.type == "LBRACE") and (func.args is not None):
            for param in func.args.params:
                if isinstance(param, c_ast.EllipsisParam):
                    break
                self.add_identifier_to_scope(param.name, param.coord)

        return self._type_modify_decl(decl=p[0], modifier=func)

    del subst_ids  # Explicit cleanup: subst() is temporary, but this isn't.

    # endregion

    @_("TIMES [ type_qualifier_list ] [ pointer ]")
    def pointer(self, p: Any):
        """Handle a pointer.

        Notes
        -----
        Pointer decls nest from inside out. This is important when different levels have different qualifiers.
        For example:

            char * const * p;

        Means "pointer to const pointer to char"

        While:

            char ** const p;

        Means "const pointer to pointer to char"

        So when we construct PtrDecl nestings, the leftmost pointer goes in as the most nested type.
        """

        nested_type = c_ast.PtrDecl(
            quals=p.type_qualifier_list or [],
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            coord=Coord.from_literal(p, p.TIMES),
        )

        if p.pointer is not None:
            tail_type = p.pointer
            while tail_type.type is not None:
                tail_type = tail_type.type
            tail_type.type = nested_type
            return p.pointer
        else:
            return nested_type

    @_("type_qualifier { type_qualifier }")
    def type_qualifier_list(self, p: Any):
        return [p.type_qualifier0, *p.type_qualifier1]

    @_("parameter_list")
    def parameter_type_list(self, p: Any):
        return p.parameter_list

    @_('parameter_list "," ELLIPSIS')
    def parameter_type_list(self, p: Any):
        p.parameter_list.params.append(c_ast.EllipsisParam(coord=Coord.from_literal(p, p.ELLIPSIS)))
        return p.parameter_list

    @_('parameter_declaration { "," parameter_declaration }')
    def parameter_list(self, p: Any):
        # single parameter
        return c_ast.ParamList(
            [p.parameter_declaration0, *p.parameter_declaration1], coord=p.parameter_declaration0.coord
        )

    @_(
        "declaration_specifiers id_declarator",
        "declaration_specifiers typeid_noparen_declarator",
    )
    def parameter_declaration(self, p: Any):
        """Handle a parameter declaration.

        Notes
        -----
        From ISO/IEC 9899:TC2, 6.7.5.3.11:

            "If, in a parameter declaration, an identifier can be treated either
            as a typedef name or as a parameter name, it shall be taken as a
            typedef name."

        Inside a parameter declaration, once we've reduced declaration specifiers,
        if we shift in an "(" and see a TYPEID, it could be either an abstract
        declarator or a declarator nested inside parens. This rule tells us to
        always treat it as an abstract declarator. Therefore, we only accept
        `id_declarator`s and `typeid_noparen_declarator`s.
        """

        spec: _DeclarationSpecifiers = p.declaration_specifiers
        if not spec.type:
            spec.type = [c_ast.IdType(["int"], coord=Coord.from_prod(self, p.declaration_specifiers))]
        return self._build_declarations(spec, decls=[{"decl": p[1]}])[0]

    @_("declaration_specifiers [ abstract_declarator ]")
    def parameter_declaration(self, p: Any):
        spec: _DeclarationSpecifiers = p.declaration_specifiers
        if not spec.type:
            spec.type = [c_ast.IdType(["int"], coord=Coord.from_prod(self, p.declaration_specifiers))]

        # Parameters can have the same names as typedefs. The trouble is that the parameter's name gets grouped into
        # declaration_specifiers, making it look like an old-style declaration; compensate.
        if len(spec.type) > 1 and len(spec.type[-1].names) == 1 and self.is_type_in_scope(spec.type[-1].names[0]):
            decl = self._build_declarations(spec, decls=[{"decl": p.abstract_declarator, "init": None}])[0]

        # This truly is an old-style parameter declaration.
        else:
            decl = c_ast.Typename(
                name="",
                quals=spec.qual,
                align=None,
                type=p.abstract_declarator or c_ast.TypeDecl(None, None, None, None),
                coord=Coord.from_prod(self, p.declaration_specifiers),
            )
            typename = spec.type
            decl = self._fix_decl_name_type(decl, typename)

        return decl

    @_('identifier { "," identifier }')
    def identifier_list(self, p: Any):
        return c_ast.ParamList([p.identifier0, *p.identifier1], coord=p.identifier0.coord)

    @_("assignment_expression")
    def initializer(self, p: Any):
        return p.assignment_expression

    @_(
        "LBRACE [ initializer_list ] RBRACE",
        'LBRACE initializer_list "," RBRACE',
    )
    def initializer(self, p: Any):
        if p.initializer_list is None:
            return c_ast.InitList([], coord=Coord.from_literal(p, p[0]))
        else:
            return p.initializer_list

    @_("[ designation ] initializer")
    def initializer_list(self, p: Any):
        init = p.initializer if (p.designation is None) else c_ast.NamedInitializer(p.designation, p.initializer)
        return c_ast.InitList([init], coord=p.initializer.coord)

    @_('initializer_list "," [ designation ] initializer')
    def initializer_list(self, p: Any):
        init = p.initializer if (p.designation is None) else c_ast.NamedInitializer(p.designation, p.initializer)
        p.initializer_list.exprs.append(init)
        return p.initializer_list

    @_("designator_list EQUALS")
    def designation(self, p: Any):
        return p.designator_list

    @_("designator { designator }")
    def designator_list(self, p: Any):
        """Handle a list of designators.

        Notes
        -----
        Designators are represented as a list of nodes, in the order in which
        they're written in the code.
        """

        return [p.designator0, *p.designator1]

    @_('"[" constant_expression "]"', '"." identifier')
    def designator(self, p: Any):
        return p[1]

    @_("specifier_qualifier_list [ abstract_declarator ]")
    def type_name(self, p: Any):
        spec_list: _DeclarationSpecifiers = p.specifier_qualifier_list
        typename = c_ast.Typename(
            name="",
            quals=spec_list.qual[:],
            align=None,
            type=p.abstract_declarator or c_ast.TypeDecl(None, None, None, None),
            coord=Coord.from_prod(self, spec_list),
        )

        return self._fix_decl_name_type(typename, spec_list.type)

    @_("pointer")
    def abstract_declarator(self, p: Any):
        dummytype = c_ast.TypeDecl(None, None, None, None)
        return self._type_modify_decl(decl=dummytype, modifier=p.pointer)

    @_("pointer direct_abstract_declarator")
    def abstract_declarator(self, p: Any):
        return self._type_modify_decl(p.direct_abstract_declarator, p.pointer)

    @_("direct_abstract_declarator")
    def abstract_declarator(self, p: Any):
        return p.direct_abstract_declarator

    # Creating and using direct_abstract_declarator_opt here
    # instead of listing both direct_abstract_declarator and the
    # lack of it in the beginning of _1 and _2 caused two
    # shift/reduce errors.
    #
    @_('"(" abstract_declarator ")"')
    def direct_abstract_declarator(self, p: Any):
        return p.abstract_declarator

    @_('direct_abstract_declarator "[" [ assignment_expression ] "]"')
    def direct_abstract_declarator(self, p: Any):
        arr = c_ast.ArrayDecl(
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            dim=p.assignment_expression,
            dim_quals=[],
            coord=p.direct_abstract_declarator.coord,
        )

        return self._type_modify_decl(decl=p.direct_abstract_declarator, modifier=arr)

    @_('"[" [ type_qualifier_list ] [ assignment_expression ] "]"')
    def direct_abstract_declarator(self, p: Any):
        if p.assignment_expression:
            dim = p.type_qualifier_list
            dim_quals: list[Any] = []
        else:
            dim = p.assignment_expression
            dim_quals = p.type_qualifier_list or []

        type_ = c_ast.TypeDecl(None, None, None, None)
        return c_ast.ArrayDecl(type=type_, dim=dim, dim_quals=dim_quals, coord=Coord.from_literal(p, p[0]))

    @_('direct_abstract_declarator "[" TIMES "]"')
    def direct_abstract_declarator(self, p: Any):
        arr = c_ast.ArrayDecl(
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            dim=c_ast.Id(p.TIMES, coord=Coord.from_literal(p, p.TIMES)),
            dim_quals=[],
            coord=p.direct_abstract_declarator.coord,
        )

        return self._type_modify_decl(decl=p.direct_abstract_declarator, modifier=arr)

    @_('"[" TIMES "]"')
    def direct_abstract_declarator(self, p: Any):
        return c_ast.ArrayDecl(
            type=c_ast.TypeDecl(None, None, None, None),
            dim=c_ast.Id(p[2], coord=Coord.from_literal(p, p[2])),
            dim_quals=[],
            coord=Coord.from_literal(p, p[0]),
        )

    @_('direct_abstract_declarator "(" [ parameter_type_list ] ")"')
    def direct_abstract_declarator(self, p: Any):
        func = c_ast.FuncDecl(
            args=p.parameter_type_list,
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            coord=p.direct_abstract_declarator.coord,
        )
        return self._type_modify_decl(decl=p.direct_abstract_declarator, modifier=func)

    @_('"(" [ parameter_type_list ] ")"')
    def direct_abstract_declarator(self, p: Any):
        return c_ast.FuncDecl(
            args=p.parameter_type_list,
            type=c_ast.TypeDecl(None, None, None, None),
            coord=Coord.from_literal(p, p[0]),
        )

    @_("declaration", "statement")
    def block_item(self, p: Any) -> list[Any]:
        """Handle a block item.

        Notes
        -----
        declaration is a list, statement isn't. To make it consistent, block_item will always be a list.
        """

        item = p[0]
        if isinstance(item, list):
            return item
        else:
            return [item]

    @_("LBRACE { block_item } RBRACE")
    def compound_statement(self, p: Any):
        """Handle a compound statement.

        Notes
        -----
        Since we made block_item a list, this just combines lists. Empty block items (plain ";") produce `[None]`,
        so ignore them.
        """

        block_items = [item for item_list in p.block_item for item in item_list if item is not None]
        return c_ast.Compound(block_items=block_items, coord=Coord.from_literal(p, p[0]))

    @_('ID ":" pragmacomp_or_statement')
    def labeled_statement(self, p: Any):
        return c_ast.Label(p.ID, p.pragmacomp_or_statement, coord=Coord.from_literal(p, p.ID))

    @_('CASE constant_expression ":" pragmacomp_or_statement')
    def labeled_statement(self, p: Any):
        return c_ast.Case(p.constant_expression, [p.pragmacomp_or_statement], coord=Coord.from_literal(p, p.CASE))

    @_('DEFAULT ":" pragmacomp_or_statement')
    def labeled_statement(self, p: Any):
        return c_ast.Default([p.pragmacomp_or_statement], coord=Coord.from_literal(p, p.DEFAULT))

    @_('IF "(" expression ")" pragmacomp_or_statement')
    def selection_statement(self, p: Any):
        return c_ast.If(p[2], p[4], None, coord=Coord.from_literal(p, p.IF))

    @_('IF "(" expression ")" statement ELSE pragmacomp_or_statement')
    def selection_statement(self, p: Any):
        return c_ast.If(p[2], p[4], p[6], coord=Coord.from_literal(p, p.IF))

    @_('SWITCH "(" expression ")" pragmacomp_or_statement')
    def selection_statement(self, p: Any):
        return fix_switch_cases(
            c_ast.Switch(p.expression, p.pragmacomp_or_statement, coord=Coord.from_literal(p, p.SWITCH))
        )

    @_('WHILE "(" expression ")" pragmacomp_or_statement')
    def iteration_statement(self, p: Any):
        return c_ast.While(p.expression, p.pragmacomp_or_statement, coord=Coord.from_literal(p, p.WHILE))

    @_('DO pragmacomp_or_statement WHILE "(" expression ")" ";"')
    def iteration_statement(self, p: Any):
        return c_ast.DoWhile(p.expression, p.pragmacomp_or_statement, coord=Coord.from_literal(p, p.DO))

    @_('FOR "(" [ expression ] ";" [ expression ] ";" [ expression ] ")" pragmacomp_or_statement')
    def iteration_statement(self, p: Any):
        return c_ast.For(p.expression0, p.expression1, p.expression2, p[8], coord=Coord.from_literal(p, p.FOR))

    @_('FOR "(" declaration [ expression ] ";" [ expression ] ")" pragmacomp_or_statement')
    def iteration_statement(self, p: Any):
        coord = Coord.from_literal(p, p.FOR)
        return c_ast.For(
            c_ast.DeclList(p.declaration, coord=coord),
            p.expression0,
            p.expression1,
            p.expression2,
            coord=coord,
        )

    @_('GOTO ID ";"')
    def jump_statement(self, p: Any):
        return c_ast.Goto(p.ID, coord=Coord.from_literal(p, p.GOTO))

    @_('BREAK ";"')
    def jump_statement(self, p: Any):
        return c_ast.Break(coord=Coord.from_literal(p, p.BREAK))

    @_('CONTINUE ";"')
    def jump_statement(self, p: Any):
        return c_ast.Continue(coord=Coord.from_literal(p, p.CONTINUE))

    @_('RETURN [ expression ] ";"')
    def jump_statement(self, p: Any):
        return c_ast.Return(p.expression, coord=Coord.from_literal(p, p.RETURN))

    @_('[ expression ] ";"')
    def expression_statement(self, p: Any):
        if p.expression is None:
            return c_ast.EmptyStatement(coord=Coord.from_literal(p, p[1]))
        else:
            return p.expression

    @_("assignment_expression")
    def expression(self, p: Any):
        return p.assignment_expression

    @_('expression "," assignment_expression')
    def expression(self, p: Any):
        if not isinstance(p.expression, c_ast.ExprList):
            p.expression = c_ast.ExprList([p.expression], coord=p.expression.coord)

        p.expression.exprs.append(p.assignment_expression)
        return p.expression

    @_("TYPEID")
    def typedef_name(self, p: Any):
        return c_ast.IdType([p.TYPEID], coord=Coord.from_literal(p, p.TYPEID))

    @_('"(" compound_statement ")"')
    def assignment_expression(self, p: Any):
        # TODO: Verify that the original name "parenthesized_compound_expression", isn't meaningful.
        return p.compound_statement

    @_("conditional_expression")
    def assignment_expression(self, p: Any):
        return p.conditional_expression

    @_("unary_expression assignment_operator assignment_expression")
    def assignment_expression(self, p: Any):
        return c_ast.Assignment(
            p.assignment_operator,
            p.unary_expression,
            p.assignment_expression,
            coord=p.assignment_operator.coord,
        )

    @_(
        "EQUALS",
        "XOREQUAL",
        "TIMESEQUAL",
        "DIVEQUAL",
        "MODEQUAL",
        "PLUSEQUAL",
        "MINUSEQUAL",
        "LSHIFTEQUAL",
        "RSHIFTEQUAL",
        "ANDEQUAL",
        "OREQUAL",
    )
    def assignment_operator(self, p: Any):
        """Handle assignment operators.

        Notes
        -----
        K&R2 defines these as many separate rules, to encode precedence and associativity. However, in our case,
        SLY's built-in precedence/associativity specification feature can take care of it.
        (see precedence declaration above)
        """

        return p[0]

    @_("conditional_expression")
    def constant_expression(self, p: Any):
        return p.conditional_expression

    @_("binary_expression")
    def conditional_expression(self, p: Any):
        return p.binary_expression

    @_('binary_expression CONDOP expression ":" conditional_expression')
    def conditional_expression(self, p: Any):
        return c_ast.TernaryOp(
            p.binary_expression,
            p.expression,
            p.conditional_expression,
            coord=p.binary_expression.coord,
        )

    @_("cast_expression")
    def binary_expression(self, p: Any):
        return p.cast_expression

    @_(
        "binary_expression TIMES binary_expression",
        "binary_expression DIVIDE binary_expression",
        "binary_expression MOD binary_expression",
        "binary_expression PLUS binary_expression",
        "binary_expression MINUS binary_expression",
        "binary_expression RSHIFT binary_expression",
        "binary_expression LSHIFT binary_expression",
        "binary_expression LT binary_expression",
        "binary_expression LE binary_expression",
        "binary_expression GE binary_expression",
        "binary_expression GT binary_expression",
        "binary_expression EQ binary_expression",
        "binary_expression NE binary_expression",
        "binary_expression AND binary_expression",
        "binary_expression OR binary_expression",
        "binary_expression XOR binary_expression",
        "binary_expression LAND binary_expression",
        "binary_expression LOR binary_expression",
    )
    def binary_expression(self, p: Any):
        return c_ast.BinaryOp(p[1], p[0], p[2], coord=p[0].coord)

    @_("unary_expression")
    def cast_expression(self, p: Any):
        return p.unary_expression

    @_('"(" type_name ")" cast_expression')
    def cast_expression(self, p: Any):
        return c_ast.Cast(p.type_name, p.cast_expression, coord=Coord.from_literal(p, p[0]))

    @_("postfix_expression")
    def unary_expression(self, p: Any):
        return p.postfix_expression

    @_("PLUSPLUS unary_expression", "MINUSMINUS unary_expression", "unary_operator cast_expression")
    def unary_expression(self, p: Any):
        return c_ast.UnaryOp(p[0], p[1], coord=p[1].coord)

    @_("SIZEOF unary_expression")
    def unary_expression(self, p: Any):
        return c_ast.UnaryOp(p[0], p[1], coord=Coord.from_literal(p, p.SIZEOF))

    @_('SIZEOF "(" type_name ")"', 'ALIGNOF_ "(" type_name ")"')
    def unary_expression(self, p: Any):
        return c_ast.UnaryOp(p[0], p.type_name, coord=Coord.from_literal(p, p[0]))

    @_("AND", "TIMES", "PLUS", "MINUS", "NOT", "LNOT")
    def unary_operator(self, p: Any):
        return p[0]

    @_("primary_expression")
    def postfix_expression(self, p: Any):
        return p.primary_expression

    @_('postfix_expression "[" expression "]"')
    def postfix_expression(self, p: Any):
        return c_ast.ArrayRef(p.postfix_expression, p.expression, coord=p.postfix_expression.coord)

    @_('postfix_expression "(" assignment_expression { "," assignment_expression } ")"')
    def postfix_expression(self, p: Any):
        arg_expr = c_ast.ExprList(
            [p.assignment_expression0, *p.assignment_expression1],
            coord=p.assignment_expression0.coord,
        )
        return c_ast.FuncCall(p.postfix_expression, arg_expr, coord=p.postfix_expression.coord)

    @_(
        'postfix_expression "." ID',
        'postfix_expression "." TYPEID',
        "postfix_expression ARROW ID",
        "postfix_expression ARROW TYPEID",
    )
    def postfix_expression(self, p: Any):
        field = c_ast.Id(p[2], coord=Coord.from_literal(p, p[2]))
        return c_ast.StructRef(p.postfix_expression, p[1], field, coord=p.postfix_expression.coord)

    @_("postfix_expression PLUSPLUS", "postfix_expression MINUSMINUS")
    def postfix_expression(self, p: Any):
        return c_ast.UnaryOp("p" + p[1], p.postfix_expression, coord=p[1].coord)

    @_('"(" type_name ")" LBRACE initializer_list [ "," ] RBRACE')
    def postfix_expression(self, p: Any):
        return c_ast.CompoundLiteral(p.type_name, p.initializer_list)

    @_("identifier", "constant", "unified_string_literal", "unified_wstring_literal")
    def primary_expression(self, p: Any):
        return p[0]

    @_('"(" expression ")"')
    def primary_expression(self, p: Any):
        return p.expression

    @_('OFFSETOF "(" type_name "," offsetof_member_designator ")"')
    def primary_expression(self, p: Any):
        coord = Coord.from_literal(p, p.OFFSETOF)
        return c_ast.FuncCall(
            c_ast.Id(p.OFFSETOF, coord=coord),
            c_ast.ExprList([p.type_name, p.offsetof_member_designator], coord=coord),
            coord=coord,
        )

    @_("identifier")
    def offsetof_member_designator(self, p: Any):
        return p.identifier

    @_('offsetof_member_designator "." identifier')
    def offsetof_member_designator(self, p: Any):
        return c_ast.StructRef(
            p.offsetof_member_designator,
            p[1],
            p.identifer,
            coord=p.offsetof_member_designator.coord,
        )

    @_('offsetof_member_designator "[" expression "]"')
    def offsetof_member_designator(self, p: Any):
        return c_ast.ArrayRef(p.offsetof_member_designator, p.expression, coord=p.offsetof_member_designator.coord)

    @_("ID")
    def identifier(self, p: Any):
        return c_ast.Id(p.ID, coord=Coord.from_literal(p, p.ID))

    @_("INT_CONST_DEC", "INT_CONST_OCT", "INT_CONST_HEX", "INT_CONST_BIN", "INT_CONST_CHAR")
    def constant(self, p: Any):
        uCount = 0
        lCount = 0
        for x in p[0][-3:]:
            if x in {"l", "L"}:
                lCount += 1
            elif x in {"u", "U"}:
                uCount += 1

        if uCount > 1:
            msg = "Constant cannot have more than one u/U suffix."
            raise ValueError(msg)
        if lCount > 2:
            msg = "Constant cannot have more than two l/L suffix."
            raise ValueError(msg)
        prefix = "unsigned " * uCount + "long " * lCount
        return c_ast.Constant(prefix + "int", p[0], coord=Coord.from_literal(p, p[0]))

    @_("FLOAT_CONST", "HEX_FLOAT_CONST")
    def constant(self, p: Any):
        if "x" in p[0].lower():
            t = "float"
        else:
            if p[0][-1] in {"f", "F"}:
                t = "float"
            elif p[0][-1] in {"l", "L"}:
                t = "long double"
            else:
                t = "double"

        return c_ast.Constant(t, p[0], coord=Coord.from_literal(p, p[0]))

    @_("CHAR_CONST", "WCHAR_CONST", "U8CHAR_CONST", "U16CHAR_CONST", "U32CHAR_CONST")
    def constant(self, p: Any):
        return c_ast.Constant("char", p[0], coord=Coord.from_literal(p, p[0]))

    @_("STRING_LITERAL")
    def unified_string_literal(self, p: Any):
        """Handle "unified" string literals.

        Notes
        -----
        The "unified" string and wstring literal rules are for supporting concatenation of adjacent string literals.
        For example, `"hello " "world"` is seen by the C compiler as a single string literal with the value
        "hello world".
        """

        # single literal
        return c_ast.Constant("string", p[0], coord=Coord.from_literal(p, p.STRING_LITERAL))

    @_("unified_string_literal STRING_LITERAL")
    def unified_string_literal(self, p: Any):
        p.unified_string_literal.value = p.unified_string_literal.value[:-1] + p.STRING_LITERAL[1:]
        return p.unified_string_literal

    @_(
        "WSTRING_LITERAL",
        "U8STRING_LITERAL",
        "U16STRING_LITERAL",
        "U32STRING_LITERAL",
    )
    def unified_wstring_literal(self, p: Any):
        return c_ast.Constant("string", p[0], coord=Coord.from_literal(p, p[0]))

    @_(
        "unified_wstring_literal WSTRING_LITERAL",
        "unified_wstring_literal U8STRING_LITERAL",
        "unified_wstring_literal U16STRING_LITERAL",
        "unified_wstring_literal U32STRING_LITERAL",
    )
    def unified_wstring_literal(self, p: Any):
        p.unified_wstring_literal.value = p.unified_wstring_literal.value.rstrip()[:-1] + p[1][2:]
        og_col_end: int = p.unified_wstring_literal.coord.col_end
        p.unified_wstring_literal.coord.col_end = og_col_end + len(p.unified_wstring_literal.value)
        return p.unified_wstring_literal

    # endregion

    @override
    def error(self, token: Any) -> NoReturn:
        if token:
            msg = "Syntax error."
            location = Coord(getattr(token, "lineno", 0), token.index, token.end)
        else:
            msg = "Parse error in input. EOF."
            location = Coord(-1, -1)

        self.context.error(msg, location, token)
