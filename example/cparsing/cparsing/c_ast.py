# ruff: noqa: A002
"""AST nodes and tools."""

import contextlib
from collections import deque
from collections.abc import Generator, MutableSequence
from io import StringIO
from types import GeneratorType, MemberDescriptorType
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional
from typing import Union as TUnion

from ._cluegen import Datum, all_clues, all_defaults, cluegen
from ._typing_compat import Self, TypeAlias, TypeGuard, override
from .utils import Coord

__all__ = (
    # Nodes
    "AST",
    "File",
    "ExprList",
    "Enumerator",
    "EnumeratorList",
    "ParamList",
    "EllipsisParam",
    "Compound",
    "For",
    "While",
    "DoWhile",
    "Goto",
    "Label",
    "Switch",
    "Case",
    "Default",
    "If",
    "Continue",
    "Break",
    "Return",
    "Assignment",
    "UnaryOp",
    "BinaryOp",
    "TernaryOp",
    "Pragma",
    "Id",
    "Constant",
    "EmptyStatement",
    "ArrayDecl",
    "ArrayRef",
    "Alignas",
    "Cast",
    "CompoundLiteral",
    "Decl",
    "DeclList",
    "Enum",
    "FuncCall",
    "FuncDecl",
    "TypeModifier",
    "FuncDef",
    "IdType",
    "InitList",
    "NamedInitializer",
    "PtrDecl",
    "StaticAssert",
    "Struct",
    "StructRef",
    "TypeDecl",
    "Typedef",
    "Typename",
    "Union",
    # Utilities
    "compare",
    "iter_child_nodes",
    "walk",
    "NodeVisitor",
    "dump",
    "unparse",
)

# ============================================================================
# region -------- AST Nodes --------
# ============================================================================


if TYPE_CHECKING:

    class AST(Datum, kw_only=True):
        _fields: ClassVar[tuple[str, ...]] = ()
        coord: Optional[Coord] = None
else:

    class AST(Datum):
        __slots__ = ("__weakref__", "coord")

        _fields: ClassVar[tuple[str, ...]] = ()

        @classmethod
        def __init_subclass__(cls, **kwargs: object) -> None:
            """Automatically populate `_fields`."""

            super().__init_subclass__(**kwargs)
            cls._fields = tuple(
                name for name, ann in all_clues(cls).items() if not isinstance(ann, MemberDescriptorType)
            )

        @cluegen
        def __init__(cls: type[Self]) -> str:  # pyright: ignore
            clues = all_clues(cls)
            defaults, mutable_defaults = all_defaults(cls, clues)

            args = ((name, f'{name}: {getattr(clue, "__name__", repr(clue))}') for name, clue in clues.items())
            args = ", ".join((f"{arg} = {defaults[name]!r}" if name in defaults else arg) for name, arg in args)
            body = "\n".join(
                (
                    *(f"    self.{name} = {name}" for name in clues if name not in mutable_defaults),
                    *(
                        f"    self.{name} = {name} if {name} is not CLUEGEN_NOTHING else {mutable_defaults[name]}"
                        for name in mutable_defaults
                    ),
                    "    self.coord = coord",
                )
            )
            return f'def __init__(self, {args}{"," if args else ""} *, coord: Optional[int] = None):\n{body}\n'  # noqa: PLE0101

        @override
        def __eq__(self, other: object) -> bool:
            """Return whether two nodes have the same values (disregarding coordinates)."""

            if not isinstance(other, type(self)):
                return NotImplemented

            return compare(self, other)


class File(AST):
    ext: list[AST]


class ExprList(AST):
    exprs: list[AST]


class Enumerator(AST):
    name: str
    value: Optional[AST] = None


class EnumeratorList(AST):
    enumerators: list[Enumerator]


class ParamList(AST):
    params: list[AST]


class EllipsisParam(AST):
    pass


class Compound(AST):
    block_items: Optional[list[AST]]


# -------- Flow control

# ---- Looping


class For(AST):
    init: Optional[AST]
    cond: Optional[AST]
    next: Optional[AST]
    stmt: AST


class While(AST):
    cond: AST
    stmt: AST


class DoWhile(AST):
    cond: AST
    stmt: AST


# ---- Other flow control


class Goto(AST):
    name: str


class Label(AST):
    name: str
    stmt: AST


class Switch(AST):
    cond: AST
    stmt: AST


class Case(AST):
    expr: AST
    stmts: list[AST]


class Default(AST):
    stmts: list[AST]


class If(AST):
    cond: Optional[AST]
    iftrue: Optional[AST]
    iffalse: Optional[AST]


class Continue(AST):
    pass


class Break(AST):
    pass


class Return(AST):
    expr: AST


# -------- Operations


class Assignment(AST):
    op: str
    left: AST
    right: AST


class UnaryOp(AST):
    op: str
    expr: AST


class BinaryOp(AST):
    op: str
    left: AST
    right: AST


class TernaryOp(AST):
    cond: AST
    iftrue: AST
    iffalse: AST


# -------- Base


class Pragma(AST):
    string: str


class Id(AST):
    name: str


class Constant(AST):
    type: str
    value: str


class EmptyStatement(AST):
    pass


# -------- Other


class IdType(AST):
    names: list[str]


# ---- Struct/Union/Enum
class Struct(AST):
    name: Optional[str]
    decls: Optional[list[AST]] = None


class Union(AST):
    name: Optional[str]
    decls: Optional[list[AST]]


class Enum(AST):
    name: Optional[str]
    values: Optional[EnumeratorList]


# ---- Type declaration
class TypeDecl(AST):
    declname: Optional[str] = None
    quals: Optional[list[str]] = []
    align: Optional[Any] = None
    type: Optional[TUnion[IdType, Struct, Union, Enum]] = None


# ---- Type modifiers
class ArrayDecl(AST):
    type: TUnion[TypeDecl, "PtrDecl", "ArrayDecl"]
    dim: Optional[TUnion[Constant, Id]]
    dim_quals: list[str]


class FuncDecl(AST):
    args: Optional[ParamList]
    type: TUnion[TypeDecl, "PtrDecl"]


class PtrDecl(AST):
    quals: list[str]
    type: TUnion[TypeDecl, "PtrDecl", FuncDecl, ArrayDecl]


TypeModifier = TUnion[PtrDecl, FuncDecl, ArrayDecl]


# ---- Parent decl
class Decl(AST):
    name: Optional[str]
    type: TUnion[TypeDecl, TypeModifier, IdType, Struct, Union, Enum]
    quals: list[str] = []
    align: list["Alignas"] = []
    storage: list[str] = []
    funcspec: list[Any] = []
    init: Optional[AST] = None
    bitsize: Optional[AST] = None


# ---- Rest
class ArrayRef(AST):
    name: AST
    subscript: AST


class Alignas(AST):
    alignment: AST


class Cast(AST):
    to_type: AST
    expr: AST


class CompoundLiteral(AST):
    type: AST
    init: AST


class DeclList(AST):
    decls: list[Decl]


class FuncCall(AST):
    name: Id
    args: ExprList


class FuncDef(AST):
    decl: AST
    param_decls: Optional[list[AST]]
    body: AST


class InitList(AST):
    exprs: list[AST] = []


class NamedInitializer(AST):
    name: list[AST]
    expr: AST


class StaticAssert(AST):
    cond: AST
    message: Optional[AST]


class StructRef(AST):
    name: AST
    type: Any
    field: AST


class Typedef(AST):
    name: Optional[str]
    quals: list[str]
    storage: list[Any]
    type: AST


class Typename(AST):
    name: Optional[str]
    quals: list[str]
    align: Optional[Any]
    type: TUnion[TypeDecl, TypeModifier]


# endregion


# ============================================================================
# region -------- AST Tools
#
# Functions and classes that help interact with ASTs, e.g. node visitors.
# ============================================================================


def compare(first_node: TUnion[AST, MutableSequence[AST]], second_node: TUnion[AST, MutableSequence[AST]]) -> bool:
    """Compare two AST nodes for equality, to see if they have the same field structure with the same values.

    This only takes into account fields present in a node's _fields list while ignoring "coord".

    Notes
    -----
    The algorithm is based on https://stackoverflow.com/a/19598419, but modified to be iterative instead of recursive.
    """

    nodes: deque[tuple[TUnion[AST, list[AST], Any], TUnion[AST, list[AST], Any]]] = deque([(first_node, second_node)])

    while nodes:
        node1, node2 = nodes.pop()

        if type(node1) is not type(node2):
            return False

        if isinstance(node1, AST):
            nodes.extend((getattr(node1, field), getattr(node2, field)) for field in node1._fields if field != "coord")
            continue

        if isinstance(node1, list):
            assert isinstance(node2, list)

            # zip(..., strict=True) is only on >=3.10.
            if len(node1) != len(node2):
                return False
            nodes.extend(zip(node1, node2))

            continue

        if node1 != node2:
            return False

    return True


def iter_child_nodes(node: AST) -> Generator[AST, Any, None]:
    for field in node._fields:
        potential_subnode = getattr(node, field)

        if isinstance(potential_subnode, AST):
            yield potential_subnode

        elif isinstance(potential_subnode, list):
            for subsub in potential_subnode:  # pyright: ignore [reportUnknownVariableType]
                if isinstance(subsub, AST):
                    yield subsub


def walk(node: AST) -> Generator[AST, Any, None]:
    stack: deque[AST] = deque([node])
    while stack:
        curr_node = stack.popleft()
        stack.extend(iter_child_nodes(curr_node))
        yield curr_node


class NodeVisitor:
    """Visitor pattern for an AST.

    Implemention is based on a talk by David Beazley called "Generators: The Final Frontier".
    """

    def _visit(self, node: AST) -> Generator[Any, Any, Any]:
        result: Any = getattr(self, f"visit_{type(node).__name__}", self.generic_visit)(node)
        if isinstance(result, GeneratorType):
            result = yield from result
        return result

    def visit(self, node: AST) -> Any:
        """Visit a node."""

        stack: deque[Generator[Any, Any, Any]] = deque([self._visit(node)])
        result: Any = None

        while stack:
            try:
                node = stack[-1].send(result)
            except StopIteration as exc:
                stack.pop()
                result = exc.value
            else:
                stack.append(self._visit(node))
                result = None

        return result

    def generic_visit(self, node: AST) -> Generator[AST, Any, Any]:
        """Called if no explicit visitor function exists for a node."""

        yield from iter_child_nodes(node)


# ========
# region ---- Pretty Printer
# ========


class _NodePrettyPrinter(NodeVisitor):
    def __init__(
        self,
        indent: Optional[TUnion[str, int]] = None,
        *,
        annotate_fields: bool = True,
        include_coords: bool = False,
    ) -> None:
        self.indent = (" " * indent) if isinstance(indent, int) else indent
        self.annotate_fields = annotate_fields
        self.include_coords = include_coords

        self.buffer = StringIO()
        self.indent_level = 0

    @property
    def prefix(self) -> str:
        if self.indent is not None:
            prefix = f"\n{self.indent * self.indent_level}"
        else:
            prefix = ""

        return prefix

    @property
    def sep(self) -> str:
        if self.indent is not None:
            sep = f",\n{self.indent * self.indent_level}"
        else:
            sep = ", "
        return sep

    def __enter__(self):
        return self

    def __exit__(self, *exc_info: object):
        self.buffer.close()

    def write(self, s: str, /) -> None:
        self.buffer.write(s)

    def remove_extra_separator(self) -> None:
        self.buffer.seek(self.buffer.tell() - len(self.sep))
        self.buffer.truncate()

    @contextlib.contextmanager
    def add_indent_level(self, val: int = 1) -> Generator[None]:
        self.indent_level += val
        try:
            yield
        finally:
            self.indent_level -= val

    @contextlib.contextmanager
    def delimit(self, start: str, end: str) -> Generator[None]:
        self.write(start)
        try:
            yield
        finally:
            self.write(end)

    def generic_visit_list(self, list_field: list[Any]) -> Generator[AST, Any, None]:
        if not list_field:
            self.write("[]")
        else:
            with self.add_indent_level(), self.delimit("[", "]"):
                self.write(self.prefix)

                for subfield in list_field:
                    if isinstance(subfield, AST):
                        yield subfield
                    else:
                        # This AST model assumes no nested lists, so this is the only alternative.
                        # TODO: Consider using str() instead of repr() to avoid double-wrapping string subfields?
                        self.write(repr(subfield))

                    self.write(self.sep)

                self.remove_extra_separator()

    @override
    def generic_visit(self, node: AST) -> Generator[Any, Any, None]:
        with self.add_indent_level():
            self.write(f"{type(node).__name__}")

            with self.delimit("(", ")"):
                self.write(self.prefix)

                # Determine which fields will be displayed.
                node_fields = node._fields
                if self.include_coords and node.coord:
                    node_fields += ("coord",)

                for field_name in node_fields:
                    field = getattr(node, field_name)

                    if self.annotate_fields:
                        self.write(f"{field_name}=")

                    if isinstance(field, AST):
                        yield field
                    elif isinstance(field, list):
                        yield from self.generic_visit_list(field)  # pyright: ignore [reportUnknownArgumentType]
                    else:
                        self.write(repr(field))

                    self.write(self.sep)

                self.remove_extra_separator()


def dump(
    node: AST,
    indent: Optional[TUnion[str, int]] = None,
    *,
    annotate_fields: bool = True,
    include_coords: bool = False,
) -> str:
    """Give a formatted string representation of an AST.

    Parameters
    ----------
    node: AST
        The AST to format as a string.
    annotate_fields: bool, default=True
        Whether the returned string will show the names and the values for fields. If False, the result string will be
        more compact by omitting unambiguous field names. Default is True.
    include_coords: bool, default=False
        Whether to display coordinates for each node. Default is False.
    indent: str | None, optional
        The indent level to pretty-print the tree with. Default is None, which selects the single line representation.

    Returns
    -------
    str
        The formatted string representation of the given AST.
    """

    with _NodePrettyPrinter(indent, annotate_fields=annotate_fields, include_coords=include_coords) as visitor:
        visitor.visit(node)
        return visitor.buffer.getvalue()


# endregion

# ========
# region ---- Unparser
# ========


_SimpleNode: TypeAlias = TUnion[Constant, Id, ArrayRef, StructRef, FuncCall]


class _Unparser(NodeVisitor):
    # TODO: Consider either writing to an IO buffer here for consistency in implementation with _NodePrettyPrinter.

    # fmt: off
    precedence_map: ClassVar[dict[str, int]] = {
        "||": 0,
        "&&": 1,
        "|":  2,
        "^":  3,
        "&":  4,
        "==": 5, "!=": 5,
        ">":  6, ">=": 6, "<": 6, "<=": 6,
        ">>": 7, "<<": 7,
        "+":  8, "-":  8,
        "*":  9, "/":  9, "%": 9,
    }
    """Precedence map of binary operators.

    Notes
    -----
    Should be in sync with `c_parser.CParser.precedence`. Higher numbers mean stronger binding.
    """
    # fmt: on

    def __init__(self, *, reduce_parentheses: bool = False) -> None:
        self.reduce_parentheses = reduce_parentheses

        self.indent_level = 0

    @property
    def indent(self) -> str:
        return " " * self.indent_level

    @contextlib.contextmanager
    def add_indent_level(self, val: int = 1) -> Generator[None, Any, None]:
        self.indent_level += val
        try:
            yield
        finally:
            self.indent_level -= val

    @staticmethod
    def is_simple_node(node: AST) -> TypeGuard[_SimpleNode]:
        return isinstance(node, (Constant, Id, ArrayRef, StructRef, FuncCall))

    @override
    def generic_visit(self, node: Optional[AST]) -> Generator[AST, Any, str]:
        if node is not None:
            yield from super().generic_visit(node)
        return ""

    def _visit_expression(self, node: AST) -> Generator[AST, str, str]:
        expr = yield node
        if isinstance(node, InitList):
            return f"{{{expr}}}"
        elif isinstance(node, ExprList):
            return f"({expr})"
        else:
            return f"{expr}"

    def _parenthesize_if(self, node: AST, condition: Callable[[AST], bool]) -> Generator[AST, str, str]:
        """Visits "n" and returns its string representation, parenthesized if the condition function applied to the
        node returns True.
        """

        result = yield from self._visit_expression(node)
        return f"({result})" if condition(node) else result

    def _parenthesize_unless_simple(self, node: AST) -> Generator[AST, str, str]:
        return (yield from self._parenthesize_if(node, lambda n: not self.is_simple_node(n)))

    def _generate_struct_union_body(self, members: list[AST]) -> Generator[AST, str, str]:
        results: list[str] = []
        for decl in members:
            results.append((yield from self._generate_stmt(decl)))
        return "".join(results)

    def _generate_stmt(self, node: AST, add_indent: bool = False) -> Generator[AST, str, str]:
        """Generation from a statement node. This method exists as a wrapper for individual visit_* methods to handle
        different treatment of some statements in this context.
        """

        with contextlib.ExitStack() as ctx:
            if add_indent:
                ctx.enter_context(self.add_indent_level(2))
            indent = self.indent

        result = yield node

        if isinstance(
            node,
            (
                Decl,
                Assignment,
                Cast,
                UnaryOp,
                BinaryOp,
                TernaryOp,
                FuncCall,
                ArrayRef,
                StructRef,
                Constant,
                Id,
                Typedef,
                ExprList,
            ),
        ):
            # These can also appear in an expression context so no semicolon
            # is added to them automatically
            #
            return f"{indent}{result};\n"
        elif isinstance(node, Compound):
            # No extra indentation required before the opening brace of a
            # compound - because it consists of multiple lines it has to
            # compute its own indentation.
            #
            return result
        elif isinstance(node, If):
            return f"{indent}{result}"
        else:
            return f"{indent}{result}\n"

    def _generate_decl(self, node: Decl) -> Generator[Any, str, str]:
        """Generation from a Decl node."""

        results: list[str] = []

        if node.funcspec:
            results.append(" ".join(node.funcspec) + " ")
        if node.storage:
            results.append(" ".join(node.storage) + " ")
        if node.align:
            align = yield node.align[0]
            results.append(f"{align} ")

        results.append((yield from self._generate_type(node.type)))
        return "".join(results)

    def _generate_type(
        self,
        node: AST,
        modifiers: Optional[list[TypeModifier]] = None,
        emit_declname: bool = True,
    ) -> Generator[AST, str, str]:
        """Recursive generation from a type node.

        Parameters
        ----------
        node: AST
            The type node.
        modifiers: Optional[list[TypeModifier]], default=None
            List that collects the PtrDecl, ArrayDecl and FuncDecl modifiers encountered on the way down to a TypeDecl,
            to allow proper generation from it.
        """

        if modifiers is None:
            modifiers = []

        if isinstance(node, TypeDecl):
            results: list[str] = []
            if node.quals:
                results.append(" ".join(node.quals) + " ")
            results.append((yield node.type))

            nstr = [node.declname if (node.declname and emit_declname) else ""]
            # Resolve modifiers.
            # Wrap in parens to distinguish pointer to array and pointer to
            # function syntax.
            #
            for i, modifier in enumerate(modifiers):
                if isinstance(modifier, ArrayDecl):
                    if i != 0 and isinstance(modifiers[i - 1], PtrDecl):
                        nstr.insert(0, "(")
                        nstr.append(")")

                    nstr.append("[")
                    if modifier.dim_quals:
                        nstr.append(" ".join(modifier.dim_quals) + " ")
                    modifier_dim_str = yield modifier.dim
                    nstr.append(f"{modifier_dim_str}]")

                elif isinstance(modifier, FuncDecl):
                    if i != 0 and isinstance(modifiers[i - 1], PtrDecl):
                        nstr.insert(0, "(")
                        nstr.append(")")
                    modifier_args_str = yield modifier.args
                    nstr.append(f"({modifier_args_str})")

                elif isinstance(modifier, PtrDecl):
                    if modifier.quals:
                        modifier_quals_str = " ".join(modifier.quals)
                        if nstr:
                            nstr.insert(0, f"* {modifier_quals_str} ")
                        else:
                            nstr = [f"* {modifier_quals_str}"]
                    else:
                        nstr.insert(0, "*")
            if nstr:
                results.append(" " + "".join(nstr))
            return "".join(results)
        elif isinstance(node, Decl):
            return (yield from self._generate_decl(node.type))
        elif isinstance(node, Typename):
            return (yield from self._generate_type(node.type, emit_declname=emit_declname))
        elif isinstance(node, IdType):
            return " ".join(node.names) + " "
        elif isinstance(node, (ArrayDecl, PtrDecl, FuncDecl)):
            return (yield from self._generate_type(node.type, [*modifiers, node], emit_declname=emit_declname))
        else:
            return (yield node)

    def visit_Constant(self, node: Constant) -> str:
        return node.value

    def visit_Identifier(self, node: Id) -> str:
        return node.name

    def visit_Pragma(self, node: Pragma) -> str:
        ret = ["#pragma"]
        if node.string:
            ret.append(f"{node.string}")
        return " ".join(ret)

    def visit_ArrayRef(self, node: ArrayRef) -> Generator[AST, str, str]:
        arrref = yield from self._parenthesize_unless_simple(node.name)
        subscript = yield node.subscript
        return f"{arrref}[{subscript}]"

    def visit_StructRef(self, node: StructRef) -> Generator[AST, str, str]:
        sref = yield from self._parenthesize_unless_simple(node.name)
        field = yield node.field
        return f"{sref}{node.type}{field}"

    def visit_FuncCall(self, node: FuncCall) -> Generator[AST, str, str]:
        fref = yield from self._parenthesize_unless_simple(node.name)
        args = yield node.args
        return f"{fref}({args})"

    def visit_UnaryOp(self, node: UnaryOp) -> Generator[AST, str, str]:
        if node.op == "sizeof":
            expr = yield node.expr
            return f"sizeof{expr}"
        else:
            operand = yield from self._parenthesize_unless_simple(node.expr)
            if node.op == "p++":
                return f"{operand}++"
            elif node.op == "p--":
                return f"{operand}--"
            else:
                return f"{node.op}{operand}"

    def visit_BinaryOp(self, node: BinaryOp) -> Generator[AST, str, str]:
        # Note: all binary operators are left-to-right associative
        #
        # If `n.left.op` has a stronger or equally binding precedence in
        # comparison to `n.op`, no parenthesis are needed for the left:
        # e.g., `(a*b) + c` is equivalent to `a*b + c`, as well as
        #       `(a+b) - c` is equivalent to `a+b - c` (same precedence).
        # If the left operator is weaker binding than the current, then
        # parentheses are necessary:
        # e.g., `(a+b) * c` is NOT equivalent to `a+b * c`.
        left = yield from self._parenthesize_if(
            node.left,
            lambda d: not (
                self.is_simple_node(d)
                or (
                    self.reduce_parentheses
                    and isinstance(d, BinaryOp)
                    and self.precedence_map[d.op] >= self.precedence_map[node.op]
                )
            ),
        )

        # If `n.right.op` has a stronger -but not equal- binding precedence,
        # parenthesis can be omitted on the right:
        # e.g., `a + (b*c)` is equivalent to `a + b*c`.
        # If the right operator is weaker or equally binding, then parentheses
        # are necessary:
        # e.g., `a * (b+c)` is NOT equivalent to `a * b+c` and
        #       `a - (b+c)` is NOT equivalent to `a - b+c` (same precedence).
        right = yield from self._parenthesize_if(
            node.right,
            lambda d: not (
                self.is_simple_node(d)
                or (
                    self.reduce_parentheses
                    and isinstance(d, BinaryOp)
                    and self.precedence_map[d.op] > self.precedence_map[node.op]
                )
            ),
        )

        return f"{left} {node.op} {right}"

    def visit_Assignment(self, node: Assignment) -> Generator[AST, str, str]:
        right = yield from self._parenthesize_if(node.right, lambda n: isinstance(n, Assignment))
        left = yield node.left
        return f"{left} {node.op} {right}"

    def visit_IdType(self, node: IdType) -> str:
        return " ".join(node.names)

    def visit_Decl(self, node: Decl, *, no_type: bool = False) -> Generator[AST, str, str]:
        # no_type is used when a Decl is part of a DeclList, where the type is
        # explicitly only for the first declaration in a list.
        #
        if no_type:
            result: list[str] = [node.name]
        else:
            decl = yield from self._generate_decl(node)
            result = [decl]

        if node.bitsize:
            bitsize = yield node.bitsize
            result.append(f" : {bitsize}")
        if node.init:
            init = yield from self._visit_expression(node.init)
            result.append(f" = {init}")
        return "".join(result)

    def visit_DeclList(self, node: DeclList) -> Generator[AST, str, str]:
        result: list[str] = []
        result.append((yield node.decls[0]))

        if len(node.decls) > 1:
            result.append(", ")

            decls_list: list[str] = []
            for decl in node.decls[1:]:
                decls_list.append((yield from self.visit_Decl(decl, no_type=True)))
            result.append(", ".join(decls_list))

        return "".join(result)

    def visit_Typedef(self, node: Typedef) -> Generator[AST, str, str]:
        result: list[str] = []
        if node.storage:
            result.append(" ".join(node.storage) + " ")

        result.append((yield from self._generate_type(node.type)))
        return "".join(result)

    def visit_Cast(self, node: Cast) -> Generator[AST, str, str]:
        expr_str = yield from self._parenthesize_unless_simple(node.expr)
        type_ = yield from self._generate_type(node.to_type, emit_declname=False)
        return f"({type_}) {expr_str}"

    def visit_ExprList(self, node: ExprList) -> Generator[AST, str, str]:
        visited_subexprs: list[str] = []
        for expr in node.exprs:
            visited_subexprs.append((yield from self._visit_expression(expr)))

        return ", ".join(visited_subexprs)

    def visit_InitList(self, node: InitList) -> Generator[AST, str, str]:
        visited_subexprs: list[str] = []
        for expr in node.exprs:
            visited_subexprs.append((yield from self._visit_expression(expr)))

        return ", ".join(visited_subexprs)

    def visit_Enum(self, node: Enum) -> Generator[AST, str, str]:
        results: list[str] = [f"enum {(node.name or '')}"]

        members = None if node.values is None else node.values.enumerators
        if members is not None:
            # None means no members
            # Empty sequence means an empty list of members
            results.append("\n")
            results.append(self.indent)
            with self.add_indent_level(2):
                results.append("{\n")

                # `[:-2] + "\n"` removes the final `,` from the enumerator list
                enum_body: list[str] = []
                for value in members:
                    enum_body.append((yield value))
                results.append("".join(enum_body)[:-2] + "\n")

            results.append(self.indent + "}")
        return "".join(results)

    def visit_Alignas(self, node: Alignas) -> Generator[AST, str, str]:
        alignment = yield node.alignment
        return f"_Alignas({alignment})"

    def visit_Enumerator(self, node: Enumerator) -> Generator[AST, str, str]:
        if not node.value:
            return f"{self.indent}{node.name},\n"
        else:
            value = yield node.value
            return f"{self.indent}{node.name} = {value}"

    def visit_FuncDef(self, node: FuncDef) -> Generator[AST, str, str]:
        decl = yield node.decl
        self.indent_level = 0
        body = yield node.body
        if node.param_decls:
            knrdecls: list[str] = []
            for p in node.param_decls:
                knrdecls.append((yield p))

            krndecls_str = ";\n".join(knrdecls)
            return f"{decl}\n{krndecls_str};\n{body}\n"
        else:
            return f"{decl}\n{body}\n"

    def visit_File(self, node: File) -> Generator[AST, str, str]:
        results: list[str] = []
        for ext in node.ext:
            result = yield ext
            if isinstance(ext, FuncDef):
                results.append(result)
            elif isinstance(ext, Pragma):
                results.append(f"{result}\n")
            else:
                results.append(f"{result};\n")
        return "".join(results)

    def visit_Compound(self, node: Compound) -> Generator[AST, str, str]:
        results: list[str] = [self.indent + "{\n"]
        with self.add_indent_level(2):
            if node.block_items:
                block_statements: list[str] = []
                for stmt in node.block_items:
                    block_statements.append((yield from self._generate_stmt(stmt)))

                results.append("".join(block_statements))

        results.append(self.indent + "}\n")
        return "".join(results)

    def visit_CompoundLiteral(self, n: CompoundLiteral) -> Generator[AST, str, str]:
        type_ = yield n.type
        init = yield n.init

        return f"({type_}){{{init}}}"

    def visit_EmptyStatement(self, n: EmptyStatement) -> str:
        return ";"

    def visit_ParamList(self, node: ParamList) -> Generator[AST, str, str]:
        results: list[str] = []
        for param in node.params:
            results.append((yield param))
        return ", ".join(results)

    def visit_Return(self, node: Return) -> Generator[AST, str, str]:
        result = ["return"]
        if node.expr:
            expr = yield node.expr
            result.append(f" {expr}")
        return f"{result};"

    def visit_Break(self, node: Break) -> str:
        return "break;"

    def visit_Continue(self, node: Continue) -> str:
        return "continue;"

    def visit_TernaryOp(self, node: TernaryOp) -> Generator[AST, str, str]:
        cond = yield from self._visit_expression(node.cond)
        iftrue = yield from self._visit_expression(node.iftrue)
        iffalse = yield from self._visit_expression(node.iffalse)

        return f"({cond}) ? ({iftrue}) : ({iffalse})"

    def visit_If(self, n: If) -> Generator[AST, str, str]:
        results = ["if ("]
        if n.cond:
            results.append((yield n.cond))
        results.append(")\n")
        results.append((yield from self._generate_stmt(n.iftrue, add_indent=True)))
        if n.iffalse:
            results.append(f"{self.indent}else\n")
            results.append((yield from self._generate_stmt(n.iffalse, add_indent=True)))
        return "".join(results)

    def visit_For(self, node: For) -> Generator[AST, str, str]:
        results = ["for ("]

        if node.init:
            results.append((yield node.init))
        results.append(";")

        if node.cond:
            cond = yield node.cond
            results.append(f" {cond}")
        results.append(";")

        if node.next:
            next_ = yield node.next
            results.append(f" {next_}")
        results.append(")\n")

        results.append((yield from self._generate_stmt(node.stmt, add_indent=True)))
        return "".join(results)

    def visit_While(self, node: While) -> Generator[AST, str, str]:
        results = ["while ("]
        if node.cond:
            results.append((yield node.cond))
        results.append(")\n")
        results.append((yield from self._generate_stmt(node.stmt, add_indent=True)))
        return "".join(results)

    def visit_DoWhile(self, node: DoWhile) -> Generator[AST, str, str]:
        results = ["do\n"]
        results.append((yield from self._generate_stmt(node.stmt, add_indent=True)))
        results.append(f"{self.indent}while (")

        if node.cond:
            results.append((yield node.cond))
        results.append(");")
        return "".join(results)

    def visit_StaticAssert(self, node: StaticAssert) -> Generator[AST, str, str]:
        results = ["_Static_assert(", (yield node.cond)]

        if node.message:
            results.append(",")
            results.append((yield node.message))
        results.append(")")
        return "".join(results)

    def visit_Switch(self, node: Switch) -> Generator[AST, str, str]:
        cond = yield node.cond
        stmt = yield from self._generate_stmt(node.stmt, add_indent=True)

        return "".join((f"switch ({cond})\n", stmt))

    def visit_Case(self, node: Case) -> Generator[AST, str, str]:
        expr = yield node
        stmts: list[str] = []
        for stmt in node.stmts:
            stmts.append((yield from self._generate_stmt(stmt, add_indent=True)))

        return "".join((f"case {expr}:\n", *stmts))

    def visit_Default(self, node: Default) -> Generator[AST, str, str]:
        s = "default:\n"
        stmts: list[str] = []
        for stmt in node.stmts:
            stmts.append((yield from self._generate_stmt(stmt, add_indent=True)))

        return "".join((s, *stmts))

    def visit_Label(self, node: Label) -> Generator[AST, str, str]:
        stmt = yield from self._generate_stmt(node.stmt)
        return f"{node.name}:\n{stmt}"

    def visit_Goto(self, node: Goto) -> str:
        return f"goto {node.name};"

    def visit_EllipsisParam(self, node: EllipsisParam) -> str:
        return "..."

    def visit_Struct(self, node: Struct) -> Generator[AST, str, str]:
        results: list[str] = [f"struct {(node.name or '')}"]

        members = node.decls
        if members is not None:
            # None means no members
            # Empty sequence means an empty list of members
            results.append("\n")
            results.append(self.indent)
            with self.add_indent_level(2):
                results.append("{\n")
                results.append((yield from self._generate_struct_union_body(members)))

            results.append(self.indent + "}")
        return "".join(results)

    def visit_Union(self, node: Union) -> Generator[AST, str, str]:
        results: list[str] = [f"union {(node.name or '')}"]

        members = node.decls
        if members is not None:
            # None means no members
            # Empty sequence means an empty list of members
            results.append("\n")
            results.append(self.indent)
            with self.add_indent_level(2):
                results.append("{\n")
                results.append((yield from self._generate_struct_union_body(members)))

            results.append(self.indent + "}")
        return "".join(results)

    def visit_Typename(self, node: Typename) -> Generator[AST, str, str]:
        return (yield from self._generate_type(node.type))

    def visit_NamedInitializer(self, node: NamedInitializer) -> Generator[AST, str, str]:
        results: list[str] = []

        for name in node.name:
            if isinstance(name, Id):
                results.append(f".{name.name}")
            else:
                name_str = yield name
                results.append(f"[{name_str}]")

        expr = yield from self._visit_expression(node.expr)
        results.append(f" = {expr}")
        return "".join(results)

    def visit_FuncDecl(self, node: FuncDecl) -> Generator[AST, str, str]:
        return (yield from self._generate_type(node))

    def visit_ArrayDecl(self, node: ArrayDecl) -> Generator[AST, str, str]:
        return (yield from self._generate_type(node, emit_declname=False))

    def visit_TypeDecl(self, node: TypeDecl) -> Generator[AST, str, str]:
        return (yield from self._generate_type(node, emit_declname=False))

    def visit_PtrDecl(self, node: PtrDecl) -> Generator[AST, str, str]:
        return (yield from self._generate_type(node, emit_declname=False))


def unparse(node: AST, *, reduce_parentheses: bool = False) -> str:
    """Unparse an AST object, generating a code string that would produce an equivalent AST object if parsed back."""

    unparser = _Unparser(reduce_parentheses=reduce_parentheses)
    return unparser.visit(node)


# endregion

# endregion
