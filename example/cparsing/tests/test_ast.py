import weakref

from cparsing import c_ast
from cparsing.utils import Coord


def test_BinaryOp():
    b1 = c_ast.BinaryOp(op="+", left=c_ast.Constant(type="int", value="6"), right=c_ast.Id(name="joe"))

    assert isinstance(b1.left, c_ast.Constant)

    assert b1.left.type == "int"
    assert b1.left.value == "6"

    assert isinstance(b1.right, c_ast.Id)
    assert b1.right.name == "joe"


def test_weakref_works_on_nodes():
    c1 = c_ast.Constant(type="float", value="3.14")
    wr = weakref.ref(c1)
    cref = wr()

    assert cref
    assert cref.type == "float"
    assert weakref.getweakrefcount(c1) == 1


def test_weakref_works_on_coord():
    coord = Coord("a", 2, *(0, 0))
    wr = weakref.ref(coord)
    cref = wr()

    assert cref
    assert cref.line_start == 2
    assert weakref.getweakrefcount(coord) == 1


class ConstantVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.values: list[str] = []

    def visit_Constant(self, node: c_ast.Constant) -> None:
        self.values.append(node.value)


def test_scalar_children():
    b1 = c_ast.BinaryOp(op="+", left=c_ast.Constant(type="int", value="6"), right=c_ast.Id(name="joe"))

    cv = ConstantVisitor()
    cv.visit(b1)

    assert cv.values == ["6"]

    b2 = c_ast.BinaryOp(op="*", left=c_ast.Constant(type="int", value="111"), right=b1)
    b3 = c_ast.BinaryOp(op="^", left=b2, right=b1)

    cv = ConstantVisitor()
    cv.visit(b3)

    assert cv.values == ["111", "6", "6"]


def tests_list_children():
    c1 = c_ast.Constant(type="float", value="5.6")
    c2 = c_ast.Constant(type="char", value="t")

    b1 = c_ast.BinaryOp(op="+", left=c1, right=c2)
    b2 = c_ast.BinaryOp(op="-", left=b1, right=c2)

    comp = c_ast.Compound(block_items=[b1, b2, c1, c2])

    cv = ConstantVisitor()
    cv.visit(comp)

    assert cv.values == ["5.6", "t", "5.6", "t", "t", "5.6", "t"]


def test_dump():
    c1 = c_ast.Constant(type="float", value="5.6")
    c2 = c_ast.Constant(type="char", value="t")

    b1 = c_ast.BinaryOp(op="+", left=c1, right=c2)
    b2 = c_ast.BinaryOp(op="-", left=b1, right=c2)

    comp = c_ast.Compound(block_items=[b1, b2, c1, c2])

    expected = """\
Compound(
    block_items=[
        BinaryOp(
            op='+',
            left=Constant(
                type='float',
                value='5.6'),
            right=Constant(
                type='char',
                value='t')),
        BinaryOp(
            op='-',
            left=BinaryOp(
                op='+',
                left=Constant(
                    type='float',
                    value='5.6'),
                right=Constant(
                    type='char',
                    value='t')),
            right=Constant(
                type='char',
                value='t')),
        Constant(
            type='float',
            value='5.6'),
        Constant(
            type='char',
            value='t')])\
"""

    assert c_ast.dump(comp, 4) == expected
