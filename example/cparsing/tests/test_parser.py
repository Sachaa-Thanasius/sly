from typing import Any, Optional, Union

import pytest
from cparsing import ast as c_ast
from cparsing import parse
from cparsing.parser import CParseError
from cparsing.utils import Coord

# ============================================================================
# region -------- Helpers
# ============================================================================

_MISSING: Any = object()

# from pathlib import Path
# SAMPLE_CFILES_PATH = Path().resolve(strict=True) / "tests" / "c_files"


def DeclWithDefaults(
    name: str,
    type: c_ast.AST,  # noqa: A002
    quals: list[str] = _MISSING,
    align: list[c_ast.Alignas] = _MISSING,
    storage: list[str] = _MISSING,
    funcspec: list[Any] = _MISSING,
    init: Optional[c_ast.AST] = None,
    bitsize: Optional[c_ast.AST] = None,
) -> c_ast.Decl:
    """Create a c_ast.Decl with common defaults for various fields if they aren't passed in.

    Defaults may be added to the actual node eventually, but for now, this eases repetition in testing.
    """

    if quals is _MISSING:
        quals = []
    if align is _MISSING:
        align = []
    if storage is _MISSING:
        storage = []
    if funcspec is _MISSING:
        funcspec = []
    return c_ast.Decl(name, quals, align, storage, funcspec, type, init, bitsize)


# endregion

# ============================================================================
# region -------- Tests
# ============================================================================

# ========
# region ---- Fundamentals
# ========


@pytest.mark.parametrize(
    ("test_input", "expected_length"),
    [
        pytest.param("int a; char c;", 2, id="nonempty file"),
        pytest.param("", 0, id="empty file"),
    ],
)
def test_ast_File(test_input: str, expected_length: int):
    tree = parse(test_input)
    assert isinstance(tree, c_ast.File)
    assert len(tree.ext) == expected_length


def test_empty_toplevel_decl():
    code = "int foo;;"
    tree = parse(code)
    assert isinstance(tree, c_ast.File)
    assert len(tree.ext) == 1

    expected_decl = DeclWithDefaults("foo", c_ast.TypeDecl("foo", [], None, c_ast.IdType(["int"])))
    assert tree.ext[0] == expected_decl


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (";", c_ast.File([])),
        (
            ";int foo;",
            c_ast.File([DeclWithDefaults(name="foo", type=c_ast.TypeDecl("foo", [], None, c_ast.IdType(["int"])))]),
        ),
    ],
)
def test_initial_semi(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    assert tree == expected


@pytest.mark.xfail()
def test_coords():
    """Tests the "coordinates" of parsed elements - file name, line and column numbers, with modification
    inserted by #line directives.
    """

    from cparsing.utils import Coord

    coord1 = parse("int a;").ext[0].coord
    print(coord1)
    assert parse("int a;").ext[0].coord == Coord("", 1, *(5, 0))

    t1 = """\
    int a;
    int b;\n\n
    int c;
    """
    f1 = parse(t1, filename="test.c")
    assert f1.ext[0].coord == Coord("test.c", 2, *(13, 0))
    assert f1.ext[1].coord == Coord("test.c", 3, *(13, 0))
    assert f1.ext[2].coord == Coord("test.c", 6, *(13, 0))

    t1_1 = """\
    int main() {
        k = p;
        printf("%d", b);
        return 0;
    }"""
    f1_1 = parse(t1_1, filename="test.c")
    assert f1_1.ext[0].body.block_items[0].coord == Coord("test.c", 3, *(13, 0))
    assert f1_1.ext[0].body.block_items[1].coord == Coord("test.c", 4, *(13, 0))

    t1_2 = """\
    int main () {
        int p = (int) k;
    }"""
    f1_2 = parse(t1_2, filename="test.c")
    # make sure that the Cast has a coord (issue 23)
    assert f1_2.ext[0].body.block_items[0].init.coord == Coord("test.c", 3, *(21, 0))

    t2 = """\
    #line 99
    int c;
    """
    f2 = parse(t2)
    assert f2.ext[0].coord == Coord("", 99, *(13, 0))

    t3 = """\
    int dsf;
    char p;
    #line 3000 "in.h"
    char d;
    """
    f3 = parse(t3, filename="test.c")
    assert f3.ext[0].coord == Coord("test.c", 2, *(13, 0))
    assert f3.ext[1].coord == Coord("test.c", 3, *(14, 0))
    assert f3.ext[2].coord == Coord("in.h", 3000, *(14, 0))

    t4 = """\
    #line 20 "restore.h"
    int maydler(char);

    #line 30 "includes/daween.ph"
    long j, k;

    #line 50000
    char* ro;
    """
    f4 = parse(t4, filename="myb.c")
    assert f4.ext[0].coord == Coord("restore.h", 20, *(13, 0))
    assert f4.ext[1].coord == Coord("includes/daween.ph", 30, *(14, 0))
    assert f4.ext[2].coord == Coord("includes/daween.ph", 30, *(17, 0))
    assert f4.ext[3].coord == Coord("includes/daween.ph", 50000, *(13, 0))

    t5 = """\
    int
    #line 99
    c;
    """
    f5 = parse(t5)
    assert f5.ext[0].coord == Coord("", 99, *(9, 0))

    # coord for ellipsis
    t6 = """\
    int foo(int j,
            ...) {
    }"""
    f6 = parse(t6)
    assert f6.ext[0].decl.type.args.params[1].coord == Coord("", 3, *(17, 0))


@pytest.mark.xfail()
def test_forloop_coord() -> None:
    from cparsing.utils import Coord

    test_input = """\
void foo() {
    for(int z=0; z<4;
        z++){}
}
"""

    tree = parse(test_input, filename="f.c")
    for_loop = tree.ext[0].body.block_items[0]

    assert isinstance(for_loop, c_ast.For)
    assert for_loop.init.coord == Coord("f.c", 2, 13)
    assert for_loop.cond.coord == Coord("f.c", 2, 26)
    assert for_loop.next.coord == Coord("f.c", 3, 17)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int a;",
            DeclWithDefaults("a", c_ast.TypeDecl("a", [], None, c_ast.IdType(["int"]))),
        ),
        (
            "unsigned int a;",
            DeclWithDefaults("a", c_ast.TypeDecl("a", [], None, c_ast.IdType(["unsigned", "int"]))),
        ),
        (
            "_Bool a;",
            DeclWithDefaults("a", c_ast.TypeDecl("a", [], None, c_ast.IdType(["_Bool"]))),
        ),
        (
            "float _Complex fcc;",
            DeclWithDefaults("fcc", c_ast.TypeDecl("fcc", [], None, c_ast.IdType(["float", "_Complex"]))),
        ),
        (
            "char* string;",
            DeclWithDefaults(
                name="string",
                type=c_ast.PtrDecl([], type=c_ast.TypeDecl("string", [], None, c_ast.IdType(["char"]))),
            ),
        ),
        (
            "long ar[15];",
            DeclWithDefaults(
                name="ar",
                type=c_ast.ArrayDecl(
                    type=c_ast.TypeDecl("ar", [], None, c_ast.IdType(["long"])),
                    dim=c_ast.Constant(type="int", value="15"),
                    dim_quals=[],
                ),
            ),
        ),
        (
            "long long ar[15];",
            DeclWithDefaults(
                name="ar",
                type=c_ast.ArrayDecl(
                    type=c_ast.TypeDecl("ar", [], None, c_ast.IdType(["long", "long"])),
                    dim=c_ast.Constant(type="int", value="15"),
                    dim_quals=[],
                ),
            ),
        ),
        (
            "unsigned ar[];",
            DeclWithDefaults(
                name="ar",
                type=c_ast.ArrayDecl(
                    type=c_ast.TypeDecl("ar", [], None, c_ast.IdType(["unsigned"])),
                    dim=None,
                    dim_quals=[],
                ),
            ),
        ),
        (
            "int strlen(char* s);",
            DeclWithDefaults(
                name="strlen",
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            DeclWithDefaults(
                                name="s",
                                type=c_ast.PtrDecl(
                                    quals=[],
                                    type=c_ast.TypeDecl("s", [], None, c_ast.IdType(["char"])),
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("strlen", [], None, c_ast.IdType(["int"])),
                ),
            ),
        ),
        (
            "int strcmp(char* s1, char* s2);",
            DeclWithDefaults(
                name="strcmp",
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            DeclWithDefaults(
                                name="s1",
                                type=c_ast.PtrDecl(
                                    quals=[],
                                    type=c_ast.TypeDecl("s1", [], None, c_ast.IdType(["char"])),
                                ),
                            ),
                            DeclWithDefaults(
                                name="s2",
                                type=c_ast.PtrDecl(
                                    quals=[],
                                    type=c_ast.TypeDecl("s2", [], None, c_ast.IdType(["char"])),
                                ),
                            ),
                        ]
                    ),
                    type=c_ast.TypeDecl("strcmp", [], None, c_ast.IdType(["int"])),
                ),
            ),
        ),
        pytest.param(
            "extern foobar(foo, bar);",
            DeclWithDefaults(
                name="foobar",
                storage=["extern"],
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList([c_ast.Id("foo"), c_ast.Id("bar")]),
                    type=c_ast.TypeDecl("foobar", [], None, c_ast.IdType(["int"])),
                ),
            ),
            id="function return values and parameters may not have type information",
        ),
        pytest.param(
            "__int128 a;",
            DeclWithDefaults(name="a", type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["__int128"]))),
            id=(
                "__int128: it isn't part of the core C99 or C11 standards, but is mentioned in both documents"
                "under 'Common Extensions'."
            ),
        ),
    ],
)
def test_simple_decls(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    decl = tree.ext[0]
    assert decl == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "char** ar2D;",
            DeclWithDefaults(
                name="ar2D",
                type=c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("ar2D", [], None, c_ast.IdType(["char"]))),
                ),
            ),
        ),
        (
            "int (*a)[1][2];",
            DeclWithDefaults(
                name="a",
                type=c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.ArrayDecl(
                        type=c_ast.ArrayDecl(
                            type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["int"])),
                            dim=c_ast.Constant("int", "2"),
                            dim_quals=[],
                        ),
                        dim=c_ast.Constant("int", "1"),
                        dim_quals=[],
                    ),
                ),
            ),
        ),
        (
            "int *a[1][2];",
            DeclWithDefaults(
                name="a",
                type=c_ast.ArrayDecl(
                    type=c_ast.ArrayDecl(
                        type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["int"]))),
                        dim=c_ast.Constant("int", "2"),
                        dim_quals=[],
                    ),
                    dim=c_ast.Constant("int", "1"),
                    dim_quals=[],
                ),
            ),
        ),
        (
            "char* const* p;",
            DeclWithDefaults(
                name="p",
                type=c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.PtrDecl(quals=["const"], type=c_ast.TypeDecl("p", [], None, c_ast.IdType(["char"]))),
                ),
            ),
        ),
        (
            "const char* const* p;",
            DeclWithDefaults(
                name="p",
                quals=["const"],
                type=c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.PtrDecl(
                        quals=["const"],
                        type=c_ast.TypeDecl("p", quals=["const"], align=None, type=c_ast.IdType(["char"])),
                    ),
                ),
            ),
        ),
        (
            "char* * const p;",
            DeclWithDefaults(
                name="p",
                type=c_ast.PtrDecl(
                    quals=["const"],
                    type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("p", [], None, c_ast.IdType(["char"]))),
                ),
            ),
        ),
        (
            "char ***ar3D[40];",
            DeclWithDefaults(
                name="ar3D",
                type=c_ast.ArrayDecl(
                    type=c_ast.PtrDecl(
                        quals=[],
                        type=c_ast.PtrDecl(
                            quals=[],
                            type=c_ast.PtrDecl(
                                quals=[],
                                type=c_ast.TypeDecl("ar3D", [], None, c_ast.IdType(["char"])),
                            ),
                        ),
                    ),
                    dim=c_ast.Constant("int", "40"),
                    dim_quals=[],
                ),
            ),
        ),
        (
            "char (***ar3D)[40];",
            DeclWithDefaults(
                name="ar3D",
                type=c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.PtrDecl(
                        quals=[],
                        type=c_ast.PtrDecl(
                            quals=[],
                            type=c_ast.ArrayDecl(
                                type=c_ast.TypeDecl("ar3D", [], None, c_ast.IdType(["char"])),
                                dim=c_ast.Constant("int", "40"),
                                dim_quals=[],
                            ),
                        ),
                    ),
                ),
            ),
        ),
        (
            "int (*const*const x)(char, int);",
            DeclWithDefaults(
                name="x",
                type=c_ast.PtrDecl(
                    quals=["const"],
                    type=c_ast.PtrDecl(
                        quals=["const"],
                        type=c_ast.FuncDecl(
                            args=c_ast.ParamList(
                                [
                                    c_ast.Typename(
                                        name=None,
                                        quals=[],
                                        align=None,
                                        type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["char"])),
                                    ),
                                    c_ast.Typename(
                                        name=None,
                                        quals=[],
                                        align=None,
                                        type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["int"])),
                                    ),
                                ]
                            ),
                            type=c_ast.TypeDecl("x", [], None, c_ast.IdType(["int"])),
                        ),
                    ),
                ),
            ),
        ),
        (
            "int (*x[4])(char, int);",
            DeclWithDefaults(
                name="x",
                type=c_ast.ArrayDecl(
                    type=c_ast.PtrDecl(
                        quals=[],
                        type=c_ast.FuncDecl(
                            args=c_ast.ParamList(
                                [
                                    c_ast.Typename(
                                        name=None,
                                        quals=[],
                                        align=None,
                                        type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["char"])),
                                    ),
                                    c_ast.Typename(
                                        name=None,
                                        quals=[],
                                        align=None,
                                        type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["int"])),
                                    ),
                                ]
                            ),
                            type=c_ast.TypeDecl("x", [], None, c_ast.IdType(["int"])),
                        ),
                    ),
                    dim=c_ast.Constant("int", "4"),
                    dim_quals=[],
                ),
            ),
        ),
        (
            "char *(*(**foo [][8])())[];",
            DeclWithDefaults(
                name="foo",
                type=c_ast.ArrayDecl(
                    type=c_ast.ArrayDecl(
                        type=c_ast.PtrDecl(
                            quals=[],
                            type=c_ast.PtrDecl(
                                quals=[],
                                type=c_ast.FuncDecl(
                                    args=None,
                                    type=c_ast.PtrDecl(
                                        quals=[],
                                        type=c_ast.ArrayDecl(
                                            type=c_ast.PtrDecl(
                                                quals=[],
                                                type=c_ast.TypeDecl("foo", [], None, c_ast.IdType(["char"])),
                                            ),
                                            dim=None,
                                            dim_quals=[],
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        dim=c_ast.Constant("int", "8"),
                        dim_quals=[],
                    ),
                    dim=None,
                    dim_quals=[],
                ),
            ),
        ),
        pytest.param(
            "int (*k)(int);",
            DeclWithDefaults(
                name="k",
                type=c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [
                                c_ast.Typename(
                                    name=None,
                                    quals=[],
                                    align=None,
                                    type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["int"])),
                                )
                            ]
                        ),
                        type=c_ast.TypeDecl("k", [], None, c_ast.IdType(["int"])),
                    ),
                ),
            ),
            id="unnamed function pointer parameters w/o quals",
        ),
        pytest.param(
            "int (*k)(const int);",
            DeclWithDefaults(
                name="k",
                type=c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [
                                c_ast.Typename(
                                    name=None,
                                    quals=["const"],
                                    align=None,
                                    type=c_ast.TypeDecl(None, ["const"], None, c_ast.IdType(["int"])),
                                )
                            ]
                        ),
                        type=c_ast.TypeDecl("k", [], None, c_ast.IdType(["int"])),
                    ),
                ),
            ),
            id="unnamed function pointer parameters w/ quals",
        ),
        pytest.param(
            "int (*k)(int q);",
            DeclWithDefaults(
                name="k",
                type=c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [DeclWithDefaults(name="q", type=c_ast.TypeDecl("q", [], None, c_ast.IdType(["int"])))]
                        ),
                        type=c_ast.TypeDecl("k", [], None, c_ast.IdType(["int"])),
                    ),
                ),
            ),
            id="named function pointer parameters w/o quals",
        ),
        pytest.param(
            "int (*k)(const volatile int q);",
            DeclWithDefaults(
                name="k",
                type=c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [
                                DeclWithDefaults(
                                    name="q",
                                    quals=["const", "volatile"],
                                    type=c_ast.TypeDecl("q", ["const", "volatile"], None, c_ast.IdType(["int"])),
                                )
                            ]
                        ),
                        type=c_ast.TypeDecl("k", [], None, c_ast.IdType(["int"])),
                    ),
                ),
            ),
            id="named function pointer parameters w/ quals 1",
        ),
        pytest.param(
            "int (*k)(_Atomic volatile int q);",
            DeclWithDefaults(
                name="k",
                type=c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [
                                DeclWithDefaults(
                                    name="q",
                                    quals=["_Atomic", "volatile"],
                                    type=c_ast.TypeDecl(
                                        "q",
                                        quals=["_Atomic", "volatile"],
                                        align=None,
                                        type=c_ast.IdType(["int"]),
                                    ),
                                )
                            ]
                        ),
                        type=c_ast.TypeDecl("k", [], None, c_ast.IdType(["int"])),
                    ),
                ),
            ),
            id="named function pointer parameters w/ quals 2",
        ),
        pytest.param(
            "int (*k)(const volatile int* q);",
            DeclWithDefaults(
                name="k",
                type=c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [
                                DeclWithDefaults(
                                    name="q",
                                    quals=["const", "volatile"],
                                    type=c_ast.PtrDecl(
                                        quals=[],
                                        type=c_ast.TypeDecl(
                                            "q",
                                            quals=["const", "volatile"],
                                            align=None,
                                            type=c_ast.IdType(["int"]),
                                        ),
                                    ),
                                )
                            ]
                        ),
                        type=c_ast.TypeDecl("k", [], None, c_ast.IdType(["int"])),
                    ),
                ),
            ),
            id="named function pointer parameters w/ quals 3",
        ),
        pytest.param(
            "int (*k)(restrict int* q);",
            DeclWithDefaults(
                name="k",
                type=c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [
                                DeclWithDefaults(
                                    name="q",
                                    quals=["restrict"],
                                    type=c_ast.PtrDecl(
                                        quals=[],
                                        type=c_ast.TypeDecl(
                                            "q",
                                            quals=["restrict"],
                                            align=None,
                                            type=c_ast.IdType(["int"]),
                                        ),
                                    ),
                                )
                            ]
                        ),
                        type=c_ast.TypeDecl("k", [], None, c_ast.IdType(["int"])),
                    ),
                ),
            ),
            id="restrict qualifier",
        ),
    ],
)
def test_nested_decls(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    decl = tree.ext[0]
    assert decl == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        pytest.param(
            "int zz(int p[static 10]);",
            DeclWithDefaults(
                name="zz",
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            DeclWithDefaults(
                                name="p",
                                type=c_ast.ArrayDecl(
                                    type=c_ast.TypeDecl("p", [], None, c_ast.IdType(["int"])),
                                    dim=c_ast.Constant("int", "10"),
                                    dim_quals=["static"],
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("zz", [], None, c_ast.IdType(["int"])),
                ),
            ),
            id="named function parameter 1",
        ),
        pytest.param(
            "int zz(int p[const 10]);",
            DeclWithDefaults(
                name="zz",
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            DeclWithDefaults(
                                name="p",
                                type=c_ast.ArrayDecl(
                                    type=c_ast.TypeDecl("p", [], None, c_ast.IdType(["int"])),
                                    dim=c_ast.Constant("int", "10"),
                                    dim_quals=["const"],
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("zz", [], None, c_ast.IdType(["int"])),
                ),
            ),
            id="named function parameter 2",
        ),
        pytest.param(
            "int zz(int p[restrict][5]);",
            DeclWithDefaults(
                name="zz",
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            DeclWithDefaults(
                                name="p",
                                type=c_ast.ArrayDecl(
                                    type=c_ast.ArrayDecl(
                                        type=c_ast.TypeDecl("p", [], None, c_ast.IdType(["int"])),
                                        dim=c_ast.Constant("int", "5"),
                                        dim_quals=[],
                                    ),
                                    dim=None,
                                    dim_quals=["restrict"],
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("zz", [], None, c_ast.IdType(["int"])),
                ),
            ),
            id="named function parameter 3",
        ),
        pytest.param(
            "int zz(int p[const restrict static 10][5]);",
            DeclWithDefaults(
                name="zz",
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            DeclWithDefaults(
                                name="p",
                                type=c_ast.ArrayDecl(
                                    type=c_ast.ArrayDecl(
                                        type=c_ast.TypeDecl("p", [], None, c_ast.IdType(["int"])),
                                        dim=c_ast.Constant("int", "5"),
                                        dim_quals=[],
                                    ),
                                    dim=c_ast.Constant("int", "10"),
                                    dim_quals=["const", "restrict", "static"],
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("zz", [], None, c_ast.IdType(["int"])),
                ),
            ),
            id="named function parameter 4",
        ),
        pytest.param(
            "int zz(int [const 10]);",
            DeclWithDefaults(
                name="zz",
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Typename(
                                name=None,
                                quals=[],
                                align=None,
                                type=c_ast.ArrayDecl(
                                    type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["int"])),
                                    dim=c_ast.Constant("int", "10"),
                                    dim_quals=["const"],
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("zz", [], None, c_ast.IdType(["int"])),
                ),
            ),
            id="unnamed function parameter 1",
        ),
        pytest.param(
            "int zz(int [restrict][5]);",
            DeclWithDefaults(
                name="zz",
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Typename(
                                name=None,
                                quals=[],
                                align=None,
                                type=c_ast.ArrayDecl(
                                    type=c_ast.ArrayDecl(
                                        type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["int"])),
                                        dim=c_ast.Constant("int", "5"),
                                        dim_quals=[],
                                    ),
                                    dim=None,
                                    dim_quals=["restrict"],
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("zz", [], None, c_ast.IdType(["int"])),
                ),
            ),
            id="unnamed function parameter 2",
        ),
        pytest.param(
            "int zz(int [const restrict volatile 10][5]);",
            DeclWithDefaults(
                name="zz",
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Typename(
                                name=None,
                                quals=[],
                                align=None,
                                type=c_ast.ArrayDecl(
                                    type=c_ast.ArrayDecl(
                                        type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["int"])),
                                        dim=c_ast.Constant("int", "5"),
                                        dim_quals=[],
                                    ),
                                    dim=c_ast.Constant("int", "10"),
                                    dim_quals=["const", "restrict", "volatile"],
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("zz", [], None, c_ast.IdType(["int"])),
                ),
            ),
            id="unnamed function parameter 3",
        ),
    ],
)
def test_func_decls_with_array_dim_qualifiers(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    decl = tree.ext[0]
    assert decl == expected


@pytest.mark.parametrize(
    ("test_input", "index", "expected_quals", "expected_storage"),
    [
        ("extern int p;", 0, [], ["extern"]),
        ("_Thread_local int p;", 0, [], ["_Thread_local"]),
        ("const long p = 6;", 0, ["const"], []),
        ("_Atomic int p;", 0, ["_Atomic"], []),
        ("_Atomic restrict int* p;", 0, ["_Atomic", "restrict"], []),
        ("static const int p, q, r;", 0, ["const"], ["static"]),
        ("static const int p, q, r;", 1, ["const"], ["static"]),
        ("static const int p, q, r;", 2, ["const"], ["static"]),
        ("static char * const p;", 0, [], ["static"]),
    ],
)
def test_qualifiers_storage_specifiers_1(
    test_input: str,
    index: int,
    expected_quals: list[str],
    expected_storage: list[str],
):
    tree = parse(test_input).ext[index]

    assert isinstance(tree, c_ast.Decl)
    assert tree.quals == expected_quals
    assert tree.storage == expected_storage


def test_qualifiers_storage_specifiers_2():
    test_input = "static char * const p;"
    tree = parse(test_input)
    pdecl = tree.ext[0].type
    assert isinstance(pdecl, c_ast.PtrDecl)
    assert pdecl.quals == ["const"]


@pytest.mark.parametrize(
    ("test_input", "index", "expected"),
    [
        (
            "_Atomic(int) ai;",
            0,
            DeclWithDefaults(
                name="ai",
                quals=["_Atomic"],
                type=c_ast.TypeDecl("ai", quals=["_Atomic"], align=None, type=c_ast.IdType(["int"])),
            ),
        ),
        (
            "_Atomic(int*) ai;",
            0,
            DeclWithDefaults(
                name="ai",
                type=c_ast.PtrDecl(quals=["_Atomic"], type=c_ast.TypeDecl("ai", [], None, c_ast.IdType(["int"]))),
            ),
        ),
        (
            "_Atomic(_Atomic(int)*) aai;",
            0,
            DeclWithDefaults(
                name="aai",
                quals=["_Atomic"],
                type=c_ast.PtrDecl(
                    quals=["_Atomic"],
                    type=c_ast.TypeDecl("aai", quals=["_Atomic"], align=None, type=c_ast.IdType(["int"])),
                ),
            ),
        ),
        pytest.param(
            "_Atomic(int) foo, bar;",
            slice(0, 2),
            [
                DeclWithDefaults(
                    name="foo",
                    quals=["_Atomic"],
                    type=c_ast.TypeDecl("foo", quals=["_Atomic"], align=None, type=c_ast.IdType(["int"])),
                ),
                DeclWithDefaults(
                    name="bar",
                    quals=["_Atomic"],
                    type=c_ast.TypeDecl("foo", quals=["_Atomic"], align=None, type=c_ast.IdType(["int"])),
                ),
            ],
            id="multiple declarations",
        ),
        pytest.param(
            "typedef _Atomic(int) atomic_int;",
            0,
            c_ast.Typedef(
                name="atomic_int",
                quals=["_Atomic"],
                storage=["typedef"],
                type=c_ast.TypeDecl("atomic_int", quals=["_Atomic"], align=None, type=c_ast.IdType(["int"])),
            ),
            id="typedefs with _Atomic specifiers 1",
        ),
        pytest.param(
            "typedef _Atomic(_Atomic(_Atomic(int (*)(void)) *) *) t;",
            0,
            c_ast.Typedef(
                name="t",
                quals=[],
                storage=["typedef"],
                type=c_ast.PtrDecl(
                    quals=["_Atomic"],
                    type=c_ast.PtrDecl(
                        quals=["_Atomic"],
                        type=c_ast.PtrDecl(
                            quals=["_Atomic"],
                            type=c_ast.FuncDecl(
                                args=c_ast.ParamList(
                                    [
                                        c_ast.Typename(
                                            name=None,
                                            quals=[],
                                            align=None,
                                            type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["void"])),
                                        )
                                    ]
                                ),
                                type=c_ast.TypeDecl("t", [], None, type=c_ast.IdType(["int"])),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ],
)
def test_atomic_specifier(test_input: str, index: Union[int, slice], expected: Union[c_ast.AST, list[c_ast.AST]]):
    tree = parse(test_input)
    decl = tree.ext[index]
    assert c_ast.compare_asts(decl, expected)


def test_sizeof() -> None:
    test_input = """\
void foo()
{
    int a = sizeof k;
    int b = sizeof(int);
    int c = sizeof(int**);;

    char* p = "just to make sure this parses w/o error...";
    int d = sizeof(int());
}
"""

    block_item_init_cases = (
        (0, c_ast.UnaryOp(op="sizeof", expr=c_ast.Id(name="k"))),
        (
            1,
            c_ast.UnaryOp(
                op="sizeof",
                expr=c_ast.Typename(
                    name=None, quals=[], align=None, type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["int"]))
                ),
            ),
        ),
        (
            2,
            c_ast.UnaryOp(
                op="sizeof",
                expr=c_ast.Typename(
                    name=None,
                    quals=[],
                    align=None,
                    type=c_ast.PtrDecl(
                        quals=[],
                        type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["int"]))),
                    ),
                ),
            ),
        ),
    )

    tree = parse(test_input)
    compound = tree.ext[0].body
    assert isinstance(compound, c_ast.Compound)

    for index, expected in block_item_init_cases:
        found_init = compound.block_items[index].init
        assert isinstance(found_init, c_ast.UnaryOp)
        assert found_init == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int a = _Alignof(int);",
            DeclWithDefaults(
                name="a",
                type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["int"])),
                init=c_ast.UnaryOp(
                    op="_Alignof",
                    expr=c_ast.Typename(
                        name=None,
                        quals=[],
                        align=None,
                        type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["int"])),
                    ),
                ),
            ),
        ),
        (
            "_Alignas(_Alignof(int)) char a;",
            DeclWithDefaults(
                name="a",
                align=[
                    c_ast.Alignas(
                        alignment=c_ast.UnaryOp(
                            op="_Alignof",
                            expr=c_ast.Typename(
                                name=None,
                                quals=[],
                                align=None,
                                type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["int"])),
                            ),
                        )
                    )
                ],
                type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["char"])),
            ),
        ),
        (
            "_Alignas(4) char a;",
            DeclWithDefaults(
                name="a",
                align=[c_ast.Alignas(alignment=c_ast.Constant("int", "4"))],
                type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["char"])),
            ),
        ),
        (
            "_Alignas(int) char a;",
            DeclWithDefaults(
                name="a",
                align=[
                    c_ast.Alignas(
                        alignment=c_ast.Typename(
                            name=None,
                            quals=[],
                            align=None,
                            type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["int"])),
                        )
                    )
                ],
                type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["char"])),
            ),
        ),
    ],
)
def test_alignof(test_input: str, expected: c_ast.AST) -> None:
    tree = parse(test_input)
    decl = tree.ext[0]
    assert decl == expected


def test_offsetof():
    test_input = """\
void foo() {
    int a = offsetof(struct S, p);
    a.b = offsetof(struct sockaddr, sp) + strlen(bar);
    int a = offsetof(struct S, p.q.r);
    int a = offsetof(struct S, p[5].q[4][5]);
}
"""

    expected_list = [
        DeclWithDefaults(
            name="a",
            type=c_ast.TypeDecl("a", [], None, c_ast.IdType(names=["int"])),
            init=c_ast.FuncCall(
                name=c_ast.Id("offsetof"),
                args=c_ast.ExprList(
                    [
                        c_ast.Typename(
                            name=None,
                            quals=[],
                            align=None,
                            type=c_ast.TypeDecl(None, [], None, c_ast.Struct(name="S", decls=None)),
                        ),
                        c_ast.Id("p"),
                    ]
                ),
            ),
        ),
        c_ast.Assignment(
            op="=",
            left=c_ast.StructRef(name=c_ast.Id("a"), type=".", field=c_ast.Id("b")),
            right=c_ast.BinaryOp(
                op="+",
                left=c_ast.FuncCall(
                    name=c_ast.Id("offsetof"),
                    args=c_ast.ExprList(
                        [
                            c_ast.Typename(
                                name=None,
                                quals=[],
                                align=None,
                                type=c_ast.TypeDecl(None, [], None, c_ast.Struct(name="sockaddr", decls=None)),
                            ),
                            c_ast.Id("sp"),
                        ]
                    ),
                ),
                right=c_ast.FuncCall(
                    name=c_ast.Id("strlen"),
                    args=c_ast.ExprList([c_ast.Id("bar")]),
                ),
            ),
        ),
        DeclWithDefaults(
            name="a",
            type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["int"])),
            init=c_ast.FuncCall(
                name=c_ast.Id("offsetof"),
                args=c_ast.ExprList(
                    [
                        c_ast.Typename(
                            name=None,
                            quals=[],
                            align=None,
                            type=c_ast.TypeDecl(None, [], None, c_ast.Struct(name="S", decls=None)),
                        ),
                        c_ast.StructRef(
                            name=c_ast.StructRef(name=c_ast.Id("p"), type=".", field=c_ast.Id("q")),
                            type=".",
                            field=c_ast.Id("r"),
                        ),
                    ]
                ),
            ),
        ),
        DeclWithDefaults(
            name="a",
            type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["int"])),
            init=c_ast.FuncCall(
                name=c_ast.Id("offsetof"),
                args=c_ast.ExprList(
                    [
                        c_ast.Typename(
                            name=None,
                            quals=[],
                            align=None,
                            type=c_ast.TypeDecl(None, [], None, type=c_ast.Struct(name="S", decls=None)),
                        ),
                        c_ast.ArrayRef(
                            name=c_ast.ArrayRef(
                                name=c_ast.StructRef(
                                    name=c_ast.ArrayRef(name=c_ast.Id("p"), subscript=c_ast.Constant("int", "5")),
                                    type=".",
                                    field=c_ast.Id("q"),
                                ),
                                subscript=c_ast.Constant("int", "4"),
                            ),
                            subscript=c_ast.Constant("int", "5"),
                        ),
                    ]
                ),
            ),
        ),
    ]

    tree = parse(test_input)
    assert c_ast.compare_asts(tree.ext[0].body.block_items, expected_list)


def test_compound_statement() -> None:
    test_input = """\
void foo() {
}
"""

    tree = parse(test_input)

    compound = tree.ext[0].body
    assert isinstance(compound, c_ast.Compound)
    assert compound.coord == Coord("", 2, 0)


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        pytest.param(
            0,
            c_ast.CompoundLiteral(
                type=c_ast.Typename(
                    name=None,
                    quals=[],
                    align=None,
                    type=c_ast.TypeDecl(None, [], None, c_ast.IdType(names=["long", "long"])),
                ),
                init=c_ast.InitList([c_ast.Id("k")]),
            ),
            id="C99 compound literal feature 1",
        ),
        pytest.param(
            1,
            c_ast.CompoundLiteral(
                type=c_ast.Typename(
                    name=None,
                    quals=[],
                    align=None,
                    type=c_ast.TypeDecl(None, [], None, c_ast.Struct(name="jk", decls=None)),
                ),
                init=c_ast.InitList(
                    [
                        c_ast.NamedInitializer(
                            name=[c_ast.Id("a")],
                            expr=c_ast.InitList([c_ast.Constant("int", "1"), c_ast.Constant(type="int", value="2")]),
                        ),
                        c_ast.NamedInitializer(name=[c_ast.Id("b"), c_ast.Constant("int", "0")], expr=c_ast.Id("t")),
                    ]
                ),
            ),
            id="C99 compound literal feature 2",
        ),
    ],
)
def test_compound_literals(index: int, expected: c_ast.CompoundLiteral) -> None:
    test_input = r"""
void foo() {
    p = (long long){k};
    tc = (struct jk){.a = {1, 2}, .b[0] = t};
}"""

    tree = parse(test_input)

    compound = tree.ext[0].body.block_items[index].right
    assert isinstance(compound, c_ast.CompoundLiteral)
    assert compound == expected


def test_parenthesized_compounds() -> None:
    test_input = r"""
    void foo() {
        int a;
        ({});
        ({ 1; });
        ({ 1; 2; });
        int b = ({ 1; });
        int c, d = ({ int x = 1; x + 2; });
        a = ({ int x = 1; 2 * x; });
    }"""

    expected = [
        DeclWithDefaults(name="a", type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["int"]))),
        c_ast.Compound(block_items=None),
        c_ast.Compound([c_ast.Constant("int", value="1")]),
        c_ast.Compound([c_ast.Constant("int", value="1"), c_ast.Constant("int", "2")]),
        DeclWithDefaults(
            name="b",
            type=c_ast.TypeDecl("b", [], None, c_ast.IdType(names=["int"])),
            init=c_ast.Compound([c_ast.Constant("int", "1")]),
        ),
        DeclWithDefaults(name="c", type=c_ast.TypeDecl("c", [], None, c_ast.IdType(["int"]))),
        DeclWithDefaults(
            name="d",
            type=c_ast.TypeDecl("d", [], None, c_ast.IdType(names=["int"])),
            init=c_ast.Compound(
                [
                    DeclWithDefaults(
                        name="x",
                        type=c_ast.TypeDecl("x", [], None, c_ast.IdType(["int"])),
                        init=c_ast.Constant("int", "1"),
                    ),
                    c_ast.BinaryOp(op="+", left=c_ast.Id("x"), right=c_ast.Constant("int", "2")),
                ]
            ),
        ),
        c_ast.Assignment(
            op="=",
            left=c_ast.Id("a"),
            right=c_ast.Compound(
                [
                    DeclWithDefaults(
                        name="x",
                        type=c_ast.TypeDecl("x", [], None, c_ast.IdType(["int"])),
                        init=c_ast.Constant("int", "1"),
                    ),
                    c_ast.BinaryOp(op="*", left=c_ast.Constant("int", "2"), right=c_ast.Id("x")),
                ]
            ),
        ),
    ]

    tree = parse(test_input)
    block_items = tree.ext[0].body.block_items
    assert c_ast.compare_asts(block_items, expected)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("enum mycolor op;", c_ast.Enum(name="mycolor", values=None)),
        (
            "enum mysize {large=20, small, medium} shoes;",
            c_ast.Enum(
                name="mysize",
                values=c_ast.EnumeratorList(
                    [
                        c_ast.Enumerator(name="large", value=c_ast.Constant("int", "20")),
                        c_ast.Enumerator(name="small", value=None),
                        c_ast.Enumerator(name="medium", value=None),
                    ]
                ),
            ),
        ),
        pytest.param(
            "enum\n{\n    red,\n    blue,\n    green,\n} color;",
            c_ast.Enum(
                name=None,
                values=c_ast.EnumeratorList(
                    [
                        c_ast.Enumerator(name="red", value=None),
                        c_ast.Enumerator(name="blue", value=None),
                        c_ast.Enumerator(name="green", value=None),
                    ]
                ),
            ),
            id="enum with trailing comma (C99 feature)",
        ),
    ],
)
def test_enums(test_input: str, expected: c_ast.AST) -> None:
    tree = parse(test_input)
    enum_type = tree.ext[0].type.type  # type: ignore
    assert isinstance(enum_type, c_ast.Enum)
    assert enum_type == expected


@pytest.mark.parametrize(
    ("test_input", "index", "expected"),
    [
        pytest.param(
            "typedef void* node;\nnode k;",
            slice(2),
            [
                c_ast.Typedef(
                    name="node",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.PtrDecl(
                        quals=[],
                        type=c_ast.TypeDecl("node", [], None, c_ast.IdType(["void"])),
                    ),
                ),
                DeclWithDefaults(name="k", type=c_ast.TypeDecl("k", [], None, c_ast.IdType(["node"]))),
            ],
            id="with typedef",
        ),
        (
            "typedef int T;\ntypedef T *pT;\n\npT aa, bb;",
            3,
            DeclWithDefaults(name="bb", type=c_ast.TypeDecl("bb", [], None, c_ast.IdType(["pT"]))),
        ),
        (
            "typedef char* __builtin_va_list;\ntypedef __builtin_va_list __gnuc_va_list;",
            1,
            c_ast.Typedef(
                name="__gnuc_va_list",
                quals=[],
                storage=["typedef"],
                type=c_ast.TypeDecl("__gnuc_va_list", [], None, c_ast.IdType(["__builtin_va_list"])),
            ),
        ),
        (
            "typedef struct tagHash Hash;",
            0,
            c_ast.Typedef(
                name="Hash",
                quals=[],
                storage=["typedef"],
                type=c_ast.TypeDecl("Hash", [], None, c_ast.Struct(name="tagHash", decls=None)),
            ),
        ),
        (
            "typedef int (* const * const T)(void);",
            0,
            c_ast.Typedef(
                name="T",
                quals=[],
                storage=["typedef"],
                type=c_ast.PtrDecl(
                    quals=["const"],
                    type=c_ast.PtrDecl(
                        quals=["const"],
                        type=c_ast.FuncDecl(
                            args=c_ast.ParamList(
                                [
                                    c_ast.Typename(
                                        name=None,
                                        quals=[],
                                        align=None,
                                        type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["void"])),
                                    )
                                ]
                            ),
                            type=c_ast.TypeDecl("T", [], None, c_ast.IdType(["int"])),
                        ),
                    ),
                ),
            ),
        ),
    ],
)
def test_typedef(test_input: str, index: Union[int, slice], expected: Union[c_ast.AST, list[c_ast.AST]]) -> None:
    tree = parse(test_input)
    assert c_ast.compare_asts(tree.ext[index], expected)


@pytest.mark.parametrize(
    "test_input",
    [pytest.param("node k;", id="without typedef")],
)
def test_typedef_error(test_input: str):
    with pytest.raises(CParseError):
        parse(test_input)


@pytest.mark.parametrize(
    ("test_input", "index", "expected"),
    [
        (
            "struct {\n    int id;\n    char* name;\n} joe;",
            0,
            DeclWithDefaults(
                name="joe",
                type=c_ast.TypeDecl(
                    "joe",
                    [],
                    None,
                    c_ast.Struct(
                        name=None,
                        decls=[
                            DeclWithDefaults(
                                name="id",
                                type=c_ast.TypeDecl("id", [], None, c_ast.IdType(names=["int"])),
                            ),
                            DeclWithDefaults(
                                name="name",
                                type=c_ast.PtrDecl(
                                    quals=[],
                                    type=c_ast.TypeDecl("name", [], None, c_ast.IdType(names=["char"])),
                                ),
                            ),
                        ],
                    ),
                ),
            ),
        ),
        (
            "struct node p;",
            0,
            DeclWithDefaults(name="p", type=c_ast.TypeDecl("p", [], None, c_ast.Struct("node", None))),
        ),
        (
            "union pri ra;",
            0,
            DeclWithDefaults(name="ra", type=c_ast.TypeDecl("ra", [], None, c_ast.Union(name="pri", decls=None))),
        ),
        (
            "struct node* p;",
            0,
            DeclWithDefaults(
                name="p",
                type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("p", [], None, c_ast.Struct(name="node", decls=None))),
            ),
        ),
        (
            "struct node;",
            0,
            DeclWithDefaults(name=None, type=c_ast.Struct(name="node", decls=None)),
        ),
        (
            "union\n"
            "{\n"
            "    struct\n"
            "    {\n"
            "        int type;\n"
            "    } n;\n"
            "\n"
            "    struct\n"
            "    {\n"
            "        int type;\n"
            "        int intnode;\n"
            "    } ni;\n"
            "} u;",
            0,
            DeclWithDefaults(
                name="u",
                type=c_ast.TypeDecl(
                    declname="u",
                    quals=[],
                    align=None,
                    type=c_ast.Union(
                        name=None,
                        decls=[
                            DeclWithDefaults(
                                name="n",
                                type=c_ast.TypeDecl(
                                    "n",
                                    [],
                                    None,
                                    c_ast.Struct(
                                        name=None,
                                        decls=[
                                            DeclWithDefaults(
                                                name="type",
                                                type=c_ast.TypeDecl("type", [], None, c_ast.IdType(names=["int"])),
                                            )
                                        ],
                                    ),
                                ),
                            ),
                            DeclWithDefaults(
                                name="ni",
                                type=c_ast.TypeDecl(
                                    "ni",
                                    [],
                                    None,
                                    c_ast.Struct(
                                        name=None,
                                        decls=[
                                            DeclWithDefaults(
                                                name="type",
                                                type=c_ast.TypeDecl("type", [], None, c_ast.IdType(["int"])),
                                            ),
                                            DeclWithDefaults(
                                                name="intnode",
                                                type=c_ast.TypeDecl("intnode", [], None, c_ast.IdType(["int"])),
                                            ),
                                        ],
                                    ),
                                ),
                            ),
                        ],
                    ),
                ),
            ),
        ),
        (
            "typedef struct foo_tag\n" "{\n" "    void* data;\n" "} foo, *pfoo;",
            slice(2),
            [
                c_ast.Typedef(
                    name="foo",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.TypeDecl(
                        declname="foo",
                        quals=[],
                        align=None,
                        type=c_ast.Struct(
                            name="foo_tag",
                            decls=[
                                DeclWithDefaults(
                                    name="data",
                                    type=c_ast.PtrDecl(
                                        quals=[],
                                        type=c_ast.TypeDecl("data", [], None, c_ast.IdType(["void"])),
                                    ),
                                )
                            ],
                        ),
                    ),
                ),
                c_ast.Typedef(
                    name="pfoo",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.PtrDecl(
                        quals=[],
                        type=c_ast.TypeDecl(
                            "pfoo",
                            [],
                            None,
                            type=c_ast.Struct(
                                name="foo_tag",
                                decls=[
                                    DeclWithDefaults(
                                        name="data",
                                        type=c_ast.PtrDecl(
                                            quals=[],
                                            type=c_ast.TypeDecl(
                                                "data",
                                                [],
                                                None,
                                                c_ast.IdType(["void"]),
                                            ),
                                        ),
                                    )
                                ],
                            ),
                        ),
                    ),
                ),
            ],
        ),
        (
            "typedef enum tagReturnCode {SUCCESS, FAIL} ReturnCode;\n"
            "\n"
            "typedef struct tagEntry\n"
            "{\n"
            "    char* key;\n"
            "    char* value;\n"
            "} Entry;\n"
            "\n"
            "\n"
            "typedef struct tagNode\n"
            "{\n"
            "    Entry* entry;\n"
            "\n"
            "    struct tagNode* next;\n"
            "} Node;\n"
            "\n"
            "typedef struct tagHash\n"
            "{\n"
            "    unsigned int table_size;\n"
            "\n"
            "    Node** heads;\n"
            "\n"
            "} Hash;\n",
            3,
            c_ast.Typedef(
                name="Hash",
                quals=[],
                storage=["typedef"],
                type=c_ast.TypeDecl(
                    declname="Hash",
                    quals=[],
                    align=None,
                    type=c_ast.Struct(
                        name="tagHash",
                        decls=[
                            DeclWithDefaults(
                                name="table_size",
                                type=c_ast.TypeDecl("table_size", [], None, c_ast.IdType(["unsigned", "int"])),
                            ),
                            DeclWithDefaults(
                                name="heads",
                                type=c_ast.PtrDecl(
                                    quals=[],
                                    type=c_ast.PtrDecl(
                                        quals=[],
                                        type=c_ast.TypeDecl("heads", [], None, c_ast.IdType(["Node"])),
                                    ),
                                ),
                            ),
                        ],
                    ),
                ),
            ),
        ),
    ],
)
def test_struct_union(test_input: str, index: Union[int, slice], expected: Union[c_ast.AST, list[c_ast.AST]]) -> None:
    tree = parse(test_input)
    type_ = tree.ext[index]
    assert c_ast.compare_asts(type_, expected)


@pytest.mark.xfail()
def test_struct_with_line_pp():
    test_input = r"""
struct _on_exit_args {
    void *  _fnargs[32];
    void *  _dso_handle[32];

    long _fntypes;
    #line 77 "D:\eli\cpp_stuff\libc_include/sys/reent.h"

    long _is_cxa;
};
"""

    s7_ast = parse(test_input, filename="test.c")
    assert s7_ast.ext[0].type.decls[2].coord == Coord("test.c", 6, 22)
    assert s7_ast.ext[0].type.decls[3].coord == Coord(r"D:\eli\cpp_stuff\libc_include/sys/reent.h", 78, 22)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "struct Foo {\n   enum Bar { A = 1 };\n};",
            DeclWithDefaults(
                name=None,
                type=c_ast.Struct(
                    name="Foo",
                    decls=[  # type: ignore
                        DeclWithDefaults(
                            name=None,
                            type=c_ast.Enum(
                                name="Bar",
                                values=c_ast.EnumeratorList([c_ast.Enumerator("A", c_ast.Constant("int", "1"))]),
                            ),
                        )
                    ],
                ),
            ),
        ),
        (
            "struct Foo {\n    enum Bar { A = 1, B, C } bar;\n    enum Baz { D = A } baz;\n} foo;",
            DeclWithDefaults(
                name="foo",
                type=c_ast.TypeDecl(
                    declname="foo",
                    quals=[],
                    align=None,
                    type=c_ast.Struct(
                        name="Foo",
                        decls=[
                            DeclWithDefaults(
                                name="bar",
                                type=c_ast.TypeDecl(
                                    declname="bar",
                                    quals=[],
                                    align=None,
                                    type=c_ast.Enum(
                                        name="Bar",
                                        values=c_ast.EnumeratorList(
                                            [
                                                c_ast.Enumerator("A", c_ast.Constant("int", "1")),
                                                c_ast.Enumerator("B", None),
                                                c_ast.Enumerator("C", None),
                                            ]
                                        ),
                                    ),
                                ),
                            ),
                            DeclWithDefaults(
                                name="baz",
                                type=c_ast.TypeDecl(
                                    declname="baz",
                                    quals=[],
                                    align=None,
                                    type=c_ast.Enum(
                                        name="Baz", values=c_ast.EnumeratorList([c_ast.Enumerator("D", c_ast.Id("A"))])
                                    ),
                                ),
                            ),
                        ],
                    ),
                ),
            ),
        ),
    ],
)
def test_struct_enum(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    type_ = tree.ext[0]
    assert type_ == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "struct {\n    int a;;\n} foo;",
            DeclWithDefaults(
                name="foo",
                type=c_ast.TypeDecl(
                    declname="foo",
                    quals=[],
                    align=None,
                    type=c_ast.Struct(
                        name=None,
                        decls=[
                            DeclWithDefaults(
                                name="a",
                                type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["int"])),
                            )
                        ],
                    ),
                ),
            ),
        ),
        (
            "struct {\n    int a;;;;\n    float b, c;\n    ;;\n    char d;\n} foo;",
            DeclWithDefaults(
                name="foo",
                type=c_ast.TypeDecl(
                    declname="foo",
                    quals=[],
                    align=None,
                    type=c_ast.Struct(
                        name=None,
                        decls=[
                            DeclWithDefaults(
                                name="a",
                                type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["int"])),
                            ),
                            DeclWithDefaults(
                                name="b",
                                type=c_ast.TypeDecl("b", [], None, c_ast.IdType(["float"])),
                            ),
                            DeclWithDefaults(
                                name="c",
                                type=c_ast.TypeDecl("c", [], None, c_ast.IdType(["float"])),
                            ),
                            DeclWithDefaults(
                                name="d",
                                type=c_ast.TypeDecl("d", [], None, c_ast.IdType(["char"])),
                            ),
                        ],
                    ),
                ),
            ),
        ),
    ],
)
def test_struct_with_extra_semis_inside(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    type_ = tree.ext[0]
    assert type_ == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "struct {\n    ;int a;\n} foo;",
            DeclWithDefaults(
                name="foo",
                type=c_ast.TypeDecl(
                    declname="foo",
                    quals=[],
                    align=None,
                    type=c_ast.Struct(
                        name=None,
                        decls=[DeclWithDefaults(name="a", type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["int"])))],
                    ),
                ),
            ),
        )
    ],
)
def test_struct_with_initial_semi(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    type_ = tree.ext[0]
    assert type_ == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "union\n"
            "{\n"
            "    union\n"
            "    {\n"
            "        int i;\n"
            "        long l;\n"
            "    };\n"
            "\n"
            "    struct\n"
            "    {\n"
            "        int type;\n"
            "        int intnode;\n"
            "    };\n"
            "} u;",
            c_ast.FuncDef(
                decl=DeclWithDefaults(
                    name="foo",
                    type=c_ast.FuncDecl(args=None, type=c_ast.TypeDecl("foo", [], None, c_ast.IdType(["void"]))),
                ),
                param_decls=None,
                body=c_ast.Compound(
                    [  # type: ignore
                        DeclWithDefaults(name="a", type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["int"]))),
                        c_ast.Compound(block_items=None),
                        c_ast.Compound(block_items=[c_ast.Constant("int", "1")]),
                        c_ast.Compound(block_items=[c_ast.Constant("int", "1"), c_ast.Constant("int", "2")]),
                        DeclWithDefaults(
                            name="b",
                            type=c_ast.TypeDecl("b", [], None, c_ast.IdType(["int"])),
                            init=c_ast.Compound([c_ast.Constant("int", "1")]),
                        ),
                        DeclWithDefaults(name="c", type=c_ast.TypeDecl("c", [], None, c_ast.IdType(["int"]))),
                        DeclWithDefaults(
                            name="d",
                            type=c_ast.TypeDecl("d", [], None, c_ast.IdType(["int"])),
                            init=c_ast.Compound(
                                [
                                    DeclWithDefaults(
                                        name="x",
                                        type=c_ast.TypeDecl("x", [], None, c_ast.IdType(["int"])),
                                        init=c_ast.Constant("int", "1"),
                                    ),
                                    c_ast.BinaryOp(op="+", left=c_ast.Id("x"), right=c_ast.Constant("int", "2")),
                                ]
                            ),
                        ),
                        c_ast.Assignment(
                            op="=",
                            left=c_ast.Id("a"),
                            right=c_ast.Compound(
                                [
                                    DeclWithDefaults(
                                        name="x",
                                        type=c_ast.TypeDecl("x", [], None, c_ast.IdType(["int"])),
                                        init=c_ast.Constant("int", "1"),
                                    ),
                                    c_ast.BinaryOp(op="*", left=c_ast.Constant("int", "2"), right=c_ast.Id("x")),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
        ),
        pytest.param(
            "struct v {\n"
            "    union {\n"
            "        struct { int i, j; };\n"
            "        struct { long k, l; } w;\n"
            "    };\n"
            "    int m;\n"
            "} v1;\n",
            DeclWithDefaults(
                name="v1",
                type=c_ast.TypeDecl(
                    declname="v1",
                    quals=[],
                    align=None,
                    type=c_ast.Struct(
                        name="v",
                        decls=[  # type: ignore
                            DeclWithDefaults(
                                name=None,
                                type=c_ast.Union(
                                    name=None,
                                    decls=[  # type: ignore
                                        DeclWithDefaults(
                                            name=None,
                                            type=c_ast.Struct(
                                                name=None,
                                                decls=[
                                                    DeclWithDefaults(
                                                        name="i",
                                                        type=c_ast.TypeDecl("i", [], None, c_ast.IdType(["int"])),
                                                    ),
                                                    DeclWithDefaults(
                                                        name="j",
                                                        type=c_ast.TypeDecl("j", [], None, c_ast.IdType(["int"])),
                                                    ),
                                                ],
                                            ),
                                        ),
                                        DeclWithDefaults(
                                            name="w",
                                            type=c_ast.TypeDecl(
                                                declname="w",
                                                quals=[],
                                                align=None,
                                                type=c_ast.Struct(
                                                    name=None,
                                                    decls=[
                                                        DeclWithDefaults(
                                                            name="k",
                                                            type=c_ast.TypeDecl("k", [], None, c_ast.IdType(["long"])),
                                                        ),
                                                        DeclWithDefaults(
                                                            name="l",
                                                            type=c_ast.TypeDecl("l", [], None, c_ast.IdType(["long"])),
                                                        ),
                                                    ],
                                                ),
                                            ),
                                        ),
                                    ],
                                ),
                            ),
                            DeclWithDefaults(name="m", type=c_ast.TypeDecl("m", [], None, c_ast.IdType(["int"]))),
                        ],
                    ),
                ),
            ),
            id="ISO/IEC 9899:201x Committee Draft 2010-11-16, N1539, section 6.7.2.1, par. 19, example 1",
        ),
        (
            "struct v {\n    int i;\n    float;\n} v2;",
            DeclWithDefaults(
                name="v2",
                type=c_ast.TypeDecl(
                    declname="v2",
                    quals=[],
                    align=None,
                    type=c_ast.Struct(
                        name="v",
                        decls=[  # type: ignore
                            DeclWithDefaults(name="i", type=c_ast.TypeDecl("i", [], None, c_ast.IdType(["int"]))),
                            DeclWithDefaults(name=None, type=c_ast.IdType(["float"])),
                        ],
                    ),
                ),
            ),
        ),
    ],
)
def test_anonymous_struct_union(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    type_ = tree.ext[0]
    assert type_ == expected


def test_struct_members_namespace():
    """Tests that structure/union member names reside in a separate
    namespace and can be named after existing types.
    """

    test_input = """
typedef int Name;
typedef Name NameArray[10];

struct {
    Name Name;
    Name NameArray[3];
} sye;

void main(void)
{
    sye.Name = 1;
}
        """

    tree = parse(test_input)

    expected2 = DeclWithDefaults(
        name="sye",
        type=c_ast.TypeDecl(
            declname="sye",
            quals=[],
            align=None,
            type=c_ast.Struct(
                name=None,
                decls=[
                    DeclWithDefaults(name="Name", type=c_ast.TypeDecl("Name", [], None, c_ast.IdType(["Name"]))),
                    DeclWithDefaults(
                        name="NameArray",
                        type=c_ast.ArrayDecl(
                            type=c_ast.TypeDecl("NameArray", [], None, c_ast.IdType(["Name"])),
                            dim=c_ast.Constant("int", "3"),
                            dim_quals=[],
                        ),
                    ),
                ],
            ),
        ),
    )

    assert tree.ext[2] == expected2
    assert tree.ext[3].body.block_items[0].left.field.name == "Name"


def test_struct_bitfields():
    # a struct with two bitfields, one unnamed
    s1 = """\
struct {
    int k:6;
    int :2;
} joe;
"""

    tree = parse(s1)
    parsed_struct = tree.ext[0]

    expected = DeclWithDefaults(
        name="joe",
        type=c_ast.TypeDecl(
            declname="joe",
            quals=[],
            align=None,
            type=c_ast.Struct(
                name=None,
                decls=[  # type: ignore
                    DeclWithDefaults(
                        name="k",
                        type=c_ast.TypeDecl("k", [], None, c_ast.IdType(["int"])),
                        bitsize=c_ast.Constant("int", "6"),
                    ),
                    DeclWithDefaults(
                        name=None,
                        type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["int"])),
                        bitsize=c_ast.Constant("int", "2"),
                    ),
                ],
            ),
        ),
    )

    assert parsed_struct == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "struct foo { };",
            DeclWithDefaults(name=None, type=c_ast.Struct(name="foo", decls=[])),
        ),
        (
            "struct { } foo;",
            DeclWithDefaults(name="foo", type=c_ast.TypeDecl("foo", [], None, c_ast.Struct(name=None, decls=[]))),
        ),
        (
            "union { } foo;",
            DeclWithDefaults(name="foo", type=c_ast.TypeDecl("foo", [], None, c_ast.Union(name=None, decls=[]))),
        ),
    ],
)
def test_struct_empty(test_input: str, expected: c_ast.AST):
    """Tests that parsing an empty struct works.

    Empty structs do NOT follow C99 (See 6.2.5-20 of the C99 standard).
    This is nevertheless supported by some compilers (clang, gcc),
    especially when using FORTIFY code.
    Some compilers (visual) will fail to compile with an error.
    """

    # an empty struct. This is NOT C99 compliant
    tree = parse(test_input)
    empty_struct = tree.ext[0]
    assert empty_struct == expected


@pytest.mark.parametrize(
    ("test_input", "index", "expected"),
    [
        (
            "typedef int tagEntry;\n" "\n" "struct tagEntry\n" "{\n" "    char* key;\n" "    char* value;\n" "} Entry;",
            1,
            DeclWithDefaults(
                name="Entry",
                type=c_ast.TypeDecl(
                    declname="Entry",
                    quals=[],
                    align=None,
                    type=c_ast.Struct(
                        name="tagEntry",
                        decls=[
                            DeclWithDefaults(
                                name="key",
                                type=c_ast.PtrDecl(
                                    quals=[], type=c_ast.TypeDecl("key", [], None, c_ast.IdType(["char"]))
                                ),
                            ),
                            DeclWithDefaults(
                                name="value",
                                type=c_ast.PtrDecl(
                                    quals=[], type=c_ast.TypeDecl("value", [], None, c_ast.IdType(["char"]))
                                ),
                            ),
                        ],
                    ),
                ),
            ),
        ),
        (
            "struct tagEntry;\n"
            "\n"
            "typedef struct tagEntry tagEntry;\n"
            "\n"
            "struct tagEntry\n"
            "{\n"
            "    char* key;\n"
            "    char* value;\n"
            "} Entry;",
            2,
            DeclWithDefaults(
                name="Entry",
                type=c_ast.TypeDecl(
                    declname="Entry",
                    quals=[],
                    align=None,
                    type=c_ast.Struct(
                        name="tagEntry",
                        decls=[
                            DeclWithDefaults(
                                name="key",
                                type=c_ast.PtrDecl(
                                    quals=[],
                                    type=c_ast.TypeDecl("key", [], None, c_ast.IdType(["char"])),
                                ),
                            ),
                            DeclWithDefaults(
                                name="value",
                                type=c_ast.PtrDecl(
                                    quals=[],
                                    type=c_ast.TypeDecl("value", [], None, c_ast.IdType(["char"])),
                                ),
                            ),
                        ],
                    ),
                ),
            ),
        ),
        (
            "typedef int mytag;\n\nenum mytag {ABC, CDE};\nenum mytag joe;\n",
            1,
            DeclWithDefaults(
                name=None,
                type=c_ast.Enum(
                    name="mytag",
                    values=c_ast.EnumeratorList([c_ast.Enumerator("ABC", None), c_ast.Enumerator("CDE", None)]),
                ),
            ),
        ),
    ],
)
def test_tags_namespace(test_input: str, index: Union[int, slice], expected: Union[c_ast.AST, list[c_ast.AST]]):
    """Tests that the tags of structs/unions/enums reside in a separate namespace and
    can be named after existing types.
    """

    tree = parse(test_input)
    assert c_ast.compare_asts(tree.ext[index], expected)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int a, b;",
            c_ast.File(
                [
                    DeclWithDefaults(name="a", type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["int"]))),
                    DeclWithDefaults(name="b", type=c_ast.TypeDecl("b", [], None, c_ast.IdType(["int"]))),
                ]
            ),
        ),
        (
            "char* p, notp, ar[4];",
            c_ast.File(
                [
                    DeclWithDefaults(
                        name="p",
                        type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("p", [], None, c_ast.IdType(["char"]))),
                    ),
                    DeclWithDefaults(name="notp", type=c_ast.TypeDecl("notp", [], None, type=c_ast.IdType(["char"]))),
                    DeclWithDefaults(
                        name="ar",
                        type=c_ast.ArrayDecl(
                            type=c_ast.TypeDecl("ar", [], None, c_ast.IdType(["char"])),
                            dim=c_ast.Constant("int", "4"),
                            dim_quals=[],
                        ),
                    ),
                ]
            ),
        ),
    ],
)
def test_multi_decls(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    assert tree == expected


@pytest.mark.parametrize("test_input", ["int enum {ab, cd} fubr;", "enum kid char brbr;"])
def test_invalid_multiple_types_error(test_input: str):
    with pytest.raises(CParseError):
        parse(test_input)


def test_invalid_typedef_storage_qual_error():
    """Tests that using typedef as a storage qualifier is correctly flagged as an error."""

    test_input = "typedef const int foo(int a) { return 0; }"
    with pytest.raises(CParseError):
        parse(test_input)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "typedef int numbertype;\ntypedef int numbertype;",
            [
                c_ast.Typedef(
                    name="numbertype",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.TypeDecl("numbertype", [], None, c_ast.IdType(["int"])),
                ),
                c_ast.Typedef(
                    name="numbertype",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.TypeDecl("numbertype", [], None, c_ast.IdType(["int"])),
                ),
            ],
        ),
        (
            "typedef int (*funcptr)(int x);\ntypedef int (*funcptr)(int x);",
            [
                c_ast.Typedef(
                    name="funcptr",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.PtrDecl(
                        quals=[],
                        type=c_ast.FuncDecl(
                            args=c_ast.ParamList(
                                [DeclWithDefaults(name="x", type=c_ast.TypeDecl("x", [], None, c_ast.IdType(["int"])))]
                            ),
                            type=c_ast.TypeDecl("funcptr", [], None, c_ast.IdType(["int"])),
                        ),
                    ),
                ),
                c_ast.Typedef(
                    name="funcptr",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.PtrDecl(
                        quals=[],
                        type=c_ast.FuncDecl(
                            args=c_ast.ParamList(
                                [DeclWithDefaults(name="x", type=c_ast.TypeDecl("x", [], None, c_ast.IdType(["int"])))]
                            ),
                            type=c_ast.TypeDecl("funcptr", [], None, c_ast.IdType(["int"])),
                        ),
                    ),
                ),
            ],
        ),
        (
            "typedef int numberarray[5];\ntypedef int numberarray[5];",
            [
                c_ast.Typedef(
                    name="numberarray",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.ArrayDecl(
                        type=c_ast.TypeDecl("numberarray", [], None, c_ast.IdType(["int"])),
                        dim=c_ast.Constant("int", "5"),
                        dim_quals=[],
                    ),
                ),
                c_ast.Typedef(
                    name="numberarray",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.ArrayDecl(
                        type=c_ast.TypeDecl("numberarray", [], None, c_ast.IdType(["int"])),
                        dim=c_ast.Constant("int", "5"),
                        dim_quals=[],
                    ),
                ),
            ],
        ),
    ],
)
def test_duplicate_typedef(test_input: str, expected: list[c_ast.AST]):
    """Tests that redeclarations of existing types are parsed correctly. This is non-standard, but allowed by many
    compilers.
    """

    tree = parse(test_input)
    assert c_ast.compare_asts(tree.ext, expected)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int a = 16;",
            DeclWithDefaults(
                name="a", type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["int"])), init=c_ast.Constant("int", "16")
            ),
        ),
        (
            "float f = 0xEF.56p1;",
            DeclWithDefaults(
                name="f",
                type=c_ast.TypeDecl("f", [], None, c_ast.IdType(["float"])),
                init=c_ast.Constant("float", "0xEF.56p1"),
            ),
        ),
        (
            "int bitmask = 0b1001010;",
            DeclWithDefaults(
                name="bitmask",
                type=c_ast.TypeDecl("bitmask", [], None, c_ast.IdType(["int"])),
                init=c_ast.Constant("int", "0b1001010"),
            ),
        ),
        (
            "long ar[] = {7, 8, 9};",
            DeclWithDefaults(
                name="ar",
                type=c_ast.ArrayDecl(
                    type=c_ast.TypeDecl("ar", [], None, c_ast.IdType(["long"])), dim=None, dim_quals=[]
                ),
                init=c_ast.InitList(
                    [c_ast.Constant("int", "7"), c_ast.Constant("int", "8"), c_ast.Constant("int", "9")]
                ),
            ),
        ),
        (
            "long ar[4] = {};",
            DeclWithDefaults(
                name="ar",
                type=c_ast.ArrayDecl(
                    type=c_ast.TypeDecl("ar", [], None, c_ast.IdType(["long"])),
                    dim=c_ast.Constant("int", "4"),
                    dim_quals=[],
                ),
                init=c_ast.InitList([]),
            ),
        ),
        (
            "char p = j;",
            DeclWithDefaults(name="p", type=c_ast.TypeDecl("p", [], None, c_ast.IdType(["char"])), init=c_ast.Id("j")),
        ),
        (
            "char x = 'c', *p = {0, 1, 2, {4, 5}, 6};",
            [
                DeclWithDefaults(
                    name="x",
                    type=c_ast.TypeDecl("x", [], None, c_ast.IdType(["char"])),
                    init=c_ast.Constant("char", "'c'"),
                ),
                DeclWithDefaults(
                    name="p",
                    type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("p", [], None, c_ast.IdType(["char"]))),
                    init=c_ast.InitList(
                        [
                            c_ast.Constant("int", "0"),
                            c_ast.Constant("int", "1"),
                            c_ast.Constant("int", "2"),
                            c_ast.InitList([c_ast.Constant("int", "4"), c_ast.Constant("int", "5")]),
                            c_ast.Constant("int", "6"),
                        ]
                    ),
                ),
            ],
        ),
        (
            "float d = 1.0;",
            DeclWithDefaults(
                name="d",
                type=c_ast.TypeDecl("d", [], None, c_ast.IdType(["float"])),
                init=c_ast.Constant("double", "1.0"),
            ),
        ),
        (
            "float ld = 1.0l;",
            DeclWithDefaults(
                name="ld",
                type=c_ast.TypeDecl("ld", [], None, c_ast.IdType(["float"])),
                init=c_ast.Constant("long double", "1.0l"),
            ),
        ),
        (
            "float ld = 1.0L;",
            DeclWithDefaults(
                name="ld",
                type=c_ast.TypeDecl("ld", [], None, c_ast.IdType(["float"])),
                init=c_ast.Constant("long double", "1.0L"),
            ),
        ),
        (
            "float ld = 1.0f;",
            DeclWithDefaults(
                name="ld",
                type=c_ast.TypeDecl("ld", [], None, c_ast.IdType(["float"])),
                init=c_ast.Constant("float", "1.0f"),
            ),
        ),
        (
            "float ld = 1.0F;",
            DeclWithDefaults(
                name="ld",
                type=c_ast.TypeDecl("ld", [], None, c_ast.IdType(["float"])),
                init=c_ast.Constant("float", "1.0F"),
            ),
        ),
        (
            "float ld = 0xDE.38p0;",
            DeclWithDefaults(
                name="ld",
                type=c_ast.TypeDecl("ld", [], None, c_ast.IdType(["float"])),
                init=c_ast.Constant("float", "0xDE.38p0"),
            ),
        ),
        (
            "int i = 1;",
            DeclWithDefaults(
                name="i",
                type=c_ast.TypeDecl("i", [], None, c_ast.IdType(["int"])),
                init=c_ast.Constant("int", "1"),
            ),
        ),
        (
            "long int li = 1l;",
            DeclWithDefaults(
                name="li",
                type=c_ast.TypeDecl("li", [], None, c_ast.IdType(["long", "int"])),
                init=c_ast.Constant("long int", "1l"),
            ),
        ),
        (
            "unsigned int ui = 1u;",
            DeclWithDefaults(
                name="ui",
                type=c_ast.TypeDecl("ui", [], None, c_ast.IdType(["unsigned", "int"])),
                init=c_ast.Constant("unsigned int", "1u"),
            ),
        ),
        (
            "unsigned long long int ulli = 1LLU;",
            DeclWithDefaults(
                name="ulli",
                type=c_ast.TypeDecl("ulli", [], None, c_ast.IdType(["unsigned", "long", "long", "int"])),
                init=c_ast.Constant("unsigned long long int", "1LLU"),
            ),
        ),
    ],
)
def test_decl_inits(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)

    if isinstance(expected, list):
        assert c_ast.compare_asts(tree.ext, expected)
    else:
        assert tree.ext[0] == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int a = {.k = 16};",
            c_ast.InitList([c_ast.NamedInitializer(name=[c_ast.Id("k")], expr=c_ast.Constant("int", "16"))]),
        ),
        (
            "int a = { [0].a = {1}, [1].a[0] = 2 };",
            c_ast.InitList(
                [
                    c_ast.NamedInitializer(
                        name=[c_ast.Constant("int", "0"), c_ast.Id("a")],
                        expr=c_ast.InitList([c_ast.Constant("int", "1")]),
                    ),
                    c_ast.NamedInitializer(
                        name=[c_ast.Constant("int", "1"), c_ast.Id("a"), c_ast.Constant("int", "0")],
                        expr=c_ast.Constant("int", "2"),
                    ),
                ]
            ),
        ),
        (
            "int a = { .a = 1, .c = 3, 4, .b = 5};",
            c_ast.InitList(
                [
                    c_ast.NamedInitializer(name=[c_ast.Id("a")], expr=c_ast.Constant("int", "1")),
                    c_ast.NamedInitializer(name=[c_ast.Id("c")], expr=c_ast.Constant("int", "3")),
                    c_ast.Constant("int", "4"),
                    c_ast.NamedInitializer(name=[c_ast.Id("b")], expr=c_ast.Constant("int", "5")),
                ]
            ),
        ),
    ],
)
def test_decl_named_inits(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    init = tree.ext[0].init

    assert isinstance(init, c_ast.InitList)
    assert init == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int factorial(int p)\n{\n    return 3;\n}",
            c_ast.FuncDef(
                decl=DeclWithDefaults(
                    name="factorial",
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [DeclWithDefaults(name="p", type=c_ast.TypeDecl("p", [], None, c_ast.IdType(["int"])))]
                        ),
                        type=c_ast.TypeDecl("factorial", [], None, c_ast.IdType(["int"])),
                    ),
                ),
                param_decls=None,
                body=c_ast.Compound([c_ast.Return(expr=c_ast.Constant("int", "3"))]),
            ),
        ),
        (
            "char* zzz(int p, char* c)\n"
            "{\n"
            "    int a;\n"
            "    char b;\n"
            "\n"
            "    a = b + 2;\n"
            "    return 3;\n"
            "}",
            c_ast.FuncDef(
                decl=DeclWithDefaults(
                    name="zzz",
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [
                                DeclWithDefaults(name="p", type=c_ast.TypeDecl("p", [], None, c_ast.IdType(["int"]))),
                                DeclWithDefaults(
                                    name="c",
                                    type=c_ast.PtrDecl(
                                        quals=[], type=c_ast.TypeDecl("c", [], None, c_ast.IdType(["char"]))
                                    ),
                                ),
                            ]
                        ),
                        type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("zzz", [], None, c_ast.IdType(["char"]))),
                    ),
                ),
                param_decls=None,
                body=c_ast.Compound(
                    [
                        DeclWithDefaults(name="a", type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["int"]))),
                        DeclWithDefaults(name="b", type=c_ast.TypeDecl("b", [], None, c_ast.IdType(["char"]))),
                        c_ast.Assignment(
                            op="=",
                            left=c_ast.Id("a"),
                            right=c_ast.BinaryOp(op="+", left=c_ast.Id("b"), right=c_ast.Constant("int", "2")),
                        ),
                        c_ast.Return(expr=c_ast.Constant("int", "3")),
                    ]
                ),
            ),
        ),
        (
            "char* zzz(p, c)\n"
            "long p, *c;\n"
            "{\n"
            "    int a;\n"
            "    char b;\n"
            "\n"
            "    a = b + 2;\n"
            "    return 3;\n"
            "}",
            c_ast.FuncDef(
                decl=DeclWithDefaults(
                    name="zzz",
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList([c_ast.Id("p"), c_ast.Id("c")]),
                        type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("zzz", [], None, c_ast.IdType(["char"]))),
                    ),
                ),
                param_decls=[
                    DeclWithDefaults(name="p", type=c_ast.TypeDecl("p", [], None, c_ast.IdType(["long"]))),
                    DeclWithDefaults(
                        name="c",
                        type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("c", [], None, c_ast.IdType(["long"]))),
                    ),
                ],
                body=c_ast.Compound(
                    [
                        DeclWithDefaults(name="a", type=c_ast.TypeDecl("a", [], None, c_ast.IdType(["int"]))),
                        DeclWithDefaults(name="b", type=c_ast.TypeDecl("b", [], None, c_ast.IdType(["char"]))),
                        c_ast.Assignment(
                            op="=",
                            left=c_ast.Id("a"),
                            right=c_ast.BinaryOp(op="+", left=c_ast.Id("b"), right=c_ast.Constant("int", "2")),
                        ),
                        c_ast.Return(expr=c_ast.Constant("int", "3")),
                    ]
                ),
            ),
        ),
        pytest.param(
            "que(p)\n{\n    return 3;\n}",
            c_ast.FuncDef(
                decl=DeclWithDefaults(
                    name="que",
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList([c_ast.Id("p")]),
                        type=c_ast.TypeDecl("que", [], None, c_ast.IdType(["int"])),
                    ),
                ),
                param_decls=None,
                body=c_ast.Compound([c_ast.Return(expr=c_ast.Constant("int", "3"))]),
            ),
            id="function return values and parameters may not have type information",
        ),
    ],
)
def test_function_definitions(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    assert tree.ext[0] == expected


def test_static_assert():
    test_input = """\
_Static_assert(1, "123");
int factorial(int p)
{
    _Static_assert(2, "456");
    _Static_assert(3);
}
"""

    tree = parse(test_input)

    expected_assert_1 = c_ast.StaticAssert(cond=c_ast.Constant("int", "1"), message=c_ast.Constant("string", '"123"'))
    assert tree.ext[0] == expected_assert_1

    expected_assert_2 = c_ast.StaticAssert(cond=c_ast.Constant("int", "2"), message=c_ast.Constant("string", '"456"'))
    assert tree.ext[1].body.block_items[0] == expected_assert_2

    expected_assert_3 = c_ast.StaticAssert(cond=c_ast.Constant("int", "3"), message=None)
    assert tree.ext[1].body.block_items[2] == expected_assert_3


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        pytest.param(
            'char* s = "hello";',
            c_ast.Constant("string", '"hello"'),
            id="simple string, for reference",
        ),
        (
            'char* s = "hello" " world";',
            c_ast.Constant("string", '"hello world"'),
        ),
        (
            'char* s = "" "foobar";',
            c_ast.Constant("string", '"foobar"'),
        ),
        (
            r'char* s = "foo\"" "bar";',
            c_ast.Constant("string", r'"foo\"bar"'),
        ),
    ],
)
def test_unified_string_literals(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)

    assert tree.ext[0].init == expected


@pytest.mark.xfail(reason="Not unsupported yet. See issue 392.")
def test_escapes_in_unified_string_literals():
    # This is not correct based on the the C spec, but testing it here to
    # see the behavior in action. Will have to fix this
    # for https://github.com/eliben/pycparser/issues/392
    #
    # The spec says in section 6.4.5 that "escape sequences are converted
    # into single members of the execution character set just prior to
    # adjacent string literal concatenation".

    test_input = r'char* s = "\1" "23";'

    with pytest.raises(CParseError):
        tree = parse(test_input)

        expected = c_ast.Constant("string", r'"\123"')
        assert tree.ext[0].init == expected


def test_unified_string_literals_issue_6():
    test_input = r"""
int main() {
    fprintf(stderr,
    "Wrong Params?\n"
    "Usage:\n"
    "%s <binary_file_path>\n",
    argv[0]
    );
}
    """

    tree = parse(test_input)

    assert tree.ext[0].body.block_items[0].args.exprs[1].value == r'"Wrong Params?\nUsage:\n%s <binary_file_path>\n"'


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            'char* s = L"hello" L"world";',
            c_ast.Constant("string", 'L"helloworld"'),
        ),
        (
            'char* s = L"hello " L"world" L" and I";',
            c_ast.Constant("string", 'L"hello world and I"'),
        ),
    ],
)
def test_unified_wstring_literals(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)

    assert tree.ext[0].init == expected


def test_inline_specifier():
    test_input = "static inline void inlinefoo(void);"
    tree = parse(test_input)

    assert tree.ext[0].funcspec == ["inline"]


def test_noreturn_specifier():
    test_input = "static _Noreturn void noreturnfoo(void);"
    tree = parse(test_input)

    assert tree.ext[0].funcspec == ["_Noreturn"]


def test_variable_length_array():
    test_input = r"""
int main() {
    int size;
    int var[size = 5];

    int var2[*];
}
"""

    tree = parse(test_input)

    expected_dim_1 = c_ast.Assignment(op="=", left=c_ast.Id("size"), right=c_ast.Constant("int", "5"))
    assert tree.ext[0].body.block_items[1].type.dim == expected_dim_1

    expected_dim_2 = c_ast.Id("*")
    assert tree.ext[0].body.block_items[2].type.dim == expected_dim_2


def test_pragma():
    test_input = r"""
#pragma bar
void main() {
    #pragma foo
    for(;;) {}
    #pragma baz
    {
        int i = 0;
    }
    #pragma
}
struct s {
#pragma baz
} s;
_Pragma("other \"string\"")
"""

    tree = parse(test_input)

    pragma1 = tree.ext[0]
    assert pragma1 == c_ast.Pragma("bar")
    assert pragma1.coord.line_start == 2

    pragma2 = tree.ext[1].body.block_items[0]
    assert pragma2 == c_ast.Pragma("foo")
    assert pragma2.coord.line_start == 4

    pragma3 = tree.ext[1].body.block_items[2]
    assert pragma3 == c_ast.Pragma("baz")
    assert pragma3.coord.line_start == 6

    pragma4 = tree.ext[1].body.block_items[4]
    assert pragma4 == c_ast.Pragma("")
    assert pragma4.coord.line_start == 10

    pragma5 = tree.ext[2].body.block_items[0]
    assert pragma5 == c_ast.Pragma("baz")
    assert pragma5.coord.line_start == 13

    pragma6 = tree.ext[3]
    assert pragma6 == c_ast.Pragma(r'"other \"string\""')
    assert pragma6.coord.line_start == 15


def test_pragmacomp_or_statement():
    test_input = r"""
void main() {
    int sum = 0;
    for (int i; i < 3; i++)
        #pragma omp critical
        sum += 1;

    while(sum < 10)
        #pragma omp critical
        sum += 1;

    mylabel:
        #pragma foo
        sum += 10;

    if (sum > 10)
        #pragma bar
        #pragma baz
        sum = 10;

    switch (sum)
    case 10:
        #pragma foo
        sum = 20;
}
"""
    tree = parse(test_input)

    expected_list = [
        DeclWithDefaults(
            name="sum",
            type=c_ast.TypeDecl("sum", [], None, c_ast.IdType(["int"])),
            init=c_ast.Constant("int", "0"),
        ),
        c_ast.For(
            init=c_ast.DeclList(
                [DeclWithDefaults(name="i", type=c_ast.TypeDecl("i", [], None, c_ast.IdType(["int"])))]
            ),
            cond=c_ast.BinaryOp(op="<", left=c_ast.Id(name="i"), right=c_ast.Constant("int", "3")),
            next=c_ast.UnaryOp(op="p++", expr=c_ast.Id("i")),
            stmt=c_ast.Compound(
                [
                    c_ast.Pragma("omp critical"),
                    c_ast.Assignment(op="+=", left=c_ast.Id("sum"), right=c_ast.Constant("int", "1")),
                ]
            ),
        ),
        c_ast.While(
            cond=c_ast.BinaryOp(op="<", left=c_ast.Id("sum"), right=c_ast.Constant("int", "10")),
            stmt=c_ast.Compound(
                [
                    c_ast.Pragma("omp critical"),
                    c_ast.Assignment(op="+=", left=c_ast.Id("sum"), right=c_ast.Constant("int", "1")),
                ]
            ),
        ),
        c_ast.Label(
            name="mylabel",
            stmt=c_ast.Compound(
                [
                    c_ast.Pragma("foo"),
                    c_ast.Assignment(op="+=", left=c_ast.Id("sum"), right=c_ast.Constant("int", "10")),
                ]
            ),
        ),
        c_ast.If(
            cond=c_ast.BinaryOp(op=">", left=c_ast.Id("sum"), right=c_ast.Constant("int", "10")),
            iftrue=c_ast.Compound(
                [
                    c_ast.Pragma("bar"),
                    c_ast.Pragma("baz"),
                    c_ast.Assignment(op="=", left=c_ast.Id("sum"), right=c_ast.Constant("int", "10")),
                ]
            ),
            iffalse=None,
        ),
        c_ast.Switch(
            cond=c_ast.Id("sum"),
            stmt=c_ast.Case(
                expr=c_ast.Constant("int", "10"),
                stmts=[
                    c_ast.Compound(
                        [
                            c_ast.Pragma(string="foo"),
                            c_ast.Assignment(op="=", left=c_ast.Id("sum"), right=c_ast.Constant("int", "20")),
                        ]
                    )
                ],
            ),
        ),
    ]

    assert c_ast.compare_asts(tree.ext[0].body.block_items, expected_list)


# endregion


# ========
# region ---- Parsing whole chunks of code
#
# Since we don't want to rely on the structure of ASTs too much, most of these tests are implemented with walk().
# ========


def match_constants(code: Union[str, c_ast.AST], constants: list[str]) -> bool:
    """Check that the list of all Constant values (by 'preorder' appearance) in the chunk of code is as given."""

    tree = parse(code) if isinstance(code, str) else code

    return [node.value for node in c_ast.walk(tree) if isinstance(node, c_ast.Constant)] == constants


def match_number_of_id_refs(code: Union[str, c_ast.AST], name: str, num: int) -> bool:
    """Check that the number of references to the ID with the given name matches the expected number."""

    tree = parse(code) if isinstance(code, str) else code

    return sum(1 for node in c_ast.walk(tree) if isinstance(node, c_ast.Id) and node.name == name) == num


def match_number_of_node_instances(code: Union[str, c_ast.AST], type: type[c_ast.AST], num: int) -> None:  # noqa: A002
    """Check that the amount of klass nodes in the code is the expected number."""

    tree = parse(code) if isinstance(code, str) else code

    assert sum(1 for node in c_ast.walk(tree) if isinstance(node, type)) == num


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int k = (r + 10.0) >> 6 + 8 << (3 & 0x14);",
            ["10.0", "6", "8", "3", "0x14"],
        ),
        (
            r"""char n = '\n', *prefix = "st_";""",
            [r"'\n'", '"st_"'],
        ),
        (
            "int main() {\n"
            "    int i = 5, j = 6, k = 1;\n"
            "    if ((i=j && k == 1) || k > j)\n"
            '        printf("Hello, world\n");\n'
            "    return 0;\n"
            "}",
            ["5", "6", "1", "1", '"Hello, world\\n"', "0"],
        ),
    ],
)
def test_expressions_constants(test_input: str, expected: list[str]) -> None:
    tree = parse(test_input)
    assert match_constants(tree, expected)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int main() {\n"
            "    int i = 5, j = 6, k = 1;\n"
            "    if ((i=j && k == 1) || k > j)\n"
            '        printf("Hello, world\n");\n'
            "    return 0;\n"
            "}",
            [("i", 1), ("j", 2)],
        )
    ],
)
def test_expressions_id_refs(test_input: str, expected: list[tuple[str, int]]):
    tree = parse(test_input)
    for id_, num in expected:
        assert match_number_of_id_refs(tree, id_, num)


@pytest.mark.parametrize(
    ("test_input", "expected_constants", "expected_id_ref_counts", "expected_node_instance_counts"),
    [
        (
            r"""
void foo(){
if (sp == 1)
    if (optind >= argc ||
        argv[optind][0] != '-' || argv[optind][1] == '\0')
            return -1;
    else if (strcmp(argv[optind], "--") == 0) {
        optind++;
        return -1;
    }
}
""",
            ["1", "0", r"'-'", "1", r"'\0'", "1", r'"--"', "0", "1"],
            [("argv", 3), ("optind", 5)],
            [(c_ast.If, 3), (c_ast.Return, 2), (c_ast.FuncCall, 1), (c_ast.BinaryOp, 7)],  # FuncCall is strcmp
        ),
        pytest.param(
            r"""
typedef int Hash, Node;

void HashDestroy(Hash* hash)
{
    unsigned int i;

    if (hash == NULL)
        return;

    for (i = 0; i < hash->table_size; ++i)
    {
        Node* temp = hash->heads[i];

        while (temp != NULL)
        {
            Node* temp2 = temp;

            free(temp->entry->key);
            free(temp->entry->value);
            free(temp->entry);

            temp = temp->next;

            free(temp2);
        }
    }

    free(hash->heads);
    hash->heads = NULL;

    free(hash);
}
""",
            ["0"],
            [("hash", 6), ("i", 4)],  # declarations don't count
            [(c_ast.FuncCall, 6), (c_ast.FuncDef, 1), (c_ast.For, 1), (c_ast.While, 1), (c_ast.StructRef, 10)],
            id="Hash and Node were defined as int to pacify the parser that sees they're used as types",
        ),
        (
            r"""
void x(void) {
    int a, b;
    if (a < b)
    do {
        a = 0;
    } while (0);
    else if (a == b) {
    a = 1;
    }
}
""",
            ["0", "0", "1"],
            [("a", 4)],
            [(c_ast.DoWhile, 1)],
        ),
    ],
)
def test_statements(
    test_input: str,
    expected_constants: list[str],
    expected_id_ref_counts: list[tuple[str, int]],
    expected_node_instance_counts: list[tuple[type[c_ast.AST], int]],
):
    tree = parse(test_input)

    assert match_constants(tree, expected_constants)

    for id_, count in expected_id_ref_counts:
        assert match_number_of_id_refs(tree, id_, count)

    for node_type, count in expected_node_instance_counts:
        assert match_number_of_node_instances(tree, node_type, count)


def test_empty_statements():
    test_input = r"""
void foo(void){
    ;
    return;;

    ;
}
"""

    tree = parse(test_input)

    assert match_number_of_node_instances(tree, c_ast.EmptyStatement, 3)
    assert match_number_of_node_instances(tree, c_ast.Return, 1)

    assert tree.ext[0].body.block_items[0].coord.line_start == 3
    assert tree.ext[0].body.block_items[1].coord.line_start == 4
    assert tree.ext[0].body.block_items[2].coord.line_start == 4
    assert tree.ext[0].body.block_items[3].coord.line_start == 6


def test_switch_statement():
    def is_case_node(node: Any, const_value: str) -> bool:
        return isinstance(node, c_ast.Case) and isinstance(node.expr, c_ast.Constant) and node.expr.value == const_value

    test_input_1 = r"""
int foo(void) {
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
    return 0;
}
"""

    tree1 = parse(test_input_1)
    switch = tree1.ext[0].body.block_items[0]

    block = switch.stmt.block_items
    assert len(block) == 4

    assert is_case_node(block[0], "10")
    assert len(block[0].stmts) == 3

    assert is_case_node(block[1], "20")
    assert len(block[1].stmts) == 0

    assert is_case_node(block[2], "30")
    assert len(block[2].stmts) == 1

    assert isinstance(block[3], c_ast.Default)

    test_input_2 = r"""
int foo(void) {
    switch (myvar) {
        default:
            joe = moe;
            return 10;
        case 10:
        case 20:
        case 30:
        case 40:
            break;
    }
    return 0;
}
"""

    tree2 = parse(test_input_2)
    switch = tree2.ext[0].body.block_items[0]

    block = switch.stmt.block_items
    assert len(block) == 5

    assert isinstance(block[0], c_ast.Default)
    assert len(block[0].stmts) == 2

    assert is_case_node(block[1], "10")
    assert len(block[1].stmts) == 0

    assert is_case_node(block[2], "20")
    assert len(block[2].stmts) == 0

    assert is_case_node(block[3], "30")
    assert len(block[3].stmts) == 0

    assert is_case_node(block[4], "40")
    assert len(block[4].stmts) == 1

    test_input_3 = r"""
int foo(void) {
    switch (myvar) {
    }
    return 0;
}
"""

    tree3 = parse(test_input_3)
    switch = tree3.ext[0].body.block_items[0]

    assert switch.stmt.block_items == []


@pytest.mark.parametrize(
    ("test_input", "expected_i_ref_count", "expected_For_instance_count"),
    [
        pytest.param(
            r"""
void x(void)
{
    int i;
    for (i = 0; i < 5; ++i) {
        x = 50;
    }
}
""",
            3,
            1,
            id="3 refs for i since the declaration doesn't count in the visitor.",
        ),
        pytest.param(
            r"""
void x(void)
{
    for (int i = 0; i < 5; ++i) {
        x = 50;
    }
}
""",
            2,
            1,
            id="2 refs for i since the declaration doesn't count in the visitor.",
        ),
        (
            r"""
void x(void) {
    for (int i = 0;;)
        i;
}
""",
            1,
            1,
        ),
    ],
)
def test_for_statement(test_input: str, expected_i_ref_count: int, expected_For_instance_count: int):
    tree = parse(test_input)
    assert match_number_of_id_refs(tree, "i", expected_i_ref_count)
    assert match_number_of_node_instances(tree, c_ast.For, expected_For_instance_count)


def test_whole_file():
    # See how pycparser handles a whole, real C file.

    with SAMPLE_CFILES_PATH.joinpath("memmgr_with_h.c").open(encoding="utf-8") as fp:
        code = fp.read()

    test_input = parse(code)

    assert match_number_of_node_instances(test_input, c_ast.FuncDef, 5)

    # each FuncDef also has a FuncDecl. 4 declarations
    # + 5 definitions, overall 9
    assert match_number_of_node_instances(test_input, c_ast.FuncDecl, 9)

    assert match_number_of_node_instances(test_input, c_ast.Typedef, 4)

    assert test_input.ext[4].coord
    assert test_input.ext[4].coord.line_start == 88
    assert test_input.ext[4].coord.filename == "./memmgr.h"

    assert test_input.ext[6].coord
    assert test_input.ext[6].coord.line_start == 10
    assert test_input.ext[6].coord.filename == "memmgr.c"


# def test_whole_file_with_stdio():
#     # Parse a whole file with stdio.h included by cpp

#     with SAMPLE_CFILES_PATH.joinpath("cppd_with_stdio_h.c").open(encoding="utf-8") as fp:
#         code = fp.read()

#     test_input = parse(code)

#     assert isinstance(test_input.ext[0], c_ast.Typedef)
#     assert test_input.ext[0].coord
#     assert test_input.ext[0].coord.line_start == 213
#     assert test_input.ext[0].coord.filename == r"D:\eli\cpp_stuff\libc_include/stddef.h"

#     assert isinstance(test_input.ext[-1], c_ast.FuncDef)
#     assert test_input.ext[-1].coord
#     assert test_input.ext[-1].coord.line_start == 15
#     assert test_input.ext[-1].coord.filename == "example_c_file.c"

#     assert isinstance(test_input.ext[-8], c_ast.Typedef)
#     assert isinstance(test_input.ext[-8].type, c_ast.TypeDecl)
#     assert test_input.ext[-8].name == "cookie_io_functions_t"


# endregion


# ========
# region ---- Issues related to typedef-name problem.
# ========


def test_innerscope_typedef():
    # should succeed since TT is not a type in bar
    test_input = r"""
void foo() {
    typedef char TT;
    TT x;
}
void bar() {
    unsigned TT;
}
"""
    assert isinstance(parse(test_input), c_ast.File)


def test_innerscope_typedef_error():
    # should fail since TT is not a type in bar
    test_input = r"""
void foo() {
    typedef char TT;
    TT x;
}
void bar() {
    TT y;
}
"""

    with pytest.raises(CParseError):
        parse(test_input)


@pytest.mark.parametrize(
    ("test_input", "expected_inner_param_1", "expected_inner_param_2"),
    [
        pytest.param(
            "typedef char TT;\nint foo(int (aa));\nint bar(int (TT));",
            DeclWithDefaults(name="aa", type=c_ast.TypeDecl("aa", [], None, c_ast.IdType(["int"]))),
            c_ast.Typename(
                name=None,
                quals=[],
                align=None,
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Typename(
                                name=None,
                                quals=[],
                                align=None,
                                type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["TT"])),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["int"])),
                ),
            ),
            id="foo takes an int named aa; bar takes a function taking a TT",
        ),
        pytest.param(
            "typedef char TT;\nint foo(int (aa (char)));\nint bar(int (TT (char)));",
            DeclWithDefaults(
                name="aa",
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Typename(
                                name=None,
                                quals=[],
                                align=None,
                                type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["char"])),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("aa", [], None, c_ast.IdType(["int"])),
                ),
            ),
            c_ast.Typename(
                name=None,
                quals=[],
                align=None,
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Typename(
                                name=None,
                                quals=[],
                                align=None,
                                type=c_ast.FuncDecl(
                                    args=c_ast.ParamList(
                                        [
                                            c_ast.Typename(
                                                name=None,
                                                quals=[],
                                                align=None,
                                                type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["char"])),
                                            )
                                        ]
                                    ),
                                    type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["TT"])),
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["int"])),
                ),
            ),
            id="foo takes a function taking a char; bar takes a function taking a function taking a char",
        ),
        pytest.param(
            "typedef char TT;\nint foo(int (aa[]));\nint bar(int (TT[]));",
            DeclWithDefaults(
                name="aa",
                type=c_ast.ArrayDecl(
                    type=c_ast.TypeDecl("aa", [], None, c_ast.IdType(["int"])), dim=None, dim_quals=[]
                ),
            ),
            c_ast.Typename(
                name=None,
                quals=[],
                align=None,
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Typename(
                                name=None,
                                quals=[],
                                align=None,
                                type=c_ast.ArrayDecl(
                                    type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["TT"])), dim=None, dim_quals=[]
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["int"])),
                ),
            ),
            id="foo takes an int array named aa; bar takes a function taking a TT array",
        ),
    ],
)
def test_ambiguous_parameters(test_input: str, expected_inner_param_1: c_ast.AST, expected_inner_param_2: c_ast.AST):
    # From ISO/IEC 9899:TC2, 6.7.5.3.11:
    # "If, in a parameter declaration, an identifier can be treated either
    #  as a typedef name or as a parameter name, it shall be taken as a
    #  typedef name."

    tree = parse(test_input)
    assert tree.ext[1].type.args.params[0] == expected_inner_param_1
    assert tree.ext[2].type.args.params[0] == expected_inner_param_2


def test_innerscope_reuse_typedef_name():
    # identifiers can be reused in inner scopes; the original should be
    # restored at the end of the block

    test_input_1 = r"""
typedef char TT;
void foo(void) {
    unsigned TT;
    TT = 10;
}
TT x = 5;
"""
    tree1 = parse(test_input_1)

    expected_before_end = DeclWithDefaults(name="TT", type=c_ast.TypeDecl("TT", [], None, c_ast.IdType(["unsigned"])))
    expected_after_end = DeclWithDefaults(
        name="x", type=c_ast.TypeDecl("x", [], None, c_ast.IdType(["TT"])), init=c_ast.Constant("int", "5")
    )
    assert tree1.ext[1].body.block_items[0] == expected_before_end
    assert tree1.ext[2] == expected_after_end

    # this should be recognized even with an initializer
    test_input_2 = r"""
typedef char TT;
void foo(void) {
    unsigned TT = 10;
}
"""
    tree2 = parse(test_input_2)

    expected = DeclWithDefaults(
        name="TT", type=c_ast.TypeDecl("TT", [], None, c_ast.IdType(["unsigned"])), init=c_ast.Constant("int", "10")
    )
    assert tree2.ext[1].body.block_items[0] == expected

    # before the second local variable, TT is a type; after, it's a
    # variable
    test_input_3 = r"""
typedef char TT;
void foo(void) {
    TT tt = sizeof(TT);
    unsigned TT = 10;
}
"""
    tree3 = parse(test_input_3)

    expected_before_end = DeclWithDefaults(
        name="tt",
        type=c_ast.TypeDecl("tt", [], None, c_ast.IdType(["TT"])),
        init=c_ast.UnaryOp(
            op="sizeof",
            expr=c_ast.Typename(
                name=None, quals=[], align=None, type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["TT"]))
            ),
        ),
    )
    expected_after_end = DeclWithDefaults(
        name="TT",
        type=c_ast.TypeDecl("TT", [], None, c_ast.IdType(names=["unsigned"])),
        init=c_ast.Constant("int", "10"),
    )
    assert tree3.ext[1].body.block_items[0] == expected_before_end
    assert tree3.ext[1].body.block_items[1] == expected_after_end

    # a variable and its type can even share the same name
    test_input_4 = r"""
typedef char TT;
void foo(void) {
    TT TT = sizeof(TT);
    unsigned uu = TT * 2;
}
"""
    tree4 = parse(test_input_4)

    expected_before_end = DeclWithDefaults(
        name="TT",
        type=c_ast.TypeDecl("TT", [], None, c_ast.IdType(["TT"])),
        init=c_ast.UnaryOp(
            op="sizeof",
            expr=c_ast.Typename(
                name=None, quals=[], align=None, type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["TT"]))
            ),
        ),
    )

    expected_after_end = DeclWithDefaults(
        name="uu",
        type=c_ast.TypeDecl("uu", [], None, c_ast.IdType(names=["unsigned"])),
        init=c_ast.BinaryOp(op="*", left=c_ast.Id("TT"), right=c_ast.Constant("int", "2")),
    )

    assert tree4.ext[1].body.block_items[0] == expected_before_end
    assert tree4.ext[1].body.block_items[1] == expected_after_end

    # ensure an error is raised if a type, redeclared as a variable, is
    # used as a type
    test_input_5 = r"""
typedef char TT;
void foo(void) {
    unsigned TT = 10;
    TT erroneous = 20;
}
"""
    with pytest.raises(CParseError):
        parse(test_input_5)

    # reusing a type name should work with multiple declarators
    test_input_6 = r"""
typedef char TT;
void foo(void) {
    unsigned TT, uu;
}
"""
    tree6 = parse(test_input_6)

    expected_before_end = DeclWithDefaults(name="TT", type=c_ast.TypeDecl("TT", [], None, c_ast.IdType(["unsigned"])))
    expected_after_end = DeclWithDefaults(name="uu", type=c_ast.TypeDecl("uu", [], None, c_ast.IdType(["unsigned"])))

    assert tree6.ext[1].body.block_items[0] == expected_before_end
    assert tree6.ext[1].body.block_items[1] == expected_after_end

    # reusing a type name should work after a pointer
    test_input_7 = r"""
typedef char TT;
void foo(void) {
    unsigned * TT;
}
"""
    tree7 = parse(test_input_7)

    expected = DeclWithDefaults(
        name="TT",
        type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("TT", [], None, c_ast.IdType(["unsigned"]))),
    )
    assert tree7.ext[1].body.block_items[0] == expected

    # redefine a name in the middle of a multi-declarator declaration
    test_input_8 = r"""
typedef char TT;
void foo(void) {
    int tt = sizeof(TT), TT, uu = sizeof(TT);
    int uu = sizeof(tt);
}
"""
    tree8 = parse(test_input_8)

    expected_first = DeclWithDefaults(
        name="tt",
        type=c_ast.TypeDecl("tt", [], None, c_ast.IdType(["int"])),
        init=c_ast.UnaryOp(
            op="sizeof",
            expr=c_ast.Typename(
                name=None, quals=[], align=None, type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["TT"]))
            ),
        ),
    )
    expected_second = DeclWithDefaults(name="TT", type=c_ast.TypeDecl("TT", [], None, c_ast.IdType(["int"])))
    expected_third = DeclWithDefaults(
        name="uu",
        type=c_ast.TypeDecl("uu", [], None, c_ast.IdType(["int"])),
        init=c_ast.UnaryOp(
            op="sizeof",
            expr=c_ast.Typename(
                name=None, quals=[], align=None, type=c_ast.TypeDecl(None, [], None, c_ast.IdType(["TT"]))
            ),
        ),
    )

    assert tree8.ext[1].body.block_items[0] == expected_first
    assert tree8.ext[1].body.block_items[1] == expected_second
    assert tree8.ext[1].body.block_items[2] == expected_third

    # Don't test this until we have support for it
    # self.assertEqual(expand_init(items[0].init),
    #     ['UnaryOp', 'sizeof', ['Typename', ['TypeDecl', ['IdentifierType', ['TT']]]]])
    # self.assertEqual(expand_init(items[2].init),
    #     ['UnaryOp', 'sizeof', ['ID', 'TT']])


def test_parameter_reuse_typedef_name():
    # identifiers can be reused as parameter names; parameter name scope
    # begins and ends with the function body; it's important that TT is
    # used immediately before the LBRACE or after the RBRACE, to test
    # a corner case
    test_input_1 = r"""
typedef char TT;
void foo(unsigned TT, TT bar) {
    TT = 10;
}
TT x = 5;
"""
    tree1 = parse(test_input_1)

    expected1 = c_ast.FuncDef(
        decl=DeclWithDefaults(
            name="foo",
            type=c_ast.FuncDecl(
                args=c_ast.ParamList(
                    [
                        DeclWithDefaults(name="TT", type=c_ast.TypeDecl("TT", [], None, c_ast.IdType(["unsigned"]))),
                        DeclWithDefaults(name="bar", type=c_ast.TypeDecl("bar", [], None, c_ast.IdType(["TT"]))),
                    ]
                ),
                type=c_ast.TypeDecl("foo", [], None, c_ast.IdType(["void"])),
            ),
        ),
        param_decls=None,
        body=c_ast.Compound([c_ast.Assignment(op="=", left=c_ast.Id("TT"), right=c_ast.Constant("int", "10"))]),
    )

    assert tree1.ext[1] == expected1

    # the scope of a parameter name in a function declaration ends at the
    # end of the declaration...so it is effectively never used; it's
    # important that TT is used immediately after the declaration, to
    # test a corner case
    test_input_2 = r"""
typedef char TT;
void foo(unsigned TT, TT bar);
TT x = 5;
"""

    tree2 = parse(test_input_2)

    expected2 = DeclWithDefaults(
        name="foo",
        type=c_ast.FuncDecl(
            args=c_ast.ParamList(
                [
                    DeclWithDefaults(name="TT", type=c_ast.TypeDecl("TT", [], None, c_ast.IdType(["unsigned"]))),
                    DeclWithDefaults(name="bar", type=c_ast.TypeDecl("bar", [], None, c_ast.IdType(["TT"]))),
                ]
            ),
            type=c_ast.TypeDecl("foo", [], None, c_ast.IdType(["void"])),
        ),
    )
    assert tree2.ext[1] == expected2

    # ensure an error is raised if a type, redeclared as a parameter, is
    # used as a type
    test_input_3 = r"""
typedef char TT;
void foo(unsigned TT, TT bar) {
    TT erroneous = 20;
}
"""

    with pytest.raises(CParseError):
        parse(test_input_3)


def test_nested_function_decls():
    # parameter names of nested function declarations must not escape into
    # the top-level function _definition's_ scope; the following must
    # succeed because TT is still a typedef inside foo's body
    test_input = r"""
typedef char TT;
void foo(unsigned bar(int TT)) {
    TT x = 10;
}
"""
    assert isinstance(parse(test_input), c_ast.File)


def test_samescope_reuse_name():
    # a typedef name cannot be reused as an object name in the same scope
    test_input_1 = r"""
typedef char TT;
char TT = 5;
"""
    with pytest.raises(CParseError):
        parse(test_input_1)

    # ...and vice-versa
    test_input_2 = r"""
char TT = 5;
typedef char TT;
"""

    with pytest.raises(CParseError):
        parse(test_input_2)


# endregion

# endregion
