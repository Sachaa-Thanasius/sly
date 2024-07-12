import platform
from typing import Literal

import pytest
from cparsing import c_ast, parse

# ============================================================================
# region -------- Helpers
# ============================================================================


def cpp_path() -> Literal["gcc", "cpp"]:
    """Path to cpp command."""

    if platform.system() == "Darwin":
        return "gcc"
    return "cpp"


def cpp_args(args: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Turn args into a suitable format for passing to cpp."""

    if platform.system() == "Darwin":
        return ("-E", *args)
    return args


def convert_c_to_c(src: str, *, reduce_parentheses: bool = False) -> str:
    return c_ast.unparse(parse(src), reduce_parentheses=reduce_parentheses)


def assert_c_to_c_is_correct(src: str, *, reduce_parentheses: bool = False):
    """Check that the c2c translation was correct by parsing the code generated by c2c for `src` and comparing the AST
    of that with the original AST.
    """

    reparsed_src = convert_c_to_c(src, reduce_parentheses=reduce_parentheses)
    assert c_ast.compare(parse(src), parse(reparsed_src))


# endregion


# ============================================================================
# region -------- Tests
# ============================================================================


@pytest.mark.parametrize(
    "test_input",
    [
        "int a;",
        "int b, a;",
        "int c, b, a;",
        "auto int a;",
        "register int a;",
        "_Thread_local int a;",
    ],
)
def test_trivial_decls(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        "int a;",
        "int b, a;",
        "int c, b, a;",
        "auto int a;",
        "register int a;",
        "_Thread_local int a;",
    ],
)
def test_complex_decls(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        "_Alignas(32) int b;",
        "int _Alignas(32) a;",
        "_Alignas(32) _Atomic(int) b;",
        "_Atomic(int) _Alignas(32) b;",
        "_Alignas(long long) int a;",
        "int _Alignas(long long) a;",
        """\
typedef struct node_t {
    _Alignas(64) void* next;
    int data;
} node;
""",
        """\
typedef struct node_t {
    void _Alignas(64) * next;
    int data;
} node;
""",
    ],
)
def test_alignment(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        """\
int main(void)
{
    int a, b;
    (a == 0) ? (b = 1) : (b = 2);
}""",
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_ternary(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        """\
int main() {
    int b = (int) f;
    int c = (int*) f;
}""",
        """\
int main() {
    int a = (int) b + 8;
    int t = (int) c;
}
""",
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_casts(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize("test_input", ["int arr[] = {1, 2, 3};"])
def test_initlist(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        """\
int main(void)
{
    int a;
    int b = a++;
    int c = ++a;
    int d = a--;
    int e = --a;
}
""",
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_exprs(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        pytest.param(
            """\
int main() {
    int a;
    a = 5;
    ;
    b = - - a;
    return a;
}""",
            id="statements (note two minuses here)",
        ),
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_statements(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        """\
typedef struct node_t {
    struct node_t* next;
    int data;
} node;
""",
    ],
)
def test_struct_decl(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        """\
int main(argc, argv)
int argc;
char** argv;
{
    return 0;
}
""",
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_krstyle(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        """\
int main() {
    switch (myvar) {
    case 10:
    {
        k = 10;
        p = k + 1;
        break;
    }
    case 20:
    case 30:
        return 20;
    default:
        break;
    }
}
""",
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_switchcase(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        """\
int main()
{
    int i[1][1] = { { 1 } };
}""",
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_nest_initializer_list(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        """\
struct test
{
    int i;
    struct test_i_t
    {
        int k;
    } test_i;
    int j;
};
struct test test_var = {.i = 0, .test_i = {.k = 1}, .j = 2};
""",
    ],
)
def test_nest_named_initializer(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        """\
int main()
{
    int i[1] = { (1, 2) };
}""",
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_expr_list_in_initializer_list(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        """\
_Noreturn int x(void) {
    abort();
}
""",
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_noreturn(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        """\
void x() {
    if (i < j)
        tmp = C[i], C[i] = C[j], C[j] = tmp;
    if (i <= j)
        i++, j--;
}
""",
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_expr_list_with_semi(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        """\
void x() {
    (a = b, (b = c, c = a));
}
""",
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_exprlist_with_subexprlist(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        pytest.param(
            """\
void f(int x) { return x; }
int main(void) { f((1, 2)); return 0; }
""",
            id="comma operator funcarg",
        ),
        pytest.param(
            """\
void f() {
    (0, 0) ? (0, 0) : (0, 0);
}
""",
            id="comma op in ternary",
        ),
        pytest.param(
            """
void f() {
    i = (a, b, c);
}
""",
            id="comma op reassignment",
        ),
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_comma_operator(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        """\
#pragma foo
void f() {
    #pragma bar
    i = (a, b, c);
    if (d)
        #pragma qux
        j = e;
    if (d)
        #pragma qux
        #pragma quux
        j = e;
}
typedef struct s {
#pragma baz
} s;
""",
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_pragma(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        'char **foo = (char *[]){ "x", "y", "z" };',
        "int i = ++(int){ 1 };",
        "struct foo_s foo = (struct foo_s){ 1, 2 };",
    ],
)
def test_compound_literal(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        """\
enum e
{
    a,
    b = 2,
    c = 3
};
""",
        """\
enum f
{
    g = 4,
    h,
    i
};
""",
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_enum(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize("test_input", ["typedef enum EnumName EnumTypedefName;"])
def test_enum_typedef(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize("test_input", ["int g(const int a[const 20]){}"])
@pytest.mark.xfail(reason="TODO")
def test_array_decl(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize("test_input", ["const int ** const  x;"])
def test_ptr_decl(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [pytest.param(f"int x = {'sizeof(' * 30}1{')' * 30}", id="30 sizeofs wrapped around 1")],
)
@pytest.mark.xfail(reason="TODO")
def test_nested_sizeof(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        '_Static_assert(sizeof(int) == sizeof(int), "123");',
        'int main() { _Static_assert(sizeof(int) == sizeof(int), "123"); } ',
        "_Static_assert(sizeof(int) == sizeof(int));",
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_static_assert(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        "_Atomic int x;",
        "_Atomic int* x;",
        "int* _Atomic x;",
    ],
)
def test_atomic_qual(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        pytest.param("\nint main() {\n}", marks=pytest.mark.xfail(reason="TODO"), id="issue 36"),
        pytest.param(
            """\
int main(void)
{
    unsigned size;
    size = sizeof(size);
    return 0;
}""",
            marks=pytest.mark.xfail(reason="TODO"),
            id="issue 37",
        ),
        pytest.param(
            "\nstruct foo;",
            id="issue 66, 1; A non-existing body must not be generated (previous valid behavior, still working)",
        ),
        pytest.param(
            "\nstruct foo {};",
            marks=pytest.mark.xfail(reason="TODO"),
            id="issue 66, 2; An empty body must be generated (added behavior)",
        ),
        pytest.param(
            """\
void x(void) {
    int i = (9, k);
}
""",
            marks=pytest.mark.xfail(reason="TODO"),
            id="issue 83",
        ),
        pytest.param(
            """\
void x(void) {
    for (int i = 0;;)
        i;
}
""",
            marks=pytest.mark.xfail(reason="TODO"),
            id="issue 84",
        ),
        pytest.param(
            "int array[3] = {[0] = 0, [1] = 1, [1+1] = 2};",
            id="issue 246",
        ),
    ],
)
def test_issues(test_input: str):
    assert_c_to_c_is_correct(test_input)


@pytest.mark.parametrize(
    ("test_input", "expected_stubs"),
    [
        (
            "void noop(void);\nvoid *something(void *thing);\nint add(int x, int y);\n",
            (
                "void noop(void)",
                "void *something(void *thing)",
                "int add(int x, int y)",
            ),
        )
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_partial_funcdecl_generation(test_input: str, expected_stubs: tuple[str, ...]):
    tree = parse(test_input)
    stubs = [c_ast.unparse(node) for node in c_ast.walk(tree) if isinstance(node, c_ast.FuncDecl)]

    assert len(stubs) == len(expected_stubs)
    assert all(expected_stub in stubs for expected_stub in expected_stubs)


@pytest.mark.xfail(reason="TODO")
def test_array_decl_subnodes():
    tree = parse("const int a[const 20];")

    assert c_ast.unparse(tree.ext[0].type) == "const int [const 20]"  # type: ignore
    assert c_ast.unparse(tree.ext[0].type.type) == "const int"  # type: ignore


@pytest.mark.xfail(reason="TODO")
def test_ptr_decl_subnodes():
    tree = parse("const int ** const  x;")

    assert c_ast.unparse(tree.ext[0].type) == "const int ** const"  # type: ignore
    assert c_ast.unparse(tree.ext[0].type.type) == "const int *"  # type: ignore
    assert c_ast.unparse(tree.ext[0].type.type.type) == "const int"  # type: ignore


@pytest.mark.parametrize(
    ("test_input", "expected_reparsed_input"),
    [
        ("_Atomic(int) x;", "_Atomic int x;\n"),
        ("_Atomic(int*) x;", "int * _Atomic x;\n"),
        ("_Atomic(_Atomic(int)*) x;", "_Atomic int * _Atomic x;\n"),
        ("typedef _Atomic(int) atomic_int;", "typedef _Atomic int atomic_int;\n"),
        pytest.param(
            "typedef _Atomic(_Atomic(_Atomic(int (*)(void)) *) *) t;",
            "typedef int (* _Atomic * _Atomic * _Atomic t)(void);\n",
            marks=pytest.mark.xfail(reason="TODO"),
        ),
        pytest.param(
            """\
typedef struct node_t {
    _Atomic(void*) a;
    _Atomic(void) *b;
    _Atomic void *c;
} node;
""",
            """\
typedef struct node_t
{
  void * _Atomic a;
  _Atomic void *b;
  _Atomic void *c;
} node;

""",
            marks=pytest.mark.xfail(reason="TODO"),
        ),
    ],
)
def test_atomic_specifier_into_qualifier(test_input: str, expected_reparsed_input: str):
    """Test that the _Atomic specifier gets turned into a qualifier."""

    assert convert_c_to_c(test_input) == expected_reparsed_input
    assert_c_to_c_is_correct(test_input)

    # TODO: Regeneration with multiple qualifiers is not fully supported.
    # Ref: https://github.com/eliben/pycparser/issues/433
    # assert is_c_to_c_correct('auto const _Atomic(int *) a;')


@pytest.mark.parametrize("test_input", ["int x = a + b + c + d;"])
@pytest.mark.parametrize(
    ("reduce_parentheses", "expected_reparsed_input"),
    [
        (False, "int x = ((a + b) + c) + d;\n"),
        (True, "int x = a + b + c + d;\n"),
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_reduce_parentheses_binaryops(test_input: str, reduce_parentheses: bool, expected_reparsed_input: str):
    assert convert_c_to_c(test_input, reduce_parentheses=reduce_parentheses) == expected_reparsed_input


@pytest.mark.parametrize(
    "test_input",
    [
        "int x = a*b*c*d;",
        "int x = a+b*c*d;",
        "int x = a*b+c*d;",
        "int x = a*b*c+d;",
        "int x = (a+b)*c*d;",
        "int x = (a+b)*(c+d);",
        "int x = (a+b)/(c-d);",
        "int x = a+b-c-d;",
        "int x = a+(b-c)-d;",
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_minimum_parentheses_binaryops(test_input: str):
    """Test code snippets with the minimum number of (necessary) parentheses."""

    assert_c_to_c_is_correct(test_input, reduce_parentheses=True)
    reparsed_source = convert_c_to_c(test_input, reduce_parentheses=True)
    assert reparsed_source.count("(") == test_input.count("(")


@pytest.mark.xfail(reason="TODO")
def test_to_type():
    test_input = "int *x;"
    test_func = c_ast.FuncCall(c_ast.Id("test_fun"), c_ast.ExprList([]))

    tree = parse(test_input)
    assert c_ast.unparse(c_ast.Cast(tree.ext[0].type, test_func)) == "(int *) test_fun()"  # type: ignore
    assert c_ast.unparse(c_ast.Cast(tree.ext[0].type.type, test_func)) == "(int) test_fun()"  # type: ignore


@pytest.mark.xfail("Need the C files for this test.")
@pytest.mark.skipif(platform.system() != "Linux", reason="cpp only works on Unix")
def test_to_type_with_cpp():
    from pathlib import Path

    from cparsing import parse_file

    SAMPLE_CFILES_PATH = Path("./tests/c_files").resolve(strict=True)
    test_func = c_ast.FuncCall(c_ast.Id("test_fun"), c_ast.ExprList([]))
    memmgr_path = SAMPLE_CFILES_PATH / "memmgr.h"

    tree = parse_file(memmgr_path, use_cpp=True, cpp_path=cpp_path(), cpp_args=cpp_args())
    assert c_ast.unparse(c_ast.Cast(tree.ext[-3].type.type, test_func)) == "(void *) test_fun()"  # type: ignore
    assert c_ast.unparse(c_ast.Cast(tree.ext[-3].type.type.type, test_func)) == "(void) test_fun()"  # type: ignore


@pytest.mark.parametrize(
    ("test_tree", "expected"),
    [
        (c_ast.If(None, None, None), "if ()\n  \n"),
        (c_ast.If(None, None, c_ast.If(None, None, None)), "if ()\n  \nelse\n  if ()\n  \n"),
        (
            c_ast.If(None, None, c_ast.If(None, None, c_ast.If(None, None, None))),
            "if ()\n  \nelse\n  if ()\n  \nelse\n  if ()\n  \n",
        ),
        (
            c_ast.If(
                None,
                c_ast.Compound([]),
                c_ast.If(None, c_ast.Compound([]), c_ast.If(None, c_ast.Compound([]), None)),
            ),
            "if ()\n{\n}\nelse\n  if ()\n{\n}\nelse\n  if ()\n{\n}\n",
        ),
    ],
)
def test_nested_else_if_line_breaks(test_tree: c_ast.AST, expected: str):
    assert c_ast.unparse(test_tree) == expected


# endregion
