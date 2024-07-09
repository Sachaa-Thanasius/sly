import os
from collections import ChainMap
from collections.abc import Sequence
from typing import Union

from .c_ast import File
from .c_lexer import CLexer
from .c_parser import CParser

_StrPath = Union[str, os.PathLike[str]]


__all__ = ("parse", "preprocess_file", "parse_file")


def parse(source: str, filename: str = "", parser_type: type[CParser] = CParser) -> File:
    scope_stack: ChainMap[str, bool] = ChainMap()
    lexer = CLexer(scope_stack)
    parser = parser_type(scope_stack)
    return parser.parse(lexer.tokenize(source))


def preprocess_file(filename: str, cpp_path: str = "cpp", cpp_args: Sequence[str] = ()) -> str:
    """Preprocess a file using cpp.

    Arguments
    ---------
    filename: str
        The name of the file to preprocess.
    cpp_path: str, default="cpp"
        The path to the cpp compiler. Default is "cpp", which assumes it's already on PATH.
    cpp_args: Sequence[str], default=()
        A sequence of command line arguments for cpp, e.g. [r"-I../utils/fake_libc_include"]. Default is an empty
        tuple. Raw strings are recommended, especially when passing in paths.

    Returns
    -------
    str
        The preprocessed file's contents.

    Raises
    ------
    RuntimeError
        If the invocation of cpp failed. This will display the original error.
    """

    import subprocess

    cmd = [cpp_path, *cpp_args, filename]

    try:
        # Note the use of universal_newlines to treat all newlines as \n for Python's purpose
        preprocessed_text = subprocess.check_output(cmd, universal_newlines=True)  # noqa: S603
    except OSError as err:
        msg = 'Unable to invoke "cpp". Make sure its path was passed correctly.'
        raise RuntimeError(msg) from err
    else:
        return preprocessed_text


def parse_file(
    file: _StrPath,
    encoding: str = "utf-8",
    *,
    use_cpp: bool = False,
    cpp_path: str = "cpp",
    cpp_args: Sequence[str] = (),
    parser_type: type[CParser] = CParser,
) -> File:
    if use_cpp:
        source = preprocess_file(os.fspath(file), cpp_path, cpp_args)
    else:
        with open(file, encoding=encoding) as fp:
            source = fp.read()

    return parse(source, parser_type=parser_type)
