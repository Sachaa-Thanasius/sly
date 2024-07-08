"""Support docstring-parsing classes."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .lex import Lexer
    from .yacc import Parser

__all__ = ("DocParseMeta",)


class DocParseMeta(type):
    '''Metaclass that processes the class docstring through a parser and incorporates the result into the resulting
    class definition.

    Extended Summary
    ----------------
    This allows Python classes to be defined with alternative syntax.

    Examples
    --------
    To use this class, you first need to define a lexer and parser:

        from sly import Lexer, Parser
        class MyLexer(Lexer):
            ...

        class MyParser(Parser):
            ...

    You then need to define a metaclass that inherits from DocParseMeta. This class must specify the associated lexer
    and parser classes. For example:

        class MyDocParseMeta(DocParseMeta):
            lexer = MyLexer
            parser = MyParser

    This metaclass is then used as a base for processing user-defined classes:

        class Base(metaclass=MyDocParseMeta):
            pass

        class Spam(Base):
            """
            doc string is parsed
            ...
            """

    It is expected that the MyParser() class would return a dictionary. This dictionary is used to create the final
    class Spam in this example.
    '''

    if TYPE_CHECKING:
        lexer: type[Lexer]
        parser: type[Parser]

    def __new__(cls, clsname: str, bases: tuple[type, ...], namespace: dict[str, Any]):
        if "__doc__" in namespace:
            lexer = cls.lexer()
            parser = cls.parser()
            lexer.cls_name = parser.cls_name = clsname  # pyright: ignore # Runtime attribute assignment.
            lexer.cls_qualname = parser.cls_qualname = namespace["__qualname__"]  # pyright: ignore # Runtime attribute assignment.
            lexer.cls_module = parser.cls_module = namespace["__module__"]  # pyright: ignore # Runtime attribute assignment.
            parsedict = parser.parse(lexer.tokenize(namespace["__doc__"]))
            if not isinstance(parsedict, dict):
                raise ValueError("Parser must return a dictionary")
            namespace.update(parsedict)  # pyright: ignore [reportUnknownArgumentType] # It's enough that it's a dict.
        return super().__new__(cls, clsname, bases, namespace)

    @classmethod
    def __init_subclass__(cls) -> None:
        if not (hasattr(cls, "parser") and hasattr(cls, "lexer")):
            raise RuntimeError("This class must have a parser and lexer.")
