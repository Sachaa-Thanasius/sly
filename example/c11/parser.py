# pyright: reportRedeclaration=none

from __future__ import annotations

from sly import Parser
from sly.yacc import YaccProduction as Prod

from .context import CNameContext
from .lexer import CLexer


TYPE_CHECKING = False

if TYPE_CHECKING:
    from sly.types import _


class CParser(Parser):
    def __init__(self, ctx: CNameContext):
        self.ctx = ctx

    tokens = CLexer.tokens

    @_("ID TYPE")
    def typedef_name(self, p: Prod): ...

    @_("ID VARIABLE")
    def var_name(self, p: Prod): ...

    @_("typedef_name")
    def typedef_name_spec(self, p: Prod): ...

    @_("typedef_name", "var_name")
    def general_identifier(self, p: Prod): ...
