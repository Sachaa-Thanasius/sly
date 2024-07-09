from difflib import Differ

import pytest
from cparsing import c_ast

_differ = Differ()


def pytest_assertrepr_compare(config: pytest.Config, op: str, left: object, right: object):
    """Get a better output for comparisons between AST nodes.

    This attempts to emulate pytest's comparison output for lists and dicts.
    """

    if op == "==" and isinstance(left, c_ast.AST) and isinstance(right, c_ast.AST):
        verbosity = config.get_verbosity()

        # Truncate the reprs with centered ellipses.
        left_node_repr = repr(left)
        if len(left_node_repr) > 30:
            left_node_repr = left_node_repr[:14] + "..." + left_node_repr[16:]

        right_node_repr = repr(right)
        if len(right_node_repr) > 30:
            right_node_repr = right_node_repr[:14] + "..." + right_node_repr[16:]

        # By default, only compare the reprs.
        displayed_comparison = [
            f"{left_node_repr} == {right_node_repr}",
            "",
            "Differing items:",
            f"{left!r} != {right!r}",
        ]

        # Augment the comparison based on verbosity.
        if verbosity == 0:
            displayed_comparison.append("Use -v to get more diff")
        elif verbosity == 1:
            dump_diff = _get_ast_diff(left, right)

            displayed_comparison.extend(("", "Full diff:", *dump_diff[:2]))

            if len(dump_diff) > 2:
                displayed_comparison.append(
                    f"...Full output truncated ({len(dump_diff) - 2} lines hidden), use '-vv' to show"
                )
        elif verbosity >= 2:
            dump_diff = _get_ast_diff(left, right)

            displayed_comparison.extend(("", "Full diff:", *dump_diff))

        return displayed_comparison

    return None


def _get_ast_diff(left: c_ast.AST, right: c_ast.AST) -> list[str]:
    left_dump = c_ast.dump(left, indent=4).splitlines()
    right_dump = c_ast.dump(right, indent=4).splitlines()
    return list(_differ.compare(left_dump, right_dump))
