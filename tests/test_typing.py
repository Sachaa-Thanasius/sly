from __future__ import annotations

import importlib
import pkgutil
import sys
import types

import pytest

import sly


if sys.version_info >= (3, 14):  # pragma: >=3.14 cover
    from annotationlib import get_annotations
else:  # pragma: <3.14 cover
    from inspect import get_annotations


def can_have_annotations(obj: object) -> bool:
    return isinstance(obj, (type, types.ModuleType)) or callable(obj)


@pytest.mark.parametrize(
    "mod",
    [
        importlib.import_module(mod_info.name)
        for mod_info in pkgutil.iter_modules(sly.__spec__.submodule_search_locations, prefix=f"{sly.__spec__.name}.")
        if mod_info.name != f"{sly.__spec__.name}.types"
    ],
)
def test_library_annotations_are_valid(mod: types.ModuleType):
    get_annotations(mod, eval_str=True)

    for obj in filter(can_have_annotations, mod.__dict__.values()):
        get_annotations(obj, eval_str=True)


def test_cannot_import_sly_types():
    expected_msg = "This module is not meant to be imported at runtime; see docstring for more details."

    with pytest.raises(ImportError) as exc_info:
        import sly.types  # noqa: F401, PLC0415

    assert exc_info.value.args[0] == expected_msg

    with pytest.raises(ImportError) as exc_info:
        from sly.types import _  # noqa: F401, PLC0415

    assert exc_info.value.args[0] == expected_msg
