from __future__ import annotations

import importlib.util
import os
import types
from collections.abc import Sequence

import sly


def create_code(found_parsers: dict[str, type[sly.Parser]]) -> str:
    output_blocks: list[str] = []
    return "\n\n".join(output_blocks)


def module_from_file_location(name: str, filepath: str | os.PathLike[str]) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, filepath)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main(args: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("-f", "--filepath", action="store", required=True)
    parser.add_argument("-o", "--output", action="store", default="standalone_parser.py")

    p_args = parser.parse_args(args)

    _head, tail = os.path.split(p_args.filepath)
    filename, _ext = os.path.splitext(tail)  # noqa: PTH122

    module = module_from_file_location(filename, p_args.filepath)

    found_parsers = {
        name: obj
        for name, obj in module.__dict__.items()
        if isinstance(obj, type) and (obj is not sly.Parser) and issubclass(obj, sly.Parser)
    }

    output_code = create_code(found_parsers)

    with open(p_args.output, "w") as f:
        f.write(output_code)


if __name__ == "__main__":
    main()
