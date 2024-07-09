"""An attempt at making a C parser with sly. Heavily based on pycparser, which uses ply."""

from ._main import parse, parse_file, preprocess_file

__all__ = ("parse", "preprocess_file", "parse_file")
