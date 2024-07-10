"""An attempt at making a C parser with sly. Heavily based on pycparser, which uses ply."""

from .c_context import CContext, parse, parse_file, preprocess_file

__all__ = ("CContext", "parse", "preprocess_file", "parse_file")
