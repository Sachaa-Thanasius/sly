"""A collection of large regex patterns helpful for tokenization of C."""

# Identifiers

_digit = r"[0-9]"
_hexadecimal_digit = r"[0-9A-Fa-f]"
_nondigit = r"[a-zA-Z_]"

_universal_character_name = rf"\\u{_hexadecimal_digit}{{4}}|\\U{_hexadecimal_digit}{{8}}"

_identifier_nondigit = rf"{_nondigit}|({_universal_character_name})"
identifier = rf"({_identifier_nondigit})(({_identifier_nondigit})|{_digit})*"


# Integer constants

_nonzero_digit = r"[1-9]"
_decimal_constant = rf"{_nonzero_digit}{_digit}*"

_octal_digit = r"[0-7]"
_octal_constant = rf"0{_octal_digit}*"

_hexadecimal_prefix = r"0[xX]"
_hexadecimal_constant = rf"{_hexadecimal_prefix}{_hexadecimal_digit}+"

_unsigned_suffix = r"[uU]"
_long_suffix = r"[lL]"
_long_long_suffix = r"ll|LL"
_integer_suffix = "|".join(
    (
        rf"({_unsigned_suffix}{_long_suffix}?)",
        rf"({_unsigned_suffix}({_long_long_suffix}))",
        rf"({_long_suffix}{_unsigned_suffix}?)",
        rf"(({_long_long_suffix}){_unsigned_suffix}?)",
    )
)

_integer_constant = "|".join(
    (
        rf"({_decimal_constant}({_integer_suffix})?)",
        rf"({_octal_constant}({_integer_suffix})?)",
        rf"({_hexadecimal_constant}({_integer_suffix})?)",
    )
)


# Floating constants

_sign = r"[-+]"
_digit_sequence = rf"{_digit}+"
_floating_suffix = r"[flFL]"

_fractional_constant = "|".join(
    (
        rf"({_digit_sequence}?\.{_digit_sequence})",
        rf"({_digit_sequence}\.)",
    )
)

_exponent_part = rf"[eE]{_sign}?{_digit_sequence}"
_decimal_floating_constant = "|".join(
    (
        rf"(({_fractional_constant}){_exponent_part}?{_floating_suffix}?)",
        rf"({_digit_sequence}{_exponent_part}{_floating_suffix}?)",
    )
)

_hexadecimal_digit_sequence = rf"{_hexadecimal_digit}+"
_hexadecimal_fractional_constant = "|".join(
    (
        rf"(({_hexadecimal_digit_sequence})?\.{_hexadecimal_digit_sequence})",
        rf"({_hexadecimal_digit_sequence}\.)",
    )
)
_binary_exponent_part = rf"[pP]{_sign}?{_digit_sequence}"
_hexadecimal_floating_constant = "|".join(
    (
        rf"({_hexadecimal_prefix}({_hexadecimal_fractional_constant})({_binary_exponent_part}){_floating_suffix}?)",
        rf"({_hexadecimal_prefix}{_hexadecimal_digit_sequence}{_binary_exponent_part}{_floating_suffix}?)",
    )
)


# Constants

constant = "|".join(
    (
        rf"({_integer_constant})",
        rf"({_decimal_floating_constant})",
        rf"({_hexadecimal_floating_constant})",
    )
)


# Preprocessing numbers

preprocessing_number = r"\.?[0-9]([0-9A-Za-z_\.]|[eEpP][+-])*"


# Character and string constants

_simple_escape_sequence = r"""\\['"?\\abfnrtv]"""
_octal_escape_sequence = rf"\\({_octal_digit}{{1,3}})"
_hexadecimal_escape_sequence = rf"\\x{_hexadecimal_digit}+"
escape_sequence = "|".join(
    (
        f"({_simple_escape_sequence})",
        f"({_octal_escape_sequence})",
        f"({_hexadecimal_escape_sequence})",
        f"({_universal_character_name})",
    )
)
