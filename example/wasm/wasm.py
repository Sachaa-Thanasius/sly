"""wasm.py: Experimental builder for Wasm binary encoding. Use at your own peril."""
# Author: David Beazley (@dabeaz)
# Copyright (C) 2019
# http://www.dabeaz.com

import enum
import json
import struct
from collections import defaultdict


def encode_unsigned(value) -> bytes:
    """Produce an LEB128 encoded unsigned integer."""

    parts: list[int] = []
    while value:
        parts.append((value & 0x7F) | 0x80)
        value >>= 7
    if not parts:
        parts.append(0)
    parts[-1] &= 0x7F
    return bytes(parts)


def encode_signed(value) -> bytes:
    """Produce a LEB128 encoded signed integer."""

    parts: list[int] = []
    if value < 0:
        # Sign extend the value up to a multiple of 7 bits
        value = (1 << (value.bit_length() + (7 - value.bit_length() % 7))) + value
        negative = True
    else:
        negative = False
    while value:
        parts.append((value & 0x7F) | 0x80)
        value >>= 7
    if not parts or (not negative and parts[-1] & 0x40):
        parts.append(0)
    parts[-1] &= 0x7F
    return bytes(parts)


assert encode_unsigned(624485) == bytes([0xE5, 0x8E, 0x26])
assert encode_unsigned(127) == bytes([0x7F])
assert encode_signed(-624485) == bytes([0x9B, 0xF1, 0x59])
assert encode_signed(127) == bytes([0xFF, 0x00])


def encode_f64(value) -> bytes:
    """
    Encode a 64-bit floating point as little endian
    """
    return struct.pack("<d", value)


def encode_f32(value) -> bytes:
    """
    Encode a 32-bit floating point as little endian.
    """
    return struct.pack("<f", value)


def encode_name(value):
    """
    Encode a name as UTF-8
    """
    data = value.encode("utf-8")
    return encode_vector(data)


def encode_vector(items):
    """
    Items is a list of encoded value or bytess
    """
    if isinstance(items, bytes):
        return encode_unsigned(len(items)) + items
    else:
        return encode_unsigned(len(items)) + b"".join(items)


# ------------------------------------------------------------
# Instruction encoding enums.
#
# Wasm defines 4 core data types [i32, i64, f32, f64].  These type
# names are used in various places (specifying functions, globals,
# etc.).  However, the type names are also used as a namespace for
# type-specific instructions such as i32.add.  We're going to use
# Python enums to set up this arrangement in a clever way that
# makes it possible to do both of these tasks.

# Metaclass for instruction encoding categories. The class itself
# can be used as an integer when encoding instructions.


class HexEnumMeta(enum.EnumMeta):
    def __int__(cls):
        return int(cls._encoding)

    __index__ = __int__

    def __repr__(cls):
        return cls.__name__

    @classmethod
    def __prepare__(meta, name, bases, *, encoding=0):
        return super().__prepare__(name, bases)

    @staticmethod
    def __new__(meta, clsname, bases, methods, *, encoding=0):
        cls = super().__new__(meta, clsname, bases, methods)
        cls._encoding = encoding
        return cls


class HexEnum(enum.IntEnum):
    def __repr__(self):
        return f"<{self!s}: 0x{self:x}>"


HexEnum.__class__ = HexEnumMeta


class i32(HexEnum, encoding=0x7F):
    eqz = 0x45
    eq = 0x46
    ne = 0x47
    lt_s = 0x48
    lt_u = 0x49
    gt_s = 0x4A
    gt_u = 0x4B
    le_s = 0x4C
    le_u = 0x4D
    ge_s = 0x4E
    ge_u = 0x4F
    clz = 0x67
    ctz = 0x68
    popcnt = 0x69
    add = 0x6A
    sub = 0x6B
    mul = 0x6C
    div_s = 0x6D
    div_u = 0x6E
    rem_s = 0x6F
    rem_u = 0x70
    and_ = 0x71
    or_ = 0x72
    xor = 0x73
    shl = 0x74
    shr_s = 0x75
    shr_u = 0x76
    rotl = 0x77
    rotr = 0x78
    wrap_i64 = 0xA7
    trunc_f32_s = 0xA8
    trunc_f32_u = 0xA9
    trunc_f64_s = 0xAA
    trunc_f64_u = 0xAB
    reinterpret_f32 = 0xBC
    load = 0x28
    load8_s = 0x2C
    load8_u = 0x2D
    load16_s = 0x2E
    load16_u = 0x2F
    store = 0x36
    store8 = 0x3A
    store16 = 0x3B
    const = 0x41


class i64(HexEnum, encoding=0x7E):
    eqz = 0x50
    eq = 0x51
    ne = 0x52
    lt_s = 0x53
    lt_u = 0x54
    gt_s = 0x55
    gt_u = 0x56
    le_s = 0x57
    le_u = 0x58
    ge_s = 0x59
    ge_u = 0x5A
    clz = 0x79
    ctz = 0x7A
    popcnt = 0x7B
    add = 0x7C
    sub = 0x7D
    mul = 0x7E
    div_s = 0x7F
    div_u = 0x80
    rem_s = 0x81
    rem_u = 0x82
    and_ = 0x83
    or_ = 0x84
    xor = 0x85
    shl = 0x86
    shr_s = 0x87
    shr_u = 0x88
    rotl = 0x89
    rotr = 0x8A
    extend_i32_s = 0xAC
    extend_i32_u = 0xAD
    trunc_f32_s = 0xAE
    trunc_f32_u = 0xAF
    trunc_f64_s = 0xB0
    trunc_f64_u = 0xB1
    reinterpret_f64 = 0xBD
    load = 0x29
    load8_s = 0x30
    load8_u = 0x31
    load16_s = 0x32
    load16_u = 0x33
    load32_s = 0x34
    load32_u = 0x35
    store = 0x37
    store8 = 0x3C
    store16 = 0x3D
    store32 = 0x3E
    const = 0x42


class f32(HexEnum, encoding=0x7D):
    eq = 0x5B
    ne = 0x5C
    lt = 0x5D
    gt = 0x5E
    le = 0x5F
    ge = 0x60
    abs = 0x8B
    neg = 0x8C
    ceil = 0x8D
    floor = 0x8E
    trunc = 0x8F
    nearest = 0x90
    sqrt = 0x91
    add = 0x92
    sub = 0x93
    mul = 0x94
    div = 0x95
    min = 0x96
    max = 0x97
    copysign = 0x98
    convert_i32_s = 0xB2
    convert_i32_u = 0xB3
    convert_i64_s = 0xB4
    convert_i64_u = 0xB5
    demote_f64 = 0xB6
    reinterpret_i32 = 0xBE
    load = 0x2A
    store = 0x38
    const = 0x43


class f64(HexEnum, encoding=0x7C):
    eq = 0x61
    ne = 0x62
    lt = 0x63
    gt = 0x64
    le = 0x65
    ge = 0x66
    abs = 0x99
    neg = 0x9A
    ceil = 0x9B
    floor = 0x9C
    trunc = 0x9D
    nearest = 0x9E
    sqrt = 0x9F
    add = 0xA0
    sub = 0xA1
    mul = 0xA2
    div = 0xA3
    min = 0xA4
    max = 0xA5
    copysign = 0xA6
    convert_i32_s = 0xB7
    convert_i32_u = 0xB8
    convert_i64_s = 0xB9
    convert_i64_u = 0xBA
    promote_f32 = 0xBB
    reinterpret_i64 = 0xBF
    load = 0x2B
    store = 0x39
    const = 0x44


class local(HexEnum):
    get = 0x20
    set = 0x21
    tee = 0x22


class global_(HexEnum):
    get = 0x23
    set = 0x24


global_.__name__ = "global"

# Special void type for block returns
void = 0x40


# ------------------------------------------------------------
def encode_function_type(parameters, results):
    """
    parameters is a vector of value types
    results is a vector value types
    """
    enc_parms = bytes(parameters)
    enc_results = bytes(results)
    return b"\x60" + encode_vector(enc_parms) + encode_vector(enc_results)


def encode_limits(min, max=None):
    if max is None:
        return b"\x00" + encode_unsigned(min)
    else:
        return b"\x01" + encode_unsigned(min) + encode_unsigned(max)


def encode_table_type(elemtype, min, max=None):
    return b"\x70" + encode_limits(min, max)


def encode_global_type(value_type, mut=True):
    return bytes([value_type, mut])


# ----------------------------------------------------------------------
# Instruction builders
#
# Wasm instructions are grouped into different namespaces.  For example:
#
#     i32.add()
#     local.get()
#     memory.size()
#     ...
#
# The classes that follow implement the namespace for different instruction
# categories.

# Builder for the local.* namespace


class SubBuilder:
    def __init__(self, builder):
        self._builder = builder

    def _append(self, instr):
        self._builder._code.append(instr)


class LocalBuilder(SubBuilder):
    def get(self, localidx):
        self._append([local.get, *encode_unsigned(localidx)])

    def set(self, localidx):
        self._append([local.set, *encode_unsigned(localidx)])

    def tee(self, localidx):
        self._append([local.tee, *encode_unsigned(localidx)])


class GlobalBuilder(SubBuilder):
    def get(self, glob):
        if isinstance(glob, int):
            globidx = glob
        else:
            globidx = glob.idx
        self._append([global_.get, *encode_unsigned(globidx)])

    def set(self, glob):
        if isinstance(glob, int):
            globidx = glob
        else:
            globidx = glob.idx
        self._append([global_.set, *encode_unsigned(globidx)])


class MemoryBuilder(SubBuilder):
    def size(self):
        self._append([0x3F, 0x00])

    def grow(self):
        self._append([0x40, 0x00])


class OpBuilder(SubBuilder):
    _optable = None  # To be supplied by subclasses

    # Memory ops
    def load(self, align, offset):
        self._append([self._optable.load, *encode_unsigned(align), *encode_unsigned(offset)])

    def load8_s(self, align, offset):
        self._append([self._optable.load8_s, *encode_unsigned(align), *encode_unsigned(offset)])

    def load8_u(self, align, offset):
        self._append([self._optable.load8_u, *encode_unsigned(align), *encode_unsigned(offset)])

    def load16_s(self, align, offset):
        self._append([self._optable.load16_s, *encode_unsigned(align), *encode_unsigned(offset)])

    def load16_u(self, align, offset):
        self._append([self._optable.load16_u, *encode_unsigned(align), *encode_unsigned(offset)])

    def load32_s(self, align, offset):
        self._append([self._optable.load32_s, *encode_unsigned(align), *encode_unsigned(offset)])

    def load32_u(self, align, offset):
        self._append([self._optable.load32_u, *encode_unsigned(align), *encode_unsigned(offset)])

    def store(self, align, offset):
        self._append([self._optable.store, *encode_unsigned(align), *encode_unsigned(offset)])

    def store8(self, align, offset):
        self._append([self._optable.store8, *encode_unsigned(align), *encode_unsigned(offset)])

    def store16(self, align, offset):
        self._append([self._optable.store16, *encode_unsigned(align), *encode_unsigned(offset)])

    def store32(self, align, offset):
        self._append([self._optable.store32, *encode_unsigned(align), *encode_unsigned(offset)])

    def __getattr__(self, key):
        def call():
            self._append([getattr(self._optable, key)])

        return call


class I32OpBuilder(OpBuilder):
    _optable = i32

    def const(self, value):
        self._append([self._optable.const, *encode_signed(value)])


class I64OpBuilder(OpBuilder):
    _optable = i64

    def const(self, value):
        self._append([self._optable.const, *encode_signed(value)])


class F32OpBuilder(OpBuilder):
    _optable = f32

    def const(self, value):
        self._append([self._optable.const, *encode_f32(value)])


class F64OpBuilder(OpBuilder):
    _optable = f64

    def const(self, value):
        self._append([self._optable.const, *encode_f64(value)])


def _flatten(instr):
    for x in instr:
        if isinstance(x, list):
            yield from _flatten(x)
        else:
            yield x


# High-level class that allows instructions to be easily encoded.
class InstructionBuilder:
    def __init__(self):
        self._code = []
        self.local = LocalBuilder(self)
        self.global_ = GlobalBuilder(self)
        self.i32 = I32OpBuilder(self)
        self.i64 = I64OpBuilder(self)
        self.f32 = F32OpBuilder(self)
        self.f64 = F64OpBuilder(self)

        # Control-flow stack.
        self._control = [None]

    def __iter__(self):
        return iter(self._code)

    # Resolve a human-readable label into control-stack index
    def _resolve_label(self, label):
        if isinstance(label, int):
            return label
        index = self._control.index(label)
        return len(label) - 1 - index

    # Control flow instructions
    def unreachable(self):
        self._code.append([0x01])

    def nop(self):
        self._code.append([0x01])

    def block_start(self, result_type, label=None):
        self._code.append([0x02, result_type])
        self._control.append(label)
        return len(self._control)

    def block_end(self):
        self._code.append([0x0B])
        self._control.pop()

    def loop_start(self, result_type, label=None):
        self._code.append([0x03, result_type])
        self._control.append(label)
        return len(self._control)

    def if_start(self, result_type, label=None):
        self._code.append([0x04, result_type])
        self._control.append(label)

    def else_start(self):
        self._code.append([0x05])

    def br(self, label):
        labelidx = self._resolve_label(label)
        self._code.append([0x0C, *encode_unsigned(labelidx)])

    def br_if(self, label):
        labelidx = self._resolve_label(label)
        self._code.append([0x0D, *encode_unsigned(labelidx)])

    def br_table(self, labels, label):
        enc_labels = [encode_unsigned(self._resolve_label(idx)) for idx in labels]
        self._code.append([0x0E, *encode_vector(enc_labels), *encode_unsigned(self._resolve_label(label))])

    def return_(self):
        self._code.append([0x0F])

    def call(self, func):
        if isinstance(func, (ImportFunction, Function)):
            self._code.append([0x10, *encode_unsigned(func._idx)])
        else:
            self._code.append([0x10, *encode_unsigned(func)])

    def call_indirect(self, typesig):
        if isinstance(typesig, Type):
            typeidx = typesig.idx
        else:
            typeidx = typesig
        self._code.append([0x11, *encode_unsigned(typeidx), 0x00])

    def drop(self):
        self._code.append([0x1A])

    def select(self):
        self._code.append([0x1B])


class Type:
    def __init__(self, parms, results, idx):
        self.parms = parms
        self.results = results
        self.idx = idx

    def __repr__(self):
        return f"{self.parms!r} -> {self.results!r}"


class ImportFunction:
    def __init__(self, name, typesig, idx):
        self._name = name
        self._typesig = typesig
        self._idx = idx

    def __repr__(self):
        return f"ImportFunction({self._name}, {self._typesig}, {self._idx})"


class Function(InstructionBuilder):
    def __init__(self, name, typesig, idx, export=True):
        super().__init__()
        self._name = name
        self._typesig = typesig
        self._locals = list(typesig.parms)
        self._export = export
        self._idx = idx

    def __repr__(self):
        return f"Function({self._name}, {self._typesig}, {self._idx})"

    # Allocate a new local variable of a given type
    def alloc(self, valuetype):
        self._locals.append(valuetype)
        return len(self.locals) - 1


class ImportGlobal:
    def __init__(self, name, valtype, idx):
        self.name = name
        self.valtype = valtype
        self.idx = idx

    def __repr__(self):
        return f"ImportGlobal({self.name}, {self.valtype}, {self.idx})"


class Global:
    def __init__(self, name, valtype, initializer, idx):
        self.name = name
        self.valtype = valtype
        self.initializer = initializer
        self.idx = idx

    def __repr__(self):
        return f"Global({self.name}, {self.valtype}, {self.initializer}, {self.idx})"


class Module:
    def __init__(self):
        # Vector of function type signatures.  Signatures are reused
        # if more than one function has the same signature.
        self.type_section = []

        # Vector of imported entities.  These can be functions, globals,
        # tables, and memories
        self.import_section = []

        # There are 4 basic entities within a Wasm file. Functions,
        # globals, memories, and tables.  Each kind of entity is
        # stored in a separate list and is indexed by an integer
        # index starting at 0.   Imported entities must always
        # go before entities defined in the Wasm module itself.
        self.funcidx = 0
        self.globalidx = 0
        self.memoryidx = 0
        self.tableidx = 0

        self.function_section = []  # Vector of typeidx
        self.global_section = []  # Vector of globals
        self.table_section = []  # Vector of tables
        self.memory_section = []  # Vector of memories

        # Exported entities.  A module may export functions, globals,
        # tables, and memories

        self.export_section = []  # Vector of exports

        # Optional start function.  A function that executes upon loading
        self.start_section = None  # Optional start function

        # Initialization of table elements
        self.element_section = []

        # Code section for function bodies.
        self.code_section = []

        # Data section contains data segments
        self.data_section = []

        # List of function objects (to help with encoding)
        self.functions = []

        # Output for JS/Html
        self.js_exports = ""
        self.html_exports = ""
        self.js_imports = defaultdict(dict)

    def add_type(self, parms, results):
        enc = encode_function_type(parms, results)
        if enc in self.type_section:
            return Type(parms, results, self.type_section.index(enc))
        else:
            self.type_section.append(enc)
            return Type(parms, results, len(self.type_section) - 1)

    def import_function(self, module, name, parms, results):
        if len(self.function_section) > 0:
            raise RuntimeError("function imports must go before first function definition")

        typesig = self.add_type(parms, results)
        code = encode_name(module) + encode_name(name) + b"\x00" + encode_unsigned(typesig.idx)
        self.import_section.append(code)
        self.js_imports[module][name] = f"function: {typesig}"
        self.funcidx += 1
        return ImportFunction(f"{module}.{name}", typesig, self.funcidx - 1)

    def import_table(self, module, name, elemtype, min, max=None):
        code = encode_name(module) + encode_name(name) + b"\x01" + encode_table_type(elemtype, min, max)
        self.import_section.append(code)
        self.js_imports[module][name] = "table:"
        self.tableidx += 1
        return self.tableidx - 1

    def import_memtype(self, module, name, min, max=None):
        code = encode_name(module) + encode_name(name) + b"\x02" + encode_limits(min, max)
        self.import_section.append(code)
        self.js_imports[module][name] = "memory:"
        self.memoryidx += 1
        return self.memoryidx - 1

    def import_global(self, module, name, value_type):
        if len(self.global_section) > 0:
            raise RuntimeError("global imports must go before first global definition")

        code = encode_name(module) + encode_name(name) + b"\x03" + encode_global_type(value_type, False)
        self.import_section.append(code)
        self.js_imports[module][name] = f"global: {value_type}"
        self.globalidx += 1
        return ImportGlobal(f"{module}.{name}", value_type, self.globalidx - 1)

    def add_function(self, name, parms, results, export=True):
        typesig = self.add_type(parms, results)
        func = Function(name, typesig, self.funcidx, export)
        self.funcidx += 1
        self.functions.append(func)
        self.function_section.append(encode_unsigned(typesig.idx))
        self.html_exports += f'<p><tt>{name}({", ".join(str(p) for p in parms)}) -> {results[0]!s}</tt></p>\n'
        return func

    def add_table(self, elemtype, min, max=None):
        self.table_section.append(encode_table_type(elemtype, min, max))
        self.tableidx += 1
        return self.tableidx - 1

    def add_memory(self, min, max=None):
        self.memory_section.append(encode_limits(min, max))
        self.memoryidx += 1
        return self.memoryidx - 1

    def add_global(self, name, value_type, initializer, mutable=True, export=True):
        code = encode_global_type(value_type, mutuable)
        expr = InstructionBuilder()
        getattr(expr, str(valtype)).const(initializer)
        expr.finalize()
        code += expr._code
        self.global_section.append(code)
        if export:
            self.export_global(name, self.globalidx)
        self.globalidx += 1
        return Global(name, value_type, initializer, self.globalidx - 1)

    def export_function(self, name, funcidx):
        code = encode_name(name) + b"\x00" + encode_unsigned(funcidx)
        self.export_section.append(code)
        self.js_exports += f"window.{name} = results.instance.exports.{name};\n"

    def export_table(self, name, tableidx):
        code = encode_name(name) + b"\x01" + encode_unsigned(tableidx)
        self.export_section.append(code)

    def export_memory(self, name, memidx):
        code = encode_name(name) + b"\x02" + encode_unsigned(memidx)
        self.export_section.append(code)

    def export_global(self, name, globalidx):
        code = encode_name(name) + b"\x03" + encode_unsigned(globalidx)
        self.export_section.append(code)

    def start_function(self, funcidx):
        self.start = encode_unsigned(funcidx)

    def add_element(self, tableidx, expr, funcidxs):
        code = encode_unsigned(tableidx) + expr.code
        code += encode_vector([encode_unsigned(i) for i in funcidxs])
        self.element_section.append(code)

    def add_function_code(self, locals, expr):
        # Locals is a list of valtypes [i32, i32, etc...]
        # expr is an expression representing the actual code (InstructionBuilder)

        locs = [encode_unsigned(1) + bytes([loc]) for loc in locals]
        locs_code = encode_vector(locs)
        func_code = locs_code + bytes(_flatten(expr))
        code = encode_unsigned(len(func_code)) + func_code
        self.code_section.append(code)

    def add_data(self, memidx, expr, data):
        # data is bytes
        code = encode_unsigned(memidx) + expr.code + encode_vector([data[i : i + 1] for i in range(len(data))])
        self.data_section.append(code)

    def _encode_section_vector(self, sectionid, contents):
        if not contents:
            return b""
        contents_code = encode_vector(contents)
        code = bytes([sectionid]) + encode_unsigned(len(contents_code)) + contents_code
        return code

    def encode(self):
        for func in self.functions:
            self.add_function_code(func._locals, func._code)
            if func._export:
                self.export_function(func._name, func._idx)

        # Encode the whole module
        code = b"\x00\x61\x73\x6d\x01\x00\x00\x00"
        code += self._encode_section_vector(1, self.type_section)
        code += self._encode_section_vector(2, self.import_section)
        code += self._encode_section_vector(3, self.function_section)
        code += self._encode_section_vector(4, self.table_section)
        code += self._encode_section_vector(5, self.memory_section)
        code += self._encode_section_vector(6, self.global_section)
        code += self._encode_section_vector(7, self.export_section)
        if self.start_section:
            code += encode_unsigned(8) + self.start_section
        code += self._encode_section_vector(9, self.element_section)
        code += self._encode_section_vector(10, self.code_section)
        code += self._encode_section_vector(11, self.data_section)
        return code

    def write_wasm(self, modname):
        with open(f"{modname}.wasm", "wb") as f:
            f.write(self.encode())

    def write_html(self, modname):
        with open(f"{modname}.html", "w") as f:
            f.write(
                js_template.format(
                    module=modname,
                    imports=json.dumps(self.js_imports, indent=4),
                    exports=self.js_exports,
                    exports_html=self.html_exports,
                )
            )


js_template = """
<html>
<body>
  <script>
    var imports = {imports};

    fetch("{module}.wasm").then(response =>
      response.arrayBuffer()
    ).then(bytes =>
           WebAssembly.instantiate(bytes, imports)
    ).then(results => {{
      {exports}
    }});
  </script>

<h3>module {module}</h3>

<p>
The following exports are made. Access from the JS console.
</p>

{exports_html}
</body>
</html>
"""


def test1():
    mod = Module()

    # An external function import.  Note:  All imports MUST go first.
    # Indexing affects function indexing for functions defined in the module.

    # Import some functions from JS
    # math_sin = mod.import_function('util', 'sin', [f64], [f64])
    # math_cos = mod.import_function('util', 'cos', [f64], [f64])

    # Import a function from another module entirely
    # fact = mod.import_function('recurse', 'fact', [i32], [i32])

    # Import a global variable (from JS?)
    # FOO = mod.import_global('util', 'FOO', f64)

    # A more complicated function
    dsquared_func = mod.add_function("dsquared", [f64, f64], [f64])
    dsquared_func.local.get(0)
    dsquared_func.local.get(0)
    dsquared_func.f64.mul()
    dsquared_func.local.get(1)
    dsquared_func.local.get(1)
    dsquared_func.f64.mul()
    dsquared_func.f64.add()
    dsquared_func.block_end()

    # A function calling another function
    distance = mod.add_function("distance", [f64, f64], [f64])
    distance.local.get(0)
    distance.local.get(1)
    distance.call(dsquared_func)
    distance.f64.sqrt()
    distance.block_end()

    # A function calling out to JS
    # ext = mod.add_function('ext', [f64, f64], [f64])
    # ext.local.get(0)
    # ext.call(math_sin)
    # ext.local.get(1)
    # ext.call(math_cos)
    # ext.f64.add()
    # ext.block_end()

    # A function calling across modules
    # tenf = mod.add_function('tenfact', [i32], [i32])
    # tenf.local.get(0)
    # tenf.call(fact)
    # tenf.i32.const(10)
    # tenf.i32.mul()
    # tenf.block_end()

    # A function accessing an imported global variable
    # gf = mod.add_function('gf', [f64], [f64])
    # gf.global_.get(FOO)
    # gf.local.get(0)
    # gf.f64.mul()
    # gf.block_end()

    # Memory
    mod.add_memory(1)
    mod.export_memory("memory", 0)

    # Function that returns a byte value
    getval = mod.add_function("getval", [i32], [i32])
    getval.local.get(0)
    getval.i32.load8_u(0, 0)
    getval.block_end()

    # Function that sets a byte value
    setval = mod.add_function("setval", [i32, i32], [i32])
    setval.local.get(0)  # Memory address
    setval.local.get(1)  # value
    setval.i32.store8(0, 0)
    setval.i32.const(1)
    setval.block_end()
    return mod


def test2():
    mod = Module()

    fact = mod.add_function("fact", [i32], [i32])
    fact.local.get(0)
    fact.i32.const(1)
    fact.i32.lt_s()
    fact.if_start(i32)
    fact.i32.const(1)
    fact.else_start()
    fact.local.get(0)
    fact.local.get(0)
    fact.i32.const(1)
    fact.i32.sub()
    fact.call(fact)
    fact.i32.mul()
    fact.block_end()
    fact.block_end()

    return mod


if __name__ == "__main__":
    mod = test1()

    mod.write_wasm("test")
    mod.write_html("test")
