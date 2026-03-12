from __future__ import annotations

import builtins
import math
import struct

from typing import Any, Sequence, Tuple


def sign(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def floor(value: float) -> float:
    return float(math.floor(value))


def ceil(value: float) -> float:
    return float(math.ceil(value))


def trunc(value: float) -> float:
    return float(math.trunc(value))


def round(value: float) -> float:
    return float(builtins.round(value))


def abs_value(value: Any) -> float:
    return float(abs(value))


def mod(x: float, y: float) -> float:
    return float(x % y)


def modf(x: float, _unused: Any = None) -> Tuple[float, float]:
    frac, whole = math.modf(x)
    return float(frac), float(whole)


def minimum(x: float, y: float) -> float:
    return float(x if x <= y else y)


def maximum(x: float, y: float) -> float:
    return float(x if x >= y else y)


def clip(x: float, min_value: float, max_value: float) -> float:
    return float(min(max(x, min_value), max_value))


def interp(x: float, xp: Sequence[float], fp: Sequence[float]) -> float:
    if len(xp) != len(fp):
        raise ValueError("xp and fp must have the same length")
    if len(xp) == 0:
        raise ValueError("xp and fp must be non-empty")
    if len(xp) == 1:
        return float(fp[0])

    if x <= xp[0]:
        return float(fp[0])
    if x >= xp[-1]:
        return float(fp[-1])

    for index in range(1, len(xp)):
        if x <= xp[index]:
            x0 = xp[index - 1]
            x1 = xp[index]
            y0 = fp[index - 1]
            y1 = fp[index]

            if x1 == x0:
                return float(y0)

            t = (x - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))

    return float(fp[-1])


def isnan(value: float) -> bool:
    return math.isnan(value)


def isinf(value: float) -> bool:
    return math.isinf(value)


def float_bits_to_int(value: float) -> int:
    return int(struct.unpack("=i", struct.pack("=f", float(value)))[0])


def float_bits_to_uint(value: float) -> int:
    return int(struct.unpack("=I", struct.pack("=f", float(value)))[0])


def int_bits_to_float(value: int) -> float:
    return float(struct.unpack("=f", struct.pack("=i", int(value)))[0])


def uint_bits_to_float(value: int) -> float:
    return float(struct.unpack("=f", struct.pack("=I", int(value)))[0])


def power(x: float, y: float) -> float:
    return float(math.pow(x, y))


def exp(value: float) -> float:
    return float(math.exp(value))


def exp2(value: float) -> float:
    if hasattr(math, "exp2"):
        return float(math.exp2(value))
    return float(math.pow(2.0, value))


def log(value: float) -> float:
    return float(math.log(value))


def log2(value: float) -> float:
    return float(math.log2(value))


def sqrt(value: float) -> float:
    return float(math.sqrt(value))


def sin(value: float) -> float:
    return float(math.sin(value))


def cos(value: float) -> float:
    return float(math.cos(value))


def tan(value: float) -> float:
    return float(math.tan(value))


def arcsin(value: float) -> float:
    return float(math.asin(value))


def arccos(value: float) -> float:
    return float(math.acos(value))


def arctan(value: float) -> float:
    return float(math.atan(value))


def arctan2(y: float, x: float) -> float:
    return float(math.atan2(y, x))


def sinh(value: float) -> float:
    return float(math.sinh(value))


def cosh(value: float) -> float:
    return float(math.cosh(value))


def tanh(value: float) -> float:
    return float(math.tanh(value))


def arcsinh(value: float) -> float:
    return float(math.asinh(value))


def arccosh(value: float) -> float:
    return float(math.acosh(value))


def arctanh(value: float) -> float:
    return float(math.atanh(value))


def dot(x: Any, y: Any) -> float:
    if isinstance(x, (int, float, complex)) and isinstance(y, (int, float, complex)):
        return float(x * y)

    return float(sum(a * b for a, b in zip(x, y)))
