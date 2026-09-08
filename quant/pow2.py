"""Fixed-position arithmetic for the frozen XINT8 producer dialect.

Source contract: Quark 0.11rc1 quantization/quant_utils.py:364-379,
527-535, 814-833, 1000-1011. See results/quant/notes_xint8_dialect.log.
This is producer arithmetic, not a simulation of the DPU's rounding mode.
"""
from dataclasses import dataclass
import operator

import numpy as np

ACT_DTYPE, ACT_ZP = np.uint8, 128
W_DTYPE, B_DTYPE = np.int8, np.int8


def qrange(dtype: str) -> tuple[int, int]:
    """Producer clipping bounds; INT8 deliberately excludes -128."""
    if dtype == "int8":
        return -127, 127
    if dtype == "uint8":
        return 0, 255
    raise ValueError(f"Unsupported XINT8 dtype: {dtype!r}")


def _position(pos: int) -> int:
    if isinstance(pos, (bool, np.bool_)):
        raise ValueError("Position must be an integer, not bool")
    try:
        pos = operator.index(pos)
    except TypeError as exc:
        raise ValueError("Position must be an integer") from exc
    if not -127 <= pos <= 127:
        raise ValueError("Position must be in [-127, 127]")
    return pos


def _zero_point(zp: int, dtype: str) -> int:
    low, high = qrange(dtype)
    if isinstance(zp, (bool, np.bool_)):
        raise ValueError("Zero point must be an integer, not bool")
    try:
        zp = operator.index(zp)
    except TypeError as exc:
        raise ValueError("Zero point must be an integer") from exc
    if not low <= zp <= high:
        raise ValueError(f"Zero point {zp} outside {dtype} producer range")
    return zp


@dataclass(frozen=True)
class TensorQ:
    name: str
    dtype: str
    pos: int
    zp: int
    source: str

    def __post_init__(self) -> None:
        if not self.name or not self.source:
            raise ValueError("Tensor name and position source must be nonempty")
        _position(self.pos)
        _zero_point(self.zp, self.dtype)


def scale2pos(scale: float) -> int:
    """Nearest position, ties to even; finite positive scale clamped as Quark does.

    Reject invalid scales rather than accepting Quark's negative/zero clamp.
    Reading an exact dialect table must additionally check pos2scale(pos) == scale.
    """
    value = np.asarray(scale)
    if value.shape != () or not np.isfinite(value) or value <= 0:
        raise ValueError("Scale must be a finite positive scalar")
    bounded = min(max(float(value), 2.0 ** -127), 2.0 ** 127)
    return int(np.rint(-np.log2(bounded)))


def pos2scale(pos: int) -> np.float32:
    return np.float32(np.ldexp(1.0, -_position(pos)))


def _float_input(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x)
    if value.dtype.kind not in "fiu" or not np.all(np.isfinite(value)):
        raise ValueError("Input must contain finite real numbers")
    with np.errstate(over="ignore"):
        value = value.astype(np.float32)
    if not np.all(np.isfinite(value)):
        raise ValueError("Input must be representable in float32")
    return value


def quantize(x: np.ndarray, pos: int, zp: int, dtype: str) -> np.ndarray:
    """float32 divide, ties-to-even rounding, zero point, symmetric clipping."""
    scale = pos2scale(pos)
    zp = _zero_point(zp, dtype)
    value = _float_input(x)
    # Division may overflow for finite inputs at extreme positions; clip saturates it.
    with np.errstate(over="ignore"):
        rounded = np.rint(value / scale) + np.float32(zp)
    return np.asarray(np.clip(rounded, *qrange(dtype)), dtype=dtype)


def dequantize(q: np.ndarray, pos: int, zp: int) -> np.ndarray:
    value = np.asarray(q)
    _zero_point(zp, str(value.dtype))
    return np.asarray((value.astype(np.float32) - np.float32(zp)) * pos2scale(pos), dtype=np.float32)


def sqerr(x: np.ndarray, pos: int, zp: int, dtype: str) -> float:
    """Float32 summed squared error, matching Quark's MinMSE accumulation dtype.

    Position search lives in calib.py.
    """
    value = _float_input(x)
    restored = dequantize(quantize(value, pos, zp, dtype), pos, zp)
    return float(np.sum((restored - value) ** 2, dtype=np.float32))
