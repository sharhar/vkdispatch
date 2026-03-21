import vkdispatch as vd

from typing import Iterable, List, Optional


_COMPLEX_PRECISION_ORDER = (vd.complex32, vd.complex64, vd.complex128)
_COMPLEX_PRECISION_RANK = {dtype: rank for rank, dtype in enumerate(_COMPLEX_PRECISION_ORDER)}


def is_complex_precision(dtype) -> bool:
    return dtype in _COMPLEX_PRECISION_RANK


def validate_complex_precision(dtype, *, arg_name: str) -> None:
    if not is_complex_precision(dtype):
        raise ValueError(f"{arg_name} must be one of complex32, complex64, or complex128 (got {dtype})")


def promote_complex_precisions(dtypes: Iterable) -> vd.dtype:
    candidates = list(dtypes)
    if len(candidates) == 0:
        raise ValueError("At least one complex dtype is required for promotion")

    for candidate in candidates:
        validate_complex_precision(candidate, arg_name="dtype")

    return max(candidates, key=lambda dtype: _COMPLEX_PRECISION_RANK[dtype])


def default_compute_precision(io_precisions: Iterable) -> vd.dtype:
    promoted = promote_complex_precisions(io_precisions)

    # Default to at least complex64 for numerical stability.
    if _COMPLEX_PRECISION_RANK[promoted] < _COMPLEX_PRECISION_RANK[vd.complex64]:
        return vd.complex64

    return promoted


def supports_complex_precision(dtype) -> bool:
    validate_complex_precision(dtype, arg_name="dtype")
    scalar_type = dtype.child_type

    for device in vd.get_context().device_infos:
        if scalar_type == vd.float16:
            if device.float_16_support != 1:
                return False

            # Half precision in storage buffers typically needs one of these capabilities.
            if (
                device.storage_buffer_16_bit_access != 1
                and device.uniform_and_storage_buffer_16_bit_access != 1
            ):
                return False

        if scalar_type == vd.float64 and device.float_64_support != 1:
            return False

    return True


def ensure_supported_complex_precision(dtype, *, role: str) -> None:
    if not supports_complex_precision(dtype):
        raise ValueError(f"{role} precision '{dtype.name}' is not supported on the active device set")


def resolve_compute_precision(io_precisions: List, compute_precision: Optional[vd.dtype]) -> vd.dtype:
    if compute_precision is not None:
        validate_complex_precision(compute_precision, arg_name="compute_type")
        ensure_supported_complex_precision(compute_precision, role="Compute")
        return compute_precision

    for io_precision in io_precisions:
        validate_complex_precision(io_precision, arg_name="io_precision")

    if len(io_precisions) == 0:
        for candidate in (vd.complex64, vd.complex32):
            if supports_complex_precision(candidate):
                return candidate

        raise ValueError(
            "Unable to resolve a default compute precision supported by all active devices"
        )

    target = default_compute_precision(io_precisions)
    if supports_complex_precision(target):
        return target

    # Auto fallback: drop from complex128 to complex64 when fp64 is unsupported.
    for candidate in (vd.complex64, vd.complex32):
        if (
            _COMPLEX_PRECISION_RANK[candidate] <= _COMPLEX_PRECISION_RANK[target]
            and supports_complex_precision(candidate)
        ):
            return candidate

    raise ValueError(
        "Unable to resolve an auto compute precision supported by all active devices"
    )
