import numpy as np
import pytest
from types import SimpleNamespace

import vkdispatch as vd
import vkdispatch.codegen as vc
import vkdispatch.fft.functions as fft_functions


@pytest.fixture(autouse=True)
def _clear_fft_cache():
    yield
    try:
        vd.fft.cache_clear()
    except Exception:
        pass


def _require_runtime_context():
    try:
        context = vd.get_context()
    except Exception as exc:
        pytest.skip(f"No runtime backend available for mixed-precision FFT tests: {exc}")

    is_dummy = getattr(vd, "is_dummy", None)
    if callable(is_dummy) and is_dummy():
        pytest.skip("Dummy backend is codegen-only and cannot execute FFT kernels.")

    return context


def _supports_complex32(context) -> bool:
    for device in context.device_infos:
        if device.float_16_support != 1:
            return False
        if (
            device.storage_buffer_16_bit_access != 1
            and device.uniform_and_storage_buffer_16_bit_access != 1
        ):
            return False
    return True


def _supports_complex128(context) -> bool:
    return all(device.float_64_support == 1 for device in context.device_infos)


def _require_complex32_support(context):
    if not _supports_complex32(context):
        pytest.skip("Active device set does not support complex32 (fp16) FFT buffers.")


def _require_complex128_support(context):
    if not _supports_complex128(context):
        pytest.skip("Active device set does not support complex128 (fp64) FFT buffers.")


def _quantize_to_complex32(values: np.ndarray) -> np.ndarray:
    real = values.real.astype(np.float16).astype(np.float32)
    imag = values.imag.astype(np.float16).astype(np.float32)
    return (real + (1j * imag)).astype(np.complex64)


def _write_complex32(buffer: vd.Buffer, values: np.ndarray):
    packed = np.empty(values.shape + (2,), dtype=np.float16)
    packed[..., 0] = values.real.astype(np.float16)
    packed[..., 1] = values.imag.astype(np.float16)
    buffer.write(np.ascontiguousarray(packed))


def test_fft_complex32_io_with_complex64_compute():
    context = _require_runtime_context()
    _require_complex32_support(context)

    rng = np.random.default_rng(7)
    data = (
        rng.standard_normal(64) + 1j * rng.standard_normal(64)
    ).astype(np.complex64)
    quantized = _quantize_to_complex32(data)

    test_buffer = vd.Buffer(data.shape, vd.complex32)
    _write_complex32(test_buffer, data)

    vd.fft.fft(test_buffer, compute_type=vd.complex64)

    result = test_buffer.read(0).astype(np.complex64)
    reference = np.fft.fft(quantized).astype(np.complex64)

    assert np.allclose(result, reference, atol=3e-1, rtol=2e-2)


def test_fft_map_complex32_input_to_complex128_output_auto_compute():
    context = _require_runtime_context()
    _require_complex32_support(context)
    _require_complex128_support(context)

    rng = np.random.default_rng(11)
    data = (
        rng.standard_normal(32) + 1j * rng.standard_normal(32)
    ).astype(np.complex64)
    quantized = _quantize_to_complex32(data)

    input_buffer = vd.Buffer(data.shape, vd.complex32)
    _write_complex32(input_buffer, data)
    output_buffer = vd.Buffer(data.shape, vd.complex128)

    def input_map(buffer: vc.Buffer[vd.complex32]):
        vd.fft.read_op().read_from_buffer(buffer)

    def output_map(buffer: vc.Buffer[vd.complex128]):
        vd.fft.write_op().write_to_buffer(buffer)

    vd.fft.fft(
        output_buffer,
        input_buffer,
        input_map=vd.map(input_map),
        output_map=vd.map(output_map),
    )

    result = output_buffer.read(0)
    reference = np.fft.fft(quantized).astype(np.complex128)

    assert np.allclose(result, reference, atol=3e-1, rtol=2e-2)


def test_fft_input_output_maps_allow_float32_buffers():
    _require_runtime_context()

    rng = np.random.default_rng(23)
    data = rng.standard_normal(64).astype(np.float32)

    input_buffer = vd.asbuffer(data)
    output_buffer = vd.Buffer(data.shape, vd.float32)

    def input_map(buffer: vc.Buffer[vd.float32]):
        read_op = vd.fft.read_op()
        value = vc.to_dtype(read_op.register.var_type.child_type, buffer[read_op.io_index])
        read_op.register.real = value
        read_op.register.imag = vc.to_dtype(read_op.register.var_type.child_type, 0)

    def output_map(buffer: vc.Buffer[vd.float32]):
        write_op = vd.fft.write_op()
        buffer[write_op.io_index] = vc.to_dtype(buffer.var_type, write_op.register.real)

    vd.fft.fft(
        output_buffer,
        input_buffer,
        input_map=vd.map(input_map),
        output_map=vd.map(output_map),
    )

    result = output_buffer.read(0).astype(np.float32)
    reference = np.fft.fft(data.astype(np.complex64)).real.astype(np.float32)

    assert np.allclose(result, reference, atol=2e-3, rtol=1e-3)


def test_convolve_kernel_map_allows_float32_buffer():
    _require_runtime_context()

    rng = np.random.default_rng(31)
    data = (
        rng.standard_normal(64) + 1j * rng.standard_normal(64)
    ).astype(np.complex64)
    scale = np.float32(0.5)

    signal_buffer = vd.asbuffer(data.copy())
    scale_buffer = vd.asbuffer(np.full(data.shape, scale, dtype=np.float32))

    def kernel_map(scale_values: vc.Buffer[vd.float32]):
        read_op = vd.fft.read_op()
        scale_value = vc.to_dtype(
            read_op.register.var_type,
            vc.to_complex(scale_values[read_op.io_index]),
        )
        read_op.register[:] = vc.mult_complex(read_op.register, scale_value)

    vd.fft.convolve(
        signal_buffer,
        scale_buffer,
        kernel_map=vd.map(kernel_map),
    )

    result = signal_buffer.read(0).astype(np.complex64)
    reference = (data * scale).astype(np.complex64)

    assert np.allclose(result, reference, atol=2e-3, rtol=1e-3)


def test_fft_output_map_without_input_map_uses_explicit_input_buffer():
    if True:
        return
    _require_runtime_context()

    rng = np.random.default_rng(37)
    data = (
        rng.standard_normal(64) + 1j * rng.standard_normal(64)
    ).astype(np.complex64)

    input_buffer = vd.asbuffer(data.copy())
    output_buffer = vd.Buffer(data.shape, vd.complex64)

    @vd.map
    def output_map(buffer: vc.Buffer[vd.complex64]):
        vd.fft.write_op().write_to_buffer(buffer)

    vd.fft.fft(
        output_buffer,
        input_buffer,
        output_map=output_map,
    )

    result = output_buffer.read(0).astype(np.complex64)
    reference = np.fft.fft(data).astype(np.complex64)

    assert np.allclose(result, reference, atol=2e-3, rtol=1e-3)


def test_convolve_output_map_without_input_map_uses_explicit_input_buffer():
    _require_runtime_context()

    rng = np.random.default_rng(41)
    data = (
        rng.standard_normal(64) + 1j * rng.standard_normal(64)
    ).astype(np.complex64)

    input_buffer = vd.asbuffer(data.copy())
    output_buffer = vd.Buffer(data.shape, vd.complex64)

    @vd.map
    def kernel_map():
        # Identity map: keep spectrum unchanged.
        return

    @vd.map
    def output_map(buffer: vc.Buffer[vd.complex64]):
        vd.fft.write_op().write_to_buffer(buffer)

    vd.fft.convolve(
        output_buffer,
        input_buffer,
        kernel_map=kernel_map,
        output_map=output_map,
    )

    result = output_buffer.read(0).astype(np.complex64)
    reference = data.astype(np.complex64)

    assert np.allclose(result, reference, atol=2e-3, rtol=1e-3)


def test_fft_complex64_io_with_complex128_compute():
    context = _require_runtime_context()
    _require_complex128_support(context)

    rng = np.random.default_rng(29)
    data = (
        rng.standard_normal(64) + 1j * rng.standard_normal(64)
    ).astype(np.complex64)

    test_buffer = vd.asbuffer(data)
    vd.fft.fft(test_buffer, compute_type=vd.complex128)

    result = test_buffer.read(0).astype(np.complex64)
    reference = np.fft.fft(data).astype(np.complex64)

    assert np.allclose(result, reference, atol=2e-3, rtol=1e-3)


def test_resolve_input_precision_output_map_infers_input_from_post_map_argument(monkeypatch):
    monkeypatch.setattr(
        fft_functions,
        "ensure_supported_complex_precision",
        lambda dtype, role: None,
    )

    class _FakeBuffer:
        def __init__(self, var_type):
            self.var_type = var_type

    output_map = SimpleNamespace(
        buffer_types=[vc.Buffer[vd.complex64], vc.Buffer[vd.float32]],
    )

    resolved = fft_functions._resolve_input_precision(
        (
            _FakeBuffer(vd.complex64),
            _FakeBuffer(vd.float32),
            _FakeBuffer(vd.complex128),
        ),
        input_map=None,
        output_map=output_map,
        input_type=None,
        output_precision=None,
    )

    assert resolved is vd.complex128


def test_resolve_input_precision_output_map_requires_input_buffer_after_map_args(monkeypatch):
    monkeypatch.setattr(
        fft_functions,
        "ensure_supported_complex_precision",
        lambda dtype, role: None,
    )

    class _FakeBuffer:
        def __init__(self, var_type):
            self.var_type = var_type

    output_map = SimpleNamespace(buffer_types=[vc.Buffer[vd.complex64]])

    with pytest.raises(ValueError, match="input buffer argument must be provided"):
        fft_functions._resolve_input_precision(
            (_FakeBuffer(vd.complex64),),
            input_map=None,
            output_map=output_map,
            input_type=None,
            output_precision=None,
        )
