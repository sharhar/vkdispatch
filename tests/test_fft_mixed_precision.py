import numpy as np
import pytest

import vkdispatch as vd
import vkdispatch.codegen as vc


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

    if vd.is_dummy():
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
