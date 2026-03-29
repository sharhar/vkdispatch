import vkdispatch as vd
import vkdispatch.codegen as vc

from vkdispatch.base.dtype import to_vector

import numpy as np
import pytest

from typing import Tuple

def run_index_ravel(shape: Tuple[int, ...], index: Tuple[int, ...], shape_static: bool):
    var_type =  to_vector(vd.uint32, len(shape))

    buffer = vd.Buffer(shape, var_type=var_type)

    @vd.shader("buff.size")
    def test_shader(buff: vc.Buff[var_type]): # pyright: ignore[reportInvalidTypeForm]
        ind = vc.global_invocation_id().x
        buff[ind] = vc.ravel_index(
            ind,
            shape if shape_static else buff.shape
        ).swizzle("xyz"[:len(shape)])

    test_shader(buffer)

    result_value = buffer.read(0)

    assert tuple(result_value[index]) == tuple(index), f"Expected index {index}, got {tuple(result_value[index])} for shape {shape} with shape_static={shape_static}"

    buffer.destroy()

def test_index_ravel():
    for _ in range(100):
        shape_len = np.random.choice([2, 3])
        shape = tuple(np.random.randint(1, 100) for _ in range(shape_len))
        index = tuple(np.random.randint(0, shape[i]) for i in range(shape_len))

        run_index_ravel(shape, index, False)
        run_index_ravel(shape, index, True)

def run_index_unravel(shape: Tuple[int, ...], index: Tuple[int, ...], input_static: bool, shape_static: bool):
    data = np.random.rand(*shape).astype(np.float32)
    buffer = vd.asbuffer(data)

    result_buffer = vd.Buffer((1,), var_type=vd.float32)

    index_type = vd.int32

    if len(index) == 2:
        index_type = vd.ivec2
    elif len(index) == 3:
        index_type = vd.ivec3

    if input_static and shape_static:
        @vd.shader(1)
        def test_shader(buff: vc.Buff[vc.f32], buff_in: vc.Buff[vc.f32]):
            buff[0] = buff_in[vc.unravel_index(index, shape)]
    elif input_static and not shape_static:
        @vd.shader(1)
        def test_shader(buff: vc.Buff[vc.f32], buff_in: vc.Buff[vc.f32]):
            buff[0] = buff_in[vc.unravel_index(index, buff_in.shape)]
    elif not input_static and shape_static:
        @vd.shader(1)
        def test_shader(buff: vc.Buff[vc.f32], buff_in: vc.Buff[vc.f32]):
            index_vec = vc.new_register(index_type, *index)
            buff[0] = buff_in[vc.unravel_index(index_vec, shape)]
    elif not input_static and not shape_static:
        @vd.shader(1)
        def test_shader(buff: vc.Buff[vc.f32], buff_in: vc.Buff[vc.f32]):
            index_vec = vc.new_register(index_type, *index)
            buff[0] = buff_in[vc.unravel_index(index_vec, buff_in.shape)]

    test_shader(result_buffer, buffer)

    result_value = result_buffer.read(0)[0]
    reference_value = data[index]

    assert np.isclose(result_value, reference_value, atol=1e-5), f"Expected {reference_value}, got {result_value}"

    buffer.destroy()
    result_buffer.destroy()

def test_index_unravel():
    for _ in range(100):
        shape_len = np.random.choice([1, 2, 3])
        shape = tuple(np.random.randint(1, 100) for _ in range(shape_len))
        index = tuple(np.random.randint(0, shape[i]) for i in range(shape_len))

        run_index_unravel(shape, index, False, False)
        run_index_unravel(shape, index, False, True)
        run_index_unravel(shape, index, True, False)
        run_index_unravel(shape, index, True, True)