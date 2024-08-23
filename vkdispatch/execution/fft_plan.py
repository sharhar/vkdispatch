import typing

import numpy as np

import vkdispatch as vd
import vkdispatch_native


class FFTPlan:
    def __init__(self, shape: typing.Tuple[int, ...], do_r2c: bool = False):
        assert len(shape) > 0 and len(shape) < 4, "shape must be 1D, 2D, or 3D"

        self.shape = shape
        self.do_r2c = do_r2c
        self.mem_size = (
            np.prod(shape) * np.dtype(np.complex64).itemsize
        )  # currently only support complex64

        self._handle = vkdispatch_native.stage_fft_plan_create(
            vd.get_context_handle(), list(reversed(self.shape)), self.mem_size, 1 if do_r2c else 0 
        )
        vd.check_for_errors()

    def record(self, command_list: vd.CommandList, buffer: vd.Buffer, inverse: bool = False):
        assert buffer.var_type == vd.complex64, "buffer must be of dtype complex64"
        assert buffer.mem_size == self.mem_size, "buffer size must match plan size"

        vkdispatch_native.stage_fft_record(
            command_list._handle, self._handle, buffer._handle, 1 if inverse else -1
        )
        vd.check_for_errors()

    def record_forward(self, command_list: vd.CommandList, buffer: vd.Buffer):
        self.record(command_list, buffer, False)

    def record_inverse(self, command_list: vd.CommandList, buffer: vd.Buffer):
        self.record(command_list, buffer, True)


__fft_plans = {}


def get_fft_plan(buffer_handle: int, shape: typing.Tuple[int, ...], do_r2c: bool) -> FFTPlan:
    global __fft_plans

    fft_plan_key = (buffer_handle, *shape, do_r2c)

    if fft_plan_key not in __fft_plans:
        __fft_plans[fft_plan_key] = FFTPlan(shape, do_r2c)

    return __fft_plans[fft_plan_key]


def reset_fft_plans():
    global __fft_plans
    __fft_plans = {}


class FFTDispatcher:
    def __init__(self, inverse: bool = False, do_r2c: bool = False):
        self.__inverse = inverse
        self.__do_r2c = do_r2c

    def __call__(self, buffer: vd.Buffer, cmd_list: vd.CommandList = None):
        my_cmd_list = cmd_list

        if my_cmd_list is None:
            my_cmd_list = vd.global_cmd_list()

        plan = get_fft_plan(buffer._handle, buffer.shape, self.__do_r2c)
        plan.record(my_cmd_list, buffer, self.__inverse)
        
        if my_cmd_list.submit_on_record:
            my_cmd_list.submit()


fft = FFTDispatcher(False, False)
ifft = FFTDispatcher(True, False)

rfft = FFTDispatcher(False, True)
rifft = FFTDispatcher(True, True)
