import typing

import numpy as np

import vkdispatch as vd
import vkdispatch_native


class FFTPlan:
    def __init__(self, shape: typing.Tuple[int, ...]):
        assert len(shape) > 0 and len(shape) < 4, "shape must be 1D, 2D, or 3D"

        self.shape = shape
        self.mem_size = (
            np.prod(shape) * np.dtype(np.complex64).itemsize
        )  # currently only support complex64

        self._handle = vkdispatch_native.stage_fft_plan_create(
            vd.get_context_handle(), list(reversed(self.shape)), self.mem_size
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


def get_fft_plan(buffer_handle: int, shape: typing.Tuple[int, ...]) -> FFTPlan:
    global __fft_plans

    if shape not in __fft_plans:
        __fft_plans[(buffer_handle, *shape)] = FFTPlan(shape)

    return __fft_plans[(buffer_handle, *shape)]


def reset_fft_plans():
    global __fft_plans
    __fft_plans = {}


class FFTDispatcher:
    def __init__(self, inverse: bool = False):
        self.__inverse = inverse

    def __getitem__(self, cmd_list: vd.CommandList):

        if cmd_list is None:
            return self.__call__

        my_cmd_list: typing.List[vd.CommandList] = [cmd_list]

        def wrapper_func(buffer: vd.Buffer):
            plan = get_fft_plan(buffer._handle, buffer.shape)
            plan.record(my_cmd_list[0], buffer, self.__inverse)

        return wrapper_func

    def __call__(self, buffer: vd.Buffer):
        cmd_list = vd.get_command_list()
        self[cmd_list](buffer)
        cmd_list.submit()


fft = FFTDispatcher(False)
ifft = FFTDispatcher(True)
