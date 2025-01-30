import typing

import numpy as np

import vkdispatch_native

from .errors import check_for_errors

from .buffer import Buffer
from .context import get_context_handle
from .command_list import CommandList

from .dtype import complex64

class FFTPlan:
    def __init__(self, shape: typing.Tuple[int, ...], do_r2c: bool = False):
        assert len(shape) > 0 and len(shape) < 4, "shape must be 1D, 2D, or 3D"

        self.shape = shape
        self.do_r2c = do_r2c
        self.mem_size = (
            np.prod(shape) * np.dtype(np.complex64).itemsize
        )  # currently only support complex64

        self._handle = vkdispatch_native.stage_fft_plan_create(
            get_context_handle(), list(reversed(self.shape)), self.mem_size, 1 if do_r2c else 0 
        )
        check_for_errors()

    def record(self, command_list: CommandList, buffer: Buffer, inverse: bool = False):
        assert buffer.var_type == complex64, "buffer must be of dtype complex64"
        assert buffer.mem_size == self.mem_size, "buffer size must match plan size"

        vkdispatch_native.stage_fft_record(
            command_list._handle, self._handle, buffer._handle, 1 if inverse else -1
        )
        check_for_errors()

    def record_forward(self, command_list: CommandList, buffer: Buffer):
        self.record(command_list, buffer, False)

    def record_inverse(self, command_list: CommandList, buffer: Buffer):
        self.record(command_list, buffer, True)

