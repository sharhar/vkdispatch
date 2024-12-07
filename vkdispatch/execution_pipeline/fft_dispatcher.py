
import typing

import vkdispatch as vd

__fft_plans = {}

def get_fft_plan(buffer_handle: int, shape: typing.Tuple[int, ...], do_r2c: bool) -> vd.FFTPlan:
    global __fft_plans

    fft_plan_key = (buffer_handle, *shape, do_r2c)

    if fft_plan_key not in __fft_plans:
        __fft_plans[fft_plan_key] = vd.FFTPlan(shape, do_r2c)

    return __fft_plans[fft_plan_key]


def reset_fft_plans():
    global __fft_plans
    __fft_plans = {}


class FFTDispatcher:
    def __init__(self, inverse: bool = False, do_r2c: bool = False):
        self.__inverse = inverse
        self.__do_r2c = do_r2c

    def __call__(self, buffer: vd.Buffer, cmd_list: typing.Optional[vd.CommandList] = None, cmd_stream: typing.Optional[vd.CommandStream] = None):
        my_cmd_list = cmd_list

        if cmd_stream is not None:
            if my_cmd_list is not None:
                raise ValueError("Cannot specify both cmd_list and cmd_stream")
            my_cmd_list = cmd_stream

        if my_cmd_list is None:
            my_cmd_list = vd.global_cmd_stream()

        plan = get_fft_plan(buffer._handle, buffer.shape, self.__do_r2c)
        plan.record(my_cmd_list, buffer, self.__inverse)
        
        if isinstance(my_cmd_list, vd.CommandStream):
            if my_cmd_list.submit_on_record:
                my_cmd_list.submit()


fft = FFTDispatcher(False, False)
ifft = FFTDispatcher(True, False)

rfft = FFTDispatcher(False, True)
rifft = FFTDispatcher(True, True)
